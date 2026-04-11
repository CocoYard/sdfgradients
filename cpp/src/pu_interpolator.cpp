#include "pu_interpolator.h"
#include "kdtree.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <deque>
#include <iostream>
#include <chrono>

namespace sdf {

PUInterpolator::PUInterpolator(const std::string& kernel, double overlap,
                                 int min_points, int max_points,
                                 const std::string& partition)
    : kernel_(kernel), partition_type_(partition), overlap_(overlap),
      min_points_(min_points), max_points_(max_points)
{
    use_box_ = (partition == "box");
}

// ── Wendland C2 weight ──────────────────────────────────────────────

double PUInterpolator::wendland_weight(double r, double radius) {
    double s = std::clamp(r / radius, 0.0, 1.0);
    double t = 1.0 - s;
    return t * t * t * t * (4.0 * s + 1.0);
}

double PUInterpolator::box_weight(const Eigen::Vector3d& pt,
                                   const Eigen::Vector3d& center,
                                   const Eigen::Vector3d& half_ext) {
    double w = 1.0;
    for (int k = 0; k < 3; k++) {
        double s = std::clamp(std::abs(pt(k) - center(k)) / half_ext(k), 0.0, 1.0);
        double t = 1.0 - s;
        w *= t * t * t * t * (4.0 * s + 1.0);
    }
    return w;
}

// ── Deduplication ───────────────────────────────────────────────────

void PUInterpolator::deduplicate(Eigen::MatrixXd& points, Eigen::VectorXd& values, double tol) {
    int n = (int)points.rows();
    if (n == 0) return;

    KDTree3D tree(points);
    std::vector<bool> keep(n, true);

    for (int i = 0; i < n; i++) {
        if (!keep[i]) continue;
        Eigen::Vector3d pt = points.row(i);
        auto neighbors = tree.query_ball_point(pt, tol);
        for (int j : neighbors) {
            if (j > i) keep[j] = false;
        }
    }

    std::vector<int> kept;
    for (int i = 0; i < n; i++)
        if (keep[i]) kept.push_back(i);

    Eigen::MatrixXd new_pts(kept.size(), points.cols());
    Eigen::VectorXd new_vals(kept.size());
    for (int i = 0; i < (int)kept.size(); i++) {
        new_pts.row(i) = points.row(kept[i]);
        new_vals(i) = values(kept[i]);
    }
    points = new_pts;
    values = new_vals;
}

// ── KDTree partition ─────────────────────────────────────────────────


static void subdivide_recursive(const Eigen::MatrixXd& points,
                                 const std::vector<int>& indices,
                                 int max_points,
                                 std::vector<std::vector<int>>& leaves) {
    if ((int)indices.size() <= max_points) {
        leaves.push_back(indices);
        return;
    }
    // Find axis with largest spread
    Eigen::Vector3d min_pt = Eigen::Vector3d::Constant(std::numeric_limits<double>::max());
    Eigen::Vector3d max_pt = Eigen::Vector3d::Constant(-std::numeric_limits<double>::max());
    for (int i : indices) {
        for (int k = 0; k < 3; k++) {
            min_pt(k) = std::min(min_pt(k), points(i, k));
            max_pt(k) = std::max(max_pt(k), points(i, k));
        }
    }
    Eigen::Vector3d spread = max_pt - min_pt;
    int axis;
    spread.maxCoeff(&axis);

    // Median split
    std::vector<double> coords(indices.size());
    for (int i = 0; i < (int)indices.size(); i++)
        coords[i] = points(indices[i], axis);
    std::nth_element(coords.begin(), coords.begin() + coords.size() / 2, coords.end());
    double median = coords[coords.size() / 2];

    std::vector<int> left, right;
    for (int i : indices) {
        if (points(i, axis) <= median)
            left.push_back(i);
        else
            right.push_back(i);
    }
    if (left.empty() || right.empty()) {
        leaves.push_back(indices);
        return;
    }
    subdivide_recursive(points, left, max_points, leaves);
    subdivide_recursive(points, right, max_points, leaves);
}

// ── fit ─────────────────────────────────────────────────────────────

void PUInterpolator::fit(const Eigen::MatrixXd& points,
                          const Eigen::VectorXd& values,
                          const Eigen::MatrixXd* gradients,
                          const Eigen::VectorXi* mask) {
    Eigen::MatrixXd pts;
    Eigen::VectorXd vals;

    // Add projection points if gradients given
    if (gradients) {
        Eigen::MatrixXd projections;
        if (mask) {
            int count = mask->sum();
            projections.resize(count, points.cols());
            int k = 0;
            for (int i = 0; i < (int)points.rows(); i++) {
                if ((*mask)(i))
                    projections.row(k++) = points.row(i) - values(i) * gradients->row(i);
            }
        } else {
            projections.resize(points.rows(), points.cols());
            for (int i = 0; i < (int)points.rows(); i++)
                projections.row(i) = points.row(i) - values(i) * gradients->row(i);
        }
        pts.resize(points.rows() + projections.rows(), points.cols());
        pts << points, projections;
        vals.resize(values.size() + projections.rows());
        vals << values, Eigen::VectorXd::Zero(projections.rows());
    } else {
        pts = points;
        vals = values;
    }

    // Filter by distance if too many
    if (vals.size() > 5000) {
        std::vector<int> keep;
        for (int i = 0; i < (int)vals.size(); i++)
            if (std::abs(vals(i)) < dist_threshold_) keep.push_back(i);
        Eigen::MatrixXd pf(keep.size(), pts.cols());
        Eigen::VectorXd vf(keep.size());
        for (int i = 0; i < (int)keep.size(); i++) {
            pf.row(i) = pts.row(keep[i]);
            vf(i) = vals(keep[i]);
        }
        std::cout << "Warning: too many points, only keeping " << keep.size()
                  << " points with abs(value) < " << dist_threshold_ << " for fitting.\n";
        pts = pf;
        vals = vf;
    }

    // Deduplicate
    deduplicate(pts, vals);
    int dim = (int)pts.cols();
    int min_pts = std::max(min_points_, dim + 2);

    using clock = std::chrono::high_resolution_clock;
    auto t0 = clock::now();

    // Build KDTree for partitioning
    KDTree3D tree(pts);
    auto t1 = clock::now();
    std::cout << "  [PU fit] build KDTree: "
              << std::chrono::duration<double>(t1-t0).count() << "s\n";

    // ── Partition ───────────────────────────────────────────────────
    int n = (int)pts.rows();
    std::vector<int> all_idx(n);
    std::iota(all_idx.begin(), all_idx.end(), 0);

    std::vector<std::vector<int>> leaves;
    subdivide_recursive(pts, all_idx, max_points_, leaves);

    struct PatchInfo {
        Eigen::Vector3d center;
        Eigen::Vector3d half_ext;
        std::vector<int> ext_idx;
    };
    std::vector<PatchInfo> patches_info;

    std::deque<std::vector<int>> queue(leaves.begin(), leaves.end());

    while (!queue.empty()) {
        auto leaf_idx = queue.front();
        queue.pop_front();

        if (use_box_) {
            Eigen::Vector3d min_pt = Eigen::Vector3d::Constant(std::numeric_limits<double>::max());
            Eigen::Vector3d max_pt = Eigen::Vector3d::Constant(-std::numeric_limits<double>::max());
            for (int i : leaf_idx)
                for (int k = 0; k < 3; k++) {
                    min_pt(k) = std::min(min_pt(k), pts(i, k));
                    max_pt(k) = std::max(max_pt(k), pts(i, k));
                }
            Eigen::Vector3d spreads = max_pt - min_pt;
            Eigen::Vector3d center = min_pt + spreads * 0.5;
            Eigen::Vector3d half_core = spreads * 0.5;
            double delta = half_core.maxCoeff() * overlap_;
            Eigen::Vector3d half_ext = half_core + Eigen::Vector3d::Constant(delta);

            std::vector<int> ext_idx;
            if (overlap_ == 0.0) {
                ext_idx = leaf_idx;
            } else {
                double bsphere_r = half_ext.norm();
                auto candidates = tree.query_ball_point(center, bsphere_r);
                for (int c : candidates) {
                    Eigen::Vector3d diff = (pts.row(c).transpose() - center).cwiseAbs();
                    if ((diff.array() <= half_ext.array()).all())
                        ext_idx.push_back(c);
                }
            }

            if ((int)ext_idx.size() > max_ext_points_ && (int)leaf_idx.size() > 2) {
                // Further split
                Eigen::Vector3d sp_min = Eigen::Vector3d::Constant(std::numeric_limits<double>::max());
                Eigen::Vector3d sp_max = Eigen::Vector3d::Constant(-std::numeric_limits<double>::max());
                for (int i : leaf_idx)
                    for (int k = 0; k < 3; k++) {
                        sp_min(k) = std::min(sp_min(k), pts(i, k));
                        sp_max(k) = std::max(sp_max(k), pts(i, k));
                    }
                int axis;
                (sp_max - sp_min).maxCoeff(&axis);
                std::vector<double> coords;
                for (int i : leaf_idx) coords.push_back(pts(i, axis));
                std::nth_element(coords.begin(), coords.begin() + coords.size() / 2, coords.end());
                double med = coords[coords.size() / 2];

                std::vector<int> left, right;
                for (int i : leaf_idx) {
                    if (pts(i, axis) <= med) left.push_back(i);
                    else right.push_back(i);
                }
                if (!left.empty() && !right.empty()) {
                    queue.push_back(left);
                    queue.push_back(right);
                    continue;
                }
            }

            patches_info.push_back({center, half_ext, ext_idx});

        } else {
            // Sphere partition
            Eigen::Vector3d center = Eigen::Vector3d::Zero();
            for (int i : leaf_idx) center += pts.row(i).transpose();
            center /= (double)leaf_idx.size();

            double r_core = 0.0;
            for (int i : leaf_idx)
                r_core = std::max(r_core, (pts.row(i).transpose() - center).norm());
            double R_ext = r_core * (1.0 + overlap_);
            auto ext_idx = tree.query_ball_point(center, R_ext);

            if ((int)ext_idx.size() > max_ext_points_ && (int)leaf_idx.size() > 2) {
                Eigen::Vector3d sp_min = Eigen::Vector3d::Constant(std::numeric_limits<double>::max());
                Eigen::Vector3d sp_max = Eigen::Vector3d::Constant(-std::numeric_limits<double>::max());
                for (int i : leaf_idx)
                    for (int k = 0; k < 3; k++) {
                        sp_min(k) = std::min(sp_min(k), pts(i, k));
                        sp_max(k) = std::max(sp_max(k), pts(i, k));
                    }
                int axis;
                (sp_max - sp_min).maxCoeff(&axis);
                std::vector<double> coords;
                for (int i : leaf_idx) coords.push_back(pts(i, axis));
                std::nth_element(coords.begin(), coords.begin() + coords.size() / 2, coords.end());
                double med = coords[coords.size() / 2];

                std::vector<int> left, right;
                for (int i : leaf_idx) {
                    if (pts(i, axis) <= med) left.push_back(i);
                    else right.push_back(i);
                }
                if (left.empty() || right.empty()) {
                    left.clear(); right.clear();
                    for (int i : leaf_idx) {
                        if (pts(i, axis) < med) left.push_back(i);
                        else right.push_back(i);
                    }
                }
                if (left.empty() || right.empty()) {
                    double c_axis = center(axis);
                    left.clear(); right.clear();
                    for (int i : leaf_idx) {
                        if (pts(i, axis) < c_axis) left.push_back(i);
                        else right.push_back(i);
                    }
                }
                if (!left.empty() && !right.empty()) {
                    queue.push_back(left);
                    queue.push_back(right);
                    continue;
                }
            }

            patches_info.push_back({center, Eigen::Vector3d::Constant(R_ext), ext_idx});
        }
    }

    // Sphere mode: remove contained patches
    if (!use_box_) {
        int np = (int)patches_info.size();
        std::vector<double> radii(np);
        for (int i = 0; i < np; i++) radii[i] = patches_info[i].half_ext(0);

        std::vector<int> sort_idx(np);
        std::iota(sort_idx.begin(), sort_idx.end(), 0);
        std::sort(sort_idx.begin(), sort_idx.end(),
                  [&](int a, int b) { return radii[a] > radii[b]; });

        std::vector<bool> keep(np, true);
        for (int si = 0; si < np; si++) {
            int i = sort_idx[si];
            if (!keep[i]) continue;
            for (int sj = si + 1; sj < np; sj++) {
                int j = sort_idx[sj];
                if (!keep[j]) continue;
                double dist = (patches_info[i].center - patches_info[j].center).norm();
                if (dist <= radii[i] - radii[j])
                    keep[j] = false;
            }
        }

        std::vector<PatchInfo> filtered;
        for (int i = 0; i < np; i++)
            if (keep[i]) filtered.push_back(std::move(patches_info[i]));
        patches_info = std::move(filtered);
    }

    auto t2 = clock::now();
    std::cout << "  [PU fit] greedy cover: "
              << std::chrono::duration<double>(t2-t1).count() << "s  ("
              << patches_info.size() << " patches)\n";

    // ── Fit local interpolators ────────────────────────────────────
    patches_.clear();
    patches_.reserve(patches_info.size());
    double t_local_fit = 0;
    std::vector<int> patch_sizes;

    for (auto& pi : patches_info) {
        if ((int)pi.ext_idx.size() < min_pts) continue;

        Eigen::MatrixXd local_pts(pi.ext_idx.size(), dim);
        Eigen::VectorXd local_vals(pi.ext_idx.size());
        for (int i = 0; i < (int)pi.ext_idx.size(); i++) {
            local_pts.row(i) = pts.row(pi.ext_idx[i]);
            local_vals(i) = vals(pi.ext_idx[i]);
        }
        patch_sizes.push_back((int)pi.ext_idx.size());

        auto tf0 = clock::now();
        auto interp = std::make_unique<DuchonInterpolator>(kernel_);
        interp->fit(local_pts, local_vals);
        t_local_fit += std::chrono::duration<double>(clock::now() - tf0).count();

        Patch patch;
        patch.center = pi.center;
        patch.half_ext = pi.half_ext;
        patch.bsphere_radius = pi.half_ext.norm();
        patch.interp = std::move(interp);
        patches_.push_back(std::move(patch));
    }

    auto t3 = clock::now();
    std::cout << "  [PU fit] patch loop: "
              << std::chrono::duration<double>(t3-t2).count() << "s  (local fit: "
              << t_local_fit << "s)\n";
    if (!patch_sizes.empty()) {
        int ps_min = *std::min_element(patch_sizes.begin(), patch_sizes.end());
        int ps_max = *std::max_element(patch_sizes.begin(), patch_sizes.end());
        double ps_mean = std::accumulate(patch_sizes.begin(), patch_sizes.end(), 0.0) / patch_sizes.size();
        std::cout << "  [PU fit] patch sizes: min=" << ps_min
                  << ", max=" << ps_max << ", mean=" << (int)ps_mean << "\n";
    }

    // Build patch center KDTree for fallback
    int np = (int)patches_.size();
    patch_centers_.resize(np, 3);
    patch_radii_.resize(np);
    for (int i = 0; i < np; i++) {
        patch_centers_.row(i) = patches_[i].center.transpose();
        patch_radii_(i) = patches_[i].bsphere_radius;
    }

    trained_ = true;
    std::cout << "  [PU fit] total: "
              << std::chrono::duration<double>(clock::now()-t0).count() << "s  ("
              << patches_.size() << " patches)\n";
}

// ── predict ─────────────────────────────────────────────────────────

Eigen::VectorXd PUInterpolator::predict(const Eigen::MatrixXd& x_new, int chunk_size) const {
    int M = (int)x_new.rows();
    if (M > chunk_size) {
        Eigen::VectorXd result(M);
        for (int s = 0; s < M; s += chunk_size) {
            int len = std::min(chunk_size, M - s);
            result.segment(s, len) = predict(x_new.middleRows(s, len), chunk_size);
        }
        return result;
    }

    Eigen::VectorXd result = Eigen::VectorXd::Zero(M);
    Eigen::VectorXd weight_sum = Eigen::VectorXd::Zero(M);

    for (const auto& patch : patches_) {
        if (use_box_) {
            // Find points inside this box
            std::vector<int> idx;
            for (int i = 0; i < M; i++) {
                Eigen::Vector3d diff = (x_new.row(i).transpose() - patch.center).cwiseAbs();
                if ((diff.array() <= patch.half_ext.array()).all())
                    idx.push_back(i);
            }
            if (idx.empty()) continue;

            Eigen::MatrixXd pts(idx.size(), 3);
            for (int i = 0; i < (int)idx.size(); i++)
                pts.row(i) = x_new.row(idx[i]);

            Eigen::VectorXd v = patch.interp->predict(pts);
            for (int i = 0; i < (int)idx.size(); i++) {
                double w = box_weight(x_new.row(idx[i]).transpose(), patch.center, patch.half_ext);
                result(idx[i]) += w * v(i);
                weight_sum(idx[i]) += w;
            }
        } else {
            double R = patch.half_ext(0);  // sphere radius
            std::vector<int> idx;
            for (int i = 0; i < M; i++) {
                double dist = (x_new.row(i).transpose() - patch.center).norm();
                if (dist <= R) idx.push_back(i);
            }
            if (idx.empty()) continue;

            Eigen::MatrixXd pts(idx.size(), 3);
            for (int i = 0; i < (int)idx.size(); i++)
                pts.row(i) = x_new.row(idx[i]);

            Eigen::VectorXd v = patch.interp->predict(pts);
            for (int i = 0; i < (int)idx.size(); i++) {
                double dist = (x_new.row(idx[i]).transpose() - patch.center).norm();
                double w = wendland_weight(dist, R);
                result(idx[i]) += w * v(i);
                weight_sum(idx[i]) += w;
            }
        }
    }

    // Normalize by weight sum
    for (int i = 0; i < M; i++) {
        if (weight_sum(i) > 0)
            result(i) /= weight_sum(i);
    }

    // Fallback: uncovered points use nearest patch
    KDTree3D ptree(patch_centers_);
    for (int i = 0; i < M; i++) {
        if (weight_sum(i) <= 0) {
            Eigen::Vector3d pt = x_new.row(i);
            auto [dist_sq, nearest] = ptree.query_nearest(pt);
            Eigen::MatrixXd qpt = x_new.middleRows(i, 1);
            result(i) = patches_[nearest].interp->predict(qpt)(0);
        }
    }

    return result;
}

// ── predict_gradients ───────────────────────────────────────────────

Eigen::MatrixXd PUInterpolator::predict_gradients(const Eigen::MatrixXd& x_new, int chunk_size) const {
    int M = (int)x_new.rows();
    if (M > chunk_size) {
        Eigen::MatrixXd result(M, 3);
        for (int s = 0; s < M; s += chunk_size) {
            int len = std::min(chunk_size, M - s);
            result.middleRows(s, len) = predict_gradients(x_new.middleRows(s, len), chunk_size);
        }
        return result;
    }

    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(M, 3);
    Eigen::VectorXd weight_sum = Eigen::VectorXd::Zero(M);

    for (const auto& patch : patches_) {
        if (use_box_) {
            std::vector<int> idx;
            for (int i = 0; i < M; i++) {
                Eigen::Vector3d diff = (x_new.row(i).transpose() - patch.center).cwiseAbs();
                if ((diff.array() <= patch.half_ext.array()).all())
                    idx.push_back(i);
            }
            if (idx.empty()) continue;

            Eigen::MatrixXd pts(idx.size(), 3);
            for (int i = 0; i < (int)idx.size(); i++)
                pts.row(i) = x_new.row(idx[i]);

            Eigen::MatrixXd g = patch.interp->predict_gradients(pts);
            for (int i = 0; i < (int)idx.size(); i++) {
                double w = box_weight(x_new.row(idx[i]).transpose(), patch.center, patch.half_ext);
                result.row(idx[i]) += w * g.row(i);
                weight_sum(idx[i]) += w;
            }
        } else {
            double R = patch.half_ext(0);
            std::vector<int> idx;
            for (int i = 0; i < M; i++) {
                double dist = (x_new.row(i).transpose() - patch.center).norm();
                if (dist <= R) idx.push_back(i);
            }
            if (idx.empty()) continue;

            Eigen::MatrixXd pts(idx.size(), 3);
            for (int i = 0; i < (int)idx.size(); i++)
                pts.row(i) = x_new.row(idx[i]);

            Eigen::MatrixXd g = patch.interp->predict_gradients(pts);
            for (int i = 0; i < (int)idx.size(); i++) {
                double dist = (x_new.row(idx[i]).transpose() - patch.center).norm();
                double w = wendland_weight(dist, R);
                result.row(idx[i]) += w * g.row(i);
                weight_sum(idx[i]) += w;
            }
        }
    }

    for (int i = 0; i < M; i++) {
        if (weight_sum(i) > 0)
            result.row(i) /= weight_sum(i);
    }

    // Fallback for uncovered
    KDTree3D ptree(patch_centers_);
    for (int i = 0; i < M; i++) {
        if (weight_sum(i) <= 0) {
            Eigen::Vector3d pt = x_new.row(i);
            auto [dist_sq, nearest] = ptree.query_nearest(pt);
            Eigen::MatrixXd qpt = x_new.middleRows(i, 1);
            result.row(i) = patches_[nearest].interp->predict_gradients(qpt).row(0);
        }
    }

    return result;
}

}  // namespace sdf
