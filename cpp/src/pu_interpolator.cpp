#include "pu_interpolator.h"
#include "kdtree.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <deque>
#include <iostream>
#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace sdf {

PUInterpolator::PUInterpolator(const std::string& kernel, double overlap,
                                 int min_points, int max_points,
                                 double reg, const std::string& partition)
    : kernel_(kernel), partition_type_(partition), overlap_(overlap),
      min_points_(min_points), max_points_(max_points), reg_(reg)
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
    std::cout << "  [PU fit] deduplication removed: " << (n - kept.size()) << " points\n";
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

// ── kdtree_partition ────────────────────────────────────────────────

std::vector<PUInterpolator::PatchInfo> PUInterpolator::kdtree_partition(
    const Eigen::MatrixXd& pts, KDTree3D& tree)
{
    int n = (int)pts.rows();
    std::vector<int> all_idx(n);
    std::iota(all_idx.begin(), all_idx.end(), 0);

    std::vector<std::vector<int>> leaves;
    subdivide_recursive(pts, all_idx, max_points_, leaves);

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

    return patches_info;
}

// ── fit ─────────────────────────────────────────────────────────────

void PUInterpolator::fit(const Eigen::MatrixXd& points,
                          const Eigen::VectorXd& values,
                          const Eigen::MatrixXd* gradients,
                          const Eigen::VectorXi* mask) {
    sample_points_ = points;
    sample_values_ = values;

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
        pts = pf;
        vals = vf;
    }

    // Deduplicate
    deduplicate(pts, vals, 1e-4);
    if (vals.size() > 5000) {
        std::cout << "  [PU fit] Warning: too many points, only keeping " << pts.rows()
                  << " points with abs(value) < " << dist_threshold_ << " for fitting.\n";
    }
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
    auto patches_info = kdtree_partition(pts, tree);

    auto t2 = clock::now();
    std::cout << "  [PU fit] greedy cover: "
              << std::chrono::duration<double>(t2-t1).count() << "s  ("
              << patches_info.size() << " patches)\n";

    // ── Fit local interpolators ────────────────────────────────────
    // Each patch fit is independent. Parallelize over patches and compact
    // into patches_ in a serial pass. t_local_fit becomes a sum over
    // threads (CPU-time-like), no longer wall time.
    patches_.clear();
    patches_.reserve(patches_info.size());
    double t_local_fit = 0;
    std::vector<int> patch_sizes;

    const int P = (int)patches_info.size();
    std::vector<Patch> tmp_patches(P);
    std::vector<int> tmp_sizes(P, 0);
    std::vector<char> valid(P, 0);

    #pragma omp parallel for schedule(dynamic, 4) reduction(+:t_local_fit)
    for (int p = 0; p < P; p++) {
        auto& pi = patches_info[p];
        if ((int)pi.ext_idx.size() < min_pts) continue;

        Eigen::MatrixXd local_pts(pi.ext_idx.size(), dim);
        Eigen::VectorXd local_vals(pi.ext_idx.size());
        for (int i = 0; i < (int)pi.ext_idx.size(); i++) {
            local_pts.row(i) = pts.row(pi.ext_idx[i]);
            local_vals(i) = vals(pi.ext_idx[i]);
        }
        tmp_sizes[p] = (int)pi.ext_idx.size();

        auto tf0 = clock::now();
        auto interp = std::make_unique<DuchonInterpolator>(kernel_, reg_);
        interp->fit(local_pts, local_vals);
        t_local_fit += std::chrono::duration<double>(clock::now() - tf0).count();

        tmp_patches[p].center = pi.center;
        tmp_patches[p].half_ext = pi.half_ext;
        tmp_patches[p].bsphere_radius = pi.half_ext.norm();
        tmp_patches[p].interp = std::move(interp);
        valid[p] = 1;
    }

    for (int p = 0; p < P; p++) {
        if (!valid[p]) continue;
        patches_.push_back(std::move(tmp_patches[p]));
        patch_sizes.push_back(tmp_sizes[p]);
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

    // Cache the patch-center KDTree so predict() never rebuilds it.
    patch_tree_ = std::make_unique<KDTree3D>(patch_centers_);

    // Build BVH over patch AABBs for fast point-in-patch queries.
    build_patch_bvh();

    trained_ = true;
    std::cout << "  [PU fit] total: "
              << std::chrono::duration<double>(clock::now()-t0).count() << "s  ("
              << patches_.size() << " patches)\n";
}

// ── Patch BVH ───────────────────────────────────────────────────────
//
// Each patch has an AABB: center ± half_ext for box mode, center ± R for
// sphere mode. A query point is inside a patch iff it lies in the AABB
// (exact for box; a conservative superset for sphere, so callers must
// still do the dist ≤ R check after).
static void patch_bvh_build_recursive(
    std::vector<PUInterpolator::PatchBVHNode>& nodes,
    std::vector<int>& leaf_indices,
    const double* lo_flat, const double* hi_flat,
    int* idx_buf, int start, int count, int leaf_size)
{
    int node_id = (int)nodes.size();
    nodes.push_back({});

    double lo0 = 1e300, lo1 = 1e300, lo2 = 1e300;
    double hi0 = -1e300, hi1 = -1e300, hi2 = -1e300;
    for (int k = start; k < start + count; k++) {
        int i = idx_buf[k];
        lo0 = std::min(lo0, lo_flat[i*3+0]); hi0 = std::max(hi0, hi_flat[i*3+0]);
        lo1 = std::min(lo1, lo_flat[i*3+1]); hi1 = std::max(hi1, hi_flat[i*3+1]);
        lo2 = std::min(lo2, lo_flat[i*3+2]); hi2 = std::max(hi2, hi_flat[i*3+2]);
    }
    nodes[node_id].lo[0] = lo0; nodes[node_id].lo[1] = lo1; nodes[node_id].lo[2] = lo2;
    nodes[node_id].hi[0] = hi0; nodes[node_id].hi[1] = hi1; nodes[node_id].hi[2] = hi2;

    if (count <= leaf_size) {
        nodes[node_id].left = -1;
        nodes[node_id].right = -1;
        nodes[node_id].leaf_start = (int)leaf_indices.size();
        nodes[node_id].leaf_count = count;
        for (int k = start; k < start + count; k++)
            leaf_indices.push_back(idx_buf[k]);
        return;
    }

    double ext0 = hi0 - lo0, ext1 = hi1 - lo1, ext2 = hi2 - lo2;
    int axis = 0;
    if (ext1 > ext0 && ext1 >= ext2) axis = 1;
    else if (ext2 > ext0 && ext2 > ext1) axis = 2;

    // Sort by AABB center on chosen axis
    std::sort(idx_buf + start, idx_buf + start + count, [&](int a, int b) {
        double ca = 0.5 * (lo_flat[a*3+axis] + hi_flat[a*3+axis]);
        double cb = 0.5 * (lo_flat[b*3+axis] + hi_flat[b*3+axis]);
        return ca < cb;
    });

    int mid = count / 2;
    nodes[node_id].leaf_start = -1;
    nodes[node_id].leaf_count = 0;

    int left_id = (int)nodes.size();
    patch_bvh_build_recursive(nodes, leaf_indices, lo_flat, hi_flat,
                              idx_buf, start, mid, leaf_size);
    int right_id = (int)nodes.size();
    patch_bvh_build_recursive(nodes, leaf_indices, lo_flat, hi_flat,
                              idx_buf, start + mid, count - mid, leaf_size);
    nodes[node_id].left = left_id;
    nodes[node_id].right = right_id;
}

void PUInterpolator::build_patch_bvh() {
    int np = (int)patches_.size();
    patch_bvh_nodes_.clear();
    patch_bvh_leaves_.clear();
    patch_aabb_lo_.assign(3 * np, 0.0);
    patch_aabb_hi_.assign(3 * np, 0.0);
    if (np == 0) return;

    for (int i = 0; i < np; i++) {
        const auto& p = patches_[i];
        // Sphere mode: half_ext = (R,R,R) already (set in fit()); box mode:
        // per-axis half-extents. Either way AABB = center ± half_ext.
        for (int k = 0; k < 3; k++) {
            patch_aabb_lo_[i*3+k] = p.center(k) - p.half_ext(k);
            patch_aabb_hi_[i*3+k] = p.center(k) + p.half_ext(k);
        }
    }
    // Sphere mode stores half_ext as (R,R,R)? Check: in kdtree_partition the
    // sphere branch pushes {center, Constant(R_ext), ext_idx}, so yes.

    std::vector<int> idx_buf(np);
    std::iota(idx_buf.begin(), idx_buf.end(), 0);
    patch_bvh_nodes_.reserve(2 * np);
    patch_bvh_leaves_.reserve(np);
    patch_bvh_build_recursive(patch_bvh_nodes_, patch_bvh_leaves_,
                              patch_aabb_lo_.data(), patch_aabb_hi_.data(),
                              idx_buf.data(), 0, np, 8);
}

void PUInterpolator::query_patches_containing(
    const Eigen::Vector3d& pt, std::vector<int>& out) const
{
    out.clear();
    if (patch_bvh_nodes_.empty()) return;
    double qx = pt(0), qy = pt(1), qz = pt(2);

    int stack[128];
    int sp = 0;
    stack[sp++] = 0;
    while (sp > 0) {
        int nid = stack[--sp];
        const auto& nd = patch_bvh_nodes_[nid];
        if (qx < nd.lo[0] || qx > nd.hi[0] ||
            qy < nd.lo[1] || qy > nd.hi[1] ||
            qz < nd.lo[2] || qz > nd.hi[2])
            continue;
        if (nd.left == -1) {
            for (int k = nd.leaf_start; k < nd.leaf_start + nd.leaf_count; k++) {
                int i = patch_bvh_leaves_[k];
                // Exact per-patch containment test.
                if (use_box_) {
                    const auto& p = patches_[i];
                    Eigen::Vector3d d = (pt - p.center).cwiseAbs();
                    if ((d.array() <= p.half_ext.array()).all())
                        out.push_back(i);
                } else {
                    const auto& p = patches_[i];
                    double R = p.half_ext(0);
                    if ((pt - p.center).squaredNorm() <= R * R)
                        out.push_back(i);
                }
            }
        } else {
            stack[sp++] = nd.left;
            stack[sp++] = nd.right;
        }
    }
}

// ── predict ─────────────────────────────────────────────────────────

Eigen::VectorXd PUInterpolator::predict(const Eigen::MatrixXd& x_new, int /*chunk_size*/) const {
    int M = (int)x_new.rows();
    Eigen::VectorXd result = Eigen::VectorXd::Zero(M);
    Eigen::VectorXd weight_sum = Eigen::VectorXd::Zero(M);
    if (M == 0) return result;

    // Pass 1 (parallel): for each query point, find containing patches via BVH
    // and invert into patch → list of query indices.
    int np = (int)patches_.size();
    std::vector<std::vector<int>> per_patch_idx(np);

    #pragma omp parallel
    {
        std::vector<std::vector<int>> local(np);
        std::vector<int> hits;
        #pragma omp for schedule(static)
        for (int i = 0; i < M; i++) {
            Eigen::Vector3d pt = x_new.row(i);
            query_patches_containing(pt, hits);
            for (int p : hits) local[p].push_back(i);
        }
        #pragma omp critical
        {
            for (int p = 0; p < np; p++) {
                auto& dst = per_patch_idx[p];
                auto& src = local[p];
                dst.insert(dst.end(), src.begin(), src.end());
            }
        }
    }

    // Pass 2 (parallel over patches): call patch.interp->predict on its batch,
    // then accumulate into result/weight_sum. Different patches may touch the
    // same query index, so accumulation needs atomics.
    #pragma omp parallel for schedule(dynamic, 1)
    for (int p = 0; p < np; p++) {
        const auto& idx = per_patch_idx[p];
        if (idx.empty()) continue;
        const auto& patch = patches_[p];

        Eigen::MatrixXd pts((int)idx.size(), 3);
        for (int i = 0; i < (int)idx.size(); i++)
            pts.row(i) = x_new.row(idx[i]);

        Eigen::VectorXd v = patch.interp->predict(pts);
        if (use_box_) {
            for (int i = 0; i < (int)idx.size(); i++) {
                double w = box_weight(x_new.row(idx[i]).transpose(),
                                      patch.center, patch.half_ext);
                double add_r = w * v(i);
                #pragma omp atomic
                result(idx[i]) += add_r;
                #pragma omp atomic
                weight_sum(idx[i]) += w;
            }
        } else {
            double R = patch.half_ext(0);
            for (int i = 0; i < (int)idx.size(); i++) {
                double dist = (x_new.row(idx[i]).transpose() - patch.center).norm();
                double w = wendland_weight(dist, R);
                double add_r = w * v(i);
                #pragma omp atomic
                result(idx[i]) += add_r;
                #pragma omp atomic
                weight_sum(idx[i]) += w;
            }
        }
    }

    // Normalize by weight sum
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        if (weight_sum(i) > 0)
            result(i) /= weight_sum(i);
    }

    // Fallback: uncovered points use nearest patch.
    #pragma omp parallel for schedule(dynamic, 256)
    for (int i = 0; i < M; i++) {
        if (weight_sum(i) <= 0) {
            Eigen::Vector3d pt = x_new.row(i);
            auto [dist_sq, nearest] = patch_tree_->query_nearest(pt);
            Eigen::MatrixXd qpt = x_new.middleRows(i, 1);
            result(i) = patches_[nearest].interp->predict(qpt)(0);
        }
    }

    return result;
}

// ── predict_gradients ───────────────────────────────────────────────

Eigen::MatrixXd PUInterpolator::predict_gradients(const Eigen::MatrixXd& x_new, int /*chunk_size*/) const {
    int M = (int)x_new.rows();
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(M, 3);
    Eigen::VectorXd weight_sum = Eigen::VectorXd::Zero(M);
    if (M == 0) return result;

    int np = (int)patches_.size();
    std::vector<std::vector<int>> per_patch_idx(np);

    #pragma omp parallel
    {
        std::vector<std::vector<int>> local(np);
        std::vector<int> hits;
        #pragma omp for schedule(static)
        for (int i = 0; i < M; i++) {
            Eigen::Vector3d pt = x_new.row(i);
            query_patches_containing(pt, hits);
            for (int p : hits) local[p].push_back(i);
        }
        #pragma omp critical
        {
            for (int p = 0; p < np; p++) {
                auto& dst = per_patch_idx[p];
                auto& src = local[p];
                dst.insert(dst.end(), src.begin(), src.end());
            }
        }
    }

    #pragma omp parallel for schedule(dynamic, 1)
    for (int p = 0; p < np; p++) {
        const auto& idx = per_patch_idx[p];
        if (idx.empty()) continue;
        const auto& patch = patches_[p];

        Eigen::MatrixXd pts((int)idx.size(), 3);
        for (int i = 0; i < (int)idx.size(); i++)
            pts.row(i) = x_new.row(idx[i]);

        Eigen::MatrixXd g = patch.interp->predict_gradients(pts);
        if (use_box_) {
            for (int i = 0; i < (int)idx.size(); i++) {
                double w = box_weight(x_new.row(idx[i]).transpose(),
                                      patch.center, patch.half_ext);
                double gx = w * g(i, 0), gy = w * g(i, 1), gz = w * g(i, 2);
                #pragma omp atomic
                result(idx[i], 0) += gx;
                #pragma omp atomic
                result(idx[i], 1) += gy;
                #pragma omp atomic
                result(idx[i], 2) += gz;
                #pragma omp atomic
                weight_sum(idx[i]) += w;
            }
        } else {
            double R = patch.half_ext(0);
            for (int i = 0; i < (int)idx.size(); i++) {
                double dist = (x_new.row(idx[i]).transpose() - patch.center).norm();
                double w = wendland_weight(dist, R);
                double gx = w * g(i, 0), gy = w * g(i, 1), gz = w * g(i, 2);
                #pragma omp atomic
                result(idx[i], 0) += gx;
                #pragma omp atomic
                result(idx[i], 1) += gy;
                #pragma omp atomic
                result(idx[i], 2) += gz;
                #pragma omp atomic
                weight_sum(idx[i]) += w;
            }
        }
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        if (weight_sum(i) > 0)
            result.row(i) /= weight_sum(i);
    }

    #pragma omp parallel for schedule(dynamic, 256)
    for (int i = 0; i < M; i++) {
        if (weight_sum(i) <= 0) {
            Eigen::Vector3d pt = x_new.row(i);
            auto [dist_sq, nearest] = patch_tree_->query_nearest(pt);
            Eigen::MatrixXd qpt = x_new.middleRows(i, 1);
            result.row(i) = patches_[nearest].interp->predict_gradients(qpt).row(0);
        }
    }

    return result;
}

}  // namespace sdf
