#include "pu_interpolator.h"
#include "kdtree.h"
#include "dedup.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <deque>
#include <unordered_set>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace sdf {

PUInterpolator::PUInterpolator(const std::string& kernel, double overlap,
                                 int min_points, int max_points,
                                 double reg, const std::string& partition,
                                 bool verbose, bool pair_local)
    : kernel_(kernel), partition_type_(partition), overlap_(overlap),
      min_points_(min_points), max_points_(max_points), reg_(reg)
{
    verbose_ = verbose;
    use_box_ = (partition == "box");
    pair_local_ = pair_local;
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

// ── Weight value + spatial gradient ─────────────────────────────────
// The PU field is f = Σ w_p f_p / Σ w_p, whose gradient
//   ∇f = (Σ w_p ∇f_p + Σ ∇w_p f_p − f Σ ∇w_p) / W
// needs ∇w_p, the spatial derivative of each partition weight. These
// helpers return the same w as wendland_weight()/box_weight() and also
// fill `grad` with ∇w.
namespace {

// Wendland C2 profile g(s) = (1-s)^4 (4s+1) on s∈[0,1], and g'(s) = -20 s (1-s)^3.
inline double wprofile(double s)      { double t = 1.0 - s; return t*t*t*t*(4.0*s + 1.0); }
inline double wprofile_grad(double s) { double t = 1.0 - s; return -20.0 * s * t*t*t; }

inline double wendland_weight_grad(const Eigen::Vector3d& pt,
                                    const Eigen::Vector3d& center,
                                    double R, Eigen::Vector3d& grad) {
    Eigen::Vector3d d = pt - center;
    double r = d.norm();
    double s = std::clamp(r / R, 0.0, 1.0);
    double w = wprofile(s);
    // dw/dr = g'(s)/R, ∇r = d/r. Zero at r≈0 (g'(0)=0) and for s≥1 (outside support).
    if (r > 1e-12 && s < 1.0)
        grad = (wprofile_grad(s) / (R * r)) * d;
    else
        grad.setZero();
    return w;
}

inline double box_weight_grad(const Eigen::Vector3d& pt,
                              const Eigen::Vector3d& center,
                              const Eigen::Vector3d& half_ext,
                              Eigen::Vector3d& grad) {
    double g[3], gp[3];
    double w = 1.0;
    for (int k = 0; k < 3; k++) {
        double diff = pt(k) - center(k);
        double s = std::clamp(std::abs(diff) / half_ext(k), 0.0, 1.0);
        g[k]  = wprofile(s);
        // ∂s/∂x_k = sign(diff)/h_k for s∈(0,1), else 0.
        gp[k] = (s < 1.0) ? wprofile_grad(s) * ((diff >= 0) ? 1.0 : -1.0) / half_ext(k)
                          : 0.0;
        w *= g[k];
    }
    // ∂w/∂x_k = (Π_{j≠k} g_j) · ∂g_k/∂x_k
    grad(0) = g[1] * g[2] * gp[0];
    grad(1) = g[0] * g[2] * gp[1];
    grad(2) = g[0] * g[1] * gp[2];
    return w;
}

}  // namespace

// ── Deduplication ───────────────────────────────────────────────────

void PUInterpolator::deduplicate(Eigen::MatrixXd& points, Eigen::VectorXd& values,
                                  double tol, std::vector<int>* partner) const {
    int n = (int)points.rows();
    if (n == 0) return;

    // Shared greedy spatial dedup (non-zero values survive over zero-valued
    // projection duplicates). PU additionally remaps `partner` below.
    std::vector<int> kept = dedup_keep_indices(points, values, tol);

    Eigen::MatrixXd new_pts(kept.size(), points.cols());
    Eigen::VectorXd new_vals(kept.size());
    for (int i = 0; i < (int)kept.size(); i++) {
        new_pts.row(i) = points.row(kept[i]);
        new_vals(i) = values(kept[i]);
    }

    // Remap partner indices onto the surviving rows.
    if (partner) {
        std::vector<int> old_to_new(n, -1);
        for (int i = 0; i < (int)kept.size(); i++) old_to_new[kept[i]] = i;
        std::vector<int> new_partner(kept.size());
        for (int i = 0; i < (int)kept.size(); i++) {
            int p = (*partner)[kept[i]];
            new_partner[i] = (p >= 0) ? old_to_new[p] : -1;
        }
        *partner = std::move(new_partner);
    }

    if (verbose_)
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

    // partner[i] = row index (into pts/vals) of point i's paired point, or -1.
    // An input point and its surface projection form a pair: together they
    // encode the gradient (direction + signed distance) at that location.
    // We track pairing through filtering/dedup so each local RBF solve can be
    // augmented with any missing partner — see the patch loop below.
    std::vector<int> partner;

    // Add projection points if gradients given
    if (gradients) {
        int N = (int)points.rows();
        // proj_src[k] = input row index that produced projection k.
        std::vector<int> proj_src;
        Eigen::MatrixXd projections;
        if (mask) {
            int count = mask->sum();
            projections.resize(count, points.cols());
            proj_src.reserve(count);
            int k = 0;
            for (int i = 0; i < N; i++) {
                if ((*mask)(i)) {
                    projections.row(k++) = points.row(i) - values(i) * gradients->row(i);
                    proj_src.push_back(i);
                }
            }
        } else {
            projections.resize(N, points.cols());
            proj_src.reserve(N);
            for (int i = 0; i < N; i++) {
                projections.row(i) = points.row(i) - values(i) * gradients->row(i);
                proj_src.push_back(i);
            }
        }
        int M = (int)projections.rows();
        pts.resize(N + M, points.cols());
        pts << points, projections;
        vals.resize(N + M);
        vals << values, Eigen::VectorXd::Zero(M);

        // Link each input to its projection and vice versa. Inputs without a
        // projection (mask == 0) stay unpaired (-1).
        partner.assign(N + M, -1);
        for (int k = 0; k < M; k++) {
            partner[proj_src[k]] = N + k;   // input -> its projection
            partner[N + k] = proj_src[k];   // projection -> its input
        }
    } else {
        pts = points;
        vals = values;
        partner.assign((int)points.rows(), -1);
    }

    // Filter by distance if too many
    if (vals.size() > 5000) {
        std::vector<int> keep;
        for (int i = 0; i < (int)vals.size(); i++)
            if (std::abs(vals(i)) < dist_threshold_) keep.push_back(i);
        Eigen::MatrixXd pf(keep.size(), pts.cols());
        Eigen::VectorXd vf(keep.size());
        std::vector<int> old_to_new(vals.size(), -1);
        for (int i = 0; i < (int)keep.size(); i++) old_to_new[keep[i]] = i;
        std::vector<int> pf_partner(keep.size());
        for (int i = 0; i < (int)keep.size(); i++) {
            pf.row(i) = pts.row(keep[i]);
            vf(i) = vals(keep[i]);
            int p = partner[keep[i]];
            pf_partner[i] = (p >= 0) ? old_to_new[p] : -1;
        }
        pts = pf;
        vals = vf;
        partner = std::move(pf_partner);
    }

    // Deduplicate
    deduplicate(pts, vals, 5e-4, &partner);
    if (vals.size() > 5000 && verbose_) {
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
    if (verbose_)
        std::cout << "  [PU fit] build KDTree: "
                  << std::chrono::duration<double>(t1-t0).count() << "s\n";

    // ── Partition ───────────────────────────────────────────────────
    auto patches_info = kdtree_partition(pts, tree);

    auto t2 = clock::now();
    if (verbose_)
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

        // Pair the local solve: the patch shape (ext_idx, center, half_ext)
        // is left untouched, but for every point in the patch whose partner
        // (input <-> projection) is absent, we add that partner as an extra
        // RBF constraint so each point's gradient information appears as a
        // complete pair. This only affects the local solve, not the blend.
        std::vector<int> local_idx = pi.ext_idx;
        if (pair_local_) {
            std::unordered_set<int> present(pi.ext_idx.begin(), pi.ext_idx.end());
            for (int gi : pi.ext_idx) {
                int q = partner[gi];
                if (q >= 0 && present.insert(q).second)
                    local_idx.push_back(q);
            }
        }

        Eigen::MatrixXd local_pts(local_idx.size(), dim);
        Eigen::VectorXd local_vals(local_idx.size());
        for (int i = 0; i < (int)local_idx.size(); i++) {
            local_pts.row(i) = pts.row(local_idx[i]);
            local_vals(i) = vals(local_idx[i]);
        }
        tmp_sizes[p] = (int)local_idx.size();

        auto tf0 = clock::now();
        auto interp = std::make_unique<DuchonInterpolator>(kernel_, reg_);
        interp->set_dedup(false);   // PU already deduped globally; avoid redundant per-patch dedup
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
    if (verbose_)
        std::cout << "  [PU fit] patch loop: "
                  << std::chrono::duration<double>(t3-t2).count() << "s  (local fit: "
                  << t_local_fit << "s)\n";
    if (!patch_sizes.empty() && verbose_) {
        int ps_min = *std::min_element(patch_sizes.begin(), patch_sizes.end());
        int ps_max = *std::max_element(patch_sizes.begin(), patch_sizes.end());
        double ps_mean = std::accumulate(patch_sizes.begin(), patch_sizes.end(), 0.0) / patch_sizes.size();
        std::cout << "  [PU fit] patch sizes: min=" << ps_min
                  << ", max=" << ps_max << ", mean=" << (int)ps_mean << "\n";
    }

    // Optional: dump per-fit stats as JSONL when PU_FIT_LOG is set.
    // One line per fit() call: {"n_constraints":N,"n_patches":P,"patch_sizes":[...]}.
    // Caller clears the file between runs and reads the last line for the
    // last iteration's fit.
    if (const char* log_path = std::getenv("PU_FIT_LOG")) {
        std::ofstream f(log_path, std::ios::app);
        if (f) {
            f << "{\"n_constraints\":" << (int)pts.rows()
              << ",\"n_patches\":" << (int)patch_sizes.size()
              << ",\"patch_sizes\":[";
            for (size_t i = 0; i < patch_sizes.size(); i++) {
                if (i) f << ',';
                f << patch_sizes[i];
            }
            f << "]}\n";
        }
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
    if (verbose_)
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
    if (M == 0) return result;

    // Partition-of-unity gradient needs the weight-derivative term, so we
    // accumulate five quantities per query point and combine at the end:
    //   W   = Σ w_p              (scalar)
    //   Vw  = Σ w_p f_p          (scalar, = W·f)
    //   G   = Σ w_p ∇f_p         (vec3)
    //   A   = Σ ∇w_p f_p         (vec3)
    //   B   = Σ ∇w_p             (vec3)
    //   ∇f = (G + A − f·B) / W,  f = Vw / W
    Eigen::VectorXd W  = Eigen::VectorXd::Zero(M);
    Eigen::VectorXd Vw = Eigen::VectorXd::Zero(M);
    Eigen::MatrixXd G  = Eigen::MatrixXd::Zero(M, 3);
    Eigen::MatrixXd A  = Eigen::MatrixXd::Zero(M, 3);
    Eigen::MatrixXd B  = Eigen::MatrixXd::Zero(M, 3);

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

        // Both the local value f_p and gradient ∇f_p are needed.
        Eigen::MatrixXd g = patch.interp->predict_gradients(pts);
        Eigen::VectorXd fp = patch.interp->predict(pts);
        double R = patch.half_ext(0);

        for (int i = 0; i < (int)idx.size(); i++) {
            Eigen::Vector3d pt = x_new.row(idx[i]).transpose();
            Eigen::Vector3d dw;
            double w = use_box_ ? box_weight_grad(pt, patch.center, patch.half_ext, dw)
                                : wendland_weight_grad(pt, patch.center, R, dw);
            double f = fp(i);
            int r = idx[i];
            #pragma omp atomic
            W(r)  += w;
            #pragma omp atomic
            Vw(r) += w * f;
            #pragma omp atomic
            G(r, 0) += w * g(i, 0);
            #pragma omp atomic
            G(r, 1) += w * g(i, 1);
            #pragma omp atomic
            G(r, 2) += w * g(i, 2);
            #pragma omp atomic
            A(r, 0) += dw(0) * f;
            #pragma omp atomic
            A(r, 1) += dw(1) * f;
            #pragma omp atomic
            A(r, 2) += dw(2) * f;
            #pragma omp atomic
            B(r, 0) += dw(0);
            #pragma omp atomic
            B(r, 1) += dw(1);
            #pragma omp atomic
            B(r, 2) += dw(2);
        }
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        if (W(i) > 0) {
            double f = Vw(i) / W(i);
            result.row(i) = (G.row(i) + A.row(i) - f * B.row(i)) / W(i);
        }
    }

    // Fallback: uncovered points sit outside every patch support, where ∇w_p=0
    // for all patches, so the nearest patch's raw gradient is already correct.
    #pragma omp parallel for schedule(dynamic, 256)
    for (int i = 0; i < M; i++) {
        if (W(i) <= 0) {
            Eigen::Vector3d pt = x_new.row(i);
            auto [dist_sq, nearest] = patch_tree_->query_nearest(pt);
            Eigen::MatrixXd qpt = x_new.middleRows(i, 1);
            result.row(i) = patches_[nearest].interp->predict_gradients(qpt).row(0);
        }
    }

    return result;
}

}  // namespace sdf
