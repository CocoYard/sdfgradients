#include "interpolator.h"
#include "thread_policy.h"
#include <cassert>  // libigl's march_cube.cpp uses assert() without including <cassert>
#include <igl/marching_cubes.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
#include <set>
#include <chrono>
#include <iostream>

using sdf::thread_policy;
using sdf::set_threads;
using sdf::restore_threads;

namespace sdf {

// ── Point-in-sphere BVH over sample AABBs ───────────────────────────
// Finds the "best containing" sphere for a query point g: among all samples
// k with dist(g, p_k) < |s_k|, returns the one maximizing margin |s_k|-dist.
// KD-tree nearest is wrong here because a distant sample with large |s| can
// contain g while the geometrically-nearest sample does not.
namespace prefill_bvh {

struct Node {
    float lo[3], hi[3];
    int left, right;       // -1 if leaf
    int leaf_start, leaf_count;
};

struct Tree {
    std::vector<Node> nodes;
    std::vector<int> leaf_indices;
    const float* px;
    const float* py;
    const float* pz;
    const float* absv;   // |sample_values|
    const float* absv2;  // |sample_values|^2  (pre-squared for leaf test)
};

static void build(Tree& T, int* idx_buf, int start, int count, int leaf_size) {
    int nid = (int)T.nodes.size();
    T.nodes.push_back(Node());

    float lo0 = 1e30f, lo1 = 1e30f, lo2 = 1e30f;
    float hi0 = -1e30f, hi1 = -1e30f, hi2 = -1e30f;
    for (int k = start; k < start + count; k++) {
        int i = idx_buf[k];
        float r = T.absv[i];
        float x = T.px[i], y = T.py[i], z = T.pz[i];
        if (x - r < lo0) lo0 = x - r;  if (x + r > hi0) hi0 = x + r;
        if (y - r < lo1) lo1 = y - r;  if (y + r > hi1) hi1 = y + r;
        if (z - r < lo2) lo2 = z - r;  if (z + r > hi2) hi2 = z + r;
    }
    T.nodes[nid].lo[0] = lo0; T.nodes[nid].lo[1] = lo1; T.nodes[nid].lo[2] = lo2;
    T.nodes[nid].hi[0] = hi0; T.nodes[nid].hi[1] = hi1; T.nodes[nid].hi[2] = hi2;

    if (count <= leaf_size) {
        T.nodes[nid].left = -1;
        T.nodes[nid].right = -1;
        T.nodes[nid].leaf_start = (int)T.leaf_indices.size();
        T.nodes[nid].leaf_count = count;
        for (int k = start; k < start + count; k++)
            T.leaf_indices.push_back(idx_buf[k]);
        return;
    }

    float ext0 = hi0 - lo0, ext1 = hi1 - lo1, ext2 = hi2 - lo2;
    int axis = 0;
    if (ext1 > ext0 && ext1 >= ext2) axis = 1;
    else if (ext2 > ext0 && ext2 > ext1) axis = 2;
    const float* axp = (axis == 0) ? T.px : (axis == 1) ? T.py : T.pz;
    std::sort(idx_buf + start, idx_buf + start + count,
              [axp](int a, int b) { return axp[a] < axp[b]; });

    int mid = count / 2;
    T.nodes[nid].leaf_start = -1;
    T.nodes[nid].leaf_count = 0;
    int left_id = (int)T.nodes.size();
    build(T, idx_buf, start, mid, leaf_size);
    int right_id = (int)T.nodes.size();
    build(T, idx_buf, start + mid, count - mid, leaf_size);
    T.nodes[nid].left = left_id;
    T.nodes[nid].right = right_id;
}

// Walk the BVH, find the sample sphere containing g with the largest margin.
// Returns best margin (|s_k|-dist) and the sample index via best_idx, or
// (-1, -1) if no sphere contains g.
static inline float query(const Tree& T, float gx, float gy, float gz,
                          int& best_idx) {
    best_idx = -1;
    float best_margin = -1.0f;
    if (T.nodes.empty()) return best_margin;
    int stack[64];
    int sp = 0;
    stack[sp++] = 0;
    const Node* nodes = T.nodes.data();
    const int* leaves = T.leaf_indices.data();
    const float* px = T.px; const float* py = T.py; const float* pz = T.pz;
    const float* absv = T.absv; const float* absv2 = T.absv2;
    // Track best_d2 so we can prune leaf tests via the squared comparison
    // (margin > best ⇔ r - d > best ⇔ d < r - best ⇔ d² < (r-best)²).
    float best_d2 = 0.0f;
    while (sp > 0) {
        int nid = stack[--sp];
        const Node& nd = nodes[nid];
        if (gx < nd.lo[0] || gx > nd.hi[0] ||
            gy < nd.lo[1] || gy > nd.hi[1] ||
            gz < nd.lo[2] || gz > nd.hi[2])
            continue;
        if (nd.left == -1) {
            int end = nd.leaf_start + nd.leaf_count;
            for (int k = nd.leaf_start; k < end; k++) {
                int i = leaves[k];
                float dx = gx - px[i];
                float dy = gy - py[i];
                float dz = gz - pz[i];
                float d2 = dx*dx + dy*dy + dz*dz;
                if (d2 < absv2[i]) {
                    float r = absv[i];
                    float margin = r - std::sqrt(d2);
                    if (margin > best_margin) {
                        best_margin = margin;
                        best_d2 = d2;
                        best_idx = i;
                    }
                }
            }
        } else {
            stack[sp++] = nd.left;
            stack[sp++] = nd.right;
        }
    }
    (void)best_d2;
    return best_margin;
}

}  // namespace prefill_bvh

// ── Fibonacci sphere (uniform directions on S2) ────────────────────

static Eigen::MatrixXd fibonacci_sphere(int n) {
    Eigen::MatrixXd dirs(n, 3);
    const double golden = (1.0 + std::sqrt(5.0)) / 2.0;
    for (int i = 0; i < n; i++) {
        double theta = 2.0 * M_PI * i / golden;
        double phi = std::acos(1.0 - 2.0 * (i + 0.5) / n);
        dirs(i, 0) = std::sin(phi) * std::cos(theta);
        dirs(i, 1) = std::sin(phi) * std::sin(theta);
        dirs(i, 2) = std::cos(phi);
    }
    return dirs;
}

// ── Tangent frame for a set of directions ───────────────────────────

static void tangent_frame(const Eigen::MatrixXd& d,
                          Eigen::MatrixXd& t1, Eigen::MatrixXd& t2) {
    int N = (int)d.rows();
    t1.resize(N, 3);
    t2.resize(N, 3);
    for (int i = 0; i < N; i++) {
        Eigen::Vector3d di = d.row(i);
        Eigen::Vector3d ref = (std::abs(di(0)) < 0.9)
                              ? Eigen::Vector3d(1, 0, 0)
                              : Eigen::Vector3d(0, 1, 0);
        Eigen::Vector3d u = di.cross(ref);
        u.normalize();
        Eigen::Vector3d v = di.cross(u);
        t1.row(i) = u;
        t2.row(i) = v;
    }
}

// ── Cone directions around best_dirs ─────────────────────────────────

static Eigen::MatrixXd cone_dirs_for_point(const Eigen::Vector3d& best_dir,
                                            const Eigen::Vector3d& u,
                                            const Eigen::Vector3d& v,
                                            double half_angle, int n) {
    const double golden = (1.0 + std::sqrt(5.0)) / 2.0;
    Eigen::MatrixXd dirs(n, 3);
    for (int i = 0; i < n; i++) {
        double r = half_angle * std::sqrt((i + 0.5) / n);
        double alpha = 2.0 * M_PI * i / golden;
        double cos_r = std::cos(r), sin_r = std::sin(r);
        double cos_a = std::cos(alpha), sin_a = std::sin(alpha);
        dirs.row(i) = cos_r * best_dir + sin_r * (cos_a * u + sin_a * v);
    }
    return dirs;
}

// ── sample_best_gradients ────────────────────────────────────────────

Eigen::MatrixXd Interpolator::sample_best_gradients(
    const Eigen::MatrixXd& points,
    const Eigen::VectorXd& sdf_values,
    int num_coarse,
    int refine_steps,
    int num_refine,
    const Eigen::MatrixXd* initial_guess,
    int chunk_size) const
{
    int batch_size = (int)points.rows();
    Eigen::VectorXd sign(batch_size);
    for (int i = 0; i < batch_size; i++)
        sign(i) = (sdf_values(i) > 0) ? 1.0 : -1.0;

    // Determine which points have valid initial guesses
    std::vector<bool> valid_mask(batch_size, false);
    if (initial_guess) {
        for (int i = 0; i < batch_size; i++)
            valid_mask[i] = !initial_guess->row(i).array().isNaN().any();
    }

    Eigen::MatrixXd best_dirs = Eigen::MatrixXd::Zero(batch_size, 3);

    // Copy valid initial guesses
    if (initial_guess) {
        for (int i = 0; i < batch_size; i++)
            if (valid_mask[i])
                best_dirs.row(i) = initial_guess->row(i);
    }

    // Fibonacci sweep for invalid (or all if no initial_guess)
    std::vector<int> invalid_idx;
    for (int i = 0; i < batch_size; i++)
        if (!valid_mask[i]) invalid_idx.push_back(i);

    if (!invalid_idx.empty()) {
        int n_inv = (int)invalid_idx.size();
        Eigen::MatrixXd fib = fibonacci_sphere(num_coarse);  // (num_coarse, 3)

        // Build samples: for each invalid point, evaluate all fibonacci directions
        Eigen::MatrixXd samples(n_inv * num_coarse, 3);
        for (int ii = 0; ii < n_inv; ii++) {
            int i = invalid_idx[ii];
            for (int d = 0; d < num_coarse; d++) {
                samples.row(ii * num_coarse + d) =
                    points.row(i) - sdf_values(i) * fib.row(d);
            }
        }
        Eigen::VectorXd preds = predict(samples, chunk_size);

        // Find best direction per invalid point
        for (int ii = 0; ii < n_inv; ii++) {
            int i = invalid_idx[ii];
            double best_obj = std::numeric_limits<double>::max();
            int best_idx = 0;
            for (int d = 0; d < num_coarse; d++) {
                double obj = preds(ii * num_coarse + d) * sign(i);
                if (obj < best_obj) { best_obj = obj; best_idx = d; }
            }
            best_dirs.row(i) = fib.row(best_idx);
        }
    }

    // Iterative cone refinement
    double half_angle = M_PI / std::sqrt((double)num_coarse);

    Eigen::MatrixXd t1, t2;

    for (int step = 0; step < refine_steps; step++) {
        tangent_frame(best_dirs, t1, t2);

        // Build samples: for each point, evaluate num_refine directions in cone
        Eigen::MatrixXd samples(batch_size * num_refine, 3);
        // Store per-point cone directions to look up the best
        Eigen::MatrixXd all_cone_dirs(batch_size * num_refine, 3);

        for (int i = 0; i < batch_size; i++) {
            Eigen::MatrixXd cdirs = cone_dirs_for_point(
                best_dirs.row(i), t1.row(i), t2.row(i), half_angle, num_refine);
            for (int d = 0; d < num_refine; d++) {
                all_cone_dirs.row(i * num_refine + d) = cdirs.row(d);
                samples.row(i * num_refine + d) =
                    points.row(i) - sdf_values(i) * cdirs.row(d);
            }
        }

        Eigen::VectorXd preds = predict(samples, chunk_size);

        // Pick best per point
        for (int i = 0; i < batch_size; i++) {
            double best_obj = std::numeric_limits<double>::max();
            int best_local = 0;
            for (int d = 0; d < num_refine; d++) {
                double obj = preds(i * num_refine + d) * sign(i);
                if (obj < best_obj) { best_obj = obj; best_local = d; }
            }
            best_dirs.row(i) = all_cone_dirs.row(i * num_refine + best_local);
            double norm = best_dirs.row(i).norm();
            if (norm > 1e-15) best_dirs.row(i) /= norm;
        }

        half_angle /= std::sqrt((double)num_refine);
    }

    return best_dirs;
}

// ── optimize_best_gradients ─────────────────────────────────────────

Eigen::MatrixXd Interpolator::optimize_best_gradients(
    const Eigen::MatrixXd& points,
    const Eigen::VectorXd& sdf_values,
    int num_coarse,
    int optim_steps,
    double lr,
    const Eigen::MatrixXd* initial_guess,
    int chunk_size) const
{
    int N = (int)points.rows();

    // Bump to predict thread count for the duration: surrounding
    // iterative_projection_3d runs under fit's reduced pool, but this
    // routine is throughput-bound (N-wide loops + predict / predict_gradients).
    int _saved = set_threads(thread_policy().predict);

    Eigen::VectorXd sgn(N);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++)
        sgn(i) = (sdf_values(i) > 0) ? 1.0 : -1.0;

    std::vector<char> valid_mask(N, 0);
    if (initial_guess) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++)
            valid_mask[i] = !initial_guess->row(i).array().isNaN().any();
    }

    Eigen::MatrixXd dirs = Eigen::MatrixXd::Zero(N, 3);
    if (initial_guess) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++)
            if (valid_mask[i])
                dirs.row(i) = initial_guess->row(i);
    }

    // Fibonacci coarse sweep for points without a valid initial guess
    std::vector<int> invalid_idx;
    invalid_idx.reserve(N);
    for (int i = 0; i < N; i++)
        if (!valid_mask[i]) invalid_idx.push_back(i);

    if (!invalid_idx.empty()) {
        int n_inv = (int)invalid_idx.size();
        Eigen::MatrixXd fib = fibonacci_sphere(num_coarse);

        Eigen::MatrixXd samples(n_inv * num_coarse, 3);
        #pragma omp parallel for schedule(static)
        for (int ii = 0; ii < n_inv; ii++) {
            int i = invalid_idx[ii];
            for (int d = 0; d < num_coarse; d++)
                samples.row(ii * num_coarse + d) =
                    points.row(i) - sdf_values(i) * fib.row(d);
        }
        Eigen::VectorXd preds = predict(samples, chunk_size);

        #pragma omp parallel for schedule(static)
        for (int ii = 0; ii < n_inv; ii++) {
            int i = invalid_idx[ii];
            double best_obj = std::numeric_limits<double>::max();
            int best_idx = 0;
            for (int d = 0; d < num_coarse; d++) {
                double obj = preds(ii * num_coarse + d) * sgn(i);
                if (obj < best_obj) { best_obj = obj; best_idx = d; }
            }
            dirs.row(i) = fib.row(best_idx);
        }
    }

    // Projected gradient descent on S²
    // obj_i = sgn(i) * sdf(p_i - s_i * g_i)   — minimize
    // ∂obj/∂g = sgn(i) * (-s_i) * ∇sdf = -|s_i| * ∇sdf
    // tangent projection: grad_tan = grad - (grad · g) * g
    // retraction: g ← normalize(g - lr * grad_tan)
    Eigen::MatrixXd proj(N, 3);
    for (int step = 0; step < optim_steps; step++) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++)
            proj.row(i) = points.row(i) - sdf_values(i) * dirs.row(i);

        Eigen::MatrixXd sdf_grad = predict_gradients(proj, chunk_size);

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++) {
            double abs_s = std::abs(sdf_values(i));
            if (abs_s < 1e-15) continue;

            Eigen::RowVector3d raw = -abs_s * sdf_grad.row(i);
            double dot = raw.dot(dirs.row(i));
            Eigen::RowVector3d grad_tan = raw - dot * dirs.row(i);

            Eigen::RowVector3d updated = dirs.row(i) - lr * grad_tan;
            double norm = updated.norm();
            if (norm > 1e-15)
                dirs.row(i) = updated / norm;
        }
    }

    restore_threads(_saved);
    return dirs;
}

// ── extract_surface (narrow-band marching cubes via libigl) ──────────
//
// Two-pass strategy:
//   1. Coarse grid (~1/4 res per axis): one predict() call, builds a cheap
//      SDF approximation over the whole bbox.
//   2. Mark coarse cells that either straddle the iso level or sit within
//      ~2× coarse-cell-diagonal of it (Lipschitz safety margin). These are
//      the only cells that can produce triangles.
//   3. Fill the fine S array with trilinear interpolation from the coarse
//      values everywhere, then overwrite fine vertices inside active coarse
//      cells with a real predict() call. Inactive regions keep smooth,
//      same-sign interpolated values so libigl MC produces no spurious
//      crossings there.
//
// Net effect: fine predict() only runs on the ~O(n²) vertices that actually
// matter, cutting cost dramatically vs. a dense O(n³) predict.

void Interpolator::extract_surface(
    const Eigen::Vector3d& bbox_min,
    const Eigen::Vector3d& bbox_max,
    int nx, int ny, int nz,
    double iso,
    Eigen::MatrixXd& V,
    Eigen::MatrixXi& F,
    int chunk_size,
    bool lipschitz_postfix) const
{
    if (nx < 2 || ny < 2 || nz < 2)
        throw std::invalid_argument("extract_surface: nx, ny, nz must each be >= 2");

    using clk = std::chrono::steady_clock;
    auto t_total = clk::now();
    auto sec_since = [](const clk::time_point& t) {
        return std::chrono::duration<double>(clk::now() - t).count();
    };

#if 1  // plain marching cubes for speed comparison

    const int N = nx * ny * nz;
    const Eigen::Vector3d step(
        (bbox_max.x() - bbox_min.x()) / (nx - 1),
        (bbox_max.y() - bbox_min.y()) / (ny - 1),
        (bbox_max.z() - bbox_min.z()) / (nz - 1));

    auto fine_idx = [&](int xi, int yi, int zi) {
        return xi + nx * (yi + ny * zi);
    };

    // libigl marching_cubes expects GV.row(x + nx*(y + ny*z))
    auto t_gv = clk::now();
    Eigen::MatrixXd GV(N, 3);
    for (int zi = 0; zi < nz; zi++)
        for (int yi = 0; yi < ny; yi++)
            for (int xi = 0; xi < nx; xi++) {
                int idx = fine_idx(xi, yi, zi);
                GV(idx, 0) = bbox_min.x() + xi * step.x();
                GV(idx, 1) = bbox_min.y() + yi * step.y();
                GV(idx, 2) = bbox_min.z() + zi * step.z();
            }

    std::cout << "[extract_surface] build GV: " << sec_since(t_gv) << " s\n";

    // ── Coarse pass ──────────────────────────────────────────────────
    auto t_coarse = clk::now();
    const int R = 4;
    const int cnx = std::max(2, (nx + R - 1) / R);
    const int cny = std::max(2, (ny + R - 1) / R);
    const int cnz = std::max(2, (nz + R - 1) / R);
    const int CN = cnx * cny * cnz;
    const Eigen::Vector3d cstep(
        (bbox_max.x() - bbox_min.x()) / (cnx - 1),
        (bbox_max.y() - bbox_min.y()) / (cny - 1),
        (bbox_max.z() - bbox_min.z()) / (cnz - 1));

    auto cidx_at = [&](int ci, int cj, int ck) {
        return ci + cnx * (cj + cny * ck);
    };

    Eigen::MatrixXd CGV(CN, 3);
    for (int ck = 0; ck < cnz; ck++)
        for (int cj = 0; cj < cny; cj++)
            for (int ci = 0; ci < cnx; ci++) {
                int idx = cidx_at(ci, cj, ck);
                CGV(idx, 0) = bbox_min.x() + ci * cstep.x();
                CGV(idx, 1) = bbox_min.y() + cj * cstep.y();
                CGV(idx, 2) = bbox_min.z() + ck * cstep.z();
            }
    Eigen::VectorXd CS = this->predict(CGV, chunk_size);
    std::cout << "[extract_surface] coarse predict (" << CN
              << " pts): " << sec_since(t_coarse) << " s\n";

    // ── Fill fine S by trilinear interpolation from coarse ───────────
    auto t_tri = clk::now();
    Eigen::VectorXd S(N);
    for (int zi = 0; zi < nz; zi++) {
        double fz = (zi * step.z()) / cstep.z();
        int ck = std::min((int)fz, cnz - 2);
        double tz = fz - ck;
        for (int yi = 0; yi < ny; yi++) {
            double fy = (yi * step.y()) / cstep.y();
            int cj = std::min((int)fy, cny - 2);
            double ty = fy - cj;
            for (int xi = 0; xi < nx; xi++) {
                double fx = (xi * step.x()) / cstep.x();
                int ci = std::min((int)fx, cnx - 2);
                double tx = fx - ci;
                double c000 = CS(cidx_at(ci  , cj  , ck  ));
                double c100 = CS(cidx_at(ci+1, cj  , ck  ));
                double c010 = CS(cidx_at(ci  , cj+1, ck  ));
                double c110 = CS(cidx_at(ci+1, cj+1, ck  ));
                double c001 = CS(cidx_at(ci  , cj  , ck+1));
                double c101 = CS(cidx_at(ci+1, cj  , ck+1));
                double c011 = CS(cidx_at(ci  , cj+1, ck+1));
                double c111 = CS(cidx_at(ci+1, cj+1, ck+1));
                double c00 = c000*(1-tx) + c100*tx;
                double c10 = c010*(1-tx) + c110*tx;
                double c01 = c001*(1-tx) + c101*tx;
                double c11 = c011*(1-tx) + c111*tx;
                double c0  = c00 *(1-ty) + c10 *ty;
                double c1  = c01 *(1-ty) + c11 *ty;
                S(fine_idx(xi, yi, zi)) = c0*(1-tz) + c1*tz;
            }
        }
    }

    std::cout << "[extract_surface] trilinear fill S: " << sec_since(t_tri) << " s\n";

    // ── Narrow-band detection ────────────────────────────────────────
    auto t_nb = clk::now();
    // Active = sign change across cell OR any corner within `tau` of iso.
    // tau = 2 × coarse cell diagonal: a Lipschitz-1 safety margin wide
    // enough to catch features that slightly undershoot the coarse grid.
    const double tau = 2.0 * cstep.norm();

    std::vector<uint8_t> need_refine(N, 0);
    for (int ck = 0; ck < cnz - 1; ck++)
    for (int cj = 0; cj < cny - 1; cj++)
    for (int ci = 0; ci < cnx - 1; ci++) {
        double vmin =  std::numeric_limits<double>::infinity();
        double vmax = -std::numeric_limits<double>::infinity();
        double amin =  std::numeric_limits<double>::infinity();
        for (int dc = 0; dc < 8; dc++) {
            int ci2 = ci + (dc & 1);
            int cj2 = cj + ((dc >> 1) & 1);
            int ck2 = ck + ((dc >> 2) & 1);
            double v = CS(cidx_at(ci2, cj2, ck2));
            vmin = std::min(vmin, v);
            vmax = std::max(vmax, v);
            amin = std::min(amin, std::abs(v - iso));
        }
        bool has_cross = (vmin - iso) * (vmax - iso) <= 0.0;
        bool near_iso  = amin < tau;
        if (!has_cross && !near_iso) continue;

        // Fine-vertex range covering this coarse cell (inclusive, so a shared
        // boundary vertex is marked active by whichever neighbor is active).
        double xs =  ci      * cstep.x() / step.x();
        double xe = (ci + 1) * cstep.x() / step.x();
        double ys =  cj      * cstep.y() / step.y();
        double ye = (cj + 1) * cstep.y() / step.y();
        double zs =  ck      * cstep.z() / step.z();
        double ze = (ck + 1) * cstep.z() / step.z();
        int xi0 = std::max(0,      (int)std::floor(xs - 1e-9));
        int xi1 = std::min(nx - 1, (int)std::ceil (xe + 1e-9));
        int yi0 = std::max(0,      (int)std::floor(ys - 1e-9));
        int yi1 = std::min(ny - 1, (int)std::ceil (ye + 1e-9));
        int zi0 = std::max(0,      (int)std::floor(zs - 1e-9));
        int zi1 = std::min(nz - 1, (int)std::ceil (ze + 1e-9));
        for (int zi = zi0; zi <= zi1; zi++)
        for (int yi = yi0; yi <= yi1; yi++)
        for (int xi = xi0; xi <= xi1; xi++)
            need_refine[fine_idx(xi, yi, zi)] = 1;
    }

    std::cout << "[extract_surface] narrow-band mark: " << sec_since(t_nb) << " s\n";

    // ── Fine predict on narrow band ──────────────────────────────────
    auto t_fine = clk::now();
    std::vector<int> refine_idx;
    refine_idx.reserve(N / 4);
    for (int i = 0; i < N; i++)
        if (need_refine[i]) refine_idx.push_back(i);

    if (!refine_idx.empty()) {
        const int M = (int)refine_idx.size();
        Eigen::MatrixXd RGV(M, 3);
        for (int k = 0; k < M; k++)
            RGV.row(k) = GV.row(refine_idx[k]);
        int saved = set_threads(thread_policy().predict);
        Eigen::VectorXd RS = this->predict(RGV, chunk_size);
        restore_threads(saved);
        for (int k = 0; k < M; k++)
            S(refine_idx[k]) = RS(k);
    }

    std::cout << "[extract_surface] fine predict ("
              << refine_idx.size() << " pts): "
              << sec_since(t_fine) << " s\n";

    // ── Lipschitz post-fix on narrow band + adaptive expansion ──────
    // Walk BVH over sample AABBs for each narrow-band vertex. If predict()
    // violates the Lipschitz bound (wrong sign or magnitude too small),
    // correct it. When a sign flip is detected, the true surface may have
    // shifted outside the current narrow band, so we expand: mark the
    // neighboring coarse cells as active, predict their fine vertices, and
    // post-fix again. This prevents holes without scanning the whole grid.
    int n_sign_flip = 0, n_mag_clamp = 0, n_expanded = 0;
    if (lipschitz_postfix && sample_points_.rows() > 0 && !refine_idx.empty()) {
        auto t_fix = clk::now();
        const int nS = (int)sample_points_.rows();
        std::vector<float> spx(nS), spy(nS), spz(nS), sabs(nS), sabs2(nS);
        std::vector<signed char> ssign(nS);
        for (int i = 0; i < nS; i++) {
            spx[i] = (float)sample_points_(i, 0);
            spy[i] = (float)sample_points_(i, 1);
            spz[i] = (float)sample_points_(i, 2);
            float a = std::abs((float)sample_values_(i));
            sabs[i] = a;
            sabs2[i] = a * a;
            ssign[i] = (sample_values_(i) >= 0) ? 1 : -1;
        }
        prefill_bvh::Tree T;
        T.px = spx.data(); T.py = spy.data(); T.pz = spz.data();
        T.absv = sabs.data(); T.absv2 = sabs2.data();
        T.nodes.reserve(2 * nS / 16 + 16);
        T.leaf_indices.reserve(nS);
        std::vector<int> idx_buf(nS);
        for (int i = 0; i < nS; i++) idx_buf[i] = i;
        prefill_bvh::build(T, idx_buf.data(), 0, nS, 16);

        // Lambda: post-fix a set of vertices, return indices where sign flipped
        auto do_postfix = [&](const std::vector<int>& verts,
                              int& flips, int& clamps) -> std::vector<int> {
            const int M = (int)verts.size();
            std::vector<uint8_t> is_flip(M, 0);
            int local_flips = 0, local_clamps = 0;
            #pragma omp parallel for schedule(dynamic, 1024) \
                    reduction(+: local_flips, local_clamps)
            for (int j = 0; j < M; j++) {
                int idx = verts[j];
                float gx = (float)GV(idx, 0);
                float gy = (float)GV(idx, 1);
                float gz = (float)GV(idx, 2);
                int hit;
                float margin = prefill_bvh::query(T, gx, gy, gz, hit);
                if (hit < 0) continue;
                double s_sign = ssign[hit];
                double cur = S(idx);
                bool sign_ok = ((cur >= 0) ? 1.0 : -1.0) == s_sign;
                double abs_cur = sign_ok ? std::abs(cur) : 0.0;
                double fixed = s_sign * std::max(abs_cur, (double)margin);
                if (fixed != cur) {
                    if (!sign_ok) {
                        local_flips++;
                        is_flip[j] = 1;
                    } else {
                        local_clamps++;
                    }
                    S(idx) = fixed;
                }
            }
            flips += local_flips;
            clamps += local_clamps;
            std::vector<int> flip_verts;
            for (int j = 0; j < M; j++)
                if (is_flip[j]) flip_verts.push_back(verts[j]);
            return flip_verts;
        };

        // First pass: post-fix on the original narrow band
        auto flip_verts = do_postfix(refine_idx, n_sign_flip, n_mag_clamp);

        // Expansion: for each sign-flipped vertex, activate neighboring coarse
        // cells and predict + post-fix the newly added fine vertices.
        if (!flip_verts.empty()) {
            // Find coarse cells of flipped vertices and their neighbors
            std::set<int> expand_cells;
            const double inv_csx = 1.0 / cstep.x();
            const double inv_csy = 1.0 / cstep.y();
            const double inv_csz = 1.0 / cstep.z();
            for (int idx : flip_verts) {
                int ci = std::min((int)((GV(idx, 0) - bbox_min.x()) * inv_csx), cnx - 2);
                int cj = std::min((int)((GV(idx, 1) - bbox_min.y()) * inv_csy), cny - 2);
                int ck = std::min((int)((GV(idx, 2) - bbox_min.z()) * inv_csz), cnz - 2);
                // Mark this cell and its 26 neighbors
                for (int dz = -1; dz <= 1; dz++)
                for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++) {
                    int ni = ci + dx, nj = cj + dy, nk = ck + dz;
                    if (ni >= 0 && ni < cnx-1 && nj >= 0 && nj < cny-1 &&
                        nk >= 0 && nk < cnz-1)
                        expand_cells.insert(cidx_at(ni, nj, nk));
                }
            }

            // Collect new fine vertices from expanded cells
            std::vector<int> new_verts;
            for (int cell_id : expand_cells) {
                // Decode coarse cell index back to (ci, cj, ck)
                int ci = cell_id % cnx;
                int cj = (cell_id / cnx) % cny;
                int ck = cell_id / (cnx * cny);
                double xs =  ci      * cstep.x() / step.x();
                double xe = (ci + 1) * cstep.x() / step.x();
                double ys =  cj      * cstep.y() / step.y();
                double ye = (cj + 1) * cstep.y() / step.y();
                double zs =  ck      * cstep.z() / step.z();
                double ze = (ck + 1) * cstep.z() / step.z();
                int xi0 = std::max(0,      (int)std::floor(xs - 1e-9));
                int xi1 = std::min(nx - 1, (int)std::ceil (xe + 1e-9));
                int yi0 = std::max(0,      (int)std::floor(ys - 1e-9));
                int yi1 = std::min(ny - 1, (int)std::ceil (ye + 1e-9));
                int zi0 = std::max(0,      (int)std::floor(zs - 1e-9));
                int zi1 = std::min(nz - 1, (int)std::ceil (ze + 1e-9));
                for (int zi = zi0; zi <= zi1; zi++)
                for (int yi = yi0; yi <= yi1; yi++)
                for (int xi = xi0; xi <= xi1; xi++) {
                    int fidx = fine_idx(xi, yi, zi);
                    if (!need_refine[fidx]) {
                        need_refine[fidx] = 1;
                        new_verts.push_back(fidx);
                    }
                }
            }

            // Predict on newly added vertices
            if (!new_verts.empty()) {
                n_expanded = (int)new_verts.size();
                Eigen::MatrixXd EGV(n_expanded, 3);
                for (int k = 0; k < n_expanded; k++)
                    EGV.row(k) = GV.row(new_verts[k]);
                Eigen::VectorXd ES = this->predict(EGV, chunk_size);
                for (int k = 0; k < n_expanded; k++)
                    S(new_verts[k]) = ES(k);

                // Second pass post-fix on expanded vertices
                int flip2 = 0, clamp2 = 0;
                do_postfix(new_verts, flip2, clamp2);
                n_sign_flip += flip2;
                n_mag_clamp += clamp2;
            }
        }

        std::cout << "[extract_surface] lipschitz post-fix: "
                  << n_sign_flip << " sign flips, "
                  << n_mag_clamp << " magnitude clamps, "
                  << n_expanded << " expanded ("
                  << sec_since(t_fix) << " s)\n";
    }

    // ── Marching cubes ───────────────────────────────────────────────
    auto t_mc = clk::now();
    igl::marching_cubes(S, GV,
                        (unsigned)nx, (unsigned)ny, (unsigned)nz,
                        iso, V, F);
    std::cout << "[extract_surface] marching_cubes: " << sec_since(t_mc) << " s\n";
    std::cout << "[extract_surface] total: " << sec_since(t_total) << " s\n";

#else  // plain marching cubes

    const int N = nx * ny * nz;
    const Eigen::Vector3d step(
        (bbox_max.x() - bbox_min.x()) / (nx - 1),
        (bbox_max.y() - bbox_min.y()) / (ny - 1),
        (bbox_max.z() - bbox_min.z()) / (nz - 1));

    Eigen::MatrixXd GV(N, 3);
    for (int zi = 0; zi < nz; zi++)
        for (int yi = 0; yi < ny; yi++)
            for (int xi = 0; xi < nx; xi++) {
                int idx = xi + nx * (yi + ny * zi);
                GV(idx, 0) = bbox_min.x() + xi * step.x();
                GV(idx, 1) = bbox_min.y() + yi * step.y();
                GV(idx, 2) = bbox_min.z() + zi * step.z();
            }

    // ── Lipschitz pre-fill ───────────────────────────────────────────
    // For each grid point g, find nearest hint point p_i (value s_i).
    // If |s_i| > dist(g, p_i), the zero level set cannot reach g,
    // so fill S(g) = sign(s_i) * (|s_i| - dist) and skip the interpolator.
    auto t_prefill = clk::now();
    Eigen::VectorXd S(N);
    std::vector<bool> needs_interp(N, true);
    int prefilled = 0;

    if (hint_pts_.rows() > 0) {
        for (int idx = 0; idx < N; idx++) {
            Eigen::Vector3d g = GV.row(idx);
            double best_margin = -1.0;
            double best_val    =  0.0;
            for (int k = 0; k < (int)hint_pts_.rows(); k++) {
                double dist   = (g - hint_pts_.row(k).transpose()).norm();
                double s      = hint_vals_(k);
                double margin = std::abs(s) - dist;
                if (margin > best_margin) {
                    best_margin = margin;
                    best_val    = (s >= 0 ? 1.0 : -1.0) * margin;
                }
            }
            if (best_margin > 0.0) {
                S(idx) = best_val;
                needs_interp[idx] = false;
                prefilled++;
            }
        }
    }
    std::cout << "[extract_surface] lipschitz pre-fill: " << prefilled << " / " << N
              << " pts  (" << sec_since(t_prefill) << " s)\n";

    // ── Interpolator for remaining uncertain points ──────────────────
    auto t_pred = clk::now();
    std::vector<int> interp_idx;
    interp_idx.reserve(N - prefilled);
    for (int i = 0; i < N; i++)
        if (needs_interp[i]) interp_idx.push_back(i);

    if (!interp_idx.empty()) {
        const int M = (int)interp_idx.size();
        Eigen::MatrixXd QV(M, 3);
        for (int k = 0; k < M; k++) QV.row(k) = GV.row(interp_idx[k]);
        Eigen::VectorXd QS = this->predict(QV, chunk_size);
        for (int k = 0; k < M; k++) S(interp_idx[k]) = QS(k);
    }
    std::cout << "[extract_surface] predict (" << interp_idx.size() << " pts): "
              << sec_since(t_pred) << " s\n";

    auto t_mc = clk::now();
    igl::marching_cubes(S, GV, (unsigned)nx, (unsigned)ny, (unsigned)nz, iso, V, F);
    std::cout << "[extract_surface] marching_cubes: " << sec_since(t_mc) << " s\n";
    std::cout << "[extract_surface] total: " << sec_since(t_total) << " s\n";

#endif
}

}  // namespace sdf
