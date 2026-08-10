#include "interpolator.h"
#include "thread_policy.h"
#include <cassert>  // libigl's march_cube.cpp uses assert() without including <cassert>
#include <igl/marching_cubes.h>
#include <algorithm>
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

// Accumulates wall-time spent in RBF evaluation (predict / predict_gradients)
// inside optimize_best_gradients, so the runtime decomposition can report how
// much of "gradient optimization" is actually RBF interpolation. Reset by
// iterative_projection_3d at the start of each optimization loop.
double g_rbf_eval_s = 0.0;

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

// Reference solver in optimize_lbfgspp.cpp; returns the RBF evaluation count.
namespace lbfgspp_opt {
long long optimize(const Interpolator& interp,
                   const Eigen::MatrixXd& points,
                   const Eigen::VectorXd& sdf_values,
                   const std::vector<int>& act,
                   int max_iter,
                   Eigen::MatrixXd& dirs);
}

Eigen::MatrixXd Interpolator::optimize_best_gradients(
    const Eigen::MatrixXd& points,
    const Eigen::VectorXd& sdf_values,
    int num_coarse,
    int optim_steps,
    double lr,
    const Eigen::MatrixXd* initial_guess,
    int chunk_size,
    const std::vector<char>* frozen,
    GradOpt method) const
{
    // Frozen rows: gather the active points, recurse on the compact arrays
    // (frozen = nullptr, so no second gather), scatter the refined rows back.
    // Frozen points keep their initial_guess gradient and never reach
    // predict()/predict_gradients().
    if (frozen) {
        int N_all = (int)points.rows();
        std::vector<int> act;
        act.reserve(N_all);
        for (int i = 0; i < N_all; i++)
            if (i >= (int)frozen->size() || !(*frozen)[i]) act.push_back(i);
        int M = (int)act.size();
        if (M < N_all) {
            Eigen::MatrixXd P(M, 3);
            Eigen::VectorXd S(M);
            Eigen::MatrixXd G(initial_guess ? M : 0, 3);
            for (int k = 0; k < M; k++) {
                P.row(k) = points.row(act[k]);
                S(k) = sdf_values(act[k]);
                if (initial_guess) G.row(k) = initial_guess->row(act[k]);
            }
            Eigen::MatrixXd active_dirs = optimize_best_gradients(
                P, S, num_coarse, optim_steps, lr,
                initial_guess ? &G : nullptr, chunk_size, nullptr, method);
            Eigen::MatrixXd out = initial_guess
                ? *initial_guess
                : Eigen::MatrixXd::Constant(
                      N_all, 3, std::numeric_limits<double>::quiet_NaN());
            for (int k = 0; k < M; k++)
                out.row(act[k]) = active_dirs.row(k);
            return out;
        }
    }

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
        auto _tec = std::chrono::steady_clock::now();
        Eigen::VectorXd preds = predict(samples, chunk_size);
        double _sweep_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - _tec).count();
        g_rbf_eval_s += _sweep_s;
        // The sweep costs num_coarse predict rows per point, so it dwarfs the
        // refinement whenever many rows arrive without a usable initial guess.
        if (verbose())
            std::cout << "  [optimize_best_gradients] fibonacci sweep: "
                      << n_inv << " / " << N << " pts, " << _sweep_s << " s\n";

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

    // Original solver, kept selectable for comparison: gradient ascent on
    // ⟨∇D̃(q), g⟩ over S² with retraction by normalization. The gradient is
    // unit-normalized so the effective step size lr is the same at every point
    // regardless of ‖∇D̃‖ (which is not 1 in general because the RBF
    // interpolant is not strictly Eikonal).
    //   q  = p − s·g
    //   n̂ = ∇D̃(q) / ‖∇D̃(q)‖
    //   g  ← normalize(g + lr · n̂)
    if (method == GradOpt::GradientAscent) {
        Eigen::MatrixXd proj(N, 3);
        for (int step = 0; step < optim_steps; step++) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < N; i++)
                proj.row(i) = points.row(i) - sdf_values(i) * dirs.row(i);

            auto _te = std::chrono::steady_clock::now();
            Eigen::MatrixXd sdf_grad = predict_gradients(proj, chunk_size);
            g_rbf_eval_s += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - _te).count();

            #pragma omp parallel for schedule(static)
            for (int i = 0; i < N; i++) {
                Eigen::RowVector3d n = sdf_grad.row(i);
                double gn = n.norm();
                if (gn < 1e-15) continue;
                n /= gn;

                Eigen::RowVector3d updated = dirs.row(i) + lr * n;
                double norm = updated.norm();
                if (norm > 1e-15)
                    dirs.row(i) = updated / norm;
            }
        }
        restore_threads(_saved);
        return dirs;
    }

    // Reference path: hand every point to its own LBFGS++ solve.
    if (method == GradOpt::LBFGSpp) {
        std::vector<int> all(N);
        for (int i = 0; i < N; i++) all[i] = i;
        auto _tl = std::chrono::steady_clock::now();
        lbfgspp_opt::optimize(*this, points, sdf_values, all, optim_steps, dirs);
        g_rbf_eval_s += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - _tl).count();
        restore_threads(_saved);
        return dirs;
    }

    // ── Riemannian BFGS on S² ───────────────────────────────────────
    // Per point we minimise, over the unit sphere,
    //     f(g) = D̃(q) / s,        q = p − s·g
    // (dividing by the signed distance s both orients the objective — the
    // minimum is the direction whose projection lands deepest on the correct
    // side — and cancels the chain-rule factor, so ∇f = −∇D̃(q) exactly. Both
    // f and ∇f are then O(1) at every point and a plain H₀ = I is well scaled,
    // where the raw Euclidean gradient would carry a per-point |s| factor.)
    //
    // The sphere constraint is handled the usual manifold way:
    //     retraction        R(g, v) = normalize(g + v)
    //     vector transport  T(v)    = (I − g'g'ᵀ) v          (projection)
    // The inverse Hessian H is kept as a 3×3 matrix per point, transported the
    // same way and given the identity on the normal direction
    // (H ← P H P + g'g'ᵀ) so it stays positive definite; search directions are
    // re-projected onto the tangent plane regardless.
    //
    // Cost is the reason for the loop shape. The RBF evaluation dominates, and
    // what it charges for is the *number* of batched calls (patch dispatch,
    // per-call setup), not just the row count — so this runs exactly one fused
    // predict_with_gradients() per iteration, the same call count as the plain
    // gradient descent it replaces, and folds the Armijo line search into the
    // outer loop instead of giving it its own evaluations:
    //   · each iteration evaluates (f, ∇f) at one candidate per point;
    //   · sufficient decrease accepted ⇒ that same ∇f closes the (s, y)
    //     curvature pair, updates H and proposes the next candidate;
    //   · rejected ⇒ the step is halved and re-proposed from the last accepted
    //     iterate (the gradient at the rejected point is simply discarded).
    // Converged and retired points are compacted out, so the calls shrink as
    // the loop proceeds. `lr` is no longer a fixed step size — it only caps the
    // first (steepest descent) step; after that the line search sets the length.
    // Convergence test on ‖γ‖. γ is the tangential part of −∇D̃(q) and the
    // interpolant is near-Eikonal (‖∇D̃‖ ≈ 1), so ‖γ‖ ≈ sin∠(∇D̃(q), g): the
    // tolerance reads directly as an angle, and 1e-4 ≈ 0.006°, an order of
    // magnitude below the accuracy the interpolant itself supports. Retiring
    // points at a *reachable* tolerance is what lets the batch shrink — with a
    // tolerance nothing ever meets, every iteration stays full width.
    constexpr double kGradTol  = 1e-4;
    constexpr double kArmijoC1 = 1e-4;  // sufficient-decrease constant
    constexpr double kMaxStep  = 1.0;   // cap on ‖t·d‖ (≈45° of rotation)
    constexpr int    kMaxBT    = 8;     // backtracks before restarting/retiring

    // Active set: points still being optimised. s == 0 means q = p whatever g
    // is — the objective is constant there, so those keep their initial
    // direction and never enter the batch.
    Eigen::VectorXd inv_s(N);
    std::vector<int> act;
    act.reserve(N);
    for (int i = 0; i < N; i++) {
        double s = sdf_values(i);
        inv_s(i) = (std::abs(s) > 1e-12) ? 1.0 / s : 0.0;
        if (inv_s(i) != 0.0) act.push_back(i);
    }

    // `dirs` holds the last accepted iterate; `cand` the one being evaluated.
    Eigen::MatrixXd cand = dirs;
    Eigen::MatrixXd cproj(N, 3);   // p − s·cand, the query point for cand
    Eigen::VectorXd fval(N);       // f at the accepted iterate
    Eigen::MatrixXd gacc(N, 3);    // tangent gradient at the accepted iterate
    Eigen::MatrixXd svec(N, 3);    // pending step t·d, one half of the BFGS pair
    Eigen::MatrixXd dirn(N, 3);    // search direction d
    Eigen::VectorXd dder(N);       // ⟨d, γ⟩, the Armijo directional derivative
    Eigen::VectorXd tstep(N);      // current step length
    std::vector<Eigen::Matrix3d> Hinv(N, Eigen::Matrix3d::Identity());
    std::vector<char> scaled(N, 0);   // H ≠ I: a curvature pair has been applied
    std::vector<char> was_sd(N, 0);   // current direction is steepest descent
    std::vector<unsigned char> nbt(N, 0);   // backtracks on the current direction
    std::vector<char> keep(N, 0);

    // Propose the candidate for the next evaluation: cand = R(g, t·d). Also
    // records the step, which becomes the `s` of the BFGS pair if it is
    // accepted. d ⟂ g and ‖t·d‖ ≤ 1, so ‖g + t·d‖ ≥ 1 — the retraction is safe.
    auto propose = [&](int i) {
        svec.row(i) = tstep(i) * dirn.row(i);
        Eigen::RowVector3d gt = dirs.row(i) + svec.row(i);
        gt /= gt.norm();
        cand.row(i)  = gt;
        cproj.row(i) = points.row(i) - sdf_values(i) * gt;
    };

    // Search direction from the tangent gradient γ at the accepted iterate g.
    // Falls back to steepest descent if H has stopped producing descent.
    // Returns false when no usable direction exists (γ in the null space).
    auto set_direction = [&](int i, const Eigen::Vector3d& g,
                             const Eigen::Vector3d& gamma, double cap) {
        bool sd = (scaled[i] == 0);
        Eigen::Vector3d d = -(Hinv[i] * gamma);
        d -= d.dot(g) * g;
        if (!(d.dot(gamma) < 0.0)) {
            Hinv[i].setIdentity();
            scaled[i] = 0;
            d = -gamma;
            sd = true;
        }
        double dn = d.norm();
        if (!(dn > 0.0)) return false;
        dirn.row(i) = d.transpose();
        dder(i)     = d.dot(gamma);
        tstep(i)    = std::min(1.0, cap / dn);
        was_sd[i]   = sd ? 1 : 0;
        nbt[i]      = 0;
        return true;
    };

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++)
        cproj.row(i) = points.row(i) - sdf_values(i) * cand.row(i);

    for (int step = 0; step < optim_steps && !act.empty(); step++) {
        int M = (int)act.size();

        Eigen::MatrixXd Q(M, 3);
        #pragma omp parallel for schedule(static)
        for (int k = 0; k < M; k++) Q.row(k) = cproj.row(act[k]);

        auto _te = std::chrono::steady_clock::now();
        Eigen::VectorXd val;
        Eigen::MatrixXd sdf_grad;
        predict_with_gradients(Q, val, sdf_grad, chunk_size);
        g_rbf_eval_s += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - _te).count();

        // Accept/reject, BFGS update and next candidate — all per-point 3×3
        // work, so it runs parallel over the active set; the compaction of the
        // retired points is done serially afterwards.
        const double cap = (step == 0) ? lr : kMaxStep;
        #pragma omp parallel for schedule(static)
        for (int k = 0; k < M; k++) {
            int i = act[k];
            keep[i] = 1;
            double f_new = val(k) * inv_s(i);

            // Iteration 0 evaluates the initial direction itself: there is
            // nothing to compare against, so it is accepted by definition.
            if (step > 0 &&
                !(f_new <= fval(i) + kArmijoC1 * tstep(i) * dder(i))) {
                if (++nbt[i] < kMaxBT) {
                    tstep(i) *= 0.5;
                } else if (!was_sd[i]) {
                    // The quasi-Newton direction is not working here. Restart
                    // from steepest descent — γ at the accepted iterate is
                    // already known, so no extra evaluation is needed.
                    Hinv[i].setIdentity();
                    scaled[i] = 0;
                    if (!set_direction(i, dirs.row(i).transpose(),
                                       gacc.row(i).transpose(), cap)) {
                        keep[i] = 0;
                        continue;
                    }
                } else {
                    // Steepest descent backtracked to nothing: no step from
                    // here decreases f, so the point is at the noise floor.
                    keep[i] = 0;
                    continue;
                }
                propose(i);
                continue;
            }

            Eigen::Vector3d g = cand.row(i).transpose();
            Eigen::Vector3d gamma = -sdf_grad.row(k).transpose();
            gamma -= gamma.dot(g) * g;             // project onto T_g S²
            if (!gamma.allFinite()) { keep[i] = 0; continue; }

            if (step > 0) {
                // Close the curvature pair spanning the accepted step.
                Eigen::Matrix3d P =
                    Eigen::Matrix3d::Identity() - g * g.transpose();
                Eigen::Vector3d sv = P * svec.row(i).transpose();
                Eigen::Vector3d y  = gamma - P * gacc.row(i).transpose();
                double sy = sv.dot(y);
                Hinv[i] = P * Hinv[i] * P + g * g.transpose();
                // Skip on failed curvature (sᵀy ≤ 0): the objective is not
                // convex on S², and Armijo alone does not guarantee it.
                if (sy > 1e-14 * sv.norm() * y.norm()) {
                    if (!scaled[i]) {
                        double yy = y.squaredNorm();
                        if (yy > 0.0) Hinv[i] *= sy / yy;   // Nocedal (6.20)
                        scaled[i] = 1;
                    }
                    double rho = 1.0 / sy;
                    Eigen::Matrix3d V = Eigen::Matrix3d::Identity()
                                      - rho * sv * y.transpose();
                    Hinv[i] = V * Hinv[i] * V.transpose()
                            + rho * sv * sv.transpose();
                }
            }

            dirs.row(i) = cand.row(i);
            fval(i)     = f_new;
            gacc.row(i) = gamma.transpose();

            if (gamma.norm() <= kGradTol ||
                !set_direction(i, g, gamma, cap)) {
                keep[i] = 0;             // converged
                continue;
            }
            propose(i);
        }

        std::vector<int> nxt;
        nxt.reserve(M);
        for (int k = 0; k < M; k++)
            if (keep[act[k]]) nxt.push_back(act[k]);
        act.swap(nxt);
    }

    restore_threads(_saved);
    return dirs;
}

// ── Dual contouring ──────────────────────────────────────────────────
// Given an evaluated SDF grid and an interpolator with analytic gradients,
// build a triangle mesh of the iso level set. For each sign-changing grid
// edge we evaluate the gradient at the interpolated intersection point; for
// each cell with at least one such edge we solve a QEF (min Σ (nᵢ·(v-pᵢ))²)
// via truncated-SVD, biased toward the cell centroid, and clamp the result
// to the cell bbox. Every sign-changing edge then emits a quad of the 4
// dual vertices of the cells sharing it, split into two triangles with
// winding chosen so the outward normal follows sdf gradient sign.
namespace dc_impl {

struct EdgeIsect {
    float t;                      // linear intersection parameter ∈ [0,1]
    int xi, yi, zi;               // base (lower-index) grid vertex
    signed char axis;             // 0=x, 1=y, 2=z
    signed char neg_end;          // 0 if S(base)<iso, 1 if S(base+axis)<iso
};

static void dual_contour(
    const Interpolator& interp,
    const Eigen::VectorXd& S,
    const Eigen::MatrixXd& GV,
    int nx, int ny, int nz,
    double iso,
    int chunk_size,
    Eigen::MatrixXd& V_out,
    Eigen::MatrixXi& F_out)
{
    using clk = std::chrono::steady_clock;
    auto sec_since = [](const clk::time_point& t) {
        return std::chrono::duration<double>(clk::now() - t).count();
    };
    const int N = nx * ny * nz;
    auto vidx = [&](int xi, int yi, int zi) {
        return xi + nx * (yi + ny * zi);
    };

    // ── 1. Find all sign-changing edges ──────────────────────────────
    // Linear interpolation of the grid samples gives the *initial guess*
    // for t; the true root on the interpolator is found in step 2.
    auto t_edge = clk::now();
    std::vector<EdgeIsect> edges;
    edges.reserve(N / 4);
    std::vector<int> edge_map(3 * N, -1);
    auto emap = [&](int xi, int yi, int zi, int axis) {
        return 3 * vidx(xi, yi, zi) + axis;
    };
    auto try_edge = [&](int xi, int yi, int zi, int axis,
                        int xi2, int yi2, int zi2) {
        double sa = S(vidx(xi, yi, zi));
        double sb = S(vidx(xi2, yi2, zi2));
        bool a_neg = sa < iso, b_neg = sb < iso;
        if (a_neg == b_neg) return;
        double denom = sb - sa;
        float t = (std::abs(denom) > 1e-30) ? float((iso - sa) / denom) : 0.5f;
        if (t < 0.f) t = 0.f; else if (t > 1.f) t = 1.f;
        edge_map[emap(xi, yi, zi, axis)] = (int)edges.size();
        EdgeIsect ei;
        ei.t = t;
        ei.xi = xi; ei.yi = yi; ei.zi = zi;
        ei.axis = (signed char)axis;
        ei.neg_end = a_neg ? 0 : 1;
        edges.push_back(ei);
    };
    for (int zi = 0; zi < nz;   zi++)
    for (int yi = 0; yi < ny;   yi++)
    for (int xi = 0; xi < nx-1; xi++) try_edge(xi, yi, zi, 0, xi+1, yi, zi);
    for (int zi = 0; zi < nz;   zi++)
    for (int yi = 0; yi < ny-1; yi++)
    for (int xi = 0; xi < nx;   xi++) try_edge(xi, yi, zi, 1, xi, yi+1, zi);
    for (int zi = 0; zi < nz-1; zi++)
    for (int yi = 0; yi < ny;   yi++)
    for (int xi = 0; xi < nx;   xi++) try_edge(xi, yi, zi, 2, xi, yi, zi+1);

    const int nedges = (int)edges.size();
    if (nedges == 0) {
        V_out.resize(0, 3); F_out.resize(0, 3);
        return;
    }

    // Edge endpoints and directions, reused throughout.
    Eigen::MatrixXd PA(nedges, 3), ED(nedges, 3);   // pa, (pb - pa)
    for (int e = 0; e < nedges; e++) {
        const auto& ei = edges[e];
        PA.row(e) = GV.row(vidx(ei.xi, ei.yi, ei.zi));
        int xi2 = ei.xi + (ei.axis == 0);
        int yi2 = ei.yi + (ei.axis == 1);
        int zi2 = ei.zi + (ei.axis == 2);
        ED.row(e) = GV.row(vidx(xi2, yi2, zi2)).transpose() - PA.row(e).transpose();
    }

    // ── 2. Root-find each intersection on the interpolator ───────────
    // Safeguarded Newton along the edge: keep a bracket [t_neg, t_pos]
    // (f < iso at t_neg, f > iso at t_pos, known from the grid signs),
    // take Newton steps t ← t − (f − iso)/(∇f·d), and fall back to
    // bisection whenever the step leaves the bracket or ∇f·d ≈ 0.
    // Each iteration is ONE fused batched predict over the still-active
    // edges; converged edges drop out, so total cost ≈ 2–4 batched calls.
    auto t_root = clk::now();

    Eigen::MatrixXd EP(nedges, 3);                  // intersection positions
    Eigen::MatrixXd EN(nedges, 3);                  // gradients at EP
    EN.setZero();

    std::vector<double> tcur(nedges), tneg(nedges), tpos(nedges);
    for (int e = 0; e < nedges; e++) {
        tcur[e] = double(edges[e].t);
        tneg[e] = (edges[e].neg_end == 0) ? 0.0 : 1.0;
        tpos[e] = (edges[e].neg_end == 0) ? 1.0 : 0.0;
        EP.row(e) = PA.row(e) + tcur[e] * ED.row(e);
    }

    const int    max_newton = 12;     // Newton needs 2–4; bisection tail more
    const double t_tol      = 1e-7;   // |dt| below this ⇒ position converged
                                      // (relative to edge length, i.e. ~1e-7·h)
    std::vector<int> active(nedges);
    for (int e = 0; e < nedges; e++) active[e] = e;

    int n_evals = 0;
    {
        int saved = set_threads(thread_policy().predict);
        Eigen::MatrixXd EPa, Ga;
        Eigen::VectorXd Fa;
        for (int it = 0; it < max_newton && !active.empty(); it++) {
            const int na = (int)active.size();
            EPa.resize(na, 3);
            for (int k = 0; k < na; k++) EPa.row(k) = EP.row(active[k]);

            interp.predict_with_gradients(EPa, Fa, Ga, chunk_size);
            n_evals++;

            std::vector<int> next;
            next.reserve(na);
            for (int k = 0; k < na; k++) {
                const int e = active[k];
                const double f = Fa(k) - iso;

                // Gradient at the point we just evaluated. If the edge
                // converges this round, tcur moves by |dt| < t_tol, so this
                // gradient is at the final position up to O(t_tol·h).
                EN.row(e) = Ga.row(k);

                // Shrink the bracket with the sign of f.
                if (f < 0.0) tneg[e] = tcur[e];
                else         tpos[e] = tcur[e];

                // Newton step along the edge; bisect if invalid.
                const double df = Ga.row(k).dot(ED.row(e));
                double tn;
                bool newton_ok = std::abs(df) > 1e-300;
                if (newton_ok) {
                    tn = tcur[e] - f / df;
                    const double lo = std::min(tneg[e], tpos[e]);
                    const double hi = std::max(tneg[e], tpos[e]);
                    newton_ok = (tn > lo && tn < hi);
                }
                if (!newton_ok) tn = 0.5 * (tneg[e] + tpos[e]);

                const double dt = tn - tcur[e];
                tcur[e] = tn;
                EP.row(e) = PA.row(e) + tcur[e] * ED.row(e);

                if (std::abs(dt) > t_tol) next.push_back(e);
            }
            active.swap(next);
        }
        restore_threads(saved);
    }

    // Persist refined parameters (float storage, as before).
    for (int e = 0; e < nedges; e++) edges[e].t = (float)tcur[e];

    for (int e = 0; e < nedges; e++) {
        double n = EN.row(e).norm();
        if (n > 1e-12) EN.row(e) /= n;
        else EN.row(e).setZero();
    }

    if (interp.verbose()) {
        std::printf("[dc] %d edges, root-finding: %d fused evals, "
                    "%d unconverged, %.3fs\n",
                    nedges, n_evals, (int)active.size(), sec_since(t_root));
    }

    // ── 3. Per-cell QEF: min Σ (nᵢ·(v-pᵢ))², biased to centroid ─────
    auto t_qef = clk::now();
    const int Cx = nx - 1, Cy = ny - 1, Cz = nz - 1;
    const int NC = Cx * Cy * Cz;
    auto cidx = [&](int ci, int cj, int ck) {
        return ci + Cx * (cj + Cy * ck);
    };
    std::vector<int> cell_vertex(NC, -1);
    std::vector<Eigen::Vector3d> verts;
    verts.reserve(nedges / 3);

    for (int ck = 0; ck < Cz; ck++)
    for (int cj = 0; cj < Cy; cj++)
    for (int ci = 0; ci < Cx; ci++) {
        int ids[12] = {
            edge_map[emap(ci, cj  , ck  , 0)],
            edge_map[emap(ci, cj+1, ck  , 0)],
            edge_map[emap(ci, cj  , ck+1, 0)],
            edge_map[emap(ci, cj+1, ck+1, 0)],
            edge_map[emap(ci  , cj, ck  , 1)],
            edge_map[emap(ci+1, cj, ck  , 1)],
            edge_map[emap(ci  , cj, ck+1, 1)],
            edge_map[emap(ci+1, cj, ck+1, 1)],
            edge_map[emap(ci  , cj  , ck, 2)],
            edge_map[emap(ci+1, cj  , ck, 2)],
            edge_map[emap(ci  , cj+1, ck, 2)],
            edge_map[emap(ci+1, cj+1, ck, 2)],
        };
        Eigen::Matrix3d AtA = Eigen::Matrix3d::Zero();
        Eigen::Vector3d Atb = Eigen::Vector3d::Zero();
        Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
        int nact = 0;
        for (int e = 0; e < 12; e++) {
            if (ids[e] < 0) continue;
            Eigen::Vector3d p = EP.row(ids[e]);
            Eigen::Vector3d n = EN.row(ids[e]);
            AtA += n * n.transpose();
            Atb += n * (n.dot(p));
            centroid += p;
            nact++;
        }
        if (nact == 0) continue;
        centroid /= double(nact);

        // Solve (AtA)(v-c) = Atb - AtA·c via truncated SVD pinv. Truncating
        // small singular values is what biases v toward c along underdetermined
        // directions (plane / edge normals) while preserving sharp features.
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(
            AtA, Eigen::ComputeFullU | Eigen::ComputeFullV);
        const Eigen::Vector3d& sv = svd.singularValues();
        const double trunc = 1e-3 * sv(0);
        Eigen::Vector3d inv_sv;
        for (int k = 0; k < 3; k++)
            inv_sv(k) = (sv(k) > trunc) ? (1.0 / sv(k)) : 0.0;
        Eigen::Matrix3d pinv =
            svd.matrixV() * inv_sv.asDiagonal() * svd.matrixU().transpose();
        Eigen::Vector3d v = centroid + pinv * (Atb - AtA * centroid);

        Eigen::Vector3d cmin = GV.row(vidx(ci,   cj,   ck  ));
        Eigen::Vector3d cmax = GV.row(vidx(ci+1, cj+1, ck+1));
        for (int k = 0; k < 3; k++)
            v(k) = std::max(cmin(k), std::min(cmax(k), v(k)));

        cell_vertex[cidx(ci, cj, ck)] = (int)verts.size();
        verts.push_back(v);
    }

    // ── 4. Emit triangles per sign-changing edge ─────────────────────
    // Cell ordering is CCW when viewed looking in +axis direction so that
    // triangle winding yields an outward normal aligned with +axis when the
    // edge goes neg→pos (neg_end == 0). Flip otherwise.
    auto t_tri = clk::now();
    std::vector<std::array<int, 3>> tris;
    tris.reserve(nedges * 2);
    for (int e = 0; e < nedges; e++) {
        const auto& ei = edges[e];
        int xi = ei.xi, yi = ei.yi, zi = ei.zi;
        int c[4];
        if (ei.axis == 0) {
            if (yi == 0 || yi >= Cy || zi == 0 || zi >= Cz) continue;
            c[0] = cidx(xi, yi-1, zi-1);
            c[1] = cidx(xi, yi  , zi-1);
            c[2] = cidx(xi, yi  , zi  );
            c[3] = cidx(xi, yi-1, zi  );
        } else if (ei.axis == 1) {
            if (xi == 0 || xi >= Cx || zi == 0 || zi >= Cz) continue;
            c[0] = cidx(xi-1, yi, zi-1);
            c[1] = cidx(xi-1, yi, zi  );
            c[2] = cidx(xi  , yi, zi  );
            c[3] = cidx(xi  , yi, zi-1);
        } else {
            if (xi == 0 || xi >= Cx || yi == 0 || yi >= Cy) continue;
            c[0] = cidx(xi-1, yi-1, zi);
            c[1] = cidx(xi  , yi-1, zi);
            c[2] = cidx(xi  , yi  , zi);
            c[3] = cidx(xi-1, yi  , zi);
        }
        int v[4];
        bool ok = true;
        for (int k = 0; k < 4; k++) {
            v[k] = cell_vertex[c[k]];
            if (v[k] < 0) { ok = false; break; }
        }
        if (!ok) continue;

        if (ei.neg_end == 0) {
            tris.push_back({v[0], v[1], v[2]});
            tris.push_back({v[0], v[2], v[3]});
        } else {
            tris.push_back({v[0], v[2], v[1]});
            tris.push_back({v[0], v[3], v[2]});
        }
    }

    V_out.resize((int)verts.size(), 3);
    for (int i = 0; i < (int)verts.size(); i++) V_out.row(i) = verts[i];
    F_out.resize((int)tris.size(), 3);
    for (int i = 0; i < (int)tris.size(); i++) {
        F_out(i, 0) = tris[i][0];
        F_out(i, 1) = tris[i][1];
        F_out(i, 2) = tris[i][2];
    }
}

}  // namespace dc_impl

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
    bool lipschitz_postfix,
    bool use_dual_contouring) const
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

    if (verbose_)
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
    if (verbose_)
        std::cout << "[extract_surface] coarse predict (" << CN
                  << " pts): " << sec_since(t_coarse) << " s\n";

    // ── Dummy-fill fine S from nearest coarse corner ─────────────────
    // Inactive fine vertices never contribute to the extracted surface
    // (Lipschitz-1 keeps the iso-surface out of inactive coarse cells and
    // their boundaries), so any sign-correct placeholder works. Copying the
    // c000 corner of the containing coarse cell gives correct sign and
    // |value| >= tau for free, at 1 lookup per fine vertex instead of 8.
    // Active fine vertices get overwritten by predict() below.
    auto t_tri = clk::now();
    Eigen::VectorXd S(N);
    for (int zi = 0; zi < nz; zi++) {
        int ck = std::min((int)((zi * step.z()) / cstep.z()), cnz - 2);
        for (int yi = 0; yi < ny; yi++) {
            int cj = std::min((int)((yi * step.y()) / cstep.y()), cny - 2);
            for (int xi = 0; xi < nx; xi++) {
                int ci = std::min((int)((xi * step.x()) / cstep.x()), cnx - 2);
                S(fine_idx(xi, yi, zi)) = CS(cidx_at(ci, cj, ck));
            }
        }
    }

    if (verbose_)
        std::cout << "[extract_surface] dummy fill S: " << sec_since(t_tri) << " s\n";

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

    if (verbose_)
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

    if (verbose_)
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

        if (verbose_)
            std::cout << "[extract_surface] lipschitz post-fix: "
                      << n_sign_flip << " sign flips, "
                      << n_mag_clamp << " magnitude clamps, "
                      << n_expanded << " expanded ("
                      << sec_since(t_fix) << " s)\n";
    }

    // ── Dummy-fill leakage check ─────────────────────────────────────
    // A fine cell with a sign change produces mesh output. If any such cell
    // has a corner that was never refined, dummy-fill values are leaking
    // into the mesh.
    if (verbose_) {
        long long n_active_cells = 0, n_leaky_cells = 0;
        for (int zi = 0; zi < nz - 1; zi++)
        for (int yi = 0; yi < ny - 1; yi++)
        for (int xi = 0; xi < nx - 1; xi++) {
            double vmin =  std::numeric_limits<double>::infinity();
            double vmax = -std::numeric_limits<double>::infinity();
            bool any_dummy = false;
            for (int dc = 0; dc < 8; dc++) {
                int xi2 = xi + (dc & 1);
                int yi2 = yi + ((dc >> 1) & 1);
                int zi2 = zi + ((dc >> 2) & 1);
                int fidx = fine_idx(xi2, yi2, zi2);
                double v = S(fidx);
                vmin = std::min(vmin, v);
                vmax = std::max(vmax, v);
                if (!need_refine[fidx]) any_dummy = true;
            }
            if ((vmin - iso) * (vmax - iso) <= 0.0) {
                n_active_cells++;
                if (any_dummy) n_leaky_cells++;
            }
        }
        std::cout << "[extract_surface] dummy-fill leak check: "
                  << n_leaky_cells << " / " << n_active_cells
                  << " sign-change cells touch a non-refined vertex\n";
    }

    // ── Surface extraction ───────────────────────────────────────────
    auto t_mc = clk::now();
    if (use_dual_contouring) {
        dc_impl::dual_contour(*this, S, GV, nx, ny, nz, iso, chunk_size, V, F);
        if (verbose_)
            std::cout << "[extract_surface] dual_contouring: "
                      << sec_since(t_mc) << " s\n";
    } else {
        igl::marching_cubes(S, GV,
                            (unsigned)nx, (unsigned)ny, (unsigned)nz,
                            iso, V, F);
        if (verbose_)
            std::cout << "[extract_surface] marching_cubes: "
                      << sec_since(t_mc) << " s\n";
    }
    if (verbose_)
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
    if (verbose_)
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
    if (verbose_)
        std::cout << "[extract_surface] predict (" << interp_idx.size() << " pts): "
                  << sec_since(t_pred) << " s\n";

    auto t_mc = clk::now();
    igl::marching_cubes(S, GV, (unsigned)nx, (unsigned)ny, (unsigned)nz, iso, V, F);
    if (verbose_) {
        std::cout << "[extract_surface] marching_cubes: " << sec_since(t_mc) << " s\n";
        std::cout << "[extract_surface] total: " << sec_since(t_total) << " s\n";
    }

#endif
}

}  // namespace sdf
