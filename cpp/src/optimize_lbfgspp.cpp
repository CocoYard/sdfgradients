// Reference implementation of the S² gradient search built on the LBFGS++
// library (yixuan/LBFGSpp), kept alongside the in-house batched BFGS in
// interpolator.cpp so the two can be compared on accuracy and wall time.
//
// The structural difference is where the iteration loop lives. LBFGS++ owns
// its loop and calls the objective back one problem at a time, so each RBF
// evaluation here is a single query point; the in-house solver keeps all N
// points in lockstep and evaluates them in one batched call. Parallelism is
// therefore across points (one independent solve per point) rather than inside
// the interpolator.
//
// The sphere constraint is removed by a chart instead of a retraction: each
// point optimises two unconstrained coordinates (u, v) in the gnomonic chart
// around its initial direction g₀,
//     ĝ(u, v) = normalize(g₀ + u·e₁ + v·e₂),      (e₁, e₂) ⟂ g₀ orthonormal
// which is a diffeomorphism onto the open hemisphere around g₀ — the Fibonacci
// sweep always starts inside it, and the objective's minimiser is nearby.

#include "interpolator.h"

#ifdef SDF_HAVE_LBFGSPP

#include <LBFGS.h>
#include <cmath>
#include <stdexcept>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace sdf {
namespace lbfgspp_opt {

namespace {

// f(u, v) = D̃(p − s·ĝ) / s, with ∇ pulled back through the chart.
// Dividing by s orients the objective (the minimum is the direction whose
// projection lands deepest on the correct side) and cancels the chain-rule
// factor, so ∇_ĝ f = −∇D̃(q) — the same objective the batched solver uses.
struct ChartObjective {
    const Interpolator* interp;
    Eigen::RowVector3d p, g0, e1, e2;
    double s, inv_s;
    long long nevals = 0;

    double operator()(const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
        Eigen::RowVector3d w = g0 + x(0) * e1 + x(1) * e2;
        double n = w.norm();
        Eigen::RowVector3d gh = w / n;

        Eigen::MatrixXd Q(1, 3);
        Q.row(0) = p - s * gh;
        Eigen::VectorXd v;
        Eigen::MatrixXd G;
        interp->predict_with_gradients(Q, v, G, 1);
        nevals++;

        Eigen::RowVector3d df = -G.row(0);            // ∇_ĝ f
        // dĝ/dx_j = (I − ĝĝᵀ) e_j / ‖w‖
        Eigen::RowVector3d t1 = (e1 - gh * gh.dot(e1)) / n;
        Eigen::RowVector3d t2 = (e2 - gh * gh.dot(e2)) / n;
        grad.resize(2);
        grad(0) = df.dot(t1);
        grad(1) = df.dot(t2);
        return v(0) * inv_s;
    }
};

// Any unit vector orthogonal to g, chosen from the smallest component of g so
// the cross product never degenerates.
Eigen::RowVector3d orthogonal_to(const Eigen::RowVector3d& g) {
    Eigen::RowVector3d a(0, 0, 0);
    int k = 0;
    if (std::abs(g(1)) < std::abs(g(k))) k = 1;
    if (std::abs(g(2)) < std::abs(g(k))) k = 2;
    a(k) = 1.0;
    Eigen::RowVector3d e = g.cross(a);
    return e / e.norm();
}

}  // namespace

long long optimize(const Interpolator& interp,
                   const Eigen::MatrixXd& points,
                   const Eigen::VectorXd& sdf_values,
                   const std::vector<int>& act,
                   int max_iter,
                   Eigen::MatrixXd& dirs) {
    // Matched to the in-house solver so the comparison is apples-to-apples.
    // Both differences below were previously charging LBFGS++ for work the
    // batched BFGS never does:
    //   · epsilon — LBFGS++ tests ‖g‖ in chart coordinates, which at the chart
    //     origin *is* the tangent gradient on S², the same quantity the batched
    //     solver retires points on. It was set to 1e-9 against that solver's
    //     1e-4, so every point ground on five orders of magnitude further.
    //   · line search — the default is strong Wolfe, whose curvature condition
    //     costs extra (f, ∇f) evaluations per step. The batched solver accepts
    //     on Armijo alone, so use the same rule and the same backtrack budget.
    LBFGSpp::LBFGSParam<double> param;
    // Tuned on the analytic-sphere harness, measuring RBF evaluations per point
    // (the only thing that costs here) against angular accuracy:
    //   m       2 vs 6      identical eval count — the chart has exactly two
    //                       variables, so a longer history stores nothing
    //                       L-BFGS can still use. 2 is the full-memory choice.
    //   epsilon 1e-4        matches the angular scale that matters: ‖g‖ in
    //                       chart coordinates is the tangent gradient on S²,
    //                       and ‖∇D̃‖ ≈ 1, so 1e-4 ≈ 0.006°. Loosening to 1e-2
    //                       saves 1.5 evals/pt and costs nothing measurable;
    //                       it is kept tight because the saving is small.
    //   linesearch          Armijo backtracking, not the default strong Wolfe:
    //                       the curvature condition costs extra evaluations for
    //                       no accuracy here (6.8 vs 8.4 evals/pt, same result).
    //   past/delta          the f-stagnation stop looks tempting (4.7 evals/pt)
    //                       but terminates after ~1 step: 2.75° vs 0.024°. Off.
    param.m = 2;
    param.epsilon = 1e-4;
    param.epsilon_rel = 0.0;
    param.max_iterations = std::max(1, max_iter);
    param.linesearch = LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING_ARMIJO;
    param.ftol = 1e-4;
    param.max_linesearch = 8;

    const int M = (int)act.size();
    long long nevals = 0;

    #pragma omp parallel for schedule(dynamic, 64) reduction(+ : nevals)
    for (int k = 0; k < M; k++) {
        int i = act[k];
        double s = sdf_values(i);
        if (std::abs(s) <= 1e-12) continue;   // q = p whatever g is

        ChartObjective obj;
        obj.interp = &interp;
        obj.p      = points.row(i);
        obj.g0     = dirs.row(i);
        obj.g0    /= obj.g0.norm();
        obj.e1     = orthogonal_to(obj.g0);
        obj.e2     = obj.g0.cross(obj.e1);
        obj.s      = s;
        obj.inv_s  = 1.0 / s;

        Eigen::VectorXd x = Eigen::VectorXd::Zero(2);
        double fx = 0.0;
        // The line search is a template parameter, not a param field:
        // the default LineSearchNocedalWright accepts only strong Wolfe and
        // rejects param.linesearch at use time.
        LBFGSpp::LBFGSSolver<double, LBFGSpp::LineSearchBacktracking> solver(param);
        try {
            solver.minimize(obj, x, fx);
        } catch (const std::runtime_error&) {
            // Thrown when the line search cannot make progress. `x` is left at
            // the best iterate reached, which is what we want. Deliberately
            // narrow: std::invalid_argument from parameter validation must
            // escape rather than be silently swallowed into a no-op solve.
        }
        nevals += obj.nevals;

        Eigen::RowVector3d w = obj.g0 + x(0) * obj.e1 + x(1) * obj.e2;
        double n = w.norm();
        if (n > 1e-15 && w.allFinite()) dirs.row(i) = w / n;
    }
    return nevals;
}

}  // namespace lbfgspp_opt
}  // namespace sdf

#else  // !SDF_HAVE_LBFGSPP

#include <stdexcept>

namespace sdf {
namespace lbfgspp_opt {

long long optimize(const Interpolator&, const Eigen::MatrixXd&,
                   const Eigen::VectorXd&, const std::vector<int>&,
                   int, Eigen::MatrixXd&) {
    throw std::runtime_error(
        "GradOpt::LBFGSpp requires the LBFGS++ headers; reconfigure with "
        "-DSDF_WITH_LBFGSPP=ON");
}

}  // namespace lbfgspp_opt
}  // namespace sdf

#endif
