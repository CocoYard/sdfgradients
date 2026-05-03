#pragma once

#include "types.h"
#include "interpolator.h"
#include <Eigen/Dense>
#include <memory>

namespace sdf {

/// Iteratively refine SDF gradients by projecting onto the zero level set.
/// Mirrors Python iterative_projection_3d() in optimization.py.
///
/// Each iteration:
///   1. sample_best_gradients to find directions minimizing |interpolant(projection)|
///   2. clamp_gradients_to_arcs for visibility
///   3. skip updates that would make visible projections invisible
///   4. refit the interpolator with updated gradients
///
/// @param points         (N, 3) sample points
/// @param values         (N,)   signed distances
/// @param init_gradients (N, 3) initial gradient estimates (NaN allowed)
/// @param interpolator   fitted interpolator (Duchon or PU)
/// @param options        algorithm options (includes batch, degenerate_pts, ngbrs_list)
/// @param num_iter       number of projection-refit iterations
/// @param num_coarse     Fibonacci directions for initial sweep
/// @param optim_steps    projected gradient descent steps per iteration
/// @param lr             learning rate for projected gradient descent
/// @param gt_gradients   optional ground truth for diagnostics
/// @return               refined (N, 3) gradient directions
Eigen::MatrixXd iterative_projection_3d(
    const Eigen::MatrixXd& points,
    const Eigen::VectorXd& values,
    const Eigen::MatrixXd& init_gradients,
    Interpolator& interpolator,
    Options& options,
    int num_iter      = 10,
    int num_coarse    = 64,
    int optim_steps   = 10,
    double lr         = 0.2,
    const Eigen::MatrixXd* gt_gradients = nullptr);

}  // namespace sdf
