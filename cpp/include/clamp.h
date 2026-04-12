#pragma once

#include "types.h"
#include <Eigen/Dense>
#include <functional>

namespace sdf {

/// Clamp gradients to visible arc boundaries.
///
/// For each point whose projection lands inside a neighbor's sphere,
/// clamp the gradient to the closest point on the exposed arcs.
/// Points with degenerate arcs are skipped (their gradients are set separately).
///
/// Modifies `gradients` in-place.
///
/// @param points         (N, 3) sample points
/// @param values         (N,)   signed distances
/// @param gradients      (N, 3) gradient directions (modified in-place)
/// @param degenerate_pts sphere index -> list of degenerate surface points
/// @param batch          batch arc/cap data from compute_exposed_batch
/// @param ngbrs_list     neighbor lists per sphere
/// @param tolerance      clamping tolerance parameters
/// @return number of clamped gradients
int clamp_gradients_to_arcs(
    const Eigen::MatrixXd& points,
    const Eigen::VectorXd& values,
    Eigen::MatrixXd& gradients,
    const std::unordered_map<int, std::vector<Eigen::Vector3d>>& degenerate_pts,
    const Options::BatchData& batch,
    const std::vector<std::vector<int>>& ngbrs_list,
    const Tolerance& tolerance);

}  // namespace sdf
