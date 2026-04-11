#pragma once

#include "types.h"
#include "interpolator.h"
#include <Eigen/Dense>
#include <memory>

namespace sdf {

/// Top-level entry point, mirrors main_algorithm() in SDF_to_surface_3D.py.
///
/// Pipeline:
///   1. get_visible_arcs (sphere_intersect + compute_exposed_batch)
///   2. init_gradients_by_degenerate_pts
///   3. iterative_projection_3d
///   4. final projection + visibility check
///
/// @param sdf_points  (N, 3)
/// @param sdf_values  (N,)
/// @param options     algorithm parameters (modified in-place with runtime state)
/// @return            MainResult { projections (N,3), visibility_mask (N,) }
MainResult main_algorithm(
    const Eigen::MatrixXd& sdf_points,
    const Eigen::VectorXd& sdf_values,
    Options& options);

/// Compute visible arcs and degenerate points for all spheres.
/// Populates options.batch, options.degenerate_pts, options.ngbrs_list.
void get_visible_arcs(
    const Eigen::MatrixXd& sdf_points,
    const Eigen::VectorXd& sdf_values,
    Options& options);

/// Initialize gradients using degenerate points.
/// Mirrors init_gradients_by_degenerate_pts() in SDF_to_surface_3D.py.
///
/// @return (N, 3) initial gradients (NaN for points without degenerate info)
Eigen::MatrixXd init_gradients_by_degenerate_pts(
    const Eigen::MatrixXd& sdf_points,
    const Eigen::VectorXd& sdf_values,
    Interpolator& interpolator,
    Options& options);

}  // namespace sdf
