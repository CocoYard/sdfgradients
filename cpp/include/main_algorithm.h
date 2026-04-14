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

/// Initialize gradients from collinear point pairs.
/// A pair (i, j) is collinear when |sdf_i - sdf_j| / |p_i - p_j| ≈ 1,
/// meaning the two points lie along the gradient direction.
/// Only searches within options.ngbrs_list neighbors.
/// Returns (N, 3) gradients; points with no qualifying pair remain NaN.
Eigen::MatrixXd init_gradients_by_collinear_pairs(
    const Eigen::MatrixXd& sdf_points,
    const Eigen::VectorXd& sdf_values,
    const Options& options,
    double ratio_tol = 1e-6);

/// Initialize gradients using degenerate points.
/// Mirrors init_gradients_by_degenerate_pts() in SDF_to_surface_3D.py.
///
/// @return (N, 3) initial gradients (NaN for points without degenerate info)
Eigen::MatrixXd init_gradients_by_degenerate_pts(
    const Eigen::MatrixXd& sdf_points,
    const Eigen::VectorXd& sdf_values,
    Interpolator& interpolator,
    Options& options);

/// Export SDF points, projections, and visibility mask as two PLY files.
///
/// File 1  out/projection_{name}_{grid_len}_{max_iters}.ply
///           visible pairs: gray SDF vertices + blue projection vertices + edges
/// File 2  out/projection_{name}_{grid_len}_{max_iters}_ln.ply
///           invisible pairs: gray SDF vertices + red projection vertices + edges
///
/// Edges are PLY edge elements; MeshLab lets you set each file's line color separately.
///
/// @param sdf_points  (N, 3)
/// @param projections (N, 3)
/// @param vis         (N,) non-zero = visible
/// @param options     used for name, grid_len, max_iters
/// @param out_dir     output directory (default "out")
void export_projection_ply(
    const Eigen::MatrixXd& sdf_points,
    const Eigen::MatrixXd& projections,
    const Eigen::VectorXi& vis,
    const Options& options,
    const std::string& out_dir = "out");

}  // namespace sdf
