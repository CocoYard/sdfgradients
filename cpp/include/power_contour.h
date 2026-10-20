#pragma once

#include <Eigen/Dense>

namespace power_contour {

/// Power-contour surface from SDF samples and their tangent points (projections).
///
/// Every SDF sample i is a ball of center points.row(i) and radius |sdf_values(i)|,
/// signed by sdf_values(i) (>= 0 counts as positive). Every valid tangent point i
/// is a ball of radius 0 with the OPPOSITE sign of sample i (the projection of an
/// outside sample is labeled inside and vice versa). The power diagram of all the
/// balls is built (CGAL Regular_triangulation_3), and the power contour is the set
/// of power-diagram faces separating a positive cell from a negative cell. Each
/// such face is dual to a regular-triangulation edge with endpoints of opposite
/// sign; its polygon is the ring of power centers of the cells around that edge,
/// fan-triangulated and oriented so the normal points from negative to positive.
///
/// @param points       (N, 3) SDF sample positions
/// @param sdf_values   (N,)   signed distances
/// @param tangent_pts  (N, 3) tangent points, 1:1 with points; NaN rows are skipped
/// @param clip_to_bbox drop faces with a vertex outside the bbox of points
///                      (faces near the convex hull can have far-away power centers)
/// @param out_V        (M, 3) vertices (the power centers used by the faces)
/// @param out_F        (K, 3) triangles
/// @param out_sites    optional (K, 2): per triangle, the (positive, negative) site
///                      pair whose shared face it lies on; site i < N is sample i,
///                      site N + i is the tangent point of sample i
void extract(
    const Eigen::MatrixXd& points,
    const Eigen::VectorXd& sdf_values,
    const Eigen::MatrixXd& tangent_pts,
    bool clip_to_bbox,
    bool verbose,
    Eigen::MatrixXd& out_V,
    Eigen::MatrixXi& out_F,
    Eigen::MatrixXi* out_sites = nullptr);

}  // namespace power_contour
