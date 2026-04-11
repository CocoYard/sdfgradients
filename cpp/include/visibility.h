#pragma once

#include <Eigen/Dense>

namespace sdf {

/// Check if query points are visible (not inside any SDF sphere).
///
/// A point is visible if for every SDF point j:
///   ||query[i] - sdf_points[j]|| >= |sdf_values[j]| - epsilon
///
/// @param query_points  (N, 3)
/// @param sdf_points    (M, 3)
/// @param sdf_values    (M,)
/// @param epsilon       small margin (default 1e-8)
/// @return              (N,) boolean vector: 1 = visible, 0 = occluded
Eigen::VectorXi are_points_visible(
    const Eigen::MatrixXd& query_points,
    const Eigen::MatrixXd& sdf_points,
    const Eigen::VectorXd& sdf_values,
    double epsilon = 1e-8);

}  // namespace sdf
