#pragma once

#include <Eigen/Dense>
#include <vector>

namespace sdf {

class SphereBVH;

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

/// Overload: fast path iterates ngbrs_list[i] first (curated 2048 neighbors
/// per sphere); if no occluder is found, falls back to SphereBVH for the
/// authoritative answer. This is correct because a sphere-containment hit
/// in the neighbor list is always truly occluded; a miss only means the
/// occluder (if any) is outside the curated list and must be confirmed
/// against the full BVH.
/// frozen[i]=1 points (degenerate/short-arc, mask built from
/// options.degenerate_pts) are treated as visible (occluder check skipped).
Eigen::VectorXi are_points_visible(
    const Eigen::MatrixXd& query_points,
    const Eigen::VectorXd& sdf_values,
    const std::vector<char>& frozen,
    const std::vector<std::vector<int>>& ngbrs_list,
    const SphereBVH& bvh,
    double epsilon = 1e-8,
    bool verbose = true);

}  // namespace sdf
