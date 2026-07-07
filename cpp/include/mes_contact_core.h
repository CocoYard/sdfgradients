#pragma once

#include <Eigen/Dense>

namespace mes_contact_core {

/// Compute maximal-empty-sphere contact points and outward normals for
/// each SDF sample point. Ports the core of `mes_contact.cpp` in the
/// project root to a pure-C++ API.
///
/// @param points       (N, 3) sample positions
/// @param sdf_values   (N,)   signed distances
/// @param filter_bbox  skip contact spheres outside the input bbox
/// @param debug_level  0 = silent; >0 prints CGAL summaries
/// @param out_pts      (N, 3) contact points; NaN row if none found
/// @param out_normals  (N, 3) outward normals; NaN row if none found
/// @param out_spheres  optional: filled with the maximal empty spheres
///                      found along the way, as (M, 4) rows of
///                      (x, y, z, radius). radius > 0 means the sphere was
///                      built from outside (sdf >= 0) samples, radius < 0
///                      from inside (sdf < 0) samples. Left untouched if
///                      nullptr.
void contact_points_from_sdf(
    const Eigen::MatrixXd& points,
    const Eigen::VectorXd& sdf_values,
    bool filter_bbox,
    int  debug_level,
    Eigen::MatrixXd& out_pts,
    Eigen::MatrixXd& out_normals,
    Eigen::MatrixXd* out_spheres = nullptr);

}  // namespace mes_contact_core
