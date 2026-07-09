#pragma once

#include "types.h"
#include <Eigen/Dense>

namespace sdf {

/// Compute visible arcs and degenerate points for all spheres.
/// Populates options.batch, options.degenerate_pts, options.ngbrs_list.
void get_visible_arcs(
    const Eigen::MatrixXd& sdf_points,
    const Eigen::VectorXd& sdf_values,
    Options& options);

}  // namespace sdf
