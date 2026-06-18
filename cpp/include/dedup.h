#pragma once

/// Shared spatial deduplication for RBF training inputs.
///
/// Near-coincident points make the RBF kernel matrix have (near-)identical
/// rows; with reg=0 that matrix is singular and the linear solve is unstable,
/// poisoning the predicted field. Both DuchonInterpolator (directly) and
/// PUInterpolator (before splitting into patches) clean their inputs with this.

#include <Eigen/Dense>
#include <vector>
#include <numeric>
#include <algorithm>
#include "kdtree.h"

namespace sdf {

/// Greedy spatial dedup of `points` within Euclidean `tol`. Non-zero-valued
/// points are processed first so they survive over zero-valued duplicates
/// (projection points carry value 0 and are less trustworthy). Returns the
/// ascending indices of the surviving rows.
inline std::vector<int> dedup_keep_indices(const Eigen::MatrixXd& points,
                                           const Eigen::VectorXd& values,
                                           double tol) {
    int n = (int)points.rows();
    std::vector<int> kept;
    if (n == 0) return kept;

    KDTree3D tree(points);
    std::vector<bool> keep(n, true);

    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        bool za = values(a) == 0.0, zb = values(b) == 0.0;
        if (za != zb) return !za;   // non-zero first
        return a < b;
    });

    for (int i : order) {
        if (!keep[i]) continue;
        Eigen::Vector3d pt = points.row(i);
        for (int j : tree.query_ball_point(pt, tol))
            if (j != i && keep[j]) keep[j] = false;
    }

    kept.reserve(n);
    for (int i = 0; i < n; i++)
        if (keep[i]) kept.push_back(i);
    return kept;
}

}  // namespace sdf
