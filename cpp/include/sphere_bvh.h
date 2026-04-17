#pragma once

#include <Eigen/Dense>
#include <vector>

namespace sdf {

// Median-split BVH over a set of spheres. Built once, reused by visibility
// and clamp so they don't need a giant per-sphere neighbor list just to do
// "is point q inside any sphere?" checks.
class SphereBVH {
public:
    // Build from (N, 3) centers + (N,) radii. Stores a float copy internally;
    // the caller's matrices do not need to outlive this object.
    SphereBVH(const Eigen::MatrixXd& centers, const Eigen::VectorXd& radii,
              int leaf_size = 16);

    // True iff query point lies strictly inside some sphere j (with margin
    // epsilon):  |q - c_j|^2  <  (|r_j| - epsilon)^2, r_j - eps > 0.
    // exclude_idx: skip sphere j == exclude_idx (pass -1 to disable).
    bool point_inside_any(double qx, double qy, double qz,
                          double epsilon, int exclude_idx = -1) const;

    // Fast-path for callers that already have a small candidate list
    // (e.g. ngbrs_list[i]): check `count` indices from `indices` against
    // the stored float sphere data. No tree traversal.
    bool any_sphere_contains(double qx, double qy, double qz,
                             double epsilon,
                             const int* indices, int count) const;

    int size() const { return (int)cx_.size(); }

private:
    struct Node {
        float lo[3], hi[3];
        int left, right;       // -1 if leaf
        int leaf_start, leaf_count;
    };

    void build_recursive(int* idx_buf, int start, int count, int leaf_size);

    std::vector<Node> nodes_;
    std::vector<int>  leaves_;
    std::vector<float> cx_, cy_, cz_, ra_;
};

}  // namespace sdf
