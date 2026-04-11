#pragma once

/// Lightweight KDTree wrapper around nanoflann for 3D points stored in Eigen matrices.
/// Provides kNN search, radius search, and ball-tree all-pairs query.

#include <Eigen/Dense>
#include <nanoflann.hpp>
#include <vector>
#include <memory>

namespace sdf {

/// Adaptor: lets nanoflann index an Eigen::MatrixXd (N x 3, row-major points).
struct EigenMatAdaptor {
    const Eigen::MatrixXd& pts;
    EigenMatAdaptor(const Eigen::MatrixXd& m) : pts(m) {}
    inline size_t kdtree_get_point_count() const { return (size_t)pts.rows(); }
    inline double kdtree_get_pt(size_t idx, size_t dim) const { return pts(idx, dim); }
    template <class BBOX> bool kdtree_get_bbox(BBOX&) const { return false; }
};

using KDTreeIndex = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, EigenMatAdaptor>,
    EigenMatAdaptor, 3, int>;

class KDTree3D {
public:
    /// Build the tree from an (N, 3) matrix. The matrix must outlive the tree.
    explicit KDTree3D(const Eigen::MatrixXd& points, int leaf_size = 16)
        : adaptor_(points)
    {
        tree_ = std::make_unique<KDTreeIndex>(
            3, adaptor_, nanoflann::KDTreeSingleIndexAdaptorParams(leaf_size));
        tree_->buildIndex();
    }

    /// k-nearest-neighbor query. Returns (distances, indices), both length k.
    /// Distances are *squared* Euclidean.
    void query(const Eigen::Vector3d& pt, int k,
               std::vector<double>& out_dists_sq,
               std::vector<int>& out_indices) const
    {
        out_dists_sq.resize(k);
        out_indices.resize(k);
        nanoflann::KNNResultSet<double, int> resultSet(k);
        resultSet.init(out_indices.data(), out_dists_sq.data());
        tree_->findNeighbors(resultSet, pt.data());
    }

    /// Radius query: find all points within Euclidean distance `radius`.
    /// Returns indices. Internally searches with radius^2.
    std::vector<int> query_ball_point(const Eigen::Vector3d& center, double radius) const {
        std::vector<nanoflann::ResultItem<int, double>> matches;
        nanoflann::SearchParameters params;
        params.sorted = false;
        tree_->radiusSearch(center.data(), radius * radius, matches, params);
        std::vector<int> result;
        result.reserve(matches.size());
        for (auto& m : matches) result.push_back(m.first);
        return result;
    }

    /// Find the single nearest neighbor. Returns (squared_distance, index).
    std::pair<double, int> query_nearest(const Eigen::Vector3d& pt) const {
        int idx;
        double dist_sq;
        nanoflann::KNNResultSet<double, int> resultSet(1);
        resultSet.init(&idx, &dist_sq);
        tree_->findNeighbors(resultSet, pt.data());
        return {dist_sq, idx};
    }

private:
    EigenMatAdaptor adaptor_;
    std::unique_ptr<KDTreeIndex> tree_;
};

}  // namespace sdf
