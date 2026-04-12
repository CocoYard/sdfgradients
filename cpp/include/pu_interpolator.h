#pragma once

#include "interpolator.h"
#include "duchon_interpolator.h"
#include "kdtree.h"
#include <memory>
#include <vector>

namespace sdf {

/// Partition of Unity interpolator.
/// Mirrors Python PUInterpolator.
///
/// Decomposes the domain into overlapping patches (KDTree median split),
/// fits a local DuchonInterpolator on each patch, and blends predictions
/// with Wendland C2 weights.
class PUInterpolator : public Interpolator {
public:
    PUInterpolator(const std::string& kernel = "cubic",
                   double overlap = 0.25,
                   int min_points = 10,
                   int max_points = 200,
                   const std::string& partition = "box");

    void fit(const Eigen::MatrixXd& points,
             const Eigen::VectorXd& values,
             const Eigen::MatrixXd* gradients = nullptr,
             const Eigen::VectorXi* mask = nullptr) override;

    Eigen::VectorXd predict(const Eigen::MatrixXd& x_new, int chunk_size = 5000) const override;

    Eigen::MatrixXd predict_gradients(const Eigen::MatrixXd& x_new, int chunk_size = 5000) const override;

    bool is_trained() const override { return trained_; }

private:
    struct Patch {
        Eigen::Vector3d center;
        Eigen::Vector3d half_ext;   // box: half-extents; sphere: (R, R, R)
        double bsphere_radius;
        std::unique_ptr<DuchonInterpolator> interp;
    };

    static double wendland_weight(double r, double radius);
    static double box_weight(const Eigen::Vector3d& pt,
                             const Eigen::Vector3d& center,
                             const Eigen::Vector3d& half_ext);

    static void deduplicate(Eigen::MatrixXd& points, Eigen::VectorXd& values, double tol = 1e-8);

    struct PatchInfo {
        Eigen::Vector3d center;
        Eigen::Vector3d half_ext;
        std::vector<int> ext_idx;
    };

    std::vector<PatchInfo> kdtree_partition(const Eigen::MatrixXd& pts, KDTree3D& tree);

    std::string kernel_;
    std::string partition_type_;
    double overlap_;
    int min_points_;
    int max_points_;
    int max_ext_points_ = 675;

    std::vector<Patch> patches_;
    bool trained_ = false;
    bool use_box_ = false;
    double dist_threshold_ = 0.2;

    // For fallback nearest-patch lookup (tree cached after fit(), reused every predict())
    Eigen::MatrixXd patch_centers_;
    Eigen::VectorXd patch_radii_;
    std::unique_ptr<KDTree3D> patch_tree_;

    // BVH over patch AABBs for fast point→containing-patches queries.
public:
    struct PatchBVHNode {
        double lo[3], hi[3];
        int left, right;           // -1 if leaf
        int leaf_start, leaf_count;
    };
private:
    std::vector<PatchBVHNode> patch_bvh_nodes_;
    std::vector<int> patch_bvh_leaves_;
    std::vector<double> patch_aabb_lo_, patch_aabb_hi_;  // flat 3*np

    void build_patch_bvh();
    void query_patches_containing(
        const Eigen::Vector3d& pt,
        std::vector<int>& out) const;
};

}  // namespace sdf
