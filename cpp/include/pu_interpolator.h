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
                   double reg=1e-5,
                   const std::string& partition = "box",
                   bool verbose = true,
                   bool pair_local = true);

    void fit(const Eigen::MatrixXd& points,
             const Eigen::VectorXd& values,
             const Eigen::MatrixXd* gradients = nullptr,
             const Eigen::VectorXi* mask = nullptr) override;

    Eigen::VectorXd predict(const Eigen::MatrixXd& x_new, int chunk_size = 5000) const override;

    Eigen::MatrixXd predict_gradients(const Eigen::MatrixXd& x_new, int chunk_size = 5000) const override;

    /// Fused evaluation. The blended value costs nothing extra here: the
    /// gradient blend already forms f = Σw·f_p / Σw per query point, and the
    /// per-patch solves already run through Duchon's fused kernel.
    void predict_with_gradients(const Eigen::MatrixXd& x_new,
                                Eigen::VectorXd& values,
                                Eigen::MatrixXd& grads,
                                int chunk_size = 5000) const override;

private:
    /// Shared implementation of predict_gradients / predict_with_gradients.
    /// `values_out` may be null when only the gradient is wanted.
    void eval_gradients(const Eigen::MatrixXd& x_new,
                        Eigen::VectorXd* values_out,
                        Eigen::MatrixXd& grads) const;

    /// Same result as eval_gradients, computed one query point at a time.
    ///
    /// eval_gradients inverts the query set into a patch → points table so each
    /// patch can solve its whole batch at once. That table costs O(#patches)
    /// per call no matter how many points are asked for — with far fewer points
    /// than patches it is pure overhead, and the table is almost all empty.
    /// This path skips it: per point, walk the patches containing it and blend
    /// on the spot. Chosen by eval_gradients when the query set is small
    /// relative to the patch count.
    void eval_gradients_pointwise(const Eigen::MatrixXd& x_new,
                                  Eigen::VectorXd* values_out,
                                  Eigen::MatrixXd& grads) const;

public:
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

    // If `partner` is non-null it is treated as a per-point index array
    // (partner[i] = row index of i's paired point, or -1) and remapped in
    // lockstep with the surviving points; pairs whose partner is dropped
    // become -1.
    void deduplicate(Eigen::MatrixXd& points, Eigen::VectorXd& values,
                     double tol = 1e-8, std::vector<int>* partner = nullptr) const;

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
    bool pair_local_ = true;   // augment each local solve with missing input/projection partners
    /// This value is used to filter out far-away points when solving interpolation because far field 
    /// effects are negligible for the 0-level-set. If no noise, this can be set to a small value like 0.2.
    double dist_threshold_ = 2;

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
    double reg_ = 0;

    void build_patch_bvh();
    void query_patches_containing(
        const Eigen::Vector3d& pt,
        std::vector<int>& out) const;
};

}  // namespace sdf
