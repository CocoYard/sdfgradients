#pragma once

#include "interpolator.h"
#include "duchon_interpolator.h"
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

    // For fallback nearest-patch lookup
    Eigen::MatrixXd patch_centers_;
    Eigen::VectorXd patch_radii_;
};

}  // namespace sdf
