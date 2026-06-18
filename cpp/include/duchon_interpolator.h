#pragma once

#include "interpolator.h"

namespace sdf {

/// Duchon RBF interpolator (cubic kernel r^3 for 3D).
/// Mirrors Python DuchonInterpolator.
///
/// Solves:
///   [ K  P ] [ alpha ]   [ values ]
///   [ P' 0 ] [  p,q  ] = [   0    ]
///
/// predict(x) = sum_j alpha_j * kernel(||x - x_j||) + x . p + q
class DuchonInterpolator : public Interpolator {
public:
    explicit DuchonInterpolator(const std::string& kernel = "cubic", double reg = 1e-5);

    void fit(const Eigen::MatrixXd& points,
             const Eigen::VectorXd& values,
             const Eigen::MatrixXd* gradients = nullptr,
             const Eigen::VectorXi* mask = nullptr) override;

    Eigen::VectorXd predict(const Eigen::MatrixXd& x_new, int chunk_size = 500) const override;

    Eigen::MatrixXd predict_gradients(const Eigen::MatrixXd& x_new, int chunk_size = 500) const override;

    bool is_trained() const override { return trained_; }

private:
    double kernel_eval(double r) const;

    // Single-block (no chunking) kernels. The public predict/predict_gradients
    // split the query set into blocks and run these in parallel across blocks.
    Eigen::VectorXd predict_block(const Eigen::MatrixXd& x_new) const;
    Eigen::MatrixXd predict_gradients_block(const Eigen::MatrixXd& x_new) const;

    void compute_coefficients(const Eigen::MatrixXd& pts,
                              const Eigen::VectorXd& vals);

    std::string kernel_type_;
    Eigen::MatrixXd points_;    // (N, 3)
    Eigen::VectorXd alpha_;     // (N,)
    Eigen::VectorXd p_;         // (3,)
    double q_ = 0.0;
    bool trained_ = false;
    double dist_threshold_ = 0.2;
    double reg_ = 1e-5; // regularization
};

}  // namespace sdf
