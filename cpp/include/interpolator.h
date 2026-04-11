#pragma once

#include <Eigen/Dense>
#include <string>
#include <memory>

namespace sdf {

/// Abstract base class for interpolation, mirroring Python's Interpolator ABC.
/// Subclasses must implement fit(), predict(), predict_gradients().
/// sample_best_gradients() is a concrete method that calls predict().
class Interpolator {
public:
    virtual ~Interpolator() = default;

    /// Fit the interpolator to (points, values).
    /// @param gradients  if non-null, projection points are added (points - values * gradients)
    /// @param mask       if non-null, only masked points contribute projections
    virtual void fit(const Eigen::MatrixXd& points,
                     const Eigen::VectorXd& values,
                     const Eigen::MatrixXd* gradients = nullptr,
                     const Eigen::VectorXi* mask = nullptr) = 0;

    /// Predict SDF values at query points. (M, 3) -> (M,)
    virtual Eigen::VectorXd predict(const Eigen::MatrixXd& x_new, int chunk_size = 500) const = 0;

    /// Predict gradients at query points. (M, 3) -> (M, 3)
    virtual Eigen::MatrixXd predict_gradients(const Eigen::MatrixXd& x_new, int chunk_size = 500) const = 0;

    virtual bool is_trained() const = 0;

    /// Find best gradient directions via coarse Fibonacci sweep + cone refinement.
    /// Concrete method — works for any subclass since it only calls predict().
    /// Mirrors Python Interpolator._sample_best_gradients_3d().
    ///
    /// @param points        (N, 3)
    /// @param sdf_values    (N,)
    /// @param num_coarse    Fibonacci directions for initial sweep
    /// @param refine_steps  cone-refinement iterations
    /// @param num_refine    directions per refinement step
    /// @param initial_guess (N, 3) optional; NaN rows trigger Fibonacci sweep
    /// @param chunk_size    batch size for predict() calls
    /// @return              (N, 3) unit gradient directions
    Eigen::MatrixXd sample_best_gradients(
        const Eigen::MatrixXd& points,
        const Eigen::VectorXd& sdf_values,
        int num_coarse   = 24,
        int refine_steps = 4,
        int num_refine   = 5,
        const Eigen::MatrixXd* initial_guess = nullptr,
        int chunk_size   = 200) const;
};

}  // namespace sdf
