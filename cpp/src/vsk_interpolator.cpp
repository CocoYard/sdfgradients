#include "vsk_interpolator.h"
#include <stdexcept>

namespace sdf {

VSKInterpolator::VSKInterpolator(std::shared_ptr<Interpolator> psi,
                                 const std::string& kernel, double overlap,
                                 double reg, const std::string& partition,
                                 bool verbose, bool pair_local)
    : PUInterpolator(kernel, overlap, reg, partition, verbose, pair_local),
      psi_(std::move(psi))
{
    if (!psi_ || !psi_->is_trained())
        throw std::invalid_argument("VSKInterpolator: psi must be a fitted interpolator");
}

Eigen::MatrixXd VSKInterpolator::lift(const Eigen::MatrixXd& x, Eigen::MatrixXd* grad_psi) const {
    Eigen::VectorXd v;
    if (grad_psi)
        psi_->predict_with_gradients(x, v, *grad_psi);
    else
        v = psi_->predict(x);
    Eigen::MatrixXd X(x.rows(), 4);
    X << x, v;
    return X;
}

}  // namespace sdf
