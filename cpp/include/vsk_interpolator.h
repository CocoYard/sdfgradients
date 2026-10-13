#pragma once

#include "pu_interpolator.h"
#include <memory>

namespace sdf {

/// Variably scaled kernel (VSK) interpolator, after Bozzini, Lenarduzzi,
/// Rossini and Schaback: every point is lifted one dimension up,
///   x  ->  (x, psi(x)),
/// and the PU local RBF solves run on the lifted points with the ordinary
/// stationary kernel, so they stay as well-posed as the plain ones. The patch
/// partition and the blend weights stay in x.
///
/// psi is a plug-in scalar function: any fitted Interpolator. For now it is the
/// RBF interpolant of the input SDF samples.
class VSKInterpolator : public PUInterpolator {
public:
    VSKInterpolator(std::shared_ptr<Interpolator> psi,
                    const std::string& kernel = "cubic",
                    double overlap = 0.25,
                    double reg = 1e-5,
                    const std::string& partition = "box",
                    bool verbose = true,
                    bool pair_local = true);

protected:
    Eigen::MatrixXd lift(const Eigen::MatrixXd& x, Eigen::MatrixXd* grad_psi) const override;

private:
    std::shared_ptr<Interpolator> psi_;
};

}  // namespace sdf
