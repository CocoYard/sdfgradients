#include "duchon_interpolator.h"
#include <cmath>
#include <iostream>

namespace sdf {

DuchonInterpolator::DuchonInterpolator(const std::string& kernel)
    : kernel_type_(kernel) {}

double DuchonInterpolator::kernel_eval(double r) const {
    if (kernel_type_ == "thin_plate") {
        return r * r * std::log(r + 1e-10);
    }
    return r * r * r;  // cubic
}

// ── fit ─────────────────────────────────────────────────────────────

void DuchonInterpolator::fit(const Eigen::MatrixXd& points,
                              const Eigen::VectorXd& values,
                              const Eigen::MatrixXd* gradients,
                              const Eigen::VectorXi* mask) {
    Eigen::MatrixXd pts;
    Eigen::VectorXd vals;

    if (gradients) {
        // Add projection points: proj = point - value * gradient
        Eigen::MatrixXd projections;
        if (mask) {
            // Only use masked points for projections
            int count = mask->sum();
            projections.resize(count, points.cols());
            int k = 0;
            for (int i = 0; i < (int)points.rows(); i++) {
                if ((*mask)(i)) {
                    projections.row(k++) = points.row(i) - values(i) * gradients->row(i);
                }
            }
        } else {
            projections.resize(points.rows(), points.cols());
            for (int i = 0; i < (int)points.rows(); i++) {
                projections.row(i) = points.row(i) - values(i) * gradients->row(i);
            }
        }
        pts.resize(points.rows() + projections.rows(), points.cols());
        pts << points, projections;
        vals.resize(values.size() + projections.rows());
        vals << values, Eigen::VectorXd::Zero(projections.rows());
    } else {
        pts = points;
        vals = values;
    }

    // If too many points, keep only those with small |value|
    if (vals.size() > 5000) {
        std::vector<int> keep;
        for (int i = 0; i < (int)vals.size(); i++) {
            if (std::abs(vals(i)) < dist_threshold_)
                keep.push_back(i);
        }
        Eigen::MatrixXd pts_filtered(keep.size(), pts.cols());
        Eigen::VectorXd vals_filtered(keep.size());
        for (int i = 0; i < (int)keep.size(); i++) {
            pts_filtered.row(i) = pts.row(keep[i]);
            vals_filtered(i) = vals(keep[i]);
        }
        std::cout << "Warning: too many points, only keeping " << keep.size()
                  << " points with abs(value) < " << dist_threshold_ << " for fitting.\n";
        pts = pts_filtered;
        vals = vals_filtered;
    }

    points_ = pts;
    compute_coefficients(pts, vals);
    trained_ = true;
}

// ── compute_coefficients ────────────────────────────────────────────

void DuchonInterpolator::compute_coefficients(const Eigen::MatrixXd& pts,
                                               const Eigen::VectorXd& vals) {
    int n = (int)pts.rows();
    int d = (int)pts.cols();
    int size = n + d + 1;

    // Compute pairwise distance matrix
    Eigen::MatrixXd K = Eigen::MatrixXd::Zero(size, size);
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            double r = (pts.row(i) - pts.row(j)).norm();
            double v = kernel_eval(r);
            K(i, j) = v;
            K(j, i) = v;
        }
        // diagonal = 0 (kernel(0) = 0 for cubic and thin_plate after fill_diagonal(0))
    }

    // P = [points | 1]  shape: (n, d+1)
    // K[:n, n:] = P;  K[n:, :n] = P.T
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < d; k++) {
            K(i, n + k) = pts(i, k);
            K(n + k, i) = pts(i, k);
        }
        K(i, n + d) = 1.0;
        K(n + d, i) = 1.0;
    }

    // RHS
    Eigen::VectorXd y = Eigen::VectorXd::Zero(size);
    y.head(n) = vals;

    // Solve via lstsq (SVD)
    Eigen::VectorXd coeffs = Eigen::BDCSVD<Eigen::MatrixXd, Eigen::ComputeThinU | Eigen::ComputeThinV>(K).solve(y);

    alpha_ = coeffs.head(n);
    p_ = coeffs.segment(n, d);
    q_ = coeffs(n + d);
}

// ── predict ─────────────────────────────────────────────────────────

Eigen::VectorXd DuchonInterpolator::predict(const Eigen::MatrixXd& x_new,
                                             int chunk_size) const {
    int M = (int)x_new.rows();
    if (M > chunk_size) {
        Eigen::VectorXd result(M);
        for (int s = 0; s < M; s += chunk_size) {
            int len = std::min(chunk_size, M - s);
            result.segment(s, len) = predict(x_new.middleRows(s, len), chunk_size);
        }
        return result;
    }

    int N = (int)points_.rows();

    // Compute pairwise distance matrix: dist(i,j) = ||x_new[i] - points_[j]||
    // Using ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b
    Eigen::VectorXd x_sq = x_new.rowwise().squaredNorm();          // (M,)
    Eigen::VectorXd p_sq = points_.rowwise().squaredNorm();         // (N,)
    Eigen::MatrixXd dist2 = x_sq.replicate(1, N) + p_sq.transpose().replicate(M, 1)
                            - 2.0 * x_new * points_.transpose();   // (M, N)
    // Clamp negatives from numerical error
    dist2 = dist2.cwiseMax(0.0);
    Eigen::MatrixXd dist = dist2.cwiseSqrt();                      // (M, N)

    // Apply kernel elementwise
    Eigen::MatrixXd K(M, N);
    if (kernel_type_ == "thin_plate") {
        // r^2 * log(r + eps)
        K = dist2.array() * (dist.array() + 1e-10).log();
    } else {
        // cubic: r^3
        K = dist.array() * dist2.array();
    }

    // result = K * alpha + x_new * p + q
    Eigen::VectorXd result = K * alpha_ + x_new * p_ + Eigen::VectorXd::Constant(M, q_);

    return result;
}

// ── predict_gradients ───────────────────────────────────────────────

Eigen::MatrixXd DuchonInterpolator::predict_gradients(const Eigen::MatrixXd& x_new,
                                                       int chunk_size) const {
    int M = (int)x_new.rows();
    if (M > chunk_size) {
        Eigen::MatrixXd result(M, 3);
        for (int s = 0; s < M; s += chunk_size) {
            int len = std::min(chunk_size, M - s);
            result.middleRows(s, len) = predict_gradients(x_new.middleRows(s, len), chunk_size);
        }
        return result;
    }

    int N = (int)points_.rows();
    int dim = (int)points_.cols();

    // dist2(i,j) = ||x_new[i] - points_[j]||^2
    Eigen::VectorXd x_sq = x_new.rowwise().squaredNorm();
    Eigen::VectorXd p_sq = points_.rowwise().squaredNorm();
    Eigen::MatrixXd dist2 = x_sq.replicate(1, N) + p_sq.transpose().replicate(M, 1)
                            - 2.0 * x_new * points_.transpose();
    dist2 = dist2.cwiseMax(0.0);
    Eigen::MatrixXd dist = dist2.cwiseSqrt().cwiseMax(1e-10);  // (M, N)

    // kernel gradient scalar: d kernel / d r
    // cubic: d/dr(r^3) = 3r^2, so d/dx = 3r^2 * (diff/r) = 3r * diff
    //   => weight per (i,j) = alpha_j * 3 * r
    // thin_plate: d/dr(r^2 log r) = 2r log r + r = r(2 log r + 1)
    //   => weight per (i,j) = alpha_j * (2 log r + 1)
    Eigen::MatrixXd W(M, N);  // W(i,j) = alpha_j * (d kernel/dr / r)
    if (kernel_type_ == "thin_plate") {
        // (2*log(r) + 1) * diff => per-component: weight = alpha_j * (2*log(r)+1)
        W = (2.0 * dist.array().log() + 1.0).matrix();
    } else {
        // 3*r * diff => per-component: weight = alpha_j * 3 * r
        W = 3.0 * dist;
    }
    // Multiply each column j by alpha_j
    W = W * alpha_.asDiagonal();  // (M, N)

    // grads(i, k) = p_(k) + sum_j W(i,j) * (x_new(i,k) - points_(j,k))
    // = p_(k) + x_new(i,k) * sum_j W(i,j) - (W * points_)(i,k)
    Eigen::VectorXd W_rowsum = W.rowwise().sum();  // (M,)
    Eigen::MatrixXd scaled_x = (x_new.array().colwise() * W_rowsum.array()).matrix();
    Eigen::MatrixXd grads = scaled_x - W * points_;
    grads.rowwise() += p_.transpose();

    return grads;
}

}  // namespace sdf
