#include "duchon_interpolator.h"
#include <cmath>
#include <chrono>
#include <iostream>
#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace sdf {

DuchonInterpolator::DuchonInterpolator(const std::string& kernel, double reg)
    : kernel_type_(kernel), reg_(reg) {}

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
    sample_points_ = points;
    sample_values_ = values;

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

    auto t0 = std::chrono::high_resolution_clock::now();

    // Compute pairwise distance matrix (vectorized)
    Eigen::MatrixXd K = Eigen::MatrixXd::Zero(size, size);
    {
        Eigen::VectorXd sq = pts.rowwise().squaredNorm();                  // (n,)
        Eigen::MatrixXd dist2 = sq.replicate(1, n) + sq.transpose().replicate(n, 1)
                                - 2.0 * pts * pts.transpose();            // (n, n)
        dist2 = dist2.cwiseMax(0.0);

        if (kernel_type_ == "thin_plate") {
            Eigen::MatrixXd dist = dist2.cwiseSqrt();
            K.topLeftCorner(n, n) = dist2.array() * (dist.array() + 1e-10).log();
        } else {
            // cubic: r^3 = sqrt(dist2)^3 = dist2 * sqrt(dist2)
            Eigen::MatrixXd dist = dist2.cwiseSqrt();
            K.topLeftCorner(n, n) = dist.array() * dist2.array();
        }
    }

    // P = [points | 1]  shape: (n, d+1)
    // K[:n, n:] = P;  K[n:, :n] = P.T
    K.block(0, n, n, d) = pts;
    K.block(n, 0, d, n) = pts.transpose();
    K.block(0, n + d, n, 1).setOnes();
    K.block(n + d, 0, 1, n).setOnes();
    K.topLeftCorner(n, n).diagonal().array() += reg_;

    auto t1 = std::chrono::high_resolution_clock::now();

    // RHS
    Eigen::VectorXd y = Eigen::VectorXd::Zero(size);
    y.head(n) = vals;

    // Solve system via PartialPivLU decomposition
    Eigen::VectorXd coeffs = K.partialPivLu().solve(y);

    auto t2 = std::chrono::high_resolution_clock::now();
    double build_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double solve_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    // std::cout << "[Duchon] n=" << n << "  build K: " << build_ms
    //           << "ms  solve: " << solve_ms << "ms\n";

    alpha_ = coeffs.head(n);
    p_ = coeffs.segment(n, d);
    q_ = coeffs(n + d);
}

// ── chunking helper ─────────────────────────────────────────────────
//
// Pick a per-block row count that (a) never exceeds `chunk_size` — which caps
// the peak M×N scratch matrix — and (b) is small enough to hand every OpenMP
// worker several blocks for load balancing. Without (b), a query set only
// slightly larger than chunk_size would split into 1-2 blocks and leave most
// cores idle.
//
// When already inside a parallel region (e.g. PUInterpolator calls this from a
// parallel-for over patches), we do NOT subdivide and the caller runs the
// blocks serially — the outer loop already saturates the cores, and splitting
// here would only add overhead (or, with OMP_NESTED on, oversubscribe).
static inline int duchon_block_rows(int M, int chunk_size) {
    int block = std::max(chunk_size, 1);
#ifdef USE_OPENMP
    if (!omp_in_parallel()) {
        int nthreads = omp_get_max_threads();
        if (nthreads > 1) {
            int target_blocks = nthreads * 4;
            int even = (M + target_blocks - 1) / target_blocks;
            block = std::min(block, std::max(even, 1));
        }
    }
#endif
    return block;
}

// ── predict ─────────────────────────────────────────────────────────
//
// The heavy per-block work — the elementwise kernel transform (pow/log) and
// the K·alpha matrix-vector product — is not parallelized by Eigen, so we
// parallelize across blocks here. Eigen's own GEMM threading would nest inside
// this loop, but OpenMP nested parallelism is off by default, so each block's
// GEMM runs single-threaded; that is exactly what we want (no oversubscription).

Eigen::VectorXd DuchonInterpolator::predict(const Eigen::MatrixXd& x_new,
                                             int chunk_size) const {
    const int M = (int)x_new.rows();
    if (M == 0) return Eigen::VectorXd(0);

    const int block = duchon_block_rows(M, chunk_size);
    const int nblocks = (M + block - 1) / block;

    Eigen::VectorXd result(M);
    #pragma omp parallel for schedule(dynamic, 1) if(!omp_in_parallel())
    for (int b = 0; b < nblocks; b++) {
        const int s = b * block;
        const int len = std::min(block, M - s);
        result.segment(s, len) = predict_block(x_new.middleRows(s, len));
    }
    return result;
}

Eigen::VectorXd DuchonInterpolator::predict_block(const Eigen::MatrixXd& x_new) const {
    int M = (int)x_new.rows();
    int N = (int)points_.rows();

    // Compute pairwise squared-distance matrix (M×N), then convert in-place to
    // kernel values.  Peak allocation is 1×(M×N) instead of the previous 3×(M×N).
    //   cubic:      r³  = (r²)^(3/2)
    //   thin_plate: r²·log(r+ε) = dist2 · log(√dist2 + ε)
    Eigen::VectorXd x_sq = x_new.rowwise().squaredNorm();          // (M,)
    Eigen::VectorXd p_sq = points_.rowwise().squaredNorm();         // (N,)
    Eigen::MatrixXd dist2 = x_sq.replicate(1, N) + p_sq.transpose().replicate(M, 1)
                            - 2.0 * x_new * points_.transpose();   // (M, N)
    dist2 = dist2.cwiseMax(0.0);

    // Overwrite dist2 with kernel values in-place (no extra M×N matrix)
    if (kernel_type_ == "thin_plate") {
        dist2.array() *= (dist2.array().sqrt() + 1e-10).log();
    } else {
        dist2 = dist2.array().pow(1.5);
    }

    // result = K * alpha + x_new * p + q  (dist2 now holds K)
    Eigen::VectorXd result = dist2 * alpha_ + x_new * p_ + Eigen::VectorXd::Constant(M, q_);

    return result;
}

// ── predict_gradients ───────────────────────────────────────────────

Eigen::MatrixXd DuchonInterpolator::predict_gradients(const Eigen::MatrixXd& x_new,
                                                       int chunk_size) const {
    const int M = (int)x_new.rows();
    if (M == 0) return Eigen::MatrixXd(0, 3);

    const int block = duchon_block_rows(M, chunk_size);
    const int nblocks = (M + block - 1) / block;

    Eigen::MatrixXd result(M, 3);
    #pragma omp parallel for schedule(dynamic, 1) if(!omp_in_parallel())
    for (int b = 0; b < nblocks; b++) {
        const int s = b * block;
        const int len = std::min(block, M - s);
        result.middleRows(s, len) = predict_gradients_block(x_new.middleRows(s, len));
    }
    return result;
}

Eigen::MatrixXd DuchonInterpolator::predict_gradients_block(const Eigen::MatrixXd& x_new) const {
    int M = (int)x_new.rows();
    int N = (int)points_.rows();
    int dim = (int)points_.cols();

    // Compute squared distances (M×N), then overwrite in-place with the per-(i,j)
    // gradient weight W(i,j) = alpha_j · (d kernel/dr).
    // Peak allocation is 1×(M×N) instead of the previous 3×(M×N).
    //
    // Gradient weights (d kernel/dr, as a function of r = sqrt(dist2)):
    //   cubic:      d(r³)/dr = 3r      → W = 3·sqrt(dist2)
    //   thin_plate: d(r²·log r)/dr = r·(2·log r + 1)
    //               The actual gradient vector is W·(x-p)/r, so the scalar
    //               factor per diff component is (2·log r + 1) = W/r.
    //               Since we factor out /r later via the rowsum trick,
    //               we store (2·log r + 1) directly.
    Eigen::VectorXd x_sq = x_new.rowwise().squaredNorm();
    Eigen::VectorXd p_sq = points_.rowwise().squaredNorm();
    Eigen::MatrixXd dist2 = x_sq.replicate(1, N) + p_sq.transpose().replicate(M, 1)
                            - 2.0 * x_new * points_.transpose();
    dist2 = dist2.cwiseMax(0.0);

    // Overwrite dist2 with W (no extra M×N allocation)
    if (kernel_type_ == "thin_plate") {
        dist2 = (2.0 * dist2.array().sqrt().cwiseMax(1e-10).log() + 1.0).matrix();
    } else {
        dist2 = (3.0 * dist2.array().sqrt().cwiseMax(1e-10)).matrix();
    }
    // W = dist2; scale each column j by alpha_j
    dist2 = dist2 * alpha_.asDiagonal();

    // grads(i,k) = p_(k) + x(i,k)·Σ_j W(i,j) − (W·points_)(i,k)
    Eigen::VectorXd W_rowsum = dist2.rowwise().sum();
    Eigen::MatrixXd scaled_x = (x_new.array().colwise() * W_rowsum.array()).matrix();
    Eigen::MatrixXd grads = scaled_x - dist2 * points_;
    grads.rowwise() += p_.transpose();

    return grads;
}

}  // namespace sdf
