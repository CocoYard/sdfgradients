#include "duchon_interpolator.h"
#include "dedup.h"
#include <cmath>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <Eigen/SVD>
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

    const bool hermite = normals_in_.rows() > 0;
    if (hermite) {
        if (normals_in_.rows() != points.rows() || normals_in_.cols() != 3 || points.cols() != 3)
            throw std::invalid_argument("DuchonInterpolator: normals must be (N, 3) for (N, 3) points");
        if (gradients)
            throw std::invalid_argument("DuchonInterpolator: normals and gradients are exclusive");
        if (kernel_type_ != "cubic" || reduced_poly_)
            throw std::invalid_argument("DuchonInterpolator: normals need the cubic kernel and the full polynomial");
    }
    // src[i] = fit() row of training point i, so the normals follow dedup/filtering
    std::vector<int> src;

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
    src.resize(pts.rows());
    for (int i = 0; i < (int)src.size(); i++) src[i] = i;

    // Deduplicate near-coincident training points (e.g. short-arc degenerate
    // projections that collapse onto the same point). With reg=0 duplicate rows
    // make the RBF kernel matrix singular and the solve unstable, poisoning the
    // field. PU dedups globally before building patches, so it disables this.
    if (dedup_) {
        std::vector<int> kept = dedup_keep_indices(pts, vals, 5e-4);
        if ((int)kept.size() < (int)pts.rows()) {
            if (verbose_)
                std::cout << "[Duchon] dedup removed " << (pts.rows() - (int)kept.size())
                          << " points\n";
            Eigen::MatrixXd dp((int)kept.size(), pts.cols());
            Eigen::VectorXd dv((int)kept.size());
            std::vector<int> ds((int)kept.size());
            for (int i = 0; i < (int)kept.size(); i++) {
                dp.row(i) = pts.row(kept[i]);
                dv(i)     = vals(kept[i]);
                ds[i]     = src[kept[i]];
            }
            src = std::move(ds);
            pts = std::move(dp);
            vals = std::move(dv);
        }
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
        std::vector<int> src_filtered(keep.size());
        for (int i = 0; i < (int)keep.size(); i++) {
            pts_filtered.row(i) = pts.row(keep[i]);
            vals_filtered(i) = vals(keep[i]);
            src_filtered[i] = src[keep[i]];
        }
        src = std::move(src_filtered);
        std::cout << "Warning: too many points, only keeping " << keep.size()
                  << " points with abs(value) < " << dist_threshold_ << " for fitting.\n";
        pts = pts_filtered;
        vals = vals_filtered;
    }

    Eigen::MatrixXd nrm;
    if (hermite) {
        nrm.resize(pts.rows(), 3);
        for (int i = 0; i < (int)src.size(); i++) nrm.row(i) = normals_in_.row(src[i]);
    }

    points_ = pts;
    compute_coefficients(pts, vals, nrm);
    trained_ = true;
}

// ── compute_coefficients ────────────────────────────────────────────

void DuchonInterpolator::compute_coefficients(const Eigen::MatrixXd& pts,
                                               const Eigen::VectorXd& vals,
                                               const Eigen::MatrixXd& normals) {
    int n = (int)pts.rows();
    int d = (int)pts.cols();

    // Points with a finite normal take the three extra rows grad f(c_k) = n_k.
    std::vector<int> herm;
    for (int i = 0; i < (int)normals.rows(); i++)
        if (normals.row(i).allFinite()) herm.push_back(i);
    const int m = (int)herm.size();
    herm_pts_.resize(m, 3);
    for (int k = 0; k < m; k++) herm_pts_.row(k) = pts.row(herm[k]);

    // Polynomial basis: [X 1], or with reduced_poly_ the coordinates along the
    // directions the points actually span, [(X - mu) B 1]. An affine function
    // on the hull is still one on R^d, so it maps back to p, q below.
    Eigen::RowVectorXd mu = Eigen::RowVectorXd::Zero(d);
    Eigen::MatrixXd B = Eigen::MatrixXd::Identity(d, d);
    if (reduced_poly_) {
        mu = pts.colwise().mean();
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(pts.rowwise() - mu, Eigen::ComputeThinV);
        const Eigen::VectorXd& sv = svd.singularValues();
        int r = 0;
        while (r < (int)sv.size() && sv(r) > 1e-3 * sv(0)) r++;
        B = svd.matrixV().leftCols(r);
    }
    const int r = (int)B.cols();
    const int pc = n + 3 * m;          // first polynomial row/column
    const int size = pc + r + 1;

    // Rows/columns: [0, n) values -> alpha, [n, pc) gradients (3 per Hermite
    // point) -> beta, [pc, size) polynomial -> p, q.
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

    // Hermite blocks for phi = r^3, with grad phi(x) = 3 |x| x and
    // Hessian 3 (|x| I + x x^T / |x|) (0 at x = 0):
    //   value-gradient    K(i, (k,a))     = -3 |c_i - e_k| (c_i - e_k)_a
    //   gradient-gradient K((k,a), (l,b)) = -3 (r d_ab + x_a x_b / r),  x = e_k - e_l
    for (int k = 0; k < m; k++) {
        const Eigen::RowVector3d e = herm_pts_.row(k);
        for (int i = 0; i < n; i++) {
            const Eigen::RowVector3d x = pts.row(i) - e;
            const double rx = x.norm();
            for (int a = 0; a < 3; a++) {
                const double v = -3.0 * rx * x(a);
                K(i, n + 3 * k + a) = v;
                K(n + 3 * k + a, i) = v;
            }
        }
        for (int l = 0; l < m; l++) {
            const Eigen::RowVector3d x = e - herm_pts_.row(l);
            const double rx = x.norm();
            if (rx < 1e-14) continue;
            for (int a = 0; a < 3; a++)
                for (int b = 0; b < 3; b++)
                    K(n + 3 * k + a, n + 3 * l + b) =
                        -3.0 * ((a == b ? rx : 0.0) + x(a) * x(b) / rx);
        }
    }

    // Polynomial: value rows hold [(X - mu) B, 1], gradient rows its derivative B^T e_a.
    const Eigen::MatrixXd P = (pts.rowwise() - mu) * B;
    K.block(0, pc, n, r) = P;
    K.block(pc, 0, r, n) = P.transpose();
    K.block(0, pc + r, n, 1).setOnes();
    K.block(pc + r, 0, 1, n).setOnes();
    for (int k = 0; k < m; k++)
        for (int a = 0; a < 3; a++) {
            K.block(n + 3 * k + a, pc, 1, r) = B.row(a);
            K.block(pc, n + 3 * k + a, r, 1) = B.row(a).transpose();
        }
    K.topLeftCorner(n, n).diagonal().array() += reg_;

    Eigen::VectorXd y = Eigen::VectorXd::Zero(size);
    y.head(n) = vals;
    for (int k = 0; k < m; k++)
        y.segment(n + 3 * k, 3) = normals.row(herm[k]).transpose();

    Eigen::VectorXd coeffs = K.partialPivLu().solve(y);

    alpha_ = coeffs.head(n);
    beta_.resize(m, 3);
    for (int k = 0; k < m; k++) beta_.row(k) = coeffs.segment(n + 3 * k, 3).transpose();
    p_ = B * coeffs.segment(pc, r);
    q_ = coeffs(pc + r) - mu.dot(p_);
}

// ── Hermite terms ───────────────────────────────────────────────────
//
// Each Hermite point adds g_k(x) = -3 r s with r = |x - e_k|, s = beta_k . (x - e_k),
// whose gradient is -3 (s/r (x - e_k) + r beta_k).

void DuchonInterpolator::add_hermite_terms(const Eigen::MatrixXd& x_new,
                                           Eigen::VectorXd* value,
                                           Eigen::MatrixXd* grad) const {
    const int m = (int)herm_pts_.rows();
    if (m == 0) return;
    const int M = (int)x_new.rows();

    Eigen::MatrixXd R = x_new.rowwise().squaredNorm().replicate(1, m)
                        + herm_pts_.rowwise().squaredNorm().transpose().replicate(M, 1)
                        - 2.0 * x_new * herm_pts_.transpose();               // (M, m)
    R = R.cwiseMax(0.0).cwiseSqrt();
    const Eigen::VectorXd eb = herm_pts_.cwiseProduct(beta_).rowwise().sum(); // e_k . beta_k
    Eigen::MatrixXd S = x_new * beta_.transpose();                           // (M, m)
    S.rowwise() -= eb.transpose();

    if (value) *value -= 3.0 * R.cwiseProduct(S).rowwise().sum();
    if (grad) {
        Eigen::MatrixXd W = (R.array() > 1e-14).select(S.array() / R.array(), 0.0).matrix();
        Eigen::MatrixXd g = x_new.array().colwise() * W.rowwise().sum().array();
        g += R * beta_ - W * herm_pts_;
        *grad -= 3.0 * g;
    }
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
    add_hermite_terms(x_new, &result, nullptr);

    return result;
}

// ── predict_gradients ───────────────────────────────────────────────

Eigen::MatrixXd DuchonInterpolator::predict_gradients(const Eigen::MatrixXd& x_new,
                                                       int chunk_size) const {
    const int M = (int)x_new.rows();
    if (M == 0) return Eigen::MatrixXd(0, points_.cols());

    const int block = duchon_block_rows(M, chunk_size);
    const int nblocks = (M + block - 1) / block;

    Eigen::MatrixXd result(M, points_.cols());
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
    add_hermite_terms(x_new, nullptr, &grads);

    return grads;
}

// ── predict_with_gradients (fused) ──────────────────────────────────

void DuchonInterpolator::predict_with_gradients(const Eigen::MatrixXd& x_new,
                                                 Eigen::VectorXd& values,
                                                 Eigen::MatrixXd& grads,
                                                 int chunk_size) const {
    const int M = (int)x_new.rows();
    values.resize(M);
    grads.resize(M, points_.cols());
    if (M == 0) return;

    const int block = duchon_block_rows(M, chunk_size);
    const int nblocks = (M + block - 1) / block;

    #pragma omp parallel for schedule(dynamic, 1) if(!omp_in_parallel())
    for (int b = 0; b < nblocks; b++) {
        const int s = b * block;
        const int len = std::min(block, M - s);
        Eigen::VectorXd v;
        Eigen::MatrixXd g;
        predict_with_gradients_block(x_new.middleRows(s, len), v, g);
        values.segment(s, len) = v;
        grads.middleRows(s, len) = g;
    }
}

void DuchonInterpolator::predict_with_gradients_block(const Eigen::MatrixXd& x_new,
                                                      Eigen::VectorXd& value_out,
                                                      Eigen::MatrixXd& grad_out) const {
    const int M = (int)x_new.rows();
    const int N = (int)points_.rows();

    // Shared squared-distance matrix — the dominant M×N GEMM, built once here
    // instead of once per predict()/predict_gradients() call. Peak M×N
    // allocation is 2 (dist2 + r); dist2 is recycled to hold the value kernel
    // and then the gradient weight.
    Eigen::VectorXd x_sq = x_new.rowwise().squaredNorm();
    Eigen::VectorXd p_sq = points_.rowwise().squaredNorm();
    Eigen::MatrixXd dist2 = x_sq.replicate(1, N) + p_sq.transpose().replicate(M, 1)
                            - 2.0 * x_new * points_.transpose();
    dist2 = dist2.cwiseMax(0.0);

    // r holds sqrt(dist2) (clamped), then log(r) for thin_plate. Shared between
    // the value kernel and the gradient weight so the sqrt/log runs once.
    Eigen::MatrixXd r = dist2.array().sqrt().cwiseMax(1e-10).matrix();
    if (kernel_type_ == "thin_plate")
        r = r.array().log().matrix();

    // Value: dist2 becomes the kernel K (r³ for cubic, dist2·log r for
    // thin_plate), then value_out = K·alpha + x·p + q.
    dist2.array() *= r.array();
    value_out = dist2 * alpha_ + x_new * p_ + Eigen::VectorXd::Constant(M, q_);

    // Gradient: recycle dist2 as the weight W (d kernel/dr, as stored in
    // predict_gradients_block), then apply the same rowsum trick.
    if (kernel_type_ == "thin_plate")
        dist2 = (2.0 * r.array() + 1.0).matrix();
    else
        dist2 = (3.0 * r.array()).matrix();
    dist2 = dist2 * alpha_.asDiagonal();

    Eigen::VectorXd W_rowsum = dist2.rowwise().sum();
    Eigen::MatrixXd scaled_x = (x_new.array().colwise() * W_rowsum.array()).matrix();
    grad_out = scaled_x - dist2 * points_;
    grad_out.rowwise() += p_.transpose();
    add_hermite_terms(x_new, &value_out, &grad_out);
}

}  // namespace sdf
