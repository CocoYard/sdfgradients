#include "interpolator.h"
#include <cmath>
#include <limits>

namespace sdf {

// ── Fibonacci sphere (uniform directions on S2) ────────────────────

static Eigen::MatrixXd fibonacci_sphere(int n) {
    Eigen::MatrixXd dirs(n, 3);
    const double golden = (1.0 + std::sqrt(5.0)) / 2.0;
    for (int i = 0; i < n; i++) {
        double theta = 2.0 * M_PI * i / golden;
        double phi = std::acos(1.0 - 2.0 * (i + 0.5) / n);
        dirs(i, 0) = std::sin(phi) * std::cos(theta);
        dirs(i, 1) = std::sin(phi) * std::sin(theta);
        dirs(i, 2) = std::cos(phi);
    }
    return dirs;
}

// ── Tangent frame for a set of directions ───────────────────────────

static void tangent_frame(const Eigen::MatrixXd& d,
                          Eigen::MatrixXd& t1, Eigen::MatrixXd& t2) {
    int N = (int)d.rows();
    t1.resize(N, 3);
    t2.resize(N, 3);
    for (int i = 0; i < N; i++) {
        Eigen::Vector3d di = d.row(i);
        Eigen::Vector3d ref = (std::abs(di(0)) < 0.9)
                              ? Eigen::Vector3d(1, 0, 0)
                              : Eigen::Vector3d(0, 1, 0);
        Eigen::Vector3d u = di.cross(ref);
        u.normalize();
        Eigen::Vector3d v = di.cross(u);
        t1.row(i) = u;
        t2.row(i) = v;
    }
}

// ── Cone directions around best_dirs ─────────────────────────────────

static Eigen::MatrixXd cone_dirs_for_point(const Eigen::Vector3d& best_dir,
                                            const Eigen::Vector3d& u,
                                            const Eigen::Vector3d& v,
                                            double half_angle, int n) {
    const double golden = (1.0 + std::sqrt(5.0)) / 2.0;
    Eigen::MatrixXd dirs(n, 3);
    for (int i = 0; i < n; i++) {
        double r = half_angle * std::sqrt((i + 0.5) / n);
        double alpha = 2.0 * M_PI * i / golden;
        double cos_r = std::cos(r), sin_r = std::sin(r);
        double cos_a = std::cos(alpha), sin_a = std::sin(alpha);
        dirs.row(i) = cos_r * best_dir + sin_r * (cos_a * u + sin_a * v);
    }
    return dirs;
}

// ── sample_best_gradients ────────────────────────────────────────────

Eigen::MatrixXd Interpolator::sample_best_gradients(
    const Eigen::MatrixXd& points,
    const Eigen::VectorXd& sdf_values,
    int num_coarse,
    int refine_steps,
    int num_refine,
    const Eigen::MatrixXd* initial_guess,
    int chunk_size) const
{
    int batch_size = (int)points.rows();
    Eigen::VectorXd sign(batch_size);
    for (int i = 0; i < batch_size; i++)
        sign(i) = (sdf_values(i) > 0) ? 1.0 : -1.0;

    // Determine which points have valid initial guesses
    std::vector<bool> valid_mask(batch_size, false);
    if (initial_guess) {
        for (int i = 0; i < batch_size; i++)
            valid_mask[i] = !initial_guess->row(i).array().isNaN().any();
    }

    Eigen::MatrixXd best_dirs = Eigen::MatrixXd::Zero(batch_size, 3);

    // Copy valid initial guesses
    if (initial_guess) {
        for (int i = 0; i < batch_size; i++)
            if (valid_mask[i])
                best_dirs.row(i) = initial_guess->row(i);
    }

    // Fibonacci sweep for invalid (or all if no initial_guess)
    std::vector<int> invalid_idx;
    for (int i = 0; i < batch_size; i++)
        if (!valid_mask[i]) invalid_idx.push_back(i);

    if (!invalid_idx.empty()) {
        int n_inv = (int)invalid_idx.size();
        Eigen::MatrixXd fib = fibonacci_sphere(num_coarse);  // (num_coarse, 3)

        // Build samples: for each invalid point, evaluate all fibonacci directions
        Eigen::MatrixXd samples(n_inv * num_coarse, 3);
        for (int ii = 0; ii < n_inv; ii++) {
            int i = invalid_idx[ii];
            for (int d = 0; d < num_coarse; d++) {
                samples.row(ii * num_coarse + d) =
                    points.row(i) - sdf_values(i) * fib.row(d);
            }
        }
        Eigen::VectorXd preds = predict(samples, chunk_size);

        // Find best direction per invalid point
        for (int ii = 0; ii < n_inv; ii++) {
            int i = invalid_idx[ii];
            double best_obj = std::numeric_limits<double>::max();
            int best_idx = 0;
            for (int d = 0; d < num_coarse; d++) {
                double obj = preds(ii * num_coarse + d) * sign(i);
                if (obj < best_obj) { best_obj = obj; best_idx = d; }
            }
            best_dirs.row(i) = fib.row(best_idx);
        }
    }

    // Iterative cone refinement
    double half_angle = M_PI / std::sqrt((double)num_coarse);

    Eigen::MatrixXd t1, t2;

    for (int step = 0; step < refine_steps; step++) {
        tangent_frame(best_dirs, t1, t2);

        // Build samples: for each point, evaluate num_refine directions in cone
        Eigen::MatrixXd samples(batch_size * num_refine, 3);
        // Store per-point cone directions to look up the best
        Eigen::MatrixXd all_cone_dirs(batch_size * num_refine, 3);

        for (int i = 0; i < batch_size; i++) {
            Eigen::MatrixXd cdirs = cone_dirs_for_point(
                best_dirs.row(i), t1.row(i), t2.row(i), half_angle, num_refine);
            for (int d = 0; d < num_refine; d++) {
                all_cone_dirs.row(i * num_refine + d) = cdirs.row(d);
                samples.row(i * num_refine + d) =
                    points.row(i) - sdf_values(i) * cdirs.row(d);
            }
        }

        Eigen::VectorXd preds = predict(samples, chunk_size);

        // Pick best per point
        for (int i = 0; i < batch_size; i++) {
            double best_obj = std::numeric_limits<double>::max();
            int best_local = 0;
            for (int d = 0; d < num_refine; d++) {
                double obj = preds(i * num_refine + d) * sign(i);
                if (obj < best_obj) { best_obj = obj; best_local = d; }
            }
            best_dirs.row(i) = all_cone_dirs.row(i * num_refine + best_local);
            double norm = best_dirs.row(i).norm();
            if (norm > 1e-15) best_dirs.row(i) /= norm;
        }

        half_angle /= std::sqrt((double)num_refine);
    }

    return best_dirs;
}

}  // namespace sdf
