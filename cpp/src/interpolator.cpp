#include "interpolator.h"
#include <cassert>  // libigl's march_cube.cpp uses assert() without including <cassert>
#include <igl/marching_cubes.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
#include <chrono>
#include <iostream>

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

// ── extract_surface (narrow-band marching cubes via libigl) ──────────
//
// Two-pass strategy:
//   1. Coarse grid (~1/4 res per axis): one predict() call, builds a cheap
//      SDF approximation over the whole bbox.
//   2. Mark coarse cells that either straddle the iso level or sit within
//      ~2× coarse-cell-diagonal of it (Lipschitz safety margin). These are
//      the only cells that can produce triangles.
//   3. Fill the fine S array with trilinear interpolation from the coarse
//      values everywhere, then overwrite fine vertices inside active coarse
//      cells with a real predict() call. Inactive regions keep smooth,
//      same-sign interpolated values so libigl MC produces no spurious
//      crossings there.
//
// Net effect: fine predict() only runs on the ~O(n²) vertices that actually
// matter, cutting cost dramatically vs. a dense O(n³) predict.

void Interpolator::extract_surface(
    const Eigen::Vector3d& bbox_min,
    const Eigen::Vector3d& bbox_max,
    int nx, int ny, int nz,
    double iso,
    Eigen::MatrixXd& V,
    Eigen::MatrixXi& F,
    int chunk_size) const
{
    if (nx < 2 || ny < 2 || nz < 2)
        throw std::invalid_argument("extract_surface: nx, ny, nz must each be >= 2");

    using clk = std::chrono::steady_clock;
    auto t_total = clk::now();
    auto sec_since = [](const clk::time_point& t) {
        return std::chrono::duration<double>(clk::now() - t).count();
    };

    const int N = nx * ny * nz;
    const Eigen::Vector3d step(
        (bbox_max.x() - bbox_min.x()) / (nx - 1),
        (bbox_max.y() - bbox_min.y()) / (ny - 1),
        (bbox_max.z() - bbox_min.z()) / (nz - 1));

    auto fine_idx = [&](int xi, int yi, int zi) {
        return xi + nx * (yi + ny * zi);
    };

    // libigl marching_cubes expects GV.row(x + nx*(y + ny*z))
    auto t_gv = clk::now();
    Eigen::MatrixXd GV(N, 3);
    for (int zi = 0; zi < nz; zi++)
        for (int yi = 0; yi < ny; yi++)
            for (int xi = 0; xi < nx; xi++) {
                int idx = fine_idx(xi, yi, zi);
                GV(idx, 0) = bbox_min.x() + xi * step.x();
                GV(idx, 1) = bbox_min.y() + yi * step.y();
                GV(idx, 2) = bbox_min.z() + zi * step.z();
            }

    std::cout << "[extract_surface] build GV: " << sec_since(t_gv) << " s\n";

    // ── Coarse pass ──────────────────────────────────────────────────
    auto t_coarse = clk::now();
    const int R = 4;
    const int cnx = std::max(2, (nx + R - 1) / R);
    const int cny = std::max(2, (ny + R - 1) / R);
    const int cnz = std::max(2, (nz + R - 1) / R);
    const int CN = cnx * cny * cnz;
    const Eigen::Vector3d cstep(
        (bbox_max.x() - bbox_min.x()) / (cnx - 1),
        (bbox_max.y() - bbox_min.y()) / (cny - 1),
        (bbox_max.z() - bbox_min.z()) / (cnz - 1));

    auto cidx_at = [&](int ci, int cj, int ck) {
        return ci + cnx * (cj + cny * ck);
    };

    Eigen::MatrixXd CGV(CN, 3);
    for (int ck = 0; ck < cnz; ck++)
        for (int cj = 0; cj < cny; cj++)
            for (int ci = 0; ci < cnx; ci++) {
                int idx = cidx_at(ci, cj, ck);
                CGV(idx, 0) = bbox_min.x() + ci * cstep.x();
                CGV(idx, 1) = bbox_min.y() + cj * cstep.y();
                CGV(idx, 2) = bbox_min.z() + ck * cstep.z();
            }
    Eigen::VectorXd CS = this->predict(CGV, chunk_size);
    std::cout << "[extract_surface] coarse predict (" << CN
              << " pts): " << sec_since(t_coarse) << " s\n";

    // ── Fill fine S by trilinear interpolation from coarse ───────────
    auto t_tri = clk::now();
    Eigen::VectorXd S(N);
    for (int zi = 0; zi < nz; zi++) {
        double fz = (zi * step.z()) / cstep.z();
        int ck = std::min((int)fz, cnz - 2);
        double tz = fz - ck;
        for (int yi = 0; yi < ny; yi++) {
            double fy = (yi * step.y()) / cstep.y();
            int cj = std::min((int)fy, cny - 2);
            double ty = fy - cj;
            for (int xi = 0; xi < nx; xi++) {
                double fx = (xi * step.x()) / cstep.x();
                int ci = std::min((int)fx, cnx - 2);
                double tx = fx - ci;
                double c000 = CS(cidx_at(ci  , cj  , ck  ));
                double c100 = CS(cidx_at(ci+1, cj  , ck  ));
                double c010 = CS(cidx_at(ci  , cj+1, ck  ));
                double c110 = CS(cidx_at(ci+1, cj+1, ck  ));
                double c001 = CS(cidx_at(ci  , cj  , ck+1));
                double c101 = CS(cidx_at(ci+1, cj  , ck+1));
                double c011 = CS(cidx_at(ci  , cj+1, ck+1));
                double c111 = CS(cidx_at(ci+1, cj+1, ck+1));
                double c00 = c000*(1-tx) + c100*tx;
                double c10 = c010*(1-tx) + c110*tx;
                double c01 = c001*(1-tx) + c101*tx;
                double c11 = c011*(1-tx) + c111*tx;
                double c0  = c00 *(1-ty) + c10 *ty;
                double c1  = c01 *(1-ty) + c11 *ty;
                S(fine_idx(xi, yi, zi)) = c0*(1-tz) + c1*tz;
            }
        }
    }

    std::cout << "[extract_surface] trilinear fill S: " << sec_since(t_tri) << " s\n";

    // ── Narrow-band detection ────────────────────────────────────────
    auto t_nb = clk::now();
    // Active = sign change across cell OR any corner within `tau` of iso.
    // tau = 2 × coarse cell diagonal: a Lipschitz-1 safety margin wide
    // enough to catch features that slightly undershoot the coarse grid.
    const double tau = 2.0 * cstep.norm();

    std::vector<uint8_t> need_refine(N, 0);
    for (int ck = 0; ck < cnz - 1; ck++)
    for (int cj = 0; cj < cny - 1; cj++)
    for (int ci = 0; ci < cnx - 1; ci++) {
        double vmin =  std::numeric_limits<double>::infinity();
        double vmax = -std::numeric_limits<double>::infinity();
        double amin =  std::numeric_limits<double>::infinity();
        for (int dc = 0; dc < 8; dc++) {
            int ci2 = ci + (dc & 1);
            int cj2 = cj + ((dc >> 1) & 1);
            int ck2 = ck + ((dc >> 2) & 1);
            double v = CS(cidx_at(ci2, cj2, ck2));
            vmin = std::min(vmin, v);
            vmax = std::max(vmax, v);
            amin = std::min(amin, std::abs(v - iso));
        }
        bool has_cross = (vmin - iso) * (vmax - iso) <= 0.0;
        bool near_iso  = amin < tau;
        if (!has_cross && !near_iso) continue;

        // Fine-vertex range covering this coarse cell (inclusive, so a shared
        // boundary vertex is marked active by whichever neighbor is active).
        double xs =  ci      * cstep.x() / step.x();
        double xe = (ci + 1) * cstep.x() / step.x();
        double ys =  cj      * cstep.y() / step.y();
        double ye = (cj + 1) * cstep.y() / step.y();
        double zs =  ck      * cstep.z() / step.z();
        double ze = (ck + 1) * cstep.z() / step.z();
        int xi0 = std::max(0,      (int)std::floor(xs - 1e-9));
        int xi1 = std::min(nx - 1, (int)std::ceil (xe + 1e-9));
        int yi0 = std::max(0,      (int)std::floor(ys - 1e-9));
        int yi1 = std::min(ny - 1, (int)std::ceil (ye + 1e-9));
        int zi0 = std::max(0,      (int)std::floor(zs - 1e-9));
        int zi1 = std::min(nz - 1, (int)std::ceil (ze + 1e-9));
        for (int zi = zi0; zi <= zi1; zi++)
        for (int yi = yi0; yi <= yi1; yi++)
        for (int xi = xi0; xi <= xi1; xi++)
            need_refine[fine_idx(xi, yi, zi)] = 1;
    }

    std::cout << "[extract_surface] narrow-band mark: " << sec_since(t_nb) << " s\n";

    // ── Fine predict on narrow band ──────────────────────────────────
    auto t_fine = clk::now();
    std::vector<int> refine_idx;
    refine_idx.reserve(N / 4);
    for (int i = 0; i < N; i++)
        if (need_refine[i]) refine_idx.push_back(i);

    if (!refine_idx.empty()) {
        const int M = (int)refine_idx.size();
        Eigen::MatrixXd RGV(M, 3);
        for (int k = 0; k < M; k++)
            RGV.row(k) = GV.row(refine_idx[k]);
        Eigen::VectorXd RS = this->predict(RGV, chunk_size);
        for (int k = 0; k < M; k++)
            S(refine_idx[k]) = RS(k);
    }

    std::cout << "[extract_surface] fine predict ("
              << refine_idx.size() << " pts): "
              << sec_since(t_fine) << " s\n";

    // ── Marching cubes ───────────────────────────────────────────────
    auto t_mc = clk::now();
    igl::marching_cubes(S, GV,
                        (unsigned)nx, (unsigned)ny, (unsigned)nz,
                        iso, V, F);
    std::cout << "[extract_surface] marching_cubes: " << sec_since(t_mc) << " s\n";
    std::cout << "[extract_surface] total: " << sec_since(t_total) << " s\n";
}

}  // namespace sdf
