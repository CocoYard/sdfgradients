#include "main_algorithm.h"
#include "duchon_interpolator.h"
#include "pu_interpolator.h"
#include "optimization.h"
#include "visibility.h"
#include "kdtree.h"
#include <iostream>
#include <cmath>
#include <limits>
#include <algorithm>
#include <chrono>

// ── Interface to existing C++ pybind modules ────────────────────────
// These functions mirror the pybind11 wrappers in bench/sphere_intersect.cpp
// and bench/sphere_exposed_pybind.cpp. To link, either:
//   (a) refactor those files to separate core logic from pybind, or
//   (b) compile them into this project with a thin adapter layer.
//
// For now, we declare the C++ core functions we need and provide
// implementations that call into the existing code.

// Forward declarations for existing C++ functions.
// These must be provided at link time (see CMakeLists.txt).
namespace sphere_intersect_core {
    // Find all sphere-sphere intersections. Returns CSR (offsets, neighbors).
    void find_intersections(const double* centers, const double* radii, int n,
                            std::vector<int>& offsets, std::vector<int>& neighbors);
}

namespace sphere_exposed_core {
    // Compute exposed arcs/caps/degenerate points for all spheres.
    void compute_exposed_batch(
        const double* centers, const double* radii, int n,
        const int* nbr_indices, const int* nbr_offsets,
        double tol, double degen_tol, double merge_tol, double tangent_tol,
        sdf::Options::BatchData& out);
}

namespace sdf {

// ── filter_degenerate_pts ───────────────────────────────────────────

static void filter_degenerate_pts(
    std::unordered_map<int, std::vector<Eigen::Vector3d>>& degenerate_pts,
    const Interpolator& interpolator,
    double dist_tol = 0.1)
{
    std::vector<int> to_remove;
    for (auto& [idx, pts] : degenerate_pts) {
        if ((int)pts.size() != 1) {
            to_remove.push_back(idx);
            continue;
        }
        Eigen::MatrixXd pt(1, 3);
        pt.row(0) = pts[0].transpose();
        double pred = interpolator.predict(pt)(0);
        if (std::abs(pred) > dist_tol) {
            std::cout << "Degenerate point " << idx
                      << " is too far from the surface with predicted sdf " << pred
                      << ", removing it.\n";
            to_remove.push_back(idx);
        }
    }
    for (int idx : to_remove) degenerate_pts.erase(idx);
    std::cout << "Filtered out " << to_remove.size()
              << " degenerate points. Remaining: " << degenerate_pts.size() << "\n";
}

// ── init_gradients_by_degenerate_pts ────────────────────────────────

Eigen::MatrixXd init_gradients_by_degenerate_pts(
    const Eigen::MatrixXd& sdf_points,
    const Eigen::VectorXd& sdf_values,
    Interpolator& interpolator,
    Options& options)
{
    int N = (int)sdf_points.rows();
    auto& degenerate_pts = options.degenerate_pts;
    std::cout << "Initializing gradients using " << degenerate_pts.size()
              << " degenerate points...\n";
    // 1. Initial fit without gradients
    interpolator.fit(sdf_points, sdf_values);
    std::cout << "======== first fit done with input " << N << " points\n";

    // Filter degenerate points
    filter_degenerate_pts(degenerate_pts, interpolator);

    // Initialize gradients to NaN
    Eigen::MatrixXd init_grads = Eigen::MatrixXd::Constant(N, 3, std::numeric_limits<double>::quiet_NaN());

    // Add degenerate points as zero-value constraints
    Eigen::MatrixXd to_train_points = sdf_points;
    Eigen::VectorXd to_train_sdf = sdf_values;

    std::vector<Eigen::Vector3d> pts_to_add;
    for (auto& [i, pts] : degenerate_pts) {
        pts_to_add.push_back(pts[0]);
        init_grads.row(i) = (sdf_points.row(i) - pts[0].transpose()) / (sdf_values(i) + 1e-10);
    }

    if (!pts_to_add.empty()) {
        int n_add = (int)pts_to_add.size();
        Eigen::MatrixXd new_pts(N + n_add, 3);
        new_pts.topRows(N) = sdf_points;
        for (int i = 0; i < n_add; i++)
            new_pts.row(N + i) = pts_to_add[i].transpose();

        Eigen::VectorXd new_vals(N + n_add);
        new_vals.head(N) = sdf_values;
        new_vals.tail(n_add).setZero();

        to_train_points = new_pts;
        to_train_sdf = new_vals;
    }

    std::cout << "After adding points for degenerate arcs, total points: "
              << to_train_points.rows() << "\n";
    interpolator.fit(to_train_points, to_train_sdf);
    std::cout << "======== second fit done with input " << to_train_points.rows()
              << " points (including " << pts_to_add.size() << " degenerate arc points)\n";
    std::cout << "initial gradient estimation done\n";

    // Debug check
    for (auto& [i, pts] : degenerate_pts) {
        if ((int)pts.size() != 1) {
            std::cout << "ERRRRRRRRRRR\n";
        }
        init_grads.row(i) = (sdf_points.row(i) - pts[0].transpose()) / (sdf_values(i) + 1e-10);
        Eigen::Vector3d proj = sdf_points.row(i).transpose() - sdf_values(i) * init_grads.row(i).transpose();
        Eigen::MatrixXd proj_mat(1, 3);
        proj_mat.row(0) = proj.transpose();
        double pred_sdf = interpolator.predict(proj_mat)(0);
        if (pred_sdf > 0.6) {
            std::cout << "Warning: For degenerate point " << i
                      << ", the projected point is still outside with predicted sdf "
                      << pred_sdf << ".\n";
        }
    }

    return init_grads;
}

// ── get_visible_arcs ────────────────────────────────────────────────

void get_visible_arcs(
    const Eigen::MatrixXd& sdf_points,
    const Eigen::VectorXd& sdf_values,
    Options& options)
{
    int N = (int)sdf_points.rows();
    Eigen::VectorXd radii = sdf_values.cwiseAbs();

    // Scope block: row-major copy + flat CSR arrays are only needed for the two C
    // function calls.  Wrapping them here releases ~(24N + M·4) bytes before the
    // heavier degenerate-pts extraction below.
    {
        // Eigen default is column-major; C functions expect row-major (centers[i*3+k])
        Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> pts_rm = sdf_points;
        std::vector<int> offsets, neighbors;
        sphere_intersect_core::find_intersections(
            pts_rm.data(), radii.data(), N, offsets, neighbors);

        // Call compute_exposed_batch while neighbors/offsets are still alive.
        // This avoids keeping pts_rm + neighbors alive past their last use.
        sphere_exposed_core::compute_exposed_batch(
            pts_rm.data(), radii.data(), N,
            neighbors.data(), offsets.data(),
            1e-4, 1e-6, 1e-12, 1e-8,
            options.batch);

        // Build per-sphere neighbor lists from the CSR arrays before they are freed.
        // ngbrs_list[i] is the only copy of neighbor data that persists beyond this
        // block; the flat neighbors + offsets vectors are freed at end of scope.
        options.ngbrs_list.resize(N);
        for (int i = 0; i < N; i++) {
            options.ngbrs_list[i].assign(
                neighbors.begin() + offsets[i],
                neighbors.begin() + offsets[i + 1]);
        }
    }  // pts_rm, offsets, neighbors freed here

    // Count fully covered spheres
    int fully_covered = 0;
    for (int i = 0; i < N; i++) {
        bool no_arcs = (options.batch.n_arcs[i] == 0);
        bool no_pts = (options.batch.n_points[i] == 0);
        bool has_ngbrs = !options.ngbrs_list[i].empty();
        if (no_arcs && no_pts && has_ngbrs) fully_covered++;
    }
    std::cout << fully_covered << " fully covered spheres\n";

    // Extract degenerate points
    options.degenerate_pts.clear();
    int n_total_pts = (int)options.batch.point_sphere_idx.size();
    for (int p = 0; p < n_total_pts; p++) {
        int idx = options.batch.point_sphere_idx[p];
        options.degenerate_pts[idx].push_back(
            options.batch.point_positions.row(p).transpose());
    }
}

// ── main_algorithm ──────────────────────────────────────────────────

MainResult main_algorithm(
    const Eigen::MatrixXd& sdf_points,
    const Eigen::VectorXd& sdf_values,
    Options& options)
{
    int N = (int)sdf_points.rows();

    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    auto ms_since = [](const clk::time_point& t) {
        return std::chrono::duration<double, std::milli>(clk::now() - t).count();
    };

    // Step 1: Compute visible arcs + degenerate points
    auto t1 = clk::now();
    get_visible_arcs(sdf_points, sdf_values, options);
    if (options.turn_off_short_arcs)
        options.degenerate_pts.clear();
    std::cout << "[main_algorithm] get_visible_arcs: " << ms_since(t1)/1000.0 << " s\n";

    // Create interpolator
    auto t2 = clk::now();
    std::shared_ptr<Interpolator> interpolator;
    if (options.interpolator_type == "Duchon") {
        interpolator = std::make_shared<DuchonInterpolator>("cubic");
    } else {
        interpolator = std::make_shared<PUInterpolator>(
            "cubic", options.interp_overlap,
            10, 200, options.interp_partition);
    }
    std::cout << "[main_algorithm] interpolator ctor: " << ms_since(t2)/1000.0 << " s\n";

    // Step 1b: Initial gradient estimation using degenerate points
    auto t3 = clk::now();
    Eigen::MatrixXd init_grads = init_gradients_by_degenerate_pts(
        sdf_points, sdf_values, *interpolator, options);
    std::cout << "[main_algorithm] init_gradients_by_degenerate_pts: "
              << ms_since(t3)/1000.0 << " s\n";

    // Step 2: Iterative optimization
    auto t4 = clk::now();
    Eigen::MatrixXd gradients = iterative_projection_3d(
        sdf_points, sdf_values, init_grads,
        *interpolator, options,
        options.max_iters);
    std::cout << "[main_algorithm] iterative_projection_3d: "
              << ms_since(t4)/1000.0 << " s\n";

    // Final projection + visibility check
    auto t5 = clk::now();
    Eigen::MatrixXd projections(N, 3);
    for (int i = 0; i < N; i++)
        projections.row(i) = sdf_points.row(i) - sdf_values(i) * gradients.row(i);

    Eigen::VectorXi vis = are_points_visible(
        projections, sdf_points, sdf_values, options.ngbrs_list);
    std::cout << "[main_algorithm] final projection + visibility: "
              << ms_since(t5)/1000.0 << " s\n";
    std::cout << "[main_algorithm] total: " << ms_since(t0)/1000.0 << " s\n";

    return {projections, vis, interpolator};
}

}  // namespace sdf
