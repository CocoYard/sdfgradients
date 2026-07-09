#include "get_visible_arcs.h"
#include <iostream>
#include <fstream>
#include <chrono>

namespace sphere_intersect_core {
    // Find all sphere-sphere intersections by power diagram. Fills per-sphere adjacency lists.
    void find_intersections_by_power_diagram(const double* centers, const double* radii, int n,
                            std::vector<std::vector<int>>& out_neighbors,
                            int* out_hidden = nullptr);
    // Find all sphere-sphere intersections. Fills per-sphere adjacency lists.
    void find_intersections(const double* centers, const double* radii, int n,
                            std::vector<std::vector<int>>& out_neighbors);
}

namespace sphere_exposed_core {
    // Compute exposed arcs/caps/degenerate points for all spheres.
    void compute_exposed_batch(
        const double* centers, const double* radii, int n,
        const std::vector<std::vector<int>>& nbrs,
        double interval_eps, double degen_tol, double merge_tol, double tangent_tol,
        sdf::Options::BatchData& out);
}

namespace sdf {
using clk = std::chrono::steady_clock;

auto ms_since = [](const clk::time_point& t) {
    return std::chrono::duration<double, std::milli>(clk::now() - t).count();
};

// ── get_visible_arcs ────────────────────────────────────────────────

void get_visible_arcs(
    const Eigen::MatrixXd& sdf_points,
    const Eigen::VectorXd& sdf_values,
    Options& options)
{
    // Build persistent BVH over the SDF spheres. Visibility + clamp query
    // this instead of scanning ngbrs_list[i] — exact, tiny memory, and lets
    // us cap ngbrs_list at MAX_KEEP without losing occluder accuracy.
    {
        auto t = clk::now();
        options.sphere_bvh = std::make_unique<SphereBVH>(sdf_points, sdf_values);
        if (options.verbose)
            std::cout << "[get_visible_arcs] build SphereBVH: "
                      << ms_since(t)/1000.0 << " s\n";
    }

    if (options.turn_off_short_arcs && !options.clamp) {
        // no need to compute arcs if we're not using them for clamping or gradient init
        // But the visibility check is still needed for the final output, so we keep the BVH build and just skip the arc/degenerate pt extraction
        options.ngbrs_list = std::vector<std::vector<int>>(sdf_points.rows());  // empty neighbor lists
        return;
    }
    int N = (int)sdf_points.rows();
    Eigen::VectorXd radii = sdf_values.cwiseAbs();


    // Scope block: row-major copy + flat CSR arrays are only needed for the two C
    // function calls.  Wrapping them here releases ~(24N + M·4) bytes before the
    // heavier degenerate-pts extraction below.
    {
        // Eigen default is column-major; C functions expect row-major (centers[i*3+k])
        Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> pts_rm = sdf_points;
        auto t = clk::now();
        sphere_intersect_core::find_intersections_by_power_diagram(
            pts_rm.data(), radii.data(), N, options.ngbrs_list,
            &options.hidden_points);
        if (false) {
            std::ofstream hf("logs/degen_stats.txt", std::ios::app);
            hf << options.grid_len << " " << options.hidden_points << "\n";
        }
        // sphere_intersect_core::find_intersections(
        //     pts_rm.data(), radii.data(), N, options.ngbrs_list);
        // print the distribution of neighbor counts
        std::cout << "Neighbor count distribution (capped at 20):\n";
        for (size_t i = 0; i < 20; i++) {
            std::cout << "  " << i << ": " << options.ngbrs_list[i].size() << "\n";
        }
        if (options.verbose)
            std::cout << "[get_visible_arcs] find_intersections: " << ms_since(t)/1000.0 << " s\n";

        // tol args (in order): interval_eps, degen_tol, merge_tol, tangent_tol.
        //   interval_eps = 1e-4  : skip_tol in intersect_intervals — angular
        //                          slack on interval bounds (radians)
        //   degen_tol    = 1e-7  : collapse exposed region to a tangent point
        //                          when total arc length < this. Length (mesh).
        //   merge_tol    = 1e-12 : merge near-touching intervals (radians)
        //   tangent_tol  = 1e-8  : internal-containment surface-gap tolerance
        //                          for emitting a tangent degen point (length)
        // The cap-dedup / parallel-cut tolerances are file-scope constants in
        // existing_modules_adapter.cpp (DEDUP_COS, DEDUP_LEN).
        auto t_arc = clk::now();
        sphere_exposed_core::compute_exposed_batch(
            pts_rm.data(), radii.data(), N,
            options.ngbrs_list,
            1e-4, 1e-5, 1e-12, 1e-8,
            options.batch);
        if (options.verbose)
            std::cout << "[get_visible_arcs] compute_exposed_batch: "
                      << ms_since(t_arc)/1000.0 << " s\n";
    }  // pts_rm freed here

    // Count fully covered spheres
    int fully_covered = 0;
    for (int i = 0; i < N; i++) {
        bool no_arcs = (options.batch.n_arcs[i] == 0);
        bool no_pts = (options.batch.n_points[i] == 0);
        bool has_ngbrs = !options.ngbrs_list[i].empty();
        if (no_arcs && no_pts && has_ngbrs) fully_covered++;
    }
    options.fully_covered = fully_covered;
    if (options.verbose)
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

}  // namespace sdf
