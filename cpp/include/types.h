#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <cmath>
#include <optional>

#include "sphere_bvh.h"

namespace sdf {

// ── Tolerance parameters ────────────────────────────────────────────

struct Tolerance {
    double clamp_radius_ratio = 0.2;   // for clamping to nearest arc point
    double clamp_sdf_tol      = 1e-2;  // for clamping to optimal visible boundary point
    double float_tol          = 1e-8;
    double angle_tol          = 15.0 * M_PI / 180.0;
};

// ── Algorithm options ───────────────────────────────────────────────

struct Options {
    std::string name   = "default";
    int grid_len       = 20;
    int max_iters      = 10;
    bool clamp         = true;
    bool turn_off_short_arcs = false;
    double reg = 1e-5;  // regularization for DuchonInterpolator
    bool use_MES = false;  // whether to use MES normals for invisible points with small visibility gain
    bool export_projections = true;   // write PLY visualization files after main_algorithm
    bool export_short_arcs  = true;   // write PLY of degenerate-arc points after init
    bool verbose = true;              // if false, main_algorithm suppresses stdout logging

    std::string interpolator_type = "PU";     // "PU" or "Duchon"
    std::string interp_partition  = "box";    // "box" or "sphere"
    double interp_overlap         = 0.5;
    std::string iter_gradient_finding = "optimize";  // "optimize" or "sample"

    Tolerance tolerance;
    bool tolerance_initialized = false;

    // Runtime state (populated by get_visible_arcs, consumed by optimization)
    // Degenerate points: sphere index -> list of surface points
    std::unordered_map<int, std::vector<Eigen::Vector3d>> degenerate_pts;

    // Batch result from compute_exposed_batch (opaque, passed through)
    // Arc data in CSR-like format, mirroring the Python batch dict
    struct BatchData {
        std::vector<int>    n_arcs;           // per-sphere arc count
        std::vector<int>    n_points;         // per-sphere degenerate point count
        // Arc geometry (concatenated across all spheres)
        std::vector<int>    arc_sphere_idx;
        std::vector<int>    arc_cap_idx;
        std::vector<double> arc_start;
        std::vector<double> arc_end;
        // Cap geometry (concatenated across all spheres)
        std::vector<int>    cap_sphere_idx;
        Eigen::MatrixXd     cap_normals;      // (total_caps, 3)
        Eigen::VectorXd     cap_d;
        Eigen::MatrixXd     cap_centers;      // (total_caps, 3)
        Eigen::VectorXd     cap_radii;
        Eigen::MatrixXd     cap_u;            // (total_caps, 3)
        Eigen::MatrixXd     cap_v;            // (total_caps, 3)
        // Degenerate point positions
        std::vector<int>    point_sphere_idx;
        Eigen::MatrixXd     point_positions;  // (total_pts, 3)
    };
    BatchData batch;

    // Neighbor lists: ngbrs_list[i] = list of neighbor indices for sphere i.
    // Truncated at MAX_NEIGHBORS_SCANNED (2048) inside find_intersections —
    // downstream compute_exposed_batch never scans beyond that. For pure
    // "is point inside some sphere?" tests, use sphere_bvh (below) instead;
    // that path is both exact and orders of magnitude smaller in memory.
    std::vector<std::vector<int>> ngbrs_list;

    // Built once in get_visible_arcs; consumed by visibility + clamp. Kept
    // as unique_ptr so Options stays movable without dragging the full
    // SphereBVH definition into this header.
    std::unique_ptr<SphereBVH> sphere_bvh;

    // If set, skip iterative optimization and use these gradients directly
    std::optional<Eigen::MatrixXd> gt_gradients;
};

// ── Result of main_algorithm ────────────────────────────────────────

// Forward declaration (defined in interpolator.h)
class Interpolator;

struct MainResult {
    Eigen::MatrixXd projections;                     // (N, 3) projected surface points
    Eigen::VectorXi visibility_mask;                 // (N,)   1 = visible, 0 = occluded
    std::shared_ptr<Interpolator> interpolator;      // trained interpolator
};

}  // namespace sdf
