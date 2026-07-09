#include "clamp.h"
#include "sphere_bvh.h"
#include <cmath>
#include <iostream>
#include <chrono>
#include <atomic>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace sdf {

// ── Precompute per-sphere cap row mapping for fast arc queries ────

// For efficiency, precompute a mapping: sphere_idx -> list of cap rows
// so query_closest_on_arcs doesn't need to scan all caps each time.
struct SphereCapMap {
    // cap_rows[sphere_idx] = list of rows in batch.cap_* arrays
    std::unordered_map<int, std::vector<int>> cap_rows;

    void build(const Options::BatchData& batch) {
        cap_rows.clear();
        int n = (int)batch.cap_sphere_idx.size();
        for (int c = 0; c < n; c++)
            cap_rows[batch.cap_sphere_idx[c]].push_back(c);
    }
};

// For efficiency, precompute a mapping: sphere_idx -> list of arc rows
// so query_closest_fast doesn't need to scan all arcs each time.
struct SphereArcMap {
    std::unordered_map<int, std::vector<int>> arc_rows;

    void build(const Options::BatchData& batch) {
        arc_rows.clear();
        int n = (int)batch.arc_sphere_idx.size();
        for (int a = 0; a < n; a++)
            arc_rows[batch.arc_sphere_idx[a]].push_back(a);
    }
};

static void query_closest_fast(
    const Eigen::Vector3d& query_pt,
    const Options::BatchData& batch,
    int sphere_idx,
    const SphereCapMap& cap_map,
    const SphereArcMap& arc_map,
    Eigen::Vector3d& closest_out,
    double& dist_out)
{
    constexpr double TWO_PI = 2.0 * M_PI;
    constexpr double PI = M_PI;

    double best_dist2 = 1e30;
    Eigen::Vector3d best_pt = Eigen::Vector3d::Zero();

    auto it = cap_map.cap_rows.find(sphere_idx);
    if (it == cap_map.cap_rows.end()) {
        closest_out = best_pt;
        dist_out = std::sqrt(best_dist2);
        return;
    }
    const auto& sphere_cap_rows = it->second;

    auto arc_it = arc_map.arc_rows.find(sphere_idx);
    if (arc_it == arc_map.arc_rows.end()) {
        closest_out = best_pt;
        dist_out = std::sqrt(best_dist2);
        return;
    }
    const auto& sphere_arc_rows = arc_it->second;

    for (int a : sphere_arc_rows) {
        int arc_cap = batch.arc_cap_idx[a];
        if (arc_cap < 0 || arc_cap >= (int)sphere_cap_rows.size()) continue;
        int cap_row = sphere_cap_rows[arc_cap];

        double cx = batch.cap_centers(cap_row, 0);
        double cy = batch.cap_centers(cap_row, 1);
        double cz = batch.cap_centers(cap_row, 2);
        double R  = batch.cap_radii(cap_row);
        double ux = batch.cap_u(cap_row, 0), uy = batch.cap_u(cap_row, 1), uz = batch.cap_u(cap_row, 2);
        double vx = batch.cap_v(cap_row, 0), vy = batch.cap_v(cap_row, 1), vz = batch.cap_v(cap_row, 2);

        double dx = query_pt(0) - cx;
        double dy = query_pt(1) - cy;
        double dz = query_pt(2) - cz;
        double proj_u = dx * ux + dy * uy + dz * uz;
        double proj_v = dx * vx + dy * vy + dz * vz;

        double t = std::atan2(proj_v, proj_u);
        if (t < 0) t += TWO_PI;

        double t_s = batch.arc_start[a];
        double t_e = batch.arc_end[a];
        double dt = std::fmod(t - t_s, TWO_PI);
        if (dt < 0) dt += TWO_PI;

        double t_clamped;
        if (dt <= t_e - t_s) {
            t_clamped = t;
        } else {
            double d_start = std::fmod(t_s - t, TWO_PI);
            if (d_start < 0) d_start += TWO_PI;
            if (d_start > PI) d_start = TWO_PI - d_start;
            double d_end = std::fmod(t_e - t, TWO_PI);
            if (d_end < 0) d_end += TWO_PI;
            if (d_end > PI) d_end = TWO_PI - d_end;
            t_clamped = (d_start <= d_end) ? t_s : t_e;
        }

        double cost = std::cos(t_clamped), sint = std::sin(t_clamped);
        double qx = cx + R * (cost * ux + sint * vx);
        double qy = cy + R * (cost * uy + sint * vy);
        double qz = cz + R * (cost * uz + sint * vz);

        double dist2 = (query_pt(0) - qx) * (query_pt(0) - qx)
                      + (query_pt(1) - qy) * (query_pt(1) - qy)
                      + (query_pt(2) - qz) * (query_pt(2) - qz);
        if (dist2 < best_dist2) {
            best_dist2 = dist2;
            best_pt = Eigen::Vector3d(qx, qy, qz);
        }
    }

    closest_out = best_pt;
    dist_out = std::sqrt(best_dist2);
}

// ── clamp_gradients_to_arcs ──────────────────────────────────────────

int clamp_gradients_to_arcs(
    const Eigen::MatrixXd& points,
    const Eigen::VectorXd& values,
    Eigen::MatrixXd& gradients,
    const std::unordered_map<int, std::vector<Eigen::Vector3d>>& degenerate_pts,
    const Options::BatchData& batch,
    const std::vector<std::vector<int>>& ngbrs_list,
    const SphereBVH& bvh,
    const Tolerance& tolerance)
{
    int N = (int)points.rows();
    double clamp_tol = tolerance.clamp_sdf_tol;
    double float_tol = tolerance.float_tol;

    // Precompute per-sphere cap/arc mappings for fast arc queries
    SphereCapMap cap_map;
    cap_map.build(batch);
    SphereArcMap arc_map;
    arc_map.build(batch);

    // Compute all projections
    Eigen::MatrixXd projections(N, 3);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++)
        projections.row(i) = points.row(i) - values(i) * gradients.row(i);

    int debug_cnt = 0;
    int clamped_cnt = 0;
    std::atomic<long long> fast_hits{0};
    std::atomic<long long> bvh_fallbacks{0};
    std::atomic<long long> arc_queries{0};
    auto t_loop0 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for reduction(+:clamped_cnt) schedule(dynamic, 1024)
    for (int i = 0; i < N; i++) {
        // Skip degenerate-arc points
        if (degenerate_pts.count(i)) continue;

        double pix = projections(i, 0), piy = projections(i, 1), piz = projections(i, 2);
        // Fast path: does any of the 2048 curated neighbors contain proj(i)?
        const auto& row = ngbrs_list[i];
        bool inside = !row.empty() &&
            bvh.any_sphere_contains(pix, piy, piz, 0, row.data(), (int)row.size());
        if (inside) {
            fast_hits.fetch_add(1, std::memory_order_relaxed);
        } else {
            // Fall back to full BVH to confirm visibility (miss in fast path
            // is inconclusive — occluder may be outside the curated list).
            bvh_fallbacks.fetch_add(1, std::memory_order_relaxed);
            inside = bvh.point_inside_any(pix, piy, piz, 0, /*exclude_idx=*/i);
        }
        if (!inside) continue;

        // Try clamping to closest arc point
        Eigen::Vector3d closest;
        double distance;
        arc_queries.fetch_add(1, std::memory_order_relaxed);
        query_closest_fast(projections.row(i).transpose(), batch, i, cap_map, arc_map, closest, distance);

        if (distance < 0.001) {
        // if (true) {
            gradients.row(i) = (points.row(i).transpose() - closest).transpose() / (values(i) + 1e-10);
            clamped_cnt++;
            continue;
        }
        // Otherwise keep original gradient
    }
    auto t_loop1 = std::chrono::high_resolution_clock::now();
    std::cout << "  [clamp] N=" << N
              << " clamp_tol=" << clamp_tol
              << " fast_hits=" << fast_hits.load()
              << " bvh_fallbacks=" << bvh_fallbacks.load()
              << " arc_queries=" << arc_queries.load()
              << " wall=" << std::chrono::duration<double>(t_loop1 - t_loop0).count() << "s\n";

    if (debug_cnt > 0)
        std::cout << "\n there are " << debug_cnt << " samples without any arcs\n";
    std::cout << " there are " << clamped_cnt << " samples clamped to arcs\n";
    return clamped_cnt;
}

}  // namespace sdf
