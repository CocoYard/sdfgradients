#include "visibility.h"
#include "sphere_bvh.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <atomic>
#include <chrono>
#include <iostream>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace sdf {

// ── Public API ──────────────────────────────────────────────────────

Eigen::VectorXi are_points_visible(
    const Eigen::MatrixXd& query_points,
    const Eigen::MatrixXd& sdf_points,
    const Eigen::VectorXd& sdf_values,
    double epsilon)
{
    int N = (int)query_points.rows();
    int M = (int)sdf_points.rows();
    Eigen::VectorXi result = Eigen::VectorXi::Ones(N);
    if (M == 0 || N == 0) return result;

    SphereBVH bvh(sdf_points, sdf_values);

    #pragma omp parallel for schedule(dynamic, 256)
    for (int i = 0; i < N; i++) {
        if (query_points.row(i).array().isNaN().any()) {
            result(i) = 0;
            continue;
        }
        if (bvh.point_inside_any(query_points(i, 0), query_points(i, 1),
                                 query_points(i, 2), epsilon))
            result(i) = 0;
    }
    return result;
}

Eigen::VectorXi are_points_visible(
    const Eigen::MatrixXd& query_points,
    const Eigen::VectorXd& /*sdf_values*/,
    const std::unordered_map<int, std::vector<Eigen::Vector3d>>& degenerate_pts,
    const std::vector<std::vector<int>>& ngbrs_list,
    const SphereBVH& bvh,
    double epsilon)
{
    int N = (int)query_points.rows();
    Eigen::VectorXi result = Eigen::VectorXi::Ones(N);
    if (N == 0) return result;

    auto t0 = std::chrono::high_resolution_clock::now();
    std::atomic<long long> fast_hits{0};
    std::atomic<long long> bvh_fallbacks{0};

    #pragma omp parallel for schedule(dynamic, 256)
    for (int i = 0; i < N; i++) {
        if (degenerate_pts.count(i) > 0) continue;
        if (query_points.row(i).array().isNaN().any()) {
            result(i) = 0;
            continue;
        }
        double qx = query_points(i, 0), qy = query_points(i, 1), qz = query_points(i, 2);
        const auto& row = ngbrs_list[i];
        bool occluded = !row.empty() &&
            bvh.any_sphere_contains(qx, qy, qz, epsilon, row.data(), (int)row.size());
        if (occluded) {
            fast_hits.fetch_add(1, std::memory_order_relaxed);
        } else {
            // Fast path missed — the true occluder (if any) is outside the
            // curated 2048; confirm against the full BVH. Exclude sphere i
            // since q = proj(i) sits on its surface.
            bvh_fallbacks.fetch_add(1, std::memory_order_relaxed);
            occluded = bvh.point_inside_any(qx, qy, qz, epsilon, /*exclude_idx=*/i);
        }
        if (occluded) result(i) = 0;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cerr << "  [are_points_visible] N=" << N
              << " fast_hits=" << fast_hits.load()
              << " bvh_fallbacks=" << bvh_fallbacks.load()
              << " total=" << std::chrono::duration<double>(t1 - t0).count() << "s\n";
    return result;
}

}  // namespace sdf
