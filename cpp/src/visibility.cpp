#include "visibility.h"
#include <cmath>
#include <algorithm>
#include <vector>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace sdf {

// ── BVH for sphere containment queries ──────────────────────────────

struct BVHNode {
    float lo[3], hi[3];   // AABB of this subtree (spheres expanded by radius)
    int left, right;       // child indices, -1 if leaf
    int leaf_start, leaf_count;
};

static void bvh_build(
    std::vector<BVHNode>& nodes,
    std::vector<int>& leaf_indices,
    const float* cx, const float* cy, const float* cz, const float* ra,
    int* idx_buf, int start, int count, int leaf_size)
{
    int node_id = (int)nodes.size();
    nodes.push_back(BVHNode());

    // Compute AABB (sphere centers ± radii)
    float lo0 = 1e30f, lo1 = 1e30f, lo2 = 1e30f;
    float hi0 = -1e30f, hi1 = -1e30f, hi2 = -1e30f;
    for (int k = start; k < start + count; k++) {
        int i = idx_buf[k];
        lo0 = std::min(lo0, cx[i] - ra[i]); hi0 = std::max(hi0, cx[i] + ra[i]);
        lo1 = std::min(lo1, cy[i] - ra[i]); hi1 = std::max(hi1, cy[i] + ra[i]);
        lo2 = std::min(lo2, cz[i] - ra[i]); hi2 = std::max(hi2, cz[i] + ra[i]);
    }
    nodes[node_id].lo[0] = lo0; nodes[node_id].lo[1] = lo1; nodes[node_id].lo[2] = lo2;
    nodes[node_id].hi[0] = hi0; nodes[node_id].hi[1] = hi1; nodes[node_id].hi[2] = hi2;

    if (count <= leaf_size) {
        nodes[node_id].left = -1;
        nodes[node_id].right = -1;
        nodes[node_id].leaf_start = (int)leaf_indices.size();
        nodes[node_id].leaf_count = count;
        for (int k = start; k < start + count; k++)
            leaf_indices.push_back(idx_buf[k]);
        return;
    }

    // Pick longest axis, median split
    float ext0 = hi0 - lo0, ext1 = hi1 - lo1, ext2 = hi2 - lo2;
    int axis = 0;
    if (ext1 > ext0 && ext1 >= ext2) axis = 1;
    else if (ext2 > ext0 && ext2 > ext1) axis = 2;

    const float* axis_ptr = (axis == 0) ? cx : (axis == 1) ? cy : cz;
    std::sort(idx_buf + start, idx_buf + start + count, [axis_ptr](int a, int b) {
        return axis_ptr[a] < axis_ptr[b];
    });

    int mid = count / 2;
    nodes[node_id].leaf_start = -1;
    nodes[node_id].leaf_count = 0;

    int left_id = (int)nodes.size();
    bvh_build(nodes, leaf_indices, cx, cy, cz, ra, idx_buf, start, mid, leaf_size);
    int right_id = (int)nodes.size();
    bvh_build(nodes, leaf_indices, cx, cy, cz, ra, idx_buf, start + mid, count - mid, leaf_size);

    nodes[node_id].left = left_id;
    nodes[node_id].right = right_id;
}

// Query: is point (qx,qy,qz) inside any sphere? Returns true if inside.
static bool bvh_point_inside_any(
    const std::vector<BVHNode>& nodes,
    const std::vector<int>& leaf_indices,
    const float* cx, const float* cy, const float* cz, const float* ra,
    float qx, float qy, float qz, float eps)
{
    int stack[64];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        int nid = stack[--sp];
        const BVHNode& nd = nodes[nid];

        // Point-vs-AABB: is point inside this AABB?
        if (qx < nd.lo[0] || qx > nd.hi[0] ||
            qy < nd.lo[1] || qy > nd.hi[1] ||
            qz < nd.lo[2] || qz > nd.hi[2])
            continue;

        if (nd.left == -1) {
            // Leaf: exact sphere containment check
            for (int k = nd.leaf_start; k < nd.leaf_start + nd.leaf_count; k++) {
                int j = leaf_indices[k];
                float dx = qx - cx[j];
                float dy = qy - cy[j];
                float dz = qz - cz[j];
                float threshold = ra[j] - eps;
                if (dx*dx + dy*dy + dz*dz < threshold * threshold)
                    return true;
            }
        } else {
            stack[sp++] = nd.left;
            stack[sp++] = nd.right;
        }
    }
    return false;
}

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

    if (M == 0) return result;

    // Prepare float arrays for BVH
    std::vector<float> cx(M), cy(M), cz(M), ra(M);
    for (int i = 0; i < M; i++) {
        cx[i] = (float)sdf_points(i, 0);
        cy[i] = (float)sdf_points(i, 1);
        cz[i] = (float)sdf_points(i, 2);
        ra[i] = std::max((float)std::abs(sdf_values(i)), 1e-10f);
    }

    // Build BVH
    std::vector<BVHNode> nodes;
    std::vector<int> leaf_indices;
    nodes.reserve(2 * M);
    leaf_indices.reserve(M);
    std::vector<int> idx_buf(M);
    for (int i = 0; i < M; i++) idx_buf[i] = i;
    bvh_build(nodes, leaf_indices, cx.data(), cy.data(), cz.data(), ra.data(),
              idx_buf.data(), 0, M, 16);

    float eps_f = (float)epsilon;

    #pragma omp parallel for schedule(dynamic, 256)
    for (int i = 0; i < N; i++) {
        if (query_points.row(i).array().isNaN().any()) {
            result(i) = 0;
            continue;
        }
        float qx = (float)query_points(i, 0);
        float qy = (float)query_points(i, 1);
        float qz = (float)query_points(i, 2);
        if (bvh_point_inside_any(nodes, leaf_indices, cx.data(), cy.data(), cz.data(), ra.data(),
                                 qx, qy, qz, eps_f))
            result(i) = 0;
    }
    return result;
}

// Overload using a precomputed candidate list per query point.
Eigen::VectorXi are_points_visible(
    const Eigen::MatrixXd& query_points,
    const Eigen::MatrixXd& sdf_points,
    const Eigen::VectorXd& sdf_values,
    const std::unordered_map<int, std::vector<Eigen::Vector3d>>& degenerate_pts,
    const std::vector<std::vector<int>>& ngbrs_list,
    double epsilon)
{
    int N = (int)query_points.rows();
    Eigen::VectorXi result = Eigen::VectorXi::Ones(N);
    if (N == 0) return result;

    #pragma omp parallel for schedule(dynamic, 256)
    for (int i = 0; i < N; i++) {
        if (degenerate_pts.count(i) > 0) {
            continue;  // skip degenerate points, treat them as visible
        }
        if (query_points.row(i).array().isNaN().any()) {
            result(i) = 0;
            continue;
        }
        const Eigen::RowVector3d q = query_points.row(i);
        const auto& cands = ngbrs_list[i];
        for (int j : cands) {
            double r = std::abs(sdf_values(j)) - epsilon;
            if (r <= 0.0) continue;
            double d2 = (q - sdf_points.row(j)).squaredNorm();
            if (d2 < r * r) { result(i) = 0; break; }
        }
    }
    return result;
}

}  // namespace sdf
