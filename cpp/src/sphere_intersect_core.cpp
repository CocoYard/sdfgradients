// Pure-C++ sphere intersection using a median-split BVH.
// Deliberately does not include types.h / Eigen — keeps this TU free of
// Eigen template machinery.
//
//   namespace sphere_intersect_core::find_intersections(...)

#include <vector>
#include <cmath>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace sphere_intersect_core {

struct BVHNode {
    float lo[3], hi[3];
    int left, right;
    int leaf_start, leaf_count;
};

static void bvh_build(
    std::vector<BVHNode>& nodes,
    std::vector<int>& leaf_indices,
    const float* cx, const float* cy, const float* cz,
    const float* s_lo_x, const float* s_lo_y, const float* s_lo_z,
    const float* s_hi_x, const float* s_hi_y, const float* s_hi_z,
    int* idx_buf, int start, int count, int leaf_size)
{
    int node_id = (int)nodes.size();
    nodes.push_back(BVHNode());

    float lo0 = 1e30f, lo1 = 1e30f, lo2 = 1e30f;
    float hi0 = -1e30f, hi1 = -1e30f, hi2 = -1e30f;
    for (int k = start; k < start + count; k++) {
        int i = idx_buf[k];
        lo0 = std::min(lo0, s_lo_x[i]); hi0 = std::max(hi0, s_hi_x[i]);
        lo1 = std::min(lo1, s_lo_y[i]); hi1 = std::max(hi1, s_hi_y[i]);
        lo2 = std::min(lo2, s_lo_z[i]); hi2 = std::max(hi2, s_hi_z[i]);
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
    bvh_build(nodes, leaf_indices, cx, cy, cz,
              s_lo_x, s_lo_y, s_lo_z, s_hi_x, s_hi_y, s_hi_z,
              idx_buf, start, mid, leaf_size);

    int right_id = (int)nodes.size();
    bvh_build(nodes, leaf_indices, cx, cy, cz,
              s_lo_x, s_lo_y, s_lo_z, s_hi_x, s_hi_y, s_hi_z,
              idx_buf, start + mid, count - mid, leaf_size);

    nodes[node_id].left = left_id;
    nodes[node_id].right = right_id;
}

static void bvh_query(
    const std::vector<BVHNode>& nodes,
    const std::vector<int>& leaf_indices,
    const float* cx, const float* cy, const float* cz, const float* ra,
    int qi, float qx, float qy, float qz, float qr,
    float q_lo0, float q_lo1, float q_lo2,
    float q_hi0, float q_hi1, float q_hi2,
    std::vector<int>& out)
{
    int stack[64];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        int nid = stack[--sp];
        const BVHNode& nd = nodes[nid];

        if (q_lo0 > nd.hi[0] || q_hi0 < nd.lo[0] ||
            q_lo1 > nd.hi[1] || q_hi1 < nd.lo[1] ||
            q_lo2 > nd.hi[2] || q_hi2 < nd.lo[2])
            continue;

        if (nd.left == -1) {
            for (int k = nd.leaf_start; k < nd.leaf_start + nd.leaf_count; k++) {
                int j = leaf_indices[k];
                if (j == qi) continue;
                float dx = qx - cx[j];
                float dy = qy - cy[j];
                float dz = qz - cz[j];
                float sr = qr + ra[j];
                if (dx * dx + dy * dy + dz * dz < sr * sr)
                    out.push_back(j);
            }
        } else {
            stack[sp++] = nd.left;
            stack[sp++] = nd.right;
        }
    }
}

void find_intersections(const double* centers, const double* radii, int n,
                        std::vector<int>& offsets, std::vector<int>& neighbors) {
    if (n == 0) {
        offsets = {0};
        neighbors.clear();
        return;
    }

    std::vector<float> cx(n), cy(n), cz(n), ra(n);
    std::vector<float> s_lo_x(n), s_lo_y(n), s_lo_z(n);
    std::vector<float> s_hi_x(n), s_hi_y(n), s_hi_z(n);
    for (int i = 0; i < n; i++) {
        cx[i] = (float)centers[i*3];
        cy[i] = (float)centers[i*3+1];
        cz[i] = (float)centers[i*3+2];
        ra[i] = std::max((float)radii[i], 0.0f);
        s_lo_x[i] = cx[i] - ra[i]; s_hi_x[i] = cx[i] + ra[i];
        s_lo_y[i] = cy[i] - ra[i]; s_hi_y[i] = cy[i] + ra[i];
        s_lo_z[i] = cz[i] - ra[i]; s_hi_z[i] = cz[i] + ra[i];
    }

    const int leaf_size = 16;
    std::vector<BVHNode> nodes;
    std::vector<int> leaf_indices;
    nodes.reserve(2 * n / leaf_size + 16);
    leaf_indices.reserve(n);

    std::vector<int> idx_buf(n);
    for (int i = 0; i < n; i++) idx_buf[i] = i;

    bvh_build(nodes, leaf_indices,
              cx.data(), cy.data(), cz.data(),
              s_lo_x.data(), s_lo_y.data(), s_lo_z.data(),
              s_hi_x.data(), s_hi_y.data(), s_hi_z.data(),
              idx_buf.data(), 0, n, leaf_size);

    std::vector<std::vector<int>> result(n);

    #pragma omp parallel
    {
        std::vector<int> buf;
        #pragma omp for schedule(dynamic, 256)
        for (int i = 0; i < n; i++) {
            buf.clear();
            bvh_query(nodes, leaf_indices,
                      cx.data(), cy.data(), cz.data(), ra.data(),
                      i, cx[i], cy[i], cz[i], ra[i],
                      s_lo_x[i], s_lo_y[i], s_lo_z[i],
                      s_hi_x[i], s_hi_y[i], s_hi_z[i],
                      buf);
            std::sort(buf.begin(), buf.end());
            buf.erase(std::unique(buf.begin(), buf.end()), buf.end());
            result[i] = std::move(buf);
        }
    }

    offsets.resize(n + 1, 0);
    for (int i = 0; i < n; i++) offsets[i+1] = offsets[i] + (int)result[i].size();
    neighbors.resize(offsets[n]);
    // Downstream compute_exposed_batch only scans the first MAX_NEIGHBORS_SCANNED
    // (=2048) neighbors per sphere, so we only need the top-K by descending
    // |radius| in sorted order — the tail can be left unordered. nth_element
    // + sort of the head cuts this stage from ~10s (full sort of 1.3B items)
    // to ~2s on bunny/grid=50.
    constexpr int TOP_K = 512;
    std::vector<float> absr(n);
    for (int i = 0; i < n; i++) absr[i] = std::abs((float)radii[i]);
    const float* absr_p = absr.data();
    #pragma omp parallel for schedule(dynamic, 256)
    for (int i = 0; i < n; i++) {
        auto& row = result[i];
        int sz = (int)row.size();
        int k = sz < TOP_K ? sz : TOP_K;
        auto cmp = [absr_p](int a, int b) { return absr_p[a] > absr_p[b]; };
        if (k < sz) {
            std::nth_element(row.begin(), row.begin() + k, row.end(), cmp);
        }
        std::sort(row.begin(), row.begin() + k, cmp);
        int base = offsets[i];
        for (int j = 0; j < sz; j++) neighbors[base + j] = row[j];
    }
}

}  // namespace sphere_intersect_core
