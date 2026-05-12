// Pure-C++ sphere intersection using a median-split BVH.
// Deliberately does not include types.h / Eigen — keeps this TU free of
// Eigen template machinery.
//
//   namespace sphere_intersect_core::find_intersections(...)

#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <limits>
#include <cstdint>
#include <random>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "full_compute_switch.h"

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
                        std::vector<std::vector<int>>& out_neighbors) {
    out_neighbors.clear();
    if (n == 0) return;

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

    // Arrangement on the per-thread raw hit buffer `buf` (size = sz), then
    // truncate to MAX_KEEP when copying into result[i]:
    //   [0, NEAREST)                 : NEAREST nearest by center distance, sorted asc
    //   [NEAREST, NEAREST+TOP_LARGEST): TOP_LARGEST largest by |radius| (from the remainder)
    //   [NEAREST+TOP_LARGEST, MAX_KEEP): uniform random samples from the remainder
    // Doing arrangement on the full `buf` before truncation keeps the curated
    // picks exact; doing truncation in the same loop keeps peak RSS at the
    // truncated size (critical for N in the millions).
#ifdef FULL_COMPUTE_30CUBED
    constexpr int MAX_KEEP    = std::numeric_limits<int>::max();
#else
    constexpr int MAX_KEEP    = 1024;
#endif
    constexpr int NEAREST     = 256;
    constexpr int TOP_LARGEST = 64;
    std::vector<float> absr_sq(n);
    for (int i = 0; i < n; i++) absr_sq[i] = (float)(radii[i]*radii[i]);
    const float* absr_sq_p = absr_sq.data();
    const float* cx_p = cx.data(), *cy_p = cy.data(), *cz_p = cz.data();

    #pragma omp parallel
    {
        std::vector<int> buf;
        std::vector<std::pair<float, int>> sd_tmp;  // (signed_d, neighbor_id), reused across i

        #pragma omp for schedule(dynamic, 256)
        for (int i = 0; i < n; i++) {
            std::mt19937 rng((uint32_t)(0x9E3779B1u ^ (uint32_t)i));  // seed by i, not tid: see Fisher-Yates section below
            buf.clear();
            bvh_query(nodes, leaf_indices,
                      cx.data(), cy.data(), cz.data(), ra.data(),
                      i, cx[i], cy[i], cz[i], ra[i],
                      s_lo_x[i], s_lo_y[i], s_lo_z[i],
                      s_hi_x[i], s_hi_y[i], s_hi_z[i],
                      buf);

            int sz = (int)buf.size();
            if (sz == 0) continue;
            float r_sq_i = absr_sq[i];
            float qx = cx_p[i], qy = cy_p[i], qz = cz_p[i];
            auto dist2_of = [qx, qy, qz, cx_p, cy_p, cz_p](int j) {
                float dx = qx - cx_p[j], dy = qy - cy_p[j], dz = qz - cz_p[j];
                return dx * dx + dy * dy + dz * dz;
            };
            auto cmp_dist = [&dist2_of](int a, int b) { return dist2_of(a) < dist2_of(b); };

            // [0, nearest_k): 256 nearest, sorted asc by distance — picked
            // from the FULL raw list so selection is exact.
            int nearest_k = sz < NEAREST ? sz : NEAREST;
            if (nearest_k < sz)
                std::nth_element(buf.begin(), buf.begin() + nearest_k, buf.end(), cmp_dist);
            std::sort(buf.begin(), buf.begin() + nearest_k, cmp_dist);

            // [nearest_k, front_used): TOP_LARGEST by smallest signed distance
            // (= largest coverage). Precompute signed_d once per candidate so
            // nth_element/sort compare plain floats instead of recomputing
            // sqrt+div on every comparison.
            int front_used = nearest_k;
            if (sz > front_used) {
                int after_nearest = sz - nearest_k;
                int largest_k = after_nearest < TOP_LARGEST ? after_nearest : TOP_LARGEST;
                sd_tmp.clear();
                sd_tmp.reserve(after_nearest);
                for (int k = nearest_k; k < sz; k++) {
                    int j = buf[k];
                    float d2 = dist2_of(j);
                    float sd = (d2 + r_sq_i - absr_sq_p[j]) / std::sqrt(d2);
                    sd_tmp.emplace_back(sd, j);
                }
                if (largest_k < after_nearest)
                    std::nth_element(sd_tmp.begin(), sd_tmp.begin() + largest_k, sd_tmp.end());
                std::sort(sd_tmp.begin(), sd_tmp.begin() + largest_k);
                // Write the whole range back: top-K (sorted) up front, then the
                // remainder (nth_element-partitioned). Must write the remainder
                // too so that the Fisher-Yates sampling pool below contains no
                // duplicates of the already-picked top-K ids.
                for (int k = 0; k < after_nearest; k++)
                    buf[nearest_k + k] = sd_tmp[k].second;
                front_used += largest_k;
            }

            // Fill remaining kept slots [front_used, MAX_KEEP) with a
            // uniform random sample from [front_used, sz) via Fisher-Yates.
            // The RNG above is seeded by `i` (not OpenMP thread id) so this
            // sample is bit-exact across runs regardless of thread scheduling.
            if (sz > front_used) {
                int remain = sz - front_used;
                int cap = MAX_KEEP - front_used;
                int take = remain < cap ? remain : cap;
                for (int s = 0; s < take; s++) {
                    int lo = front_used + s;
                    int hi = sz - 1;
                    std::uniform_int_distribution<int> dist(lo, hi);
                    int pick = dist(rng);
                    if (pick != lo) std::swap(buf[lo], buf[pick]);
                }
            }

            int keep = sz < MAX_KEEP ? sz : MAX_KEEP;
            result[i].assign(buf.begin(), buf.begin() + keep);
        }
    }
    out_neighbors = std::move(result);
}

}  // namespace sphere_intersect_core
