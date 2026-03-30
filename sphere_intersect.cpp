/*
 * sphere_intersect.cpp  (v3)
 *
 * Four spatial acceleration methods for sphere intersection detection.
 * All return CSR format: (offsets, neighbors).
 *
 * Methods:
 *   1. find_intersections        — Multi-level spatial hash (v2, existing)
 *   2. find_intersections_sap    — 3-axis Sweep and Prune
 *   3. find_intersections_grid   — Uniform grid with mailbox stamping
 *   4. find_intersections_bvh    — BVH with median split
 *
 * Compile (macOS with OpenMP):
 *   c++ -O3 -shared -fPIC -undefined dynamic_lookup \
 *     -Xpreprocessor -fopenmp \
 *     -I/opt/homebrew/opt/libomp/include \
 *     -L/opt/homebrew/opt/libomp/lib -lomp \
 *     $(python3 -m pybind11 --includes) \
 *     -o sphere_intersect$(python3-config --extension-suffix) \
 *     sphere_intersect.cpp
 *
 * Without OpenMP (single-threaded):
 *   c++ -O3 -shared -fPIC -undefined dynamic_lookup \
 *     $(python3 -m pybind11 --includes) \
 *     -o sphere_intersect$(python3-config --extension-suffix) \
 *     sphere_intersect.cpp
 *
 * Linux:
 *   c++ -O3 -shared -fPIC -fopenmp \
 *     $(python3 -m pybind11 --includes) \
 *     -o sphere_intersect$(python3-config --extension-suffix) \
 *     sphere_intersect.cpp
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

/* ─── helpers ─── */

static inline int64_t grid_key(int gx, int gy, int gz) {
    return ((int64_t)(gx + 500000) * 1000001LL + (int64_t)(gy + 500000))
           * 1000001LL + (int64_t)(gz + 500000);
}

/* Build CSR output from per-sphere neighbor lists */
static py::tuple build_csr(const std::vector<std::vector<int>> &result, int n) {
    std::vector<int> offsets(n + 1, 0);
    for (int i = 0; i < n; i++)
        offsets[i + 1] = offsets[i] + (int)result[i].size();
    int total = offsets[n];

    py::array_t<int> py_off(n + 1);
    py::array_t<int> py_nbr(total);
    auto poff = py_off.mutable_unchecked<1>();
    auto pnbr = py_nbr.mutable_unchecked<1>();
    for (int i = 0; i <= n; i++) poff(i) = offsets[i];

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        int base = offsets[i];
        for (int k = 0; k < (int)result[i].size(); k++)
            pnbr(base + k) = result[i][k];
    }
    return py::make_tuple(py_off, py_nbr);
}

/* ═══════════════════════════════════════════════════════════════════
 * METHOD 1: Multi-level spatial hash (existing v2)
 * ═══════════════════════════════════════════════════════════════════ */

struct SphereRef {
    int idx;
    float r;
};

struct GridLevel {
    float cell_size, inv_cell;
    std::unordered_map<int64_t, std::vector<SphereRef>> cells;

    void init(float cs) {
        cell_size = cs;
        inv_cell = 1.0f / cs;
    }

    void insert(int orig_idx, float x, float y, float z, float r) {
        int lo_x = (int)std::floor((x - r) * inv_cell);
        int hi_x = (int)std::floor((x + r) * inv_cell);
        int lo_y = (int)std::floor((y - r) * inv_cell);
        int hi_y = (int)std::floor((y + r) * inv_cell);
        int lo_z = (int)std::floor((z - r) * inv_cell);
        int hi_z = (int)std::floor((z + r) * inv_cell);
        for (int gx = lo_x; gx <= hi_x; gx++)
            for (int gy = lo_y; gy <= hi_y; gy++)
                for (int gz = lo_z; gz <= hi_z; gz++)
                    cells[grid_key(gx, gy, gz)].push_back({orig_idx, r});
    }
};

py::tuple find_intersections(
    py::array_t<double> centers_arr,
    py::array_t<double> radii_arr)
{
    auto c = centers_arr.unchecked<2>();
    auto r = radii_arr.unchecked<1>();
    int n = (int)c.shape(0);

    if (n == 0) {
        py::array_t<int> off(1); *off.mutable_data(0) = 0;
        return py::make_tuple(off, py::array_t<int>(0));
    }

    std::vector<float> cx(n), cy(n), cz(n), ra(n);
    float min_r = 1e30f, max_r = 0;
    for (int i = 0; i < n; i++) {
        cx[i] = (float)c(i, 0); cy[i] = (float)c(i, 1); cz[i] = (float)c(i, 2);
        ra[i] = std::max((float)r(i), 1e-10f);
        min_r = std::min(min_r, ra[i]);
        max_r = std::max(max_r, ra[i]);
    }

    float ratio = 4.0f;
    float base_cell = 2.0f * min_r;
    if (base_cell < 1e-8f) base_cell = 1e-8f;

    int n_levels = 1;
    { float cs = base_cell; while (cs < 2.0f * max_r) { cs *= ratio; n_levels++; } }
    if (n_levels > 20) n_levels = 20;

    std::vector<float> level_cs(n_levels);
    for (int lv = 0; lv < n_levels; lv++)
        level_cs[lv] = base_cell * std::pow(ratio, (float)lv);

    std::vector<int> home(n);
    for (int i = 0; i < n; i++) {
        int lv = 0;
        float need = 2.0f * ra[i];
        while (lv < n_levels - 1 && level_cs[lv] < need) lv++;
        home[i] = lv;
    }

    std::vector<GridLevel> levels(n_levels);
    for (int lv = 0; lv < n_levels; lv++)
        levels[lv].init(level_cs[lv]);

    for (int i = 0; i < n; i++) {
        for (int lv = home[i]; lv < n_levels; lv++)
            levels[lv].insert(i, cx[i], cy[i], cz[i], ra[i]);
    }

    std::vector<std::vector<int>> result(n);

    #pragma omp parallel
    {
        std::vector<int> buf;
        #pragma omp for schedule(dynamic, 256)
        for (int i = 0; i < n; i++) {
            buf.clear();
            int lv = home[i];
            float inv = levels[lv].inv_cell;
            float xi = cx[i], yi = cy[i], zi = cz[i], ri = ra[i];

            int gcx = (int)std::floor(xi * inv);
            int gcy = (int)std::floor(yi * inv);
            int gcz = (int)std::floor(zi * inv);

            for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++)
            for (int dz = -1; dz <= 1; dz++) {
                auto it = levels[lv].cells.find(
                    grid_key(gcx + dx, gcy + dy, gcz + dz));
                if (it == levels[lv].cells.end()) continue;
                for (const auto &s : it->second) {
                    if (s.idx == i) continue;
                    float ddx = xi - cx[s.idx];
                    float ddy = yi - cy[s.idx];
                    float ddz = zi - cz[s.idx];
                    float sr = ri + s.r;
                    if (ddx * ddx + ddy * ddy + ddz * ddz < sr * sr)
                        buf.push_back(s.idx);
                }
            }

            std::sort(buf.begin(), buf.end());
            buf.erase(std::unique(buf.begin(), buf.end()), buf.end());
            result[i] = buf;
        }
    }

    return build_csr(result, n);
}

/* ═══════════════════════════════════════════════════════════════════
 * METHOD 2: 3-Axis Sweep and Prune
 *
 * Sort by x-min.  For each sphere i, sweep forward while x-intervals
 * overlap.  Filter by y-interval, then z-interval, then exact dist².
 *
 * Complexity: O(n log n) + O(n * k_xyz)
 * ═══════════════════════════════════════════════════════════════════ */

py::tuple find_intersections_sap(
    py::array_t<double> centers_arr,
    py::array_t<double> radii_arr)
{
    auto c = centers_arr.unchecked<2>();
    auto r = radii_arr.unchecked<1>();
    int n = (int)c.shape(0);

    if (n == 0) {
        py::array_t<int> off(1); *off.mutable_data(0) = 0;
        return py::make_tuple(off, py::array_t<int>(0));
    }

    /* Copy data and compute per-axis intervals */
    std::vector<float> cx(n), cy(n), cz(n), ra(n);
    std::vector<float> x_lo(n), x_hi(n), y_lo(n), y_hi(n), z_lo(n), z_hi(n);
    for (int i = 0; i < n; i++) {
        cx[i] = (float)c(i, 0); cy[i] = (float)c(i, 1); cz[i] = (float)c(i, 2);
        ra[i] = std::max((float)r(i), 0.0f);
        x_lo[i] = cx[i] - ra[i]; x_hi[i] = cx[i] + ra[i];
        y_lo[i] = cy[i] - ra[i]; y_hi[i] = cy[i] + ra[i];
        z_lo[i] = cz[i] - ra[i]; z_hi[i] = cz[i] + ra[i];
    }

    /* Sort by x-min */
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return x_lo[a] < x_lo[b];
    });

    /* Build sorted x_lo/x_hi for fast sweep */
    std::vector<float> sorted_x_lo(n), sorted_x_hi(n);
    for (int si = 0; si < n; si++) {
        sorted_x_lo[si] = x_lo[order[si]];
        sorted_x_hi[si] = x_hi[order[si]];
    }

    /* Sweep — single-threaded because forward scan is sequential */
    std::vector<std::vector<int>> result(n);

    for (int si = 0; si < n; si++) {
        int i = order[si];
        float xi_hi_val = sorted_x_hi[si];
        float yi_lo_val = y_lo[i], yi_hi_val = y_hi[i];
        float zi_lo_val = z_lo[i], zi_hi_val = z_hi[i];
        float ri = ra[i];
        float cxi = cx[i], cyi = cy[i], czi = cz[i];

        for (int sj = si + 1; sj < n && sorted_x_lo[sj] < xi_hi_val; sj++) {
            int j = order[sj];

            /* Y-interval overlap */
            if (y_lo[j] >= yi_hi_val || y_hi[j] <= yi_lo_val) continue;
            /* Z-interval overlap */
            if (z_lo[j] >= zi_hi_val || z_hi[j] <= zi_lo_val) continue;

            /* Exact distance² check */
            float dx = cxi - cx[j];
            float dy = cyi - cy[j];
            float dz = czi - cz[j];
            float sr = ri + ra[j];
            if (dx * dx + dy * dy + dz * dz < sr * sr) {
                result[i].push_back(j);
                result[j].push_back(i);
            }
        }
    }

    /* Sort each neighbor list */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++)
        std::sort(result[i].begin(), result[i].end());

    return build_csr(result, n);
}

/* ═══════════════════════════════════════════════════════════════════
 * METHOD 3: Uniform Grid with Mailbox Stamping
 *
 * cell_size = 2 * r_max.  Each sphere inserted into all cells its
 * bbox overlaps.  Query: 27 neighbor cells, mailbox stamp avoids
 * duplicate pair tests.
 *
 * Complexity: O(n) expected for uniform distributions.
 * ═══════════════════════════════════════════════════════════════════ */

py::tuple find_intersections_grid(
    py::array_t<double> centers_arr,
    py::array_t<double> radii_arr)
{
    auto c = centers_arr.unchecked<2>();
    auto r = radii_arr.unchecked<1>();
    int n = (int)c.shape(0);

    if (n == 0) {
        py::array_t<int> off(1); *off.mutable_data(0) = 0;
        return py::make_tuple(off, py::array_t<int>(0));
    }

    std::vector<float> cx(n), cy(n), cz(n), ra(n);
    float max_r = 0;
    for (int i = 0; i < n; i++) {
        cx[i] = (float)c(i, 0); cy[i] = (float)c(i, 1); cz[i] = (float)c(i, 2);
        ra[i] = std::max((float)r(i), 1e-10f);
        max_r = std::max(max_r, ra[i]);
    }

    float cell_size = 2.0f * max_r;
    if (cell_size < 1e-8f) cell_size = 1e-8f;
    float inv_cell = 1.0f / cell_size;

    /* Build grid: insert each sphere into all cells its bbox overlaps */
    std::unordered_map<int64_t, std::vector<int>> grid;
    grid.reserve(n * 2);

    for (int i = 0; i < n; i++) {
        int lo_x = (int)std::floor((cx[i] - ra[i]) * inv_cell);
        int hi_x = (int)std::floor((cx[i] + ra[i]) * inv_cell);
        int lo_y = (int)std::floor((cy[i] - ra[i]) * inv_cell);
        int hi_y = (int)std::floor((cy[i] + ra[i]) * inv_cell);
        int lo_z = (int)std::floor((cz[i] - ra[i]) * inv_cell);
        int hi_z = (int)std::floor((cz[i] + ra[i]) * inv_cell);
        for (int gx = lo_x; gx <= hi_x; gx++)
            for (int gy = lo_y; gy <= hi_y; gy++)
                for (int gz = lo_z; gz <= hi_z; gz++)
                    grid[grid_key(gx, gy, gz)].push_back(i);
    }

    /* Query: each sphere checks center cell + 26 neighbors.
     * Parallel-safe: each sphere only writes to its own result[i].
     * No j>i restriction, no mailbox needed — dedup via sort+unique. */
    std::vector<std::vector<int>> result(n);

    #pragma omp parallel
    {
        std::vector<int> buf;
        #pragma omp for schedule(dynamic, 256)
        for (int i = 0; i < n; i++) {
            buf.clear();
            float xi = cx[i], yi = cy[i], zi = cz[i], ri = ra[i];

            int gcx = (int)std::floor(xi * inv_cell);
            int gcy = (int)std::floor(yi * inv_cell);
            int gcz = (int)std::floor(zi * inv_cell);

            for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++)
            for (int dz = -1; dz <= 1; dz++) {
                auto it = grid.find(grid_key(gcx + dx, gcy + dy, gcz + dz));
                if (it == grid.end()) continue;
                for (int j : it->second) {
                    if (j == i) continue;
                    float ddx = xi - cx[j];
                    float ddy = yi - cy[j];
                    float ddz = zi - cz[j];
                    float sr = ri + ra[j];
                    if (ddx * ddx + ddy * ddy + ddz * ddz < sr * sr)
                        buf.push_back(j);
                }
            }

            /* Dedup: a sphere inserted into multiple cells may appear
             * in multiple neighbor cells of this query */
            std::sort(buf.begin(), buf.end());
            buf.erase(std::unique(buf.begin(), buf.end()), buf.end());
            result[i] = buf;
        }
    }

    return build_csr(result, n);
}

/* ═══════════════════════════════════════════════════════════════════
 * METHOD 4: BVH with Median Split
 *
 * Top-down build: at each node, pick longest AABB axis, sort by
 * center, split at median.  Leaves hold ≤ leaf_size spheres.
 * Query prunes via AABB-vs-AABB overlap.
 *
 * Complexity: O(n log n) build + O(n log n) expected queries.
 * ═══════════════════════════════════════════════════════════════════ */

struct BVHNode {
    float lo[3], hi[3];      /* AABB of this subtree */
    int left, right;          /* child indices, -1 if leaf */
    int leaf_start, leaf_count; /* range into sorted index array */
};

static void bvh_build(
    std::vector<BVHNode> &nodes,
    std::vector<int> &leaf_indices,  /* permuted sphere indices */
    const float *cx, const float *cy, const float *cz, const float *ra,
    const float *s_lo_x, const float *s_lo_y, const float *s_lo_z,
    const float *s_hi_x, const float *s_hi_y, const float *s_hi_z,
    int *idx_buf,  /* workspace: indices to partition */
    int start, int count,
    int leaf_size)
{
    int node_id = (int)nodes.size();
    nodes.push_back(BVHNode());
    BVHNode &node = nodes[node_id];

    /* Compute AABB */
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
        /* Leaf */
        nodes[node_id].left = -1;
        nodes[node_id].right = -1;
        nodes[node_id].leaf_start = (int)leaf_indices.size();
        nodes[node_id].leaf_count = count;
        for (int k = start; k < start + count; k++)
            leaf_indices.push_back(idx_buf[k]);
        return;
    }

    /* Pick longest axis */
    float ext0 = hi0 - lo0, ext1 = hi1 - lo1, ext2 = hi2 - lo2;
    int axis = 0;
    if (ext1 > ext0 && ext1 >= ext2) axis = 1;
    else if (ext2 > ext0 && ext2 > ext1) axis = 2;

    /* Sort by center along axis */
    const float *axis_ptr = (axis == 0) ? cx : (axis == 1) ? cy : cz;
    std::sort(idx_buf + start, idx_buf + start + count, [axis_ptr](int a, int b) {
        return axis_ptr[a] < axis_ptr[b];
    });

    int mid = count / 2;

    /* Build children — reserve node_id since vector may reallocate */
    nodes[node_id].leaf_start = -1;
    nodes[node_id].leaf_count = 0;

    int left_id = (int)nodes.size();
    /* We can't hold a reference to nodes[node_id] across recursive calls
     * because the vector may reallocate.  Record left_id, build, then set. */
    bvh_build(nodes, leaf_indices, cx, cy, cz, ra,
              s_lo_x, s_lo_y, s_lo_z, s_hi_x, s_hi_y, s_hi_z,
              idx_buf, start, mid, leaf_size);

    int right_id = (int)nodes.size();
    bvh_build(nodes, leaf_indices, cx, cy, cz, ra,
              s_lo_x, s_lo_y, s_lo_z, s_hi_x, s_hi_y, s_hi_z,
              idx_buf, start + mid, count - mid, leaf_size);

    nodes[node_id].left = left_id;
    nodes[node_id].right = right_id;
}

/* Iterative query using explicit stack to avoid deep recursion */
static void bvh_query(
    const std::vector<BVHNode> &nodes,
    const std::vector<int> &leaf_indices,
    const float *cx, const float *cy, const float *cz, const float *ra,
    int qi, float qx, float qy, float qz, float qr,
    float q_lo0, float q_lo1, float q_lo2,
    float q_hi0, float q_hi1, float q_hi2,
    std::vector<int> &out)
{
    /* Explicit stack */
    int stack[64];
    int sp = 0;
    stack[sp++] = 0;  /* root */

    while (sp > 0) {
        int nid = stack[--sp];
        const BVHNode &nd = nodes[nid];

        /* AABB overlap test */
        if (q_lo0 > nd.hi[0] || q_hi0 < nd.lo[0] ||
            q_lo1 > nd.hi[1] || q_hi1 < nd.lo[1] ||
            q_lo2 > nd.hi[2] || q_hi2 < nd.lo[2])
            continue;

        if (nd.left == -1) {
            /* Leaf: exact check */
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

py::tuple find_intersections_bvh(
    py::array_t<double> centers_arr,
    py::array_t<double> radii_arr,
    int leaf_size = 16)
{
    auto c = centers_arr.unchecked<2>();
    auto r = radii_arr.unchecked<1>();
    int n = (int)c.shape(0);

    if (n == 0) {
        py::array_t<int> off(1); *off.mutable_data(0) = 0;
        return py::make_tuple(off, py::array_t<int>(0));
    }

    std::vector<float> cx(n), cy(n), cz(n), ra(n);
    std::vector<float> s_lo_x(n), s_lo_y(n), s_lo_z(n);
    std::vector<float> s_hi_x(n), s_hi_y(n), s_hi_z(n);
    for (int i = 0; i < n; i++) {
        cx[i] = (float)c(i, 0); cy[i] = (float)c(i, 1); cz[i] = (float)c(i, 2);
        ra[i] = std::max((float)r(i), 0.0f);
        s_lo_x[i] = cx[i] - ra[i]; s_hi_x[i] = cx[i] + ra[i];
        s_lo_y[i] = cy[i] - ra[i]; s_hi_y[i] = cy[i] + ra[i];
        s_lo_z[i] = cz[i] - ra[i]; s_hi_z[i] = cz[i] + ra[i];
    }

    /* Build BVH */
    std::vector<BVHNode> nodes;
    std::vector<int> leaf_indices;
    nodes.reserve(2 * n / leaf_size);
    leaf_indices.reserve(n);

    std::vector<int> idx_buf(n);
    std::iota(idx_buf.begin(), idx_buf.end(), 0);

    bvh_build(nodes, leaf_indices,
              cx.data(), cy.data(), cz.data(), ra.data(),
              s_lo_x.data(), s_lo_y.data(), s_lo_z.data(),
              s_hi_x.data(), s_hi_y.data(), s_hi_z.data(),
              idx_buf.data(), 0, n, leaf_size);

    /* Query all spheres — parallelizable since each writes to its own result[i] */
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
            result[i] = buf;
        }
    }

    return build_csr(result, n);
}

/* ═══════════════════════════════════════════════════════════════════
 * pybind11 module
 * ═══════════════════════════════════════════════════════════════════ */

PYBIND11_MODULE(sphere_intersect, m) {
    m.doc() = "Fast sphere intersection detection (4 methods, CSR output)";

    m.def("find_intersections", &find_intersections,
          py::arg("centers"), py::arg("radii"),
          "Multi-level spatial hash. Returns (offsets, neighbors) CSR.");

    m.def("find_intersections_sap", &find_intersections_sap,
          py::arg("centers"), py::arg("radii"),
          "3-axis Sweep and Prune. Returns (offsets, neighbors) CSR.");

    m.def("find_intersections_grid", &find_intersections_grid,
          py::arg("centers"), py::arg("radii"),
          "Uniform grid with mailbox stamping. Returns (offsets, neighbors) CSR.");

    m.def("find_intersections_bvh", &find_intersections_bvh,
          py::arg("centers"), py::arg("radii"),
          py::arg("leaf_size") = 16,
          "BVH with median split. Returns (offsets, neighbors) CSR.");
}