/*
 * sphere_intersect.cpp  (v2)
 *
 * Multi-level spatial hash for spheres with large radius variance.
 *
 * Key insight: the v1 bug was that a big sphere querying a small-sphere
 * level had to scan (2*big_r / small_cell)^3 cells — potentially millions.
 *
 * Fix: each sphere is inserted into its "home" level AND all coarser
 * levels above. When querying, a sphere only searches its own home
 * level's grid, checking 3^3 = 27 cells. Because smaller spheres
 * have been promoted into coarser grids, all pairs are found.
 *
 * Why this works: if sphere A (big, level 3) and sphere B (small, level 0)
 * intersect, then B was inserted into level 3's grid (promoted up).
 * When A queries level 3's 27 cells, it finds B there.
 *
 * Cost per sphere: always 27 cells × avg population per cell.
 * Total inserts: each sphere inserted into (n_levels - home_level) grids,
 * at most ~27 cells per grid. With ~5 levels, that's ≤ 5*27 = 135 inserts.
 *
 * Compile (macOS):
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
#include <cmath>
#include <algorithm>
#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

static inline int64_t grid_key(int gx, int gy, int gz) {
    return ((int64_t)(gx + 500000) * 1000001LL + (int64_t)(gy + 500000))
           * 1000001LL + (int64_t)(gz + 500000);
}

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

    /* Insert a sphere into all cells it overlaps in this grid */
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
                    cells[grid_key(gx,gy,gz)].push_back({orig_idx, r});
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

    /* Copy to float */
    std::vector<float> cx(n), cy(n), cz(n), ra(n);
    float min_r = 1e30f, max_r = 0;
    for (int i = 0; i < n; i++) {
        cx[i]=(float)c(i,0); cy[i]=(float)c(i,1); cz[i]=(float)c(i,2);
        ra[i] = std::max((float)r(i), 1e-10f);
        min_r = std::min(min_r, ra[i]);
        max_r = std::max(max_r, ra[i]);
    }

    /* Build levels: level k has cell_size = base * ratio^k */
    float ratio = 4.0f;
    float base_cell = 2.0f * min_r;
    if (base_cell < 1e-8f) base_cell = 1e-8f;

    int n_levels = 1;
    { float cs = base_cell; while (cs < 2.0f * max_r) { cs *= ratio; n_levels++; } }
    if (n_levels > 20) n_levels = 20;

    std::vector<float> level_cs(n_levels);
    for (int lv = 0; lv < n_levels; lv++)
        level_cs[lv] = base_cell * std::pow(ratio, (float)lv);

    /* Assign home level: finest level where cell_size >= 2*radius */
    std::vector<int> home(n);
    for (int i = 0; i < n; i++) {
        int lv = 0;
        float need = 2.0f * ra[i];
        while (lv < n_levels - 1 && level_cs[lv] < need) lv++;
        home[i] = lv;
    }

    /* Build grids — insert each sphere into home level and all above */
    std::vector<GridLevel> levels(n_levels);
    for (int lv = 0; lv < n_levels; lv++)
        levels[lv].init(level_cs[lv]);

    for (int i = 0; i < n; i++) {
        for (int lv = home[i]; lv < n_levels; lv++)
            levels[lv].insert(i, cx[i], cy[i], cz[i], ra[i]);
    }

    /* Query: each sphere searches ONLY its home level, 3x3x3 cells */
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
                    grid_key(gcx+dx, gcy+dy, gcz+dz));
                if (it == levels[lv].cells.end()) continue;
                for (const auto &s : it->second) {
                    if (s.idx == i) continue;
                    float ddx = xi - cx[s.idx];
                    float ddy = yi - cy[s.idx];
                    float ddz = zi - cz[s.idx];
                    float sr = ri + s.r;
                    if (ddx*ddx + ddy*ddy + ddz*ddz < sr*sr)
                        buf.push_back(s.idx);
                }
            }

            /* Deduplicate — a sphere promoted into multiple cells
             * of this level could appear more than once */
            std::sort(buf.begin(), buf.end());
            buf.erase(std::unique(buf.begin(), buf.end()), buf.end());

            result[i] = std::move(buf);
            buf = std::vector<int>();
        }
    }

    /* Build CSR */
    std::vector<int> offsets(n + 1, 0);
    for (int i = 0; i < n; i++)
        offsets[i+1] = offsets[i] + (int)result[i].size();
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

PYBIND11_MODULE(sphere_intersect, m) {
    m.doc() = "Fast sphere intersection (multi-level spatial hash v2)";
    m.def("find_intersections", &find_intersections,
          py::arg("centers"), py::arg("radii"),
          "Returns (offsets, neighbors) in CSR format.");
}