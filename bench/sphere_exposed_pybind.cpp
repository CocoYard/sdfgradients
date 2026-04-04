/*
 * sphere_exposed_pybind.cpp
 *
 * pybind11 reimplementation of sphere_exposed_cpp.cpp
 * Same algorithm, cleaner Python binding layer.
 *
 * Compile (macOS ARM64):
 *   c++ -O3 -shared -fPIC -undefined dynamic_lookup -std=c++17 \
 *     $(python3 -m pybind11 --includes) \
 *     -o sphere_exposed_pybind$(python3-config --extension-suffix) \
 *     sphere_exposed_pybind.cpp -lm
 *
 * Compile (Linux):
 *   c++ -O3 -shared -fPIC -std=c++17 \
 *     $(python3 -m pybind11 --includes) \
 *     -o sphere_exposed_pybind$(python3-config --extension-suffix) \
 *     sphere_exposed_pybind.cpp -lm
 *
 * With OpenMP (optional):
 *   macOS: add  -Xpreprocessor -fopenmp -lomp
 *   Linux:  add  -fopenmp
 *
 * Install pybind11 if needed:
 *   pip install pybind11
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>   // std::vector <-> Python list automatic conversion

#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <stdexcept>

namespace py = pybind11;
using namespace pybind11::literals;  // enables "key"_a syntax for py::dict

/* ═══════════════════════════════════════════════════════════════════
 * Constants
 * ═══════════════════════════════════════════════════════════════════ */
static constexpr double PI     = 3.14159265358979323846;
static constexpr double TWO_PI = 2.0 * PI;
static constexpr double EPS    = 1e-14;
static constexpr double EPS12  = 1e-12;
static constexpr double EPS10  = 1e-10;

static constexpr int MAX_CAPS      = 512;
static constexpr int MAX_ARCS      = 2048;
static constexpr int MAX_INTERVALS = 512;
static constexpr int MAX_DEGEN_PTS = 256;

/* ═══════════════════════════════════════════════════════════════════
 * Tolerances
 * ═══════════════════════════════════════════════════════════════════ */
struct Tolerances {
    double tol;
    double degen_tol;
    double merge_tol;
};

/* ═══════════════════════════════════════════════════════════════════
 * Vec3
 * ═══════════════════════════════════════════════════════════════════ */
struct Vec3 {
    double x, y, z;
    Vec3() : x(0), y(0), z(0) {}
    Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
};

static inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
static inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
static inline Vec3 operator*(Vec3 a, double s) { return {a.x*s, a.y*s, a.z*s}; }
static inline Vec3 operator*(double s, Vec3 a) { return a*s; }
static inline double dot(Vec3 a, Vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static inline Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}
static inline double length(Vec3 a) { return std::sqrt(dot(a, a)); }
static inline Vec3 normalize(Vec3 a) {
    double l = length(a);
    return l < EPS ? Vec3{0,0,0} : a * (1.0/l);
}
static inline double clampd(double x, double lo, double hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}
static inline double fmod_pos(double x, double m) {
    double r = std::fmod(x, m);
    return r < 0 ? r + m : r;
}

/* ═══════════════════════════════════════════════════════════════════
 * Data structures
 * ═══════════════════════════════════════════════════════════════════ */
struct Cap {
    Vec3 normal;
    double d;
    Vec3 circle_center;
    double circle_radius;
    Vec3 local_u, local_v;
    double phi;
    int sphere_idx;
};

struct Interval {
    double start, end;
};

struct BoundaryArc {
    int cap_idx;
    double t_start, t_end;
};

/* ═══════════════════════════════════════════════════════════════════
 * Geometry helpers (identical to original)
 * ═══════════════════════════════════════════════════════════════════ */

static Vec3 perpendicular_unit(Vec3 n) {
    Vec3 ref = (std::fabs(n.x) < 0.9) ? Vec3{1,0,0} : Vec3{0,1,0};
    return normalize(cross(n, ref));
}

static bool compute_cap(Vec3 mc, double mr, Vec3 oc, double or_, int idx, Cap &out) {
    Vec3 diff = oc - mc;
    double dist = length(diff);

    if (dist < EPS) {
        if (or_ >= mr) {
            Vec3 n{1,0,0};
            out = {n, dot(n,mc) - mr - 1, mc, 0, {}, {}, PI, idx};
            return true;
        }
        return false;
    }
    if (dist >= mr + or_) return false;
    if (dist + or_ <= mr) return false;
    if (dist + mr <= or_) {
        Vec3 n = diff * (1.0/dist);
        out = {n, dot(n,mc) - mr - 1, mc, 0, {}, {}, PI, idx};
        return true;
    }

    Vec3 n = diff * (1.0/dist);
    double h = (mr*mr - or_*or_ + dist*dist) / (2.0*dist);
    Vec3 cc = mc + h * n;
    double cr = std::sqrt(std::max(0.0, mr*mr - h*h));
    double phi = std::acos(clampd(h / mr, -1.0, 1.0));
    Vec3 u = perpendicular_unit(n);
    Vec3 v = cross(n, u);
    out = {n, dot(n, mc) + h, cc, cr, u, v, phi, idx};
    return true;
}

static int compute_all_caps(Vec3 mc, double mr,
                            const double *oc, const double *or_,
                            int n_others, Cap caps[], double tol) {
    int nc = 0;
    for (int i = 0; i < n_others && nc < MAX_CAPS; i++) {
        Cap cap;
        if (!compute_cap(mc, mr, {oc[i*3], oc[i*3+1], oc[i*3+2]}, or_[i], i, cap))
            continue;

        bool dup = false;
        for (int e = nc - 1; e >= 0; e--) {
            double d = dot(cap.normal, caps[e].normal);
            if (d > 1 - tol) {
                if (std::fabs(cap.d - caps[e].d) < tol) { dup = true; break; }
                else if (cap.d > caps[e].d)             { dup = true; break; }
                else { caps[e] = caps[--nc]; break; }
            }
        }
        if (!dup) caps[nc++] = cap;
    }
    return nc;
}

static inline Vec3 point_on_circle(const Cap &cap, double t) {
    return cap.circle_center + cap.circle_radius * (std::cos(t) * cap.local_u + std::sin(t) * cap.local_v);
}

static inline bool is_inside_cap(Vec3 pt, const Cap &cap) {
    return dot(cap.normal, pt) - cap.d > EPS12;
}

static inline bool angle_in_arc(double t, double t_start, double t_end) {
    t = fmod_pos(t, TWO_PI);
    double dt = fmod_pos(t - t_start, TWO_PI);
    double arc_len = t_end - t_start;
    return dt < arc_len - EPS;
}

static int intersect_circle_with_plane(const Cap &circle_cap, const Cap &cutting_cap,
                                       double out[2]) {
    double R = circle_cap.circle_radius;
    if (R < EPS) return 0;

    double a = dot(cutting_cap.normal, circle_cap.local_u);
    double b = dot(cutting_cap.normal, circle_cap.local_v);
    double c_val = (cutting_cap.d - dot(cutting_cap.normal, circle_cap.circle_center)) / R;

    double A_amp = std::sqrt(a*a + b*b);
    if (A_amp < EPS) return 0;

    double ratio = c_val / A_amp;
    if (std::fabs(ratio) > 1.0 + EPS10) return 0;
    ratio = clampd(ratio, -1.0, 1.0);

    double alpha = std::atan2(b, a);
    double delta = std::acos(ratio);
    if (delta < EPS12) return 0;

    double t1 = fmod_pos(alpha - delta, TWO_PI);
    double t2 = fmod_pos(alpha + delta, TWO_PI);
    if (t1 <= t2) { out[0] = t1; out[1] = t2; }
    else          { out[0] = t2; out[1] = t1; }
    return 2;
}

static void clip_arc_by_cap(const BoundaryArc &arc, const Cap &cutting_cap,
                            const Cap caps[], std::vector<BoundaryArc> &result) {
    const Cap &host = caps[arc.cap_idx];

    double hits[2];
    int nhits = intersect_circle_with_plane(host, cutting_cap, hits);

    double hits_in[2];
    int n_in = 0;
    for (int i = 0; i < nhits; i++) {
        if (angle_in_arc(hits[i], arc.t_start, arc.t_end))
            hits_in[n_in++] = hits[i];
    }

    if (n_in == 0) {
        double mid_t = (arc.t_start + arc.t_end) / 2.0;
        Vec3 mid_pt = point_on_circle(host, mid_t);
        if (is_inside_cap(mid_pt, cutting_cap)) return;
        result.push_back(arc);
        return;
    }

    if (n_in == 2) {
        double k0 = fmod_pos(hits_in[0] - arc.t_start, TWO_PI);
        double k1 = fmod_pos(hits_in[1] - arc.t_start, TWO_PI);
        if (k0 > k1) std::swap(hits_in[0], hits_in[1]);
    }

    double boundaries[4];
    boundaries[0] = arc.t_start;
    for (int i = 0; i < n_in; i++) boundaries[1 + i] = hits_in[i];
    boundaries[1 + n_in] = arc.t_end;
    int nbnd = 2 + n_in;

    for (int k = 0; k < nbnd - 1; k++) {
        double t_s = boundaries[k];
        double t_e = boundaries[k + 1];
        if (t_e < t_s - EPS) t_e += TWO_PI;
        if (t_e - t_s < 1e-15) continue;

        Vec3 mid_pt = point_on_circle(host, (t_s + t_e) / 2.0);
        if (!is_inside_cap(mid_pt, cutting_cap)) {
            result.push_back({arc.cap_idx, t_s, t_e});
        }
    }
}

static int intersect_intervals(const Interval *a, int na,
                               const Interval *b, int nb,
                               Interval *out, int max_out,
                               double skip_tol, double merge_tol) {
    int cnt = 0;
    for (int i = 0; i < na; i++) {
        for (int j = 0; j < nb; j++) {
            double lo = std::max(a[i].start, b[j].start);
            double hi = std::min(a[i].end,   b[j].end);
            if (hi - lo > -skip_tol && cnt < max_out) {
                if (hi < lo) {
                    double mid = (lo + hi) / 2.0;
                    out[cnt++] = {mid - 1e-15, mid + 1e-15};
                } else {
                    out[cnt++] = {lo, hi};
                }
            }
        }
    }
    std::sort(out, out + cnt, [](const Interval &a, const Interval &b) {
        return a.start < b.start;
    });
    int m = 0;
    for (int i = 0; i < cnt; i++) {
        if (m > 0 && out[i].start <= out[m-1].end + merge_tol)
            out[m-1].end = std::max(out[m-1].end, out[i].end);
        else
            out[m++] = out[i];
    }
    return m;
}

static int compute_exposed_arcs_on_circle(int cap_idx, const Cap caps[],
                                          const int *active_caps, int n_active,
                                          Interval *result, const Tolerances &tol) {
    const Cap &host = caps[cap_idx];
    double R = host.circle_radius;
    if (R < EPS) return 0;

    Interval iv[MAX_INTERVALS], tmp[MAX_INTERVALS];
    int niv = 1;
    iv[0] = {0.0, TWO_PI};

    for (int ai = 0; ai < n_active; ai++) {
        int j = active_caps[ai];
        if (j == cap_idx) continue;
        const Cap &other = caps[j];

        double a = -dot(other.normal, host.local_u);
        double b = -dot(other.normal, host.local_v);
        double c = -(other.d - dot(other.normal, host.circle_center));

        double A_line = std::sqrt(a*a + b*b);
        if (A_line < EPS) {
            if (c > tol.tol) return 0;
            continue;
        }

        double ratio = c / (R * A_line);
        if (ratio <= -1.0 - EPS10) continue;
        if (ratio >= 1.0 + EPS10) return 0;
        ratio = clampd(ratio, -1.0, 1.0);

        double alpha = std::atan2(b, a);
        double delta = std::acos(ratio);

        double arc_s = fmod_pos(alpha - delta, TWO_PI);
        double arc_e = fmod_pos(alpha + delta, TWO_PI);

        Interval con[2];
        int ncn;
        if (arc_s < arc_e) {
            con[0] = {arc_s, arc_e}; ncn = 1;
        } else {
            con[0] = {arc_s, TWO_PI}; con[1] = {0.0, arc_e}; ncn = 2;
        }

        niv = intersect_intervals(iv, niv, con, ncn, tmp, MAX_INTERVALS,
                                  tol.tol, tol.merge_tol);
        if (niv == 0) return 0;
        std::memcpy(iv, tmp, niv * sizeof(Interval));
    }

    for (int i = 0; i < niv; i++) {
        result[i] = {std::min(iv[i].start, iv[i].end),
                     std::max(iv[i].start, iv[i].end)};
    }
    return niv;
}

static bool is_pt_exposed(Vec3 pt, const Cap caps[], int nc, int e1, int e2,
                          double tol) {
    for (int k = 0; k < nc; k++) {
        if (k == e1 || k == e2) continue;
        if (dot(caps[k].normal, pt) - caps[k].d > tol) return false;
    }
    return true;
}

static bool solve3(const double A[9], const double b[3], double x[3]) {
    double det = A[0]*(A[4]*A[8]-A[5]*A[7]) - A[1]*(A[3]*A[8]-A[5]*A[6])
               + A[2]*(A[3]*A[7]-A[4]*A[6]);
    if (std::fabs(det) < EPS) return false;
    double inv = 1.0 / det;
    x[0] = inv*(b[0]*(A[4]*A[8]-A[5]*A[7]) - A[1]*(b[1]*A[8]-A[5]*b[2]) + A[2]*(b[1]*A[7]-A[4]*b[2]));
    x[1] = inv*(A[0]*(b[1]*A[8]-A[5]*b[2]) - b[0]*(A[3]*A[8]-A[5]*A[6]) + A[2]*(A[3]*b[2]-b[1]*A[6]));
    x[2] = inv*(A[0]*(A[4]*b[2]-b[1]*A[7]) - A[1]*(A[3]*b[2]-b[1]*A[6]) + b[0]*(A[3]*A[7]-A[4]*A[6]));
    return true;
}

static int find_degen_pts(Vec3 mc, double mr, const Cap caps[], int nc,
                          Vec3 pts[], const Tolerances &tol) {
    int np = 0;
    for (int i = 0; i < nc && np < MAX_DEGEN_PTS; i++) {
        if (caps[i].circle_radius < EPS) continue;
        for (int j = i + 1; j < nc && np < MAX_DEGEN_PTS; j++) {
            if (caps[j].circle_radius < EPS) continue;

            Vec3 ni = caps[i].normal, nj = caps[j].normal;
            Vec3 ld = cross(ni, nj);
            double ldn = length(ld);
            if (ldn < 1e-12) continue;
            ld = ld * (1.0 / ldn);

            double A[9] = {ni.x,ni.y,ni.z, nj.x,nj.y,nj.z, ld.x,ld.y,ld.z};
            double bv[3] = {caps[i].d, caps[j].d, 0};
            double p0[3];
            if (!solve3(A, bv, p0)) continue;

            Vec3 p0v{p0[0], p0[1], p0[2]};
            Vec3 d = p0v - mc;
            double bc = 2 * dot(d, ld);
            double cc = dot(d, d) - mr * mr;
            double disc = bc * bc - 4 * cc;
            if (disc < -EPS10) continue;
            if (disc < 0) disc = 0;
            double sq = std::sqrt(disc);

            for (int s = -1; s <= 1; s += 2) {
                double t = (-bc + s * sq) / 2.0;
                Vec3 pt = p0v + t * ld;

                if (std::fabs(length(pt - mc) - mr) > tol.degen_tol) continue;
                if (!is_pt_exposed(pt, caps, nc, i, j, tol.tol)) continue;
                bool dup = false;
                for (int k = 0; k < np; k++) {
                    if (length(pt - pts[k]) < tol.degen_tol) { dup = true; break; }
                }
                if (!dup) pts[np++] = pt;
            }
        }
    }
    return np;
}

/* ═══════════════════════════════════════════════════════════════════
 * compute_one: same as before
 * ═══════════════════════════════════════════════════════════════════ */
static void compute_one(
    Vec3 mc, double mr, const double *oc, const double *or_, int n_others,
    const Tolerances &tol,
    int *out_ncaps,
    int *out_narcs, int *arc_cap, double *arc_s, double *arc_e, int max_arcs,
    double *out_total_arc,
    int *out_npts, Vec3 *out_pts,
    Cap *out_caps, int *out_nc,
    int *out_remap)
{
    Cap caps[MAX_CAPS];
    int nc = compute_all_caps(mc, mr, oc, or_, n_others, caps, tol.tol);
    *out_ncaps = 0;
    *out_total_arc = 0;
    *out_narcs = 0;
    *out_npts = 0;
    *out_nc = nc;

    if (out_caps)  std::memcpy(out_caps, caps, nc * sizeof(Cap));
    if (out_remap) std::memset(out_remap, -1, nc * sizeof(int));

    if (nc == 0) return;

    for (int i = 0; i < nc; i++) {
        if (caps[i].phi >= PI - EPS10) return;
    }

    int valid_caps[MAX_CAPS];
    int n_valid = 0;
    for (int i = 0; i < nc; i++) {
        if (caps[i].circle_radius >= EPS)
            valid_caps[n_valid++] = i;
    }
    if (n_valid == 0) return;

    std::vector<BoundaryArc> all_arcs;
    all_arcs.reserve(n_valid * 4);

    int active_caps[MAX_CAPS];
    int n_active = 0;

    for (int vi = 0; vi < n_valid; vi++) {
        int cap_idx = valid_caps[vi];

        std::vector<BoundaryArc> new_arcs;
        new_arcs.reserve(all_arcs.size() + 4);
        for (const auto &arc : all_arcs)
            clip_arc_by_cap(arc, caps[cap_idx], caps, new_arcs);

        Interval new_intervals[MAX_INTERVALS];
        int n_new = compute_exposed_arcs_on_circle(cap_idx, caps,
                                                   active_caps, n_active,
                                                   new_intervals, tol);
        active_caps[n_active++] = cap_idx;

        for (int a = 0; a < n_new; a++)
            new_arcs.push_back({cap_idx, new_intervals[a].start, new_intervals[a].end});
        all_arcs = std::move(new_arcs);
    }

    int arc_count[MAX_CAPS] = {};
    for (const auto &arc : all_arcs)
        if (arc.cap_idx < MAX_CAPS) arc_count[arc.cap_idx]++;

    int remap[MAX_CAPS];
    int kept = 0;
    for (int i = 0; i < nc; i++) {
        if (arc_count[i] > 0) remap[i] = kept++;
        else                  remap[i] = -1;
    }
    if (out_remap) std::memcpy(out_remap, remap, nc * sizeof(int));

    double total = 0;
    int na = 0;
    for (const auto &arc : all_arcs) {
        if (na >= max_arcs) break;
        arc_cap[na] = remap[arc.cap_idx];
        arc_s[na]   = arc.t_start;
        arc_e[na]   = arc.t_end;
        total += arc.t_end - arc.t_start;
        na++;
    }
    *out_narcs = na;
    *out_ncaps = kept;
    *out_total_arc = total;

    if (total < tol.degen_tol && nc > 0)
        *out_npts = find_degen_pts(mc, mr, caps, nc, out_pts, tol);
}

/* ═══════════════════════════════════════════════════════════════════
 * Helper: make a C-contiguous (rows, cols) float64 numpy array
 * ═══════════════════════════════════════════════════════════════════ */
static py::array_t<double> make_f64_2d(int rows, int cols) {
    return py::array_t<double>({(py::ssize_t)rows, (py::ssize_t)cols});
}
static py::array_t<double> make_f64_1d(int n) {
    return py::array_t<double>(n);
}
static py::array_t<int32_t> make_i32_1d(int n) {
    return py::array_t<int32_t>(n);
}

/* ═══════════════════════════════════════════════════════════════════
 * compute_exposed_single
 *
 * Python signature:
 *   compute_exposed_single(center, radius, other_centers, other_radii,
 *                          tol=1e-8, degen_tol=1e-6, merge_tol=1e-12) -> dict
 *
 * Returns the same keys as the original C-API version.
 * ═══════════════════════════════════════════════════════════════════ */
py::dict compute_exposed_single(
    py::array_t<double> py_mc,
    double mr,
    py::array_t<double> py_oc,
    py::array_t<double> py_or,
    double tol_v      = 1e-8,
    double degen_tol_v = 1e-6,
    double merge_tol_v = 1e-12)
{
    // ── Input validation ──────────────────────────────────────────
    auto mc_buf = py_mc.request();
    auto oc_buf = py_oc.request();
    auto or_buf = py_or.request();

    if (mc_buf.ndim != 1 || mc_buf.shape[0] != 3)
        throw std::invalid_argument("center must be shape (3,)");
    if (oc_buf.ndim != 2 || oc_buf.shape[1] != 3)
        throw std::invalid_argument("other_centers must be shape (N,3)");

    Tolerances tol{tol_v, degen_tol_v, merge_tol_v};

    int n = (int)oc_buf.shape[0];
    double *mc_ptr = static_cast<double*>(mc_buf.ptr);
    double *oc_ptr = static_cast<double*>(oc_buf.ptr);
    double *or_ptr = static_cast<double*>(or_buf.ptr);

    // ── Core computation ──────────────────────────────────────────
    int ncaps, narcs, npts, nc_all;
    double total_arc;
    int    arc_cap_buf[MAX_ARCS];
    double arc_s_buf[MAX_ARCS], arc_e_buf[MAX_ARCS];
    Vec3   dpts[MAX_DEGEN_PTS];
    Cap    all_caps[MAX_CAPS];
    int    remap[MAX_CAPS];

    compute_one(
        {mc_ptr[0], mc_ptr[1], mc_ptr[2]}, mr,
        oc_ptr, or_ptr, n, tol,
        &ncaps, &narcs, arc_cap_buf, arc_s_buf, arc_e_buf, MAX_ARCS,
        &total_arc, &npts, dpts,
        all_caps, &nc_all, remap);

    // ── arcs_by_cap dict  (compacted indices 0..ncaps-1) ─────────
    py::dict arcs_by_cap;
    for (int i = 0; i < ncaps; i++)
        arcs_by_cap[py::int_(i)] = py::list();
    for (int a = 0; a < narcs; a++) {
        py::list lst = arcs_by_cap[py::int_(arc_cap_buf[a])];
        lst.append(py::make_tuple(arc_s_buf[a], arc_e_buf[a]));
    }

    // ── exposed_points list ───────────────────────────────────────
    py::list exposed_points;
    for (int i = 0; i < npts; i++) {
        auto p = make_f64_1d(3);
        auto pm = p.mutable_unchecked<1>();
        pm(0) = dpts[i].x; pm(1) = dpts[i].y; pm(2) = dpts[i].z;
        exposed_points.append(p);
    }

    // ── Compacted cap geometry (caps that have arcs) ──────────────
    auto cap_normals = make_f64_2d(ncaps, 3);
    auto cap_d       = make_f64_1d(ncaps);
    auto cap_centers = make_f64_2d(ncaps, 3);
    auto cap_radii   = make_f64_1d(ncaps);
    auto cap_u       = make_f64_2d(ncaps, 3);
    auto cap_v       = make_f64_2d(ncaps, 3);

    auto cn = cap_normals.mutable_unchecked<2>();
    auto cd = cap_d.mutable_unchecked<1>();
    auto cc = cap_centers.mutable_unchecked<2>();
    auto cr = cap_radii.mutable_unchecked<1>();
    auto cu = cap_u.mutable_unchecked<2>();
    auto cv = cap_v.mutable_unchecked<2>();

    for (int i = 0; i < nc_all; i++) {
        int ri = remap[i];
        if (ri < 0) continue;
        cn(ri,0) = all_caps[i].normal.x;
        cn(ri,1) = all_caps[i].normal.y;
        cn(ri,2) = all_caps[i].normal.z;
        cd(ri)   = all_caps[i].d;
        cc(ri,0) = all_caps[i].circle_center.x;
        cc(ri,1) = all_caps[i].circle_center.y;
        cc(ri,2) = all_caps[i].circle_center.z;
        cr(ri)   = all_caps[i].circle_radius;
        cu(ri,0) = all_caps[i].local_u.x;
        cu(ri,1) = all_caps[i].local_u.y;
        cu(ri,2) = all_caps[i].local_u.z;
        cv(ri,0) = all_caps[i].local_v.x;
        cv(ri,1) = all_caps[i].local_v.y;
        cv(ri,2) = all_caps[i].local_v.z;
    }

    // ── ALL caps (for query_inside) ───────────────────────────────
    auto all_cap_normals = make_f64_2d(nc_all, 3);
    auto all_cap_d       = make_f64_1d(nc_all);
    auto acn = all_cap_normals.mutable_unchecked<2>();
    auto acd = all_cap_d.mutable_unchecked<1>();
    for (int i = 0; i < nc_all; i++) {
        acn(i,0) = all_caps[i].normal.x;
        acn(i,1) = all_caps[i].normal.y;
        acn(i,2) = all_caps[i].normal.z;
        acd(i)   = all_caps[i].d;
    }

    // ── Arc arrays ────────────────────────────────────────────────
    auto arc_cap_idx = make_i32_1d(narcs);
    auto arc_start   = make_f64_1d(narcs);
    auto arc_end     = make_f64_1d(narcs);
    std::memcpy(arc_cap_idx.mutable_data(), arc_cap_buf, narcs * sizeof(int32_t));
    std::memcpy(arc_start.mutable_data(),   arc_s_buf,   narcs * sizeof(double));
    std::memcpy(arc_end.mutable_data(),     arc_e_buf,   narcs * sizeof(double));

    return py::dict(
        "arcs_by_cap"_a    = arcs_by_cap,
        "exposed_points"_a = exposed_points,
        "total_arc"_a      = total_arc,
        "n_caps"_a         = ncaps,
        "cap_normals"_a    = cap_normals,
        "cap_d"_a          = cap_d,
        "cap_centers"_a    = cap_centers,
        "cap_radii"_a      = cap_radii,
        "cap_u"_a          = cap_u,
        "cap_v"_a          = cap_v,
        "all_cap_normals"_a = all_cap_normals,
        "all_cap_d"_a       = all_cap_d,
        "arc_cap_idx"_a    = arc_cap_idx,
        "arc_start"_a      = arc_start,
        "arc_end"_a        = arc_end
    );
}

/* ═══════════════════════════════════════════════════════════════════
 * compute_exposed_batch
 *
 * Python signature:
 *   compute_exposed_batch(centers, radii, nbr_indices, nbr_offsets,
 *                         tol=1e-8, degen_tol=1e-6, merge_tol=1e-12) -> dict
 * ═══════════════════════════════════════════════════════════════════ */
py::dict compute_exposed_batch(
    py::array_t<double>  py_centers,
    py::array_t<double>  py_radii,
    py::array_t<int64_t> py_nbr_idx,
    py::array_t<int64_t> py_nbr_off,
    double tol_v       = 1e-8,
    double degen_tol_v = 1e-6,
    double merge_tol_v = 1e-12)
{
    auto c_buf   = py_centers.request();
    auto r_buf   = py_radii.request();
    auto idx_buf = py_nbr_idx.request();
    auto off_buf = py_nbr_off.request();

    Tolerances tol{tol_v, degen_tol_v, merge_tol_v};

    int N = (int)c_buf.shape[0];
    double  *centers  = static_cast<double*>(c_buf.ptr);
    double  *radii    = static_cast<double*>(r_buf.ptr);
    int64_t *nbr_idx  = static_cast<int64_t*>(idx_buf.ptr);
    int64_t *nbr_off  = static_cast<int64_t*>(off_buf.ptr);

    // Per-sphere summary arrays
    auto py_ncaps = make_i32_1d(N);
    auto py_narcs = make_i32_1d(N);
    auto py_npts  = make_i32_1d(N);
    auto py_tarc  = make_f64_1d(N);
    int32_t *s_ncaps = py_ncaps.mutable_data();
    int32_t *s_narcs = py_narcs.mutable_data();
    int32_t *s_npts  = py_npts.mutable_data();
    double  *s_tarc  = py_tarc.mutable_data();

    // Growable flat storage
    std::vector<int>    fa_sphere, fa_cap;
    std::vector<double> fa_s, fa_e;
    std::vector<int>    fc_sphere, fc_cap_id;
    std::vector<double> fc_normal, fc_d, fc_center, fc_radius, fc_u, fc_v;
    std::vector<int>    fp_sphere;
    std::vector<double> fp_pos;

    // Per-sphere scratch
    int    _ac[MAX_ARCS];
    double _as[MAX_ARCS], _ae[MAX_ARCS];
    Vec3   _dp[MAX_DEGEN_PTS];
    Cap    _caps[MAX_CAPS];

    // Neighbor gather buffer
    std::vector<double> oc_buf_v, or_buf_v;

    for (int i = 0; i < N; i++) {
        int n_nbrs = (int)(nbr_off[i+1] - nbr_off[i]);
        if (n_nbrs == 0) {
            s_ncaps[i] = 0; s_narcs[i] = 0; s_npts[i] = 0; s_tarc[i] = 0;
            continue;
        }

        oc_buf_v.resize(n_nbrs * 3);
        or_buf_v.resize(n_nbrs);

        const int64_t *nb = nbr_idx + nbr_off[i];
        for (int j = 0; j < n_nbrs; j++) {
            int64_t idx = nb[j];
            oc_buf_v[j*3]   = centers[idx*3];
            oc_buf_v[j*3+1] = centers[idx*3+1];
            oc_buf_v[j*3+2] = centers[idx*3+2];
            or_buf_v[j]     = radii[idx];
        }

        int ncaps, narcs, npts, n_pcaps;
        double total_arc;
        int remap[MAX_CAPS];
        compute_one(
            {centers[i*3], centers[i*3+1], centers[i*3+2]}, radii[i],
            oc_buf_v.data(), or_buf_v.data(), n_nbrs, tol,
            &ncaps, &narcs, _ac, _as, _ae, MAX_ARCS,
            &total_arc, &npts, _dp,
            _caps, &n_pcaps, remap);

        s_ncaps[i] = ncaps;
        s_narcs[i] = narcs;
        s_npts[i]  = npts;
        s_tarc[i]  = total_arc;

        // Append caps
        for (int c = 0; c < ncaps; c++) {
            int orig = -1;
            for (int k = 0; k < n_pcaps; k++) {
                if (remap[k] == c) { orig = k; break; }
            }
            if (orig < 0) continue;
            fc_sphere.push_back(i);
            fc_cap_id.push_back(c);
            fc_normal.push_back(_caps[orig].normal.x);
            fc_normal.push_back(_caps[orig].normal.y);
            fc_normal.push_back(_caps[orig].normal.z);
            fc_d.push_back(_caps[orig].d);
            fc_center.push_back(_caps[orig].circle_center.x);
            fc_center.push_back(_caps[orig].circle_center.y);
            fc_center.push_back(_caps[orig].circle_center.z);
            fc_radius.push_back(_caps[orig].circle_radius);
            fc_u.push_back(_caps[orig].local_u.x);
            fc_u.push_back(_caps[orig].local_u.y);
            fc_u.push_back(_caps[orig].local_u.z);
            fc_v.push_back(_caps[orig].local_v.x);
            fc_v.push_back(_caps[orig].local_v.y);
            fc_v.push_back(_caps[orig].local_v.z);
        }

        // Append arcs
        for (int a = 0; a < narcs; a++) {
            fa_sphere.push_back(i);
            fa_cap.push_back(_ac[a]);
            fa_s.push_back(_as[a]);
            fa_e.push_back(_ae[a]);
        }

        // Append degen points
        for (int p = 0; p < npts; p++) {
            fp_sphere.push_back(i);
            fp_pos.push_back(_dp[p].x);
            fp_pos.push_back(_dp[p].y);
            fp_pos.push_back(_dp[p].z);
        }
    }

    // ── Wrap std::vectors into numpy arrays ───────────────────────
    // Helper lambdas that copy a vector into a new numpy array
    auto vec_to_i32_1d = [](const std::vector<int> &v) {
        auto a = make_i32_1d((int)v.size());
        std::memcpy(a.mutable_data(), v.data(), v.size() * sizeof(int32_t));
        return a;
    };
    auto vec_to_f64_1d = [](const std::vector<double> &v) {
        auto a = make_f64_1d((int)v.size());
        std::memcpy(a.mutable_data(), v.data(), v.size() * sizeof(double));
        return a;
    };
    auto vec_to_f64_2d = [](const std::vector<double> &v, int rows) {
        auto a = make_f64_2d(rows, 3);
        std::memcpy(a.mutable_data(), v.data(), rows * 3 * sizeof(double));
        return a;
    };

    int total_arcs = (int)fa_sphere.size();
    int total_caps = (int)fc_sphere.size();
    int total_pts  = (int)fp_sphere.size();

    py::array_t<double> fp_pos_arr;
    if (total_pts > 0) {
        fp_pos_arr = vec_to_f64_2d(fp_pos, total_pts);
    } else {
        fp_pos_arr = make_f64_2d(0, 3);
    }

    return py::dict(
        "n_caps"_a          = py_ncaps,
        "n_arcs"_a          = py_narcs,
        "n_points"_a        = py_npts,
        "total_arc"_a       = py_tarc,
        "arc_sphere_idx"_a  = vec_to_i32_1d(fa_sphere),
        "arc_cap_idx"_a     = vec_to_i32_1d(fa_cap),
        "arc_start"_a       = vec_to_f64_1d(fa_s),
        "arc_end"_a         = vec_to_f64_1d(fa_e),
        "point_sphere_idx"_a = vec_to_i32_1d(fp_sphere),
        "point_positions"_a = fp_pos_arr,
        "cap_sphere_idx"_a  = vec_to_i32_1d(fc_sphere),
        "cap_id"_a          = vec_to_i32_1d(fc_cap_id),
        "cap_normals"_a     = vec_to_f64_2d(fc_normal, total_caps),
        "cap_d"_a           = vec_to_f64_1d(fc_d),
        "cap_centers"_a     = vec_to_f64_2d(fc_center, total_caps),
        "cap_radii"_a       = vec_to_f64_1d(fc_radius),
        "cap_u"_a           = vec_to_f64_2d(fc_u, total_caps),
        "cap_v"_a           = vec_to_f64_2d(fc_v, total_caps)
    );
}

/* ═══════════════════════════════════════════════════════════════════
 * query_inside
 *
 * Python signature:
 *   query_inside(points(N,3), all_cap_normals(M,3), all_cap_d(M,)) -> bool(N,)
 * ═══════════════════════════════════════════════════════════════════ */
py::array_t<bool> query_inside(
    py::array_t<double> py_pts,
    py::array_t<double> py_cn,
    py::array_t<double> py_cd)
{
    auto pts_r = py_pts.unchecked<2>();
    auto cn_r  = py_cn.unchecked<2>();
    auto cd_r  = py_cd.unchecked<1>();

    int n_pts  = (int)pts_r.shape(0);
    int n_caps = (int)cn_r.shape(0);

    auto out = py::array_t<bool>(n_pts);
    auto out_r = out.mutable_unchecked<1>();

    for (int i = 0; i < n_pts; i++) {
        double px = pts_r(i,0), py_ = pts_r(i,1), pz = pts_r(i,2);
        bool exposed = true;
        for (int j = 0; j < n_caps; j++) {
            if (cn_r(j,0)*px + cn_r(j,1)*py_ + cn_r(j,2)*pz - cd_r(j) > EPS12) {
                exposed = false; break;
            }
        }
        out_r(i) = exposed;
    }
    return out;
}

/* ═══════════════════════════════════════════════════════════════════
 * query_closest_on_arcs
 *
 * Python signature:
 *   query_closest_on_arcs(points(N,3), sphere_center(3,), sphere_radius,
 *     cap_centers(K,3), cap_radii(K,), cap_u(K,3), cap_v(K,3),
 *     arc_cap_idx(M,), arc_start(M,), arc_end(M,))
 *   -> (closest(N,3), distances(N,), arc_indices(N,))
 * ═══════════════════════════════════════════════════════════════════ */
std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<int32_t>>
query_closest_on_arcs(
    py::array_t<double>  py_pts,
    py::array_t<double>  py_cc,
    py::array_t<double>  py_cr,
    py::array_t<double>  py_cu,
    py::array_t<double>  py_cv,
    py::array_t<int32_t> py_ai,
    py::array_t<double>  py_as,
    py::array_t<double>  py_ae)
{
    auto pts_r = py_pts.unchecked<2>();
    auto cc_r  = py_cc.unchecked<2>();
    auto cr_r  = py_cr.unchecked<1>();
    auto cu_r  = py_cu.unchecked<2>();
    auto cv_r  = py_cv.unchecked<2>();
    auto ai_r  = py_ai.unchecked<1>();
    auto as_r  = py_as.unchecked<1>();
    auto ae_r  = py_ae.unchecked<1>();

    int n_pts  = (int)pts_r.shape(0);
    int n_arcs = (int)ai_r.shape(0);

    auto out_closest = make_f64_2d(n_pts, 3);
    auto out_dist    = make_f64_1d(n_pts);
    auto out_arcidx  = make_i32_1d(n_pts);
    auto oc_r = out_closest.mutable_unchecked<2>();
    auto od_r = out_dist.mutable_unchecked<1>();
    auto oa_r = out_arcidx.mutable_unchecked<1>();

    for (int i = 0; i < n_pts; i++) {
        double px = pts_r(i,0), py_ = pts_r(i,1), pz = pts_r(i,2);
        double best_dist2 = 1e30;
        double best_pt[3] = {0,0,0};
        int best_arc = -1;

        for (int a = 0; a < n_arcs; a++) {
            int ci = ai_r(a);
            double cx = cc_r(ci,0), cy = cc_r(ci,1), cz = cc_r(ci,2);
            double R  = cr_r(ci);
            double ux = cu_r(ci,0), uy = cu_r(ci,1), uz = cu_r(ci,2);
            double vx = cv_r(ci,0), vy = cv_r(ci,1), vz = cv_r(ci,2);

            double dx = px-cx, dy = py_-cy, dz = pz-cz;
            double proj_u = dx*ux + dy*uy + dz*uz;
            double proj_v = dx*vx + dy*vy + dz*vz;

            double t = std::atan2(proj_v, proj_u);
            if (t < 0) t += TWO_PI;

            double t_s = as_r(a), t_e = ae_r(a);
            double dt  = std::fmod(t - t_s, TWO_PI);
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
            double qx = cx + R*(cost*ux + sint*vx);
            double qy = cy + R*(cost*uy + sint*vy);
            double qz = cz + R*(cost*uz + sint*vz);

            double dist2 = (px-qx)*(px-qx) + (py_-qy)*(py_-qy) + (pz-qz)*(pz-qz);
            if (dist2 < best_dist2) {
                best_dist2 = dist2;
                best_pt[0] = qx; best_pt[1] = qy; best_pt[2] = qz;
                best_arc = a;
            }
        }

        oc_r(i,0) = best_pt[0];
        oc_r(i,1) = best_pt[1];
        oc_r(i,2) = best_pt[2];
        od_r(i) = std::sqrt(best_dist2);
        oa_r(i) = best_arc;
    }

    return {out_closest, out_dist, out_arcidx};
}

/* ═══════════════════════════════════════════════════════════════════
 * sample_arcs
 *
 * Python signature:
 *   sample_arcs(sphere_center(3,), sphere_radius,
 *               cap_centers(K,3), cap_radii(K,), cap_u(K,3), cap_v(K,3),
 *               arc_cap_idx(M,), arc_start(M,), arc_end(M,), n_total_samples)
 *   -> (points(S,3), arc_indices(S,))
 * ═══════════════════════════════════════════════════════════════════ */
std::tuple<py::array_t<double>, py::array_t<int32_t>>
sample_arcs(
    py::array_t<double>  py_cc,
    py::array_t<double>  py_cr,
    py::array_t<double>  py_cu,
    py::array_t<double>  py_cv,
    py::array_t<int32_t> py_ai,
    py::array_t<double>  py_as,
    py::array_t<double>  py_ae,
    int n_total)
{
    auto cc_r = py_cc.unchecked<2>();
    auto cr_r = py_cr.unchecked<1>();
    auto cu_r = py_cu.unchecked<2>();
    auto cv_r = py_cv.unchecked<2>();
    auto ai_r = py_ai.unchecked<1>();
    auto as_r = py_as.unchecked<1>();
    auto ae_r = py_ae.unchecked<1>();

    int n_arcs = (int)ai_r.shape(0);

    if (n_arcs == 0) {
        return {make_f64_2d(0, 3), make_i32_1d(0)};
    }

    // Arc lengths proportional to actual 3D arc length
    std::vector<double> arc_len(n_arcs);
    double total_len = 0;
    for (int a = 0; a < n_arcs; a++) {
        arc_len[a] = (ae_r(a) - as_r(a)) * cr_r(ai_r(a));
        total_len += arc_len[a];
    }

    // Distribute samples: at least 1 per arc
    std::vector<int> n_samples(n_arcs, 1);
    int remaining = n_total - n_arcs;
    if (remaining > 0 && total_len > 0) {
        for (int a = 0; a < n_arcs; a++)
            n_samples[a] += (int)(remaining * arc_len[a] / total_len);

        int assigned = 0;
        for (int a = 0; a < n_arcs; a++) assigned += n_samples[a];
        int leftover = n_total - assigned;

        std::vector<int> order(n_arcs);
        for (int i = 0; i < n_arcs; i++) order[i] = i;
        std::sort(order.begin(), order.end(),
                  [&](int a, int b) { return arc_len[a] > arc_len[b]; });
        for (int i = 0; i < leftover && i < n_arcs; i++)
            n_samples[order[i]]++;
    }

    int total_samples = 0;
    for (int a = 0; a < n_arcs; a++) total_samples += n_samples[a];

    auto py_spts = make_f64_2d(total_samples, 3);
    auto py_sidx = make_i32_1d(total_samples);
    auto spts_r  = py_spts.mutable_unchecked<2>();
    auto sidx_r  = py_sidx.mutable_unchecked<1>();

    int si = 0;
    for (int a = 0; a < n_arcs; a++) {
        int ci = ai_r(a);
        double cx = cc_r(ci,0), cy = cc_r(ci,1), cz = cc_r(ci,2);
        double R  = cr_r(ci);
        double ux = cu_r(ci,0), uy = cu_r(ci,1), uz = cu_r(ci,2);
        double vx = cv_r(ci,0), vy = cv_r(ci,1), vz = cv_r(ci,2);
        double t_s = as_r(a), t_e = ae_r(a);
        int ns = n_samples[a];

        for (int s = 0; s < ns; s++) {
            double t = t_s + (t_e - t_s) * (s + 0.5) / ns;
            double cost = std::cos(t), sint = std::sin(t);
            spts_r(si, 0) = cx + R*(cost*ux + sint*vx);
            spts_r(si, 1) = cy + R*(cost*uy + sint*vy);
            spts_r(si, 2) = cz + R*(cost*uz + sint*vz);
            sidx_r(si) = a;
            si++;
        }
    }

    return {py_spts, py_sidx};
}

/* ═══════════════════════════════════════════════════════════════════
 * Module definition — pybind11 style
 * ═══════════════════════════════════════════════════════════════════ */
PYBIND11_MODULE(sphere_exposed_pybind, m) {
    m.doc() = "Sphere exposed region computation + query functions (pybind11)";

    m.def("compute_exposed_single", &compute_exposed_single,
        py::arg("center"),
        py::arg("radius"),
        py::arg("other_centers"),
        py::arg("other_radii"),
        py::arg("tol")        = 1e-8,
        py::arg("degen_tol")  = 1e-6,
        py::arg("merge_tol")  = 1e-12,
        R"doc(
compute_exposed_single(center, radius, other_centers, other_radii,
                       tol=1e-8, degen_tol=1e-6, merge_tol=1e-12) -> dict

Returns dict with keys:
  arcs_by_cap      : {int: [(start, end), ...]}  compacted cap indices 0..K-1
  exposed_points   : [array(3,), ...]             degenerate exposed points
  total_arc        : float
  n_caps           : int
  cap_normals      : (K,3) float64
  cap_d            : (K,)  float64
  cap_centers      : (K,3) float64
  cap_radii        : (K,)  float64
  cap_u            : (K,3) float64
  cap_v            : (K,3) float64
  all_cap_normals  : (N,3) float64   all caps (for query_inside)
  all_cap_d        : (N,)  float64
  arc_cap_idx      : (M,)  int32
  arc_start        : (M,)  float64
  arc_end          : (M,)  float64
)doc");

    m.def("compute_exposed_batch", &compute_exposed_batch,
        py::arg("centers"),
        py::arg("radii"),
        py::arg("nbr_indices"),
        py::arg("nbr_offsets"),
        py::arg("tol")        = 1e-8,
        py::arg("degen_tol")  = 1e-6,
        py::arg("merge_tol")  = 1e-12,
        R"doc(
compute_exposed_batch(centers, radii, nbr_indices, nbr_offsets,
                      tol=1e-8, degen_tol=1e-6, merge_tol=1e-12) -> dict

CSR input. arc_cap_idx values are compacted per-sphere (no empty caps).
Returns dict with keys:
  n_caps, n_arcs, n_points, total_arc   : (N,) summary arrays
  arc_sphere_idx, arc_cap_idx,
    arc_start, arc_end                  : flat arc arrays
  point_sphere_idx, point_positions     : flat degen-point arrays
  cap_sphere_idx, cap_id,
    cap_normals, cap_d, cap_centers,
    cap_radii, cap_u, cap_v             : flat cap geometry arrays
)doc");

    m.def("query_inside", &query_inside,
        py::arg("points"),
        py::arg("all_cap_normals"),
        py::arg("all_cap_d"),
        R"doc(
query_inside(points(N,3), all_cap_normals(M,3), all_cap_d(M,)) -> bool(N,)

Test whether each point is in the exposed region (not inside any cap).
)doc");

    m.def("query_closest_on_arcs", &query_closest_on_arcs,
        py::arg("points"),
        py::arg("cap_centers"),
        py::arg("cap_radii"),
        py::arg("cap_u"),
        py::arg("cap_v"),
        py::arg("arc_cap_idx"),
        py::arg("arc_start"),
        py::arg("arc_end"),
        R"doc(
query_closest_on_arcs(points(N,3),
  cap_centers(K,3), cap_radii(K,), cap_u(K,3), cap_v(K,3),
  arc_cap_idx(M,), arc_start(M,), arc_end(M,))
  -> (closest(N,3), distances(N,), arc_indices(N,))

Find the closest point on the exposed boundary arcs for each query point.
)doc");

    m.def("sample_arcs", &sample_arcs,
        py::arg("cap_centers"),
        py::arg("cap_radii"),
        py::arg("cap_u"),
        py::arg("cap_v"),
        py::arg("arc_cap_idx"),
        py::arg("arc_start"),
        py::arg("arc_end"),
        py::arg("n_total_samples"),
        R"doc(
sample_arcs(
  cap_centers(K,3), cap_radii(K,), cap_u(K,3), cap_v(K,3),
  arc_cap_idx(M,), arc_start(M,), arc_end(M,), n_total_samples)
  -> (points(S,3), arc_indices(S,))

Distribute samples across arcs proportional to arc length (≥1 per arc).
)doc");
}