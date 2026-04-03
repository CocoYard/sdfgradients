/*
 * sphere_exposed_cpp.cpp
 *
 * C++ reimplementation of the Python incremental clipping algorithm for
 * computing exposed regions on a sphere.
 *
 * Algorithm (per sphere):
 *   For each cap added incrementally:
 *     1. Clip all existing boundary arcs against the new cap
 *     2. Compute the new cap's own exposed arcs (half-plane intersection)
 *     3. Append newly exposed arcs to the boundary
 *   After all caps processed:
 *     4. Compact: remove caps with no arcs, remap arc cap indices to 0..K-1
 *
 * Compile (Linux):
 *   g++ -O3 -shared -fPIC -std=c++17 \
 *     -o sphere_exposed_cpp$(python3-config --extension-suffix) \
 *     sphere_exposed_cpp.cpp \
 *     -I$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))") \
 *     -I$(python3 -c "import numpy; print(numpy.get_include())") \
 *     $(python3-config --ldflags) -lm
 *
 * Compile (macOS):
 *   g++ -O3 -shared -fPIC -undefined dynamic_lookup -std=c++17 \
 *     -o sphere_exposed_cpp$(python3-config --extension-suffix) \
 *     sphere_exposed_cpp.cpp \
 *     -I$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))") \
 *     -I$(python3 -c "import numpy; print(numpy.get_include())") -lm
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <cmath>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <algorithm>

/* ═══════════════════════════════════════════════════════════════════
 * Constants
 * ═══════════════════════════════════════════════════════════════════ */
static constexpr double PI     = 3.14159265358979323846;
static constexpr double TWO_PI = 2.0 * PI;
static constexpr double EPS    = 1e-14;   // zero-length / normalization
static constexpr double EPS12  = 1e-12;   // is_inside_cap
static constexpr double EPS10  = 1e-10;   // phi >= pi check, ratio bounds

/* Stack limits */
static constexpr int MAX_CAPS      = 512;
static constexpr int MAX_ARCS      = 2048;
static constexpr int MAX_INTERVALS = 512;
static constexpr int MAX_DEGEN_PTS = 256;

/* ═══════════════════════════════════════════════════════════════════
 * Tolerance parameters (passed through the call chain)
 * ═══════════════════════════════════════════════════════════════════ */
struct Tolerances {
    double tol;        // skip_tol for interval intersection & parallel cap check
    double degen_tol;  // total_arc threshold to trigger degen point search
    double merge_tol;  // merge_tol for interval merging & degen point dedup
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
 * Geometry helpers
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

/* Build all caps, deduplicating near-parallel ones. Uses tol for dedup. */
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
                else if (cap.d > caps[e].d) { dup = true; break; }
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

/* Is point inside (covered by) cap? Uses EPS12 = 1e-12 */
static inline bool is_inside_cap(Vec3 pt, const Cap &cap) {
    return dot(cap.normal, pt) - cap.d > EPS12;
}

/* Is angle t inside arc [t_start, t_end)? Uses EPS = 1e-14 */
static inline bool angle_in_arc(double t, double t_start, double t_end) {
    t = fmod_pos(t, TWO_PI);
    double dt = fmod_pos(t - t_start, TWO_PI);
    double arc_len = t_end - t_start;
    return dt < arc_len - EPS;
}

/* ═══════════════════════════════════════════════════════════════════
 * Circle-plane intersection
 * ═══════════════════════════════════════════════════════════════════ */
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

/* ═══════════════════════════════════════════════════════════════════
 * Clip a single boundary arc by a cutting cap
 * ═══════════════════════════════════════════════════════════════════ */
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

/* ═══════════════════════════════════════════════════════════════════
 * Interval intersection with configurable tolerances
 * ═══════════════════════════════════════════════════════════════════ */
static int intersect_intervals(const Interval *a, int na,
                               const Interval *b, int nb,
                               Interval *out, int max_out,
                               double skip_tol, double merge_tol) {
    int cnt = 0;
    for (int i = 0; i < na; i++) {
        for (int j = 0; j < nb; j++) {
            double lo = std::max(a[i].start, b[j].start);
            double hi = std::min(a[i].end, b[j].end);
            if (hi - lo > -skip_tol && cnt < max_out) {
                if (hi < lo) {
                    double mid = (lo + hi) / 2.0;
                    out[cnt++] = {mid - skip_tol/2, mid + skip_tol/2};
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

/* ═══════════════════════════════════════════════════════════════════
 * Compute exposed arcs on one cap's circle via half-plane intersection
 * ═══════════════════════════════════════════════════════════════════ */
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
            if (c > tol.tol) return 0;  // parallel cap eats this one
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

/* ═══════════════════════════════════════════════════════════════════
 * Degenerate point detection
 * ═══════════════════════════════════════════════════════════════════ */

/* is_pt_exposed uses tol.tol as the "inside" threshold */
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

/* Find degenerate points; uses degen_tol for on-sphere check and dedup */
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
 * compute_one: incremental clipping for one sphere.
 *
 * After computing all arcs, caps with no arcs are removed and
 * arc cap indices are remapped to the compacted range 0..K-1.
 * out_ncaps returns K (only caps that have arcs).
 * ═══════════════════════════════════════════════════════════════════ */
static void compute_one(
    Vec3 mc, double mr, const double *oc, const double *or_, int n_others,
    const Tolerances &tol,
    /* outputs */
    int *out_ncaps,
    int *out_narcs, int *arc_cap, double *arc_s, double *arc_e, int max_arcs,
    double *out_total_arc,
    int *out_npts, Vec3 *out_pts)
{
    Cap caps[MAX_CAPS];
    int nc = compute_all_caps(mc, mr, oc, or_, n_others, caps, tol.tol);
    *out_ncaps = 0;
    *out_total_arc = 0;
    *out_narcs = 0;
    *out_npts = 0;

    if (nc == 0) return;

    /* Check if any cap covers the entire sphere */
    for (int i = 0; i < nc; i++) {
        if (caps[i].phi >= PI - EPS10) return;
    }

    /* Filter to valid caps (non-zero circle radius) */
    int valid_caps[MAX_CAPS];
    int n_valid = 0;
    for (int i = 0; i < nc; i++) {
        if (caps[i].circle_radius >= EPS)
            valid_caps[n_valid++] = i;
    }
    if (n_valid == 0) return;

    /* Incremental clipping */
    std::vector<BoundaryArc> all_arcs;
    all_arcs.reserve(n_valid * 4);

    int active_caps[MAX_CAPS];
    int n_active = 0;

    for (int vi = 0; vi < n_valid; vi++) {
        int cap_idx = valid_caps[vi];

        /* Clip existing arcs by the new cap */
        std::vector<BoundaryArc> new_arcs;
        new_arcs.reserve(all_arcs.size() + 4);
        for (const auto &arc : all_arcs) {
            clip_arc_by_cap(arc, caps[cap_idx], caps, new_arcs);
        }

        /* Compute exposed arcs on the new cap's circle */
        Interval new_intervals[MAX_INTERVALS];
        int n_new = compute_exposed_arcs_on_circle(cap_idx, caps,
                                                   active_caps, n_active,
                                                   new_intervals, tol);

        active_caps[n_active++] = cap_idx;

        for (int a = 0; a < n_new; a++) {
            new_arcs.push_back({cap_idx, new_intervals[a].start, new_intervals[a].end});
        }

        all_arcs = std::move(new_arcs);
    }

    /* ── Compact: remove caps with no arcs, remap cap indices ────── */

    /* Count arcs per original cap index */
    int arc_count[MAX_CAPS] = {};  // zero-init
    for (const auto &arc : all_arcs) {
        if (arc.cap_idx < MAX_CAPS) arc_count[arc.cap_idx]++;
    }

    /* Build old→new index map; -1 means cap has no arcs (removed) */
    int remap[MAX_CAPS];
    int kept = 0;
    for (int i = 0; i < nc; i++) {
        if (arc_count[i] > 0) remap[i] = kept++;
        else                  remap[i] = -1;
    }

    /* Output arcs with remapped cap indices */
    double total = 0;
    int na = 0;
    for (const auto &arc : all_arcs) {
        if (na >= max_arcs) break;
        arc_cap[na] = remap[arc.cap_idx];
        arc_s[na] = arc.t_start;
        arc_e[na] = arc.t_end;
        total += arc.t_end - arc.t_start;
        na++;
    }
    *out_narcs = na;
    *out_ncaps = kept;
    *out_total_arc = total;

    /* Degenerate point detection if total arc length is near zero */
    if (total < tol.degen_tol && nc > 0) {
        *out_npts = find_degen_pts(mc, mr, caps, nc, out_pts, tol);
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * Python API: compute_exposed_single(center, radius, other_centers,
 *             other_radii, tol=1e-8, degen_tol=1e-6, merge_tol=1e-12)
 *
 * Returns dict with:
 *   arcs_by_cap: {0: [(s,e),...], 1: [...], ...}  (compacted, no empty caps)
 *   exposed_points: [array([x,y,z]), ...]
 *   total_arc: float
 *   n_caps: int (only caps that have arcs)
 * ═══════════════════════════════════════════════════════════════════ */
static PyObject* py_compute_single(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyArrayObject *py_mc, *py_oc, *py_or;
    double mr;
    double tol_v = 1e-8, degen_tol_v = 1e-6, merge_tol_v = 1e-12;

    static const char *kwlist[] = {
        "center", "radius", "other_centers", "other_radii",
        "tol", "degen_tol", "merge_tol", NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!dO!O!|ddd",
            const_cast<char**>(kwlist),
            &PyArray_Type, &py_mc, &mr,
            &PyArray_Type, &py_oc, &PyArray_Type, &py_or,
            &tol_v, &degen_tol_v, &merge_tol_v))
        return NULL;

    Tolerances tol = {tol_v, degen_tol_v, merge_tol_v};

    int n = (int)PyArray_DIM(py_oc, 0);
    double *mc = (double*)PyArray_DATA(py_mc);

    int ncaps, narcs, npts;
    double total_arc;
    int arc_cap_buf[MAX_ARCS];
    double arc_s_buf[MAX_ARCS], arc_e_buf[MAX_ARCS];
    Vec3 dpts[MAX_DEGEN_PTS];

    compute_one({mc[0], mc[1], mc[2]}, mr,
                (double*)PyArray_DATA(py_oc), (double*)PyArray_DATA(py_or), n,
                tol,
                &ncaps, &narcs, arc_cap_buf, arc_s_buf, arc_e_buf, MAX_ARCS,
                &total_arc, &npts, dpts);

    /* Build arcs_by_cap dict with compacted indices 0..ncaps-1.
     * Every key in the dict is guaranteed to have at least one arc. */
    PyObject *ad = PyDict_New();
    for (int i = 0; i < ncaps; i++) {
        PyObject *k = PyLong_FromLong(i);
        PyObject *l = PyList_New(0);
        PyDict_SetItem(ad, k, l);
        Py_DECREF(k); Py_DECREF(l);
    }
    for (int a = 0; a < narcs; a++) {
        PyObject *k = PyLong_FromLong(arc_cap_buf[a]);
        PyObject *l = PyDict_GetItem(ad, k);
        PyObject *t = Py_BuildValue("(dd)", arc_s_buf[a], arc_e_buf[a]);
        PyList_Append(l, t);
        Py_DECREF(t); Py_DECREF(k);
    }

    /* Build degenerate points list */
    PyObject *pl = PyList_New(npts);
    for (int i = 0; i < npts; i++) {
        npy_intp d[1] = {3};
        PyObject *p = PyArray_SimpleNew(1, d, NPY_DOUBLE);
        double *pd = (double*)PyArray_DATA((PyArrayObject*)p);
        pd[0] = dpts[i].x; pd[1] = dpts[i].y; pd[2] = dpts[i].z;
        PyList_SET_ITEM(pl, i, p);
    }

    return Py_BuildValue("{s:O,s:O,s:d,s:i}",
                         "arcs_by_cap", ad,
                         "exposed_points", pl,
                         "total_arc", total_arc,
                         "n_caps", ncaps);
}

/* ═══════════════════════════════════════════════════════════════════
 * Python API: compute_exposed_batch(centers, radii, nbr_indices,
 *             nbr_offsets, tol=1e-8, degen_tol=1e-6, merge_tol=1e-12)
 *
 * CSR input, compact numpy output.
 * arc_cap_idx values are compacted per-sphere (no empty caps).
 * ═══════════════════════════════════════════════════════════════════ */
static PyObject* py_compute_batch(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyArrayObject *py_c, *py_r, *py_idx, *py_off;
    double tol_v = 1e-8, degen_tol_v = 1e-6, merge_tol_v = 1e-12;

    static const char *kwlist[] = {
        "centers", "radii", "nbr_indices", "nbr_offsets",
        "tol", "degen_tol", "merge_tol", NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!O!O!|ddd",
            const_cast<char**>(kwlist),
            &PyArray_Type, &py_c, &PyArray_Type, &py_r,
            &PyArray_Type, &py_idx, &PyArray_Type, &py_off,
            &tol_v, &degen_tol_v, &merge_tol_v))
        return NULL;

    Tolerances tol = {tol_v, degen_tol_v, merge_tol_v};

    int N = (int)PyArray_DIM(py_c, 0);
    double *centers = (double*)PyArray_DATA(py_c);
    double *radii   = (double*)PyArray_DATA(py_r);
    int64_t *nbr_idx = (int64_t*)PyArray_DATA(py_idx);
    int64_t *nbr_off = (int64_t*)PyArray_DATA(py_off);

    /* Per-sphere summary arrays */
    npy_intp dims_n[1] = {N};
    PyObject *py_ncaps = PyArray_SimpleNew(1, dims_n, NPY_INT32);
    PyObject *py_narcs = PyArray_SimpleNew(1, dims_n, NPY_INT32);
    PyObject *py_npts  = PyArray_SimpleNew(1, dims_n, NPY_INT32);
    PyObject *py_tarc  = PyArray_SimpleNew(1, dims_n, NPY_FLOAT64);
    int32_t *s_ncaps = (int32_t*)PyArray_DATA((PyArrayObject*)py_ncaps);
    int32_t *s_narcs = (int32_t*)PyArray_DATA((PyArrayObject*)py_narcs);
    int32_t *s_npts  = (int32_t*)PyArray_DATA((PyArrayObject*)py_npts);
    double  *s_tarc  = (double*)PyArray_DATA((PyArrayObject*)py_tarc);

    /* Growable flat arc storage */
    int arcs_alloc = N * 4;
    int total_arcs = 0;
    int    *fa_sphere = (int*)malloc(arcs_alloc * sizeof(int));
    int    *fa_cap    = (int*)malloc(arcs_alloc * sizeof(int));
    double *fa_s      = (double*)malloc(arcs_alloc * sizeof(double));
    double *fa_e      = (double*)malloc(arcs_alloc * sizeof(double));

    /* Growable flat point storage */
    int pts_alloc = 256;
    int total_pts = 0;
    int    *fp_sphere = (int*)malloc(pts_alloc * sizeof(int));
    double *fp_pos    = (double*)malloc(pts_alloc * 3 * sizeof(double));

    /* Per-sphere scratch */
    int    _ac[MAX_ARCS];
    double _as[MAX_ARCS], _ae[MAX_ARCS];
    Vec3   _dp[MAX_DEGEN_PTS];

    /* Neighbor gather buffer (reused across iterations) */
    int oc_alloc = 1024;
    double *oc_buf = (double*)malloc(oc_alloc * 3 * sizeof(double));
    double *or_buf = (double*)malloc(oc_alloc * sizeof(double));

    for (int i = 0; i < N; i++) {
        int n_nbrs = (int)(nbr_off[i+1] - nbr_off[i]);
        if (n_nbrs == 0) {
            s_ncaps[i] = 0; s_narcs[i] = 0; s_npts[i] = 0; s_tarc[i] = 0;
            continue;
        }

        if (n_nbrs > oc_alloc) {
            oc_alloc = n_nbrs * 2;
            oc_buf = (double*)realloc(oc_buf, oc_alloc * 3 * sizeof(double));
            or_buf = (double*)realloc(or_buf, oc_alloc * sizeof(double));
        }

        const int64_t *nb = nbr_idx + nbr_off[i];
        for (int j = 0; j < n_nbrs; j++) {
            int64_t idx = nb[j];
            oc_buf[j*3]   = centers[idx*3];
            oc_buf[j*3+1] = centers[idx*3+1];
            oc_buf[j*3+2] = centers[idx*3+2];
            or_buf[j] = radii[idx];
        }

        int ncaps, narcs, npts;
        double total_arc;
        compute_one({centers[i*3], centers[i*3+1], centers[i*3+2]}, radii[i],
                    oc_buf, or_buf, n_nbrs, tol,
                    &ncaps, &narcs, _ac, _as, _ae, MAX_ARCS,
                    &total_arc, &npts, _dp);

        s_ncaps[i] = ncaps;
        s_narcs[i] = narcs;
        s_npts[i]  = npts;
        s_tarc[i]  = total_arc;

        /* Append arcs */
        while (total_arcs + narcs > arcs_alloc) {
            arcs_alloc *= 2;
            fa_sphere = (int*)realloc(fa_sphere, arcs_alloc * sizeof(int));
            fa_cap    = (int*)realloc(fa_cap,    arcs_alloc * sizeof(int));
            fa_s      = (double*)realloc(fa_s,   arcs_alloc * sizeof(double));
            fa_e      = (double*)realloc(fa_e,   arcs_alloc * sizeof(double));
        }
        for (int a = 0; a < narcs; a++) {
            fa_sphere[total_arcs] = i;
            fa_cap[total_arcs]    = _ac[a];
            fa_s[total_arcs]      = _as[a];
            fa_e[total_arcs]      = _ae[a];
            total_arcs++;
        }

        /* Append points */
        while (total_pts + npts > pts_alloc) {
            pts_alloc *= 2;
            fp_sphere = (int*)realloc(fp_sphere, pts_alloc * sizeof(int));
            fp_pos    = (double*)realloc(fp_pos,  pts_alloc * 3 * sizeof(double));
        }
        for (int p = 0; p < npts; p++) {
            fp_sphere[total_pts] = i;
            fp_pos[total_pts*3]   = _dp[p].x;
            fp_pos[total_pts*3+1] = _dp[p].y;
            fp_pos[total_pts*3+2] = _dp[p].z;
            total_pts++;
        }
    }

    free(oc_buf);
    free(or_buf);

    /* Wrap C arrays into numpy arrays */
    auto wrap_int = [](int *ptr, int len) -> PyObject* {
        npy_intp d[1] = {len};
        PyObject *a = PyArray_SimpleNew(1, d, NPY_INT32);
        memcpy(PyArray_DATA((PyArrayObject*)a), ptr, len * sizeof(int32_t));
        free(ptr);
        return a;
    };
    auto wrap_dbl = [](double *ptr, int len) -> PyObject* {
        npy_intp d[1] = {len};
        PyObject *a = PyArray_SimpleNew(1, d, NPY_FLOAT64);
        memcpy(PyArray_DATA((PyArrayObject*)a), ptr, len * sizeof(double));
        free(ptr);
        return a;
    };

    PyObject *fa_sphere_arr = wrap_int(fa_sphere, total_arcs);
    PyObject *fa_cap_arr    = wrap_int(fa_cap, total_arcs);
    PyObject *fa_s_arr      = wrap_dbl(fa_s, total_arcs);
    PyObject *fa_e_arr      = wrap_dbl(fa_e, total_arcs);
    PyObject *fp_sphere_arr = wrap_int(fp_sphere, total_pts);

    npy_intp pd[2] = {total_pts, 3};
    PyObject *fp_pos_arr = PyArray_SimpleNew(2, pd, NPY_FLOAT64);
    memcpy(PyArray_DATA((PyArrayObject*)fp_pos_arr), fp_pos, total_pts * 3 * sizeof(double));
    free(fp_pos);

    return Py_BuildValue(
        "{s:O,s:O,s:O,s:O, s:O,s:O,s:O,s:O, s:O,s:O}",
        "n_caps", py_ncaps, "n_arcs", py_narcs, "n_points", py_npts, "total_arc", py_tarc,
        "arc_sphere_idx", fa_sphere_arr, "arc_cap_idx", fa_cap_arr,
        "arc_start", fa_s_arr, "arc_end", fa_e_arr,
        "point_sphere_idx", fp_sphere_arr, "point_positions", fp_pos_arr);
}

/* ═══════════════════════════════════════════════════════════════════
 * Module definition
 * ═══════════════════════════════════════════════════════════════════ */
static PyMethodDef methods[] = {
    {"compute_exposed_single", (PyCFunction)py_compute_single, METH_VARARGS | METH_KEYWORDS,
     "compute_exposed_single(center, radius, other_centers, other_radii,\n"
     "                       tol=1e-8, degen_tol=1e-6, merge_tol=1e-12) -> dict\n"
     "  Returns {arcs_by_cap, exposed_points, total_arc, n_caps}\n"
     "  Cap indices are compacted: only caps with arcs, numbered 0..K-1."},
    {"compute_exposed_batch", (PyCFunction)py_compute_batch, METH_VARARGS | METH_KEYWORDS,
     "compute_exposed_batch(centers, radii, nbr_indices, nbr_offsets,\n"
     "                      tol=1e-8, degen_tol=1e-6, merge_tol=1e-12) -> dict\n"
     "  CSR input. Cap indices in arc_cap_idx are compacted per-sphere."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "sphere_exposed_cpp",
    "Sphere exposed region computation (C++ incremental clipping).\n"
    "Tolerances: tol (skip/parallel/dedup), degen_tol (arc->point threshold), merge_tol (interval merge).",
    -1, methods
};

PyMODINIT_FUNC PyInit_sphere_exposed_cpp(void) {
    import_array();
    return PyModule_Create(&module);
}