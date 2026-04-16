/// Adapter that wraps the core computation logic from existing pybind modules
/// (sphere_intersect.cpp, sphere_exposed_pybind.cpp) so our pure C++ code
/// can call them without pybind11.

#include "types.h"
#include <vector>
#include <cmath>
#include <cstring>
#include <cfloat>
#include <algorithm>
#include <unordered_map>
#include <cstdio>
#include <atomic>
#include <climits>

#ifdef USE_OPENMP
#include <omp.h>
#endif

#if defined(__APPLE__)
  #include <mach/mach.h>
  static size_t mem_rss_bytes() {
      mach_task_basic_info_data_t info;
      mach_msg_type_number_t cnt = MACH_TASK_BASIC_INFO_COUNT;
      if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                    (task_info_t)&info, &cnt) != KERN_SUCCESS) return 0;
      return (size_t)info.resident_size;
  }
#elif defined(__linux__)
  #include <cstdio>
  static size_t mem_rss_bytes() {
      FILE* f = std::fopen("/proc/self/statm", "r");
      if (!f) return 0;
      long pages = 0, rss = 0;
      if (std::fscanf(f, "%ld %ld", &pages, &rss) != 2) { std::fclose(f); return 0; }
      std::fclose(f);
      return (size_t)rss * (size_t)sysconf(_SC_PAGESIZE);
  }
#else
  static size_t mem_rss_bytes() { return 0; }
#endif

#ifdef CHECK_MEM
#  define MEM_MARK(tag) std::fprintf(stderr, "[RSS] %-28s %7.2f GB\n", tag, mem_rss_bytes()/1.0e9)
#  define MEM_LOG(...) std::fprintf(stderr, __VA_ARGS__)
#else
#  define MEM_MARK(tag) ((void)0)
#  define MEM_LOG(...)  ((void)0)
#endif

// ═══════════════════════════════════════════════════════════════════
// sphere_exposed_core
//
// Full reimplementation of the per-sphere exposed region computation
// from bench/sphere_exposed_pybind.cpp, without pybind11 dependency.
// ═══════════════════════════════════════════════════════════════════

namespace sphere_exposed_core {

// ── Constants ────────────────────────────────────────────────────
static constexpr double PI     = 3.14159265358979323846;
static constexpr double TWO_PI = 2.0 * PI;
static constexpr double EPS    = 1e-14;
static constexpr double EPS12  = 1e-12;
static constexpr double EPS10  = 1e-10;

static constexpr int MAX_CAPS      = 512;
static constexpr int MAX_ARCS      = 2048;
static constexpr int MAX_INTERVALS = 512;
static constexpr int MAX_DEGEN_PTS = 256;

// ── Tolerances ───────────────────────────────────────────────────
struct Tolerances {
    double tol;
    double degen_tol;
    double merge_tol;
    double tangent_tol;
};

// ── Vec3 ─────────────────────────────────────────────────────────
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

// ── Data structures ──────────────────────────────────────────────
struct Cap {
    Vec3 normal;
    double d;
    Vec3 circle_center;
    double circle_radius;
    Vec3 local_u, local_v;
    double phi;
    int sphere_idx;
    double containment_gap;
};

struct Interval {
    double start, end;
};

struct BoundaryArc {
    int cap_idx;
    double t_start, t_end;
};

// ── Geometry helpers ─────────────────────────────────────────────

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
            double gap = or_ - mr;
            out = {n, dot(n,mc) - mr - 1, mc, 0, {}, {}, PI, idx, gap};
            return true;
        }
        return false;
    }
    if (dist >= mr + or_) return false;
    if (dist + or_ <= mr) return false;
    if (dist + mr <= or_) {
        Vec3 n = diff * (1.0/dist);
        double gap = or_ - dist - mr;
        out = {n, dot(n,mc) - mr - 1, mc, 0, {}, {}, PI, idx, gap};
        return true;
    }

    Vec3 n = diff * (1.0/dist);
    double h = (mr*mr - or_*or_ + dist*dist) / (2.0*dist);
    Vec3 cc = mc + h * n;
    double cr = std::sqrt(std::max(0.0, mr*mr - h*h));
    double phi = std::acos(clampd(h / mr, -1.0, 1.0));
    Vec3 u = perpendicular_unit(n);
    Vec3 v = cross(n, u);
    out = {n, dot(n, mc) + h, cc, cr, u, v, phi, idx, DBL_MAX};
    return true;
}

// Hard safety cap on how many neighbors the incremental loop in compute_one
// will scan per sphere. Our per-sphere loop prunes dead caps aggressively, but
// pathological cases (>30k neighbors, observed) would still waste time scanning
// neighbors that contribute nothing. Kept as a safety net.
static constexpr int MAX_NEIGHBORS_SCANNED = 2048;

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

// ── compute_one: per-sphere exposed region computation ───────────

// Single-pass incremental version: per neighbor, compute its cap, dedup,
// clip existing arcs, compute the new cap's own arcs, and prune caps whose
// arcs have all been eaten. Avoids the old two-phase pattern of building a
// full caps[] up-front and iterating again for arcs.
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
    Cap  caps[MAX_CAPS];
    int  nc = 0;
    int  arc_count[MAX_CAPS] = {};
    int  active_caps[MAX_CAPS];
    int  n_active = 0;
    bool in_active[MAX_CAPS] = {};

    std::vector<BoundaryArc> all_arcs;
    std::vector<BoundaryArc> new_arcs;
    all_arcs.reserve(16);
    new_arcs.reserve(16);

    *out_ncaps = 0;
    *out_narcs = 0;
    *out_npts = 0;
    *out_total_arc = 0;
    *out_nc = 0;

    int scan_limit = n_others < MAX_NEIGHBORS_SCANNED ? n_others : MAX_NEIGHBORS_SCANNED;

    for (int i = 0; i < scan_limit && nc < MAX_CAPS; i++) {
        Cap cap;
        if (!compute_cap(mc, mr, {oc[i*3], oc[i*3+1], oc[i*3+2]}, or_[i], i, cap))
            continue;

        // Containment (φ ≥ π): this neighbor swallows us. Emit tangent point if
        // applicable and bail — exposed region is empty.
        if (cap.phi >= PI - EPS10) {
            if (cap.containment_gap <= tol.tangent_tol && *out_npts < MAX_DEGEN_PTS) {
                out_pts[(*out_npts)++] = mc - mr * cap.normal;
            }
            *out_ncaps = 0;
            *out_narcs = 0;
            *out_nc = 0;
            return;
        }

        // Dedup against existing caps (live and frozen small-radius ones),
        // matching the original compute_all_caps logic. Dominating new replaces
        // the old in-place; duplicates are skipped.
        bool dup = false;
        int replaced = -1;
        for (int e = nc - 1; e >= 0; e--) {
            double d = dot(cap.normal, caps[e].normal);
            if (d > 1 - tol.tol) {
                if (std::fabs(cap.d - caps[e].d) < tol.tol) { dup = true; break; }
                else if (cap.d > caps[e].d)                 { dup = true; break; }
                else { replaced = e; break; }
            }
        }
        if (dup) continue;

        int new_idx;
        if (replaced >= 0) {
            new_idx = replaced;
            caps[new_idx] = cap;
            // Drop any arcs hosted by the replaced cap; the new cap will compute
            // its own arcs below.
            if (arc_count[new_idx] > 0) {
                all_arcs.erase(std::remove_if(all_arcs.begin(), all_arcs.end(),
                    [&](const BoundaryArc& a){ return a.cap_idx == new_idx; }),
                    all_arcs.end());
                arc_count[new_idx] = 0;
            }
        } else {
            new_idx = nc++;
            caps[new_idx] = cap;
        }

        // Frozen small-radius caps: kept in caps[] so find_degen_pts can see
        // them, but never participate in arc clipping.
        if (cap.circle_radius < EPS) {
            if (replaced >= 0 && in_active[new_idx]) {
                // The replaced slot was active; remove it.
                for (int k = 0; k < n_active; k++) {
                    if (active_caps[k] == new_idx) {
                        active_caps[k] = active_caps[--n_active];
                        break;
                    }
                }
                in_active[new_idx] = false;
            }
            continue;
        }

        // Clip all existing arcs by the new cap.
        new_arcs.clear();
        for (const auto& arc : all_arcs)
            clip_arc_by_cap(arc, caps[new_idx], caps, new_arcs);

        // Compute the new cap's own exposed arcs against currently active caps.
        // compute_exposed_arcs_on_circle skips new_idx internally if present.
        Interval ivs[MAX_INTERVALS];
        int n_new = compute_exposed_arcs_on_circle(new_idx, caps,
                                                   active_caps, n_active,
                                                   ivs, tol);
        for (int a = 0; a < n_new; a++)
            new_arcs.push_back({new_idx, ivs[a].start, ivs[a].end});

        all_arcs.swap(new_arcs);

        // Rebuild arc_count from the surviving arcs.
        std::memset(arc_count, 0, sizeof(int) * nc);
        for (const auto& a : all_arcs) arc_count[a.cap_idx]++;

        if (!in_active[new_idx]) {
            active_caps[n_active++] = new_idx;
            in_active[new_idx] = true;
        }

        // Prune caps whose arcs have all been eaten.
        int w = 0;
        for (int k = 0; k < n_active; k++) {
            int idx = active_caps[k];
            if (arc_count[idx] > 0) {
                active_caps[w++] = idx;
            } else {
                in_active[idx] = false;
            }
        }
        n_active = w;
    }

    // Build output: compact caps into kept order.
    int remap[MAX_CAPS];
    int kept = 0;
    for (int i = 0; i < nc; i++) {
        if (arc_count[i] > 0) remap[i] = kept++;
        else                  remap[i] = -1;
    }

    *out_nc = nc;
    if (out_caps)  std::memcpy(out_caps, caps, nc * sizeof(Cap));
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

// ── compute_exposed_batch ────────────────────────────────────────

void compute_exposed_batch(
    const double* centers, const double* radii, int n,
    const std::vector<std::vector<int>>& nbrs,
    double tol_v, double degen_tol, double merge_tol, double tangent_tol,
    sdf::Options::BatchData& out)
{
    Tolerances tol{tol_v, degen_tol, merge_tol, tangent_tol};

    MEM_MARK("batch: enter");
    MEM_LOG("[RSS] batch n=%d\n", n);

    out.n_arcs.resize(n);
    out.n_points.resize(n);

    // Per-sphere local results (computed in parallel, then concatenated).
    struct SphereLocal {
        int narcs = 0, nkept_caps = 0, npts = 0;
        std::vector<int>    arc_cap;
        std::vector<double> arc_s, arc_e;
        std::vector<Cap>    caps;   // only kept (remapped) caps, in remap order
        std::vector<Vec3>   pts;
    };
    std::vector<SphereLocal> locals(n);
    MEM_MARK("after locals(n) alloc");

    // Neighbor list stats — catches pathological cases where some sphere
    // has huge n_nbrs, which blows up compute_one internally.
    {
        int max_nb = 0, min_nb = INT_MAX;
        long long sum_nb = 0, sum_sq = 0;
        int hist[8] = {0};  // <=8, <=32, <=128, <=512, <=2k, <=8k, <=32k, >32k
        int arg_max = -1;
        for (int i = 0; i < n; i++) {
            int k = (int)nbrs[i].size();
            if (k > max_nb) { max_nb = k; arg_max = i; }
            if (k < min_nb) min_nb = k;
            sum_nb += k;
            sum_sq += (long long)k * k;
            int b = 0;
            int t = 8;
            while (b < 7 && k > t) { b++; t *= 4; }
            hist[b]++;
        }
        double mean = (double)sum_nb / n;
        MEM_LOG("[RSS] nbr stats: min=%d max=%d(at i=%d) mean=%.1f"
            " total=%lld (%.2f GB as int)\n",
            min_nb, max_nb, arg_max, mean, sum_nb, sum_nb*4/1e9);
        MEM_LOG("[RSS] nbr hist (<=8,32,128,512,2k,8k,32k,>32k): "
            "%d %d %d %d %d %d %d %d\n",
            hist[0],hist[1],hist[2],hist[3],hist[4],hist[5],hist[6],hist[7]);
        // Rough upper bound if compute_one is O(n_nbrs^2) in memory:
        MEM_LOG("[RSS] sum(n_nbrs^2)=%lld (~%.2f GB as double)\n",
            sum_sq, sum_sq*8/1e9);
    }

    std::atomic<int> progress{0};
    size_t rss_before_loop = mem_rss_bytes();

    #pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < n; i++) {
        int n_nbrs = (int)nbrs[i].size();
        if (n_nbrs == 0) {
            out.n_arcs[i] = 0;
            out.n_points[i] = 0;
            continue;
        }

        // Thread-local scratch (stack-allocated).
        int    _ac[MAX_ARCS];
        double _as[MAX_ARCS], _ae[MAX_ARCS];
        Vec3   _dp[MAX_DEGEN_PTS];
        Cap    _caps[MAX_CAPS];
        int    remap[MAX_CAPS];

        // Thread-local reusable gather buffers — reallocating per iteration
        // (with n_nbrs up to ~40k) was the main cause of RSS bloat, since the
        // allocator holds the high-water mark and doesn't return pages to OS.
        // compute_all_caps will only scan MAX_NEIGHBORS_SCANNED anyway, so cap
        // the copy.
        static thread_local std::vector<double> oc_buf_v;
        static thread_local std::vector<double> or_buf_v;
        int n_use = n_nbrs < MAX_NEIGHBORS_SCANNED ? n_nbrs : MAX_NEIGHBORS_SCANNED;
        if ((int)or_buf_v.size() < n_use) {
            oc_buf_v.resize(n_use * 3);
            or_buf_v.resize(n_use);
        }
        const int* nb = nbrs[i].data();
        for (int j = 0; j < n_use; j++) {
            int idx = nb[j];
            oc_buf_v[j*3]   = centers[idx*3];
            oc_buf_v[j*3+1] = centers[idx*3+1];
            oc_buf_v[j*3+2] = centers[idx*3+2];
            or_buf_v[j]     = radii[idx];
        }

        int ncaps, narcs, npts, n_pcaps;
        double total_arc;
        compute_one(
            {centers[i*3], centers[i*3+1], centers[i*3+2]}, radii[i],
            oc_buf_v.data(), or_buf_v.data(), n_use, tol,
            &ncaps, &narcs, _ac, _as, _ae, MAX_ARCS,
            &total_arc, &npts, _dp,
            _caps, &n_pcaps, remap);

        out.n_arcs[i] = narcs;
        out.n_points[i] = npts;

        SphereLocal& L = locals[i];
        L.narcs = narcs;
        L.npts  = npts;

        // Collect kept caps in remap order (matches original serial append order).
        L.caps.reserve(ncaps);
        for (int c = 0; c < ncaps; c++) {
            int orig = -1;
            for (int k = 0; k < n_pcaps; k++) {
                if (remap[k] == c) { orig = k; break; }
            }
            if (orig < 0) continue;
            L.caps.push_back(_caps[orig]);
        }
        L.nkept_caps = (int)L.caps.size();

        L.arc_cap.assign(_ac, _ac + narcs);
        L.arc_s.assign(_as, _as + narcs);
        L.arc_e.assign(_ae, _ae + narcs);

        L.pts.assign(_dp, _dp + npts);

        int done = progress.fetch_add(1, std::memory_order_relaxed) + 1;
        if ((done & 4095) == 0) {
            #pragma omp critical
            {
                MEM_LOG("[RSS] loop progress %d/%d  (i=%d n_nbrs=%d ncaps=%d narcs=%d)"
                    "  RSS=%.2f GB (+%.2f)\n",
                    done, n, i, n_nbrs, ncaps, narcs,
                    mem_rss_bytes()/1e9,
                    (mem_rss_bytes() - rss_before_loop)/1e9);
            }
        }
    }

    MEM_MARK("after parallel compute");

    // Tally locals capacity to see how much the intermediate copy costs.
    {
        size_t b_arc=0, b_caps=0, b_pts=0;
        size_t tot_arcs=0, tot_caps=0, tot_pts=0;
        for (int i = 0; i < n; i++) {
            const SphereLocal& L = locals[i];
            b_arc  += L.arc_cap.capacity()*sizeof(int)
                    + L.arc_s.capacity()*sizeof(double)
                    + L.arc_e.capacity()*sizeof(double);
            b_caps += L.caps.capacity()*sizeof(Cap);
            b_pts  += L.pts.capacity()*sizeof(Vec3);
            tot_arcs += L.narcs;
            tot_caps += L.nkept_caps;
            tot_pts  += L.npts;
        }
        MEM_LOG("[RSS] locals bytes: arcs=%.2fGB caps=%.2fGB pts=%.2fGB"
            " | totals arcs=%zu caps=%zu pts=%zu\n",
            b_arc/1e9, b_caps/1e9, b_pts/1e9, tot_arcs, tot_caps, tot_pts);
        MEM_LOG("[RSS] out.* projected: cap_mats=%.2fGB points=%.2fGB\n",
            tot_caps*(3+1+3+1+3+3)*sizeof(double)/1e9,
            tot_pts*3*sizeof(double)/1e9);
    }

    // Serial concat pass: prefix-sum counts, then fill flat arrays in parallel.
    std::vector<int> arc_off(n+1, 0), cap_off(n+1, 0), pt_off(n+1, 0);
    for (int i = 0; i < n; i++) {
        arc_off[i+1] = arc_off[i] + locals[i].narcs;
        cap_off[i+1] = cap_off[i] + locals[i].nkept_caps;
        pt_off[i+1]  = pt_off[i]  + locals[i].npts;
    }
    int total_arcs = arc_off[n];
    int total_caps = cap_off[n];
    int total_pts  = pt_off[n];

    std::vector<int>    fa_sphere(total_arcs), fa_cap(total_arcs);
    std::vector<double> fa_s(total_arcs), fa_e(total_arcs);
    std::vector<int>    fc_sphere(total_caps);
    std::vector<int>    fp_sphere(total_pts);

    out.cap_normals.resize(total_caps, 3);
    out.cap_d.resize(total_caps);
    out.cap_centers.resize(total_caps, 3);
    out.cap_radii.resize(total_caps);
    out.cap_u.resize(total_caps, 3);
    out.cap_v.resize(total_caps, 3);
    out.point_positions.resize(total_pts, 3);
    MEM_MARK("after out.* resize");

    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < n; i++) {
        const SphereLocal& L = locals[i];

        int ao = arc_off[i];
        for (int a = 0; a < L.narcs; a++) {
            fa_sphere[ao + a] = i;
            fa_cap[ao + a]    = L.arc_cap[a];
            fa_s[ao + a]      = L.arc_s[a];
            fa_e[ao + a]      = L.arc_e[a];
        }

        int co = cap_off[i];
        for (int c = 0; c < L.nkept_caps; c++) {
            const Cap& cp = L.caps[c];
            fc_sphere[co + c]      = i;
            out.cap_normals(co+c, 0) = cp.normal.x;
            out.cap_normals(co+c, 1) = cp.normal.y;
            out.cap_normals(co+c, 2) = cp.normal.z;
            out.cap_d(co+c)          = cp.d;
            out.cap_centers(co+c, 0) = cp.circle_center.x;
            out.cap_centers(co+c, 1) = cp.circle_center.y;
            out.cap_centers(co+c, 2) = cp.circle_center.z;
            out.cap_radii(co+c)      = cp.circle_radius;
            out.cap_u(co+c, 0)       = cp.local_u.x;
            out.cap_u(co+c, 1)       = cp.local_u.y;
            out.cap_u(co+c, 2)       = cp.local_u.z;
            out.cap_v(co+c, 0)       = cp.local_v.x;
            out.cap_v(co+c, 1)       = cp.local_v.y;
            out.cap_v(co+c, 2)       = cp.local_v.z;
        }

        int po = pt_off[i];
        for (int p = 0; p < L.npts; p++) {
            fp_sphere[po + p]            = i;
            out.point_positions(po+p, 0) = L.pts[p].x;
            out.point_positions(po+p, 1) = L.pts[p].y;
            out.point_positions(po+p, 2) = L.pts[p].z;
        }
    }

    // Pack flat index arrays into BatchData (cap/point matrices already filled).
    out.arc_sphere_idx   = std::move(fa_sphere);
    out.arc_cap_idx      = std::move(fa_cap);
    out.arc_start        = std::move(fa_s);
    out.arc_end          = std::move(fa_e);
    out.cap_sphere_idx   = std::move(fc_sphere);
    out.point_sphere_idx = std::move(fp_sphere);

    MEM_MARK("batch: exit");
}

}  // namespace sphere_exposed_core
