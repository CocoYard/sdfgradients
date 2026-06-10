/// Adapter that wraps the core computation logic from existing pybind modules
/// (sphere_intersect.cpp, sphere_exposed_pybind.cpp) so our pure C++ code
/// can call them without pybind11.

// Optional: emit one `[MAX_ARC_PER_CAP_HIST]` line per batch on stderr,
// feeding plot_max_arcs_per_cap.py. Uncomment to enable when collecting
// boxplot data; otherwise leave off (one atomic-fetch_add per sphere is
// cheap, but the dump line itself is verbose).
// #define DEBUG_ARC_HIST 1

#include "types.h"
#include "full_compute_switch.h"
#include <vector>
#include <cmath>
#include <cstring>
#include <cfloat>
#include <algorithm>
#include <unordered_map>
#include <cstdio>
#include <atomic>
#include <chrono>
#include <climits>
#include <thread>
#include "always_assert.h"

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
  #include <unistd.h>
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
// #define CHECK_MEM
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

// ── Per-sphere algorithmic limits (controlled by FULL_COMPUTE_30CUBED) ──
//
// Production values (#else branch) sized so that all per-sphere scratch is
// stack-allocated (~MB-scale per OpenMP worker) yet large enough to handle
// real geometric configurations seen on natural meshes.
//
// FULL_COMPUTE_30CUBED bumps every limit ~50× for exact reproduction of
// reference results on a 30³ stress grid; never use it in production — those
// stack frames overflow on any decently parallel run.
//
//   MAX_CAPS        Max number of distinct half-space caps tracked per
//                   sphere. Each neighbor potentially contributes one cap;
//                   `nc` (the live count) is also the inner loop bound in
//                   compute_one (`for i < scan_limit && nc < MAX_CAPS`).
//                   Reaching this limit cuts neighbor scanning short.
//
//   MAX_ARCS        Two distinct uses, same value:
//                   (a) Final per-sphere output budget — at most this many
//                       boundary arcs are written to the output buffers
//                       (`arc_cap[]/arc_s[]/arc_e[]`). When more survive,
//                       they are partial-sorted by geometric length and the
//                       top-K is kept.
//                   (b) Per-cap intermediate budget enforced by `try_push`
//                       inside the clipping loop. Without this cap, naive
//                       pathological geometry can blow `new_arcs` to 100M+
//                       entries (observed) and never return.
//
//   MAX_INTERVALS   Max number of disjoint angular intervals tracked while
//                   computing a single new cap's exposed arcs against the
//                   active-cap set (`compute_exposed_arcs_on_circle`). One
//                   interval can split into two each time a new cap clips
//                   it, so this also bounds the working set there.
//
//   MAX_DEGEN_PTS   Max number of degenerate "tangent" points emitted per
//                   sphere (collapsed corners of the visible region). Comes
//                   from two sources: containment caps (φ ≈ π) and the
//                   post-pass `find_degen_pts` over surviving short arcs.
#ifdef FULL_COMPUTE_30CUBED
static constexpr int MAX_CAPS      = 30000;
static constexpr int MAX_ARCS      = 60000;
static constexpr int MAX_INTERVALS = 30000;
static constexpr int MAX_DEGEN_PTS = 10000;
#else
static constexpr int MAX_CAPS      = 512;
static constexpr int MAX_ARCS      = 2048;
static constexpr int MAX_INTERVALS = 2048;
static constexpr int MAX_DEGEN_PTS = 256;
#endif

// Hard safety cap on how many neighbors the incremental loop in compute_one
// will scan per sphere. Our per-sphere loop prunes dead caps aggressively, but
// pathological cases (>30k neighbors, observed) would still waste time scanning
// neighbors that contribute nothing. Kept as a safety net.
#ifdef FULL_COMPUTE_30CUBED
static constexpr int MAX_NEIGHBORS_SCANNED = INT_MAX;
#else
static constexpr int MAX_NEIGHBORS_SCANNED = 2048;
#endif

// ── opt-in profiling (env: SDF_BATCH_PROGRESS=1) ─────────────────
// All sub-section timing + per-4096 progress lines are gated by this flag,
// queried once on first call. Default off so the per-iteration now_sec()
// calls (~1-2s wall on a 1M-sphere run) and atomic adds don't tax the hot
// path. Counters live regardless but are only fed when enabled.
static bool sdf_batch_verbose() {
    static const bool v = (std::getenv("SDF_BATCH_PROGRESS") != nullptr);
    return v;
}
static std::atomic<long long> g_us_compute_one{0};
static std::atomic<long long> g_us_find_degen{0};
static std::atomic<long long> g_n_find_degen{0};
static std::atomic<long long> g_us_clip{0};
static std::atomic<long long> g_us_exposed{0};
static std::atomic<long long> g_us_count{0};
static std::atomic<long long> g_us_dedup{0};

#ifdef USE_OPENMP
static inline double now_sec() { return omp_get_wtime(); }
#else
static inline double now_sec() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}
#endif

// ── Tolerances ───────────────────────────────────────────────────
struct Tolerances {
    // Angular slack used as skip_tol in intersect_intervals(). Radians.
    // The other "small slack" thresholds that historically shared this
    // field (cap dedup, parallel-cut early-out) live as file-scope
    // constants below — they're tied to mesh / geometric units and
    // shouldn't be slaved to an angular tolerance.
    double interval_eps;
    double degen_tol;
    double merge_tol;
    double tangent_tol;
};

// Internal cap-dedup / degeneracy tolerances. Not exposed in the API
// because their physical units differ from `Tolerances::interval_eps`
// (radians) and from each other:
//   DEDUP_COS      : dimensionless, threshold on 1 - cos(angle between
//                    cap normals). 1e-4 ≈ 0.81° — caps within this
//                    angle and offset are treated as the same cap.
//   DEDUP_LEN_FRAC : DIMENSIONLESS — fraction of the main sphere's
//                    radius (mr). Every use is `DEDUP_LEN_FRAC * mr`,
//                    so the effective length threshold scales with
//                    sphere size and stays scale-invariant.
//                    Applies to:
//                      - cap-plane offset dedup (cap.d - caps[e].d)
//                      - parallel-cut early-out where `c` is the
//                        signed distance from host circle center to
//                        the cutter plane; `c > DEDUP_LEN_FRAC * mr`
//                        means the whole host circle is on the
//                        covered side.
static constexpr double DEDUP_COS      = 1e-4;
static constexpr double DEDUP_LEN_FRAC = 1e-4;

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
    // Returns x mod m in [0, m). The naive `r + m` can round up to exactly m
    // when |r| ≪ m (catastrophic absorption), violating the half-open
    // contract — observed at sphere 41667 of mesh 49546 where a tangent hit
    // came out as TWO_PI instead of 0, then angle_in_arc collapsed it to 0
    // for the inside-test but boundaries kept TWO_PI raw, fabricating a full
    // 2π fake arc piece and seeding cascading explosion.
    double r = std::fmod(x, m);
    if (r < 0) r += m;
    if (r >= m) r = 0;
    return r;
}

// ── Data structures ──────────────────────────────────────────────
//
// A `Cap` is the spherical region cut off the main sphere by one neighbor
// sphere. Its boundary on the main sphere is a small circle ("cap circle").
struct Cap {
    Vec3 normal;            // cutting plane normal, oriented from main → neighbor
    double d;               // plane offset: a point x is "inside the cap"
                            // (covered by the neighbor) iff normal·x > d
    Vec3 circle_center;     // 3D center of the cap circle (lies on the cutting plane)
    double circle_radius;   // radius of the cap circle (≤ main sphere radius);
                            // 0 when the cap collapses to a point or covers everything
    Vec3 local_u, local_v;  // orthonormal basis spanning the cutting plane;
                            // any point on the cap circle is
                            //   circle_center + circle_radius·(cos t·u + sin t·v)
    double phi;             // half-opening angle of the cap measured from main
                            // sphere center: φ=0 → single tangent point,
                            // φ=π/2 → hemisphere, φ=π → cap covers the whole sphere
                            // (i.e., main is fully inside the neighbor)
    int sphere_idx;         // index of the neighbor sphere that produced this cap
    double containment_gap; // only meaningful when φ ≈ π (containment branch).
                            // Equal to neighbor.r - dist - main.r, i.e., the
                            // closest gap between the two sphere surfaces.
                            // 0 = internally tangent (Δsdf = center distance),
                            // DBL_MAX = not a containment cap.
};

// Closed angular interval on a circle, in radians.
struct Interval {
    double start, end;
};

// One contiguous arc of the exposed-region boundary, lying on a specific
// cap's circle. `t_start, t_end` parametrize the arc on that circle via
// `point_on_circle(caps[cap_idx], t)`.
struct BoundaryArc {
    int cap_idx;            // host cap whose circle this arc lies on
    double t_start, t_end;  // angular range in [0, 2π) on the host's local_u/v
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

// Variant where the host's circle-plane intersection with cutting_cap has
// already been computed (hits, nhits). The caller is expected to cache these
// across all arcs that share the same host, since intersect_circle_with_plane
// has expensive trig (atan2, acos) and the same host typically owns many arcs.
static void clip_arc_with_hits(const BoundaryArc &arc, const Cap &host,
                               const Cap &cutting_cap,
                               const double hits[2], int nhits,
                               std::vector<BoundaryArc> &result) {
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

// Original signature kept for callers that don't have hits precomputed.
static void clip_arc_by_cap(const BoundaryArc &arc, const Cap &cutting_cap,
                            const Cap caps[], std::vector<BoundaryArc> &result) {
    const Cap &host = caps[arc.cap_idx];
    double hits[2];
    int nhits = intersect_circle_with_plane(host, cutting_cap, hits);
    clip_arc_with_hits(arc, host, cutting_cap, hits, nhits, result);
}

static int intersect_intervals(const Interval *a, int na,
                               const Interval *b, int nb,
                               Interval *out, int max_out,
                               double skip_tol, double merge_tol) {
    int cnt = 0;
    for (int i = 0; i < na; i++) {
        // Dedup window: pushes from this iv[i] live in out[iv_start..cnt).
        // With nb ≤ 2 this is at most 1 entry, so the inner scan is O(1).
        const int iv_start = cnt;
        for (int j = 0; j < nb; j++) {
            double lo = std::max(a[i].start, b[j].start);
            double hi = std::min(a[i].end,   b[j].end);
            if (!(hi - lo > -skip_tol) || cnt >= max_out) continue;

            double new_lo, new_hi;
            if (hi < lo) {
                // Tolerance fudge: near-touching within skip_tol. Push a
                // placeholder at the midpoint so subsequent steps don't
                // forget the boundary.
                double mid = (lo + hi) / 2.0;
                new_lo = mid - 1e-15;
                new_hi = mid + 1e-15;
            } else {
                new_lo = lo;
                new_hi = hi;
            }

            // Per-iv dedup: if a previous push from this same iv[i] lands
            // within skip_tol at the start, this is a wrap-seam duplicate
            // (a degenerate iv point sitting on, or in the tiny gap between,
            // the two con pieces would otherwise emit one fake + one legit
            // — or two fakes — at the same location, doubling niv each
            // step). Genuine wrap-span pushes (one near 2π, one near 0)
            // sit ~π apart and survive this check.
            bool dup = false;
            for (int k = iv_start; k < cnt; k++) {
                if (std::fabs(out[k].start - new_lo) < skip_tol) {
                    dup = true;
                    break;
                }
            }
            if (!dup) out[cnt++] = {new_lo, new_hi};
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

// Per-sphere "max arcs per cap" histogram, one bin per integer in
// [0, MAX_ARCS]. Reset at start of each compute_exposed_batch, dumped at end.
// Off by default; toggle the `#define DEBUG_ARC_HIST` at the top of this
// file to back the arcs-per-cap boxplot (plot_max_arcs_per_cap.py).
#ifdef DEBUG_ARC_HIST
static std::atomic<long long> g_per_sphere_max_arc_hist[MAX_ARCS + 1];
#endif

// mr: main sphere radius — used to scale DEDUP_LEN_FRAC into a length
// threshold for the parallel-cut early-out (see DEDUP_LEN_FRAC comment).
static int compute_exposed_arcs_on_circle(int cap_idx, const Cap caps[],
                                          const int *active_caps, int n_active,
                                          Interval *result, const Tolerances &tol,
                                          double mr) {
    const Cap &host = caps[cap_idx];
    double R = host.circle_radius;
    if (R < EPS) return 0;

    // Heap-backed per-thread scratch — MAX_INTERVALS=30000 under
    // FULL_COMPUTE_30CUBED would put ~960 KB on the stack otherwise,
    // overflowing OpenMP worker stacks. iv[0] is set before any other entry
    // is read, so no cross-call zeroing is needed.
    thread_local std::vector<Interval> iv_buf(MAX_INTERVALS);
    thread_local std::vector<Interval> tmp_buf(MAX_INTERVALS);
    Interval* iv  = iv_buf.data();
    Interval* tmp = tmp_buf.data();
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
            // Other cap's plane is parallel to host's circle plane (their
            // normals are (anti)parallel). Same-direction parallels would
            // have been collapsed at compute_one's dedup, so in practice
            // this branch fires for anti-parallel cutters: two opposing
            // neighbors covering main sphere from opposite sides.
            //
            // `c` is the signed distance from host's circle center to
            // other's cutting plane, measured along other.normal. Since
            // the planes are parallel, every point on host's circle has
            // the same `c`, so the entire circle is uniformly on one side.
            //
            //   c > 0 → host circle lies on other's covered side → other
            //           swallows host's whole circle → host contributes
            //           no exposed arcs → return 0.
            //   c ≤ 0 → host circle on other's uncovered side → other
            //           does not clip host at all → continue.
            //
            // Multiply by mr so the threshold scales with main sphere
            // size (DEDUP_LEN_FRAC is a fraction, see top-of-file note).
            if (c > DEDUP_LEN_FRAC * mr) return 0; // should not happen in strict SDF
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
                                  tol.interval_eps, tol.merge_tol);
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

// Trigger condition: total arc length < degen_tol, meaning the boundary of the
// exposed region has collapsed to (near) a single point. Each surviving arc is
// then essentially a degenerate corner — its midpoint on the cap circle is a
// good 3D candidate. Dedup against existing pts within degen_tol.
//
// This replaces the earlier O(nc^3) cap-pair brute force, which re-derived
// geometry that the arc list already encodes.
static int find_degen_pts(const Cap caps[],
                          const std::vector<BoundaryArc>& all_arcs,
                          Vec3 pts[], int np_in, const Tolerances &tol) {
    int np = np_in;
    for (const auto& arc : all_arcs) {
        if (np >= MAX_DEGEN_PTS) break;
        const Cap& c = caps[arc.cap_idx];
        if (c.circle_radius < EPS) continue;
        double t_mid = 0.5 * (arc.t_start + arc.t_end);
        double cs = std::cos(t_mid), sn = std::sin(t_mid);
        Vec3 pt = c.circle_center + c.circle_radius * (cs * c.local_u + sn * c.local_v);
        bool dup = false;
        for (int k = 0; k < np; k++) {
            if (length(pt - pts[k]) < tol.degen_tol) { dup = true; break; }
        }
        if (!dup) pts[np++] = pt;
    }
    return np;
}

// ── compute_one: per-sphere exposed region computation ───────────

// Single-pass incremental version: per neighbor, compute its cap, dedup,
// clip existing arcs, compute the new cap's own arcs, and prune caps whose
// arcs have all been eaten. Avoids the old two-phase pattern of building a
// full caps[] up-front and iterating again for arcs.
// (DEBUG_ARC_HIST atomic counters moved to before compute_exposed_arcs_on_circle.)
//
// Parameter glossary (used throughout this function and its helpers):
//   mc       : main sphere center (the sphere whose exposed region we compute)
//   mr       : main sphere radius — also the length scale for tolerances
//              (DEDUP_LEN_FRAC, tangent_tol, etc.) so they stay scale-invariant
//   oc       : neighbor (other) sphere centers, packed [x,y,z, x,y,z, ...]
//   or_      : neighbor sphere radii
//   n_others : number of neighbors
//   cap      : a half-space cut produced by one neighbor sphere intersecting
//              the main sphere — represented by a unit normal and offset d
//              (point inside cap iff normal·x > d); see struct Cap above
//   host     : the cap whose own boundary circle we are clipping
//   other    : another cap being used as a cutter against host's circle
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
    // Per-thread scratch buffers, heap-backed via thread_local std::vector.
    // FULL_COMPUTE_30CUBED bumps MAX_CAPS to 30000 — the original stack-
    // resident C arrays added up to ~5 MB per frame and blew through OpenMP
    // worker stacks (default 4 MB). Storing them as thread_local vectors keeps
    // them off the stack while reusing the allocation across calls.
    //
    // The buffers tagged "ZEROED" below need to read as 0 on entry to each
    // call (matches the `= {}` zero-init of the original arrays). The
    // ScratchReset guard restores the invariant by clearing [0, nc) on every
    // return path — cheaper than zeroing the full MAX_CAPS range each call.
    thread_local std::vector<Cap>            caps_buf(MAX_CAPS);
    thread_local std::vector<int>            arc_count_buf(MAX_CAPS, 0);     // ZEROED
    thread_local std::vector<int>            active_caps_buf(MAX_CAPS);
    thread_local std::vector<unsigned char>  in_active_buf(MAX_CAPS, 0);     // ZEROED
    thread_local std::vector<double>         host_hits_buf_v(MAX_CAPS * 2);
    thread_local std::vector<int>            host_nhits_buf_v(MAX_CAPS);
    thread_local std::vector<unsigned char>  host_circle_inside_cut_buf(MAX_CAPS);
    thread_local std::vector<unsigned char>  host_seg_h0h1_inside_cut_buf(MAX_CAPS);
    thread_local std::vector<int>            host_stamp_buf(MAX_CAPS, 0);    // ZEROED
    thread_local std::vector<int>            new_arc_count_buf(MAX_CAPS);
    thread_local std::vector<int>            all_caps_buf_v(MAX_CAPS);
    thread_local std::vector<int>            arc_count2_buf(MAX_CAPS, 0);    // ZEROED
    thread_local std::vector<int>            remap_buf(MAX_CAPS);
    thread_local std::vector<Interval>       ivs_buf(MAX_INTERVALS);

    Cap*    caps                       = caps_buf.data();
    int     nc                         = 0; // number of caps currently in the caps[] list; grows up to nc = n_others
    int*    arc_count                  = arc_count_buf.data();
    int*    active_caps                = active_caps_buf.data();
    bool*   in_active                  = reinterpret_cast<bool*>(in_active_buf.data());
    int     n_active                   = 0;

    // Per-host caches for clip_arc_by_cap (stamped invalidation).
    //   host_hits_buf[h]    : intersect_circle_with_plane output for host h
    //   host_nhits_buf[h]   : 0 or 2
    //   host_circle_inside_cut[h] : when nhits=0, whether the entire host
    //                               circle is inside the cutting cap (drop all)
    //   host_seg_h0h1_inside_cut[h] : when nhits=2, whether the [h0,h1]
    //                                 segment of host circle is inside cutting
    // The two side flags let us skip per-arc midpoint trig — once we know
    // which segment of the host circle is "inside cut", any arc/piece's
    // segment can be determined by an angle comparison rather than another
    // is_inside_cap(point_on_circle(...)) call.
    double  (*host_hits_buf)[2]        = reinterpret_cast<double(*)[2]>(host_hits_buf_v.data());
    int*    host_nhits_buf             = host_nhits_buf_v.data();
    bool*   host_circle_inside_cut     = reinterpret_cast<bool*>(host_circle_inside_cut_buf.data());
    bool*   host_seg_h0h1_inside_cut   = reinterpret_cast<bool*>(host_seg_h0h1_inside_cut_buf.data());
    int*    host_stamp                 = host_stamp_buf.data();

    int*      all_caps_buf = all_caps_buf_v.data();
    int*      arc_count2   = arc_count2_buf.data();
    int*      remap        = remap_buf.data();
    Interval* ivs          = ivs_buf.data();

    struct ScratchReset {
        int* arc_count; int* host_stamp;
        unsigned char* in_active; int* arc_count2;
        const int* nc_ptr;
        ~ScratchReset() {
            int n = *nc_ptr;
            std::fill_n(arc_count,  n, 0);
            std::fill_n(host_stamp, n, 0);
            std::fill_n(in_active,  n, (unsigned char)0);
            std::fill_n(arc_count2, n, 0);
        }
    } _scratch_reset{arc_count, host_stamp,
                     in_active_buf.data(), arc_count2,
                     &nc};

    std::vector<BoundaryArc> all_arcs;
    std::vector<BoundaryArc> new_arcs;
    all_arcs.reserve(16);
    new_arcs.reserve(16);

    // Per-cap arc budget enforced while building new_arcs each outer iter.
    // Without this, pathological geometry can blow up new_arcs to 100M+.
    // Final output is still truncated by max_arcs at the end (top-K by length).
    int* new_arc_count = new_arc_count_buf.data();
    auto try_push = [&](int cap_idx, double t_s, double t_e) {
        ALWAYS_ASSERT(new_arc_count[cap_idx] < MAX_ARCS,
                      "per-cap arc budget exceeded in one outer iter");
        if (new_arc_count[cap_idx] >= MAX_ARCS) return;
        new_arc_count[cap_idx]++;
        new_arcs.push_back({cap_idx, t_s, t_e});
    };

    *out_ncaps = 0;
    *out_narcs = 0;
    *out_npts = 0;
    *out_total_arc = 0;
    *out_nc = 0;

    int scan_limit = n_others < MAX_NEIGHBORS_SCANNED ? n_others : MAX_NEIGHBORS_SCANNED;
    const bool prof = sdf_batch_verbose();
    double t_dedup_acc = 0, t_clip_acc = 0, t_exposed_acc = 0, t_count_acc = 0;

    for (int i = 0; i < scan_limit && nc < MAX_CAPS; i++) {
        Cap cap;
        if (!compute_cap(mc, mr, {oc[i*3], oc[i*3+1], oc[i*3+2]}, or_[i], i, cap))
            continue;

        // (we are fully inside the neighbor). Containment (φ ≥ π): this
        // neighbor swallows us. Emit tangent point if applicable and
        // bail — exposed region is empty.
        if (cap.phi >= PI - EPS10) {
            if (cap.containment_gap <= tol.tangent_tol && *out_npts < MAX_DEGEN_PTS) {
                out_pts[(*out_npts)++] = mc - mr * cap.normal;
            }
            *out_ncaps = 0;
            *out_narcs = 0;
            *out_nc = 0;
            return;
        }

        // Dedup against existing caps,
        // Dominating new replaces
        // the old in-place; duplicates are skipped.
        double _t_dd = prof ? now_sec() : 0.0;
        bool dup = false;
        int replaced = -1;
        for (int e = nc - 1; e >= 0; e--) {
            double d = dot(cap.normal, caps[e].normal);
            if (d > 1 - DEDUP_COS) {
                if (std::fabs(cap.d - caps[e].d) < DEDUP_LEN_FRAC * mr) { dup = true; break; }
                else if (cap.d > caps[e].d)                   { dup = true; break; }
                else { replaced = e; break; }
            }
        }
        if (prof) t_dedup_acc += now_sec() - _t_dd;
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

        // Clip all existing arcs by the new cap. Two-level caching:
        //   1. Per host, cache the circle/plane intersection (expensive trig).
        //   2. Per host, cache "which segment of host circle is inside cutting"
        //      so per-arc clipping is just angle comparisons, no trig.
        double _t_cl = prof ? now_sec() : 0.0;
        new_arcs.clear();
        // Reset only the caps we have so far (memset O(nc), not O(MAX_CAPS)).
        std::memset(new_arc_count, 0, sizeof(int) * (size_t)nc);
        const int stamp = i + 1;
        const Cap& cut = caps[new_idx];
        for (const auto& arc : all_arcs) {
            int h = arc.cap_idx;
            // stamp is used for circle/plane intersection caching. Multiple arcs can share 
            // the same host cap, so we compute the intersection once per host per new cap, 
            // then cache it via host_stamp.
            if (host_stamp[h] != stamp) {
                int nhits = intersect_circle_with_plane(caps[h], cut, host_hits_buf[h]);
                host_nhits_buf[h] = nhits;
                if (nhits == 0) {
                    // Entire host circle is on one side of cutting plane.
                    Vec3 tp = point_on_circle(caps[h], 0.0);
                    host_circle_inside_cut[h] = is_inside_cap(tp, cut);
                } else {
                    // host circle split by cutting plane into [h0,h1] (S1) and
                    // [h1, h0+2π] (S2). Test midpoint of S1 to learn which
                    // segment is "inside cut" (= covered, arcs there get dropped).
                    double mid_h = 0.5 * (host_hits_buf[h][0] + host_hits_buf[h][1]);
                    Vec3 tp = point_on_circle(caps[h], mid_h);
                    host_seg_h0h1_inside_cut[h] = is_inside_cap(tp, cut);
                }
                host_stamp[h] = stamp;
            }

            int nhits = host_nhits_buf[h];
            if (nhits == 0) {
                // Whole host circle on one side; per-arc decision is uniform.
                if (!host_circle_inside_cut[h]) try_push(arc.cap_idx, arc.t_start, arc.t_end);
                continue;
            }

            // nhits == 2: filter the two hits to those lying inside this arc.
            const double h0 = host_hits_buf[h][0];
            const double h1 = host_hits_buf[h][1];
            double hits_in[2];
            int n_in = 0;
            if (angle_in_arc(h0, arc.t_start, arc.t_end)) hits_in[n_in++] = h0;
            if (angle_in_arc(h1, arc.t_start, arc.t_end)) hits_in[n_in++] = h1;

            if (n_in == 0) {
                // Arc fully on one side. Determine side via midpoint angle vs
                // [h0,h1] segment — no trig needed (replaces the old midpoint
                // is_inside_cap test).
                double mt = fmod_pos(0.5 * (arc.t_start + arc.t_end), TWO_PI);
                bool in_S1 = (mt >= h0 && mt <= h1);
                bool inside_cut = (in_S1 == host_seg_h0h1_inside_cut[h]);
                if (!inside_cut) try_push(arc.cap_idx, arc.t_start, arc.t_end);
                continue;
            }

            if (n_in == 2) {
                double k0 = fmod_pos(hits_in[0] - arc.t_start, TWO_PI);
                double k1 = fmod_pos(hits_in[1] - arc.t_start, TWO_PI);
                if (k0 > k1) std::swap(hits_in[0], hits_in[1]);
            }

            // Arc crosses cutting plane: split into pieces; per-piece side is
            // determined by piece midpoint angle vs [h0,h1] segment.
            double boundaries[4];
            boundaries[0] = arc.t_start;
            for (int k = 0; k < n_in; k++) boundaries[1 + k] = hits_in[k];
            boundaries[1 + n_in] = arc.t_end;
            int nbnd = 2 + n_in;
            for (int k = 0; k < nbnd - 1; k++) {
                double t_s = boundaries[k];
                double t_e = boundaries[k + 1];
                if (t_e < t_s - EPS) t_e += TWO_PI;
                if (t_e - t_s < 1e-15) continue;
                double mt = fmod_pos(0.5 * (t_s + t_e), TWO_PI);
                bool in_S1 = (mt >= h0 && mt <= h1);
                bool inside_cut = (in_S1 == host_seg_h0h1_inside_cut[h]);
                if (!inside_cut) try_push(arc.cap_idx, t_s, t_e);
            }
        }
        if (prof) t_clip_acc += now_sec() - _t_cl;

        // Compute the new cap's own exposed arcs against ALL caps seen so
        // far (not just `active_caps`). The active-only optimization is
        // unsafe: a cap A whose boundary arcs have all been clipped by
        // others can still cover the interior of a region no other
        // surviving cap reaches (e.g. A is a polar cap; B+D ate A's
        // boundary at the equator but neither covers near the pole). A
        // later cap C lying inside A's interior gets tested only against
        // {B, D} and falsely retains exposed arcs. BVH masks this with
        // redundancy; RT's minimal neighbor set has no spare cap to
        // cover for an "inactive" A, so the bug shows as missing degen
        // points exactly on geometrically meaningful arcs.
        // compute_exposed_arcs_on_circle skips new_idx internally and
        // skips small-radius (frozen) caps via the A_line < EPS branch.
        int n_all = 0;
        for (int k = 0; k < nc; k++) {
            if (caps[k].circle_radius >= EPS) all_caps_buf[n_all++] = k;
        }
        double _t_ex = prof ? now_sec() : 0.0;
        int n_new = compute_exposed_arcs_on_circle(new_idx, caps,
                                                //    active_caps, n_active,
                                                   all_caps_buf, n_all,
                                                   ivs, tol, mr);
        for (int a = 0; a < n_new; a++)
            try_push(new_idx, ivs[a].start, ivs[a].end);
        if (prof) t_exposed_acc += now_sec() - _t_ex;

        all_arcs.swap(new_arcs);

        // Rebuild arc_count from the surviving arcs.
        double _t_ct = prof ? now_sec() : 0.0;
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
        if (prof) t_count_acc += now_sec() - _t_ct;
    }
    if (prof) {
        g_us_dedup.fetch_add((long long)(t_dedup_acc * 1e6), std::memory_order_relaxed);
        g_us_clip.fetch_add((long long)(t_clip_acc * 1e6), std::memory_order_relaxed);
        g_us_exposed.fetch_add((long long)(t_exposed_acc * 1e6), std::memory_order_relaxed);
        g_us_count.fetch_add((long long)(t_count_acc * 1e6), std::memory_order_relaxed);
    }

    // Build output: when more arcs survived than max_arcs, keep the
    // top-K by geometric length (radius × angular extent). nth_element is
    // O(n) avg, much cheaper than full sort. Then recompute remap from the
    // kept subset so caps that lost all their arcs to truncation get pruned.
    const int total_in = (int)all_arcs.size();
    const int keep = total_in < max_arcs ? total_in : max_arcs;

    // Indices into all_arcs, partial-sorted to put the top-keep first.
    static thread_local std::vector<std::pair<double,int>> ranked;
    ranked.clear();
    ranked.reserve((size_t)total_in);
    for (int k = 0; k < total_in; k++) {
        const auto& a = all_arcs[k];
        ranked.emplace_back(caps[a.cap_idx].circle_radius * (a.t_end - a.t_start), k);
    }
    if (keep < total_in) {
        std::nth_element(ranked.begin(), ranked.begin() + keep, ranked.end(),
            [](const std::pair<double,int>& a, const std::pair<double,int>& b){
                return a.first > b.first;
            });
        ranked.resize((size_t)keep);
    }

    // Recompute arc_count from the kept subset, then build remap.
    // arc_count2 / remap are thread_local at function top (arc_count2 enters
    // zero, restored by ScratchReset on exit).
    for (const auto& r : ranked) arc_count2[all_arcs[r.second].cap_idx]++;
    int kept_caps = 0;
    for (int i = 0; i < nc; i++) {
        if (arc_count2[i] > 0) remap[i] = kept_caps++;
        else                   remap[i] = -1;
    }
#ifdef DEBUG_ARC_HIST
    // Tally this sphere's max-arcs-per-cap into the global histogram.
    int max_apc = 0;
    for (int i = 0; i < nc; i++) {
        if (arc_count2[i] > max_apc) max_apc = arc_count2[i];
    }
    if (max_apc > MAX_ARCS) max_apc = MAX_ARCS;
    g_per_sphere_max_arc_hist[max_apc].fetch_add(1, std::memory_order_relaxed);
#endif

    *out_nc = nc;
    if (out_caps)  std::memcpy(out_caps, caps, nc * sizeof(Cap));
    if (out_remap) std::memcpy(out_remap, remap, nc * sizeof(int));

    double total = 0;
    for (int na = 0; na < keep; na++) {
        const auto& arc = all_arcs[ranked[na].second];
        arc_cap[na] = remap[arc.cap_idx];
        arc_s[na]   = arc.t_start;
        arc_e[na]   = arc.t_end;
        // Total arc length = sum over arcs of (cap circle radius) × (angular extent).
        // degen_tol is a length in mesh units, not an angle.
        total += ranked[na].first;
    }
    *out_narcs = keep;
    *out_ncaps = kept_caps;
    *out_total_arc = total;

    if (total < tol.degen_tol && nc > 0) {
        if (prof) {
            double _t = now_sec();
            *out_npts = find_degen_pts(caps, all_arcs, out_pts, *out_npts, tol);
            long long us = (long long)((now_sec() - _t) * 1e6);
            g_us_find_degen.fetch_add(us, std::memory_order_relaxed);
            g_n_find_degen.fetch_add(1, std::memory_order_relaxed);
        } else {
            *out_npts = find_degen_pts(caps, all_arcs, out_pts, *out_npts, tol);
        }
    }
}

// ── compute_exposed_batch ────────────────────────────────────────

void compute_exposed_batch(
    const double* centers, const double* radii, int n,
    const std::vector<std::vector<int>>& nbrs,
    double interval_eps, double degen_tol, double merge_tol, double tangent_tol,
    sdf::Options::BatchData& out)
{
    Tolerances tol{interval_eps, degen_tol, merge_tol, tangent_tol};
    const bool prof = sdf_batch_verbose();
    if (prof) {
        g_us_compute_one.store(0);
        g_us_find_degen.store(0);
        g_n_find_degen.store(0);
        g_us_clip.store(0);
        g_us_exposed.store(0);
        g_us_count.store(0);
        g_us_dedup.store(0);
    }
    double t_loop_start = prof ? now_sec() : 0.0;
#ifdef DEBUG_ARC_HIST
    for (int b = 0; b <= MAX_ARCS; b++) g_per_sphere_max_arc_hist[b].store(0);
#endif

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

    // ---- Watchdog: per-thread "currently working on i" + start time. ----
    int n_threads_max = 1;
#ifdef USE_OPENMP
    n_threads_max = omp_get_max_threads();
#endif
    std::vector<std::atomic<int>>    wd_cur_i(n_threads_max);
    std::vector<std::atomic<int>>    wd_cur_nbr(n_threads_max);
    std::vector<std::atomic<double>> wd_cur_t0(n_threads_max);
    for (int t = 0; t < n_threads_max; t++) {
        wd_cur_i[t].store(-1);
        wd_cur_nbr[t].store(0);
        wd_cur_t0[t].store(0.0);
    }
    std::atomic<bool> wd_stop{false};
    std::thread wd([&]{
        while (!wd_stop.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(2));
            if (wd_stop.load()) break;
            double now = now_sec();
            for (int t = 0; t < n_threads_max; t++) {
                int ci = wd_cur_i[t].load();
                if (ci < 0) continue;
                double dt = now - wd_cur_t0[t].load();
                if (dt > 3.0) {
                    std::fprintf(stderr,
                        "[WATCHDOG] tid=%d stuck on i=%d n_nbrs=%d for %.1fs\n",
                        t, ci, wd_cur_nbr[t].load(), dt);
                    std::fflush(stderr);
                }
            }
        }
    });

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
        // Watchdog: always time compute_one (cheap), warn loudly on slow spheres.
        double _t_co = now_sec();
#ifdef USE_OPENMP
        int _wd_tid = omp_get_thread_num();
#else
        int _wd_tid = 0;
#endif
        wd_cur_nbr[_wd_tid].store(n_nbrs);
        wd_cur_t0[_wd_tid].store(_t_co);
        wd_cur_i[_wd_tid].store(i);
        compute_one(
            {centers[i*3], centers[i*3+1], centers[i*3+2]}, radii[i],
            oc_buf_v.data(), or_buf_v.data(), n_use, tol,
            &ncaps, &narcs, _ac, _as, _ae, MAX_ARCS,
            &total_arc, &npts, _dp,
            _caps, &n_pcaps, remap);
        wd_cur_i[_wd_tid].store(-1);
        double _dt_co = now_sec() - _t_co;
        if (_dt_co > 0.5) {
            std::fprintf(stderr,
                "[SLOW SPHERE] i=%d n_nbrs=%d n_use=%d ncaps=%d narcs=%d npts=%d "
                "dt=%.3fs r=%.6g c=(%.6g,%.6g,%.6g)\n",
                i, n_nbrs, n_use, ncaps, narcs, npts, _dt_co, radii[i],
                centers[i*3], centers[i*3+1], centers[i*3+2]);
            std::fflush(stderr);
        }
        if (prof) g_us_compute_one.fetch_add(
            (long long)(_dt_co * 1e6),
            std::memory_order_relaxed);

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
        if (prof && (done & 4095) == 0) {
            #pragma omp critical
            {
                double t_elapsed = now_sec() - t_loop_start;
                double co_s = g_us_compute_one.load(std::memory_order_relaxed) * 1e-6;
                double fd_s = g_us_find_degen.load(std::memory_order_relaxed) * 1e-6;
                double cl_s = g_us_clip.load(std::memory_order_relaxed) * 1e-6;
                double ex_s = g_us_exposed.load(std::memory_order_relaxed) * 1e-6;
                double ct_s = g_us_count.load(std::memory_order_relaxed) * 1e-6;
                double dd_s = g_us_dedup.load(std::memory_order_relaxed) * 1e-6;
                long long fd_n = g_n_find_degen.load(std::memory_order_relaxed);
                double inv = co_s > 0 ? 100.0 / co_s : 0.0;
                std::fprintf(stderr,
                    "[batch] %d/%d (%.1f%%) wall=%.1fs co=%.1fs  "
                    "clip=%.1fs(%.0f%%) exposed=%.1fs(%.0f%%) count=%.1fs(%.0f%%) "
                    "dedup=%.1fs(%.0f%%) find_degen=%.1fs(%.0f%%, n=%lld)  "
                    "(i=%d n_nbrs=%d ncaps=%d narcs=%d)\n",
                    done, n, 100.0*done/n, t_elapsed, co_s,
                    cl_s, cl_s*inv, ex_s, ex_s*inv, ct_s, ct_s*inv,
                    dd_s, dd_s*inv, fd_s, fd_s*inv, (long long)fd_n,
                    i, n_nbrs, ncaps, narcs);
                MEM_LOG("[RSS] RSS=%.2f GB (+%.2f)\n",
                    mem_rss_bytes()/1e9,
                    (mem_rss_bytes() - rss_before_loop)/1e9);
            }
        }
    }
    if (prof) {
        double t_elapsed = now_sec() - t_loop_start;
        double co_s = g_us_compute_one.load() * 1e-6;
        double fd_s = g_us_find_degen.load() * 1e-6;
        double cl_s = g_us_clip.load() * 1e-6;
        double ex_s = g_us_exposed.load() * 1e-6;
        double ct_s = g_us_count.load() * 1e-6;
        double dd_s = g_us_dedup.load() * 1e-6;
        long long fd_n = g_n_find_degen.load();
        double inv = co_s > 0 ? 100.0 / co_s : 0.0;
        std::fprintf(stderr,
            "[batch] DONE n=%d wall=%.2fs co_sum=%.2fs  "
            "clip=%.2fs(%.0f%%) exposed=%.2fs(%.0f%%) count=%.2fs(%.0f%%) "
            "dedup=%.2fs(%.0f%%) find_degen=%.2fs(%.0f%%, n=%lld)\n",
            n, t_elapsed, co_s,
            cl_s, cl_s*inv, ex_s, ex_s*inv, ct_s, ct_s*inv,
            dd_s, dd_s*inv, fd_s, fd_s*inv, (long long)fd_n);
    }

    wd_stop.store(true);
    if (wd.joinable()) wd.join();
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
#ifdef DEBUG_ARC_HIST
    {
        // Per-sphere max-arcs-per-cap histogram dump (compact value:count).
        long long total = 0;
        long long over_cap = 0;   // # spheres with max arcs/cap > MAX_CAPS
        int observed_max = 0;
        for (int b = 0; b <= MAX_ARCS; b++) {
            long long c = g_per_sphere_max_arc_hist[b].load();
            total += c;
            if (c > 0 && b > observed_max) observed_max = b;
            if (b > MAX_CAPS) over_cap += c;
        }
        std::fprintf(stderr,
            "[MAX_ARC_PER_CAP_HIST] n=%d MAX_CAPS=%d MAX_ARCS=%d "
            "spheres=%lld observed_max=%d over_MAX_CAPS=%lld pairs(value:count):",
            n, MAX_CAPS, MAX_ARCS, total, observed_max, over_cap);
        for (int b = 0; b <= MAX_ARCS; b++) {
            long long c = g_per_sphere_max_arc_hist[b].load();
            if (c > 0) std::fprintf(stderr, " %d:%lld", b, c);
        }
        std::fprintf(stderr, "\n");
        std::fflush(stderr);
    }
#endif
}

}  // namespace sphere_exposed_core
