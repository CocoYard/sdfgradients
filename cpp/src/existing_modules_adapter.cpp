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

// ═══════════════════════════════════════════════════════════════════
// sphere_intersect_core
// ═══════════════════════════════════════════════════════════════════

namespace sphere_intersect_core {

static inline int64_t grid_key(int gx, int gy, int gz) {
    return ((int64_t)(gx + 500000) * 1000001LL + (int64_t)(gy + 500000))
           * 1000001LL + (int64_t)(gz + 500000);
}

struct SphereRef { int idx; float r; };
struct GridLevel {
    float cell_size, inv_cell;
    std::unordered_map<int64_t, std::vector<SphereRef>> cells;
    void init(float cs) { cell_size = cs; inv_cell = 1.0f / cs; }
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

void find_intersections(const double* centers, const double* radii, int n,
                        std::vector<int>& offsets, std::vector<int>& neighbors) {
    if (n == 0) {
        offsets = {0};
        neighbors.clear();
        return;
    }

    std::vector<float> cx(n), cy(n), cz(n), ra(n);
    float min_r = 1e30f, max_r = 0;
    for (int i = 0; i < n; i++) {
        cx[i] = (float)centers[i*3];
        cy[i] = (float)centers[i*3+1];
        cz[i] = (float)centers[i*3+2];
        ra[i] = std::max((float)radii[i], 1e-10f);
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
    for (int lv = 0; lv < n_levels; lv++) levels[lv].init(level_cs[lv]);
    for (int i = 0; i < n; i++)
        for (int lv = home[i]; lv < n_levels; lv++)
            levels[lv].insert(i, cx[i], cy[i], cz[i], ra[i]);

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
                auto it = levels[lv].cells.find(grid_key(gcx+dx, gcy+dy, gcz+dz));
                if (it == levels[lv].cells.end()) continue;
                for (const auto& s : it->second) {
                    if (s.idx == i) continue;
                    float ddx = xi - cx[s.idx];
                    float ddy = yi - cy[s.idx];
                    float ddz = zi - cz[s.idx];
                    float sr = ri + s.r;
                    if (ddx*ddx + ddy*ddy + ddz*ddz < sr*sr)
                        buf.push_back(s.idx);
                }
            }

            std::sort(buf.begin(), buf.end());
            buf.erase(std::unique(buf.begin(), buf.end()), buf.end());
            result[i] = buf;
            buf = std::vector<int>();
        }
    }

    offsets.resize(n + 1, 0);
    for (int i = 0; i < n; i++) offsets[i+1] = offsets[i] + (int)result[i].size();
    int total = offsets[n];
    neighbors.resize(total);
    for (int i = 0; i < n; i++) {
        // Sort neighbors by absolute radius descending (largest first)
        std::sort(result[i].begin(), result[i].end(),
                  [&](int a, int b) { return std::abs(radii[a]) > std::abs(radii[b]); });
        int base = offsets[i];
        for (int k = 0; k < (int)result[i].size(); k++)
            neighbors[base + k] = result[i][k];
    }
}

}  // namespace sphere_intersect_core

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

// ── compute_one: per-sphere exposed region computation ───────────

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
        if (caps[i].phi >= PI - EPS10) {
            if (caps[i].containment_gap <= tol.tangent_tol && *out_npts < MAX_DEGEN_PTS) {
                out_pts[(*out_npts)++] = mc - mr * caps[i].normal;
            }
            return;
        }
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

// ── compute_exposed_batch ────────────────────────────────────────

void compute_exposed_batch(
    const double* centers, const double* radii, int n,
    const int* nbr_indices, const int* nbr_offsets,
    double tol_v, double degen_tol, double merge_tol, double tangent_tol,
    sdf::Options::BatchData& out)
{
    Tolerances tol{tol_v, degen_tol, merge_tol, tangent_tol};

    out.n_arcs.resize(n);
    out.n_points.resize(n);

    // Growable flat storage
    std::vector<int>    fa_sphere, fa_cap;
    std::vector<double> fa_s, fa_e;
    std::vector<int>    fc_sphere;
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

    for (int i = 0; i < n; i++) {
        int n_nbrs = nbr_offsets[i+1] - nbr_offsets[i];
        if (n_nbrs == 0) {
            out.n_arcs[i] = 0;
            out.n_points[i] = 0;
            continue;
        }

        oc_buf_v.resize(n_nbrs * 3);
        or_buf_v.resize(n_nbrs);

        const int* nb = nbr_indices + nbr_offsets[i];
        for (int j = 0; j < n_nbrs; j++) {
            int idx = nb[j];
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

        out.n_arcs[i] = narcs;
        out.n_points[i] = npts;

        // Append caps
        for (int c = 0; c < ncaps; c++) {
            int orig = -1;
            for (int k = 0; k < n_pcaps; k++) {
                if (remap[k] == c) { orig = k; break; }
            }
            if (orig < 0) continue;
            fc_sphere.push_back(i);
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

    // Pack into BatchData
    out.arc_sphere_idx = std::move(fa_sphere);
    out.arc_cap_idx    = std::move(fa_cap);
    out.arc_start      = std::move(fa_s);
    out.arc_end        = std::move(fa_e);

    out.cap_sphere_idx = std::move(fc_sphere);

    int total_caps = (int)fc_d.size();
    out.cap_normals.resize(total_caps, 3);
    out.cap_d.resize(total_caps);
    out.cap_centers.resize(total_caps, 3);
    out.cap_radii.resize(total_caps);
    out.cap_u.resize(total_caps, 3);
    out.cap_v.resize(total_caps, 3);
    for (int c = 0; c < total_caps; c++) {
        out.cap_normals(c, 0) = fc_normal[c*3];
        out.cap_normals(c, 1) = fc_normal[c*3+1];
        out.cap_normals(c, 2) = fc_normal[c*3+2];
        out.cap_d(c) = fc_d[c];
        out.cap_centers(c, 0) = fc_center[c*3];
        out.cap_centers(c, 1) = fc_center[c*3+1];
        out.cap_centers(c, 2) = fc_center[c*3+2];
        out.cap_radii(c) = fc_radius[c];
        out.cap_u(c, 0) = fc_u[c*3];
        out.cap_u(c, 1) = fc_u[c*3+1];
        out.cap_u(c, 2) = fc_u[c*3+2];
        out.cap_v(c, 0) = fc_v[c*3];
        out.cap_v(c, 1) = fc_v[c*3+1];
        out.cap_v(c, 2) = fc_v[c*3+2];
    }

    out.point_sphere_idx = std::move(fp_sphere);
    int total_pts = (int)out.point_sphere_idx.size();
    out.point_positions.resize(total_pts, 3);
    for (int p = 0; p < total_pts; p++) {
        out.point_positions(p, 0) = fp_pos[p*3];
        out.point_positions(p, 1) = fp_pos[p*3+1];
        out.point_positions(p, 2) = fp_pos[p*3+2];
    }
}

}  // namespace sphere_exposed_core
