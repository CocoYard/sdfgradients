/*
 * sphere_exposed_ext.c  (v3 — fast + memory-efficient)
 *
 * v1 was fast but created millions of Python objects (40GB for 91k spheres).
 * v2 fixed memory but was slow due to per-sphere malloc/free.
 * v3: stack arrays in the hot path (like v1), compact numpy batch output (like v2),
 *     pre-convert neighbor arrays outside the loop.
 *
 * Compile (macOS):
 *   gcc -O3 -shared -fPIC -undefined dynamic_lookup \
 *     -o sphere_exposed_ext$(python3-config --extension-suffix) \
 *     sphere_exposed_ext.c \
 *     -I$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))") \
 *     -I$(python3 -c "import numpy; print(numpy.get_include())") -lm
 *
 * Compile (Linux):
 *   gcc -O3 -shared -fPIC \
 *     -o sphere_exposed_ext$(python3-config --extension-suffix) \
 *     sphere_exposed_ext.c \
 *     -I$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))") \
 *     -I$(python3 -c "import numpy; print(numpy.get_include())") \
 *     $(python3-config --ldflags) -lm
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════════ */
#define PI  3.14159265358979323846
#define TWO_PI (2.0 * PI)
#define EPS    1e-14
#define EPS8   1e-8
#define EPS10  1e-10

/* Stack limits — sized for up to ~500 neighbors per sphere.
 * These go on the stack inside compute_one(), ~100KB total. */
#define MAX_CAPS      512
#define MAX_INTERVALS 512   /* max arc intervals across all lines for one cap */
#define MAX_DEGEN_PTS 256

typedef struct { double x, y, z; } Vec3;

static inline Vec3 v3(double x, double y, double z) { return (Vec3){x,y,z}; }
static inline Vec3 v3_add(Vec3 a, Vec3 b) { return v3(a.x+b.x, a.y+b.y, a.z+b.z); }
static inline Vec3 v3_sub(Vec3 a, Vec3 b) { return v3(a.x-b.x, a.y-b.y, a.z-b.z); }
static inline Vec3 v3_scale(Vec3 a, double s) { return v3(a.x*s, a.y*s, a.z*s); }
static inline double v3_dot(Vec3 a, Vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static inline Vec3 v3_cross(Vec3 a, Vec3 b) {
    return v3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
static inline double v3_len(Vec3 a) { return sqrt(v3_dot(a,a)); }
static inline Vec3 v3_norm(Vec3 a) {
    double l = v3_len(a); return l < EPS ? v3(0,0,0) : v3_scale(a, 1.0/l);
}
static inline double clampd(double x, double lo, double hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}
static inline double fmod_pos(double x, double m) {
    double r = fmod(x, m); return r < 0 ? r + m : r;
}

/* ═══════════════════════════════════════════════════════════════════ */
typedef struct {
    Vec3 normal; double d;
    Vec3 circle_center; double circle_radius;
    Vec3 local_u, local_v;
    double phi; int sphere_idx;
} Cap;

typedef struct { double start, end; } Interval;

/* ═══════════════════════════════════════════════════════════════════
 * Core geometry (all stack-based)
 * ═══════════════════════════════════════════════════════════════════ */

static Vec3 perpendicular_unit(Vec3 n) {
    Vec3 ref = (fabs(n.x) < 0.9) ? v3(1,0,0) : v3(0,1,0);
    return v3_norm(v3_cross(n, ref));
}

static int compute_cap(Vec3 mc, double mr, Vec3 oc, double or_, int idx, Cap *out) {
    Vec3 diff = v3_sub(oc, mc);
    double dist = v3_len(diff);
    if (dist < EPS) {
        if (or_ >= mr) { Vec3 n=v3(1,0,0); *out=(Cap){n,v3_dot(n,mc)-mr-1,mc,0,{0},{0},PI,idx}; return 1; }
        return 0;
    }
    if (dist >= mr + or_) return 0;
    if (dist + or_ <= mr) return 0;
    if (dist + mr <= or_) {
        Vec3 n = v3_scale(diff, 1.0/dist);
        *out = (Cap){n, v3_dot(n,mc)-mr-1, mc, 0, {0},{0}, PI, idx}; return 1;
    }
    Vec3 n = v3_scale(diff, 1.0/dist);
    double h = (mr*mr - or_*or_ + dist*dist) / (2.0*dist);
    *out = (Cap){
        n, v3_dot(n,mc)+h, v3_add(mc,v3_scale(n,h)),
        sqrt(fmax(0, mr*mr - h*h)),
        perpendicular_unit(n), v3_cross(n, perpendicular_unit(n)),
        acos(clampd(h/mr,-1,1)), idx
    };
    return 1;
}

static int compute_all_caps_stack(Vec3 mc, double mr, const double *oc,
                                  const double *or_, int n_others, Cap caps[MAX_CAPS]) {
    int nc = 0;
    for (int i = 0; i < n_others && nc < MAX_CAPS; i++) {
        Cap cap;
        if (!compute_cap(mc, mr, v3(oc[i*3],oc[i*3+1],oc[i*3+2]), or_[i], i, &cap)) continue;
        int dup = 0;
        for (int e = 0; e < nc; e++) {
            double dot = v3_dot(cap.normal, caps[e].normal);
            if (dot > 1-EPS8) {
                if (fabs(cap.d-caps[e].d)<EPS8) {dup=1;break;}
                else if (cap.d>caps[e].d) {dup=1;break;}
                else {caps[e]=caps[--nc];break;}
            }
        }
        if (!dup) caps[nc++] = cap;
    }
    return nc;
}

/* ═══════════════════════════════════════════════════════════════════
 * Interval intersection (stack buffers)
 * ═══════════════════════════════════════════════════════════════════ */

static int intersect_iv(const Interval *a, int na, const Interval *b, int nb,
                        Interval *out, int max_out) {
    int cnt = 0;
    for (int i = 0; i < na; i++)
        for (int j = 0; j < nb; j++) {
            double lo = fmax(a[i].start, b[j].start);
            double hi = fmin(a[i].end, b[j].end);
            if (hi - lo > 1e-12 && cnt < max_out) out[cnt++] = (Interval){lo,hi};
        }
    /* insertion sort */
    for (int i = 1; i < cnt; i++) {
        Interval k = out[i]; int j = i-1;
        while (j>=0 && out[j].start > k.start) { out[j+1]=out[j]; j--; }
        out[j+1] = k;
    }
    /* merge */
    int m = 0;
    for (int i = 0; i < cnt; i++) {
        if (m>0 && out[i].start <= out[m-1].end+1e-12)
            out[m-1].end = fmax(out[m-1].end, out[i].end);
        else out[m++] = out[i];
    }
    return m;
}

/* ═══════════════════════════════════════════════════════════════════
 * Exposed arcs for one cap (stack-based)
 * ═══════════════════════════════════════════════════════════════════ */

static int compute_arcs_for_cap(int hi, const Cap *caps, int nc,
                                Interval *result, int *feasible) {
    *feasible = 1;
    const Cap *host = &caps[hi];
    if (host->phi >= PI-EPS10 || host->circle_radius < EPS) return 0;
    double R = host->circle_radius;

    /* Collect lines */
    int nlines = 0;
    double la[MAX_CAPS], lb[MAX_CAPS], lc[MAX_CAPS];
    for (int j = 0; j < nc; j++) {
        if (j == hi) continue;
        double a = -v3_dot(caps[j].normal, host->local_u);
        double b = -v3_dot(caps[j].normal, host->local_v);
        double c = -(caps[j].d - v3_dot(caps[j].normal, host->circle_center));
        if (fabs(a)<EPS && fabs(b)<EPS) {
            if (v3_dot(caps[j].normal, host->circle_center)-caps[j].d > EPS10)
                { *feasible=0; return 0; }
            continue;
        }
        la[nlines]=a; lb[nlines]=b; lc[nlines]=c; nlines++;
    }

    Interval iv[MAX_INTERVALS], tmp[MAX_INTERVALS], con[2];
    int niv = 1;
    iv[0] = (Interval){0, TWO_PI};

    for (int li = 0; li < nlines; li++) {
        double a=la[li], b=lb[li], c=lc[li];
        double A = sqrt(a*a+b*b);
        if (A<EPS) { if (c>EPS10) return 0; continue; }
        double ratio = c/(R*A);
        if (ratio <= -1.0-EPS10) continue;
        if (ratio >= 1.0+EPS10) return 0;
        ratio = clampd(ratio,-1,1);
        double alpha=atan2(b,a), delta=acos(ratio);
        double s=fmod_pos(alpha-delta,TWO_PI), e=fmod_pos(alpha+delta,TWO_PI);
        int ncn;
        if (s<e) {con[0]=(Interval){s,e}; ncn=1;}
        else {con[0]=(Interval){s,TWO_PI}; con[1]=(Interval){0,e}; ncn=2;}
        niv = intersect_iv(iv, niv, con, ncn, tmp, MAX_INTERVALS);
        if (niv==0) return 0;
        memcpy(iv, tmp, niv*sizeof(Interval));
    }
    memcpy(result, iv, niv*sizeof(Interval));
    return niv;
}

/* ═══════════════════════════════════════════════════════════════════
 * Degenerate point detection (stack-based)
 * ═══════════════════════════════════════════════════════════════════ */

static int is_pt_exposed(Vec3 pt, const Cap *caps, int n, int e1, int e2) {
    for (int k = 0; k < n; k++) {
        if (k==e1||k==e2) continue;
        if (v3_dot(caps[k].normal, pt)-caps[k].d > EPS8) return 0;
    }
    return 1;
}

static int solve3(const double A[9], const double b[3], double x[3]) {
    double det = A[0]*(A[4]*A[8]-A[5]*A[7])-A[1]*(A[3]*A[8]-A[5]*A[6])
                +A[2]*(A[3]*A[7]-A[4]*A[6]);
    if (fabs(det)<EPS) return 0;
    double inv=1.0/det;
    x[0]=inv*(b[0]*(A[4]*A[8]-A[5]*A[7])-A[1]*(b[1]*A[8]-A[5]*b[2])+A[2]*(b[1]*A[7]-A[4]*b[2]));
    x[1]=inv*(A[0]*(b[1]*A[8]-A[5]*b[2])-b[0]*(A[3]*A[8]-A[5]*A[6])+A[2]*(A[3]*b[2]-b[1]*A[6]));
    x[2]=inv*(A[0]*(A[4]*b[2]-b[1]*A[7])-A[1]*(A[3]*b[2]-b[1]*A[6])+b[0]*(A[3]*A[7]-A[4]*A[6]));
    return 1;
}

static int find_degen_pts(Vec3 mc, double mr, const Cap *caps, int nc,
                          Vec3 pts[MAX_DEGEN_PTS]) {
    int np=0;
    for (int i=0; i<nc && np<MAX_DEGEN_PTS; i++) {
        if (caps[i].circle_radius<EPS) continue;
        for (int j=i+1; j<nc && np<MAX_DEGEN_PTS; j++) {
            if (caps[j].circle_radius<EPS) continue;
            Vec3 ni=caps[i].normal, nj=caps[j].normal;
            Vec3 ld=v3_cross(ni,nj); double ldn=v3_len(ld);
            if (ldn<1e-12) continue;
            ld=v3_scale(ld,1.0/ldn);
            double A[9]={ni.x,ni.y,ni.z,nj.x,nj.y,nj.z,ld.x,ld.y,ld.z};
            double bv[3]={caps[i].d,caps[j].d,0}, p0[3];
            if (!solve3(A,bv,p0)) continue;
            Vec3 p0v=v3(p0[0],p0[1],p0[2]), d=v3_sub(p0v,mc);
            double bc=2*v3_dot(d,ld), cc=v3_dot(d,d)-mr*mr;
            double disc=bc*bc-4*cc;
            if (disc<-EPS10) continue;
            if (disc<0) disc=0;
            double sq=sqrt(disc);
            for (int s=-1;s<=1;s+=2) {
                double t=(-bc+s*sq)/2;
                Vec3 pt=v3_add(p0v,v3_scale(ld,t));
                if (fabs(v3_len(v3_sub(pt,mc))-mr)>1e-6) continue;
                if (!is_pt_exposed(pt,caps,nc,i,j)) continue;
                int dup=0;
                for (int k=0;k<np;k++) if (v3_len(v3_sub(pt,pts[k]))<1e-6) {dup=1;break;}
                if (!dup) pts[np++]=pt;
            }
        }
    }
    return np;
}

/* ═══════════════════════════════════════════════════════════════════
 * compute_one: stack-only, writes results to caller's arrays
 * ═══════════════════════════════════════════════════════════════════ */

/* Arc storage: caller passes pre-allocated arrays.
 * We write up to *max_arcs entries and return actual count. */
static void compute_one(
    Vec3 mc, double mr, const double *oc, const double *or_, int n_others,
    /* outputs: */
    int *out_ncaps,
    int *out_narcs, int *arc_cap, double *arc_s, double *arc_e, int max_arcs,
    double *out_total_arc,
    int *out_npts, Vec3 *out_pts /* [MAX_DEGEN_PTS] */)
{
    Cap caps[MAX_CAPS];
    int nc = compute_all_caps_stack(mc, mr, oc, or_, n_others, caps);
    *out_ncaps = nc;
    *out_total_arc = 0;

    int na = 0;
    Interval arcs_buf[MAX_INTERVALS];
    for (int i = 0; i < nc; i++) {
        int feasible;
        int n = compute_arcs_for_cap(i, caps, nc, arcs_buf, &feasible);
        for (int a = 0; a < n && na < max_arcs; a++) {
            arc_cap[na] = i;
            arc_s[na] = arcs_buf[a].start;
            arc_e[na] = arcs_buf[a].end;
            *out_total_arc += arcs_buf[a].end - arcs_buf[a].start;
            na++;
        }
    }
    *out_narcs = na;

    *out_npts = 0;
    if (*out_total_arc < 1e-6 && nc > 0) {
        *out_npts = find_degen_pts(mc, mr, caps, nc, out_pts);
    }
}


/* ═══════════════════════════════════════════════════════════════════
 * Python: compute_exposed_single → dict
 * ═══════════════════════════════════════════════════════════════════ */

static PyObject* py_compute_single(PyObject *self, PyObject *args) {
    PyArrayObject *py_mc, *py_oc, *py_or;
    double mr;
    if (!PyArg_ParseTuple(args, "O!dO!O!", &PyArray_Type, &py_mc, &mr,
                          &PyArray_Type, &py_oc, &PyArray_Type, &py_or))
        return NULL;
    int n = (int)PyArray_DIM(py_oc, 0);
    double *mc = (double*)PyArray_DATA(py_mc);

    int ncaps, narcs, npts;
    double total_arc;
    int arc_cap[MAX_INTERVALS*2]; /* generous */
    double arc_s[MAX_INTERVALS*2], arc_e[MAX_INTERVALS*2];
    Vec3 dpts[MAX_DEGEN_PTS];

    compute_one(v3(mc[0],mc[1],mc[2]), mr,
                (double*)PyArray_DATA(py_oc), (double*)PyArray_DATA(py_or), n,
                &ncaps, &narcs, arc_cap, arc_s, arc_e, MAX_INTERVALS*2,
                &total_arc, &npts, dpts);

    /* Build arcs_by_cap dict */
    PyObject *ad = PyDict_New();
    for (int i=0;i<ncaps;i++) {
        PyObject *k=PyLong_FromLong(i), *l=PyList_New(0);
        PyDict_SetItem(ad,k,l); Py_DECREF(k); Py_DECREF(l);
    }
    for (int a=0;a<narcs;a++) {
        PyObject *k=PyLong_FromLong(arc_cap[a]);
        PyObject *l=PyDict_GetItem(ad,k);
        PyObject *t=Py_BuildValue("(dd)",arc_s[a],arc_e[a]);
        PyList_Append(l,t); Py_DECREF(t); Py_DECREF(k);
    }
    /* Build points */
    PyObject *pl = PyList_New(npts);
    for (int i=0;i<npts;i++) {
        npy_intp d[1]={3};
        PyObject *p=PyArray_SimpleNew(1,d,NPY_DOUBLE);
        double *pd=(double*)PyArray_DATA((PyArrayObject*)p);
        pd[0]=dpts[i].x; pd[1]=dpts[i].y; pd[2]=dpts[i].z;
        PyObject *oc_list=PyList_New(0); /* on_caps not needed for single */
        PyList_SET_ITEM(pl,i,Py_BuildValue("(OO)",p,oc_list));
        Py_DECREF(p); Py_DECREF(oc_list);
    }
    return Py_BuildValue("{s:O,s:O,s:d,s:i}","arcs_by_cap",ad,
                         "exposed_points",pl,"total_arc",total_arc,"n_caps",ncaps);
}

/* ═══════════════════════════════════════════════════════════════════
 * Python: compute_exposed_batch → compact numpy arrays
 *
 * Input: centers (N,3), radii (N,), neighbor_indices (N,) int64 flat,
 *        neighbor_offsets (N+1,) int64 (CSR format)
 *
 * Using CSR avoids per-sphere PyList_GetItem + PyArray_FromAny overhead.
 * Call from Python:
 *   indices = np.concatenate(nbr_list)
 *   offsets = np.zeros(N+1, dtype=np.int64)
 *   for i,nb in enumerate(nbr_list): offsets[i+1] = offsets[i] + len(nb)
 *   result = ext.compute_exposed_batch(centers, radii, indices, offsets)
 * ═══════════════════════════════════════════════════════════════════ */

static PyObject* py_compute_batch(PyObject *self, PyObject *args) {
    PyArrayObject *py_c, *py_r, *py_idx, *py_off;
    if (!PyArg_ParseTuple(args, "O!O!O!O!",
            &PyArray_Type, &py_c, &PyArray_Type, &py_r,
            &PyArray_Type, &py_idx, &PyArray_Type, &py_off))
        return NULL;

    int N = (int)PyArray_DIM(py_c, 0);
    double *centers = (double*)PyArray_DATA(py_c);
    double *radii   = (double*)PyArray_DATA(py_r);
    int64_t *nbr_idx = (int64_t*)PyArray_DATA(py_idx);
    int64_t *nbr_off = (int64_t*)PyArray_DATA(py_off);

    /* Per-sphere summary */
    npy_intp dims_n[1] = {N};
    PyObject *py_ncaps = PyArray_SimpleNew(1, dims_n, NPY_INT32);
    PyObject *py_narcs = PyArray_SimpleNew(1, dims_n, NPY_INT32);
    PyObject *py_npts  = PyArray_SimpleNew(1, dims_n, NPY_INT32);
    PyObject *py_tarc  = PyArray_SimpleNew(1, dims_n, NPY_FLOAT64);
    int32_t *s_ncaps = (int32_t*)PyArray_DATA((PyArrayObject*)py_ncaps);
    int32_t *s_narcs = (int32_t*)PyArray_DATA((PyArrayObject*)py_narcs);
    int32_t *s_npts  = (int32_t*)PyArray_DATA((PyArrayObject*)py_npts);
    double  *s_tarc  = (double*)PyArray_DATA((PyArrayObject*)py_tarc);

    /* Growable flat arc/point storage */
    int arcs_alloc = N * 4, total_arcs = 0;
    int *fa_sphere = (int*)malloc(arcs_alloc * sizeof(int));
    int *fa_cap    = (int*)malloc(arcs_alloc * sizeof(int));
    double *fa_s   = (double*)malloc(arcs_alloc * sizeof(double));
    double *fa_e   = (double*)malloc(arcs_alloc * sizeof(double));

    int pts_alloc = 256, total_pts = 0;
    int *fp_sphere = (int*)malloc(pts_alloc * sizeof(int));
    double *fp_pos = (double*)malloc(pts_alloc * 3 * sizeof(double));

    /* Per-sphere scratch (on stack) */
    int    _ac[MAX_INTERVALS * 2];
    double _as[MAX_INTERVALS * 2], _ae[MAX_INTERVALS * 2];
    Vec3   _dp[MAX_DEGEN_PTS];

    /* Temporary neighbor gather buffer (reuse across iterations) */
    int oc_alloc = 1024;
    double *oc_buf = (double*)malloc(oc_alloc * 3 * sizeof(double));
    double *or_buf = (double*)malloc(oc_alloc * sizeof(double));

    for (int i = 0; i < N; i++) {
        int n_nbrs = (int)(nbr_off[i+1] - nbr_off[i]);
        if (n_nbrs == 0) {
            s_ncaps[i]=0; s_narcs[i]=0; s_npts[i]=0; s_tarc[i]=0;
            continue;
        }

        /* Grow gather buffer if needed */
        if (n_nbrs > oc_alloc) {
            oc_alloc = n_nbrs * 2;
            oc_buf = (double*)realloc(oc_buf, oc_alloc * 3 * sizeof(double));
            or_buf = (double*)realloc(or_buf, oc_alloc * sizeof(double));
        }

        /* Gather neighbor data */
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
        compute_one(v3(centers[i*3],centers[i*3+1],centers[i*3+2]), radii[i],
                    oc_buf, or_buf, n_nbrs,
                    &ncaps, &narcs, _ac, _as, _ae, MAX_INTERVALS*2,
                    &total_arc, &npts, _dp);

        s_ncaps[i] = ncaps;
        s_narcs[i] = narcs;
        s_npts[i]  = npts;
        s_tarc[i]  = total_arc;

        /* Append arcs */
        while (total_arcs + narcs > arcs_alloc) {
            arcs_alloc *= 2;
            fa_sphere = (int*)realloc(fa_sphere, arcs_alloc*sizeof(int));
            fa_cap    = (int*)realloc(fa_cap,    arcs_alloc*sizeof(int));
            fa_s      = (double*)realloc(fa_s,   arcs_alloc*sizeof(double));
            fa_e      = (double*)realloc(fa_e,   arcs_alloc*sizeof(double));
        }
        for (int a=0; a<narcs; a++) {
            fa_sphere[total_arcs]=i; fa_cap[total_arcs]=_ac[a];
            fa_s[total_arcs]=_as[a]; fa_e[total_arcs]=_ae[a];
            total_arcs++;
        }

        /* Append points */
        while (total_pts + npts > pts_alloc) {
            pts_alloc *= 2;
            fp_sphere = (int*)realloc(fp_sphere, pts_alloc*sizeof(int));
            fp_pos    = (double*)realloc(fp_pos,  pts_alloc*3*sizeof(double));
        }
        for (int p=0; p<npts; p++) {
            fp_sphere[total_pts]=i;
            fp_pos[total_pts*3]=_dp[p].x;
            fp_pos[total_pts*3+1]=_dp[p].y;
            fp_pos[total_pts*3+2]=_dp[p].z;
            total_pts++;
        }
    }

    free(oc_buf); free(or_buf);

    /* Build output numpy arrays */
    #define WRAP_INT(ptr, len) do {                               \
        npy_intp d[1]={len};                                      \
        PyObject *a=PyArray_SimpleNew(1,d,NPY_INT32);             \
        memcpy(PyArray_DATA((PyArrayObject*)a),ptr,(len)*4);      \
        free(ptr); ptr##_arr=a; } while(0)

    #define WRAP_DBL(ptr, len) do {                               \
        npy_intp d[1]={len};                                      \
        PyObject *a=PyArray_SimpleNew(1,d,NPY_FLOAT64);           \
        memcpy(PyArray_DATA((PyArrayObject*)a),ptr,(len)*8);      \
        free(ptr); ptr##_arr=a; } while(0)

    PyObject *fa_sphere_arr, *fa_cap_arr, *fa_s_arr, *fa_e_arr;
    WRAP_INT(fa_sphere, total_arcs);
    WRAP_INT(fa_cap, total_arcs);
    WRAP_DBL(fa_s, total_arcs);
    WRAP_DBL(fa_e, total_arcs);

    PyObject *fp_sphere_arr;
    WRAP_INT(fp_sphere, total_pts);

    npy_intp pd[2]={total_pts, 3};
    PyObject *fp_pos_arr = PyArray_SimpleNew(2, pd, NPY_FLOAT64);
    memcpy(PyArray_DATA((PyArrayObject*)fp_pos_arr), fp_pos, total_pts*3*8);
    free(fp_pos);

    #undef WRAP_INT
    #undef WRAP_DBL

    return Py_BuildValue(
        "{s:O,s:O,s:O,s:O, s:O,s:O,s:O,s:O, s:O,s:O}",
        "n_caps",py_ncaps, "n_arcs",py_narcs, "n_points",py_npts, "total_arc",py_tarc,
        "arc_sphere_idx",fa_sphere_arr, "arc_cap_idx",fa_cap_arr,
        "arc_start",fa_s_arr, "arc_end",fa_e_arr,
        "point_sphere_idx",fp_sphere_arr, "point_positions",fp_pos_arr);
}

/* ═══════════════════════════════════════════════════════════════════ */
static PyMethodDef methods[] = {
    {"compute_exposed_single", py_compute_single, METH_VARARGS,
     "compute_exposed_single(center, radius, other_centers, other_radii) -> dict"},
    {"compute_exposed_batch", py_compute_batch, METH_VARARGS,
     "compute_exposed_batch(centers, radii, nbr_indices, nbr_offsets) -> dict\n"
     "  nbr_indices: int64 flat array of all neighbor indices\n"
     "  nbr_offsets: int64 array of length N+1 (CSR format)"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "sphere_exposed_ext",
    "Sphere exposed region computation (v3: fast + compact)", -1, methods
};

PyMODINIT_FUNC PyInit_sphere_exposed_ext(void) {
    import_array();
    return PyModule_Create(&module);
}