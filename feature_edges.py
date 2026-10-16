"""Sharp feature curves from the thin exposed regions of the SDF spheres.

A sphere whose nearest surface point lies on a sharp edge keeps an exposed
region that is a short sliver lying ALONG the edge: the edge is a line tangent
to the sphere, and the samples pin how far away that line is but not where
along it the contact sits (see debug/vis_thin_exposed.py for the regions and
their thinness). Each such region is therefore a little piece of the edge:

    midpoint  p   the region's arc points averaged and pushed back onto the
                  sphere: a point on the edge
    direction t   main PCA axis of the arc points: the edge tangent
    half-len  hl  half the region's extent along t

On fandisk@50 with thinness >= 10 and total angle < pi, 95% of the midpoints
are within 0.05h of a GT sharp edge (median 0.0016h, h = grid spacing) and t is
within 10 degrees of the GT tangent for 97% of them.

Chaining
  The pieces are accurate across the edge but may leave gaps along it, so the
  neighbourhood is a thin cylinder along t, not a ball. Pieces i and j may link
  when
    - their directions agree within `max_angle_deg`;
    - their offset is parallel to the mean tangent: the lateral distance from
      that line is below `lat_tol`. On a circular edge the chord between two
      points is parallel to the bisector of their tangents, so this holds on
      curved edges too, without a curvature allowance;
    - the gap between their facing ends is at most `max_gap`;
    - the chord between them does not enter any SDF sphere deeper than
      `pen_tol` (tested on the tree links below): the true edge is on the
      surface, and a sphere holds no surface, so a link that cuts through a
      sphere jumps somewhere it should not (e.g. across to a parallel edge).
  All pairs that pass are candidate links; their minimum spanning tree holds
  each curve together, and each tree's longest path is a feature polyline.
  (Linking each piece to its mutual nearest neighbour fragments the curves:
  near-duplicate pieces from spheres touching the same edge point have a
  noise-level order along t.)  A closed loop comes out as one open path.

  Defaults are in units of h. Measured on fandisk@50 with lateral tolerance
  0.05h: 99.6% of the gaps to the next piece are <= 1h (median -0.12h: the
  pieces mostly overlap), and 2 of 3803 nearest links went off the GT edge.

Usage
  .venv/bin/python feature_edges.py --name fandisk --grid-len 50
  .venv/bin/python feature_edges.py --name fandisk --grid-len 20 --max-gap 2
Writes out/<name>/feature_edges_<name>_<grid>.obj (one 'l' polyline per
chain) and, when examples/<name>_sharp_edges.npz exists, scores the chains
against the GT sharp edges in it.
"""
import argparse
import os
import sys
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, 'debug'))

import vis_arcs_near_edges as vae          # noqa: E402
import vis_thin_exposed as vte             # noqa: E402


# ── one edge sample per thin region ──────────────────────────────────

def edge_samples(batch, m, sdf_points, radii, keep, samples_per_arc=8):
    """Midpoint, direction and half-length of every region in `keep`.

    `m` is vis_thin_exposed.sphere_areas(...) on `batch`, `keep` a bool mask
    over its regions. Returns a dict of arrays, one row per kept region.
    """
    arcs = m['arcs']
    reg = m['regions']
    rid = np.full(len(reg['area']), -1, np.int64)
    rid[keep] = np.arange(int(keep.sum()))
    ar = arcs['region']
    on = (ar >= 0)
    on[on] = keep[ar[on]]
    owner = rid[ar[on]]                                   # sample index of each arc
    n = int(keep.sum())

    rows = vae.arc_cap_rows(batch, len(sdf_points))[on]
    cc = np.asarray(batch['cap_centers'])[rows]
    rho = np.asarray(batch['cap_radii'])[rows]
    cu = np.asarray(batch['cap_u'])[rows]
    cv = np.asarray(batch['cap_v'])[rows]
    t0 = np.asarray(batch['arc_start'], float)[on]
    t1 = np.asarray(batch['arc_end'], float)[on]
    tt = t0[:, None] + (t1 - t0)[:, None] * np.linspace(0, 1, samples_per_arc)[None]
    q = (cc[:, None] + rho[:, None, None] * (np.cos(tt)[..., None] * cu[:, None]
                                             + np.sin(tt)[..., None] * cv[:, None]))
    q = q.reshape(-1, 3)
    o = np.repeat(owner, samples_per_arc)

    # per-sample mean and covariance of its arc points, all at once
    cnt = np.bincount(o, minlength=n).astype(float)
    mu = np.stack([np.bincount(o, weights=q[:, a], minlength=n) for a in range(3)], 1)
    mu /= cnt[:, None]
    d = q - mu[o]
    cov = np.zeros((n, 3, 3))
    for a in range(3):
        for b in range(a, 3):
            cov[:, a, b] = cov[:, b, a] = np.bincount(o, weights=d[:, a] * d[:, b],
                                                      minlength=n) / cnt
    w, U = np.linalg.eigh(cov)
    t = U[:, :, -1]

    sph = reg['sphere'][keep]
    c, r = sdf_points[sph], radii[sph]
    v = mu - c
    mid = c + r[:, None] * v / np.maximum(np.linalg.norm(v, axis=1), 1e-30)[:, None]

    s = np.einsum('ij,ij->i', q - mid[o], t[o])
    smax = np.full(n, -np.inf)
    smin = np.full(n, np.inf)
    np.maximum.at(smax, o, s)
    np.minimum.at(smin, o, s)
    return dict(mid=mid, t=t, hl=0.5 * (smax - smin), sphere=sph,
                region=np.flatnonzero(keep),
                elong=np.sqrt(w[:, -1] / np.maximum(w[:, -2], 1e-300)))


# ── chaining ─────────────────────────────────────────────────────────

def chord_penetration(a, b, sdf_points, radii, tree, step, k=16):
    """Deepest penetration into any SDF sphere along each chord a[i] -> b[i].

    Tested at points `step` apart against the k nearest sphere centers: a
    point near the surface can only be inside a sphere that reaches it, and
    those centers are close.
    """
    L = np.linalg.norm(b - a, axis=1)
    ns = np.maximum(np.ceil(L / step).astype(np.int64), 1) + 1
    link = np.repeat(np.arange(len(a)), ns)
    first = np.concatenate([[0], np.cumsum(ns)[:-1]])
    f = (np.arange(len(link)) - first[link]) / (ns[link] - 1)
    pts = a[link] + f[:, None] * (b - a)[link]
    kk = min(k, len(sdf_points))
    dist, idx = tree.query(pts, k=kk)
    dist, idx = dist.reshape(len(pts), kk), idx.reshape(len(pts), kk)
    pen = (radii[idx] - dist).max(1)
    out = np.full(len(a), -np.inf)
    np.maximum.at(out, link, pen)
    return out


def chain(samples, sdf_points, radii, h, max_gap=1.0, lat_tol=0.05,
          max_angle_deg=15.0, pen_tol=0.05, verbose=True):
    """Link the edge samples into polylines (see the module docstring).

    Every pair that passes the cylinder test is a candidate link. The minimum
    spanning tree of those links (weight: distance) keeps each curve in one
    piece without trusting a fragile "mutual nearest" choice: an edge point
    touched by several spheres yields near-duplicate samples whose order along
    t is noise (~1e-5 h), and mutual-nearest linking fragmented fandisk@50 into
    4805 chains of median length ~0. Only tree links are tested against the
    spheres; a link that cuts through one is removed, splitting the tree. Each
    tree's longest path is its polyline; the samples hanging off it are the
    near-duplicates and are not needed for the curve.

    max_gap, lat_tol and pen_tol are in units of h. Returns (polylines, info):
    each polyline is an index array into the samples, in order along the curve.
    """
    from scipy.sparse.csgraph import breadth_first_order, minimum_spanning_tree

    mid, t, hl = samples['mid'], samples['t'], samples['hl']
    n = len(mid)
    if n < 2:
        return [np.arange(n)] if n else [], dict(links=0)

    # Step 1. Candidate pairs: midpoints within max_gap plus both half-lengths
    # (half-length capped at h). A coarse filter only; most fail step 2.
    reach = max_gap * h + 2.0 * min(float(hl.max()), h)
    pairs = cKDTree(mid).query_pairs(reach, output_type='ndarray')
    I, J = pairs[:, 0], pairs[:, 1]

    # Step 2. Cylinder test: is j a piece of the same edge as i?
    #   - directions within max_angle_deg of each other;
    #   - p_j - p_i lies along the MEAN tangent tb: lateral offset <= lat_tol.
    #     On a circular edge the chord between two points is parallel to the
    #     bisector of their tangents, so curved edges pass without slack;
    #   - the gap between the facing ends (distance along tb minus both
    #     half-lengths; negative = overlap) is <= max_gap.
    # Long along the edge, thin across it: it bridges gaps without reaching
    # over to a parallel edge.
    dot = np.einsum('ij,ij->i', t[I], t[J])
    tb = t[I] + np.sign(dot)[:, None] * t[J]             # mean tangent, signs aligned
    tb /= np.maximum(np.linalg.norm(tb, axis=1), 1e-30)[:, None]
    dv = mid[J] - mid[I]
    lon = np.einsum('ij,ij->i', dv, tb)
    lat = np.linalg.norm(dv - lon[:, None] * tb, axis=1)
    gap = np.abs(lon) - hl[I] - hl[J]
    ok = ((np.abs(dot) >= np.cos(np.radians(max_angle_deg)))
          & (lat <= lat_tol * h) & (gap <= max_gap * h))
    I, J = I[ok], J[ok]
    w = np.linalg.norm(dv[ok], axis=1) + 1e-12 * h         # zero weight = no edge

    # Step 3. Minimum spanning tree of the passing pairs (weight: distance).
    # One edge point is often touched by several spheres, so pieces come in
    # near-duplicates whose order along t is noise; the tree keeps each curve
    # connected and loop-free without relying on who is whose nearest.
    tree = minimum_spanning_tree(coo_matrix((w, (I, J)), shape=(n, n))).tocoo()
    a, b, wt = tree.row, tree.col, tree.data

    # Step 4. Penetration test on the tree links: the chord is sampled every
    # 0.1h and a link is cut if any sample is deeper than pen_tol inside an SDF
    # sphere. The true edge lies on the surface and a sphere holds no surface,
    # so such a link jumps somewhere it should not.
    pen = chord_penetration(mid[a], mid[b], sdf_points, radii,
                            cKDTree(sdf_points), step=0.1 * h)
    clear = pen <= pen_tol * h
    if verbose:
        print(f'[chain] {n} samples: {len(pairs)} pairs in reach, {len(I)} pass the '
              f'cylinder test, {len(a)} tree links, {int((~clear).sum())} cut through '
              f'a sphere (> {pen_tol}h) and dropped')
    a, b, wt = a[clear], b[clear], wt[clear]
    g = coo_matrix((wt, (a, b)), shape=(n, n)).tocsr()
    g = (g + g.T).tocsr()

    # Step 5. Each remaining tree becomes one polyline: its longest path,
    # found in two sweeps (the node u farthest from any start, then the node v
    # farthest from u; the path u -> v). Nodes off that path are the
    # near-duplicates hanging off it and are dropped.
    nc, comp = connected_components(g, directed=False)
    seen = np.zeros(n, bool)
    lines = []

    def farthest(src):
        order, pred = breadth_first_order(g, src, directed=False,
                                          return_predecessors=True)
        dist = np.zeros(n)
        for v in order[1:]:                               # parents come first
            dist[v] = dist[pred[v]] + g[pred[v], v]
        return order[np.argmax(dist[order])], pred

    for s0 in range(n):
        if seen[s0]:
            continue
        members = np.flatnonzero(comp == comp[s0])
        seen[members] = True
        if len(members) == 1:
            lines.append(members)
            continue
        u, _ = farthest(s0)
        v, pred = farthest(u)
        path = [v]
        while path[-1] != u:
            path.append(pred[path[-1]])
        lines.append(np.asarray(path[::-1]))
    return lines, dict(links=len(a), candidates=len(I), cut=int((~clear).sum()))


def polyline_length(p):
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0


# ── evaluation against GT sharp edges ────────────────────────────────

def load_gt_edges(path):
    """(E, 2, 3) GT sharp-edge segments from edge_samples.extract_sharp_edges."""
    g = np.load(path)
    return g['V'][g['edges']]


def _seg_distance(q, seg, chunk=2000):
    a, d = seg[:, 0], seg[:, 1] - seg[:, 0]
    L2 = np.maximum((d ** 2).sum(1), 1e-300)
    out = np.empty(len(q))
    for c in range(0, len(q), chunk):
        w = q[c:c + chunk, None] - a[None]
        s = np.clip(np.einsum('qej,ej->qe', w, d) / L2[None], 0, 1)
        out[c:c + chunk] = np.linalg.norm(w - s[..., None] * d[None], axis=2).min(1)
    return out


def _densify(p, step):
    out = [p[:1]]
    for a, b in zip(p[:-1], p[1:]):
        k = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        out.append(a + np.linspace(0, 1, k + 1)[1:, None] * (b - a))
    return np.vstack(out)


def score(polylines, gt_seg, h):
    """Precision / coverage of the polylines against GT segments, in units of h.

    precision: share of the polylines' length (sampled every 0.1h, so a link
    that jumps off the edge counts along its whole span) within 0.1h of a GT
    edge. coverage: share of the GT edges' length (sampled every 0.1h) within
    0.5h of a polyline.
    """
    step = 0.1 * h
    pp = [_densify(p, step) for p in polylines if len(p) > 1]
    if not pp:
        return dict(precision=float('nan'), coverage=0.0)
    pts = np.vstack(pp)
    precision = float((_seg_distance(pts, gt_seg) < 0.1 * h).mean())
    gt_pts = np.vstack([_densify(s, step) for s in gt_seg])
    d, _ = cKDTree(pts).query(gt_pts)
    return dict(precision=precision, coverage=float((d < 0.5 * h).mean()))


# ── output ───────────────────────────────────────────────────────────

def thin_index(p, min_step):
    """Indices of the vertices kept by thin_out."""
    if len(p) < 3 or min_step <= 0:
        return np.arange(len(p))
    keep, last = [0], p[0]
    for i in range(1, len(p) - 1):
        if np.linalg.norm(p[i] - last) >= min_step:
            keep.append(i)
            last = p[i]
    keep.append(len(p) - 1)
    return np.asarray(keep)


def thin_out(p, min_step):
    """Keep a vertex only once it is min_step from the last kept one (and
    always the last vertex): a chain runs through many near-duplicate samples."""
    return p[thin_index(p, min_step)]


def write_obj(path, polylines, min_step=0.0):
    """Polylines as two-vertex 'l' elements, one per segment.

    A single multi-vertex 'l' element is valid OBJ, but Blender's importer
    only turned the two-vertex ones into edges and left every longer chain as
    loose vertices, so each segment is written on its own.
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        base = 1
        for p in polylines:
            p = thin_out(p, min_step)
            if len(p) < 2:
                continue
            for v in p:
                f.write(f'v {v[0]:.7f} {v[1]:.7f} {v[2]:.7f}\n')
            for i in range(len(p) - 1):
                f.write(f'l {base + i} {base + i + 1}\n')
            base += len(p)
    return path


# ── the two face normals at an edge point ───────────────────────────

def face_normals(p, t, sdf_points, sdf_values, h, radius=2.5, window=1.0,
                 tol=0.05, min_support=4, min_sep_deg=10.0):
    """Normals (n_A, n_B) of the two faces meeting at edge points p (tangent t).

    Uses the SDF values only. A sample x whose closest surface point lies on a
    face through p with normal n has signed distance s = n·(x - p), exactly for
    a planar face. The face contains the edge, so n is perpendicular to t and
    has one unknown angle in the cross-section plane: with the sample at polar
    angle phi and distance rho from p in that plane, s = rho*cos(theta - phi),
    so each sample votes theta = phi +- arccos(s/rho). The angle with the most
    samples fitting within `tol` is face A; with its samples removed, the best
    remaining angle at least `min_sep_deg` away is face B; each is refined by
    least squares on its own samples. Samples whose closest point is the edge
    itself fit neither face and drop out as outliers.

    Neighbours: SDF samples within `radius` of p and within `window` of it
    along t (all three in units of h, like `tol`). Returns nA, nB (n,3),
    outward (s > 0 outside), NaN where no face reaches `min_support` samples,
    and the support counts (n,2).
    """
    tree = cKDTree(sdf_points)
    nA = np.full((len(p), 3), np.nan)
    nB = np.full((len(p), 3), np.nan)
    support = np.zeros((len(p), 2), np.int64)
    tol_abs = tol * h
    for k, nb in enumerate(tree.query_ball_point(p, radius * h)):
        nb = np.asarray(nb)
        if len(nb) < 2 * min_support:
            continue
        tk = t[k]
        e1 = np.cross(tk, [1.0, 0.0, 0.0] if abs(tk[0]) < 0.9 else [0.0, 1.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(tk, e1)
        d = sdf_points[nb] - p[k]
        nb = nb[np.abs(d @ tk) <= window * h]
        d = sdf_points[nb] - p[k]
        q = np.stack([d @ e1, d @ e2], 1)                 # cross-section coords
        sv = sdf_values[nb]
        rho = np.linalg.norm(q, axis=1)
        ok = (rho > 1e-12) & (np.abs(sv) <= rho)
        if ok.sum() < 2 * min_support:
            continue
        q, sv, rho = q[ok], sv[ok], rho[ok]
        phi = np.arctan2(q[:, 1], q[:, 0])
        da = np.arccos(np.clip(sv / rho, -1.0, 1.0))
        cand = np.concatenate([phi + da, phi - da])
        cn = np.stack([np.cos(cand), np.sin(cand)], 1)    # (2m, 2)
        fit = np.abs(q @ cn.T - sv[:, None]) <= tol_abs   # (m, 2m)

        def best(mask_samples, exclude=None):
            score = fit[mask_samples].sum(0).astype(float)
            if exclude is not None:
                sep = np.abs(np.angle(np.exp(1j * (cand - exclude))))
                score[sep < np.radians(min_sep_deg)] = -1
            j = int(np.argmax(score))
            return j, int(score[j])

        out = []
        free = np.ones(len(sv), bool)
        prev = None
        for _ in range(2):
            j, sc = best(free, prev)
            if sc < min_support:
                break
            inl = free & fit[:, j]
            # least squares n.q = s on the inliers, then back to unit length
            n2 = np.linalg.lstsq(q[inl], sv[inl], rcond=None)[0]
            n2 /= max(np.linalg.norm(n2), 1e-30)
            out.append((n2, int(inl.sum())))
            free &= ~inl
            prev = cand[j]
        if len(out) == 2:
            (a2, ca), (b2, cb) = out
            nA[k] = a2[0] * e1 + a2[1] * e2
            nB[k] = b2[0] * e1 + b2[1] * e2
            support[k] = ca, cb
    return nA, nB, support


def chain_normals(smp, info, sdf_points, sdf_values, h, min_length=1.0,
                  min_step=0.02):
    """Thinned points of the chains at least `min_length` long (units of h),
    with their tangents and the two face normals. Rows without both normals
    are dropped. Returns dict(p, t, nA, nB)."""
    keep = [ix[thin_index(smp['mid'][ix], min_step * h)] for ix in info['paths']
            if polyline_length(smp['mid'][ix]) >= min_length * h]
    ix = np.concatenate(keep) if keep else np.zeros(0, np.int64)
    nA, nB, _ = face_normals(smp['mid'][ix], smp['t'][ix], sdf_points, sdf_values, h)
    ok = ~np.isnan(nA[:, 0])
    return dict(p=smp['mid'][ix][ok], t=smp['t'][ix][ok], nA=nA[ok], nB=nB[ok])


def gt_face_normals(path, sharp_angle_deg=15.0):
    """GT sharp-edge segments (E,2,3) with the normals of the two faces on
    either side (E,2,3), rebuilt from the npz of edge_samples.extract_sharp_edges."""
    import trimesh
    g = np.load(path)
    mesh = trimesh.Trimesh(vertices=g['V'], faces=g['F'], process=False)
    sharp = mesh.face_adjacency_angles > np.radians(sharp_angle_deg)
    seg = mesh.vertices[mesh.face_adjacency_edges[sharp]]
    return seg, mesh.face_normals[mesh.face_adjacency[sharp]]


def score_normals(p, nA, nB, gt_seg, gt_n, h):
    """Angle errors (degrees) of the estimated face normals against the GT
    faces of the nearest GT sharp edge, matching A/B to the two GT faces in
    whichever order fits better. Only points within 0.1h of a GT edge."""
    a, dvec = gt_seg[:, 0], gt_seg[:, 1] - gt_seg[:, 0]
    L2 = np.maximum((dvec ** 2).sum(1), 1e-300)
    found = ~np.isnan(nA[:, 0])
    near, jj = np.zeros(len(p), bool), np.zeros(len(p), np.int64)
    for c in range(0, len(p), 2000):
        w = p[c:c + 2000, None] - a[None]
        sp = np.clip(np.einsum('qej,ej->qe', w, dvec) / L2[None], 0, 1)
        dd = np.linalg.norm(w - sp[..., None] * dvec[None], axis=2)
        jj[c:c + 2000] = dd.argmin(1)
        near[c:c + 2000] = dd.min(1) < 0.1 * h
    use = found & near
    g1, g2 = gt_n[jj[use], 0], gt_n[jj[use], 1]
    ang = lambda u, v: np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i', u, v), -1, 1)))
    straight = np.maximum(ang(nA[use], g1), ang(nB[use], g2))
    swapped = np.maximum(ang(nA[use], g2), ang(nB[use], g1))
    return dict(err=np.minimum(straight, swapped), n_near=int(near.sum()),
                n_found=int(use.sum()))


# ── dual contouring: put the feature cells' vertices on the curves ────

def dc_vertex_cells(V, bbox_min, bbox_max, res):
    """Grid cell of every vertex of Interpolator.extract_surface's DC output.

    The C++ dual_contour emits one vertex per active cell, in increasing cell
    index (ci + Cx*(cj + Cy*ck)), clamped into that cell's box. A vertex clamped
    onto a shared face could belong to either cell; the increasing order picks
    the right one. V must be the raw output, before any vertex merging.
    Returns (cells (n,3) int, number of vertices that could not be placed).
    """
    bmin = np.asarray(bbox_min, float)
    step = (np.asarray(bbox_max, float) - bmin) / (res - 1)
    C = res - 1
    x = (np.asarray(V) - bmin) / step
    lo = np.clip(np.floor(x - 1e-7).astype(np.int64), 0, C - 1)
    hi = np.clip(np.floor(x + 1e-7).astype(np.int64), 0, C - 1)
    cells = np.empty((len(V), 3), np.int64)
    prev, bad = -1, 0
    for n_ in range(len(V)):
        best = None
        for ck in {lo[n_, 2], hi[n_, 2]}:
            for cj in {lo[n_, 1], hi[n_, 1]}:
                for ci in {lo[n_, 0], hi[n_, 0]}:
                    c = ci + C * (cj + C * ck)
                    if c > prev and (best is None or c < best[0]):
                        best = (c, ci, cj, ck)
        if best is None:                     # should not happen on raw output
            bad += 1
            best = (prev, *hi[n_])
        prev = best[0]
        cells[n_] = best[1:]
    return cells, bad


def snap_to_features(V, cells, polylines, bbox_min, bbox_max, res):
    """Move the vertex of every cell a feature polyline passes through onto
    the polyline (its closest point there).

    This stands in for the cell's QEF solution: with smooth interpolated
    normals the QEF lands on the rounded surface, while the polyline is the
    sharp edge itself. Returns (new V, mask of moved vertices, number of
    feature cells that have no vertex).
    """
    bmin = np.asarray(bbox_min, float)
    step = (np.asarray(bbox_max, float) - bmin) / (res - 1)
    C = res - 1
    q = np.vstack([_densify(p, step.min() / 8) for p in polylines if len(p) > 1])
    qc = np.clip(np.floor((q - bmin) / step).astype(np.int64), 0, C - 1)
    qkey = qc[:, 0] + C * (qc[:, 1] + C * qc[:, 2])
    vkey = cells[:, 0] + C * (cells[:, 1] + C * cells[:, 2])

    order = np.argsort(qkey, kind='stable')
    qkey_s, q_s = qkey[order], q[order]
    lo = np.searchsorted(qkey_s, vkey, 'left')
    hi = np.searchsorted(qkey_s, vkey, 'right')
    V2 = np.array(V, float, copy=True)
    moved = hi > lo
    for n_ in np.flatnonzero(moved):
        cand = q_s[lo[n_]:hi[n_]]
        V2[n_] = cand[np.argmin(((cand - V2[n_]) ** 2).sum(1))]
    n_feat_cells = len(np.unique(qkey))
    return V2, moved, n_feat_cells - int(np.isin(np.unique(qkey), vkey[moved]).sum())


def project_side_planes(V, cells, moved, feat, bbox_min, bbox_max, res,
                        radius_cells=1.5):
    """Put the DC vertices next to a feature onto the true face planes.

    For every vertex not already on a curve (`moved`) and within
    `radius_cells` cells of a feature point, pick the face it sits on (the
    nearer of the two planes through that point) and project it onto that
    plane, then clamp it back into its own cell as DC does. Next to an edge
    the interpolated gradients average the two faces, so the QEF rounds the
    edge over about a cell; the face planes are flat up to the edge itself.
    Returns (new V, mask of projected vertices).
    """
    bmin = np.asarray(bbox_min, float)
    step = (np.asarray(bbox_max, float) - bmin) / (res - 1)
    cell = step.min()
    V3 = np.array(V, float, copy=True)
    if len(feat['p']) == 0:
        return V3, np.zeros(len(V), bool)
    d, j = cKDTree(feat['p']).query(V3, distance_upper_bound=radius_cells * cell)
    use = np.isfinite(d) & ~moved
    j = j[use]
    off = V3[use] - feat['p'][j]
    dA = np.einsum('ij,ij->i', off, feat['nA'][j])
    dB = np.einsum('ij,ij->i', off, feat['nB'][j])
    a_side = np.abs(dA) <= np.abs(dB)
    n = np.where(a_side[:, None], feat['nA'][j], feat['nB'][j])
    dist = np.where(a_side, dA, dB)
    new = V3[use] - dist[:, None] * n
    lo = bmin + cells[use] * step
    V3[use] = np.clip(new, lo, lo + step)
    return V3, use


def run_dc_experiment(name, grid_len, lines, out_dir, data_dir='examples',
                      resolutions=None, feat=None):
    """Fit as test_our_method does, extract with DC, and compare the plain DC
    mesh with the one whose feature cells sit on `lines`.

    `resolutions`: DC grid resolutions to extract at from the one fit (default:
    test_our_method's adaptive one). Coarser grids round edges off more, which
    is where putting the feature cells on the curves should pay off most."""
    import SDF_to_surface_3D as sdf3d
    sys.path.insert(0, os.path.join(_HERE, 'cpp', 'build'))
    import sdf_cpp

    options = sdf3d.Options(name=name, grid_len=grid_len, use_MES=-1, clamp=True,
                            optim_steps=5, max_iters=10)
    gt, P, D, gt_grad = sdf3d.generate_test_mesh_data(
        f'{data_dir}/{name}.obj', name, grid_len=grid_len, save=False)
    options.gt_gradients, options.gt_mesh = gt_grad, gt
    t = time.perf_counter()
    result = sdf_cpp.main_algorithm(P, D, sdf3d._build_cpp_options(options))
    print(f'[dc] fit: {time.perf_counter() - t:.1f} s')

    bmin = P.min(0) - options.extract_padding
    bmax = P.max(0) + options.extract_padding
    adaptive = sdf3d._adaptive_resolution(P, grid_len)
    paths = {}
    for res in (resolutions or [adaptive]):
        paths.update(_dc_compare(result, gt, lines, bmin, bmax, res, options,
                                 out_dir, name, grid_len,
                                 suffix='' if res == adaptive else f'_res{res}',
                                 feat=feat))
    return paths


def _dc_compare(result, gt, lines, bmin, bmax, res, options, out_dir, name,
                grid_len, suffix='', feat=None):
    import trimesh
    import edgeChamfer
    from util import mesh_distances

    t = time.perf_counter()
    V, F = result.interpolator.extract_surface(
        bbox_min=bmin, bbox_max=bmax, nx=res, ny=res, nz=res, iso=0.0,
        chunk_size=5000, lipschitz_postfix=options.post_processing,
        use_dual_contouring=True)
    V, F = np.asarray(V), np.asarray(F)
    print(f'[dc] extract: {time.perf_counter() - t:.1f} s, res {res}, '
          f'{len(V)} vertices')

    cells, bad = dc_vertex_cells(V, bmin, bmax, res)
    V2, moved, missing = snap_to_features(V, cells, lines, bmin, bmax, res)
    shift = np.linalg.norm(V2[moved] - V[moved], axis=1)
    cell = ((bmax - bmin) / (res - 1)).min()
    print(f'[dc] {bad} vertices without a consistent cell; {int(moved.sum())} '
          f'feature-cell vertices moved onto the curves (shift/cell pct[50,90,max]: '
          + ', '.join(f'{x:.3g}' for x in np.percentile(shift / cell, [50, 90, 100]))
          + f'); {missing} feature cells have no DC vertex')

    variants = [('dc', V), ('dc_featsnap', V2)]
    if feat is not None:
        V3, proj = project_side_planes(V2, cells, moved, feat, bmin, bmax, res)
        sh = np.linalg.norm(V3[proj] - V2[proj], axis=1) / cell
        print(f'[dc] {int(proj.sum())} vertices next to the curves projected onto '
              'the face planes (shift/cell pct[50,90,max]: '
              + ', '.join(f'{x:.3g}' for x in np.percentile(sh, [50, 90, 100]))
              + ')' if proj.any() else '[dc] no vertex next to the curves')
        variants.append(('dc_featplane', V3))

    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for tag, verts in variants:
        mesh = trimesh.Trimesh(vertices=verts, faces=F, process=False)
        key = f'{tag}{suffix}'
        paths[key] = os.path.join(out_dir, f'{tag}_{name}_{grid_len}{suffix}.obj')
        mesh.export(paths[key])
        haus, cham, f1 = mesh_distances(mesh, gt)
        ecd, ef1 = edgeChamfer.compute_ecd(mesh, gt, sample_num=1_000_000)
        print(f'[dc] res {res:<4} {tag:<12} Hausdorff {haus:.5f}  Chamfer {cham:.7f}  F1 {f1:.4f}  '
              f'EdgeChamfer {ecd:.5f}  EdgeF1 {ef1:.4f}   -> {paths[key]}')
    return paths


# ── driver ───────────────────────────────────────────────────────────

def feature_edges(sdf_points, sdf_values, h, degen_tol=1e-5, min_thin=10.0,
                  max_area=None, min_perim_rad=0.0, max_total_angle=np.pi,
                  bbox=None, max_gap=1.0, lat_tol=0.05, max_angle_deg=15.0,
                  pen_tol=0.05, verbose=True):
    """Thin regions -> edge samples -> feature polylines.

    Returns (polylines as lists of points, samples, chain info).
    """
    sdf_points = np.ascontiguousarray(sdf_points, float)
    sdf_values = np.ascontiguousarray(sdf_values, float)
    radii = np.abs(sdf_values)

    t = time.perf_counter()
    batch, _ = vae.compute_exposed(sdf_points, sdf_values, degen_tol)
    m = vte.sphere_areas(batch, sdf_points, radii, verbose=False)
    keep = vte.select_regions(m, sdf_points, radii, min_thin=min_thin,
                              max_area=max_area, min_perim_rad=min_perim_rad,
                              max_total_angle=max_total_angle, bbox=bbox,
                              verbose=verbose)
    smp = edge_samples(batch, m, sdf_points, radii, keep)
    if verbose:
        print(f'[feature_edges] {len(smp["mid"])} edge samples from thin regions '
              f'({time.perf_counter() - t:.2f} s)')

    t = time.perf_counter()
    idx, info = chain(smp, sdf_points, radii, h, max_gap=max_gap, lat_tol=lat_tol,
                      max_angle_deg=max_angle_deg, pen_tol=pen_tol, verbose=verbose)
    lines = [smp['mid'][ix] for ix in idx]
    info['paths'] = idx                      # sample indices along each polyline
    if verbose:
        L = np.array([polyline_length(p) for p in lines]) / h
        print(f'[feature_edges] {len(lines)} chains '
              f'({int((L == 0).sum())} single samples); length/h pct[50,90,99,max]: '
              + ', '.join(f'{x:.3g}' for x in np.percentile(L, [50, 90, 99, 100]))
              + f'  ({time.perf_counter() - t:.2f} s)')
    return lines, smp, info


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--name', default='fandisk')
    ap.add_argument('--data-dir', default='examples')
    ap.add_argument('--grid-len', type=int, default=50)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--degen-tol', type=float, default=1e-5)
    ap.add_argument('--min-thin', type=float, default=10.0)
    ap.add_argument('--max-area', type=float, default=None)
    ap.add_argument('--min-perim-rad', type=float, default=0.0)
    ap.add_argument('--max-total-angle', type=float, default=np.pi)
    ap.add_argument('--max-gap', type=float, default=1.0, help='in units of h')
    ap.add_argument('--lat-tol', type=float, default=0.05, help='in units of h')
    ap.add_argument('--max-angle-deg', type=float, default=15.0)
    ap.add_argument('--pen-tol', type=float, default=0.05, help='in units of h')
    ap.add_argument('--min-length', type=float, default=0.0,
                    help='drop chains shorter than this (units of h) from the '
                         'output and the score')
    ap.add_argument('--gt', default=None,
                    help='GT sharp edges npz (default: <data-dir>/<name>_sharp_edges.npz '
                         'if it exists)')
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--normals', action='store_true',
                    help='estimate the two face normals along the chains and '
                         'score them against the GT mesh faces')
    ap.add_argument('--dc', action='store_true',
                    help='also fit as test_our_method does, extract with dual '
                         'contouring, and compare plain DC against DC with the '
                         'feature cells moved onto the chains')
    ap.add_argument('--dc-res', type=int, nargs='+', default=None,
                    help='DC resolutions to compare at (default: the adaptive one)')
    ap.add_argument('--dc-min-length', type=float, default=1.0,
                    help='chains shorter than this (units of h) are not used by --dc')
    args = ap.parse_args()

    import SDF_to_surface_3D as sdf3d
    sdf3d.data_dir = args.data_dir
    sdf3d.seed = args.seed
    _, P, V, _ = sdf3d.generate_test_mesh_data(
        f'{args.data_dir}/{args.name}.obj', args.name, grid_len=args.grid_len, save=False)
    h = float((P.max(0) - P.min(0)).max()) / (args.grid_len - 1)   # grid spacing
    print(f'{len(P)} SDF samples, h = {h:.5f}')

    lines, smp, info = feature_edges(P, V, h, degen_tol=args.degen_tol,
                                min_thin=args.min_thin, max_area=args.max_area,
                                min_perim_rad=args.min_perim_rad,
                                max_total_angle=args.max_total_angle,
                                max_gap=args.max_gap, lat_tol=args.lat_tol,
                                max_angle_deg=args.max_angle_deg, pen_tol=args.pen_tol)
    if args.min_length > 0:
        lines = [p for p in lines if polyline_length(p) >= args.min_length * h]
        print(f'{len(lines)} chains of length >= {args.min_length}h kept')

    out_dir = args.out_dir or f'out/{args.name}'
    obj = write_obj(os.path.join(out_dir, f'feature_edges_{args.name}_{args.grid_len}.obj'),
                    lines, min_step=0.02 * h)
    print(f'wrote {obj}')

    if args.normals:
        t0 = time.perf_counter()
        keep = [ix[thin_index(smp['mid'][ix], 0.02 * h)] for ix in info['paths']
                if polyline_length(smp['mid'][ix]) >= args.dc_min_length * h]
        ix = np.concatenate(keep)
        nA, nB, sup = face_normals(smp['mid'][ix], smp['t'][ix], P, V, h)
        found = ~np.isnan(nA[:, 0])
        print(f'[normals] {len(ix)} chain points, both faces found for '
              f'{int(found.sum())} ({100 * found.mean():.1f}%), support per face '
              f'median {np.median(sup[found], 0) if found.any() else "-"} '
              f'({time.perf_counter() - t0:.1f} s)')
        gt_path = args.gt or os.path.join(args.data_dir, f'{args.name}_sharp_edges.npz')
        if os.path.exists(gt_path):
            gs, gn = gt_face_normals(gt_path)
            sc = score_normals(smp['mid'][ix], nA, nB, gs, gn, h)
            e = sc['err']
            print(f'[normals] vs GT faces ({sc["n_found"]} of {sc["n_near"]} points '
                  f'on a GT edge have both): worse-of-two angle error deg '
                  f'pct[50,75,90,95,99]: '
                  + ', '.join(f'{x:.3g}' for x in np.percentile(e, [50, 75, 90, 95, 99]))
                  + f'; < 2deg: {100 * (e < 2).mean():.1f}%, < 5deg: {100 * (e < 5).mean():.1f}%')

    if args.dc:
        dc_lines = [p for p in lines if polyline_length(p) >= args.dc_min_length * h]
        print(f'[dc] using {len(dc_lines)} chains of length >= {args.dc_min_length}h')
        feat = chain_normals(smp, info, P, V, h, min_length=args.dc_min_length)
        print(f'[dc] {len(feat["p"])} chain points with both face normals')
        run_dc_experiment(args.name, args.grid_len, dc_lines, out_dir, args.data_dir,
                          resolutions=args.dc_res, feat=feat)

    gt = args.gt or os.path.join(args.data_dir, f'{args.name}_sharp_edges.npz')
    if os.path.exists(gt):
        sc = score(lines, load_gt_edges(gt), h)
        print(f'vs GT {gt}: precision {100 * sc["precision"]:.1f}% of chain length '
              f'within 0.1h of a GT edge, coverage {100 * sc["coverage"]:.1f}% of GT '
              'edge length within 0.5h of a chain')


if __name__ == '__main__':
    main()
