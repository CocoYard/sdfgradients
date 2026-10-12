"""Find and visualize the *thin* exposed regions: little area, long boundary.

`get_visible_arcs` leaves every SDF sphere with an exposed region — the part of
its surface that no neighbor sphere covers — and reports that region only by its
boundary arcs. A region can be a fat cap, or a sliver: a ribbon a few percent of
the radius wide that still runs most of the way around the sphere. Slivers are
the interesting failure shape, because the contact point the pipeline picks from
them is barely constrained sideways, so this script measures every sphere's
exposed area and perimeter, ranks the spheres by how thin that makes them, and
draws the worst ones.

Area, exactly, from the arcs (Gauss-Bonnet)
  A sphere of radius R at c is cut by caps: the plane n·p = d removes
  {p : n·p > d} (the query_inside convention in bench/sphere_exposed_pybind.cpp),
  so the exposed region is the sphere inside a convex polyhedron. Writing
  h = d - n·c for the plane's signed height above c, the boundary arc on that
  cap has geodesic curvature -h/(R*rho) and length rho*|dt|, so on each closed
  boundary loop Gauss-Bonnet (K = 1/R^2, chi = 1) gives

      A_loop = R^2 * (2*pi + sum_arcs (h/R)*|dt| - sum_corners theta)

  with theta the exterior turning angle at each corner, measured about the
  outward sphere normal. Loops are recovered by chaining arc endpoints; the
  traversal runs with *decreasing* t, which is the direction that keeps the
  exposed region on the left (verified by --self-test against the analytic
  two-sphere lens).

  A_loop is the area on the left of the loop, so a single-loop region is done.
  When a sphere has several loops (disjoint patches, or a cap poking a hole in
  the middle), h of them are hole boundaries and the total is
  sum(A_loop) - 4*pi*R^2*h. Since the total lies in [0, 4*pi*R^2], h is just
  floor(sum(A_loop) / (4*pi*R^2)). Nothing is sampled; Monte Carlo appears
  only in --self-test, as the independent check.

Thinness
  The spherical isoperimetric inequality is P^2 >= A*(4*pi - A/R^2), with
  equality exactly for a spherical cap, so

      thinness = P^2 / (A * (4*pi - A/R^2))   >= 1, = 1 for a cap

  is scale-free and measures "sliver" directly: 1 is a round cap, 10 is a ribbon
  ~10x longer than it is wide. `width = 2A/P` is the same thing in length units:
  the mean width of the ribbon.

Outputs (out/<name>/ by default)
  (the .csv and the two .html pages only with csv=True / html=True, or
   --csv / --html on the command line; the glb is always written)
  thin_exposed_<tag>_overview.html  every thin region at once on the GT mesh,
                                    boundaries bucketed by thinness
  thin_exposed_<tag>_detail.html    one thin sphere per slider step
                                    (--region-samples N also samples the
                                    exposed region itself as points)
  thin_exposed_<tag>.csv            the full ranking
  thin_exposed_<tag>.glb            boundary arcs only, as tubes, of EVERY
                                    connected region that passes both
                                    thinness and area (per region, not per
                                    sphere: a sphere's other regions are
                                    left out)

Usage
  .venv/bin/python debug/vis_thin_exposed.py --self-test
  .venv/bin/python debug/vis_thin_exposed.py --name eiffel --grid-len 30
  .venv/bin/python debug/vis_thin_exposed.py --name eiffel --grid-len 40 \
      --min-thin 8 --max-area 1e-4 --max-spheres 32
  .venv/bin/python debug/vis_thin_exposed.py --name eiffel --sort width
"""
import argparse
import os
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import vis_arcs_near_edges as vae          # noqa: E402  (shared arc/cap plumbing)

FOUR_PI = 4.0 * np.pi
PALETTE = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
           '#a65628', '#f781bf', '#00bfbf']


# ── per-arc geometry, in the traversal direction ─────────────────────

def arc_frames(batch, n_spheres, sdf_points, radii, check=True):
    """Everything the area formula needs, one row per arc.

    The traversal direction is *decreasing* t: at t the position is
    P(t) = cc + rho*(cos t * u + sin t * v), so the tangent is
    T(t) = sin t * u - cos t * v, and with (u, v, n) right-handed that is the
    direction that keeps the exposed region (n·p <= d) on the left.

    `head` is where the traversal enters the arc (t = arc_end) and `tail` where
    it leaves (t = arc_start), so a loop chains tail -> head.
    """
    rows = vae.arc_cap_rows(batch, n_spheres)
    cc = np.asarray(batch['cap_centers'])[rows]
    rho = np.asarray(batch['cap_radii'])[rows]
    cu = np.asarray(batch['cap_u'])[rows]
    cv = np.asarray(batch['cap_v'])[rows]
    cn = np.asarray(batch['cap_normals'])[rows]
    cd = np.asarray(batch['cap_d'])[rows]

    si = np.asarray(batch['arc_sphere_idx']).astype(np.int64)
    ctr = sdf_points[si]
    R = radii[si]
    h = cd - np.einsum('ij,ij->i', cn, ctr)      # plane height above the center

    if check and len(si):
        # confirms the plane convention: circle center = c + h*n, rho^2 = R^2 - h^2
        e_c = np.abs(cc - (ctr + h[:, None] * cn)).max() / max(R.max(), 1e-12)
        e_r = (np.abs(rho - np.sqrt(np.maximum(R ** 2 - h ** 2, 0.0)))
               / np.maximum(R, 1e-12)).max()
        assert e_c < 1e-6 and e_r < 1e-6, \
            f'cap plane/circle mismatch (center {e_c:.2e}, radius {e_r:.2e})'

    t0 = np.asarray(batch['arc_start'], float)
    t1 = np.asarray(batch['arc_end'], float)
    dt = t1 - t0

    def pos(t):
        return cc + rho[:, None] * (np.cos(t)[:, None] * cu + np.sin(t)[:, None] * cv)

    def tangent(t):
        return np.sin(t)[:, None] * cu - np.cos(t)[:, None] * cv

    # exact axis-aligned extent of each arc: the endpoints, plus the extrema of
    # x_a(t) = cc_a + rho*|(u_a, v_a)|*cos(t - phi_a), at t = phi_a + k*pi
    lo = np.minimum(pos(t0), pos(t1))
    hi = np.maximum(pos(t0), pos(t1))
    phi = np.arctan2(cv, cu)                       # (n_arcs, 3), one per axis
    first = phi + np.pi * np.ceil((t0[:, None] - phi) / np.pi)
    amp = rho[:, None] * np.hypot(cu, cv)
    for k in range(3):                             # dt <= 2*pi holds at most 3 extrema
        tc = first + k * np.pi
        inside = tc <= t1[:, None]
        x = cc + amp * np.cos(tc - phi)
        lo = np.where(inside, np.minimum(lo, x), lo)
        hi = np.where(inside, np.maximum(hi, x), hi)

    return dict(sphere=si, R=R, center=ctr, rho=rho, h=h, dt=dt,
                arclen=rho * dt, kappa=(h / R) * dt,
                head=pos(t1), tail=pos(t0), mid=pos(0.5 * (t0 + t1)),
                T_head=tangent(t1), T_tail=tangent(t0), lo=lo, hi=hi)


def chain_loops(arcs, tol_rel=1e-9, min_dt=1e-9, k=4):
    """Chain arcs tail -> head into closed loops.

    Returns (labels, nxt, theta, matched): `labels` is a loop id per arc (from
    the connected components of the chaining graph, -1 for dropped arcs),
    `theta` the exterior turning angle at the corner where the arc hands off to
    its successor, and `matched` False where no successor was found within
    tolerance — a loop with such an arc is open, which means the arc set is not
    a clean boundary and its area cannot be trusted.

    Three things real data does that a clean arrangement does not:
      - zero-length arcs (dt ~ 1e-15; over a third of all arcs on eiffel@30)
        pile up on single points. They carry no length and no curvature, so
        they are dropped and the corner is measured across them directly.
      - adjacent spheres share their triple points exactly, so a plain
        nearest-head search returns other spheres' heads. The search runs in
        (x, y, z, sphere id * BIG), which keeps it inside the sphere.
      - at a pinch, where two pieces of the region touch at a point, two heads
        sit on the same tail. Taking the one with the sharpest left turn is
        face tracing with the region on the left: it splits the pinch into two
        simple loops, which is what Gauss-Bonnet with chi = 1 needs.
      - arcs down to ~1e-9 R long exist, so near a pinch the other piece's
        heads are that close too. True endpoint gaps are ~1e-13 R, and the
        tolerance is 1e-9 R: at 1e-5 R, 3 of 204 eiffel@6 spheres chained
        across a pinch and came out up to 0.019 of the sphere off.
      - U-turns. When two neighbors cut along the same circle from opposite
        sides, the region is a zero-width band and its boundary runs out and
        back along one curve. There atan2(+-0, -1) returns +-pi by rounding
        noise, and one wrong sign turns area 0 into a hemisphere. The sign is
        read off which side of the incoming arc the returning arc lies on:
        on its left (or on it) the tip is a spike of the region, +pi.
    """
    n = len(arcs['sphere'])
    labels = np.full(n, -1, np.int64)
    nxt = np.full(n, -1, np.int64)
    theta = np.zeros(n)
    matched = np.ones(n, bool)
    live = np.flatnonzero(arcs['dt'] > min_dt)
    if live.size == 0:
        return labels, nxt, theta, matched

    sph = arcs['sphere'][live]
    big = 1e3 * (np.abs(arcs['center']).max() + arcs['R'].max() + 1.0)
    key = big * sph[:, None].astype(float)
    kk = min(k, live.size)
    dist, idx = cKDTree(np.hstack([arcs['head'][live], key])).query(
        np.hstack([arcs['tail'][live], key]), k=kk)
    dist = np.asarray(dist).reshape(-1, kk)
    idx = np.asarray(idx).reshape(-1, kk)
    ok = dist <= np.maximum(tol_rel * arcs['R'][live], 1e-12)[:, None]

    # exterior turning angle, about the outward sphere normal, per candidate
    t_in = arcs['T_tail'][live]
    u = (arcs['tail'][live] - arcs['center'][live]) / arcs['R'][live][:, None]
    left = np.cross(u, t_in)                  # the region's side of the incoming arc
    mid_in = arcs['mid'][live]
    th = np.empty(idx.shape)
    for c in range(kk):
        t_out = arcs['T_head'][live][idx[:, c]]
        dot = np.einsum('ij,ij->i', t_in, t_out)
        th[:, c] = np.arctan2(np.einsum('ij,ij->i', u, np.cross(t_in, t_out)), dot)
        uturn = dot < -1.0 + 1e-9
        if uturn.any():
            side = np.einsum('ij,ij->i', left,
                             arcs['mid'][live][idx[:, c]] - mid_in)
            spike = side >= -1e-9 * arcs['R'][live]
            th[:, c] = np.where(uturn, np.where(spike, np.pi, -np.pi), th[:, c])
    best = np.argmax(np.where(ok, th, -np.inf), axis=1)
    hit = ok.any(1)
    rows = np.arange(live.size)

    matched[live] = hit
    nxt[live] = np.where(hit, live[idx[rows, best]], -1)
    theta[live] = np.where(hit, th[rows, best], 0.0)

    e = live[hit]
    g = coo_matrix((np.ones(e.size), (e, nxt[e])), shape=(n, n))
    _, comp = connected_components(g, directed=False)
    # renumber so that only live arcs own loops, 0..L-1
    _, labels[live] = np.unique(comp[live], return_inverse=True)
    return labels, nxt, theta, matched


# ── areas ────────────────────────────────────────────────────────────

def loop_areas(arcs, labels):
    """Gauss-Bonnet area of the region on the left of every loop."""
    keep = labels >= 0
    arcs = {k: v[keep] for k, v in arcs.items()}
    labels = labels[keep]
    n_loops = int(labels.max()) + 1 if len(labels) else 0
    sphere = np.zeros(n_loops, np.int64)
    sphere[labels] = arcs['sphere']
    R = np.zeros(n_loops)
    R[labels] = arcs['R']
    kappa = np.bincount(labels, weights=arcs['kappa'], minlength=n_loops)
    theta = np.bincount(labels, weights=arcs['theta'], minlength=n_loops)
    perim = np.bincount(labels, weights=arcs['arclen'], minlength=n_loops)
    area = R ** 2 * (2.0 * np.pi + kappa - theta)
    return dict(sphere=sphere, R=R, area=area, perim=perim,
                n_arcs=np.bincount(labels, minlength=n_loops))


def fibonacci_sphere(m):
    """m roughly-uniform directions on the unit sphere."""
    i = np.arange(m) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / m)
    gold = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([np.cos(gold) * np.sin(phi),
                     np.sin(gold) * np.sin(phi), np.cos(phi)], axis=1)


def occluders(sdf_points, radii, i, tree):
    """The spheres that actually overlap sphere i."""
    j = np.asarray(tree.query_ball_point(sdf_points[i], radii[i] + radii.max()),
                   dtype=np.int64)
    if j.size == 0:
        return j
    d = np.linalg.norm(sdf_points[j] - sdf_points[i], axis=1)
    return j[(d < radii[j] + radii[i]) & (j != i)]


def exposed_mask(points, sdf_points, radii, occ, chunk=4096):
    """True where the point is outside every occluding sphere.

    This is the definition of exposed, tested against the neighbor spheres
    themselves. The cap planes exported in the batch are NOT a substitute: only
    caps that kept an arc are exported, and dropping the rest overestimates the
    region badly (measured: 0.65 vs 0.038 on one sphere of the random cloud).
    """
    m = np.ones(len(points), bool)
    if occ.size == 0:
        return m
    c, r = sdf_points[occ], radii[occ]
    for s in range(0, len(points), chunk):
        d = np.linalg.norm(points[s:s + chunk, None, :] - c[None, :, :], axis=2)
        m[s:s + chunk] = (d > r[None, :]).all(1)
    return m


def mc_area(sdf_points, radii, i, tree, dirs):
    """Monte-Carlo exposed area of sphere i, from the exact membership test."""
    R = float(radii[i])
    m = exposed_mask(sdf_points[i] + R * dirs, sdf_points, radii,
                     occluders(sdf_points, radii, i, tree))
    return FOUR_PI * R ** 2 * float(m.mean())


def sphere_areas(batch, sdf_points, radii, verbose=True):
    """Exposed area / perimeter / loop count for every sphere.

    Returns a dict of per-sphere arrays plus the per-arc and per-loop tables.
    """
    n = len(sdf_points)
    arcs = arc_frames(batch, n, sdf_points, radii)
    labels, nxt, theta, matched = chain_loops(arcs)
    arcs['theta'], arcs['label'], arcs['matched'] = theta, labels, matched
    loops = loop_areas(arcs, labels)

    area = np.bincount(loops['sphere'], weights=loops['area'], minlength=n)
    perim = np.bincount(loops['sphere'], weights=loops['perim'], minlength=n)
    n_loops = np.bincount(loops['sphere'], minlength=n)
    open_loop = np.zeros(n, bool)
    if len(arcs['sphere']):
        np.logical_or.at(open_loop, arcs['sphere'], ~matched)

    # loops that bound holes each contributed a whole sphere too much. The
    # true area is sum(A_loop) - 4 pi R^2 * holes and lies in [0, 4 pi R^2],
    # so the hole count is simply floor(sum / 4 pi R^2): no sampling needed.
    full = FOUR_PI * radii ** 2
    holes = np.where(open_loop, 0,
                     np.floor(area / np.maximum(full, 1e-300))).astype(np.int64)
    holes = np.clip(holes, 0, np.maximum(n_loops - 1, 0))
    area = area - holes * full
    if verbose and holes.any():
        print(f'{int((holes > 0).sum())} spheres have hole boundaries among '
              'their loops')

    area, thin, width, zero_width = thinness(area, perim, radii)
    regions, loop_region = split_regions(loops, holes, open_loop, radii)
    arcs['region'] = np.where(labels >= 0, loop_region[np.maximum(labels, 0)], -1)
    return dict(area=area, perim=perim, n_loops=n_loops, holes=holes,
                open_loop=open_loop, thin=thin, width=width, zero_width=zero_width,
                frac=area / np.maximum(full, 1e-30), arcs=arcs, loops=loops,
                regions=regions)


def thinness(area, perim, R):
    """(area, thinness, width, zero_width) for regions on spheres of radius R.

    A region with a boundary but no area (a band along one circle, see
    chain_loops) is the thinnest there is: thinness = inf, width = 0.
    """
    full = FOUR_PI * R ** 2
    area = np.clip(area, 0.0, full)
    zero_width = (perim > 0) & (area <= 1e-9 * full)
    area = np.where(zero_width, 0.0, area)
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = area * np.maximum(FOUR_PI - area / np.maximum(R, 1e-30) ** 2, 0.0)
        thin = perim ** 2 / denom
        width = np.where(perim > 0, 2.0 * area / np.maximum(perim, 1e-30), 0.0)
    thin = np.where(zero_width, np.inf,
                    np.where((denom > 0) & (perim > 0), thin, np.nan))
    return area, thin, width, zero_width


def split_regions(loops, holes, open_loop, radii):
    """Group boundary loops into connected exposed regions.

    Each outer loop bounds one region; a hole loop belongs to the region it
    punches. On a sphere with h holes, the hole loops are the h loops with the
    largest left-hand area (a hole loop's is 4 pi R^2 minus the small hole).
    When such a sphere has several outer loops, which hole sits in which is not
    resolved and its loops are merged into one region.

    Returns (regions dict of per-region arrays, loop -> region index).
    """
    nL = len(loops['area'])
    full_l = FOUR_PI * loops['R'] ** 2
    is_hole = np.zeros(nL, bool)
    region = np.arange(nL)
    merged = np.zeros(nL, bool)
    hs = np.flatnonzero(holes > 0)
    if hs.size:
        order = np.argsort(loops['sphere'], kind='stable')
        srt = loops['sphere'][order]
        for i, a, b in zip(hs, np.searchsorted(srt, hs, 'left'),
                           np.searchsorted(srt, hs, 'right')):
            ls = order[a:b]
            ls = ls[np.argsort(-loops['area'][ls])]
            hl, ol = ls[:holes[i]], ls[holes[i]:]
            is_hole[hl] = True
            if len(ol) == 1:
                region[hl] = ol[0]
            else:
                region[ls] = ls[0]
                merged[ls] = True
    # a hole loop's left-hand area is the sphere minus the hole
    contrib = np.where(is_hole, loops['area'] - full_l, loops['area'])
    uniq, rid = np.unique(region, return_inverse=True)
    n = len(uniq)
    sph = loops['sphere'][uniq]
    R = radii[sph]
    area, thin, width, zw = thinness(np.bincount(rid, weights=contrib, minlength=n),
                                     np.bincount(rid, weights=loops['perim'], minlength=n),
                                     R)
    perim = np.bincount(rid, weights=loops['perim'], minlength=n)
    return dict(sphere=sph, area=area, perim=perim, thin=thin, width=width,
                zero_width=zw, frac=area / np.maximum(FOUR_PI * R ** 2, 1e-30),
                n_loops=np.bincount(rid, minlength=n),
                merged=np.bincount(rid, weights=merged, minlength=n) > 0,
                valid=~open_loop[sph]), rid


# ── self-test ────────────────────────────────────────────────────────

def _pad(pts, vals, n_min=24, far=25.0):
    """Pad a tiny configuration with harmless far-away spheres.

    get_visible_arcs unconditionally prints ngbrs_list[0..19], so it needs at
    least 20 spheres; these sit far outside and touch nothing.
    """
    k = max(0, n_min - len(pts))
    if k == 0:
        return np.asarray(pts, float), np.asarray(vals, float)
    d = fibonacci_sphere(k) * far
    return (np.vstack([pts, d]),
            np.concatenate([vals, np.full(k, 1e-3)]))


def self_test(degen_tol=1e-9):
    """Check the area formula against an analytic lens and against Monte Carlo.

    Two spheres r1 at origin and r2 at distance D: the plane cutting sphere 1
    sits at height h = (D^2 + r1^2 - r2^2) / (2D) above the center, the cap
    towards the neighbor is removed, so the exposed area is 2*pi*r1*(r1 + h).
    Getting this right is what fixes the traversal orientation: the opposite
    orientation returns 2*pi*r1*(r1 - h) instead.
    """
    print('self-test 1: two-sphere lens vs analytic 2*pi*r1*(r1 + h)')
    worst = 0.0
    for r1, r2, D in [(0.30, 0.20, 0.40), (0.30, 0.20, 0.25),
                      (0.10, 0.50, 0.45), (0.25, 0.25, 0.10)]:
        pts, vals = _pad([[0.0, 0.0, 0.0], [D, 0.0, 0.0]], [r1, r2])
        batch, _ = vae.compute_exposed(pts, vals, degen_tol)
        got = sphere_areas(batch, pts, np.abs(vals), verbose=False)
        for i, (ri, rj) in enumerate([(r1, r2), (r2, r1)]):
            h = (D ** 2 + ri ** 2 - rj ** 2) / (2.0 * D)
            ref = 2.0 * np.pi * ri * (ri + h)
            err = abs(got['area'][i] - ref) / ref
            worst = max(worst, err)
            print(f'  r={ri:.2f} vs r={rj:.2f} at D={D:.2f}: '
                  f'area={got["area"][i]:.6f} analytic={ref:.6f} rel={err:.2e}')
    print(f'  worst relative error {worst:.2e}')
    assert worst < 1e-6, 'two-sphere area is wrong'

    print('self-test 2: random cloud, Gauss-Bonnet vs Monte Carlo')
    rng = np.random.default_rng(0)
    pts = rng.uniform(-0.5, 0.5, (60, 3))
    vals = rng.uniform(0.15, 0.45, 60)
    batch, _ = vae.compute_exposed(pts, vals, degen_tol)
    got = sphere_areas(batch, pts, np.abs(vals), verbose=False)
    dirs = fibonacci_sphere(200000)
    tree = cKDTree(pts)
    n_arcs = np.asarray(batch['n_arcs'])
    keep = np.flatnonzero(~got['open_loop'])
    err = np.array([abs(got['area'][i] - mc_area(pts, vals, int(i), tree, dirs))
                    / (FOUR_PI * vals[i] ** 2) for i in keep])
    print(f'  {len(err)}/{len(pts)} spheres, |GB - MC| / sphere area: '
          f'mean={err.mean():.2e} max={err.max():.2e}  '
          f'({int(got["open_loop"].sum())} skipped for open loops, '
          f'{int((got["n_loops"][n_arcs > 0] > 1).sum())} multi-loop, '
          f'{int((n_arcs == 0).sum())} with no boundary at all)')
    for i in keep[np.argsort(-err)][:5]:
        i = int(i)
        print(f'    sphere {i:>3}: GB={got["area"][i]:.6f} '
              f'MC={mc_area(pts, vals, i, tree, dirs):.6f} '
              f'loops={got["n_loops"][i]} holes={got["holes"][i]} '
              f'arcs={n_arcs[i]}')
    assert err.max() < 3e-3, 'Gauss-Bonnet area disagrees with Monte Carlo'
    print('self-test passed')


# ── region sampling for the picture ──────────────────────────────────

def sample_region(sdf_points, radii, i, tree, arcs_poly, n_samples,
                  min_hits=250):
    """Points of the exposed region itself, for drawing.

    A uniform sweep of the sphere finds a sliver only by luck, so when it comes
    back nearly empty the sweep is repeated inside the spherical cap that
    bounds the region's own boundary arcs, where the same sample budget buys a
    far finer grid. The cap is only used if it really does contain every
    exposed sample the uniform sweep already found.
    """
    center, R = sdf_points[i], float(radii[i])
    occ = occluders(sdf_points, radii, i, tree)
    dirs = fibonacci_sphere(n_samples)
    m = exposed_mask(center + R * dirs, sdf_points, radii, occ)
    pts = center + R * dirs[m]
    if m.sum() >= min_hits or len(arcs_poly) == 0:
        return pts, 'uniform'

    ub = (np.vstack(arcs_poly) - center) / R
    axis = ub.mean(0)
    nrm = np.linalg.norm(axis)
    if nrm < 1e-9:
        return pts, 'uniform'
    axis /= nrm
    cos_a = float(np.min(ub @ axis))
    if len(pts) and float(np.min(((pts - center) / R) @ axis)) < cos_a - 1e-6:
        return pts, 'uniform'                 # region leaves the bounding cap
    cos_a = max(-1.0, cos_a - 0.05)
    # uniform inside the cap: cos(polar angle) stratified over [cos_a, 1]
    e1 = np.cross(axis, [0.0, 0.0, 1.0] if abs(axis[2]) < 0.9 else [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    k = np.arange(n_samples) + 0.5
    cz = 1.0 - (1.0 - cos_a) * k / n_samples
    sz = np.sqrt(np.maximum(1.0 - cz ** 2, 0.0))
    gold = np.pi * (1.0 + 5.0 ** 0.5) * k
    d2 = (cz[:, None] * axis + sz[:, None] * (np.cos(gold)[:, None] * e1
                                              + np.sin(gold)[:, None] * e2))
    m2 = exposed_mask(center + R * d2, sdf_points, radii, occ)
    return center + R * d2[m2], 'bounding cap'


# ── html ─────────────────────────────────────────────────────────────

def _nan_join(polys):
    out = []
    for p in polys:
        out.append(np.asarray(p))
        out.append(np.full((1, 3), np.nan))
    return np.vstack(out) if out else np.zeros((0, 3))


def export_overview(path, mesh, picks, arcs_by_sphere, sdf_points, radii, st,
                    n_buckets=4, embed_js=False):
    """Every thin region at once: boundaries bucketed by thinness."""
    import plotly.graph_objects as go
    fig = go.Figure()
    if mesh is not None:
        fig.add_trace(go.Mesh3d(
            x=mesh.vertices[:, 0], y=mesh.vertices[:, 1], z=mesh.vertices[:, 2],
            i=mesh.faces[:, 0], j=mesh.faces[:, 1], k=mesh.faces[:, 2],
            color='lightgray', opacity=0.35, name='GT mesh', showlegend=True,
            hoverinfo='skip'))

    thin = np.asarray([st[i]['thin'] for i in picks])
    fin = np.isfinite(thin)
    groups = []                              # (label, picks) per line trace
    if (~fin).any():
        groups.append(('zero width (inf)', [p for p, f in zip(picks, fin) if not f]))
    if fin.any():
        tf = thin[fin]
        pf = [p for p, f in zip(picks, fin) if f]
        edges = np.unique(np.percentile(tf, np.linspace(0, 100, n_buckets + 1)))
        if len(edges) < 2:                   # every pick has the same thinness
            edges = np.array([tf.min(), tf.max() + 1e-12])
        bucket = np.clip(np.searchsorted(edges, tf, 'left') - 1, 0, len(edges) - 2)
        for b in reversed(range(len(edges) - 1)):
            sel = [p for p, k in zip(pf, bucket) if k == b]
            if sel:
                groups.append((f'thinness {edges[b]:.1f}-{edges[b + 1]:.1f}', sel))
    for g, (label, sel) in enumerate(groups):
        poly = _nan_join([a['poly'] for i in sel for a in arcs_by_sphere[i]])
        fig.add_trace(go.Scatter3d(
            x=poly[:, 0], y=poly[:, 1], z=poly[:, 2], mode='lines',
            line=dict(width=4, color=PALETTE[g % len(PALETTE)]),
            name=f'{label} ({len(sel)})', hoverinfo='skip'))
    # colorbar cannot show inf; zero-width bands take the top of the scale
    cap = 1.1 * thin[fin].max() if fin.any() else 1.0
    thin = np.where(fin, thin, cap)

    c = sdf_points[picks]
    fig.add_trace(go.Scatter3d(
        x=c[:, 0], y=c[:, 1], z=c[:, 2], mode='markers',
        marker=dict(size=3, color=thin, colorscale='Turbo', showscale=True,
                    colorbar=dict(title='thinness', x=1.02)),
        name='sphere centers',
        hovertext=[f'sphere {i}<br>thinness={st[i]["thin"]:.2f}'
                   f'<br>area/4pir2={st[i]["frac"]:.2e}'
                   f'<br>P/r={st[i]["perim"] / st[i]["r"]:.2f}'
                   f'<br>width/r={st[i]["width"] / st[i]["r"]:.3f}'
                   for i in picks],
        hoverinfo='text'))

    fig.update_layout(
        title=(f'{len(picks)} thin exposed regions | thinness '
               f'{thin.min():.2f}-{thin.max():.2f} '
               '(1 = round cap, large = sliver)'),
        scene=dict(aspectmode='data'), margin=dict(l=0, r=0, t=50, b=0),
        height=850)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fig.write_html(path, include_plotlyjs=True if embed_js else 'cdn')
    return path


def export_detail(path, mesh, picks, arcs_by_sphere, sdf_points, radii, st,
                  region_pts, embed_js=False):
    """One thin sphere per slider step, with the exposed region sampled."""
    import plotly.graph_objects as go
    frames = []
    for i in picks:
        center, r = sdf_points[i], float(radii[i])
        traces = []
        sub = None if mesh is None else vae._crop_mesh(mesh, center, r)
        if sub is not None:
            traces.append(go.Mesh3d(
                x=sub.vertices[:, 0], y=sub.vertices[:, 1], z=sub.vertices[:, 2],
                i=sub.faces[:, 0], j=sub.faces[:, 1], k=sub.faces[:, 2],
                color='lightgray', opacity=0.4, name='gt mesh',
                showlegend=True, hoverinfo='skip'))

        n = 40
        uu, vv = np.linspace(0, 2 * np.pi, n), np.linspace(0, np.pi, n)
        traces.append(go.Surface(
            x=center[0] + r * np.outer(np.cos(uu), np.sin(vv)),
            y=center[1] + r * np.outer(np.sin(uu), np.sin(vv)),
            z=center[2] + r * np.outer(np.ones_like(uu), np.cos(vv)),
            opacity=0.12, showscale=False, name='sphere',
            colorscale=[[0, 'steelblue'], [1, 'steelblue']], hoverinfo='skip'))

        pts, how = region_pts.get(i, (np.zeros((0, 3)), ''))
        if len(pts):
            traces.append(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode='markers',
                marker=dict(size=1.6, color='darkorange'),
                name=f'exposed region ({len(pts)} pts, {how})', hoverinfo='skip'))

        for k, lp in enumerate(st[i]['loops']):
            # arcs_of_sphere can drop an arc with a stale cap id, so the
            # positions from the loop table are filtered, not assumed
            sel = [a for a in lp['arcs'] if a < len(arcs_by_sphere[i])]
            poly = _nan_join([arcs_by_sphere[i][a]['poly'] for a in sel])
            traces.append(go.Scatter3d(
                x=poly[:, 0], y=poly[:, 1], z=poly[:, 2], mode='lines',
                line=dict(width=6, color=PALETTE[k % len(PALETTE)]),
                name=(f'loop {k}: {len(lp["arcs"])} arcs, '
                      f'A={lp["area"]:.3e}, P={lp["perim"]:.3e}'),
                hoverinfo='name'))

        traces.append(go.Scatter3d(
            x=[center[0]], y=[center[1]], z=[center[2]], mode='markers',
            marker=dict(size=3, color='black'), name='sphere center'))
        frames.append(traces)

    fig = go.Figure()
    counts = []
    for traces in frames:
        counts.append(len(traces))
        for t in traces:
            fig.add_trace(t)
    for t in fig.data:
        t.visible = False
    for t in fig.data[:counts[0]]:
        t.visible = True

    steps, off = [], 0
    for i, c in zip(picks, counts):
        vis = [False] * len(fig.data)
        for j in range(off, off + c):
            vis[j] = True
        s = st[i]
        title = (f'sphere {i} | r={s["r"]:.5f} | thinness={s["thin"]:.2f} | '
                 f'width/r={s["width"] / s["r"]:.4f} | '
                 f'area/4pir2={s["frac"]:.3e} | P/r={s["perim"] / s["r"]:.3f} | '
                 f'{s["n_loops"]} loop(s), {s["n_arcs"]} arcs, '
                 f'{s["n_caps"]} caps')
        steps.append(dict(method='update', label=str(i),
                          args=[{'visible': vis}, {'title': title}]))
        off += c

    fig.update_layout(
        sliders=[dict(active=0, currentvalue=dict(prefix='sphere idx: '),
                      steps=steps, pad=dict(t=40))],
        title=steps[0]['args'][1]['title'],
        scene=dict(aspectmode='data'), margin=dict(l=0, r=0, t=60, b=0),
        height=820)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fig.write_html(path, include_plotlyjs=True if embed_js else 'cdn')
    return path


def export_arcs_glb(path, mesh, arc_mask, batch, sdf_points, tube_radius,
                    arc_samples=32, sections=6):
    """The arcs selected by `arc_mask` as tubes, nothing else (plus the GT mesh).

    Built in one vectorized pass and written as a single mesh with per-vertex
    colors (one palette color per sphere), so it stays light for thousands of
    spheres. The tube ring around an arc on cap c is spanned by the cap normal
    and the in-plane radial direction, both exactly perpendicular to the arc.
    `arc_mask` runs over the batch's arcs, in batch order. Returns
    (path, arcs drawn).
    """
    import trimesh
    from trimesh.visual.material import PBRMaterial

    n = len(sdf_points)
    rows = vae.arc_cap_rows(batch, n)
    si = np.asarray(batch['arc_sphere_idx'])
    t0 = np.asarray(batch['arc_start'], float)
    t1 = np.asarray(batch['arc_end'], float)
    rho = np.asarray(batch['cap_radii'])[rows]
    sel = np.asarray(arc_mask, bool) & ((t1 - t0) * rho > 1e-9)
    rows, si, t0, t1, rho = rows[sel], si[sel], t0[sel], t1[sel], rho[sel]
    cc = np.asarray(batch['cap_centers'])[rows]
    cu = np.asarray(batch['cap_u'])[rows]
    cv = np.asarray(batch['cap_v'])[rows]
    cn = np.asarray(batch['cap_normals'])[rows]

    M, S, K = len(rows), arc_samples, sections
    t = t0[:, None] + (t1 - t0)[:, None] * np.linspace(0.0, 1.0, S)[None, :]
    radial = (np.cos(t)[..., None] * cu[:, None, :]
              + np.sin(t)[..., None] * cv[:, None, :])               # (M,S,3)
    center = cc[:, None, :] + rho[:, None, None] * radial
    a = np.linspace(0.0, 2.0 * np.pi, K, endpoint=False)
    ring = (np.cos(a)[None, None, :, None] * radial[:, :, None, :]
            + np.sin(a)[None, None, :, None] * cn[:, None, None, :])  # (M,S,K,3)
    verts = (center[:, :, None, :] + tube_radius * ring).reshape(-1, 3)

    i, j = np.meshgrid(np.arange(S - 1), np.arange(K), indexing='ij')
    q0 = i * K + j
    q1 = i * K + (j + 1) % K
    q2 = (i + 1) * K + (j + 1) % K
    q3 = (i + 1) * K + j
    quad = np.concatenate([np.stack([q0, q1, q2], -1).reshape(-1, 3),
                           np.stack([q0, q2, q3], -1).reshape(-1, 3)])
    faces = (quad[None, :, :] + (np.arange(M) * S * K)[:, None, None]).reshape(-1, 3)

    palette = np.array([(228, 26, 28), (55, 126, 184), (77, 175, 74),
                        (152, 78, 163), (255, 127, 0), (166, 86, 40),
                        (247, 129, 191), (0, 191, 191)], np.uint8)
    rank = np.searchsorted(np.unique(si), si)          # neighbors differ in color
    col = palette[rank % len(palette)]
    vcol = np.repeat(np.hstack([col, np.full((M, 1), 255, np.uint8)]), S * K, axis=0)

    scene = trimesh.Scene()
    if mesh is not None:
        gt = mesh.copy()
        gt.visual.material = PBRMaterial(baseColorFactor=[190, 190, 190, 110],
                                         alphaMode='BLEND')
        scene.add_geometry(gt, 'gt_mesh')
    tubes = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=vcol,
                            process=False)
    scene.add_geometry(tubes, 'thin_exposed_arcs')
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    scene.export(path)
    return path, M


# ── driver ───────────────────────────────────────────────────────────

def export_thin_exposed(mesh, sdf_points, sdf_values, out_dir, tag,
                        degen_tol=1e-5, neighbors='rt', min_thin=4.0,
                        max_area=None, min_perim_rad=0.05, sort='thin',
                        max_spheres=24, max_overview=400, max_glb=None,
                        arc_samples=64,
                        region_samples=0, glb=True, csv=False, html=False,
                        embed_js=False, bbox=None):
    """Measure every sphere's exposed region and write the thin ones out.

    `mesh` is the GT mesh in the same (normalized) frame as the samples, or
    None when there is none (e.g. SDF loaded from an npz); the pages are then
    drawn without it. `bbox` = (min_corner, max_corner): only regions whose
    arcs all lie inside it are drawn; None = the samples' bbox, False = no
    bbox filter. Writes <out_dir>/thin_exposed_<tag>.glb, plus the
    ranking .csv (csv=True) and the _overview/_detail .html pages (html=True),
    and returns the list of paths written.
    """
    import time
    t_all = time.perf_counter()
    sdf_points = np.ascontiguousarray(sdf_points, float)
    sdf_values = np.ascontiguousarray(sdf_values, float)
    radii = np.abs(sdf_values)
    written = []

    t = time.perf_counter()
    batch, degenerate_pts = vae.compute_exposed(sdf_points, sdf_values,
                                                degen_tol, source=neighbors)
    caps_by_sphere = vae.per_sphere_caps(batch)
    n_arcs = np.asarray(batch['n_arcs'])
    n_caps = np.asarray(batch['n_caps'])
    n_points = np.asarray(batch['n_points'])
    print(f'[thin_exposed] exposed arcs: {time.perf_counter() - t:.2f} s')

    t = time.perf_counter()
    m = sphere_areas(batch, sdf_points, radii)
    area, perim, thin, width = m['area'], m['perim'], m['thin'], m['width']
    print(f'[thin_exposed] areas: {time.perf_counter() - t:.2f} s')
    print(f'{int((n_arcs > 0).sum())} spheres have an exposed boundary, '
          f'{int((n_arcs == 0).sum())} do not '
          f'({int((n_points > 0).sum())} of those hold a degenerate point)')
    if m['open_loop'].any():
        print(f'{int(m["open_loop"].sum())} spheres have an unclosed arc chain; '
              'their area is unreliable and they are excluded')

    reg = m['regions']
    rlive = reg['valid'] & ~np.isnan(reg['thin'])
    if not rlive.any():
        print('no measurable exposed region')
        return written
    rR = radii[reg['sphere']]
    print(f'{int(rlive.sum())} connected exposed regions on '
          f'{len(np.unique(reg["sphere"][rlive]))} spheres '
          f'({int((reg["merged"] & rlive).sum())} merged per sphere: several '
          'outer loops plus holes)')
    zw = rlive & reg['zero_width']
    if zw.any():
        print(f'{int(zw.sum())} regions are zero-width bands (boundary, no area): '
              'thinness = inf')
    fin = rlive & ~reg['zero_width']
    if fin.any():
        pq = [5, 25, 50, 75, 95, 99]
        print(f'over the {int(fin.sum())} regions with area, percentiles {pq}:')
        for lbl, v in (('thinness', reg['thin'][fin]),
                       ('area', reg['area'][fin]),
                       ('perimeter / r', reg['perim'][fin] / rR[fin]),
                       ('width / r', reg['width'][fin] / rR[fin])):
            print(f'  {lbl:<15} ' + ', '.join(f'{x:.4g}'
                                              for x in np.percentile(v, pq)))

    # a region is drawn only if it is thin enough AND small enough
    # area is absolute, in the normalized mesh frame (unit-cube scale)
    rkeep = (rlive & (reg['thin'] >= min_thin)
             & (reg['area'] <= (np.inf if max_area is None else max_area))
             & (reg['perim'] / rR >= min_perim_rad))
    if bbox is not False:
        # and every arc of the region lies inside the bbox
        bmin, bmax = ((sdf_points.min(axis=0), sdf_points.max(axis=0)) if bbox is None
                      else (np.asarray(bbox[0], float), np.asarray(bbox[1], float)))
        eps = 1e-9 * (bmax - bmin).max()
        arcs_ = m['arcs']
        arc_out = ~(np.all(arcs_['lo'] >= bmin - eps, axis=1)
                    & np.all(arcs_['hi'] <= bmax + eps, axis=1))
        has = arcs_['region'] >= 0
        reg_out = np.bincount(arcs_['region'][has], weights=arc_out[has],
                              minlength=len(rkeep))[:len(rkeep)] > 0
        print(f'bbox {np.round(bmin, 4)} .. {np.round(bmax, 4)}: '
              f'{int((rkeep & reg_out).sum())} otherwise-drawn regions poke out, dropped')
        rkeep &= ~reg_out
    rsel = np.flatnonzero(rkeep)
    key = {'thin': -reg['thin'], 'width': reg['width'] / rR,
           'area': reg['frac'], 'perim': -reg['perim'] / rR}[sort]
    rsel = rsel[np.argsort(key[rsel], kind='stable')]
    # spheres holding a drawn region, in the order of their best region
    cand = reg['sphere'][rsel]
    cand = cand[np.sort(np.unique(cand, return_index=True)[1])]
    print(f'{rsel.size} regions on {cand.size} spheres pass thinness>={min_thin} '
          f'and area<={max_area} (P/r>={min_perim_rad})')

    os.makedirs(out_dir, exist_ok=True)
    if csv:
        csv_path = os.path.join(out_dir, f'thin_exposed_{tag}.csv')
        order = np.flatnonzero(rlive)
        order = order[np.argsort(key[order], kind='stable')]
        with open(csv_path, 'w') as f:
            f.write('sphere,region,r,area,perimeter,thinness,width,area_frac,'
                    'n_loops,merged,drawn\n')
            for k in order:
                i = reg['sphere'][k]
                f.write(f'{i},{k},{radii[i]:.9g},{reg["area"][k]:.9g},'
                        f'{reg["perim"][k]:.9g},{reg["thin"][k]:.9g},'
                        f'{reg["width"][k]:.9g},{reg["frac"][k]:.9g},'
                        f'{reg["n_loops"][k]},{int(reg["merged"][k])},'
                        f'{int(rkeep[k])}\n')
        print(f'wrote {csv_path} ({len(order)} regions, ranked by {sort})')
        written.append(csv_path)

    if rsel.size == 0:
        print('nothing to draw — lower min_thin or raise max_area')
        return written

    # per-loop breakdown, only drawn by the html pages
    loops, arcs = m['loops'], m['arcs']
    arc_rows_of = {}
    if html:
        for r, sph in enumerate(arcs['sphere']):
            arc_rows_of.setdefault(int(sph), []).append(r)

    def loop_table(i):
        by_label = {}
        for pos, r in enumerate(arc_rows_of.get(i, [])):  # pos = index in arcs_of_sphere
            if arcs['label'][r] >= 0:              # zero-length arcs own no loop
                by_label.setdefault(int(arcs['label'][r]), []).append(pos)
        return [dict(arcs=v, area=float(loops['area'][k]),
                     perim=float(loops['perim'][k]))
                for k, v in sorted(by_label.items())]

    overview = [int(i) for i in cand[:max_overview]] if html else []
    picks = [int(i) for i in cand[:max_spheres]]
    if html:
        print(f'overview: {len(overview)} spheres; detail slider: {len(picks)}')

    st = {i: dict(r=float(radii[i]), area=float(area[i]),
                  perim=float(perim[i]), thin=float(thin[i]),
                  width=float(width[i]), frac=float(m['frac'][i]),
                  n_loops=int(m['n_loops'][i]), holes=int(m['holes'][i]),
                  n_arcs=int(n_arcs[i]), n_caps=int(n_caps[i]),
                  loops=loop_table(i) if html else [])
          for i in set(overview) | set(picks)}

    print('top regions:')
    print(f'  {"sphere":>8} {"r":>9} {"thinness":>9} {"width/r":>9} '
          f'{"area":>11} {"P/r":>8} {"loops":>5}')
    for k in rsel[:max_spheres]:
        i = reg['sphere'][k]
        print(f'  {i:>8} {radii[i]:>9.5f} {reg["thin"][k]:>9.2f} '
              f'{reg["width"][k] / radii[i]:>9.4f} {reg["area"][k]:>11.3e} '
              f'{reg["perim"][k] / radii[i]:>8.3f} {reg["n_loops"][k]:>5}')

    if html:
        t = time.perf_counter()
        arcs_by_sphere = {i: vae.arcs_of_sphere(batch, caps_by_sphere, i, arc_samples)
                          for i in set(overview) | set(picks)}
        ov = export_overview(os.path.join(out_dir, f'thin_exposed_{tag}_overview.html'),
                             mesh, overview, arcs_by_sphere, sdf_points, radii, st,
                             embed_js=embed_js)
        print(f'wrote {ov} ({time.perf_counter() - t:.2f} s)')
        written.append(ov)

        t = time.perf_counter()
        region_pts = {}
        if region_samples > 0:               # opt-in: costs samples x occluders per pick
            tree = cKDTree(sdf_points)
            region_pts = {i: sample_region(sdf_points, radii, i, tree,
                                           [a['poly'] for a in arcs_by_sphere[i]],
                                           region_samples)
                          for i in picks}
        det = export_detail(os.path.join(out_dir, f'thin_exposed_{tag}_detail.html'),
                            mesh, picks, arcs_by_sphere, sdf_points, radii, st,
                            region_pts, embed_js=embed_js)
        print(f'wrote {det} ({time.perf_counter() - t:.2f} s)')
        written.append(det)

    if glb:
        # every region that passes both filters (max_glb=None), arcs only
        t = time.perf_counter()
        gsel = rsel if max_glb is None else rsel[:max_glb]
        tube_r = float(np.median(radii[reg['sphere'][gsel]])) * 0.015
        g, n_tubes = export_arcs_glb(
            os.path.join(out_dir, f'thin_exposed_{tag}.glb'), mesh,
            np.isin(m['arcs']['region'], gsel), batch, sdf_points, tube_r)
        print(f'wrote {g}: {len(gsel)} regions on '
              f'{len(np.unique(reg["sphere"][gsel]))} spheres, {n_tubes} arcs '
              f'({time.perf_counter() - t:.2f} s)')
        written.append(g)

    print(f'[thin_exposed] total: {time.perf_counter() - t_all:.2f} s')
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--name', default='eiffel')
    ap.add_argument('--data-dir', default='examples')
    ap.add_argument('--grid-len', type=int, default=30)
    ap.add_argument('--scatter', action='store_true',
                    help='random samples in the bbox instead of a grid')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--degen-tol', type=float, default=1e-5)
    ap.add_argument('--neighbors', choices=['rt', 'hash', 'hash-sym'], default='rt')
    ap.add_argument('--min-thin', type=float, default=4.0,
                    help='draw regions with P^2/(A(4pi - A/r^2)) above this '
                         '(1 = round cap, 10 = ribbon 10x longer than wide)')
    ap.add_argument('--max-area', type=float, default=None,
                    help='and whose area is at most this (absolute, normalized '
                         'mesh units); both must hold for a region to be drawn')
    ap.add_argument('--min-perim-rad', type=float, default=0.05,
                    help='keep spheres with perimeter/radius above this, which '
                         'drops numerical dust near the degeneracy threshold')
    ap.add_argument('--sort', choices=['thin', 'width', 'area', 'perim'],
                    default='thin', help='ranking used for the picks')
    ap.add_argument('--max-spheres', type=int, default=24,
                    help='spheres in the detail slider')
    ap.add_argument('--max-overview', type=int, default=400,
                    help='spheres drawn in the overview page')
    ap.add_argument('--max-glb', type=int, default=None,
                    help='cap the regions in the glb (default: all that pass)')
    ap.add_argument('--arc-samples', type=int, default=64)
    ap.add_argument('--region-samples', type=int, default=0,
                    help='also sample the exposed region itself as points in '
                         'the detail page, with this many samples per sphere '
                         '(0 = off; the arcs already outline it)')
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--no-glb', action='store_true')
    ap.add_argument('--csv', action='store_true', help='also write the ranking csv')
    ap.add_argument('--html', action='store_true',
                    help='also write the overview/detail html pages')
    ap.add_argument('--embed-plotly', action='store_true')
    ap.add_argument('--self-test', action='store_true',
                    help='validate the area formula and exit')
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    import SDF_to_surface_3D as sdf3d
    sdf3d.data_dir = args.data_dir
    sdf3d.seed = args.seed
    mesh, sdf_points, sdf_values, _ = sdf3d.generate_test_mesh_data(
        f'{args.data_dir}/{args.name}.obj', args.name,
        grid_len=args.grid_len, save=False, scatter=args.scatter)
    print(f'{len(sdf_points)} SDF samples')

    export_thin_exposed(
        mesh, sdf_points, sdf_values, args.out_dir or f'out/{args.name}',
        f'{args.name}_{args.grid_len}', degen_tol=args.degen_tol,
        neighbors=args.neighbors, min_thin=args.min_thin,
        max_area=args.max_area, min_perim_rad=args.min_perim_rad,
        sort=args.sort, max_spheres=args.max_spheres,
        max_overview=args.max_overview, max_glb=args.max_glb,
        arc_samples=args.arc_samples,
        region_samples=args.region_samples, glb=not args.no_glb,
        csv=args.csv, html=args.html,
        embed_js=args.embed_plotly)


if __name__ == '__main__':
    main()
