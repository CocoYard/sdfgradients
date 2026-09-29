"""Visualize the exposed-region boundary arcs of spheres that sit on a GT sharp edge.

`get_visible_arcs` (cpp/src/get_visible_arcs.cpp) carves every SDF sphere by its
neighbors and returns the boundary of what is left exposed: a set of arcs, each
living on a cap circle of that sphere. Near a sharp edge of the GT shape the
exposed region is where the sphere kisses the surface, so its boundary arcs are
what decides whether the edge is reconstructed or rounded off. This script picks
the spheres whose nearest GT feature is a sharp edge and draws their arcs.

Edge detection reuses the NMC criterion from edgeChamfer.extract_edge_points
(a surface sample is an "edge point" if some neighbor within EF1_RADIUS has a
near-perpendicular normal).

Sphere selection: a sphere of radius r centered at c is empty of surface, so the
nearest edge point is at distance >= r. It is "on an edge" when
    dist(c, nearest edge point) - r <= band,
i.e. the sphere touches the surface at (or very near) a crease.

Outputs
  out/<name>/arcs_near_edges_<name>_<grid>.html  interactive plotly, one sphere
                                                 per slider step
  out/<name>/arcs_near_edges_<name>_<grid>.glb   same arcs as tube meshes, for
                                                 Blender / any glTF viewer

The carve itself comes from sdf_cpp.exposed_arcs, which is the pipeline's own
step 1 (power-diagram neighbors -> compute_exposed_batch). --neighbors hash /
hash-sym reroute it through bench/sphere_exposed_pybind with the spatial-hash
neighbor list instead; both under-carve badly and are there for comparison
only. --verify audits the drawn arcs against every sphere.

Usage
  .venv/bin/python debug/vis_arcs_near_edges.py --name eiffel --grid-len 30
  .venv/bin/python debug/vis_arcs_near_edges.py --name fandisk --grid-len 40 \
      --band 0.005 --max-spheres 24
  .venv/bin/python debug/vis_arcs_near_edges.py --name eiffel --sphere-ids 12345 67890
  .venv/bin/python debug/vis_arcs_near_edges.py --name eiffel --neighbors hash --verify
"""
import argparse
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import SDF_to_surface_3D as sdf3d          # noqa: E402
from edgeChamfer import (                  # noqa: E402
    EF1_DOTPRODUCT_THRESHOLD,
    EF1_RADIUS,
    SAMPLE_NUM,
    extract_edge_points,
)


# ── exposed-region computation ───────────────────────────────────────

def symmetrize_neighbors(offsets, nbrs, n):
    """Make a CSR neighbor relation symmetric.

    sphere_intersect.find_intersections (the multi-level spatial hash in
    sphere_intersect.cpp) records each overlapping pair only ONCE, under the
    sphere with the coarser home level — i.e. the bigger one. A sphere is
    inserted into its home level and every coarser level, but the query for
    sphere i scans only i's own home level, where any larger sphere was never
    inserted. So nbrs[i] holds only neighbors of i's size or smaller.

    Only used by the 'hash-sym' source below, which exists to show what that
    list does to the carve — it is not the list the pipeline uses.
    """
    from scipy.sparse import csr_matrix
    a = csr_matrix((np.ones(len(nbrs), np.int8), nbrs, offsets), shape=(n, n))
    b = (a + a.T).tocsr()
    b.setdiag(0)
    b.eliminate_zeros()
    return b.indptr.astype(np.int64), b.indices.astype(np.int64)


def compute_exposed(sdf_points, sdf_values, degen_tol, source='rt', verbose=False):
    """Carve every sphere by its neighbors; return (batch dict, degenerate pts).

    source='rt' (default) calls sdf_cpp.exposed_arcs, i.e. exactly what the
    pipeline runs: power-diagram (regular triangulation) neighbors into
    compute_exposed_batch. Those neighbors are the right ones and not merely a
    cheap approximation: for p on sphere i, p lies inside sphere j iff
    pi_j(p) < 0 = pi_i(p), so the exposed region is exactly the part of the
    sphere inside i's power cell, whose facets are the RT edges at i.

    source='hash' / 'hash-sym' go through bench/sphere_exposed_pybind with the
    spatial-hash neighbor list instead (raw, or symmetrized). Both under-carve
    and are kept only for comparison: 'hash' misses every larger overlapping
    neighbor, and 'hash-sym' hands compute_all_caps thousands of neighbors, so
    its MAX_CAPS=512 scan limit drops the deep cuts. Run with --verify to see
    it.
    """
    if source == 'rt':
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'cpp', 'build'))
        import sdf_cpp
        batch = sdf_cpp.exposed_arcs(np.ascontiguousarray(sdf_points, float),
                                     np.ascontiguousarray(sdf_values, float),
                                     degen_tol=degen_tol, verbose=verbose)
    else:
        from bench import sphere_exposed_pybind as sep
        from bench import sphere_intersect
        radii = np.abs(sdf_values)
        offsets, neighbors = sphere_intersect.find_intersections(sdf_points, radii)
        if source == 'hash-sym':
            offsets, neighbors = symmetrize_neighbors(offsets, neighbors,
                                                      len(sdf_points))
        batch = sep.compute_exposed_batch(sdf_points, radii, neighbors, offsets,
                                          tol=1e-4, degen_tol=degen_tol)

    degenerate_pts = {}
    for idx, pos in zip(batch['point_sphere_idx'], batch['point_positions']):
        degenerate_pts.setdefault(int(idx), []).append(pos)
    return batch, degenerate_pts


def verify_arcs(batch, caps_by_sphere, sdf_points, radii, spheres, n_samples=17):
    """Penetration audit: is any drawn arc point strictly inside another sphere?

    For p on an arc of sphere i, max_{j != i} (r_j - |p - c_j|) must be <= 0:
    the cutting sphere touches p exactly, everything else stays outside. A
    positive value means p was carved away yet still shows up as boundary,
    which is what a wrong neighbor list looks like on screen.

    Returns (worst penetration / radius) per sphere.
    """
    from scipy.spatial.distance import cdist
    out = {}
    for i in spheres:
        arcs = arcs_of_sphere(batch, caps_by_sphere, i, n_samples)
        if not arcs:
            continue
        pts = np.vstack([a['poly'] for a in arcs])
        worst = -np.inf
        for s in range(0, len(pts), 400):
            pen = radii[None, :] - cdist(pts[s:s + 400], sdf_points)
            pen[:, i] = -np.inf
            worst = max(worst, float(pen.max()))
        out[i] = worst / max(radii[i], 1e-12)
    return out


# ── arc geometry ─────────────────────────────────────────────────────

def per_sphere_caps(batch):
    """Split the flat cap arrays into a per-sphere lookup.

    `arc_cap_idx` is a *per-sphere compacted* cap id (see the packing loop in
    bench/sphere_exposed_pybind.cpp), matching `cap_id` in the cap arrays, so
    caps must be grouped by sphere before an arc can find its circle.

    Returns {sphere index -> dict of cap arrays indexed by that compacted id}.
    """
    cap_sphere = np.asarray(batch['cap_sphere_idx'])
    cap_id = np.asarray(batch['cap_id'])
    out = {}
    for s in np.unique(cap_sphere):
        m = cap_sphere == s
        order = np.argsort(cap_id[m])          # compacted id -> row in the flat arrays
        rows = np.flatnonzero(m)[order]
        out[int(s)] = dict(
            centers=np.asarray(batch['cap_centers'])[rows],
            radii=np.asarray(batch['cap_radii'])[rows],
            u=np.asarray(batch['cap_u'])[rows],
            v=np.asarray(batch['cap_v'])[rows],
            normals=np.asarray(batch['cap_normals'])[rows],
            d=np.asarray(batch['cap_d'])[rows],
        )
    return out


def arc_cap_rows(batch, n_spheres):
    """Row in the flat cap arrays for every arc, vectorized.

    Caps are appended sphere by sphere with ascending compacted id, so the row
    of (sphere i, cap c) is start[i] + c. Asserted rather than assumed.
    """
    cap_sphere = np.asarray(batch['cap_sphere_idx'])
    cap_id = np.asarray(batch['cap_id'])
    counts = np.bincount(cap_sphere, minlength=n_spheres)
    starts = np.concatenate([[0], np.cumsum(counts)])[:-1]
    assert np.array_equal(cap_id, np.arange(len(cap_id)) - starts[cap_sphere]), \
        'cap arrays are not grouped by sphere with ascending cap_id'
    return starts[np.asarray(batch['arc_sphere_idx'])] + np.asarray(batch['arc_cap_idx'])


def arc_midpoints_and_lengths(batch, n_spheres):
    """(midpoint (M,3), length (M,)) for every arc in the batch."""
    rows = arc_cap_rows(batch, n_spheres)
    cc = np.asarray(batch['cap_centers'])[rows]
    cr = np.asarray(batch['cap_radii'])[rows]
    cu = np.asarray(batch['cap_u'])[rows]
    cv = np.asarray(batch['cap_v'])[rows]
    a_s = np.asarray(batch['arc_start'])
    a_e = np.asarray(batch['arc_end'])
    t = 0.5 * (a_s + a_e)
    mid = cc + cr[:, None] * (np.cos(t)[:, None] * cu + np.sin(t)[:, None] * cv)
    return mid, (a_e - a_s) * cr


def arcs_of_sphere(batch, caps_by_sphere, sphere_idx, n_samples=64):
    """Polylines for every exposed-boundary arc of one sphere.

    Arc a on cap c is traced as  center_c + R_c * (cos t * u_c + sin t * v_c)
    for t in [arc_start, arc_end]; the batch stores end >= start (sample_arcs in
    the pybind layer relies on the same convention), so no wrap handling here.

    Returns a list of dicts: polyline (S,3), length, cap id, angular span.
    """
    arc_sphere = np.asarray(batch['arc_sphere_idx'])
    rows = np.flatnonzero(arc_sphere == sphere_idx)
    if rows.size == 0:
        return []
    caps = caps_by_sphere.get(int(sphere_idx))
    if caps is None:
        return []

    arc_cap = np.asarray(batch['arc_cap_idx'])
    arc_s = np.asarray(batch['arc_start'])
    arc_e = np.asarray(batch['arc_end'])

    out = []
    for r in rows:
        ci = int(arc_cap[r])
        if ci >= len(caps['radii']):
            continue                                    # defensive: stale cap id
        t = np.linspace(arc_s[r], arc_e[r], n_samples)
        poly = (caps['centers'][ci][None, :]
                + caps['radii'][ci] * (np.cos(t)[:, None] * caps['u'][ci][None, :]
                                       + np.sin(t)[:, None] * caps['v'][ci][None, :]))
        out.append(dict(poly=poly,
                        cap=ci,
                        span=float(arc_e[r] - arc_s[r]),
                        length=float((arc_e[r] - arc_s[r]) * caps['radii'][ci])))
    return out


# ── tube meshes for the glb ──────────────────────────────────────────

def polyline_tube(poly, radius, sections=8):
    """A closed-cross-section tube swept along `poly`; returns (verts, faces).

    Frames are built from a single reference axis rather than parallel
    transport — an arc never turns enough for the ring to flip.
    """
    poly = np.asarray(poly, float)
    if len(poly) < 2:
        return None
    tang = np.gradient(poly, axis=0)
    nrm = np.linalg.norm(tang, axis=1, keepdims=True)
    tang = tang / np.where(nrm < 1e-15, 1.0, nrm)

    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(tang[0] @ ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    u = np.cross(tang, ref)
    u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-15
    v = np.cross(tang, u)

    ang = np.linspace(0, 2 * np.pi, sections, endpoint=False)
    ring = (np.cos(ang)[None, :, None] * u[:, None, :]
            + np.sin(ang)[None, :, None] * v[:, None, :])
    verts = (poly[:, None, :] + radius * ring).reshape(-1, 3)

    n = len(poly)
    faces = []
    for i in range(n - 1):
        for j in range(sections):
            a = i * sections + j
            b = i * sections + (j + 1) % sections
            c = (i + 1) * sections + (j + 1) % sections
            d = (i + 1) * sections + j
            faces.append([a, b, c])
            faces.append([a, c, d])
    return verts, np.asarray(faces, np.int64)


def export_glb(path, mesh, picks, arcs_by_sphere, sdf_points, radii,
               degenerate_pts, edge_pts, colors, tube_radius):
    import trimesh
    from trimesh.visual.material import PBRMaterial

    scene = trimesh.Scene()

    gt = mesh.copy()
    gt.visual.material = PBRMaterial(baseColorFactor=[190, 190, 190, 110],
                                     alphaMode='BLEND')
    scene.add_geometry(gt, 'gt_mesh')

    if len(edge_pts):
        scene.add_geometry(
            trimesh.PointCloud(edge_pts,
                               colors=np.tile([30, 90, 255, 255], (len(edge_pts), 1))),
            'gt_edge_points')

    for k, i in enumerate(picks):
        col = colors[k % len(colors)]
        # the carved sphere itself, translucent, so the exposed region reads
        ball = trimesh.creation.icosphere(subdivisions=3, radius=float(radii[i]))
        ball.apply_translation(sdf_points[i])
        ball.visual.material = PBRMaterial(baseColorFactor=[*col, 45],
                                           alphaMode='BLEND')
        scene.add_geometry(ball, f'sphere_{i}')

        for a, arc in enumerate(arcs_by_sphere[i]):
            tube = polyline_tube(arc['poly'], tube_radius)
            if tube is None:
                continue
            t = trimesh.Trimesh(vertices=tube[0], faces=tube[1], process=False)
            t.visual.material = PBRMaterial(baseColorFactor=[*col, 255],
                                            emissiveFactor=[c / 255 for c in col],
                                            alphaMode='OPAQUE')
            scene.add_geometry(t, f'arc_{i}_{a}')

        for p, pt in enumerate(degenerate_pts.get(i, [])):
            dot = trimesh.creation.icosphere(subdivisions=2, radius=tube_radius * 3)
            dot.apply_translation(pt)
            dot.visual.material = PBRMaterial(baseColorFactor=[255, 215, 0, 255],
                                              emissiveFactor=[1, 0.84, 0],
                                              alphaMode='OPAQUE')
            scene.add_geometry(dot, f'degen_{i}_{p}')

        center = trimesh.creation.icosphere(subdivisions=2, radius=tube_radius * 2)
        center.apply_translation(sdf_points[i])
        center.visual.material = PBRMaterial(baseColorFactor=[20, 20, 20, 255],
                                             alphaMode='OPAQUE')
        scene.add_geometry(center, f'center_{i}')

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    scene.export(path)
    return path


# ── interactive plotly page ──────────────────────────────────────────

def _crop_mesh(mesh, center, radius):
    """Faces whose centroid is within 2.5r of the sphere center."""
    keep = np.linalg.norm(mesh.triangles_center - center, axis=1) < 2.5 * radius
    if not keep.any():
        return None
    sub = mesh.submesh([np.flatnonzero(keep)], append=True)
    return sub


def export_html(path, mesh, picks, arcs_by_sphere, sdf_points, radii,
                degenerate_pts, edge_tree, edge_pts, stats, embed_js=False):
    import plotly.graph_objects as go

    frames_traces = []      # list of (list of traces) per pick
    for i in picks:
        center, r = sdf_points[i], float(radii[i])
        traces = []

        sub = _crop_mesh(mesh, center, r)
        if sub is not None:
            traces.append(go.Mesh3d(
                x=sub.vertices[:, 0], y=sub.vertices[:, 1], z=sub.vertices[:, 2],
                i=sub.faces[:, 0], j=sub.faces[:, 1], k=sub.faces[:, 2],
                color='lightgray', opacity=0.45, name='gt mesh',
                showlegend=True, hoverinfo='skip'))

        # translucent carved sphere
        n = 40
        uu = np.linspace(0, 2 * np.pi, n)
        vv = np.linspace(0, np.pi, n)
        traces.append(go.Surface(
            x=center[0] + r * np.outer(np.cos(uu), np.sin(vv)),
            y=center[1] + r * np.outer(np.sin(uu), np.sin(vv)),
            z=center[2] + r * np.outer(np.ones_like(uu), np.cos(vv)),
            opacity=0.18, showscale=False, name='sphere',
            colorscale=[[0, 'steelblue'], [1, 'steelblue']], hoverinfo='skip'))

        # boundary arcs, colored by distance from the nearest GT edge point
        for a, arc in enumerate(arcs_by_sphere[i]):
            poly = arc['poly']
            de, _ = edge_tree.query(poly)
            traces.append(go.Scatter3d(
                x=poly[:, 0], y=poly[:, 1], z=poly[:, 2], mode='lines',
                line=dict(width=7, color=de, colorscale='Turbo',
                          cmin=0.0, cmax=max(3 * EF1_RADIUS, 1e-6),
                          colorbar=dict(title='dist to<br>GT edge', x=1.02)
                          if a == 0 else None),
                name=f'arc {a} (cap {arc["cap"]}, len {arc["length"]:.4f})',
                hovertext=[f'd_edge={d:.5f}' for d in de], hoverinfo='text+name'))

        for p, pt in enumerate(degenerate_pts.get(i, [])):
            traces.append(go.Scatter3d(
                x=[pt[0]], y=[pt[1]], z=[pt[2]], mode='markers',
                marker=dict(size=6, color='gold', symbol='diamond'),
                name=f'degenerate pt {p}'))

        near = edge_pts[edge_tree.query_ball_point(center, 2.5 * r)]
        if len(near):
            traces.append(go.Scatter3d(
                x=near[:, 0], y=near[:, 1], z=near[:, 2], mode='markers',
                marker=dict(size=1.8, color='crimson'),
                name=f'GT edge points ({len(near)})'))

        traces.append(go.Scatter3d(
            x=[center[0]], y=[center[1]], z=[center[2]], mode='markers',
            marker=dict(size=3, color='black'), name='sphere center'))

        frames_traces.append(traces)

    # one figure holding every pick's traces, with a slider flipping visibility
    fig = go.Figure()
    counts = []
    for traces in frames_traces:
        counts.append(len(traces))
        for t in traces:
            fig.add_trace(t)
    for t in fig.data:
        t.visible = False
    for t in fig.data[:counts[0]]:
        t.visible = True

    steps = []
    off = 0
    for k, (i, c) in enumerate(zip(picks, counts)):
        vis = [False] * len(fig.data)
        for j in range(off, off + c):
            vis[j] = True
        st = stats[k]
        steps.append(dict(
            method='update', label=str(i),
            args=[{'visible': vis},
                  {'title': (f'sphere {i} | r={st["r"]:.5f} | '
                             f'edge gap={st["gap"]:.5f} | '
                             f'{st["n_arcs"]} arcs, {st["n_caps"]} caps, '
                             f'{st["n_points"]} degen pts | '
                             f'total arc={st["total_arc"]:.5f} '
                             f'({st["total_arc"] / max(st["r"], 1e-12):.3f} rad)')}]))
        off += c

    fig.update_layout(
        sliders=[dict(active=0, currentvalue=dict(prefix='sphere idx: '),
                      steps=steps, pad=dict(t=40))],
        title=steps[0]['args'][1]['title'],
        scene=dict(aspectmode='data'),
        margin=dict(l=0, r=0, t=60, b=0),
        height=820)

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    # 'cdn' keeps the file ~3 MB but needs the viewer's browser to reach the
    # CDN; embed_js inlines plotly.js (~7 MB extra) for offline viewing.
    fig.write_html(path, include_plotlyjs=True if embed_js else 'cdn')
    return path


# ── main ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--name', default='eiffel')
    ap.add_argument('--data-dir', default='examples')
    ap.add_argument('--grid-len', type=int, default=30)
    ap.add_argument('--scatter', action='store_true',
                    help='random samples in the bbox instead of a grid')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--degen-tol', type=float, default=1e-5,
                    help='exposed region collapses to a point below this arc length')
    ap.add_argument('--band', type=float, default=None,
                    help='sphere is "on an edge" when nearest-edge-distance - r <= band '
                         '(default: 2x the median spacing of the edge point cloud, i.e. '
                         'the resolution at which "the contact point is an edge point" '
                         'can be resolved at all)')
    ap.add_argument('--edge-radius', type=float, default=EF1_RADIUS,
                    help='neighborhood radius of the NMC edge detector')
    ap.add_argument('--edge-dot', type=float, default=EF1_DOTPRODUCT_THRESHOLD)
    ap.add_argument('--edge-samples', type=int, default=SAMPLE_NUM)
    ap.add_argument('--max-spheres', type=int, default=24,
                    help='how many qualifying spheres to draw')
    ap.add_argument('--sort', choices=['gap', 'shortest-arc', 'most-arcs', 'radius',
                                       'random'], default='gap',
                    help='which qualifying spheres to prefer; picks are then spread '
                         'evenly over that ordering')
    ap.add_argument('--sphere-ids', type=int, nargs='+', default=None,
                    help='draw exactly these sphere indices, skipping the selection')
    ap.add_argument('--arc-samples', type=int, default=64)
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--no-glb', action='store_true')
    ap.add_argument('--embed-plotly', action='store_true',
                    help='inline plotly.js into the html so it renders without '
                         'network access (adds ~7 MB)')
    ap.add_argument('--neighbors', choices=['rt', 'hash', 'hash-sym'], default='rt',
                    help="occluder set fed to the carve: 'rt' = the pipeline's "
                         "power-diagram neighbors via sdf_cpp.exposed_arcs "
                         "(correct); 'hash'/'hash-sym' = the spatial-hash list "
                         'via the bench module (under-carves, for comparison)')
    ap.add_argument('--verify', action='store_true',
                    help='audit the drawn arcs: report how far each one is '
                         'carved into another sphere (should be ~0)')
    args = ap.parse_args()

    sdf3d.data_dir = args.data_dir
    sdf3d.seed = args.seed

    # 1. SDF samples (mesh comes back normalized into a unit cube, which is what
    #    the absolute NMC edge thresholds assume)
    mesh, sdf_points, sdf_values, _ = sdf3d.generate_test_mesh_data(
        f'{args.data_dir}/{args.name}.obj', args.name,
        grid_len=args.grid_len, save=False, scatter=args.scatter)
    radii = np.abs(sdf_values)
    print(f'{len(sdf_points)} SDF samples, radius range '
          f'[{radii.min():.5f}, {radii.max():.5f}]')

    # 2. GT sharp-edge points (NMC criterion, same as edgeChamfer)
    edge_pts, _ = extract_edge_points(
        mesh, sample_num=args.edge_samples, ef1_radius=args.edge_radius,
        ef1_dotproduct_threshold=args.edge_dot)
    print(f'{len(edge_pts)} GT edge points '
          f'(radius={args.edge_radius}, |n.n\'|<{args.edge_dot})')
    if len(edge_pts) == 0:
        print('No edge points found — nothing to visualize. '
              'Try a larger --edge-radius or --edge-dot.')
        return
    edge_tree = cKDTree(edge_pts)
    edge_spacing = float(np.median(edge_tree.query(edge_pts, k=2)[0][:, 1]))
    band = args.band if args.band is not None else 2.0 * edge_spacing
    print(f'edge point spacing (median NN): {edge_spacing:.5f} -> band={band:.5f}'
          + ('' if args.band is None else ' (from --band)'))

    # 3. exposed-region boundary arcs for every sphere
    batch, degenerate_pts = compute_exposed(sdf_points, sdf_values, args.degen_tol,
                                           source=args.neighbors)
    caps_by_sphere = per_sphere_caps(batch)
    n_arcs = np.asarray(batch['n_arcs'])
    n_caps = np.asarray(batch['n_caps'])
    if 'n_neighbors' in batch:
        nb = np.asarray(batch['n_neighbors'])
        print(f'neighbors/sphere: mean={nb.mean():.1f} max={nb.max()}  '
              f'caps/sphere: mean={n_caps.mean():.1f} max={n_caps.max()}')
    n_points = np.asarray(batch['n_points'])
    if 'total_arc' in batch:
        total_arc = np.asarray(batch['total_arc'])
    else:
        _, _alen = arc_midpoints_and_lengths(batch, len(sdf_points))
        total_arc = np.bincount(np.asarray(batch['arc_sphere_idx']),
                                weights=_alen, minlength=len(sdf_points))

    # 4. spheres whose nearest GT feature is a sharp edge
    d_edge, _ = edge_tree.query(sdf_points)
    gap = d_edge - radii                      # >= 0 for an empty sphere
    on_edge = gap <= band
    qs = np.percentile(gap, [1, 5, 25, 50, 75, 95])
    print('gap = dist(center, nearest GT edge point) - r   percentiles '
          '[1,5,25,50,75,95]: ' + ', '.join(f'{q:.5f}' for q in qs))
    print(f'{on_edge.sum()} / {len(sdf_points)} spheres have gap <= band '
          f'({100.0 * on_edge.mean():.1f}%) — i.e. they touch the surface at a crease')

    if args.sphere_ids:
        picks = [i for i in args.sphere_ids if 0 <= i < len(sdf_points)]
    else:
        cand = np.flatnonzero(on_edge & (n_arcs > 0))
        if cand.size == 0:
            print('No qualifying sphere has any exposed arc. '
                  'Widen --band, or check --degen-tol.')
            cand = np.flatnonzero(on_edge)
            if cand.size == 0:
                return
        # spread the picks over the chosen ordering instead of taking the extreme
        # few, so the page shows a range of cases rather than 24 near-clones
        if args.sort == 'gap':
            cand = cand[np.argsort(gap[cand])]
        elif args.sort == 'shortest-arc':
            cand = cand[np.argsort(total_arc[cand] / np.maximum(radii[cand], 1e-12))]
        elif args.sort == 'most-arcs':
            cand = cand[np.argsort(-n_arcs[cand])]
        elif args.sort == 'radius':
            cand = cand[np.argsort(-radii[cand])]
        else:
            cand = np.random.default_rng(args.seed).permutation(cand)
        if args.sort == 'gap':
            # every qualifying sphere has gap ~ 0 by construction, so the head of
            # this list is 24 near-identical cases; spread instead
            step = max(1, cand.size // args.max_spheres)
            picks = list(cand[::step][:args.max_spheres])
        else:
            picks = list(cand[:args.max_spheres])
    picks = [int(i) for i in picks]
    print(f'drawing {len(picks)} spheres: {picks}')

    # 5. comparison of exposed geometry on-edge vs elsewhere
    def summarize(mask, label):
        if mask.sum() == 0:
            print(f'  {label:<14} (empty)')
            return
        print(f'  {label:<14} n={mask.sum():>7}  '
              f'arcs/sphere={n_arcs[mask].mean():.2f}  '
              f'caps/sphere={n_caps[mask].mean():.2f}  '
              f'degen pts/sphere={n_points[mask].mean():.3f}  '
              f'total_arc/r={np.divide(total_arc[mask], np.maximum(radii[mask], 1e-12)).mean():.3f} rad  '
              f'fully carved={int(((n_arcs[mask] == 0) & (n_points[mask] == 0)).sum())}')

    print('exposed-region statistics:')
    summarize(on_edge, 'on GT edge')
    summarize(~on_edge, 'elsewhere')

    # 6. arc-level view: how long is an arc that runs along an edge, versus one
    #    that does not? (every arc in the model, keyed by its midpoint)
    mid, alen = arc_midpoints_and_lengths(batch, len(sdf_points))
    d_arc, _ = edge_tree.query(mid)
    near = d_arc <= band
    print(f'arc statistics over all {len(alen)} arcs '
          f'({near.sum()} within band of a GT edge):')
    for m, label in ((near, 'arc on edge'), (~near, 'arc elsewhere')):
        if m.sum() == 0:
            print(f'  {label:<14} (empty)')
            continue
        q = np.percentile(alen[m], [5, 25, 50, 75, 95])
        print(f'  {label:<14} n={m.sum():>8}  '
              f'length pct[5,25,50,75,95]=' + ', '.join(f'{v:.2e}' for v in q)
              + f'  below degen_tol={int((alen[m] < args.degen_tol).sum())}')

    arcs_by_sphere = {i: arcs_of_sphere(batch, caps_by_sphere, i, args.arc_samples)
                      for i in picks}
    stats = [dict(r=float(radii[i]), gap=float(gap[i]), n_arcs=int(n_arcs[i]),
                  n_caps=int(n_caps[i]), n_points=int(n_points[i]),
                  total_arc=float(total_arc[i])) for i in picks]
    print('per-sphere (arc lengths, longest first):')
    for i, st in zip(picks, stats):
        lens = sorted((a['length'] for a in arcs_by_sphere[i]), reverse=True)
        shown = ', '.join(f'{v:.2e}' for v in lens[:8])
        if len(lens) > 8:
            shown += f', ... (+{len(lens) - 8} more)'
        print(f'  sphere {i:>7}  r={st["r"]:.5f}  gap={st["gap"]:+.5f}  '
              f'caps={st["n_caps"]:>2}  arcs={st["n_arcs"]:>2}  '
              f'degen={st["n_points"]:>2}  lengths=[{shown}]')

    out_dir = args.out_dir or f'out/{args.name}'
    tag = f'{args.name}_{args.grid_len}'
    html = export_html(os.path.join(out_dir, f'arcs_near_edges_{tag}.html'),
                       mesh, picks, arcs_by_sphere, sdf_points, radii,
                       degenerate_pts, edge_tree, edge_pts, stats,
                       embed_js=args.embed_plotly)
    print(f'wrote {html}')

    if args.verify:
        pen = verify_arcs(batch, caps_by_sphere, sdf_points, radii, picks)
        print('penetration audit (max carve depth into another sphere, /r):')
        for i in picks:
            v = pen.get(i)
            if v is None:
                continue
            flag = '  <-- drawn inside another sphere' if v > 1e-6 else ''
            print(f'  sphere {i:>7}  {v:+.3e}{flag}')
        bad = sum(1 for v in pen.values() if v > 1e-6)
        print(f'  {bad}/{len(pen)} spheres have an arc carved into a neighbor')

    if not args.no_glb:
        palette = [(228, 26, 28), (55, 126, 184), (77, 175, 74), (152, 78, 163),
                   (255, 127, 0), (166, 86, 40), (247, 129, 191), (0, 191, 191)]
        tube_r = float(np.median(radii[picks])) * 0.02
        hits = [idx for i in picks
                for idx in edge_tree.query_ball_point(sdf_points[i], 2.5 * radii[i])]
        near_edges = edge_pts[np.unique(hits)] if hits else edge_pts[:0]
        glb = export_glb(os.path.join(out_dir, f'arcs_near_edges_{tag}.glb'),
                         mesh, picks, arcs_by_sphere, sdf_points, radii,
                         degenerate_pts, near_edges, palette, tube_r)
        print(f'wrote {glb}')


if __name__ == '__main__':
    main()
