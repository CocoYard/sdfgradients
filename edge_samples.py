""" Function 1: dense samples around the GT sharp edges.

    1. extract_sharp_edges() pulls the sharp edges out of the GT mesh once and
       stores them in their own file (FANDISK_EDGES for fandisk), plus an .obj of
       the same name for viewing them.
    2. sample_edge_points(spacing, on_edge, save_vis) samples near those edges:
       spacing  -- distance between samples, met as closely as possible
       on_edge  -- True: samples exactly on the sharp edges;
                   False: samples on the faces on both sides of the edges,
                   no farther than FACE_BAND_ROWS * spacing from them
       save_vis -- write the samples out for viewing
"""
import os
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from util import normalize_mesh

FANDISK_EDGES = 'examples/fandisk_sharp_edges.npz'
SHARP_ANGLE_DEG = 15    # dihedral angle above which a mesh edge is sharp
FACE_BAND_ROWS = 3      # face samples lie within this many spacings of an edge


def extract_sharp_edges(mesh_path='examples/fandisk.obj', out_path=FANDISK_EDGES):
    """ Find the sharp edges of the mesh and save them with the mesh to out_path (.npz).

        The mesh is normalized as for the SDF samples (util.normalize_mesh), so the
        edges are in the samples' frame. Stored: V, F (normalized mesh), edges (S, 2)
        vertex indices of the sharp edges. The edges alone also go to an .obj of the
        same name (polylines), only for viewing.
    """
    mesh = normalize_mesh(trimesh.load(mesh_path))
    sharp = mesh.face_adjacency_angles > np.radians(SHARP_ANGLE_DEG)
    edges = mesh.face_adjacency_edges[sharp]
    np.savez(out_path, V=mesh.vertices, F=mesh.faces, edges=edges)
    # only the vertices on sharp edges, so viewers that draw vertices show just the edges
    obj_path = os.path.splitext(out_path)[0] + '.obj'
    used, local = np.unique(edges, return_inverse=True)
    with open(obj_path, 'w') as f:
        for v in mesh.vertices[used]:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for a, b in local.reshape(-1, 2):
            f.write(f"l {a + 1} {b + 1}\n")
    print(f"[extract_sharp_edges] {len(edges)} sharp edges (> {SHARP_ANGLE_DEG} deg) -> {out_path}, {obj_path}")
    return out_path


def _sample_segments(segs, step):
    """ Points along each segment (S, 2, 3), at most `step` apart. """
    out = []
    for a, b in segs:
        k = max(int(np.ceil(np.linalg.norm(b - a) / step)), 1)
        out.append(a + np.linspace(0, 1, k + 1)[:, None] * (b - a))
    return np.vstack(out)


def _thin(points, spacing, order):
    """ Greedy Poisson-disk thinning in the given order: keep a maximal subset with no
        two points closer than spacing. """
    pairs = cKDTree(points).query_pairs(spacing, output_type='ndarray')
    nbrs = [[] for _ in range(len(points))]
    for i, j in pairs:
        nbrs[i].append(j)
        nbrs[j].append(i)
    alive = np.ones(len(points), dtype=bool)
    keep = []
    for i in order:
        if alive[i]:
            keep.append(i)
            alive[nbrs[i]] = False
    return np.sort(np.asarray(keep, dtype=np.int64))


def sample_edge_points(spacing, on_edge=False, save_vis=False, edges_file=FANDISK_EDGES):
    """ Sample points around the sharp edges stored in edges_file (see extract_sharp_edges).

        on_edge=True: points on the sharp edges themselves; their normal is undefined (NaN).
        on_edge=False: points on the faces next to the edges, at distance
            (0, FACE_BAND_ROWS * spacing] from them, with the face normal.
        Candidates are drawn densely and thinned so that no two samples are closer than
        `spacing`; each candidate dropped has a kept sample within `spacing`.
        save_vis writes out/<name>/samples<count>_gap<spacing>_<edge|face>.ply (points and
        normals).
        Returns (points (K, 3), normals (K, 3)).
    """
    data = np.load(edges_file)
    V, F, edges = data['V'], data['F'], data['edges']
    segs = V[edges]                                                     # (S, 2, 3)
    rng = np.random.default_rng(0)

    if on_edge:
        cand = _sample_segments(segs, spacing / 20)
        nrm = np.full(cand.shape, np.nan)
    else:
        band = FACE_BAND_ROWS * spacing
        edge_tree = cKDTree(_sample_segments(segs, spacing / 20))
        mesh = trimesh.Trimesh(V, F, process=False)
        # faces that come within band of an edge: centroid distance minus the face's reach
        reach = np.linalg.norm(mesh.triangles - mesh.triangles_center[:, None], axis=2).max(axis=1)
        d_face, _ = edge_tree.query(mesh.triangles_center)
        near = np.flatnonzero(d_face - reach <= band)
        sub = mesh.submesh([near], append=True)
        count = int(np.ceil(8 * sub.area / spacing ** 2))
        cand, fidx = trimesh.sample.sample_surface(sub, count, seed=rng)
        d, _ = edge_tree.query(cand)
        keep = (d > spacing / 20) & (d <= band)
        cand, nrm = cand[keep], sub.face_normals[fidx[keep]]

    # along the edges in order, so consecutive samples land one spacing apart
    order = np.arange(len(cand)) if on_edge else rng.permutation(len(cand))
    idx = _thin(cand, spacing, order)
    pts, nrm = cand[idx], nrm[idx]
    kind = 'edge' if on_edge else 'face'
    print(f"[sample_edge_points] {len(pts)} {kind} samples at spacing {spacing:g}")

    if save_vis:
        import edgeChamfer
        name = os.path.basename(edges_file).replace('_sharp_edges.npz', '')
        out_dir = f'out/{name}'
        os.makedirs(out_dir, exist_ok=True)
        ply_path = f'{out_dir}/samples{len(pts)}_gap{spacing:g}_{kind}.ply'
        edgeChamfer.write_ply_point_normal(ply_path, pts, np.nan_to_num(nrm), color=(255, 0, 0))
        print(f"  wrote {ply_path}")
    return pts, nrm
