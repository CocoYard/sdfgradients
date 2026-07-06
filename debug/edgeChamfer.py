"""
Edge Chamfer Distance (ECD) / Edge F1 (EF1)  + edge-point .ply export

Faithful re-packaging of the metric from
    Chen & Zhang, "Neural Marching Cubes", SIGGRAPH Asia 2021.
Reference implementation:
    https://github.com/czq142857/NMC/blob/main/eval_cd_nc_f1_ecd_ef1.py

Idea: instead of a Chamfer distance over the whole surface (which is dominated
by large smooth regions and is nearly blind to sharp features), ECD first
extracts "edge points" -- points whose local neighborhood contains a large
normal deviation -- and computes a symmetric Chamfer distance only on those.

Usage:
    import trimesh
    gt   = trimesh.load("gt.obj",   force="mesh")
    pred = trimesh.load("pred.obj", force="mesh")

    # metric only
    ecd, ef1 = compute_ecd(gt, pred)

    # metric + dump edge points for inspection in Blender
    ecd, ef1 = compute_ecd(gt, pred, export_prefix="mymodel")
    #   -> writes mymodel_gt_edge.ply (green) and mymodel_pred_edge.ply (red)

    # just export edge points of a single mesh
    export_edge_points(pred, "pred_edge.ply")
"""

import numpy as np
import trimesh
from scipy.spatial import cKDTree


# ----- default parameters, taken verbatim from the NMC eval script -----
SAMPLE_NUM = 1000_000            # points sampled per mesh
EF1_RADIUS = 0.004             # neighborhood radius for edge detection
EF1_DOTPRODUCT_THRESHOLD = 0.2  # |n . n_neighbor| below this => normals ~perpendicular => edge
EF1_THRESHOLD = 0.005          # distance threshold used for EF1 precision/recall
# NOTE: EF1_RADIUS and EF1_THRESHOLD are ABSOLUTE distances. They assume both
# meshes live in a unit-ish cube (NMC normalizes to coords in [-0.5, 0.5]).
# If your meshes are at a different scale, either pre-normalize them the same
# way, or pass normalize=True below.


# ----------------------------------------------------------------------
# PLY export
# ----------------------------------------------------------------------
def write_ply_point_normal(path, points, normals=None, color=None):
    """
    Write an ASCII .ply point cloud (optionally with per-point normals and a
    single RGB color). Blender's PLY importer reads x/y/z, nx/ny/nz and
    red/green/blue. `color` is an (r, g, b) tuple in 0..255.
    """
    points = np.asarray(points, np.float64).reshape(-1, 3)
    n = len(points)
    if normals is not None:
        normals = np.asarray(normals, np.float64).reshape(-1, 3)
        assert len(normals) == n

    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if normals is not None:
            f.write("property float nx\nproperty float ny\nproperty float nz\n")
        if color is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")

        cols = [points]
        if normals is not None:
            cols.append(normals)
        rows = np.hstack(cols)
        cstr = ""
        if color is not None:
            r, g, b = (int(c) for c in color)
            cstr = f" {r} {g} {b}"
        for row in rows:
            f.write(" ".join(f"{v:.6f}" for v in row) + cstr + "\n")
    return path


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _as_trimesh(m):
    """Accept a trimesh.Trimesh or a file path."""
    if isinstance(m, trimesh.Trimesh):
        return m
    loaded = trimesh.load(m, force="mesh")
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"Could not load a single mesh from {m!r}")
    return loaded


def _joint_normalize(gt_mesh, pred_mesh):
    """
    Fit both meshes into a unit cube centered at the origin using ONE shared
    transform defined by the GT bounding box, so relative distances between the
    two meshes are preserved. GT defines the canonical scale (standard practice).
    """
    gt_mesh = gt_mesh.copy()
    pred_mesh = pred_mesh.copy()
    lo, hi = gt_mesh.bounds
    center = (lo + hi) / 2.0
    scale = float((hi - lo).max())
    if scale <= 0:
        raise ValueError("Degenerate GT bounding box.")
    for mesh in (gt_mesh, pred_mesh):
        mesh.apply_translation(-center)
        mesh.apply_scale(1.0 / scale)   # GT now spans [-0.5, 0.5] on its longest axis
    return gt_mesh, pred_mesh


def _sample_with_normals(mesh, sample_num, rng):
    """Area-weighted surface samples + the face normal of each sampled face."""
    points, face_idx = trimesh.sample.sample_surface(mesh, sample_num, seed=rng)
    normals = mesh.face_normals[face_idx]
    return np.asarray(points, np.float64), np.asarray(normals, np.float64)


def _extract_edge_points(points, normals, radius, dot_threshold):
    """
    A point is an edge sample if, within `radius`, some neighbor has
    |normal . neighbor_normal| < dot_threshold (i.e. a nearly perpendicular
    normal -> a crease/corner passes nearby). np.abs is used exactly as in NMC,
    so opposite-facing normals (~ -1) count as aligned, not as an edge.

    Returns the flagged edge points AND their normals.
    """
    # unbalanced/non-compact build is ~2x faster here since the tree is used
    # once for query_pairs and thrown away (no repeated point-queries where a
    # balanced tree would pay off).
    tree = cKDTree(points, balanced_tree=False, compact_nodes=False)
    pairs = tree.query_pairs(radius, output_type="ndarray")
    flags = np.zeros(len(points), dtype=bool)
    if len(pairs) > 0:
        i, j = pairs[:, 0], pairs[:, 1]
        dp = np.abs(np.einsum("ij,ij->i", normals[i], normals[j]))
        mask = dp < dot_threshold
        flags[i[mask]] = True
        flags[j[mask]] = True
    return (np.ascontiguousarray(points[flags]),
            np.ascontiguousarray(normals[flags]))


# ----------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------
def extract_edge_points(
    mesh,
    sample_num=SAMPLE_NUM,
    ef1_radius=EF1_RADIUS,
    ef1_dotproduct_threshold=EF1_DOTPRODUCT_THRESHOLD,
    normalize_to_unit=False,
    seed=0,
):
    """
    Sample a mesh and return (edge_points, edge_normals) using the NMC criterion.

    normalize_to_unit: if True, fit the mesh alone into a unit cube first
    (only meaningful for single-mesh inspection; for a fair GT-vs-pred pair use
    compute_ecd(..., normalize=True) so both share ONE transform).
    """
    mesh = _as_trimesh(mesh)
    if normalize_to_unit:
        mesh = mesh.copy()
        lo, hi = mesh.bounds
        mesh.apply_translation(-(lo + hi) / 2.0)
        mesh.apply_scale(1.0 / float((hi - lo).max()))
    rng = np.random.default_rng(seed)
    pts, nls = _sample_with_normals(mesh, sample_num, rng)
    return _extract_edge_points(pts, nls, ef1_radius, ef1_dotproduct_threshold)


def export_edge_points(
    mesh,
    out_path,
    color=None,
    sample_num=SAMPLE_NUM,
    ef1_radius=EF1_RADIUS,
    ef1_dotproduct_threshold=EF1_DOTPRODUCT_THRESHOLD,
    normalize_to_unit=False,
    seed=0,
):
    """
    Extract edge points from a single mesh and write them to `out_path` (.ply),
    with normals and an optional (r,g,b) color. Returns the number of points.
    """
    pts, nls = extract_edge_points(
        mesh,
        sample_num=sample_num,
        ef1_radius=ef1_radius,
        ef1_dotproduct_threshold=ef1_dotproduct_threshold,
        normalize_to_unit=normalize_to_unit,
        seed=seed,
    )
    write_ply_point_normal(out_path, pts, nls, color=color)
    print(f"[export_edge_points] {len(pts)} edge points -> {out_path}")
    return len(pts)


def compute_ecd(
    gt_mesh,
    pred_mesh,
    sample_num=SAMPLE_NUM,
    ef1_radius=EF1_RADIUS,
    ef1_dotproduct_threshold=EF1_DOTPRODUCT_THRESHOLD,
    ef1_threshold=EF1_THRESHOLD,
    normalize=False,
    return_ef1=True,
    export_prefix=None,
    seed=0,
):
    """
    Compute Edge Chamfer Distance (and, optionally, Edge F1) between two meshes.

    Parameters
    ----------
    gt_mesh, pred_mesh : trimesh.Trimesh | str
        Ground-truth and predicted meshes (objects or file paths).
    normalize : bool
        If True, jointly normalize both meshes into a unit cube (GT-defined
        scale) before measuring, so the absolute thresholds are meaningful.
    return_ef1 : bool
        If True, also return the Edge F1 score.
    export_prefix : str | None
        If set, writes "{prefix}_gt_edge.ply" (green) and
        "{prefix}_pred_edge.ply" (red) with the extracted edge points, so you
        can overlay them in Blender. Coordinates are post-normalization if
        normalize=True.
    seed : int | None
        Seed for the surface sampling (reproducibility).

    Returns
    -------
    ecd (float)                       if return_ef1 is False
    (ecd, ef1) (tuple of two floats)  if return_ef1 is True

    ECD is the symmetric sum of mean SQUARED nearest-neighbor distances between
    the two edge-point sets (matching the NMC script). Smaller is better.
    """
    gt_mesh = _as_trimesh(gt_mesh)
    pred_mesh = _as_trimesh(pred_mesh)
    if normalize:
        gt_mesh, pred_mesh = _joint_normalize(gt_mesh, pred_mesh)

    rng = np.random.default_rng(seed)
    gt_points, gt_normals = _sample_with_normals(gt_mesh, sample_num, rng)
    pred_points, pred_normals = _sample_with_normals(pred_mesh, sample_num, rng)

    gt_edge, gt_edge_n = _extract_edge_points(
        gt_points, gt_normals, ef1_radius, ef1_dotproduct_threshold
    )
    pred_edge, pred_edge_n = _extract_edge_points(
        pred_points, pred_normals, ef1_radius, ef1_dotproduct_threshold
    )

    if export_prefix is not None:
        write_ply_point_normal(f"{export_prefix}_pred_edge.ply",
                               pred_edge, pred_edge_n, color=(220, 60, 60))  # red
        print(f"[compute_ecd] exported {len(gt_edge)} gt / {len(pred_edge)} pred "
              f"edge points with prefix '{export_prefix}'")

    # Degenerate handling mirrors the NMC script.
    if len(gt_edge) == 0:
        return (0.0, 1.0) if return_ef1 else 0.0
    if len(pred_edge) == 0:
        pred_edge = np.zeros((1, 3), np.float64)

    # gt -> pred
    tree = cKDTree(pred_edge)
    d_gp, _ = tree.query(gt_edge, k=1, workers=-1)
    recall = float(np.mean(d_gp < ef1_threshold))
    gt2pred_mean_ecd = float(np.mean(np.square(d_gp)))

    # pred -> gt
    tree = cKDTree(gt_edge)
    d_pg, _ = tree.query(pred_edge, k=1, workers=-1)
    precision = float(np.mean(d_pg < ef1_threshold))
    pred2gt_mean_ecd = float(np.mean(np.square(d_pg)))

    ecd = gt2pred_mean_ecd + pred2gt_mean_ecd

    if not return_ef1:
        return ecd
    ef1 = (2 * recall * precision / (recall + precision)) if (recall + precision) > 0 else 0.0
    return ecd, ef1


if __name__ == "__main__":
    # tiny smoke test + export
    # box = trimesh.creation.box(extents=(0.6, 0.6, 0.6))

    # moved = box.copy()
    # moved.apply_translation((0.02, 0.0, 0.0))
    # ecd, ef1 = compute_ecd(box, moved, seed=0, export_prefix="_smoketest")
    # print(f"shifted cube : ECD={ecd:.3e}  EF1={ef1:.3f}")

    # export_edge_points(box, "_smoketest_single.ply", color=(60, 120, 220))
    name = "eiffel"
    gt_path = f"../gallery/examples/{name}.obj"
    mesh = trimesh.load(gt_path)
    for res in [100]:
        # gallery/out/fandisk/interpolant_100_10_shortArcs_MESforce_PU.obj
        ours_path = f"../gallery/out/{name}/interpolant_{res}_15_shortArcs_MES.obj"
        pred_mesh = trimesh.load(ours_path)
        ecd, ef1 = compute_ecd(mesh, pred_mesh, seed=0, export_prefix=f"data/ours_{res}")
        print(f"ours_{res} : ECD={ecd:.3e}  EF1={ef1:.3f}")
        for method in ['mes', 'rfta', 'sample_points']:
            pred_path = f"../gallery/out/{name}/{method}_{res}.obj"
            pred_mesh = trimesh.load(pred_path)
            ecd, ef1 = compute_ecd(mesh, pred_mesh, seed=0, export_prefix=f"data/{method}_{res}")
            print(f"{method}_{res} : ECD={ecd:.3e}  EF1={ef1:.3f}")
        print()

