import numpy as np
from scipy.spatial.distance import cdist
import trimesh
from bench import sphere_exposed_pybind as sep

def get_sphere_data(batch, i):
    """从 batch 结果里切出第 i 个球的数据"""
    arc_mask = batch['arc_sphere_idx'] == i
    cap_mask = batch['cap_sphere_idx'] == i
    pt_mask  = batch['point_sphere_idx'] == i
    return {
        'n_caps'        : int(batch['n_caps'][i]),
        'total_arc'     : float(batch['total_arc'][i]),
        'arc_cap_idx'   : batch['arc_cap_idx'][arc_mask],
        'arc_start'     : batch['arc_start'][arc_mask],
        'arc_end'       : batch['arc_end'][arc_mask],
        'cap_normals'   : batch['cap_normals'][cap_mask],   # (K,3) 法向量
        'cap_d'         : batch['cap_d'][cap_mask],         # (K,)  平面偏移
        'cap_centers'   : batch['cap_centers'][cap_mask],   # (K,3) 边界圆心
        'cap_radii'     : batch['cap_radii'][cap_mask],     # (K,)  边界圆半径
        'cap_u'         : batch['cap_u'][cap_mask],         # (K,3) 边界圆 u 轴
        'cap_v'         : batch['cap_v'][cap_mask],         # (K,3) 边界圆 v 轴
        'exposed_points': batch['point_positions'][pt_mask],# (P,3) 退化点
    }

def sample_arcs(i, batch, num_points=100):
    sp = get_sphere_data(batch, i)
    points, _ = sep.sample_arcs(
        sp['cap_centers'],
        sp['cap_radii'],
        sp['cap_u'],
        sp['cap_v'],
        sp['arc_cap_idx'],
        sp['arc_start'],
        sp['arc_end'],
        num_points
    )
    return points

def query_closest_on_arcs(query_pts, i, batch):
    sp = get_sphere_data(batch, i)
    closest, distances, _ = sep.query_closest_on_arcs(
        query_pts,
        sp['cap_centers'], sp['cap_radii'],
        sp['cap_u'],       sp['cap_v'],
        sp['arc_cap_idx'], sp['arc_start'], sp['arc_end']
    )
    return closest, distances

    
def are_points_visible(points, sdf_points, sdf_values, epsilon=1e-8):
    """
    Check if query points are visible from the SDF points, i.e. not occluded by any other SDF point's sphere.
    This is a simple visibility test based on the SDF values, which represent the radius of the sphere around each SDF point.
    A point is considered visible if it is outside the sphere of every SDF point, with a small epsilon tolerance to avoid numerical issues.
    Parameters
    ----------
    points : (N, d) array of query points to test for visibility.
    sdf_points : (M, d) array of SDF points, each with an associated SDF value.
    sdf_values : (M,) array of SDF values at the SDF points, representing the radius of the sphere around each SDF point.
    epsilon : float, optional, default=1e-8 A small margin to ensure points on the boundary are considered visible.
    Returns
    -------
    visible : (N,) boolean array, True if the corresponding point is visible, False if it is occluded by any SDF point's sphere.
    """
    # check if points are not inside any other sdf_point's sphere
    nan_mask = np.any(np.isnan(points), axis=1)
    
    dists = cdist(points, sdf_points, 'euclidean')
    radii = np.abs(sdf_values)
    inside = dists < (radii[np.newaxis, :] - epsilon)
    result = ~np.any(inside, axis=1)
    
    # Points containing NaN are marked as not visible
    result[nan_mask] = False
    
    return result

def mesh_distances(recon : trimesh.Trimesh, gt_mesh : trimesh.Trimesh, verbose=False,
                   n_samples: int = 100_000, f1_tau: float = 0.01):
    """
    Surface-sampled Hausdorff / Chamfer / F1 between two meshes.

    Both meshes are uniformly sampled on their surface (n_samples points each);
    distances are computed symmetrically between each sample set and the *other*
    mesh's surface via an AABB tree. This is fair regardless of tessellation
    density (vertex-only metrics penalise sparsely-sampled reconstructions
    unfairly even when the surface is correct).

    Hausdorff is the symmetric max (intentionally outlier-sensitive).
    Chamfer is the L1 symmetric mean of the two directional means.
    F1 uses threshold `f1_tau` on the unit-normalised scale.
    """
    import igl
    recon_pts, _ = trimesh.sample.sample_surface(recon, n_samples)
    gt_pts,    _ = trimesh.sample.sample_surface(gt_mesh, n_samples)

    gt_V = np.asarray(gt_mesh.vertices, dtype=np.float64)
    gt_F = np.asarray(gt_mesh.faces, dtype=np.int32)
    rc_V = np.asarray(recon.vertices, dtype=np.float64)
    rc_F = np.asarray(recon.faces, dtype=np.int32)

    sqrD_r2g, _, _ = igl.point_mesh_squared_distance(
        np.asarray(recon_pts, dtype=np.float64), gt_V, gt_F)
    sqrD_g2r, _, _ = igl.point_mesh_squared_distance(
        np.asarray(gt_pts, dtype=np.float64), rc_V, rc_F)
    d_r2g = np.sqrt(sqrD_r2g)
    d_g2r = np.sqrt(sqrD_g2r)

    hausdorff = max(d_r2g.max(), d_g2r.max())
    chamfer = (d_r2g.mean() + d_g2r.mean()) / 2

    precision = float((d_r2g < f1_tau).mean())
    recall    = float((d_g2r < f1_tau).mean())
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    if verbose:
        print(f"  Hausdorff: {hausdorff:.6f}  Chamfer: {chamfer:.6f}  "
              f"F1@{f1_tau}: {f1:.4f} (P={precision:.4f} R={recall:.4f})")
    return hausdorff, chamfer, f1

def _point_to_polylines_min_dist(points, polylines):
    """Min distance from each query point to the *segments* of polylines."""
    min_dists = np.full(len(points), np.inf)
    for poly in polylines:
        if len(poly) < 2:
            dists = np.linalg.norm(points - poly[0], axis=1)
            min_dists = np.minimum(min_dists, dists)
            continue
        a = poly[:-1]
        b = poly[1:]
        ab = b - a
        ab_sq = np.sum(ab ** 2, axis=1)
        chunk = max(1, 50_000 // max(len(a), 1))
        for i0 in range(0, len(points), chunk):
            i1 = min(i0 + chunk, len(points))
            pts = points[i0:i1]
            ap = pts[:, None, :] - a[None, :, :]
            t = np.sum(ap * ab[None, :, :], axis=2) / np.maximum(ab_sq[None, :], 1e-30)
            t = np.clip(t, 0.0, 1.0)
            closest = a[None, :, :] + t[:, :, None] * ab[None, :, :]
            dists = np.linalg.norm(pts[:, None, :] - closest, axis=2)
            min_dists[i0:i1] = np.minimum(min_dists[i0:i1], np.min(dists, axis=1))
    return min_dists

def _normalise_to_polyline_list(shape):
    """Convert shape to list-of-polylines.
    If shape is an ndarray with NaN separator rows, splits into separate polylines."""
    if isinstance(shape, np.ndarray):
        nan_mask = np.any(np.isnan(shape), axis=1)
        if np.any(nan_mask):
            polylines = []
            start = 0
            for i in range(len(shape)):
                if nan_mask[i]:
                    if i > start:
                        polylines.append(shape[start:i])
                    start = i + 1
            if start < len(shape):
                polylines.append(shape[start:])
            return polylines
        return [shape]
    return list(shape)

def shape_distances(shape, original_shape):
    """Compute multiple shape-difference metrics between two sets of polylines.

    Parameters
    ----------
    shape : list of (N_i, 2) arrays, or a single (N, 2) array
        One or more polylines (e.g. from marching cubes).
    original_shape : (M, 2) array or list of (M_j, 2) arrays
        The reference polyline(s).

    Returns
    -------
    dict with keys:
        hausdorff : float — max of directed Hausdorff distances (worst-case)
        hausdorff_95 : float — 95-th percentile Hausdorff (robust worst-case)
        chamfer : float — symmetric mean point-to-segment distance
        rms : float — symmetric root-mean-square point-to-segment distance
        iou : float — intersection-over-union of enclosed areas (0–1, higher=better)
    """
    shape_list = _normalise_to_polyline_list(shape)
    orig_list  = _normalise_to_polyline_list(original_shape)

    shape_verts = np.concatenate(shape_list, axis=0)
    orig_verts  = np.concatenate(orig_list, axis=0)

    # directed distances
    d_s2o = _point_to_polylines_min_dist(shape_verts, orig_list)
    d_o2s = _point_to_polylines_min_dist(orig_verts, shape_list)

    # Hausdorff
    hausdorff = float(max(np.max(d_s2o), np.max(d_o2s)))
    hausdorff_95 = float(max(np.percentile(d_s2o, 95), np.percentile(d_o2s, 95)))

    # Chamfer (symmetric mean)
    chamfer = float(np.mean(d_s2o) + np.mean(d_o2s)) / 2.0

    # RMS (symmetric root-mean-square)
    rms = float(np.sqrt((np.mean(d_s2o ** 2) + np.mean(d_o2s ** 2)) / 2.0))

    # IoU via rasterisation on a 500×500 grid in [0,1]²
    from matplotlib.path import Path
    res = 500
    xs = np.linspace(0, 1, res)
    ys = np.linspace(0, 1, res)
    grid = np.array(np.meshgrid(xs, ys)).T.reshape(-1, 2)

    def _raster(polylines):
        mask = np.zeros(len(grid), dtype=bool)
        for poly in polylines:
            if len(poly) < 3:
                continue
            # close the polygon if not already closed
            if not np.allclose(poly[0], poly[-1]):
                poly = np.vstack([poly, poly[:1]])
            path = Path(poly)
            mask |= path.contains_points(grid)
        return mask

    mask_s = _raster(shape_list)
    mask_o = _raster(orig_list)
    intersection = np.sum(mask_s & mask_o)
    union = np.sum(mask_s | mask_o)
    iou = float(intersection / max(union, 1))

    return {
        'hausdorff': hausdorff,
        'hausdorff_95': hausdorff_95,
        'chamfer': chamfer,
        'rms': rms,
        'iou': iou,
    }

def print_shape_distances(label, shape, original_shape):
    """Compute and pretty-print all shape distance metrics."""
    d = shape_distances(shape, original_shape)
    print(f"  {label}:  Hausdorff={d['hausdorff']:.4f}  H95={d['hausdorff_95']:.4f}  "
          f"Chamfer={d['chamfer']:.6f}  RMS={d['rms']:.4f}  IoU={d['iou']:.3f}")
