import numpy as np

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
