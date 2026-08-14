import numpy as np
from interpolation import CurlFree_Interpolator, Interpolator
import torch
from scipy.optimize import minimize
import visible_arcs as va
import util
from util import print_shape_distances
from util import are_points_visible
from SDF_to_surface_3D import Tolerance, Options
try:
    import mes_contact
except ImportError:
    mes_contact = None

def iterative_gradient_alignment(points, values, init_gradients, interpolator : Interpolator, visible_arcs, short_arc_idx, num_iter=10, gt=None):
    """
    Iteratively refine SDF gradients by finding the best gradient direction where 
    the interpolant gradient direction is the closest to the direction between the
    projected point and its center.

    Algorithm (each iteration):
      1. Fit a Interpolator with current gradients (use_projection=True).
      2. For each sample point, search over directions to find the one whose
         projection P - s*g lands closest to the zero level set of the
         interpolant (via sample_best_gradient).
    """
    points = np.asarray(points, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64).ravel()
    gradients = np.array(init_gradients, dtype=np.float64, copy=True)
    interpolator.fit(points, values, gradients)
    if gt is not None:
            print_shape_distances("Before refinement", interpolator.extract_zero_level_set(bounds=((0, 1), (0, 1)), resolution=400), gt)
    for it in range(num_iter):
        new_gradients = np.zeros_like(gradients)
        for i in range(len(points)):
            if i in short_arc_idx:
                new_gradients[i] = gradients[i]  # Skip points with very short visible arcs
                continue
            # ----- Find best gradient via angular search on the interpolant -----
            init_guess_angle = np.arctan2(gradients[i, 1], gradients[i, 0]) if values[i] < 0 else np.arctan2(-gradients[i, 1], -gradients[i, 0])
            grad = interpolator.sample_gradient_by_alignment(
                points[i], values[i], visible_arcs=visible_arcs[i], num_coarse=15, initial_guess=init_guess_angle
            )
            angle = np.arctan2(grad[1], grad[0]) if values[i] < 0 else np.arctan2(-grad[1], -grad[0])
            if va.angle_in_arcs(angle, visible_arcs[i]):
                new_gradients[i] = grad
            else:
                new_gradients[i] = gradients[i]  # Keep original if new grad is not in visible arcs
        # ----- Convergence diagnostic -----
        cos_sim = np.sum(gradients * new_gradients, axis=1)
        mean_cos = np.mean(cos_sim)
        max_angle_deg = np.degrees(np.arccos(np.clip(np.min(cos_sim), -1, 1)))

        # Projection error: f(P - s*g) should be ~0
        proj_pts = points - values[:, np.newaxis] * new_gradients
        proj_vals = interpolator.predict(proj_pts)
        proj_rmse = np.sqrt(np.mean(proj_vals**2))

        print(f"Iter {it+1:3d} | mean cos_sim: {mean_cos:.6f}  "
              f"max angle change: {max_angle_deg:.2f}\u00b0  "
              f"proj RMSE: {proj_rmse:.6e}")
        gradients = new_gradients
        interpolator.fit(points, values, gradients)
        # if gt is not None:
        #     print_shape_distances("    ", interpolator.extract_zero_level_set(bounds=((0, 1), (0, 1)), resolution=400), gt)
    return gradients

'''
    Compute various shape distance metrics between two shapes represented as lists of polylines.
def find_angle_point(circle_center, radius, p1, p2, is_max=True):
    cx, cy = circle_center
    p1 = np.array(p1)
    p2 = np.array(p2)
    
    def objective(theta_args):
        theta = theta_args[0] if isinstance(theta_args, np.ndarray) else theta_args
        
        # 现在 P 是一个干净的 (2,) 形状的 1D 数组
        P = np.array([cx + radius * np.cos(theta), cy + radius * np.sin(theta)])
        
        v1 = p1 - P
        v2 = p2 - P
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 1.0  
            
        cos_theta = np.dot(v1, v2) / (norm1 * norm2)
        return np.clip(cos_theta, -1.0, 1.0) if is_max else -np.clip(cos_theta, -1.0, 1.0)

    # 1. 全局粗搜
    thetas = np.linspace(0, 2 * np.pi, 36)
    # 注意这里粗搜传入的是标量，所以上面的 isinstance 检查能兼容这两种情况
    best_initial_theta = thetas[np.argmin([objective(t) for t in thetas])]

    # 2. 局部精搜
    res = minimize(
        objective, 
        x0=[best_initial_theta], 
        bounds=[(0, 2 * np.pi)],
        method='L-BFGS-B'
    )
    
    best_theta = res.x[0]
    best_point = (cx + radius * np.cos(best_theta), cy + radius * np.sin(best_theta))
    # objective returns cos(angle) when is_max, -cos(angle) when not is_max
    # so we need to undo the negation to get the actual cos(angle)
    raw = objective(best_theta)
    actual_cos = raw if is_max else -raw
    best_angle_rad = np.arccos(np.clip(actual_cos, -1.0, 1.0))
    return best_point, best_theta, best_angle_rad
    
def find_neighbors(proj_points, interpolator : Interpolator):
    """
    For each projected point, find its neighbors based on 0-level set proximity using
    the interpolator. Returns a list of neighbor indices for each projected point.

    Strategy:
      1. Extract the zero level set as a collection of polylines.
      2. Build a global arc-length parameterisation over all polyline vertices.
      3. Map each projected point to the arc-length of its nearest contour vertex.
      4. Sort projected points by arc-length and return the two adjacent ones as neighbors.
    """
    zero_level_contours = interpolator.extract_zero_level_set(bounds=((0, 1), (0, 1)), resolution=400)

    # Build flat arrays: contour vertices, their polyline id, and local arc-length within that polyline.
    # Each polyline is treated independently so that different contours are never cross-connected.
    all_contour_pts = []
    vert_poly_id = []
    vert_arc = []
    poly_closed = []   # whether each polyline is a closed loop
    all_seg_starts = []    # edge segment start points
    all_seg_ends = []      # edge segment end points
    seg_poly_ids = []      # polyline id for each segment
    seg_arc_starts = []    # arc-length at segment start
    seg_arc_ends = []      # arc-length at segment end

    for pid, poly in enumerate(zero_level_contours):
        poly = np.asarray(poly)
        if len(poly) < 2:
            all_contour_pts.append(poly[0])
            vert_poly_id.append(pid)
            vert_arc.append(0.0)
            poly_closed.append(False)
            continue
        seg_lengths = np.linalg.norm(np.diff(poly, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        # A contour is closed when its first and last vertices coincide
        is_closed = np.linalg.norm(poly[0] - poly[-1]) < 1e-6
        for pt, arc in zip(poly, cum):
            all_contour_pts.append(pt)
            vert_poly_id.append(pid)
            vert_arc.append(arc)
        poly_closed.append(is_closed)
        for j in range(len(poly) - 1):
            all_seg_starts.append(poly[j])
            all_seg_ends.append(poly[j + 1])
            seg_poly_ids.append(pid)
            seg_arc_starts.append(cum[j])
            seg_arc_ends.append(cum[j + 1])

    if len(all_contour_pts) == 0:
        return [[] for _ in range(len(proj_points))]

    N = len(proj_points)

    if len(all_seg_starts) == 0:
        # All polylines are single points — no meaningful neighbors
        return [[] for _ in range(N)]

    # Convert edge arrays for vectorized nearest-segment projection
    all_seg_starts = np.array(all_seg_starts)    # (E, 2)
    all_seg_ends = np.array(all_seg_ends)        # (E, 2)
    seg_poly_ids = np.array(seg_poly_ids)        # (E,)
    seg_arc_starts = np.array(seg_arc_starts)    # (E,)
    seg_arc_ends = np.array(seg_arc_ends)        # (E,)

    # Map each projected point to its nearest contour SEGMENT and compute
    # an interpolated arc-length.  This yields a continuous parameterisation
    # so that sorting by arc-length reliably gives true left / right neighbors.
    proj_poly_id = np.empty(N, dtype=int)
    proj_arc = np.empty(N)
    E = len(all_seg_starts)
    AB = all_seg_ends - all_seg_starts           # (E, 2)
    AB_len2 = np.sum(AB**2, axis=1)              # (E,)
    chunk = max(1, 50_000 // max(E, 1))
    for i0 in range(0, N, chunk):
        i1 = min(i0 + chunk, N)
        batch = proj_points[i0:i1]               # (B, 2)
        B = batch.shape[0]
        AP = batch[:, None, :] - all_seg_starts[None, :, :]        # (B, E, 2)
        t = np.sum(AP * AB[None, :, :], axis=2) / np.maximum(AB_len2[None, :], 1e-20)
        t = np.clip(t, 0.0, 1.0)                                   # (B, E)
        closest = all_seg_starts[None, :, :] + t[:, :, None] * AB[None, :, :]  # (B, E, 2)
        dists = np.linalg.norm(batch[:, None, :] - closest, axis=2) # (B, E)
        nearest_seg = np.argmin(dists, axis=1)                       # (B,)
        nearest_t = t[np.arange(B), nearest_seg]                     # (B,)
        proj_poly_id[i0:i1] = seg_poly_ids[nearest_seg]
        proj_arc[i0:i1] = seg_arc_starts[nearest_seg] + nearest_t * (seg_arc_ends[nearest_seg] - seg_arc_starts[nearest_seg])

    # For each projected point find arc-length neighbors *within the same polyline*.
    # Closed polylines wrap around: first and last entries are each other's neighbors.
    neighbors_idx = [[] for _ in range(N)]

    for pid in range(len(poly_closed)):
        members = np.where(proj_poly_id == pid)[0]
        if len(members) == 0:
            continue
        # Sort members by local arc-length
        sorted_members = members[np.argsort(proj_arc[members])]  # sorted original indices
        M = len(sorted_members)
        is_closed = poly_closed[pid]
        for rank in range(M):
            i = int(sorted_members[rank])
            nbrs = []
            if rank > 0:
                nbrs.append(int(sorted_members[rank - 1]))
            elif is_closed and M > 1:
                nbrs.append(int(sorted_members[M - 1]))   # wrap: first -> last
            if rank < M - 1:
                nbrs.append(int(sorted_members[rank + 1]))
            elif is_closed and M > 1:
                nbrs.append(int(sorted_members[0]))        # wrap: last -> first
            neighbors_idx[i] = nbrs

    return neighbors_idx

def iterative_smoothing(points, values, init_gradients, interpolator : Interpolator, visible_arcs, short_arc_idx, num_iter=10):
    """
    Iteratively smooth SDF gradients by refining them based on projections' neighbors.
    Algorithm (each iteration):
      1. Compute 2 neighbors for each projection of sample points onto the zero level set.
      2. For each projected point, move it on its visible arc to make the angle formed by 
         its neighbors more obtuse, thus encouraging smoother gradient directions.
      3. Repeat for num_iter iterations.

    Parameters
    ----------
    points : (N, 2) array
        Sample point coordinates.
    values : (N,) array
        Signed distance values at each sample point.
    init_gradients : (N, 2) array
        Initial gradient estimates.
    visible_arcs: a list of lists of tuples: point index -> list of visible arcs
        A collection of visible arcs that can be used to clamp the gradients.
    short_arc_idx: a set of point indices whose visible arcs are extremely short, so we skip them.
    num_iter : int
        Number of smoothing iterations (default 10).
    
    Returns
    -------
    gradients : (N, 2) array
        Smoothed gradient vectors after iterative refinement.
    """
    points = np.asarray(points, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64).ravel()
    gradients = np.array(init_gradients, dtype=np.float64, copy=True)
    proj_points = points - values[:, np.newaxis] * gradients
    neighbors_idx = find_neighbors(proj_points, interpolator)
    for it in range(num_iter):
        # Update projected points to encourage smoother angles between neighbors
        new_projected_points = proj_points.copy()
        for i in range(len(points)):
            if i in short_arc_idx:
                continue  # Skip points with very short visible arcs
            if len(neighbors_idx[i]) < 2:
                continue  # Need at least 2 neighbors to define an angle
            p_i = proj_points[i]
            n1, n2 = proj_points[neighbors_idx[i][:2]]  # Take first 2 neighbors
            _d = n2 - n1
            if np.linalg.norm(_d) < 1e-6:
                continue  # Neighbors are too close, skip to avoid instability
            if np.linalg.norm(n1 - points[i]) < abs(values[i]) or np.linalg.norm(n2 - points[i]) < abs(values[i]):
                continue  # Neighbors are inside the circle, skip to avoid instability
            # if the line segment n1-n2 intersects the circle around points[i] with radius abs(values[i]), use min angle
            _len2 = np.dot(_d, _d)
            if _len2 > 0:
                _t = np.clip(np.dot(points[i] - n1, _d) / _len2, 0.0, 1.0)
            else:
                _t = 0.0
            _closest = n1 + _t * _d
            if np.linalg.norm(_closest - points[i]) < abs(values[i]):
                # intersection between n1-n2 and circle
                best_point, best_theta, _ = find_angle_point(points[i], np.linalg.norm(points[i] - p_i), n1, n2, is_max=True)
                # check the distance between best_point and original projection
                if np.linalg.norm(best_point - p_i) < np.sqrt(2) * abs(values[i]):
                    if va.angle_in_arcs(best_theta, visible_arcs[i]):
                        new_projected_points[i] = best_point
                else:
                    if va.angle_in_arcs(best_theta + np.pi, visible_arcs[i]):
                        new_projected_points[i] = 2 * points[i] - best_point
            else:
                # no intersection, just move p_i to the point on the line n1-n2 that forms the largest angle
                best_point, best_theta, max_angle_rad = find_angle_point(points[i], np.linalg.norm(points[i] - p_i), n1, n2, is_max=True)
                if np.linalg.norm(best_point - p_i) < np.sqrt(2) * abs(values[i]):
                    if va.angle_in_arcs(best_theta, visible_arcs[i]):
                        new_projected_points[i] = best_point
                else:
                    if va.angle_in_arcs(best_theta + np.pi, visible_arcs[i]):
                        new_projected_points[i] = 2 * points[i] - best_point
            # if i == 429:
            #     print(np.linalg.norm(_closest - points[i]) < abs(values[i])) # intersection happens
            #     print(np.linalg.norm(best_point - points[i]), 2*abs(values[i]))
            #     print(f"Iter {it+1}, point {i}: center={points[i]} p_i={p_i}, n1={n1}, n2={n2}, best_point={best_point}, max_rad={max_angle_rad}, \
            #           original_theta={np.arccos(np.dot(n1 - p_i, n2 - p_i) / (np.linalg.norm(n1 - p_i) * np.linalg.norm(n2 - p_i)))}, \
            #             new_proj={new_projected_points[i]}")
        proj_points = new_projected_points
        print(f"Iter {it+1:3d} completed.")
    # compute final unit gradients: g = sign(s) * (P - P_proj) / |P - P_proj|
    direction = points - proj_points
    norms = np.linalg.norm(direction, axis=1, keepdims=True)
    s_sign = np.where(values[:, np.newaxis] >= 0, 1.0, -1.0)
    gradients = s_sign * direction / (norms + 1e-12)
    return gradients
'''

def _point_to_polylines_min_dist(points, polylines):
    """Min distance from each query point to the *segments* of polylines."""
    min_dists = np.full(len(points), np.inf)
    nearest_points = np.zeros_like(points)
    for poly in polylines:
        if len(poly) < 2:
            dists = np.linalg.norm(points - poly[0], axis=1)
            better = dists < min_dists
            min_dists[better] = dists[better]
            nearest_points[better] = poly[0]
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
            best_idx = np.argmin(dists, axis=1)
            best_dists = dists[np.arange(i1 - i0), best_idx]
            best_points = closest[np.arange(i1 - i0), best_idx]

            better = best_dists < min_dists[i0:i1]
            if np.any(better):
                global_idx = np.where(better)[0] + i0
                min_dists[global_idx] = best_dists[better]
                nearest_points[global_idx] = best_points[better]
    return min_dists, nearest_points

def iterative_projection(points, values, init_gradients, interpolator : Interpolator, visible_arcs, short_arc_idx, num_iter=10,
                         num_coarse=24, refine_steps=4, num_refine=12, optim_steps=10, lr=0.2,
                         gt=None, colinear_neighbors=None, clamp=True):
    """
    Iteratively refine SDF gradients by projecting sample points onto the zero
    level set of an interpolant. Mirrors the 3D pipeline in
    cpp/src/optimization.cpp::iterative_projection_3d so the 2D and 3D
    algorithms behave identically.

    Each iteration:
      1. Find best gradient via angular search (sample_best_gradients).
      2. (optional) Clamp updated gradients back into the visible arcs.
      3. Compute projections q = p - s*g for old & new gradients.
      4. Visibility check: for each sample, decide whether the new projection
         lies in a visible region (i.e., not inside any other sphere). Reject
         updates that would turn a visible projection into an invisible one,
         and reject updates for degenerate-arc samples.
      5. Build the fit mask = (vis_old | vis_new) and refit the interpolator
         using only those samples.
      6. Cache visibility for the next iteration.

    Note: `visible_arcs` and `colinear_neighbors` are kept in the signature
    for backwards compatibility but no longer participate in the per-update
    gating \u2014 visibility is now decided by the projection's containment in
    other spheres (consistent with the 3D version).

    Returns
    -------
    gradients : (N, 2) array
        Refined unit gradient vectors.
    interpolator : Interpolator
        The final fitted interpolator.
    """
    points = np.asarray(points, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64).ravel()
    gradients = np.array(init_gradients, dtype=np.float64, copy=True)
    N = len(points)

    # Normalize initial gradients
    norms = np.linalg.norm(gradients, axis=1, keepdims=True)
    gradients /= np.maximum(norms, 1e-12)
    init_angles = np.arctan2(gradients[:, 1], gradients[:, 0])

    if gt is not None:
        print_shape_distances("Before refinement", interpolator.extract_zero_level_set(bounds=((0, 1), (0, 1)), resolution=400), gt)

    vis_cached = None  # cached vis_old across iters

    for it in range(num_iter):
        # \u2500\u2500 Step 1: Find best gradient via projected gradient ascent \u2500\u2500
        new_gradients = interpolator.optimize_best_gradients(
            points, values,
            num_coarse=num_coarse,
            optim_steps=optim_steps,
            lr=lr,
            initial_guess=init_angles
        )

        # \u2500\u2500 Step 2: Clamp updated gradients to visible arcs \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        # Mirrors `clamp_gradients_to_arcs` in cpp/src/optimization.cpp.
        # The 2D arc-clamp uses visible_arcs (no tunable tolerance) so we
        # call it once instead of the C++ retry-with-shrinking-tol loop.
        if clamp:
            va.clamp_gradients_to_arcs(
                new_gradients, visible_arcs, short_arc_idx, values,
                skip_degenerate=True)

        # \u2500\u2500 Step 3: Visibility checks on projections \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        proj_new = points - values[:, np.newaxis] * new_gradients
        vis_new = are_points_visible(proj_new, points, values)

        if vis_cached is not None:
            vis_old = vis_cached
        else:
            proj_old = points - values[:, np.newaxis] * gradients
            vis_old = are_points_visible(proj_old, points, values)

        # \u2500\u2500 Step 4: Reject updates that lose visibility or hit degen \u2500\u2500
        skip = vis_old & ~vis_new
        if it == 0:
            # only use those whose visible arcs are not 2Pi
            for i in range(N):
                # print(f"Point {i}: visible arcs = {visible_arcs[i]}")
                if visible_arcs[i] and visible_arcs[i][0][1] - visible_arcs[i][0][0] >= 2 * np.pi - 1e-3:
                    skip[i] = True

        for i in range(N):
            if i in short_arc_idx:
                skip[i] = True
        new_gradients[skip] = gradients[skip]
        vis_next = np.where(skip, vis_old, vis_new)
        vis_cached = vis_next

        # \u2500\u2500 Visible mask = union of old and new \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        vis_mask = vis_old | vis_new
        visible_num = int(vis_mask.sum())

        # \u2500\u2500 Convergence diagnostic \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        cos_sim = np.sum(gradients * new_gradients, axis=1)
        mean_cos = float(np.mean(cos_sim))
        max_angle_deg = float(np.degrees(np.arccos(np.clip(np.min(cos_sim), -1, 1))))
        proj_pts = points - values[:, np.newaxis] * new_gradients
        proj_rmse = float(np.sqrt(np.mean(interpolator.predict(proj_pts) ** 2)))

        print(f"Iter {it+1:3d} | visible: {visible_num}/{N} ({100.0*visible_num/N:.2f}%)  "
              f"mean cos_sim: {mean_cos:.6f}  "
              f"max angle change: {max_angle_deg:.2f}\u00b0  "
              f"proj RMSE: {proj_rmse:.6e}")

        gradients = new_gradients
        init_angles = np.arctan2(gradients[:, 1], gradients[:, 0])

        # \u2500\u2500 Step 5: Refit interpolant on the visible subset \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        fit_mask = vis_mask & ~np.isnan(gradients).any(axis=1)
        interpolator.fit(points, values, gradients, mask=fit_mask)

    return gradients, interpolator

def clamp_gradients_to_arcs(points, values, gradients, degenerate_pts, batch, ngbrs_list, interpolator : Interpolator, tolerance : Tolerance):
    projections = points - values[:, np.newaxis] * gradients
    # sdf_tol = tolerance.clamp_sdf_tol
    ratio = tolerance.clamp_radius_ratio
    float_tol = tolerance.float_tol
    debug_cnt = 0
    clamp_cnt = 0
    for i in range(len(points)):
        if i in degenerate_pts:
            continue    # skip clamping for points with degenerate arcs, since we will set their gradients directly toward the angle later
        ngbrs = points[ngbrs_list[i]]
        dists = np.linalg.norm(projections[i] - ngbrs, axis=1)
        inside = dists < (np.abs(values[ngbrs_list[i]])) - float_tol
        if not np.any(inside):
            continue    # no neighbor contains the projection, skip clamping
        # 1. if it is close to some arc, clamp to the closest point on that arc
        closest, distances = util.query_closest_on_arcs(projections[i:i+1], i, batch)
        # if distances[0] < ratio * np.abs(values[i]):
        if distances[0] < tolerance.clamp_sdf_tol:
            pt = closest[0]
            gradients[i] = (points[i] - pt) / (values[i] + 1e-10)  # clamp to the closest point on the arc, with a slightly relaxed denominator to avoid over-shooting
            clamp_cnt += 1
            continue

        # # 2. otherwise, if some point on the arc has a low function value, clamp to that point
        # sample_pts = util.sample_arcs(i, batch, num_points=100)
        # if len(sample_pts) == 0:
        #     debug_cnt += 1
        #     continue
        # grads = interpolator.sample_best_gradients(points[i:i+1], values[i:i+1], given_samples=sample_pts[np.newaxis, :, :])  # given_samples shape: (1, num_points, 3)
        # proj = points[i] - values[i] * grads[0]
        # pred_sdf = interpolator.predict(proj[np.newaxis, :])
        # if -sdf_tol < pred_sdf < sdf_tol:
        #     gradients[i] = grads[0]
        #     continue
        # 3. if no suitable arc point found, keep the original gradient (which is outside visible arcs, filtered out later)
    if debug_cnt > 0:
        print(f"\n there are {debug_cnt} samples without any arcs\n")
    if clamp_cnt > 0:
        print(f"\n clamped {clamp_cnt} samples to their closest arcs\n")
    return clamp_cnt

def iterative_projection_3d(points, values, init_gradients, interpolator : Interpolator, options : Options, num_iter=10,
                         num_coarse=24, refine_steps=4, num_refine=5, gt_gradients=None):
    """
    Iteratively refine SDF gradients by projecting sample points onto the zero
    level set of an interpolant, then finding the best gradient
    direction via sample_best_gradients (coarse sweep + angular refinement).

    Algorithm (each iteration):
      1. Fit an Interpolator with current gradients (use_projection=True).
      2. For each sample point, search over directions to find the one whose
         projection P - s*g lands closest to the zero level set of the
         interpolant (via sample_best_gradients).
      3. Update gradients := best directions found.
      4. Repeat.

    Parameters
    ----------
    points : (N, 3) array
        Sample point coordinates.
    values : (N,) array
        Signed distance values at each sample point.
    init_gradients : (N, 3) array
        Initial unit gradient estimates.
    visible_arcs: a list of lists of tuples: point index -> list of visible arcs
        A collection of visible arcs that can be used to clamp the gradients.
    short_arc_idx: a set of point indices whose visible arcs are extremely short, so we skip them.
    num_iter : int
        Number of projection-refit iterations (default 10).
    num_coarse : int
        Number of uniformly spaced directions in the coarse sweep (default 24).
    refine_steps : int
        Number of zoom-in refinement iterations (default 4).
    num_refine : int
        Directions evaluated per refinement step (default 12).

    Returns
    -------
    gradients : (N, 2) array
        Refined unit gradient vectors.
    interpolator : CurlFree_Interpolator
        The final fitted interpolator (ready for predict / marching cubes).
    """
    points = np.asarray(points, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64).ravel()
    gradients = np.array(init_gradients, dtype=np.float64, copy=True)
    MES_used = False
    MES_used = True

    # interpolator.fit(points, values, gradients, force_recompute=False)
    if gt_gradients is not None:
            print(f"cos_sim mean to the ground truth gradients: {np.mean(np.sum(gt_gradients * gradients, axis=1)):.6f}")
    no_neighbor_mask = np.array([len(options.ngbrs_list[i]) == 0 for i in range(len(points))], dtype=bool)
    for it in range(num_iter):
        # ----- Step 1: Find best gradient via angular search on the interpolant -----
        new_gradients = interpolator.sample_best_gradients(
            points, values,
            num_coarse=num_coarse,
            refine_steps=refine_steps,
            num_refine=num_refine,
            initial_guess=gradients
        )
        if options.clamp:
            pre_clamp_gradients = new_gradients.copy()
            while True:
                new_gradients = pre_clamp_gradients.copy()
                clamped_cnt = clamp_gradients_to_arcs(points, values, new_gradients, options.degenerate_pts, options.batch, options.ngbrs_list, interpolator, options.tolerance)
                if clamped_cnt / len(points) > .3:
                    options.tolerance.clamp_sdf_tol /= 2
                else:
                    break

        # zero_level_contours = interpolator.extract_zero_level_set(bounds=((0, 1), (0, 1)), resolution=600)
        # _, nearest_pts = _point_to_polylines_min_dist(points, zero_level_contours)
        # dir = np.where(values[:, np.newaxis] < 0, nearest_pts - points, points - nearest_pts)
        # new_gradients = dir / np.linalg.norm(dir, axis=1, keepdims=True)

        # skip points with short visible arcs or the dir is not in visible arcs, keep original gradients for them
        projections_new = points - values[:, np.newaxis] * new_gradients
        projections_old = points - values[:, np.newaxis] * gradients

        visible_mask_new = are_points_visible(projections_new, points, values)
        visible_mask_old = are_points_visible(projections_old, points, values)
        # Don't update gradients to make visible gradients invisible
        skip_mask = visible_mask_old & ~visible_mask_new
        # Don't update gradients for short arc points
        for i in options.degenerate_pts:
            skip_mask[i] = True
        new_gradients[skip_mask] = gradients[skip_mask]

        # ----- Convergence diagnostic -----
        cos_sim = np.sum(gradients * new_gradients, axis=1)
        mean_cos = np.mean(cos_sim)
        max_angle_deg = np.degrees(np.arccos(np.clip(np.min(cos_sim), -1, 1)))

        print(f"Iter {it+1:3d} | mean cos_sim: {mean_cos:.6f}  "
              f"max angle change: {max_angle_deg:.2f}\u00b0  ")
        visible_mask = visible_mask_new | visible_mask_old
        # visible_mask = visible_mask & ~no_neighbor_mask # remove no neighbor sdf projections since they are not reliable for guiding interpolation
        visible_num = visible_mask.sum()
        print(f"Number of visible projected points: {visible_num} out of {len(points)}. Percentage: {visible_num / len(points) * 100:.2f}%")

        if not MES_used and (visible_num - visible_mask_old.sum())/len(points) < 0.01: # it is time to use MES points
            if mes_contact is None:
                raise ImportError("mes_contact module not available — build it via build_mes_contact.py (needs CGAL + GMP/MPFR)")
            print("========= Using MES points... =========")
            contact_pts, MES_normals = mes_contact.contact_points_from_sdf(points, values, debug_level=0)
            valid_mask = ~np.isnan(MES_normals).any(axis=1)
            new_gradients[valid_mask & ~visible_mask] = MES_normals[valid_mask & ~visible_mask]
            MES_used = True

        gradients = new_gradients
        # ----- Step 2: Fit interpolant with current gradients -----
        interpolator.fit(points, values, gradients, mask=visible_mask)
        if gt_gradients is not None:
            print(f"cos_sim mean to the ground truth gradients: {np.mean(np.sum(gt_gradients * new_gradients, axis=1)):.6f}")

    return gradients, interpolator


def build_two_step_macedo_matrices_with_projection(points, values, init_gradients, min_proj_distance=1e-8):
    """
    Improved two-step matrix construction using projected points to enhance gradient interpolation.
    Projected point position: P_proj = P - S * d
    
    Same strategy as CurlFree_Interpolator in interpolation.py:
    - Filter out projected points too close to existing points
    - Constrain gradients on all base points (original + valid projected)
    - Build a square system (2*N_cf+5, 2*N_cf+5) to avoid singular matrices
    
    Returns: A_grad, A_scalar, all_base_points, valid_mask
    """
    N, d = points.shape
    device, dtype = points.device, points.dtype
    
    # Compute projected points: P_proj = P - S * d
    projected_points = points - values.unsqueeze(1) * init_gradients
    
    # Filter projected points: keep only those far enough from all existing points
    valid_mask = torch.ones(N, dtype=torch.bool, device=device)
    for i in range(N):
        dists_to_orig = torch.norm(projected_points[i] - points, dim=1)
        if torch.min(dists_to_orig).item() < min_proj_distance:
            valid_mask[i] = False
            continue
        if torch.any(valid_mask[:i]):
            accepted_proj = projected_points[:i][valid_mask[:i]]
            dists_to_accepted = torch.norm(projected_points[i] - accepted_proj, dim=1)
            if dists_to_accepted.numel() > 0 and torch.min(dists_to_accepted).item() < min_proj_distance:
                valid_mask[i] = False
    
    valid_proj = projected_points[valid_mask]
    all_base_points = torch.cat([points, valid_proj], dim=0)
    N_cf = all_base_points.shape[0]
    # print(f"  Projection: {valid_mask.sum().item()}/{N} projected points accepted, "
    #       f"total CF base points: {N_cf}")
    
    # ==========================================
    # Step 1: Gradient interpolation matrix A_grad (Curl-free)
    # Based on Hessian of PHS phi(r) = r^4 log r
    # Constrain gradients on all N_cf base points -> square system (2*N_cf+5, 2*N_cf+5)
    # ==========================================
    delta = all_base_points.unsqueeze(1) - all_base_points.unsqueeze(0)
    r2 = torch.sum(delta**2, dim=-1)
    r = torch.sqrt(r2)
    log_r = torch.where(r < 1e-12, torch.zeros_like(r), torch.log(r))
    
    I_mat = torch.eye(d, device=device, dtype=dtype).view(1, 1, d, d)
    outer_product = torch.einsum('nid,nie->nide', delta, delta)
    
    H_blocks = r2.view(N_cf, N_cf, 1, 1) * (4 * log_r.view(N_cf, N_cf, 1, 1) + 1.0) * I_mat + \
               (8 * log_r.view(N_cf, N_cf, 1, 1) + 6.0) * outer_product
    
    A_grad_core = H_blocks.permute(0, 2, 1, 3).reshape(N_cf * d, N_cf * d)
    
    # Polynomial constraint matrix (2*N_cf, 5)
    P_grad = torch.zeros((2 * N_cf, 5), device=device, dtype=dtype)
    P_grad[0::2, 0] = 1.0
    P_grad[1::2, 1] = 1.0
    P_grad[0::2, 2] = all_base_points[:, 1]
    P_grad[1::2, 2] = all_base_points[:, 0]
    P_grad[0::2, 3] = all_base_points[:, 0]
    P_grad[1::2, 4] = all_base_points[:, 1]
    
    A_grad = torch.cat([
        torch.cat([A_grad_core, P_grad], dim=1),
        torch.cat([P_grad.T, torch.zeros((5, 5), device=device, dtype=dtype)], dim=1)
    ], dim=0)  # (2*N_cf+5, 2*N_cf+5) -- square matrix
    
    # ==========================================
    # Step 2: Residual interpolation matrix A_scalar (original N points only)
    # ==========================================
    delta_orig = points.unsqueeze(1) - points.unsqueeze(0)
    r2_orig = torch.sum(delta_orig**2, dim=-1)
    r_orig = torch.sqrt(r2_orig)
    log_r_orig = torch.where(r_orig < 1e-12, torch.zeros_like(r_orig), torch.log(r_orig))
    
    A_scal_core = r2_orig * log_r_orig
    
    E_scal = torch.cat([
        torch.ones((N, 1), device=device, dtype=dtype),
        points
    ], dim=1)
    A_scalar = torch.cat([
        torch.cat([A_scal_core, E_scal], dim=1),
        torch.cat([E_scal.T, torch.zeros((3, 3), device=device, dtype=dtype)], dim=1)
    ], dim=0)
    
    # Regularization
    A_grad += torch.eye(A_grad.shape[0], device=device, dtype=dtype) * 1e-10
    A_scalar += torch.eye(A_scalar.shape[0], device=device, dtype=dtype) * 1e-10
    
    return A_grad, A_scalar, all_base_points, valid_mask

def build_two_step_macedo_matrices(points):
    """
    Build two-step matrices strictly following the paper.
    Step 1 (gradient): Hessian of PHS kernel phi(r) = r^4 log(r) for gradient interpolation
    Step 2 (residual): TPS kernel phi(r) = r^2 log(r) with P_1 polynomial space (3 bases)
    """
    N, d = points.shape
    device, dtype = points.device, points.dtype
    
    delta = points.unsqueeze(1) - points.unsqueeze(0)
    r2 = torch.sum(delta**2, dim=-1)
    r = torch.sqrt(r2)
    
    # Safe log(r) computation to prevent NaN when r=0
    log_r = torch.where(r < 1e-12, torch.zeros_like(r), torch.log(r))
    
    # ==========================================
    # Step 1: Gradient interpolation matrix A_grad (Curl-free)
    # Based on Hessian of PHS phi(r) = r^4 log r:
    # H = r^2(4 log r + 1)I + (8 log r + 6)(delta @ delta^T)
    # ==========================================
    I = torch.eye(d, device=device, dtype=dtype).view(1, 1, d, d)
    outer_product = torch.einsum('nid,nie->nide', delta, delta)
    
    H_blocks = r2.view(N, N, 1, 1) * (4 * log_r.view(N, N, 1, 1) + 1.0) * I + \
               (8 * log_r.view(N, N, 1, 1) + 6.0) * outer_product
               
    A_grad_core = H_blocks.permute(0, 2, 1, 3).reshape(N * d, N * d)
    
    # Polynomial constraint matrix for gradients (2N, 5), corresponding to P_2 potential without constant term
    P_grad = torch.zeros((2 * N, 5), device=device, dtype=dtype)
    P_grad[0::2, 0] = 1.0         # constant term for px
    P_grad[1::2, 1] = 1.0         # constant term for py
    P_grad[0::2, 2] = points[:, 1]  # x-component of x*y
    P_grad[1::2, 2] = points[:, 0]  # y-component of x*y
    P_grad[0::2, 3] = points[:, 0]  # x^2 coefficient
    P_grad[1::2, 4] = points[:, 1]  # y^2 coefficient
    
    A_grad = torch.cat([
        torch.cat([A_grad_core, P_grad], dim=1),
        torch.cat([P_grad.T, torch.zeros((5, 5), device=device, dtype=dtype)], dim=1)
    ], dim=0)
    
    # ==========================================
    # Step 2: Residual interpolation matrix A_scalar
    # 2D TPS: phi = r^2 log r, P_1 polynomial space (3 bases)
    # ==========================================
    A_scal_core = r2 * log_r  # r^2 log r
    
    # Polynomial tail P_1(x, y) = a_0 + a_1*x + a_2*y
    E_scal = torch.cat([
        torch.ones((N, 1), device=device, dtype=dtype),
        points
    ], dim=1)
    A_scalar = torch.cat([
        torch.cat([A_scal_core, E_scal], dim=1),
        torch.cat([E_scal.T, torch.zeros((3, 3), device=device, dtype=dtype)], dim=1)
    ], dim=0)
    
    # Tiny perturbation to ensure safe LU factorization
    A_grad += torch.eye(A_grad.shape[0], device=device, dtype=dtype) * 1e-10
    A_scalar += torch.eye(A_scalar.shape[0], device=device, dtype=dtype) * 1e-10
    
    return A_grad, A_scalar

def _safe_r_logr(r2):
    """
    Safely compute r and log(r) from r^2, producing no inf/NaN in forward or backward pass.
    For r=0 positions, returns r=0, log_r=0 (kernel values are exactly 0 since
    grad_phi is proportional to delta=0, TPS is proportional to r^2*log_r -> 0*(-inf)=0).
    """
    eps = 1e-24
    r2_safe = r2 + eps                     # Ensure >0, finite sqrt gradient
    r = torch.sqrt(r2_safe)
    log_r = 0.5 * torch.log(r2_safe)       # log(sqrt(r2+eps))
    # Zero out log_r where r is truly ~0 (kernel is 0 on the diagonal anyway)
    mask = (r2 < 1e-20)
    log_r = log_r.masked_fill(mask, 0.0)
    r = r.masked_fill(mask, 0.0)
    return r, log_r

def evaluate_full_field(target_points, base_points, c_grad, p_grad, w_scal, p_scal):
    """
    Evaluate the total potential field f(x) = Phi_grad(x) + S_res(x)
    """
    N_base = base_points.shape[0]
    
    delta = target_points.unsqueeze(1) - base_points.unsqueeze(0)
    r2 = torch.sum(delta**2, dim=-1)
    r, log_r = _safe_r_logr(r2)
    
    # 1. Compute curl-free potential Phi_grad(x)
    grad_phi_blocks = r2.unsqueeze(-1) * (4 * log_r.unsqueeze(-1) + 1.0) * delta
    c_reshaped = c_grad.view(N_base, 2)
    Phi_rbf = torch.einsum('tbd,bd->t', grad_phi_blocks, c_reshaped)
    
    x = target_points[:, 0]
    y = target_points[:, 1]
    Phi_poly = p_grad[0]*x + p_grad[1]*y + p_grad[2]*x*y + 0.5*p_grad[3]*x**2 + 0.5*p_grad[4]*y**2
    Phi = Phi_rbf + Phi_poly
    
    # 2. Compute residual scalar field S_res(x)
    S_phi = r2 * log_r
    S_val = torch.matmul(S_phi, w_scal).squeeze()
    S_poly = p_scal[0] + p_scal[1]*x + p_scal[2]*y
    
    return Phi + (S_val + S_poly)

def evaluate_full_field_with_projection(target_points, base_points, c_grad, p_grad, w_scal, p_scal):
    """
    Evaluate the total potential field f(x) = Phi_grad(x) + S_res(x)
    
    base_points: gradient interpolation base points (original + valid projected)
    w_scal: residual interpolation coefficients, corresponding to the original N points
    """
    N_base = base_points.shape[0]
    N_orig = w_scal.shape[0]
    base_points_orig = base_points[:N_orig]
    
    # --- Gradient potential (using all base points) ---
    delta = target_points.unsqueeze(1) - base_points.unsqueeze(0)
    r2 = torch.sum(delta**2, dim=-1)
    r, log_r = _safe_r_logr(r2)
    
    grad_phi_blocks = r2.unsqueeze(-1) * (4 * log_r.unsqueeze(-1) + 1.0) * delta
    c_reshaped = c_grad.view(N_base, 2)
    Phi_rbf = torch.einsum('tbd,bd->t', grad_phi_blocks, c_reshaped)
    
    x = target_points[:, 0]
    y = target_points[:, 1]
    Phi_poly = p_grad[0]*x + p_grad[1]*y + p_grad[2]*x*y + 0.5*p_grad[3]*x**2 + 0.5*p_grad[4]*y**2
    Phi = Phi_rbf + Phi_poly
    
    # --- Residual scalar field (original base points only) ---
    delta_orig = target_points.unsqueeze(1) - base_points_orig.unsqueeze(0)
    r2_orig = torch.sum(delta_orig**2, dim=-1) # 2x2
    _, log_r_orig = _safe_r_logr(r2_orig)
    
    S_phi = r2_orig * log_r_orig
    S_val = torch.matmul(S_phi, w_scal).squeeze()
    S_poly = p_scal[0] + p_scal[1]*x + p_scal[2]*y
    
    return Phi + (S_val + S_poly)

def opt(points_np, values_np, init_grads_np, num_iter=500, lr=1e-2, rebuild_every=1, hard_eikonal=False,
        w_proj=1.0, w_smooth=1, w_init=0.01, w_eikonal=0.1, k_neighbors=6):
    """
    Optimize projected point positions to derive SDF gradients.
    
    Instead of optimizing gradient vectors directly, optimize the projected point
    positions P_proj (where P_proj = P - s*g). Gradients are then derived as:
      g = sign(s) * (P - P_proj) / |P - P_proj|
    
    This parameterization tends to produce smoother surfaces because neighboring
    projected points on the zero level set are directly regularized.
    
    Loss function design:
      1. Projection loss: f(P_proj)^2 -> 0
         Projected points should lie on the zero level set.
      2. Gradient smoothness loss (k-NN): sum w_ij (1 - g_i . g_j)^2
         Penalize inconsistent gradient directions derived from projected points.
      3. Initial position anchor: ||P_proj - P_proj_init||^2
         Prevent projected points from drifting far from initial estimate.
      4. Distance constraint: (|P - P_proj| - |s|)^2
         Projected distance should match the SDF value (eikonal consistency).
    
    Parameters:
        w_proj:      Projection loss weight (default 1.0)
        w_smooth:    Smoothness loss weight (default 1)
        w_init:      Initial position anchor weight (default 0.01)
        w_eikonal:   Distance constraint weight (default 0.1)
        k_neighbors: Number of k-NN neighbors for smoothness loss (default 6)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    print(f"Device: {device}, dtype: {dtype}")
    
    points = torch.tensor(points_np, dtype=dtype, device=device)
    values = torch.tensor(values_np, dtype=dtype, device=device).squeeze()
    N = points.shape[0]
    
    # Normalized initial gradients -> compute initial projected points
    init_grads_tensor = torch.tensor(init_grads_np, dtype=dtype, device=device)
    init_grads_tensor = init_grads_tensor / (torch.norm(init_grads_tensor, dim=1, keepdim=True) + 1e-12)
    init_proj_points = points - values.unsqueeze(1) * init_grads_tensor
    
    # sign(s) for gradient direction recovery; for s=0 use +1 as placeholder
    s_sign = torch.sign(values)
    s_sign[s_sign == 0] = 1.0
    s_sign = s_sign.unsqueeze(1)  # (N, 1)
    
    k = min(k_neighbors, N - 1)
    
    def rebuild_knn(p_proj_detached):
        """Rebuild k-NN graph based on projected point positions (on the zero level set)."""
        dists_matrix = torch.cdist(p_proj_detached, p_proj_detached)  # (N, N)
        _, idx = torch.topk(dists_matrix, k + 1, largest=False)
        idx = idx[:, 1:]  # Exclude self -> (N, k)
        # Inverse distance weights: closer neighbors have more influence
        dists = torch.gather(dists_matrix, 1, idx)  # (N, k)
        weights = 1.0 / (dists + 1e-10)
        weights = weights / weights.sum(dim=1, keepdim=True)  # Normalize
        return idx, weights
    
    # ==========================================
    # Optimization variable: projected point positions
    # ==========================================
    proj_points = init_proj_points.detach().clone().requires_grad_(True)
    opt_params = [proj_points]
    
    # Store initial projected positions for anchor loss
    init_proj_detached = init_proj_points.detach().clone()
    
    # Gentle decay weights: points closer to surface get more weight
    surface_weights = 1.0 / (1.0 + torch.abs(values))
    
    def derive_gradients(p_proj):
        """Derive unit gradient vectors from projected point positions.
        g = sign(s) * (P - P_proj) / |P - P_proj|"""
        direction = points - p_proj  # (N, 2), equals s * g
        dist = torch.norm(direction, dim=1, keepdim=True)  # |s| ideally
        grads_normed = s_sign * direction / (dist + 1e-12)
        return grads_normed, dist.squeeze()
    
    def rebuild_matrices(current_grads_detached):
        """Recompute projected points and factorize matrices using current (detached) gradients."""
        A_g, A_s, base_pts, v_mask = build_two_step_macedo_matrices_with_projection(
            points, values, current_grads_detached)
        LU_g, piv_g = torch.linalg.lu_factor(A_g)
        LU_s, piv_s = torch.linalg.lu_factor(A_s)
        return A_g, A_s, base_pts, v_mask, LU_g, piv_g, LU_s, piv_s
    
    # Initial construction
    print("Building two-step matrices with projection (initial)...")
    with torch.no_grad():
        cur_grads, _ = derive_gradients(proj_points)
        knn_idx, knn_weights = rebuild_knn(proj_points.detach())
    A_grad, A_scalar, all_base_points, valid_mask, LU_grad, pivots_grad, LU_scal, pivots_scal = \
        rebuild_matrices(cur_grads.detach())
    N_cf = all_base_points.shape[0]
    
    optimizer = torch.optim.Adam(opt_params, lr=lr)
    
    print(f"Starting optimization (rebuild every {rebuild_every} steps, k={k})...")
    print(f"Weights: proj={w_proj}, smooth={w_smooth}, init={w_init}, dist={w_eikonal}")
    
    for i in range(num_iter):
        # Recompute projected points and matrix factorization every rebuild_every steps
        if i > 0 and i % rebuild_every == 0:
            with torch.no_grad():
                cur_grads, _ = derive_gradients(proj_points)
                knn_idx, knn_weights = rebuild_knn(proj_points.detach())
            A_grad, A_scalar, all_base_points, valid_mask, LU_grad, pivots_grad, LU_scal, pivots_scal = \
                rebuild_matrices(cur_grads.detach())
            N_cf = all_base_points.shape[0]
        
        optimizer.zero_grad()
        
        # Derive gradients from current projected point positions
        grads_normalized, proj_dists = derive_gradients(proj_points)
        
        # =========================================
        # Loss 1: Distance constraint (|P - P_proj| - |s|)^2
        # Ensures the projected distance matches the SDF value (eikonal consistency)
        # =========================================
        loss_dist = torch.mean((proj_dists - torch.abs(values))**2)
        
        # Constrain gradients on all base points (projected point gradients = corresponding original gradients)
        all_grads = torch.cat([grads_normalized, grads_normalized[valid_mask]], dim=0)  # (N_cf, 2)
        
        # --- Two-step solve for interpolation coefficients ---
        # Step 1: Solve curl-free gradient coefficients
        y_grad = torch.cat([all_grads.view(-1, 1), torch.zeros((5, 1), device=device, dtype=dtype)], dim=0)
        coeffs_grad_all = torch.linalg.lu_solve(LU_grad, pivots_grad, y_grad)
        c_grad = coeffs_grad_all[:2*N_cf]
        p_grad = coeffs_grad_all[2*N_cf:]
        
        # Step 2: Compute residual and solve scalar coefficients
        Phi_at_points = evaluate_full_field_with_projection(
            points, all_base_points, c_grad, p_grad,
            torch.zeros(N, 1, device=device, dtype=dtype),
            torch.zeros(3, 1, device=device, dtype=dtype))
        residual = values - Phi_at_points
        
        y_scal = torch.cat([residual.unsqueeze(1), torch.zeros((3, 1), device=device, dtype=dtype)], dim=0)
        coeffs_scal_all = torch.linalg.lu_solve(LU_scal, pivots_scal, y_scal)
        w_scal = coeffs_scal_all[:N]
        p_scal = coeffs_scal_all[N:]
        
        # =========================================
        # Loss 2: Projection loss f(P_proj)^2 -> 0
        # Projected points should lie on the zero level set
        # =========================================
        projected_values = evaluate_full_field_with_projection(
            proj_points, all_base_points, c_grad, p_grad, w_scal, p_scal)
        loss_proj = torch.mean(surface_weights * projected_values**2)
        
        # =========================================
        # Loss 3: Laplacian smoothness on projected points
        # Each projected point should be close to the centroid of its neighbors
        # on the zero level set -> directly reduces jaggedness/sawteeth
        # =========================================
        neighbor_proj = proj_points[knn_idx]  # (N, k, 2)
        centroid = torch.sum(knn_weights.unsqueeze(-1) * neighbor_proj, dim=1)  # (N, 2)
        laplacian = proj_points - centroid  # deviation from local centroid
        loss_smooth = torch.mean(torch.sum(laplacian**2, dim=1))
        
        # =========================================
        # Loss 4: Initial position anchor (mild regularization)
        # Prevent projected points from drifting far from initial estimate
        # =========================================
        loss_init = torch.mean(torch.sum(
            (proj_points - init_proj_detached)**2, dim=1))
        
        # =========================================
        # Total loss
        # =========================================
        loss = (w_proj * loss_proj + w_smooth * loss_smooth +
                w_init * loss_init + w_eikonal * loss_dist)
        
        loss.backward()
        
        # NaN detection & gradient clipping
        if torch.isnan(proj_points.grad).any():
            print(f"  !! NaN gradient detected at step {i+1}, skipping update")
            proj_points.grad.zero_()
            continue
        torch.nn.utils.clip_grad_norm_([proj_points], max_norm=1.0)
        optimizer.step()
        
        if (i+1) % 100 == 0 or i == 0:
            print(f"Step {i+1:3d} | Proj: {loss_proj.item():.6e}  "
                  f"Smooth: {loss_smooth.item():.6e}  "
                  f"Init: {loss_init.item():.6e}  "
                  f"Dist: {loss_dist.item():.6e}  "
                  f"Total: {loss.item():.6e}")
    
    # Convert optimized projected points to unit gradients
    final_grads, _ = derive_gradients(proj_points)
    return final_grads.detach().cpu().numpy()