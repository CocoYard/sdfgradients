"""
Sphere Intersection Detection: Serial vs Parallel Performance Test

Given n spheres in 3D space, for each sphere find all other spheres it intersects with.
Two spheres intersect iff the distance between centers < sum of radii.
"""

import numpy as np
import time
from multiprocessing import Pool, shared_memory
from concurrent.futures import ProcessPoolExecutor
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist


# ═══════════════════════════════════════════════════════════════════
# Mesh SDF data generation (from user's code)
# ═══════════════════════════════════════════════════════════════════

def generate_test_mesh_data(path_to_mesh, grid_len=10):
    """
    Loads a mesh from the given path and computes signed distances
    and gradients on a regular grid around the mesh.

    Returns:
        mesh: trimesh.Trimesh
        points: (N, 3) array
        distances: (N,) array of signed distance values
        gradients: (N, 3) array
    """
    import trimesh
    import igl

    mesh = trimesh.load(path_to_mesh)

    # Normalize mesh to unit cube
    vmin = np.min(mesh.vertices, axis=0)
    vmax = np.max(mesh.vertices, axis=0)
    mesh.vertices -= (vmin + vmax) / 2
    mesh.vertices /= np.max(vmax - vmin)

    # Generate grid points around mesh bounding box
    bbox_min = np.min(mesh.vertices, axis=0) - 0.1
    bbox_max = np.max(mesh.vertices, axis=0) + 0.1
    x = np.linspace(bbox_min[0], bbox_max[0], grid_len)
    y = np.linspace(bbox_min[1], bbox_max[1], grid_len)
    z = np.linspace(bbox_min[2], bbox_max[2], grid_len)
    X, Y, Z = np.meshgrid(x, y, z)
    points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    # Compute distances to mesh
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int32)
    sq_dists, face_ids, closest = igl.point_mesh_squared_distance(points, V, F)
    distances = np.sqrt(sq_dists)

    gradients = points - closest
    norm_temp = np.linalg.norm(gradients, axis=1, keepdims=True)
    norm_temp[np.abs(norm_temp) <= 1e-8] = 1.0
    gradients /= norm_temp

    # Filter out points too close to surface
    mask = np.abs(distances) > 1e-8
    points = points[mask]
    distances = distances[mask]
    gradients = gradients[mask]

    # Use winding number for inside/outside
    W = igl.winding_number(V, F, points)
    inside = W > 0.5
    distances[inside] *= -1.0
    gradients[inside] *= -1.0

    return mesh, points, distances, gradients
# ---------------------------------------------------------------------------
# Shared data setup (for multiprocessing approaches)
# ---------------------------------------------------------------------------
_shm_centers = None
_shm_radii = None
_n = None


def _init_worker(shm_c_name, shm_r_name, shape_c, shape_r, n):
    """Attach worker to shared memory."""
    global _centers, _radii, _n, _shm_centers, _shm_radii
    _shm_centers = shared_memory.SharedMemory(name=shm_c_name)
    _shm_radii = shared_memory.SharedMemory(name=shm_r_name)
    _centers = np.ndarray(shape_c, dtype=np.float64, buffer=_shm_centers.buf)
    _radii = np.ndarray(shape_r, dtype=np.float64, buffer=_shm_radii.buf)
    _n = n


def _detect_one(i):
    """Worker function: find all spheres intersecting sphere i."""
    ci = _centers[i]
    ri = _radii[i]
    dists = np.linalg.norm(_centers - ci, axis=1)
    sum_r = _radii + ri
    mask = dists < sum_r
    mask[i] = False
    return i, np.where(mask)[0].tolist()


# ---------------------------------------------------------------------------
# Method 1: Serial (plain loop + numpy vectorized per-sphere)
# ---------------------------------------------------------------------------
def serial(centers, radii):
    n = len(centers)
    result = {}
    for i in range(n):
        dists = np.linalg.norm(centers - centers[i], axis=1)
        sum_r = radii + radii[i]
        mask = dists < sum_r
        mask[i] = False
        result[i] = np.where(mask)[0].tolist()
    return result

def serial2(centers, radii, block=15000):
    n = len(centers)
    result = {i: [] for i in range(n)}

    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        for j0 in range(0, n, block):
            j1 = min(j0 + block, n)

            dist = cdist(centers[i0:i1], centers[j0:j1])
            r_sum = radii[i0:i1, None] + radii[None, j0:j1]
            mask = dist < r_sum

            if i0 == j0:
                np.fill_diagonal(mask, False)

            rows, cols = np.where(mask)
            for r, c in zip(rows, cols):
                result[i0 + r].append(j0 + c)

    return result

# ---------------------------------------------------------------------------
# Method 2: multiprocessing.Pool + shared memory
# ---------------------------------------------------------------------------
def parallel_pool(centers, radii):
    n = len(centers)

    shm_c = shared_memory.SharedMemory(create=True, size=centers.nbytes)
    shm_r = shared_memory.SharedMemory(create=True, size=radii.nbytes)
    buf_c = np.ndarray(centers.shape, dtype=centers.dtype, buffer=shm_c.buf)
    buf_r = np.ndarray(radii.shape, dtype=radii.dtype, buffer=shm_r.buf)
    buf_c[:] = centers
    buf_r[:] = radii

    with Pool(
        initializer=_init_worker,
        initargs=(shm_c.name, shm_r.name, centers.shape, radii.shape, n),
    ) as pool:
        pairs = pool.map(_detect_one, range(n))

    shm_c.close(); shm_c.unlink()
    shm_r.close(); shm_r.unlink()

    return dict(pairs)


# ---------------------------------------------------------------------------
# Method 3: concurrent.futures.ProcessPoolExecutor + shared memory
# ---------------------------------------------------------------------------
def parallel_futures(centers, radii):
    n = len(centers)

    shm_c = shared_memory.SharedMemory(create=True, size=centers.nbytes)
    shm_r = shared_memory.SharedMemory(create=True, size=radii.nbytes)
    buf_c = np.ndarray(centers.shape, dtype=centers.dtype, buffer=shm_c.buf)
    buf_r = np.ndarray(radii.shape, dtype=radii.dtype, buffer=shm_r.buf)
    buf_c[:] = centers
    buf_r[:] = radii

    with ProcessPoolExecutor(
        initializer=_init_worker,
        initargs=(shm_c.name, shm_r.name, centers.shape, radii.shape, n),
    ) as executor:
        pairs = list(executor.map(_detect_one, range(n)))

    shm_c.close(); shm_c.unlink()
    shm_r.close(); shm_r.unlink()

    return dict(pairs)


# ---------------------------------------------------------------------------
# Method 4: joblib
# ---------------------------------------------------------------------------
def parallel_joblib(centers, radii):
    n = len(centers)

    def detect(i):
        dists = np.linalg.norm(centers - centers[i], axis=1)
        sum_r = radii + radii[i]
        mask = dists < sum_r
        mask[i] = False
        return np.where(mask)[0].tolist()

    results = Parallel(n_jobs=-1, prefer="processes")(
        delayed(detect)(i) for i in range(n)
    )
    return {i: r for i, r in enumerate(results)}


# ---------------------------------------------------------------------------
# Method 5: Fully vectorized numpy (no loop, no multiprocessing)
# ---------------------------------------------------------------------------
def fully_vectorized(centers, radii):
    # pairwise distance matrix  (n x n)
    diff = centers[:, None, :] - centers[None, :, :]   # (n, n, 3)
    dists = np.linalg.norm(diff, axis=2)                # (n, n)
    sum_r = radii[:, None] + radii[None, :]             # (n, n)
    mask = dists < sum_r
    np.fill_diagonal(mask, False)

    return {i: np.where(mask[i])[0].tolist() for i in range(len(centers))}

# ---------------------------------------------------------------------------
# Method 6: KD-Tree (using scipy.spatial.cKDTree)
# ---------------------------------------------------------------------------
def kd_tree_method(centers, radii):
    from scipy.spatial import cKDTree
    tree = cKDTree(centers)
    max_r = np.max(radii)
    
    pairs = np.array(list(tree.query_pairs(2 * max_r)))
    if len(pairs) == 0:
        return {i: [] for i in range(len(centers))}
    
    ii, jj = pairs[:, 0], pairs[:, 1]
    
    dists = np.linalg.norm(centers[ii] - centers[jj], axis=1)
    thresholds = radii[ii] + radii[jj]
    mask = dists < thresholds
    
    result = {i: set() for i in range(len(centers))}
    for i, j in pairs[mask]:
        result[i].add(j)
        result[j].add(i)
    return result


def chunked_vectorized(centers, radii, chunk_size=5000):
    from scipy.spatial.distance import cdist
    n = len(centers)
    result = {}
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        dists = cdist(centers[start:end], centers)
        sum_r = radii[start:end, None] + radii[None, :]
        mask = dists < sum_r
        for idx, i in enumerate(range(start, end)):
            mask[idx, i] = False
            result[i] = np.where(mask[idx])[0].tolist()
    return result

def _process_chunk(args):
    from scipy.spatial.distance import cdist
    start, end, centers, radii = args
    dists = cdist(centers[start:end], centers)
    sum_r = radii[start:end, None] + radii[None, :]
    mask = dists < sum_r
    result = {}
    for idx, i in enumerate(range(start, end)):
        mask[idx, i] = False
        result[i] = np.where(mask[idx])[0].tolist()
    return result

def chunked_parallel(centers, radii, chunk_size=2000):
    from concurrent.futures import ProcessPoolExecutor
    n = len(centers)
    chunks = [(s, min(s + chunk_size, n), centers, radii) 
              for s in range(0, n, chunk_size)]
    
    with ProcessPoolExecutor() as executor:
        chunk_results = executor.map(_process_chunk, chunks)
    
    result = {}
    for d in chunk_results:
        result.update(d)
    return result

def chunked_gemm_fast(centers, radii, chunk_size=5000):
    n = len(centers)
    sq_norms = np.sum(centers ** 2, axis=1)
    result = {i: [] for i in range(n)}

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = centers[start:end]

        dots = chunk @ centers.T
        dists_sq = sq_norms[start:end, None] + sq_norms[None, :] - 2 * dots
        sum_r_sq = (radii[start:end, None] + radii[None, :]) ** 2
        mask = dists_sq < sum_r_sq
        np.fill_diagonal(mask[:, start:end], False)

        rows, cols = np.nonzero(mask)
        rows += start

        if len(rows) > 0:
            split_points = np.searchsorted(rows, np.arange(start, end), side='right')
            split_points = np.concatenate([[0], split_points])
            for idx, i in enumerate(range(start, end)):
                result[i] = cols[split_points[idx]:split_points[idx+1]].tolist()

    return result

def sap(centers, radii, batch_size=10240):
    n = len(centers)
    if n < 2:
        return {}

    centers = np.asarray(centers, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)

    mins = centers[:, 0] - radii
    maxs = centers[:, 0] + radii
    order = np.argsort(mins)
    sorted_mins = mins[order]
    sorted_maxs = maxs[order]

    rights = np.searchsorted(sorted_mins, sorted_maxs, side='right')

    result = {}

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        ii, jj = [], []
        for i in range(start, end):
            count = rights[i] - i - 1
            if count > 0:
                js = np.arange(i + 1, rights[i])
                ii.append(np.full(count, i, dtype=np.intp))
                jj.append(js)

        if not ii:
            continue

        ii = np.concatenate(ii)
        jj = np.concatenate(jj)

        idx_i = order[ii]
        idx_j = order[jj]

        diff = centers[idx_i] - centers[idx_j]
        dist_sq = np.einsum('ij,ij->i', diff, diff)
        r_sum = radii[idx_i] + radii[idx_j]
        mask = dist_sq <= r_sum * r_sum

        for a, b in zip(idx_i[mask], idx_j[mask]):
            result.setdefault(int(a), []).append(int(b))
            result.setdefault(int(b), []).append(int(a))

    return result


# ═══════════════════════════════════════════════════════════════════
# NEW METHOD A: 3-Axis Sweep and Prune (Sort in x, y, z)
# ═══════════════════════════════════════════════════════════════════
# 
# Algorithm:
#   1. Sort spheres by x_min = cx - r.
#   2. Sweep: for each sphere i in sorted order, scan forward while
#      x_min[j] < x_max[i].  This gives x-axis overlap candidates.
#   3. Among candidates, filter by y-interval overlap, then z-interval.
#   4. Survivors get an exact squared-distance check.
#
# Complexity:
#   Sort: O(n log n)
#   Sweep: O(n * k_x) where k_x = avg x-axis candidates per sphere
#   With y/z filtering, effective work is O(n * k_xyz) where k_xyz << k_x
#   Best case ~O(n log n), worst case O(n^2) if all intervals overlap.
# ═══════════════════════════════════════════════════════════════════

def sweep_and_prune_3axis(centers, radii):
    """
    3-axis Sweep and Prune.

    Sort by x-min, sweep forward pruning by x-overlap, then filter
    by y-overlap and z-overlap before the exact distance check.
    """
    n = len(centers)
    if n < 2:
        return {i: [] for i in range(n)}

    centers = np.asarray(centers, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)

    # Precompute per-axis intervals: [center - r, center + r]
    lo = centers - radii[:, None]  # (n, 3)
    hi = centers + radii[:, None]  # (n, 3)

    # Sort by x-min
    order = np.argsort(lo[:, 0])
    sorted_x_lo = lo[order, 0]
    sorted_x_hi = hi[order, 0]

    # For y and z we still need the original-indexed lo/hi for filtering
    # but we'll index via order.

    result = {i: [] for i in range(n)}

    for si in range(n):
        idx_i = order[si]
        xi_hi = sorted_x_hi[si]
        yi_lo = lo[idx_i, 1];  yi_hi = hi[idx_i, 1]
        zi_lo = lo[idx_i, 2];  zi_hi = hi[idx_i, 2]
        ri = radii[idx_i]
        ci = centers[idx_i]

        # Scan forward: j = si+1, si+2, ... while x_min[j] < x_max[i]
        sj = si + 1
        while sj < n and sorted_x_lo[sj] < xi_hi:
            idx_j = order[sj]

            # Y-interval overlap check
            if lo[idx_j, 1] < yi_hi and hi[idx_j, 1] > yi_lo:
                # Z-interval overlap check
                if lo[idx_j, 2] < zi_hi and hi[idx_j, 2] > zi_lo:
                    # Exact distance check (squared to avoid sqrt)
                    dx = ci[0] - centers[idx_j, 0]
                    dy = ci[1] - centers[idx_j, 1]
                    dz = ci[2] - centers[idx_j, 2]
                    dist_sq = dx*dx + dy*dy + dz*dz
                    rsum = ri + radii[idx_j]
                    if dist_sq < rsum * rsum:
                        result[idx_i].append(idx_j)
                        result[idx_j].append(idx_i)
            sj += 1

    return result


# ═══════════════════════════════════════════════════════════════════
# NEW METHOD B: Uniform Grid with Mailboxes
# ═══════════════════════════════════════════════════════════════════
#
# Algorithm:
#   1. Choose cell_size = 2 * r_max  (guarantees intersecting spheres
#      are in the same cell or adjacent cells).
#   2. For each sphere, compute the range of cells it overlaps
#      (for varying radii a large sphere may span several cells)
#      and insert its index into each cell's list.
#   3. For each sphere i, iterate over its cell and 26 neighbors.
#      For every candidate j found there (j > i to avoid duplicates):
#        - Mailbox check: if we already tested pair (i, j), skip.
#        - Exact distance test.
#   4. The "mailbox" is a stamp per candidate so that if sphere j
#      appears in multiple neighbor cells we don't re-test it.
#
# Complexity:
#   Build grid: O(n) expected (each sphere touches O(1) cells if r ≈ r_max)
#   Query: O(n * 27 * m) where m = avg spheres per cell ≈ n / #cells
#   For uniform distributions: O(n) expected.
#   Worst case: O(n^2) if all spheres fall into one cell.
# ═══════════════════════════════════════════════════════════════════

def uniform_grid_mailbox(centers, radii):
    """
    Uniform spatial grid with mailbox duplicate elimination.

    cell_size = 2 * r_max.  Each sphere is inserted into every cell
    its bounding box overlaps.  Queries check 27 neighboring cells
    with a mailbox stamp to avoid redundant pair tests.
    """
    from collections import defaultdict

    n = len(centers)
    if n < 2:
        return {i: [] for i in range(n)}

    centers = np.asarray(centers, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)

    r_max = np.max(radii)
    cell_size = 2.0 * r_max
    if cell_size < 1e-12:
        return {i: [] for i in range(n)}
    inv_cell = 1.0 / cell_size

    # --- Build grid: insert each sphere into all cells its bbox overlaps ---
    lo = centers - radii[:, None]
    hi = centers + radii[:, None]

    # Cell indices for lo and hi corners
    cell_lo = np.floor(lo * inv_cell).astype(np.int64)  # (n, 3)
    cell_hi = np.floor(hi * inv_cell).astype(np.int64)  # (n, 3)

    grid = defaultdict(list)
    for i in range(n):
        for cx in range(cell_lo[i, 0], cell_hi[i, 0] + 1):
            for cy in range(cell_lo[i, 1], cell_hi[i, 1] + 1):
                for cz in range(cell_lo[i, 2], cell_hi[i, 2] + 1):
                    grid[(cx, cy, cz)].append(i)

    # --- Query: for each sphere, check its cell + 26 neighbors ---
    result = {i: [] for i in range(n)}
    # Mailbox: for each sphere i, track which j's we've already tested
    # We use a per-query set to avoid duplicate pair checks.
    # Since we process i < j, we only need to track per i.

    for i in range(n):
        ci = centers[i]
        ri = radii[i]

        # The cell of sphere i's center
        ccx = int(np.floor(ci[0] * inv_cell))
        ccy = int(np.floor(ci[1] * inv_cell))
        ccz = int(np.floor(ci[2] * inv_cell))

        mailbox = set()  # candidates already tested for this i

        # 27 neighbor cells (including self)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    cell_key = (ccx + dx, ccy + dy, ccz + dz)
                    cell_list = grid.get(cell_key)
                    if cell_list is None:
                        continue
                    for j in cell_list:
                        if j <= i:
                            continue
                        if j in mailbox:
                            continue
                        mailbox.add(j)
                        # Exact squared-distance check
                        ddx = ci[0] - centers[j, 0]
                        ddy = ci[1] - centers[j, 1]
                        ddz = ci[2] - centers[j, 2]
                        dist_sq = ddx*ddx + ddy*ddy + ddz*ddz
                        rsum = ri + radii[j]
                        if dist_sq < rsum * rsum:
                            result[i].append(j)
                            result[j].append(i)

    return result


# ═══════════════════════════════════════════════════════════════════
# NEW METHOD C: BVH (Bounding Volume Hierarchy)
# ═══════════════════════════════════════════════════════════════════
#
# Algorithm:
#   Build (top-down, median split):
#     1. Compute AABB of all sphere bounding boxes in the current set.
#     2. Pick the longest axis of that AABB.
#     3. Sort spheres by center coordinate along that axis.
#     4. Split at the median → left half, right half.
#     5. Recurse until a leaf contains ≤ leaf_size spheres.
#     6. Store the AABB at every internal node.
#
#   Query (for each sphere i):
#     1. Start at root.
#     2. If sphere i's bounding box does NOT intersect node's AABB → prune.
#     3. If leaf: exact distance check against all spheres in the leaf (skip self).
#     4. Else: recurse into both children.
#
#   All-pairs: for each sphere i, query the root. Use i < j to
#   record each pair only once.
#
# Complexity:
#   Build: O(n log n)  (sort at each level, log n levels)
#   Per query: O(log n) expected for well-separated distributions
#   Total: O(n log n) expected, O(n^2) worst case
# ═══════════════════════════════════════════════════════════════════

def bvh_method(centers, radii, leaf_size=16):
    """
    BVH with top-down median-split construction.

    Each leaf holds up to `leaf_size` spheres.
    Queries prune via AABB-vs-sphere overlap test.
    """
    n = len(centers)
    if n < 2:
        return {i: [] for i in range(n)}

    centers = np.asarray(centers, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)

    # Precompute per-sphere AABBs
    sphere_lo = centers - radii[:, None]  # (n, 3)
    sphere_hi = centers + radii[:, None]  # (n, 3)

    # ---- BVH node stored as flat arrays for speed ----
    # Each node: (aabb_lo[3], aabb_hi[3], left_child, right_child, leaf_start, leaf_count)
    # We'll use a list-of-tuples approach and recursive build.

    # Node representation:
    #   - Internal: children = (left, right), indices = None
    #   - Leaf:     children = None,          indices = list of sphere indices

    class BVHNode:
        __slots__ = ['lo', 'hi', 'left', 'right', 'indices']
        def __init__(self):
            self.lo = None
            self.hi = None
            self.left = None
            self.right = None
            self.indices = None

    def build(idx_list):
        node = BVHNode()
        # Compute AABB over all spheres in idx_list
        node.lo = np.min(sphere_lo[idx_list], axis=0)
        node.hi = np.max(sphere_hi[idx_list], axis=0)

        if len(idx_list) <= leaf_size:
            node.indices = idx_list
            return node

        # Split along longest axis
        extent = node.hi - node.lo
        axis = int(np.argmax(extent))
        # Sort by center along chosen axis
        axis_vals = centers[idx_list, axis]
        sorted_order = np.argsort(axis_vals)
        sorted_indices = [idx_list[k] for k in sorted_order]

        mid = len(sorted_indices) // 2
        node.left = build(sorted_indices[:mid])
        node.right = build(sorted_indices[mid:])
        return node

    idx_all = list(range(n))
    root = build(idx_all)

    # ---- Query ----
    result = {i: [] for i in range(n)}

    def query(node, i, ci, ri, si_lo, si_hi):
        """Find all j > i that intersect sphere i, starting from node."""
        # AABB vs sphere-AABB overlap test (separating axis on each dim)
        if (si_lo[0] > node.hi[0] or si_hi[0] < node.lo[0] or
            si_lo[1] > node.hi[1] or si_hi[1] < node.lo[1] or
            si_lo[2] > node.hi[2] or si_hi[2] < node.lo[2]):
            return  # no overlap → prune

        if node.indices is not None:
            # Leaf: check all spheres in this leaf
            for j in node.indices:
                if j <= i:
                    continue
                dx = ci[0] - centers[j, 0]
                dy = ci[1] - centers[j, 1]
                dz = ci[2] - centers[j, 2]
                dist_sq = dx*dx + dy*dy + dz*dz
                rsum = ri + radii[j]
                if dist_sq < rsum * rsum:
                    result[i].append(j)
                    result[j].append(i)
        else:
            query(node.left, i, ci, ri, si_lo, si_hi)
            query(node.right, i, ci, ri, si_lo, si_hi)

    for i in range(n):
        ci = centers[i]
        ri = radii[i]
        si_lo = sphere_lo[i]
        si_hi = sphere_hi[i]
        query(root, i, ci, ri, si_lo, si_hi)

    return result


# ---------------------------------------------------------------------------
# Main: benchmark all methods
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mesh, points, sdf_values, gradients = generate_test_mesh_data("examples/bunny.obj", grid_len=50)
    centers = points
    radii = np.abs(sdf_values)
    N = len(centers)
    radii_max = np.max(radii)

    # ── C++ wrappers: return list of numpy arrays (fast, no dict overhead) ──
    def _csr_to_list(csr_func, centers, radii, **kwargs):
        """Wrap a C++ CSR function → list[np.array]."""
        off, nbr = csr_func(centers, radii, **kwargs)
        off = np.asarray(off)
        nbr = np.asarray(nbr)
        return [nbr[off[i]:off[i+1]] for i in range(len(off) - 1)]

    try:
        import sphere_intersect as si
        cpp_available = True
    except ImportError:
        cpp_available = False
        print("WARNING: sphere_intersect C++ module not found, skipping C++ methods.")

    if not cpp_available:
        print("ERROR: sphere_intersect C++ module not found. Please compile it first.")
        exit(1)
    methods = [
        ("C++: Uniform Grid+Mailbox", lambda c, r: _csr_to_list(si.find_intersections_grid, c, r)),
        ("C++: 3-Axis SAP",           lambda c, r: _csr_to_list(si.find_intersections_sap, c, r)),
        ("C++: BVH (median split)",   lambda c, r: _csr_to_list(si.find_intersections_bvh, c, r)),
        ("C++: Multi-level Hash",     lambda c, r: _csr_to_list(si.find_intersections, c, r)),
    ]

    results = {}
    print(f"Sphere intersection detection benchmark  (n = {N}), radii in [0, {radii_max:.4f}])")
    print("=" * 60)

    for name, func in methods:
        t0 = time.perf_counter()
        res = func(centers, radii)
        elapsed = time.perf_counter() - t0
        lengths = [len(v) for v in res]
        total_pairs = sum(lengths) // 2
        max_neighbors = max(lengths) if lengths else 0
        median_neighbors = np.median(lengths) if lengths else 0
        print(f"{name:<35} {elapsed:>8.3f}s   ({total_pairs} pairs, max: {max_neighbors}, median: {median_neighbors:.1f})")
        results[name] = res

    # Verify correctness: use Uniform Grid as reference (simplest, least likely to have bugs)
    ref_name = "C++: Uniform Grid+Mailbox"
    ref = results[ref_name]
    print("\nCorrectness check (ref = C++: Uniform Grid+Mailbox):")
    for name, res in results.items():
        if name == ref_name:
            continue
        match = all(np.array_equal(np.sort(res[i]), np.sort(ref[i])) for i in range(N))
        status = "✓ PASS" if match else "✗ FAIL"
        print(f"  {status}  {name}")