from scipy.spatial import cKDTree
import gpytoolbox as gpy
import igl
from scipy.spatial.distance import cdist
import numpy as np
# import matplotlib
# matplotlib.use('Agg')  # Non-interactive backend, no window shown
import matplotlib.pyplot as plt
import visible_arcs as va
import iterative_projection as ip
from interpolation import Interpolator, CurlFree_Interpolator
from enum import Enum
import time
import optimization as opt
from util import print_shape_distances

class NeighborEstimation(Enum):
    VISIBLE_CONNECTIVITY = 'visible_connectivity'
    SPATIAL = 'spatial'

class GradientEstimation(Enum):
    CurlFree_OPT = 'curlfree_opt'
    INTERP_GLOBAL_OPT = 'interp_global_opt'
    INTERP_LOCAL = 'interp_local'
    ORACLE_CURLFREE = 'oracle_curlfree'
    IRLS = 'irls'
    RANSAC = 'ransac'
    FINITE = 'finite'
    LSTSQ = 'lstsq'

def generate_test_mesh_data( path_to_mesh, outbase, grid_len=10, save=False ):
    '''
    Loads a mesh from the given path and computes signed distances and gradients for its vertices.
    Parameters:
    path_to_mesh: str
        The file path to the mesh.
    Returns:
    points: (N, 3) array of vertex coordinates
        The vertices of the mesh.
    distances: (N,) array of signed distance values
        The signed distance values for each vertex.
    gradients: (N, 3) array of gradient vectors
        The gradient vectors at each vertex.
    '''
    import trimesh

    # Load the mesh
    mesh = trimesh.load(path_to_mesh)
    # Normalize the mesh to fit within a unit cube
    min = np.min( mesh.vertices, axis=0 )
    max = np.max( mesh.vertices, axis=0 )
    mesh.vertices -= (min + max) / 2
    mesh.vertices /= np.max( max - min )

    # # Generate random points in 3D space
    # radius = 2*np.max( np.linalg.norm( mesh.vertices, axis=1 ) )
    # points = np.random.uniform(-radius, radius, (num_points, 3))

    # Generate equally spaced points around the mesh bounding box
    bbox_min = np.min(mesh.vertices, axis=0) - 0.1
    bbox_max = np.max(mesh.vertices, axis=0) + 0.1
    x = np.linspace(bbox_min[0], bbox_max[0], grid_len)
    y = np.linspace(bbox_min[1], bbox_max[1], grid_len)
    z = np.linspace(bbox_min[2], bbox_max[2], grid_len)
    X, Y, Z = np.meshgrid(x, y, z)
    points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    # Find the closest points on the mesh surface
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int32)
    sq_dists, face_ids, closest = igl.point_mesh_squared_distance(points, V, F)
    distances = np.sqrt(sq_dists)

    gradients = points - closest
    # Normalize gradients
    norm_temp = np.linalg.norm(gradients, axis=1, keepdims=True)
    # Avoid division by zero
    norm_temp[np.abs(norm_temp) <= 1e-8] = 1.0
    gradients /= norm_temp

    # Filter out points that are too close to the surface (within 0.1 units), also remove respective gradients
    mask = np.abs(distances) > 1e-8
    points = points[mask]
    distances = distances[mask]
    gradients = gradients[mask]

    # Use winding number to determine inside/outside
    W = igl.winding_number(V, F, points)
    mask = W > 0.5  # Points with winding number > 0.5 are inside
    distances[mask] *= -1.0  # Invert distances for points inside the mesh
    gradients[mask] *= -1.0  # Invert gradients for points inside the mesh

    # save to file for reuse
    if save:
        np.savez("out/" + outbase + "_sdf_" + str(grid_len**3) + ".npz",
                 points=points,
                 sdf_values=distances,
                 gradients=gradients)
        print("✅ Saved SDF data:", "out/" + outbase + "_sdf_" + str(grid_len**3) + ".npz")
    return mesh, points, distances, gradients

def test_mesh(grid_len=20, path_to_sdf=None, path_to_obj=None, save_npz=False):
    """
    Test function to demonstrate the process of loading SDF data, fitting an interpolator, and visualizing the results by Marching Cubes.
    e.g. path_to_sdf='out/bunny_sdf_1000.npz', path_to_obj='examples/bunny.obj')
    """
    if path_to_sdf is not None:
        # read sdf data from file
        data = np.load(path_to_sdf)
        points = data['points']
        distances = data['sdf_values']
    else:
        base_name = path_to_obj.split('/')[-1].split('.')[0]
        mesh, points, distances, _ = generate_test_mesh_data(path_to_obj, base_name, grid_len=grid_len, save=save_npz)  # Generate new data with 4096 points

    # Create and fit the interpolator
    interpolator = Interpolator(kernel='cubic')
    timer = time.perf_counter()
    interpolator.fit(points, distances)
    print(f"  ⏱  {'Interpolator fitted':<30} {time.perf_counter() - timer:>7.2f} s")

    # visualize results using marching cubes to extract isosurface
    timer = time.perf_counter()
    verts, faces = interpolator.extract_zero_level_set(bounds=((points[:, 0].min(), points[:, 0].max()),
                                                (points[:, 1].min(), points[:, 1].max()),
                                                (points[:, 2].min(), points[:, 2].max())),
                                                resolution=100)
    print(f"  ⏱  {'Grid evaluation':<30} {time.perf_counter() - timer:>7.2f} s")
    # Extract isosurface at value 0 using marching cubes

    # --- Second window: marching cubes directly on sample points (原始网格点) ---
    # 从点坐标反推网格结构，无需插值
    xs = np.unique(np.round(points[:, 0], 8))
    ys = np.unique(np.round(points[:, 1], 8))
    zs = np.unique(np.round(points[:, 2], 8))
    nx, ny, nz = len(xs), len(ys), len(zs)
    ix = np.searchsorted(xs, np.round(points[:, 0], 8))
    iy = np.searchsorted(ys, np.round(points[:, 1], 8))
    iz = np.searchsorted(zs, np.round(points[:, 2], 8))
    grid_values_direct = np.ones((nx, ny, nz))  # 缺失点默认为外部(+1)
    grid_values_direct[ix, iy, iz] = distances
    sp = ((xs[-1]-xs[0])/(nx-1), (ys[-1]-ys[0])/(ny-1), (zs[-1]-zs[0])/(nz-1))
    from skimage.measure import marching_cubes
    verts2, faces2, _, _ = marching_cubes(grid_values_direct, level=0.0, spacing=sp)
    verts2 += np.array([xs[0], ys[0], zs[0]])
    # Export meshes to out/
    import trimesh, os
    out_dir = 'out/' + path_to_obj.split('/')[-1].split('.')[0]
    os.makedirs(out_dir, exist_ok=True)
    trimesh.Trimesh(vertices=verts, faces=faces).export(f'{out_dir}/interpolant_{grid_len}.obj')
    trimesh.Trimesh(vertices=verts2, faces=faces2).export(f'{out_dir}/sample_points_{grid_len}.obj')
    print(f"Exported: {out_dir}/interpolant_{grid_len}.obj, {out_dir}/sample_points_{grid_len}.obj")
    return plt

import itertools

def estimate_gradient_exhaustive(points, sdf_values, ind, neighbors=None, verbose=False):
    """
    Brute-force all pairs of neighbors to find the gradient that minimizes 
    the Median Squared Residual (LMS). 
    
    The returned weights are binary: 1.0 for the two points that formed the 
    best gradient, and 0.0 for everyone else.
    """
    # 1. Setup Data
    point = points[ind]
    point_sdf = sdf_values[ind]

    if neighbors is None:
        nbrs = NearestNeighbors(n_neighbors=16, algorithm='auto').fit(points)
        _, indices = nbrs.kneighbors(point.reshape(1, -1))
        indices = indices[0]
    else:
        indices = np.array(neighbors, dtype=int)
        if indices.ndim > 1:
            indices = indices.flatten()
    indices = indices[indices != ind]  # Exclude the center point itself if present
    # Local coordinate system relative to the center point
    # We enforce the plane to pass through (0, 0) in this local space
    A_local = points[indices] - point
    b_local = sdf_values[indices] - point_sdf
    
    n_neighbors = len(indices)
    dim = points.shape[1] # 2 for 2D, 3 for 3D

    # Initialize output containers
    if n_neighbors == 1:
        # Only one neighbor, we can only form a line, not a plane. 
        # The best we can do is to take the direction to that neighbor.
        grad = A_local[0] / np.linalg.norm(A_local[0])
        weights = np.ones(n_neighbors)
        if sdf_values[indices[0]] < point_sdf:
            grad *= -1.0  # Flip direction if neighbor is inside
        error = np.abs((A_local[0] @ grad - b_local[0]))
        return grad, weights, indices, error
    # If not enough neighbors to form a pair, return zero gradient
    if n_neighbors < dim:
        return np.zeros(dim), np.zeros(n_neighbors), indices, 0.0
    best_loss = float('inf')
    best_gradient = np.zeros(dim)
    best_subset_idx = [] # To store the indices of the "winning" pair
    if verbose:
        print(f"Point {ind}: Evaluating {n_neighbors} neighbors\n")
    # 2. Iterate through all combinations of 'dim' neighbors (Pairs in 2D)
    # subset_idx contains the local indices (0 to n_neighbors-1) of the chosen pair
    for subset_idx in itertools.combinations(range(n_neighbors), dim):
        # Extract the subset of points
        A_sub = A_local[list(subset_idx)]
        b_sub = b_local[list(subset_idx)]
        if verbose:
            print(f"  A_sub: \n{A_sub}\n  b_sub: {b_sub}")

        if dim == 2:
            # Check for collinearity (determinant close to zero)
            det = A_sub[0,0]*A_sub[1,1] - A_sub[0,1]*A_sub[1,0]
            if abs(det) < 1e-6:
                # Collinear: fall back to 1D gradient along the available direction
                dir_norm = np.linalg.norm(A_sub[0])
                if dir_norm < 1e-10:
                    dir_norm = np.linalg.norm(A_sub[1])
                    if dir_norm < 1e-10:
                        continue  # Both vectors are zero (overlapping points)
                    direction = A_sub[1] / dir_norm
                else:
                    direction = A_sub[0] / dir_norm
                projections = A_sub @ direction
                slope = np.dot(projections, b_sub) / np.dot(projections, projections)
                cand_grad = slope * direction
            else:
                try:
                    cand_grad = np.linalg.solve(A_sub, b_sub)
                    if verbose:
                        print(f"    Solved gradient: {cand_grad}")
                except np.linalg.LinAlgError:
                    continue
        # Check Gradient Validity (SDF property: norm should be approx 1)
        # We allow a loose tolerance to accept imperfect but reasonable gradients
        norm = np.linalg.norm(cand_grad)
        if not (0.5 < norm < 1.5):
            if verbose:
                print(f"    Rejected gradient due to norm {norm:.4f} (not in [0.5, 1.5])")
            continue
        cand_grad /= norm  # Normalize candidate gradient to unit length for fair comparison
        # 3. Validation: Evaluate this candidate gradient against ALL neighbors
        preds = A_sub @ cand_grad
        residuals = np.abs(preds - b_sub)
        
        # LMS Metric: Use Median of Residuals to be robust against 50% outliers
        loss = np.mean(residuals)  # Mean of residuals
        # loss = abs(1 - norm)

        # Alternative Metric: Mean of Squared Residuals (L2) for better sensitivity to all neighbors
        # loss = np.mean(residuals**2)
        if verbose:
            print(f"Loss: {loss:.4f}, Gradient Norm: {norm:.4f}, Residuals: {residuals}")

        if loss < best_loss:
            best_loss = loss
            best_gradient = cand_grad
            best_subset_idx = subset_idx
            
    # 4. Construct Final Weights
    # As requested, we assign weight 1.0 ONLY to the selected pair of points
    weights = np.zeros(n_neighbors)
    if len(best_subset_idx) > 0:
        weights[list(best_subset_idx)] = 1.0
    weights[indices == ind]=1  # Debug: Check if center point got weight 1.0 (it shouldn't)
    # Normalize Gradient vector
    final_norm = np.linalg.norm(best_gradient)
    if final_norm > 1e-8:
        best_gradient /= final_norm
    else:
        best_gradient = np.zeros(dim)
    return best_gradient, weights, indices, best_loss

def estimate_gradients_lstsq(points, sdf_values, neighbors=None):
    """ estimation of gradients from SDF values at given points. LSTSQ is the same as 
    to Prewitt finite difference (equal weights) if neighbors are the 8-connected grid 
    neighbors."""
    indices = neighbors
    gradients = np.zeros_like(points)
    errors = np.zeros(points.shape[0])
    for i in range(points.shape[0]):
        neighbor_points = points[indices[i]]
        neighbor_sdf = sdf_values[indices[i]]
        diffs = neighbor_points - points[i]
        sdf_diffs = neighbor_sdf - sdf_values[i]
        A = diffs
        b = sdf_diffs
        # Check if A is rank-deficient (collinear neighbors)
        if A.shape[0] >= 2 and np.linalg.matrix_rank(A, tol=1e-8) < 2:
            # Fall back to 1D gradient along the available direction
            direction = diffs[0] / np.linalg.norm(diffs[0])
            projections = diffs @ direction
            slope = np.linalg.lstsq(projections.reshape(-1, 1), sdf_diffs, rcond=None)[0][0]
            grad = slope * direction
        else:
            grad, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        gradients[i] = grad
        residuals = A @ grad - b
        errors[i] = np.sum(residuals**2) / max(len(b), 1)  # Average error per neighbor
    # Normalize gradients
    norms = np.linalg.norm(gradients, axis=1, keepdims=True)
    gradients /= np.maximum(norms, 1e-8)
    return gradients, errors

def estimate_gradients_finite_diff(points, sdf_values):
    """ estimation of gradients from SDF values at given points. This finite difference 
    uses central difference with Sobel operator (smaller weights on diagonals). The neighbors 
    are the 8-connected grid neighbors. For border points, use forward/backward difference.
    """
    n = int(points.shape[0]**0.5)
    from scipy.ndimage import sobel
    # 1. Reshape to grid
    sdf_grid = sdf_values.reshape(n, n)
    
    # 2. Apply Sobel on each axis
    # sdf_grid shape: (n_x, n_y), where axis=0 is x, axis=1 is y
    # sobel(axis=0) differentiates along x, sobel(axis=1) differentiates along y
    # Assume uniform grid spacing
    dx = points[1, 1] - points[0, 1]
    grad_x_grid = sobel(sdf_grid, axis=0, mode='nearest') / (8 * dx)
    grad_y_grid = sobel(sdf_grid, axis=1, mode='nearest') / (8 * dx)

    # Treat the boarder points with forward/backward difference
    grad_x_grid[0, :] = (sdf_grid[1, :] - sdf_grid[0, :]) / dx
    grad_x_grid[-1, :] = (sdf_grid[-1, :] - sdf_grid[-2, :]) / dx
    grad_y_grid[:, 0] = (sdf_grid[:, 1] - sdf_grid[:, 0]) / dx
    grad_y_grid[:, -1] = (sdf_grid[:, -1] - sdf_grid[:, -2]) / dx
    
    # 3. Flatten back to (N, 2)
    grad_x = grad_x_grid.flatten()
    grad_y = grad_y_grid.flatten()
    
    # Combine
    gradients = np.stack([grad_x, grad_y], axis=1) # (N, 2)

    # Compute error metric for gradient estimation. For SDF, the error for each neighbor
    # is the squared absolute difference between the SDF value and the dot product of the gradient with the neighbor vector.
    # Return the sum of the errors for all neighbors for each point.
    N = points.shape[0]
    errors = np.zeros(N)
    grad_grid = gradients.reshape(n, n, 2)
    for ix in range(n):
        for iy in range(n):
            total_err = 0.0
            count = 0
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = ix + di, iy + dj
                    if 0 <= ni < n and 0 <= nj < n:
                        idx_center = ix * n + iy
                        idx_neighbor = ni * n + nj
                        diff = points[idx_neighbor] - points[idx_center]
                        sdf_diff = sdf_values[idx_neighbor] - sdf_values[idx_center]
                        predicted = gradients[idx_center] @ diff
                        total_err += (sdf_diff - predicted)**2
                        count += 1
            errors[ix * n + iy] = total_err / max(count, 1)
    # Normalize gradients
    norms = np.linalg.norm(gradients, axis=1, keepdims=True)
    gradients /= np.maximum(norms, 1e-8)
    return gradients, errors

def estimate_gradients_irls(points, sdf_values, neighbor_list=None, interpolator: Interpolator | None = None, iters=10, sigma=0.1):
    """
    Estimate gradients using IRLS, calling estimate_gradient_irls_single per point.
    
    Args:
        points: (N, 2) coordinates.
        sdf_values: (N,) SDF values.
        neighbor_list: List of lists (your irregular indices).
        iters: IRLS iterations.
        sigma: Gaussian falloff parameter.
    Returns:
        gradients: (N, 2) unit gradients.
        errors: (N,) mean absolute residual per point.
    """
    n = points.shape[0]
    dim = points.shape[1]
    errors = np.zeros(n)
    if interpolator is not None:
        print("Initializing gradients with interpolator's best guess...")
        gradients = np.zeros((n, dim))
        if not interpolator.trained:
            interpolator.fit(points, sdf_values)
        for i in range(n):
            gradients[i] = interpolator.sample_best_gradient(points[i], sdf_values[i], num_samples=50)
    else:
        gradients, _ = estimate_gradients_RANSAC(points, sdf_values, neighbors=neighbor_list)
    for _ in range(iters):
        for i in range(n):
            nbr = neighbor_list[i] if neighbor_list is not None else None
            grad_i, weights_i, indices_i, error_i = estimate_gradient_irls_single(
                points, sdf_values, i, neighbors=nbr, last_gradients=gradients, iters=iters, sigma=sigma)
            gradients[i] = grad_i
            errors[i] = error_i
    return gradients, errors

def estimate_gradient_irls_single(points, sdf_values, ind, neighbors, last_gradients, iters=10, sigma=0.1, verbose=False):
    """
    Estimation of gradients from SDF values at a given point using IRLS.
    Only for a single point at index `ind`.
    
    Args:
        points: (N, D) array of points.
        sdf_values: (N,) array of SDF values.
        ind: Index of the point to estimate.
        neighbors: Optional list of neighbor indices. If None, KNN is used.
        last_gradients: (N, D) array of gradients from the previous iteration, used for weighting.
        iters: Number of IRLS iterations.
        sigma: Gaussian bandwidth for weighting.
        
    Returns:
        gradient: (D,) Estimated gradient vector.
        weights: (K,) Final weights of the neighbors.
        indices: (K,) Indices of the neighbors used.
        error: Mean absolute residual error for the point.
    """
    point = points[ind]
    point_sdf = sdf_values[ind]

    # 1. Handle Neighbors Input
    if neighbors is None:
        # If no neighbors provided, find k-nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=9, algorithm='auto').fit(points)
        _, indices = nbrs.kneighbors(point.reshape(1, -1))
        indices = indices[0] # Flatten to 1D array
    else:
        # Ensure indices is a numpy array
        indices = np.array(neighbors, dtype=int)
        # Handle case where neighbors might be passed as a list of lists or similar
        if indices.ndim > 1:
            indices = indices.flatten()

    # 2. Prepare Local Coordinate System
    # Calculate differences relative to the center point
    neighbor_points = points[indices]
    neighbor_sdfs = sdf_values[indices]

    # A: (K, D) -> vectors from center to neighbors
    A = neighbor_points - point  
    # b: (K,) -> SDF difference from center to neighbors
    b = neighbor_sdfs - point_sdf 

    # Note: We do NOT append np.ones here. 
    # By solving A @ gradient = b, we implicitly enforce the plane to pass 
    # exactly through (0,0) in local space, which is (point, point_sdf) in global space.
    
    # 3. Initialize Weights
    weights = np.ones(len(indices))
    gradient_similarity = last_gradients[indices] @ last_gradients[ind]

    # 4. Iterative Re-weighted Least Squares (IRLS)
    for i in range(iters):
        # Gaussian weighting: closer gradients get higher weights
        # Note: We add a small epsilon to sigma to prevent division by zero if sigma is too small
        weights = np.exp((gradient_similarity - 1) / (sigma**2))
        
        # Prepare Weighted System
        # To minimize sum(w_i * r_i^2), we multiply A and b by sqrt(w_i)
        sqrt_w = np.sqrt(weights)[:, np.newaxis] # Shape (K, 1)
        
        Aw = A * sqrt_w
        bw = b * sqrt_w.flatten()
        
        # Solve Weighted Least Squares
        gradient, _, _, _ = np.linalg.lstsq(Aw, bw, rcond=None)

    # 5. Normalization (SDF property: norm(gradient) should be 1)
    norm = np.linalg.norm(gradient)
    if norm > 1e-8:
        gradient /= norm
    else:
        # Fallback if gradient is zero (rare, usually means flat region or singular)
        gradient = np.zeros_like(gradient) 
    if verbose:
        print(f"A: \n{A}\nb: {b}\nWeights: {weights}\nGradient Similarity: {gradient_similarity}\nFinal Gradient: {gradient}")
    # compute error as weighted mean absolute residual
    residuals = A @ gradient - b
    error = np.sum(weights * np.abs(residuals)) / (np.sum(weights) + 1e-8)
    return gradient, weights, indices, error

import numpy as np
from scipy.optimize import linprog
from sklearn.neighbors import NearestNeighbors

def compute_weights_from_residuals(residuals, sigma=0.05):
    """
    Helper: Compute Gaussian weights from residuals.
    Points perfectly fitting the gradient get weight 1.0.
    """
    return np.exp(-(residuals**2) / (2 * (sigma**2)))
def estimate_gradient_l1_direct(points, sdf_values, ind, neighbors=None, sigma=0.05):
    """
    Directly solve L1 minimization: min sum(|Ax - b|) using Linear Programming.
    Returns optimal gradient, inferred weights, and indices.
    """
    # 1. Setup Data
    point = points[ind]
    point_sdf = sdf_values[ind]

    if neighbors is None:
        nbrs = NearestNeighbors(n_neighbors=40, algorithm='auto').fit(points)
        _, indices = nbrs.kneighbors(point.reshape(1, -1))
        indices = indices[0]
    else:
        indices = np.array(neighbors).flatten()
    
    # Local system: A * grad = b
    # Enforces passing through the center point (0,0) in local space
    A_local = points[indices] - point
    b_local = sdf_values[indices] - point_sdf
    
    n_neighbors, dim = A_local.shape
    
    # 2. Setup Linear Programming for L1 Minimization
    # Objective: min sum(u_i)
    # Variables z = [g_x, g_y, u_1, ..., u_n] (Size: dim + n_neighbors)
    # Constraints: -u_i <= A_i*g - b_i <= u_i
    
    # Objective function vector c: [0, 0, 1, 1, ..., 1]
    c = np.concatenate([np.zeros(dim), np.ones(n_neighbors)])
    
    # Inequality Matrix A_ub * z <= b_ub
    eye = np.eye(n_neighbors)
    # Constraint 1:  A*g - u <= b  -> [ A, -I] * z <= b
    top_A = np.hstack([A_local, -eye])
    # Constraint 2: -A*g - u <= -b -> [-A, -I] * z <= -b
    bot_A = np.hstack([-A_local, -eye])
    
    A_ub = np.vstack([top_A, bot_A])
    b_ub = np.concatenate([b_local, -b_local])
    
    # Solve (Unbounded gradient, Positive slack variables u)
    bounds = [(None, None)] * dim + [(0, None)] * n_neighbors
    
    # Using 'highs' method which is fast and robust
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        gradient = res.x[:dim]
    else:
        # Fallback to simple Least Squares if LP fails
        gradient, _, _, _ = np.linalg.lstsq(A_local, b_local, rcond=None)

    # 3. Post-processing: Calculate Weights based on the optimal gradient
    preds = A_local @ gradient
    residuals = np.abs(preds - b_local)
    weights = compute_weights_from_residuals(residuals, sigma)
    
    # Normalize Gradient
    norm = np.linalg.norm(gradient)
    gradient = gradient / max(norm, 1e-8)
    
    return gradient, weights, indices

def rate_gradient_estimation(sdf_points, points_on_surface, sdf_values, tol=1e-3):
    """ rate the gradient estimation by testing projected points enclosed by other points' circles."""
    # Vectorized version: compute all pairwise distances at once
    # Shape: (num_points, num_points)
    dists = np.linalg.norm(points_on_surface[:, np.newaxis, :] - sdf_points[np.newaxis, :, :], axis=2)
    
    # Broadcast radius to match distance matrix shape
    radii = np.abs(sdf_values)[np.newaxis, :]  # Shape: (1, num_points)
    
    # Check which points are inside other points' circles
    inside = dists < (radii - tol)  # Shape: (num_points, num_points)
    
    # Exclude self-comparison (diagonal elements)
    np.fill_diagonal(inside, False)
    
    # Count how many circles each point falls into
    wrong_count = np.sum(inside, axis=1)
    
    return wrong_count

def plot_correspondence(sdf_points, points_on_surface, plt, color='k'):
    for i in range(sdf_points.shape[0]):
        plt.plot([sdf_points[i, 0], points_on_surface[i, 0]], 
                 [sdf_points[i, 1], points_on_surface[i, 1]], color+'--', linewidth=0.5)

def setup_gradient_click_inspector(fig, ax, sdf_points, sdf_values, gradients, neighbors=None, gradient_estimation=GradientEstimation.RANSAC):
    """
    Add interactive click handler to inspect per-point gradient estimation.
    Click any point to show its neighbors, weights, gradient, and projected surface point.
    Supports RANSAC and IRLS methods.
    
    Parameters:
    -----------
    fig : matplotlib Figure
    ax : matplotlib Axes
    sdf_points : (N, 2) array
    sdf_values : (N,) array
    neighbors : dict or list, optional — neighbor indices per point
    gradient_estimation : GradientEstimation — which method to use for per-point inspection
    """
    highlighted = {'point': None, 'surface': None, 'neighbors': None, 'texts': [], 'line': None, 'lines': []}
    
    def on_click(event):
        if event.inaxes != ax:
            return
        # Clear previous highlights
        if highlighted['point'] is not None:
            highlighted['point'].remove()
        if highlighted['surface'] is not None:
            highlighted['surface'].remove()
        if highlighted['neighbors'] is not None:
            highlighted['neighbors'].remove()
        if highlighted['line'] is not None:
            highlighted['line'].remove()
        for item in highlighted['texts']:
            item.remove()
        for item in highlighted['lines']:
            item.remove()
        highlighted['texts'] = []
        highlighted['lines'] = []
        
        click_pt = np.array([event.xdata, event.ydata])
        distances_to_click = np.linalg.norm(sdf_points - click_pt, axis=1)
        idx = np.argmin(distances_to_click)
        print(f"[click] idx={idx}  pos=({sdf_points[idx, 0]:.4f}, {sdf_points[idx, 1]:.4f})  sdf={sdf_values[idx]:.4f}")
        
        # Compute single-point gradient to get weights and indices
        nbr = neighbors[idx] if neighbors is not None else None
        if gradient_estimation == GradientEstimation.RANSAC:
            gradient_i, weights_i, indices_i, loss_i = estimate_gradient_exhaustive(
                sdf_points, sdf_values, idx, neighbors=nbr, verbose=True)
        elif gradient_estimation == GradientEstimation.IRLS:
            gradient_i, weights_i, indices_i, loss_i = estimate_gradient_irls_single(
                sdf_points, sdf_values, idx, neighbors=nbr, last_gradients=gradients, iters=1, verbose=True)
        else:
            return
        point_on_surface = sdf_points[idx] - sdf_values[idx] * gradient_i
        
        # Highlight selected point
        highlighted['point'] = ax.scatter([sdf_points[idx, 0]], [sdf_points[idx, 1]],
                                          c='cyan', s=150, zorder=15, marker='*',
                                          edgecolors='white', linewidths=2)
        # Show projected surface point
        highlighted['surface'] = ax.scatter([point_on_surface[0]], [point_on_surface[1]],
                                            c='yellow', s=100, zorder=14, marker='o',
                                            edgecolors='white', linewidths=2)
        # Correspondence line
        highlighted['line'], = ax.plot([sdf_points[idx, 0], point_on_surface[0]],
                                       [sdf_points[idx, 1], point_on_surface[1]],
                                       'w--', linewidth=2, zorder=13)
        
        # Show neighbors colored by weight
        neighbor_pts = sdf_points[indices_i]
        cmap = plt.cm.RdYlGn_r
        colors = cmap(weights_i)
        highlighted['neighbors'] = ax.scatter(neighbor_pts[:, 0], neighbor_pts[:, 1],
                                              c=colors, s=50, zorder=14, edgecolors='white', linewidths=1)
        # Label each neighbor with its weight
        for k in range(len(indices_i)):
            txt = ax.annotate(f'{weights_i[k]:.5f}', (neighbor_pts[k, 0], neighbor_pts[k, 1]),
                              fontsize=7, color='white', ha='center', va='bottom', zorder=16,
                              bbox=dict(boxstyle='round,pad=0.15', facecolor='black', alpha=0.6))
            highlighted['texts'].append(txt)
            ln, = ax.plot([sdf_points[idx, 0], neighbor_pts[k, 0]],
                          [sdf_points[idx, 1], neighbor_pts[k, 1]],
                          color=colors[k], linewidth=1.5, alpha=0.6, zorder=12)
            highlighted['lines'].append(ln)
        
        ax.set_title(f'Point {idx}: SDF={sdf_values[idx]:.4f}  grad={gradient_i}  loss={loss_i*1e4:.4f}')
        fig.canvas.draw_idle()
    
    fig.canvas.mpl_connect('button_press_event', on_click)

def gradients_diff_norm(gradients1, gradients2):
    """ Compute the mean L2 norm of the difference between two sets of gradients."""
    return np.mean(np.linalg.norm(gradients1 - gradients2, axis=1))

def _polygon_area_signed(poly):
    """Signed area of a simple polygon using the shoelace formula."""
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])

def Haussdorff_distances(shape, original_shape):
    """Backward-compatible wrapper — returns only the Hausdorff distance (float)."""
    return shape_distances(shape, original_shape)['hausdorff']

def test_gradient_estimation(n, neighbor_estimation: NeighborEstimation, gradient_estimation: GradientEstimation, interpolator=None, on_gradient_neighbors=True, see_arcs=False, 
                             show_errors=False, clamp_gradients=False, iters=1000, resolution=500, path_to_image='examples/eiffel.png'):
    print(f"Testing Gradient Estimation with n={n}, neighbor_estimation={neighbor_estimation}, gradient_estimation={gradient_estimation}, on_gradient_neighbors={on_gradient_neighbors}\n")
    points, sdf_points, sdf_values = generate_2D_mesh(n=n, path_to_image=path_to_image)
    if interpolator is None:
        # Create and fit the interpolator
        interpolator = Interpolator(kernel='thin_plate')
        # interpolator.fit(sdf_points, sdf_values)

    visible_arcs = va.compute_visible_arcs(sdf_points, sdf_values)
    # for i, arcs in enumerate(visible_arcs):
    #     visible_arcs[i] = [(0, 2 * np.pi)]
    radii = np.abs(sdf_values)
    degenerate_arcs = va.get_short_arcs(visible_arcs, tol=1e-8)
    # degenerate_arcs = {}
    gradients_gt, new_points = estimate_gradients_oracle(sdf_points, sdf_values, points)

    if on_gradient_neighbors:
        colinear_neighbors = neighbors_on_gradient(sdf_points, sdf_values, tol=1e-5)
        print(f"Number of colinear neighbors: {len(colinear_neighbors)}")
    if gradient_estimation == GradientEstimation.INTERP_GLOBAL_OPT:
        init_gradients = estimate_gradients_interp_global(sdf_points, sdf_values, interpolator, visible_arcs, degenerate_arcs, colinear_neighbors if on_gradient_neighbors else None)
        init_projections = sdf_points - sdf_values[:, np.newaxis] * init_gradients
        # interpolator = CurlFree_Interpolator()
        interpolator.fit(sdf_points, sdf_values, init_gradients, force_recompute=True, use_projection=True)

        init_zero_contours = interpolator.extract_zero_level_set(bounds=((0, 1), (0, 1)), resolution=resolution)
        gradients = estimate_gradients_basicInterp_opt(sdf_points, sdf_values, init_gradients, interpolator, iters,  visible_arcs, degenerate_arcs, points)

        interpolator.fit(sdf_points, sdf_values, gradients, force_recompute=True, use_projection=True)
    elif gradient_estimation == GradientEstimation.INTERP_LOCAL:
        pass
    else:
        if neighbor_estimation == NeighborEstimation.VISIBLE_CONNECTIVITY:
            neighbors = va.find_arcs_neighbors(sdf_points, radii, visible_arcs, 1e-3)
        elif neighbor_estimation == NeighborEstimation.SPATIAL:
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=9, algorithm='auto').fit(sdf_points)
            distances, neighbors = nbrs.kneighbors(sdf_points)
            neighbors = [list(neighbors[i]) for i in range(sdf_points.shape[0])]
        if on_gradient_neighbors:
            for k, v in colinear_neighbors.items():
                neighbors[k] = v
        if gradient_estimation == GradientEstimation.CurlFree_OPT:
            # init_gradients, grad_errors = estimate_gradients_irls(sdf_points, sdf_values, neighbors, interpolator)
            init_gradients = estimate_gradients_interp_global(sdf_points, sdf_values, interpolator, visible_arcs, degenerate_arcs, colinear_neighbors if on_gradient_neighbors else None)
            interpolator = CurlFree_Interpolator()
            interpolator.fit(sdf_points, sdf_values, init_gradients, use_projection=True)
            init_zero_contours = interpolator.extract_zero_level_set(bounds=((0, 1), (0, 1)), resolution=resolution)
            init_projections = sdf_points - sdf_values[:, np.newaxis] * init_gradients
            gradients = estimate_gradients_curlfree_opt(sdf_points, sdf_values, init_gradients, interpolator, iters,  visible_arcs, degenerate_arcs)
            interpolator.fit(sdf_points, sdf_values, gradients, use_projection=True, force_recompute=True)  # Refit the interpolator with the original points and distances
        elif gradient_estimation == GradientEstimation.IRLS:
            gradients, grad_errors = estimate_gradients_irls(sdf_points, sdf_values, neighbors, interpolator)
        elif gradient_estimation == GradientEstimation.RANSAC:
            gradients, grad_errors = estimate_gradients_RANSAC(sdf_points, sdf_values, neighbors)
        elif gradient_estimation == GradientEstimation.FINITE:
            neighbor_estimation = NeighborEstimation.SPATIAL
            gradients, grad_errors = estimate_gradients_finite_diff(sdf_points, sdf_values)
        elif gradient_estimation == GradientEstimation.LSTSQ:
            gradients, grad_errors = estimate_gradients_lstsq(sdf_points, sdf_values, neighbors)
        elif gradient_estimation == GradientEstimation.ORACLE_CURLFREE:
            gradients = gradients_gt
            interpolator = CurlFree_Interpolator()
            interpolator.fit(sdf_points, sdf_values, gradients, use_projection=True, force_recompute=True)  # Refit the interpolator with the original points and distances
    if clamp_gradients:
        va.clamp_gradients_to_arcs(gradients, visible_arcs, degenerate_arcs, sdf_values)
    if gradient_estimation != GradientEstimation.CurlFree_OPT and  gradient_estimation != GradientEstimation.INTERP_GLOBAL_OPT:
        for i, angle in degenerate_arcs.items():
            # For points with degenerate arcs, set gradient directly toward the angle
            gradients[i] = np.array([-np.cos(angle), -np.sin(angle)]) if sdf_values[i] > 0 else np.array([np.cos(angle), np.sin(angle)])
            # print(f"Point {i} has degenerate arc with angle {angle:.2f} radians. Setting gradient to {gradients[i]}.")
            if 'grad_errors' in dir():
                grad_errors[i] = 0.0  # Set error to 0 for these points since we are overriding the gradient

    print("Estimated gradients shape:", gradients.shape)
    points_on_surface = yongs_algorithm(sdf_points, sdf_values, gradients)
    # points_on_surface, gradients = yongs_algorithm2(sdf_points, sdf_values, points)
    wrong_count = rate_gradient_estimation(sdf_points, points_on_surface, sdf_values, tol=1e-3)
    mask = wrong_count <= 0
    print(f"Number of points with correct projection: {np.sum(mask)} out of {len(sdf_points)}")
    mask = wrong_count >= 0
    good_sdf_points = sdf_points[mask]
    points_on_surface_wrong = points_on_surface[~mask]
    good_points_on_surface = points_on_surface[mask]
    good_gradients = gradients[mask]
    contour_segments = marching_cubes_2D(sdf_values)
    print_shape_distances('MC', contour_segments, points)

    V, E = gpy.point_cloud_to_mesh( good_points_on_surface, good_gradients,
    method='PSR',
    psr_screening_weight=10.0,
    psr_outer_boundary_type="Neumann",
    )
    poisson_contour = obj_to_points(V, E)[0]
    print_shape_distances('PSR', poisson_contour, points)

    # visualize results in 2D as heatmap
    # set the size of the figure to be just enough to hold the heatmap
    fig, ax = plt.subplots(figsize=(8, 7))
    grid_x, grid_y = np.mgrid[0:1:resolution*1j, 0:1:resolution*1j]
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
    if not interpolator.trained:
        interpolator.fit(sdf_points, sdf_values, gradients_gt)
    grid_values = interpolator.predict(grid_points).reshape(resolution, resolution)
    im = ax.imshow(grid_values.T, extent=(0, 1, 0, 1), origin='lower', cmap='viridis')
    plt.colorbar(im, ax=ax, label='Interpolated SDF Values')
    # draw the contours at multiple levels (no transpose for contour!)
    contour_levels = np.linspace(-.5, .5, 21)
    ax.contour(grid_x, grid_y, grid_values, levels=contour_levels, colors='yellow', linewidths=0.5, alpha=0.5)
    contour_segments_interpolation = marching_cubes_2D(grid_values.ravel())
    print_shape_distances('Interp', contour_segments_interpolation, points)

    # overlay the original shape
    plt.scatter(sdf_points[:, 0], sdf_points[:, 1], c='r', s=3, label='Original Grid Points')
    plt.scatter(good_points_on_surface[:, 0], good_points_on_surface[:, 1], c='yellow', s=10, label='Projected Surface Points')
    if on_gradient_neighbors:
        ids = set() # To plot the points in colinear_neighbors with different color, we need to collect their indices
        for k, v in colinear_neighbors.items():
            ids.add(k)
            for idx in v:
                ids.add(idx)
        ids = list(ids)
        # show all points in colinear_neighbors with white color
        plt.scatter(sdf_points[ids, 0], sdf_points[ids, 1], c='white', s=10, label='on-gradient Points')
    # Add error labels on each point if available
    if show_errors and 'grad_errors' in dir():
        grad_errors *= 1e4  # Scale error for better visualization
        cmap = plt.cm.coolwarm
        # Use quantile-based normalization to handle skewed distributions
        sorted_errors = np.sort(grad_errors)
        vmin = sorted_errors[int(len(sorted_errors) * 0.05)]  # 5th percentile
        vmax = sorted_errors[int(len(sorted_errors) * 0.95)]  # 95th percentile
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        for i in range(len(sdf_points)):
            color = cmap(norm(grad_errors[i]))
            ax.annotate(f'{grad_errors[i]:.2f}', (sdf_points[i, 0], sdf_points[i, 1]),
                        fontsize=8, color=color, ha='center', va='bottom')

    plt.plot(points[:, 0], points[:, 1], 'b-', linewidth=2, label='Original Shape')
    for k, seg in enumerate(contour_segments):
        plt.plot(seg[:, 0], seg[:, 1], 'k', linewidth=2, label='MC Contour' if k == 0 else None)
    plt.plot(poisson_contour[:, 0], poisson_contour[:, 1], 'm', linewidth=2, label='PSR Contour')
    for k, seg in enumerate(contour_segments_interpolation):
        plt.plot(seg[:, 0], seg[:, 1], 'y', linewidth=2, label='Interpolated Contour' if k == 0 else None)

    if see_arcs:
        # visualize visible arcs just like in test_visible_neighbors
        for i in range(len(visible_arcs)):
            # draw arc as a circle for simplicity
            center = sdf_points[i]
            radius = np.abs(sdf_values[i])
            color = '#FF6B9D' if sdf_values[i] < 0 else "#4ECDC5"
            circle = plt.Circle(center, radius, color=color, fill=False, alpha=1, linewidth=.3)
            ax.add_patch(circle)
    
    if gradient_estimation in [GradientEstimation.INTERP_GLOBAL_OPT]:
        # draw the gradients as arrows for each projected point on surface
        gradients = interpolator.predict_gradient(good_points_on_surface) / np.linalg.norm(interpolator.predict_gradient(good_points_on_surface), axis=1, keepdims=True)
        plt.quiver(good_points_on_surface[:, 0], good_points_on_surface[:, 1], gradients[:, 0], gradients[:, 1], color='cyan', scale=50, width=0.001, label='Estimated Gradients')
        # Draw zero-level contours
        for poly in init_zero_contours:
            poly = np.asarray(poly)
            if len(poly) >= 2:
                ax.plot(poly[:, 0], poly[:, 1], 'r--', linewidth=2, alpha=0.8)
        # draw initial projections
        plt.scatter(init_projections[:, 0], init_projections[:, 1], c='yellow', s=20, label='Initial Projections', alpha=1)

    plot_correspondence(good_sdf_points, good_points_on_surface, plt)
    # connect original points to projected points
    # if on_gradient_neighbors:
    #     plot_correspondence(sdf_points[ids], points_on_surface[ids], plt, color='w')
    ax.set_aspect('equal')
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    plt.title('Gradient Estimation and Surface Point Projection' + ' (' + str(n) + '^2 points ' + neighbor_estimation.value + ' + ' + gradient_estimation.value + ')')
    # plt.legend(loc='upper right')
    # Interactive click: show neighbors and weights
    if gradient_estimation in (GradientEstimation.RANSAC, GradientEstimation.IRLS):
        setup_gradient_click_inspector(fig, ax, sdf_points, sdf_values, gradients, neighbors, gradient_estimation)

    # Click any point to print its index to the console
    def _on_click_print_idx(event):
        if event.inaxes != ax:
            return
        click_pt = np.array([event.xdata, event.ydata])
        idx = int(np.argmin(np.linalg.norm(sdf_points - click_pt, axis=1)))
        print(f"[click] idx={idx}  pos=({sdf_points[idx, 0]:.4f}, {sdf_points[idx, 1]:.4f})  sdf={sdf_values[idx]:.4f}")
    fig.canvas.mpl_connect('button_press_event', _on_click_print_idx)

    grad_diff = gradients_diff_norm(gradients, gradients_gt)
    print(f"{gradient_estimation.value} n={n} Mean L2 norm of gradient difference from ground truth: {grad_diff:.4f}")
    if True:
        Vr, Er = gpy.reach_for_the_arcs(sdf_points, sdf_values, fine_tune_iters=100, batch_size=1000)
        rfta_contours = obj_to_points(Vr, Er)  # list of contour arrays
        # Build NaN-separated array for plotting and distance computation
        if len(rfta_contours) > 0:
            rfta_parts = []
            for k, seg in enumerate(rfta_contours):
                if k > 0:
                    rfta_parts.append(np.full((1, 2), np.nan))
                rfta_parts.append(seg)
            rfta_contour = np.vstack(rfta_parts)
        else:
            rfta_contour = None
        if rfta_contour is not None:
            # plt.plot handles NaN separators automatically (breaks the line)
            plt.plot(rfta_contour[:, 0], rfta_contour[:, 1], 'c', linewidth=2, label='RFTA Contour')
            print_shape_distances('RFTA', rfta_contour, points)
        plt.show()
        return Haussdorff_distances(contour_segments, points), Haussdorff_distances(poisson_contour, points), Haussdorff_distances(contour_segments_interpolation, points), Haussdorff_distances(rfta_contour, points)
    plt.show()
    return Haussdorff_distances(contour_segments, points), Haussdorff_distances(poisson_contour, points), Haussdorff_distances(contour_segments_interpolation, points), None

if __name__ == "__main__":
    # test_mesh(path_to_sdf='out/horse_sdf_8000.npz')
    t0 = time.perf_counter()
    plt = test_mesh(grid_len=20, path_to_obj='examples/holes.obj')

    elapsed = time.perf_counter() - t0
    print(f"  ⏱  {'Total execution time':<30} {elapsed:>7.2f} s")
