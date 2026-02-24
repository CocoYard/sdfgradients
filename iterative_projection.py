import gpytoolbox as gpy
import numpy as np

from SDF_to_surface import marching_cubes_2D

def estimate_gradient(points, sdf_values):
    """ estimation of gradients from SDF values at given points."""
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=8, algorithm='auto').fit(points)
    distances, indices = nbrs.kneighbors(points)
    gradients = np.zeros_like(points)
    for i in range(points.shape[0]):
        neighbor_points = points[indices[i]]
        neighbor_sdf = sdf_values[indices[i]]
        diffs = neighbor_points - points[i]
        sdf_diffs = neighbor_sdf - sdf_values[i]
        A = diffs
        b = sdf_diffs
        grad, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        gradients[i] = grad
    # Normalize gradients
    norms = np.linalg.norm(gradients, axis=1, keepdims=True)
    gradients /= np.maximum(norms, 1e-8)
    return gradients

def project_onto_surface( points, distances, vertices ):
    '''
    Given a collection of points, where each point has a signed distance value. We compute the nearest point on the curve.
    For each point, use the nearest point on the curve (considering edges, not just vertices) to project the point onto the surface.
    '''
    vertices = np.array(vertices)
    n_points = len(points)
    n_verts = len(vertices)
    
    # Create edge arrays
    v0 = vertices  # Shape: (n_verts, 2)
    v1 = np.roll(vertices, -1, axis=0)  # Next vertex (wraps around)
    
    # Broadcast for vectorized computation
    # points: (n_points, 1, 2), v0: (1, n_verts, 2), v1: (1, n_verts, 2)
    points_exp = points[:, np.newaxis, :]  # (n_points, 1, 2)
    v0_exp = v0[np.newaxis, :, :]  # (1, n_verts, 2)
    v1_exp = v1[np.newaxis, :, :]  # (1, n_verts, 2)
    
    # Edge vectors and point-to-v0 vectors
    edge_vecs = v1_exp - v0_exp  # (1, n_verts, 2)
    pt_vecs = points_exp - v0_exp  # (n_points, n_verts, 2)
    
    # Compute projection parameter t for all point-edge pairs
    edge_len_sq = np.sum(edge_vecs * edge_vecs, axis=2, keepdims=True)  # (1, n_verts, 1)
    edge_len_sq = np.maximum(edge_len_sq, 1e-10)
    
    t = np.sum(pt_vecs * edge_vecs, axis=2, keepdims=True) / edge_len_sq  # (n_points, n_verts, 1)
    t = np.clip(t, 0.0, 1.0)
    
    # Compute nearest points on all edges
    nearest_pts = v0_exp + t * edge_vecs  # (n_points, n_verts, 2)
    
    # Compute distances to all edges
    dists = np.linalg.norm(points_exp - nearest_pts, axis=2)  # (n_points, n_verts)
    
    # Find minimum distance edge for each point
    min_edge_idx = np.argmin(dists, axis=1)  # (n_points,)
    nearest_vertices = nearest_pts[np.arange(n_points), min_edge_idx]  # (n_points, 2)
    
    # Compute gradients as the normalized vector from point to nearest vertex
    gradients = nearest_vertices - points
    norms = np.linalg.norm(gradients, axis=1, keepdims=True)
    gradients /= np.maximum(norms, 1e-8)
    
    # Flip gradient for points with positive distance (outside)
    gradients[distances > 0] *= -1
    
    return points - distances[:, np.newaxis] * gradients, gradients

def rate_gradient_estimation(sdf_points, points_on_surface, sdf_values, tol=1e-5):
    """ rate the gradient estimation by testing projected points enclosed by other points' circles."""
    # Vectorized version: compute all pairwise distances at once
    # Shape: (num_points, num_points)
    dists = np.linalg.norm(points_on_surface[:, np.newaxis, :] - sdf_points[np.newaxis, :, :], axis=2)
    
    # Broadcast radius to match distance matrix shape
    radii = np.abs(sdf_values)[np.newaxis, :]  # Shape: (1, num_points)
    
    # Check which points are inside other points' circles
    inside = dists < radii - tol # Shape: (num_points, num_points)
    
    # Exclude self-comparison (diagonal elements)
    np.fill_diagonal(inside, False)
    
    # Count how many circles each point falls into
    wrong_count = np.sum(inside, axis=1)
    # wrong_count = np.zeros(sdf_points.shape[0], dtype=int)

    # Compute the Laplacian of sdf_values at sdf_points
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=8, algorithm='auto').fit(sdf_points)
    distances, indices = nbrs.kneighbors(sdf_points)
    laplacian = np.zeros(sdf_points.shape[0])
    for i in range(sdf_points.shape[0]):
        neighbor_sdf = sdf_values[indices[i]]
        laplacian[i] = np.sum(neighbor_sdf) - 8 * sdf_values[i]
    # Penalize points with negative Laplacian (indicating incorrect gradient direction)
    wrong_count += (laplacian < 0).astype(int)*2
    
    return wrong_count

def obj_to_points(V, E):
    """ Convert mesh (V, E) to list of contours (points)."""
    # V is the reconstructed vertices, E is the edge list
    # sort V by E to make contiguous order
    poisson_contour = []
    if len(E) > 0:
        start_vertex = E[0, 0]
        poisson_contour_indices = [start_vertex]
        current_vertex = start_vertex
        while True:
            next_edges = E[E[:, 0] == current_vertex]
            if len(next_edges) == 0:
                break
            next_vertex = next_edges[0, 1]
            if next_vertex == start_vertex:
                break
            poisson_contour_indices.append(next_vertex)
            current_vertex = next_vertex
        poisson_contour.append( V[poisson_contour_indices] )
    return poisson_contour

def iterative_projection(sdf_points, sdf_values, shape, max_iterations=5):
    """
    Perform iterative projection of points onto the zero level set of the SDF.
    Args:
        sdf_points (np.ndarray): Nxd array of points in d-dimensional space.
        sdf_values (np.ndarray): N array of SDF values at the corresponding points.
    """
    # initialize projected points
    gradients = estimate_gradient(sdf_points, sdf_values)
    projected_points = sdf_points - sdf_values[:, np.newaxis] * gradients
    # filter out points with wrong gradient estimation
    wrong_counts = rate_gradient_estimation(sdf_points, projected_points, sdf_values)
    mask = wrong_counts == 0
    good_projected_points = projected_points[mask]
    good_sdf_points = sdf_points[mask]
    good_sdf_values = sdf_values[mask]
    good_gradients = gradients[mask]
    # reconstruct the initial surface by Poisson surface reconstruction
    V, E = gpy.point_cloud_to_mesh( good_projected_points, good_gradients,
    method='PSR',
    psr_screening_weight=10.0,
    psr_outer_boundary_type="Neumann",
    )
    poisson_contour = marching_cubes_2D(sdf_points, sdf_values)[0]
    # poisson_contour = shape
    # poisson_contour = obj_to_points(V, E)

    for iteration in range(max_iterations):
        new_points, new_gradients = project_onto_surface( sdf_points, sdf_values, poisson_contour )
        # check gradient estimation quality
        # print("error = ", new_points - (sdf_points - sdf_values[:, np.newaxis] * new_points))

        wrong_counts = rate_gradient_estimation(sdf_points, new_points, sdf_values)
        mask = wrong_counts == 0
        good_projected_points = new_points[mask]
        good_gradients = new_gradients[mask]
        good_sdf_points = sdf_points[mask]
        print(f"Iteration {iteration+1}: {good_projected_points.shape[0]} good projected points out of {sdf_points.shape[0]}")
        V, E = gpy.point_cloud_to_mesh( good_projected_points, good_gradients,
        method='PSR',
        psr_screening_weight=10.0,
        psr_outer_boundary_type="Neumann",
        )
        poisson_contour = obj_to_points(V, E)[0]

    return poisson_contour, good_projected_points, good_sdf_points, good_gradients