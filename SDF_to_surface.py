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

class NeighborEstimation(Enum):
    VISIBLE_CONNECTIVITY = 'visible_connectivity'
    SPATIAL = 'spatial'

class GradientEstimation(Enum):
    CurlFree_OPT = 'curlfree_opt'
    INTERP_GLOBAL = 'interp_global'
    INTERP_LOCAL = 'interp_local'
    ORACLE_CURLFREE = 'oracle_curlfree'
    IRLS = 'irls'
    RANSAC = 'ransac'
    FINITE = 'finite'
    LSTSQ = 'lstsq'

def generate_test_mesh_data( path_to_mesh, outbase, num_points=500 ):
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
    grid_size = int(np.ceil(num_points ** (1/3)))
    x = np.linspace(bbox_min[0], bbox_max[0], grid_size)
    y = np.linspace(bbox_min[1], bbox_max[1], grid_size)
    z = np.linspace(bbox_min[2], bbox_max[2], grid_size)
    X, Y, Z = np.meshgrid(x, y, z)
    points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    if points.shape[0] > num_points:
        points = points[np.random.choice(points.shape[0], num_points, replace=False)]
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
    np.savez("out/" + outbase + "_sdf_" + str(num_points) + ".npz",
             points=points,
             sdf_values=distances,
             gradients=gradients)
    print("✅ Saved SDF data:", "out/" + outbase + "_sdf_" + str(num_points) + ".npz")
    return points, distances, gradients

# Example usage:
def test_circle():
        # Generate a signed distance field for a circle. 
    # Sample data points and values. 16 points in grid corners in 2D
    num_points = 16
    points = np.array([[x, y] for x in np.linspace(0, 1, 4) for y in np.linspace(0, 1, 4)])
    values = np.sqrt((points[:, 0] - 0.5)**2 + (points[:, 1] - 0.5)**2) - 0.3  # Signed distance from circle of radius 0.3 centered at (0.5, 0.5)    

    # Create and fit the interpolator
    interpolator = Interpolator(kernel='thin_plate')
    interpolator.fit(points, values)

    # visualize results in 2D as heatmap
    grid_x, grid_y = np.mgrid[-0.5:1.5:100j, -0.5:1.5:100j]
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
    grid_values = interpolator.predict(grid_points).reshape(100, 100)
    plt.imshow(grid_values.T, extent=(-0.5, 1.5, -0.5, 1.5), origin='lower', cmap='viridis')
    plt.colorbar(label='Interpolated Values')
    plt.scatter(points[:, 0], points[:, 1], c='red', label='Data Points')
    plt.legend()
    plt.title('Duchon Interpolation Heatmap')
    plt.show()

def test_sphere():
    # Generate a signed distance field for a sphere.
    points = np.array([[x, y, z] for x in np.linspace(0, 1, 5) for y in np.linspace(0, 1, 5) for z in np.linspace(0, 1, 5)])
    values = np.sqrt((points[:, 0] - 0.5)**2 + (points[:, 1] - 0.5)**2 + (points[:, 2] - 0.5)**2) - 0.3  # Signed distance from sphere of radius 0.3 centered at (0.5, 0.5, 0.5)    

    # Create and fit the interpolator
    interpolator = Interpolator(kernel='cubic')
    interpolator.fit(points, values)

    # visualize results using marching cubes to extract isosurface
    grid_resolution = 50
    grid_x, grid_y, grid_z = np.mgrid[-0.5:1.5:grid_resolution*1j, -0.5:1.5:grid_resolution*1j, -0.5:1.5:grid_resolution*1j]
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T
    grid_values = interpolator.predict(grid_points).reshape(grid_resolution, grid_resolution, grid_resolution)
    
    # Extract isosurface at value 0 using marching cubes
    from skimage import measure
    verts, faces, normals, values = measure.marching_cubes(grid_values, level=0.0, spacing=(2.0/grid_resolution, 2.0/grid_resolution, 2.0/grid_resolution))
    
    # Adjust vertices to match the grid coordinates
    verts = verts - 0.5
    
    # Plot the mesh
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    mesh = Poly3DCollection(verts[faces], alpha=0.7, edgecolor='k', linewidth=0.1)
    mesh.set_facecolor([0.5, 0.7, 1.0])
    ax.add_collection3d(mesh)
    
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_zlim(-0.5, 1.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Isosurface Extraction (Marching Cubes)')
    plt.show()

def test_mesh(path_to_sdf='out/bunny_sdf_4000.npz'):
    # read sdf data from file
    data = np.load(path_to_sdf)
    points = data['points']
    distances = data['sdf_values']

    # Create and fit the interpolator
    interpolator = Interpolator(kernel='cubic')
    interpolator.fit(points, distances)

    # Predict at new points (for example, the original points)
    predictions = interpolator.predict(points)
    print("Predictions at original points:", predictions[:10])  # Print first 10 predictions

    # visualize results using marching cubes to extract isosurface
    grid_resolution = 50
    grid_x, grid_y, grid_z = np.mgrid[-0.5:1.5:grid_resolution*1j, -0.5:1.5:grid_resolution*1j, -0.5:1.5:grid_resolution*1j]
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T
    grid_values = interpolator.predict(grid_points).reshape(grid_resolution, grid_resolution, grid_resolution)
    
    # Extract isosurface at value 0 using marching cubes
    from skimage import measure
    verts, faces, normals, values = measure.marching_cubes(grid_values, level=0.0, spacing=(2.0/grid_resolution, 2.0/grid_resolution, 2.0/grid_resolution))
    
    # Adjust vertices to match the grid coordinates
    verts = verts - 0.5
    
    # Plot the mesh
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    mesh = Poly3DCollection(verts[faces], alpha=0.7, edgecolor='k', linewidth=0.1)
    mesh.set_facecolor([0.5, 0.7, 1.0])
    ax.add_collection3d(mesh)
    
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_zlim(-0.5, 1.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Isosurface Extraction (Marching Cubes)')
    plt.show()

def generate_2D_mesh(n=4, visualize=False, path_to_image='examples/image.png'):
    """
    Generate a 2D mesh with signed distance function (SDF) values from an image contour.
    This function reads an image from 'examples/image.png', extracts the longest contour
    from the grayscale version, normalizes it to [0,1] coordinates, and computes signed
    distance values on a regular grid. Points inside the contour have negative distances,
    while points outside have positive distances.
    Parameters
    ----------
    n : int, optional
        Number of grid points along each axis for SDF sampling. Creates an n×n grid
        of sample points in [0,1]×[0,1]. Default is 4.
    visualize : bool, optional
        If True, displays a plot of the extracted 2D shape with contour outline
        and filled interior. Default is False.
    Returns
    -------
    points : numpy.ndarray
        Shape (N, 2) array containing the boundary points of the extracted contour,
        normalized to [0,1] coordinates with y-axis flipped.
    sdf_points : numpy.ndarray
        Shape (n², 2) array containing the regular grid sample points where SDF
        values were computed.
    values : numpy.ndarray
        Shape (n²,) array containing signed distance values at each grid point.
        Negative values indicate points inside the shape, positive values indicate
        points outside the shape.
    Notes
    -----
    - Requires 'examples/image.png' to exist in the current working directory
    - The function extracts contours at grayscale level 0.5
    - Uses the longest contour found in the image
    - Coordinates are normalized and y-axis is flipped for standard mathematical convention
    """
    # read image and generate 2D mesh data from examples/image.png
    from skimage import io, color, measure
    from skimage.util import img_as_float
    import warnings
    image = io.imread(path_to_image)
    # image shape = (H, W, 4)
    image = img_as_float(image[:, :, :3])  # Convert to float [0,1] RGB
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        gray_image = color.rgb2gray(image)  # Convert to grayscale
    # Extract contours at a constant value of 0.5
    contours = measure.find_contours(gray_image, level=0.5)
    # Use the longest contour
    longest_contour = max(contours, key=len)
    points = longest_contour[:, ::-1]  # Swap columns to get (x, y)
    points /= np.max(points)  # Normalize to [0, 1]
    points[:, 1] = 1.0 - points[:, 1]  # Flip y-axis
    values = np.zeros(points.shape[0])

    # visualize the 2D shape
    if visualize:
        plt.figure(figsize=(6, 6))
        plt.plot(points[:, 0], points[:, 1], 'b-', linewidth=2)
        plt.fill(points[:, 0], points[:, 1], 'lightblue', alpha=0.5)
        plt.axis('equal')
        plt.title('2D Shape from Image Contour')
    plt.show()
    # Compute signed distance values (inside/outside)
    # Sample data points and values. 16 points in grid corners in 2D
    sdf_points = np.array([[x, y] for x in np.linspace(0, 1, n) for y in np.linspace(0, 1, n)])
    # compute signed distance values to the 2D shape
    from matplotlib.path import Path
    shape_path = Path(longest_contour[:, ::-1] / np.max(longest_contour))  # Create a path for the shape
    def signed_distance(pt):
        dist = np.min(np.linalg.norm(longest_contour[:, ::-1] / np.max(longest_contour) - pt, axis=1))
        if shape_path.contains_point(pt):
            return -dist  # Inside the shape
        else:
            return dist  # Outside the shape
    values = np.array([signed_distance(pt) for pt in sdf_points])

    return points, sdf_points, values

def test_2D_mesh(interpolator=None, n=4):
    points, sdf_points, values = generate_2D_mesh(n=n)
    if interpolator is None:
        # Create and fit the interpolator
        interpolator = Interpolator(kernel='thin_plate')
        interpolator.fit(sdf_points, values)
    # visualize results in 2D as heatmap
    fig, ax = plt.subplots(figsize=(8, 7))
    grid_x, grid_y = np.mgrid[0:1:100j, 0:1:100j]
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
    grid_values = interpolator.predict(grid_points).reshape(100, 100)
    im = ax.imshow(grid_values.T, extent=(0, 1, 0, 1), origin='lower', cmap='viridis')
    plt.colorbar(im, ax=ax, label='Interpolated SDF Values')
    # overlay the original shape
    ax.plot(points[:, 0], points[:, 1], 'w-', linewidth=1, label='Original Shape')
    # draw the contours at multiple levels (no transpose for contour!)
    contour_levels = np.linspace(np.min(grid_values), np.max(grid_values), 10)
    # insert 0 level contour if not in levels
    if 0.0 not in contour_levels:
        contour_levels = np.sort( np.append( contour_levels, 0.0 ) )
    ax.contour(grid_x, grid_y, grid_values, levels=contour_levels, colors='yellow', linewidths=0.5, alpha=0.5)
    # make 0-level contour distinct
    ax.contour(grid_x, grid_y, grid_values, levels=[0.0], colors='red', linewidths=1)
    # compute interpolation errors: original values - predicted values
    predicted_values = interpolator.predict(sdf_points)
    errors = np.abs(values - predicted_values)
    # show data points with error as color (darker = larger error)
    scatter = ax.scatter(sdf_points[:, 0], sdf_points[:, 1], c=errors, s=10, cmap='hot', 
                         edgecolors='black', linewidths=0.5, label='Data Points')
    plt.colorbar(scatter, ax=ax, label='|True - Interpolated|')
    ax.set_title('Duchon Interpolation Heatmap for 2D Shape' + ' (' + str(n) + '^2 points)')
    # ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
    return interpolator

def yongs_algorithm( points, distances, gradients ):
    '''
    Given a collection of points, where each point has a signed distance value and a gradient.
    For each point, outputs the point -distance units along the gradient direction.
    Parameters:
    points: (N, d) array of point coordinates
        The input points in d-dimensional space.
    distances: (N,) array of signed distance values
        The signed distance values for each point.
    gradients: (N, d) array of gradient vectors
        The gradient vectors at each point.
    '''
    # Normalize the gradients to unit vectors
    # norm_gradients = gradients / np.linalg.norm(gradients, axis=1, keepdims=True)
    # Compute the new points by moving along the gradient direction
    new_points = points - (distances[:, np.newaxis] * gradients)
    return new_points

def estimate_gradients_curlfree_opt(points, distances, init_gradients, interpolator : CurlFree_Interpolator):
    """
    Estimate gradients by optimizing a curl-free potential function to fit the signed distance data.

    Parameters:
    -----------
    points: (N, d) array of point coordinates
        The input points in d-dimensional space.
    distances: (N,) array of signed distance values
        The signed distance values for each point.
    init_gradients: (N, d) array of initial gradient estimates
        The initial gradient estimates at each point, which can be obtained from a global interpolator or other methods.

    Returns:
    ---------
    gradients: (N, d) array of estimated gradient vectors
        The estimated gradient vectors at each input point.
    """
    # optimize the gradients to be curl-free
    gradients = opt.opt(points, distances, init_gradients, num_iter=500, lr=1e-2)
    # gradients = opt.opt(points, init_gradients, num_iter=500, lr=1e-2)
    gradients /= np.linalg.norm(gradients, axis=1, keepdims=True)  # Normalize to unit vectors
    interpolator.fit(points, distances, gradients)  # Refit the interpolator with the original points and distances
    return gradients


def estimate_gradients_interp_global(sdf_points, sdf_values, interpolator : Interpolator, visible_arcs, degenerate_arcs, colinear_neighbors=None, clamp=True):
    '''
    Estimate gradients by fitting a global Duchon interpolator to the signed distance data and evaluating its gradient.

    Parameters:
    -----------
    sdf_points: (N, d) array of point coordinates
        The input points in d-dimensional space.
    sdf_values: (N,) array of signed distance values
        The signed distance values for each point.
    interpolator: Interpolator
        A fitted Interpolator object that can predict values and gradients.
    visible_arcs: a dictionary: point index -> list of visible arcs
        A collection of visible arcs that can be used to clamp the gradients.
    degenerate_arcs: a dictionary: point index -> list of degenerate arcs
        A collection of degenerate arcs that can be used to clamp the gradients.
    colinear_neighbors: a dictionary: point index -> list of colinear neighbors, optional
        A collection of colinear neighbors that can be used for gradient estimation. Default is None.
    Returns:
    -----------
    gradients: (N, d) array of estimated gradient vectors
        The estimated gradient vectors at each input point.
    '''

    to_train_points = sdf_points.copy()
    to_train_sdf = sdf_values.copy()
    for i, angle in degenerate_arcs.items():
        # For points with degenerate arcs, set gradient directly toward the angle
        grad = np.array([-np.cos(angle), -np.sin(angle)]) if sdf_values[i] > 0 else np.array([np.cos(angle), np.sin(angle)])
        # Add a new point on the surface along this gradient direction
        new_point = sdf_points[i] - sdf_values[i] * grad
        to_train_points = np.vstack([to_train_points, new_point])
        to_train_sdf = np.append(to_train_sdf, 0)  # The SDF value at the projected point should be 0
    print(f"After adding points for degenerate arcs, total points: {len(to_train_points)}")
    interpolator.fit(to_train_points, to_train_sdf)
    gradients = interpolator.sample_best_gradients(sdf_points, sdf_values)
    # Clamp gradients to visible arcs
    if clamp:
        clamp_indices = va.clamp_gradients_to_arcs(gradients, visible_arcs, degenerate_arcs, sdf_values)
        if colinear_neighbors is not None:
            va.clamp_gradient_to_colinear_neighbors(gradients, colinear_neighbors, degenerate_arcs, sdf_points, sdf_values, clamp_indices)

    # gradients[704] *= -1.0  # manually flip the gradient for point 704 which is a special case with wrong orientation (TODO: find a more principled way to handle this)
    # gradients[764] = np.array([np.cos(5/4*np.pi), np.sin(5/4*np.pi)])  # manually set the gradient for point 764 which is a special case with wrong orientation (TODO: find a more principled way to handle this)
    return gradients

def estimate_gradients_oracle( points, distances, vertices ):
    '''
    Given a collection of points, where each point has a signed distance value. We compute the nearest point on the curve.
    For each point, outputs the nearest point on the curve (considering edges, not just vertices).
    '''
    # Construct edges by connecting consecutive vertices (assuming closed curve)
    n_verts = len(vertices)
    n_points = len(points)
    
    # Get edge start and end points
    v0 = vertices[:-1]  # All vertices except last
    v1 = vertices[1:]   # All vertices except first
    # Add closing edge
    v0 = np.vstack([v0, vertices[-1:]])
    v1 = np.vstack([v1, vertices[:1]])
    
    # Vectorized computation for all points and all edges
    # Broadcast points to shape (n_points, 1, 2) and edges to (1, n_edges, 2)
    points_exp = points[:, np.newaxis, :]  # (n_points, 1, 2)
    v0_exp = v0[np.newaxis, :, :]          # (1, n_edges, 2)
    v1_exp = v1[np.newaxis, :, :]          # (1, n_edges, 2)
    
    # Edge vectors: (1, n_edges, 2)
    edge_vecs = v1_exp - v0_exp
    # Point-to-v0 vectors: (n_points, n_edges, 2)
    pt_vecs = points_exp - v0_exp
    
    # Compute projection parameter t for all point-edge pairs
    edge_length_sq = np.sum(edge_vecs * edge_vecs, axis=2)  # (1, n_edges)
    edge_length_sq = np.maximum(edge_length_sq, 1e-10)  # Avoid division by zero
    
    t = np.sum(pt_vecs * edge_vecs, axis=2) / edge_length_sq  # (n_points, n_edges)
    t = np.clip(t, 0, 1)  # Clamp to [0, 1]
    
    # Compute projections: (n_points, n_edges, 2)
    projections = v0_exp + t[:, :, np.newaxis] * edge_vecs
    
    # Compute distances from points to projections
    dists = np.linalg.norm(points_exp - projections, axis=2)  # (n_points, n_edges)
    
    # Find minimum distance edge for each point
    min_edge_idx = np.argmin(dists, axis=1)  # (n_points,)
    
    # Extract closest points
    new_points = projections[np.arange(n_points), min_edge_idx]
    
    gradients = (points - new_points) / np.linalg.norm(points - new_points, axis=1, keepdims=True)
    for i in range(n_points):
        if distances[i] < 0:
            gradients[i] *= -1.0
    return gradients, new_points

def neighbors_on_gradient(points, sdf_values, tol=1e-3):
    """ In SDF, any 2 points on the gradient line segment should have the SDF value change equal to their distance.
    So we can use this property to find neighbors along the gradient direction.
    """
    neighbors = {}
    for i in range(points.shape[0]):
        neighbors[i] = []
        for j in range(points.shape[0]):
            if i == j:
                continue
            dist = np.linalg.norm(points[i] - points[j])
            sdf_diff = np.abs(sdf_values[i] - sdf_values[j])
            if np.abs(dist - sdf_diff) < tol:
                neighbors[i].append(j)
    # clean empty neighbors
    neighbors = {k: v for k, v in neighbors.items() if len(v) > 0}
    return neighbors

def estimate_gradients_RANSAC(points, sdf_values, neighbors=None):
    """ estimation of gradients from SDF values at given points.
    Inputs:
        points: (N, 2) coordinates of sample points
        sdf_values: (N,) corresponding SDF values
        k: number of neighbors to find for each point for RANSAC (default is 3×3 neighborhood)
    Returns:
        gradients: (N, 2) robust gradients for each point
        errors: (N,) corresponding errors for each gradient, defined as the mean of squared distance error of all selected neighbors.
    """
    gradients = np.zeros_like(points)
    errors = np.zeros(points.shape[0])
    for i in range(points.shape[0]):
        gradients[i], _, _, errors[i] = estimate_gradient_exhaustive(points, sdf_values, i, neighbors=neighbors[i] if neighbors is not None else None)
    return gradients, errors

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

# def estimate_gradients_Duchon_reflect(points, sdf_values, neighbors=None):
#     # construct a Duchon interpolator with the given points and sdf_values, then sample the gradient at each point
#     interpolator = Interpolator(kernel='thin_plate')
#     interpolator.fit(points, sdf_values)
#     interpolator.sample_best_gradient(points, sdf_values, num_samples=50)
#     to_train_points = sdf_points
#     to_train_sdf = sdf_values
#     for i, angle in degenerate_arcs.items():
#         # For points with degenerate arcs, set gradient directly toward the angle
#         grad = np.array([-np.cos(angle), -np.sin(angle)]) if sdf_values[i] > 0 else np.array([np.cos(angle), np.sin(angle)])
#         # Add a new point on the surface along this gradient direction
#         new_point = sdf_points[i] - sdf_values[i] * grad
#         to_train_points = np.vstack([to_train_points, new_point])
#         to_train_sdf = np.append(to_train_sdf, 0)  # The SDF value at the projected point should be 0
#     print(f"After adding points for degenerate arcs, total points: {len(to_train_points)}")
#     interpolator.fit(to_train_points, to_train_sdf)

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
    # wrong_count = np.zeros(sdf_points.shape[0], dtype=int)

    # # Compute the Laplacian of sdf_values at sdf_points
    # from sklearn.neighbors import NearestNeighbors
    # nbrs = NearestNeighbors(n_neighbors=8, algorithm='auto').fit(sdf_points)
    # distances, indices = nbrs.kneighbors(sdf_points)
    # laplacian = np.zeros(sdf_points.shape[0])
    # for i in range(sdf_points.shape[0]):
    #     neighbor_sdf = sdf_values[indices[i]]
    #     laplacian[i] = np.sum(neighbor_sdf) - 8 * sdf_values[i]
    # # Penalize points with negative Laplacian (indicating incorrect gradient direction)
    # wrong_count += (laplacian < 0).astype(int)*2
    
    return wrong_count

def marching_cubes_2D(sdf_points, sdf_values):
    """ Extract 0-level contour from 2D SDF values using marching squares on the grid.
    Assumes sdf_points are on a regular n×n grid in [0,1]×[0,1]."""
    from skimage import measure
    n = int(np.sqrt(len(sdf_values)))
    sdf_grid = sdf_values.reshape(n, n)
    contours = measure.find_contours(sdf_grid, level=0.0)
    # Convert pixel indices back to [0,1] coordinates
    # find_contours returns (row, col) in grid index space
    contour_points = []
    for contour in contours:
        # row → x, col → y (matching the np.linspace grid ordering)
        scaled = contour / (n - 1)
        contour_points.append(scaled)
    return contour_points

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

def Haussdorff_distances(shape, original_shape):
    """ Compute the Hausdorff distance between polylines (not point clouds).

    Parameters
    ----------
    shape : list of (N_i, 2) arrays, or a single (N, 2) array
        One or more polylines, e.g. [[seg1], [seg2]] from marching cubes.
    original_shape : (M, 2) array or list of (M_j, 2) arrays
        The reference polyline(s).

    Returns
    -------
    float  – the (symmetric) Hausdorff distance, computed as
             max(directed_H(shape→original), directed_H(original→shape))
             using point-to-segment distances.
    """

    def _point_to_polylines_min_dist(points, polylines):
        """Min distance from each query point to the *segments* of polylines."""
        min_dists = np.full(len(points), np.inf)
        for poly in polylines:
            if len(poly) < 2:
                # Degenerate single-point polyline
                dists = np.linalg.norm(points - poly[0], axis=1)
                min_dists = np.minimum(min_dists, dists)
                continue
            a = poly[:-1]   # (S, 2) segment start points
            b = poly[1:]    # (S, 2) segment end points
            ab = b - a      # (S, 2)
            ab_sq = np.sum(ab ** 2, axis=1)  # (S,)
            # Process in chunks to limit memory (M * S * 2 floats)
            chunk = max(1, 50_000 // max(len(a), 1))
            for i0 in range(0, len(points), chunk):
                i1 = min(i0 + chunk, len(points))
                pts = points[i0:i1]                          # (C, 2)
                ap = pts[:, None, :] - a[None, :, :]         # (C, S, 2)
                t = np.sum(ap * ab[None, :, :], axis=2) / np.maximum(ab_sq[None, :], 1e-30)
                t = np.clip(t, 0.0, 1.0)                     # (C, S)
                closest = a[None, :, :] + t[:, :, None] * ab[None, :, :]  # (C, S, 2)
                dists = np.linalg.norm(pts[:, None, :] - closest, axis=2) # (C, S)
                min_dists[i0:i1] = np.minimum(min_dists[i0:i1],
                                              np.min(dists, axis=1))
        return min_dists

    # ---- normalise inputs to list-of-polylines ----
    if isinstance(shape, np.ndarray):
        shape_list = [shape]
    else:
        shape_list = list(shape)

    if isinstance(original_shape, np.ndarray):
        orig_list = [original_shape]
    else:
        orig_list = list(original_shape)

    # Gather all vertices from each side
    shape_verts = np.concatenate(shape_list, axis=0)
    orig_verts  = np.concatenate(orig_list, axis=0)

    # directed Hausdorff: shape → original  (each shape vertex → nearest original segment)
    d_s2o = _point_to_polylines_min_dist(shape_verts, orig_list)
    # directed Hausdorff: original → shape  (each original vertex → nearest shape segment)
    d_o2s = _point_to_polylines_min_dist(orig_verts, shape_list)

    return float(max(np.max(d_s2o), np.max(d_o2s)))

def test_gradient_estimation(n, neighbor_estimation: NeighborEstimation, gradient_estimation: GradientEstimation, interpolator=None, on_gradient_neighbors=True, see_arcs=False, show_errors=False, clamp_gradients=True):
    points, sdf_points, sdf_values = generate_2D_mesh(n=n, path_to_image='examples/horse.png')
    if interpolator is None:
        # Create and fit the interpolator
        interpolator = Interpolator(kernel='thin_plate')
        # interpolator.fit(sdf_points, sdf_values)

    visible_arcs = va.compute_visible_arcs(sdf_points, sdf_values)
    radii = np.abs(sdf_values)
    degenerate_arcs = va.get_short_arcs(visible_arcs, tol=1e-8)
    gradients_gt, new_points = estimate_gradients_oracle(sdf_points, sdf_values, points)

    if on_gradient_neighbors:
        colinear_neighbors = neighbors_on_gradient(sdf_points, sdf_values, tol=1e-5)
    if gradient_estimation == GradientEstimation.INTERP_GLOBAL:
        gradients = estimate_gradients_interp_global(sdf_points, sdf_values, interpolator, visible_arcs, degenerate_arcs, colinear_neighbors if on_gradient_neighbors else None)
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
            gradients = estimate_gradients_curlfree_opt(sdf_points, sdf_values, init_gradients, interpolator)
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
            interpolator.fit(sdf_points, sdf_values, gradients)
            # interpolator.fit(np.append(sdf_points, new_points, axis=0), np.append(sdf_values, np.zeros(len(new_points)), axis=0))
    if clamp_gradients:
        va.clamp_gradients_to_arcs(gradients, visible_arcs, degenerate_arcs, sdf_values)

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
    contour_segments = marching_cubes_2D(sdf_points, sdf_values)
    print(f"Haussdorff distance for MC: {Haussdorff_distances(contour_segments, points):.4f}")

    V, E = gpy.point_cloud_to_mesh( good_points_on_surface, good_gradients,
    method='PSR',
    psr_screening_weight=10.0,
    psr_outer_boundary_type="Neumann",
    )
    poisson_contour = obj_to_points(V, E)[0]
    print(f"Haussdorff distance for PSR: {Haussdorff_distances(poisson_contour, points):.4f}")

    # visualize results in 2D as heatmap
    # set the size of the figure to be just enough to hold the heatmap
    fig, ax = plt.subplots(figsize=(8, 7))
    grid_x, grid_y = np.mgrid[0:1:100j, 0:1:100j]
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
    if not interpolator.trained:
        interpolator.fit(sdf_points, sdf_values, gradients_gt)
    grid_values = interpolator.predict(grid_points).reshape(100, 100)
    im = ax.imshow(grid_values.T, extent=(0, 1, 0, 1), origin='lower', cmap='viridis')
    plt.colorbar(im, ax=ax, label='Interpolated SDF Values')
    # draw the contours at multiple levels (no transpose for contour!)
    contour_levels = np.linspace(-.5, .5, 21)
    ax.contour(grid_x, grid_y, grid_values, levels=contour_levels, colors='yellow', linewidths=0.5, alpha=0.5)
    # visualize results in 2D
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
        print(f"\nNumber of colinear neighbors: {len(colinear_neighbors)}")
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

    if see_arcs:
        Vr, Er = gpy.reach_for_the_arcs(sdf_points, sdf_values, fine_tune_iters=100, batch_size=1000)
        rfta_contour = obj_to_points(Vr, Er)[0]
        # plt.plot(rfta_contour[:, 0], rfta_contour[:, 1], 'y', linewidth=2, label='RFTA Contour')
        # visualize visible arcs just like in test_visible_neighbors
        for i in range(len(visible_arcs)):
            # draw arc as a circle for simplicity
            center = sdf_points[i]
            radius = np.abs(sdf_values[i])
            color = '#FF6B9D' if sdf_values[i] < 0 else "#4ECDC5"
            circle = plt.Circle(center, radius, color=color, fill=False, alpha=1, linewidth=.3)
            ax.add_patch(circle)
        print(f"Haussdorff distance for RFTA: {Haussdorff_distances(rfta_contour, points):.4f}")

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
    
    grad_diff = gradients_diff_norm(gradients, gradients_gt)
    print(f"{gradient_estimation.value} n={n} Mean L2 norm of gradient difference from oracle: {grad_diff:.4f}")
    plt.show()
    # return Haussdorff_distances(contour_segments, points), Haussdorff_distances(poisson_contour, points), Haussdorff_distances(rfta_contour, points) if see_arcs else None
    return grad_diff

def test_visible_neighbors(n=4, show_all=False):
    points, sdf_points, sdf_values = generate_2D_mesh(n=n, path_to_image='examples/horse.png')
    visible_arcs = va.compute_visible_arcs(sdf_points, sdf_values)

    # Find degenerate arcs: only one arc with angular length < 1e-8
    degenerate_indices = []
    for i, arcs in enumerate(visible_arcs):
        if len(arcs) == 1:
            s, e = arcs[0]
            arc_len = (e - s)
            if arc_len <= 1e-8:
                degenerate_indices.append(i)
    if degenerate_indices:
        print(f"\n⚠️  {len(degenerate_indices)} degenerate arc points (1 arc, length < 1e-8):")
        for i in degenerate_indices:
            s, e = visible_arcs[i][0]
            print(f"  Point {i}: pos=({sdf_points[i,0]:.4f}, {sdf_points[i,1]:.4f}), "
                  f"sdf={sdf_values[i]:.4f}, arc=({s:.6f}, {e:.6f})")
    else:
        print("\n✅ No degenerate arc points found.")

    fig, ax = va.visualize_visible_arcs(sdf_points, sdf_values, visible_arcs, points)
    radii = np.abs(sdf_values)
    neighbors = va.find_arcs_neighbors(sdf_points, radii, visible_arcs, tol=1e-3)
    
    # If show_all is True, display all neighbor connections at once with colors
    if show_all:
        print(f"Showing all neighbor connections for {len(sdf_points)} points")
        
        # Draw all points with their neighbors in different colors
        for i in range(len(sdf_points)):
            if len(neighbors[i]) == 0:
                continue
            
            # Generate a unique color for this point
            color = plt.cm.tab20(i % 20)
            
            # Draw the center point as a star
            ax.scatter([sdf_points[i, 0]], [sdf_points[i, 1]], 
                      c=[color], s=100, zorder=10, marker='*', 
                      edgecolors='white', linewidths=1.5, alpha=0.8)
            
            # Draw neighbor points with the same color
            neighbor_coords = sdf_points[neighbors[i]]
            ax.scatter(neighbor_coords[:, 0], neighbor_coords[:, 1],
                      c=[color]*len(neighbors[i]), s=40, zorder=9, marker='o',
                      edgecolors='white', linewidths=0.8, alpha=0.5)
            
            # Draw lines to neighbors with the same color
            for j in neighbors[i]:
                ax.plot([sdf_points[i, 0], sdf_points[j, 0]], 
                       [sdf_points[i, 1], sdf_points[j, 1]], 
                       color=color, linewidth=2, alpha=0.3, zorder=8)
        
        # Print statistics
        total_connections = sum(len(n) for n in neighbors)
        points_with_neighbors = sum(1 for n in neighbors if len(n) > 0)
        print(f"Points with neighbors: {points_with_neighbors}/{len(sdf_points)}")
        print(f"Points with arcs: {sum(1 for arcs in visible_arcs if len(arcs) > 0)}/{len(sdf_points)}")
        print(f"Total neighbor connections: {total_connections}")
        print(f"Average neighbors per point: {total_connections / len(sdf_points):.2f}")
        
        ax.set_title(f'All neighbor connections shown ({len(sdf_points)} points, {points_with_neighbors} with neighbors)')
        plt.show()
        return
    
    # Storage for highlighted elements (now lists to accumulate)
    highlighted = {'lines': [], 'scatter': [], 'neighbor_scatter': []}
    selected_points = set()  # Track selected point indices
    
    def select_point(i):
        """Programmatically select point i and highlight it."""
        if i in selected_points:
            return
        
        selected_points.add(i)
        
        print(f"\n=== Point {i} ===")
        print(f"Position: ({sdf_points[i, 0]:.3f}, {sdf_points[i, 1]:.3f})")
        print(f"SDF value: {sdf_values[i]:.3f}")
        print(f"Visible arcs: {visible_arcs[i]}")
        print(f"Neighbors ({len(neighbors[i])}): {neighbors[i]}")
        
        # Generate a unique color for this point
        colors = plt.cm.tab10(len(selected_points) % 10)
        
        # Highlight the selected point
        scatter = ax.scatter([sdf_points[i, 0]], [sdf_points[i, 1]], 
                            c=[colors], s=150, zorder=10, marker='*', 
                            edgecolors='white', linewidths=2, label=f'Point {i}')
        highlighted['scatter'].append(scatter)
        
        # Highlight neighbor points
        if len(neighbors[i]) > 0:
            neighbor_coords = sdf_points[neighbors[i]]
            neighbor_scatter = ax.scatter(neighbor_coords[:, 0], neighbor_coords[:, 1],
                                         c=[colors]*len(neighbors[i]), s=60, zorder=9, marker='o',
                                         edgecolors='white', linewidths=1, alpha=0.6)
            highlighted['neighbor_scatter'].append(neighbor_scatter)
        
        # Draw lines to neighbors
        for j in neighbors[i]:
            line, = ax.plot([sdf_points[i, 0], sdf_points[j, 0]], 
                           [sdf_points[i, 1], sdf_points[j, 1]], 
                           color=colors, linewidth=4, alpha=0.5, zorder=8)
            highlighted['lines'].append(line)
        
        ax.set_title(f'{len(selected_points)} point(s) selected | Left-click: add point | Right-click: clear all')
        ax.legend(loc='upper right', fontsize=8)

    def on_click(event):
        # Only respond to clicks inside the axes
        if event.inaxes != ax:
            return
        
        # Right click to clear all
        if event.button == 3:  # Right mouse button
            for line in highlighted['lines']:
                line.remove()
            highlighted['lines'] = []
            for scatter in highlighted['scatter']:
                scatter.remove()
            highlighted['scatter'] = []
            for scatter in highlighted['neighbor_scatter']:
                scatter.remove()
            highlighted['neighbor_scatter'] = []
            selected_points.clear()
            ax.set_title('All cleared! Left-click to select points, right-click to clear')
            ax.legend().remove() if ax.get_legend() else None
            fig.canvas.draw_idle()
            return
        
        # Find nearest point
        click_point = np.array([event.xdata, event.ydata])
        distances = np.linalg.norm(sdf_points - click_point, axis=1)
        i = np.argmin(distances)
        
        select_point(i)
        fig.canvas.draw_idle()
    
    # Connect the click event
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    # Auto-select degenerate arc points
    if not show_all:
        for i in degenerate_indices:
            select_point(i)
    
    ax.set_title(f'{len(degenerate_indices)} degenerate point(s) auto-selected | Click to add more | Right-click to clear')
    plt.show()

def test_subdividing(n=4):
    points, sdf_points, sdf_values = generate_2D_mesh(n=n, path_to_image='examples/horse.png')
    sdf_points_backup = sdf_points.copy()
    sdf_values_backup = sdf_values.copy()
    # Interpolate to a denser grid for better visualization
    inter_points = np.array([[x, y] for x in np.linspace(0, 1, 50) for y in np.linspace(0, 1, 50)])
    # Add more points to the part where the original sdf are smaller than 0.1 to better visualize the interior
    interior_points = sdf_points_backup[np.abs(sdf_values_backup) < 0.01]
    # Add more points around interior_points
    for p in interior_points:
        for dx in np.linspace(-0.02, 0.02, 5):
            for dy in np.linspace(-0.02, 0.02, 5):
                new_point = p + np.array([dx, dy])
                if 0 <= new_point[0] <= 1 and 0 <= new_point[1] <= 1:
                    inter_points = np.vstack([inter_points, new_point])
    inter_points = np.vstack([inter_points, sdf_points_backup])  # Ensure original points are included
    inter_points = np.unique(inter_points, axis=0)
    sdf_points = inter_points
    print(f"Original points: {len(sdf_points_backup)}, Interpolated points: {len(inter_points)}")
    sdf_values = va.interpolate_sdf(sdf_points_backup, sdf_values_backup, inter_points, method='bilinear')

    start_time = time.time()
    fig, ax = va.visualize_circles(sdf_points, sdf_values, points)
    end_time = time.time()
    print(f"visualization on {len(sdf_points)} points took {end_time - start_time:.4f} seconds")

    # Plot the original points
    ax.scatter(sdf_points_backup[:, 0], sdf_points_backup[:, 1], c='black', s=8, label='Original Points', zorder=11)
    
    def add_point(pt):
        """Add a single point: interpolate SDF, draw circle, update arrays."""
        nonlocal sdf_points, sdf_values
        
        new_sdf = va.interpolate_sdf(sdf_points_backup, sdf_values_backup,
                                      pt.reshape(1, -1), method='bilinear')[0]
        sdf_points = np.vstack([sdf_points, pt])
        sdf_values = np.append(sdf_values, new_sdf)
        
        from matplotlib.patches import Circle
        color = 'green' if new_sdf < 0 else 'orange'
        radius = np.abs(new_sdf)
        circle = Circle(pt, radius, facecolor=color, alpha=0.5, zorder=10)
        ax.add_patch(circle)
        ax.scatter([pt[0]], [pt[1]], c='cyan', s=5, zorder=13, edgecolors='blue', linewidths=1.5)
        return new_sdf
    
    def on_click(event):
        if event.inaxes != ax:
            return
        
        click_pt = np.array([event.xdata, event.ydata])
        
        if event.button == 1:  # Left click: add single point
            new_sdf = add_point(click_pt)
            ax.set_title(f'({click_pt[0]:.3f}, {click_pt[1]:.3f}) SDF={new_sdf:.4f} | Total: {len(sdf_points)}')
            print(f"Added ({click_pt[0]:.3f}, {click_pt[1]:.3f}), SDF={new_sdf:.4f}")
        
        elif event.button == 3:  # Right click: add 10 points nearby
            # Dynamic spread based on current view range
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            spread = min(xlim[1] - xlim[0], ylim[1] - ylim[0]) * 0.02
            
            for _ in range(10):
                offset = np.random.uniform(-spread, spread, size=2)
                new_pt = click_pt + offset
                add_point(new_pt)
            
            ax.set_title(f'Added 10 points near ({click_pt[0]:.3f}, {click_pt[1]:.3f}) | Total: {len(sdf_points)}')
            print(f"Added 10 points near ({click_pt[0]:.3f}, {click_pt[1]:.3f}), spread={spread:.4f}")
        
        fig.canvas.draw_idle()
    
    # Connect the click event
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.plot(points[:, 0], points[:, 1], 'b-', linewidth=2, label='Original Shape')
    
    ax.set_title('Click anywhere to add an interpolated SDF circle')
    plt.show()

def test_single_gradient(n=4, interpolator=None):
    # to_test_index is an index of a point near the medial axis
    points, sdf_points, sdf_values = generate_2D_mesh(n=n, path_to_image='examples/horse.png')
    if interpolator is None:
        # Create and fit the interpolator
        interpolator = Interpolator(kernel='cubic')
        interpolator.fit(sdf_points, sdf_values)
    
    # Setup figure and initial plot
    fig, ax = plt.subplots(figsize=(8, 7))
    grid_x, grid_y = np.mgrid[0:1:100j, 0:1:100j]
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
    grid_values = interpolator.predict(grid_points).reshape(100, 100)
    im = ax.imshow(grid_values.T, extent=(0, 1, 0, 1), origin='lower', cmap='viridis')
    plt.colorbar(im, ax=ax, label='Interpolated SDF Values')
    # draw the contours at multiple levels (no transpose for contour!)
    contour_levels = np.linspace(np.min(grid_values), np.max(grid_values), 10)
    # insert 0 level contour if not in levels
    if 0.0 not in contour_levels:
        contour_levels = np.sort( np.append( contour_levels, 0.0 ) )
    ax.contour(grid_x, grid_y, grid_values, levels=contour_levels, colors='yellow', linewidths=0.5, alpha=0.5)
    # overlay the original shape
    ax.scatter(sdf_points[:, 0], sdf_points[:, 1], c='r', s=3, label='Original Grid Points')
    ax.plot(points[:, 0], points[:, 1], 'b-', linewidth=2, label='Original Shape')
    ax.axis('equal')
    ax.legend(loc='upper right')
    
    # Storage for highlighted elements
    highlighted = {'point': None, 'surface': None, 'neighbors': None, 'texts': [], 'line': None}
    
    def update_display(to_test_index):
        # Clear previous highlights
        if highlighted['point'] is not None:
            highlighted['point'].remove()
        if highlighted['surface'] is not None:
            highlighted['surface'].remove()
        if highlighted['neighbors'] is not None:
            highlighted['neighbors'].remove()
        if highlighted['line'] is not None:
            highlighted['line'].remove()
        for txt in highlighted['texts']:
            txt.remove()
        highlighted['texts'] = []
        
        # Compute gradient for selected point
        # gradient, weights, indices = estimate_gradient_exhaustive(sdf_points, sdf_values, to_test_index)
        gradient, weights, indices = estimate_gradient_irls_single(sdf_points, sdf_values, to_test_index)
        point_on_surface = sdf_points[to_test_index] - sdf_values[to_test_index] * gradient
        
        # Highlight selected point
        highlighted['point'] = ax.scatter([sdf_points[to_test_index, 0]], [sdf_points[to_test_index, 1]], 
                                         c='cyan', s=150, zorder=10, marker='*', 
                                         edgecolors='white', linewidths=2, label=f'Selected Point {to_test_index}')
        
        # Show projected surface point
        highlighted['surface'] = ax.scatter([point_on_surface[0]], [point_on_surface[1]], 
                                           c='yellow', s=100, zorder=9, marker='o',
                                           edgecolors='white', linewidths=2, label='Projected Surface Point')
        
        # Show neighbors with weights
        neighbor_points = sdf_points[indices]
        highlighted['neighbors'] = ax.scatter(neighbor_points[:, 0], neighbor_points[:, 1], 
                                             c='green', s=50, zorder=8, label='Neighbors Used')
        for i in range(neighbor_points.shape[0]):
            txt = ax.text(neighbor_points[i, 0], neighbor_points[i, 1], f"{weights[i]:.2f}", 
                         color='black', fontsize=8, zorder=11)
            highlighted['texts'].append(txt)
        
        # Draw correspondence line
        highlighted['line'], = ax.plot([sdf_points[to_test_index, 0], point_on_surface[0]], 
                                       [sdf_points[to_test_index, 1], point_on_surface[1]], 
                                       'k--', linewidth=2, zorder=7)
        
        # Update title and legend
        ax.set_title(f'Point {to_test_index}: SDF={sdf_values[to_test_index]:.3f}, Gradient={gradient}')
        ax.legend(loc='upper right')
        fig.canvas.draw_idle()
        
        # Print info
        print(f"\n=== Point {to_test_index} ===")
        print(f"Position: ({sdf_points[to_test_index, 0]:.3f}, {sdf_points[to_test_index, 1]:.3f})")
        print(f"SDF value: {sdf_values[to_test_index]:.3f}")
        print(f"Gradient: {gradient}")
        print(f"Projected point: ({point_on_surface[0]:.3f}, {point_on_surface[1]:.3f})")
    
    def on_click(event):
        # Only respond to clicks inside the axes
        if event.inaxes != ax:
            return
        
        # Get click coordinates
        click_x, click_y = event.xdata, event.ydata
        click_point = np.array([click_x, click_y])
        
        # Find nearest point
        distances = np.linalg.norm(sdf_points - click_point, axis=1)
        to_test_index = np.argmin(distances)
        
        # Update display with new point
        update_display(to_test_index)
    
    # Connect the click event
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    ax.set_title('Click on any point to analyze its gradient')
    plt.show()

def test_interpolation_gradients(n=4, use_sample_gradient=False):
    points, sdf_points, sdf_values = generate_2D_mesh(n=n, path_to_image='examples/horse.png')
    # Create and fit the interpolator using ALL points
    interpolator = Interpolator(kernel='cubic')
    visible_arcs = va.compute_visible_arcs(sdf_points, sdf_values)
    radii = np.abs(sdf_values)
    degenerate_arcs = va.get_short_arcs(visible_arcs, tol=1e-8)
    for i, angle in degenerate_arcs.items():
        # For points with degenerate arcs, set gradient directly toward the angle
        grad = np.array([-np.cos(angle), -np.sin(angle)]) if sdf_values[i] > 0 else np.array([np.cos(angle), np.sin(angle)])
        print(f"Point {i} has degenerate arc with angle {angle:.2f} radians. Setting gradient to {grad}.")
        # Add a new point on the surface along this gradient direction
        new_point = sdf_points[i] - sdf_values[i] * grad
        sdf_points = np.vstack([sdf_points, new_point])
        sdf_values = np.append(sdf_values, 0)  # The SDF value at the projected point should be 0
    print(f"After adding points for degenerate arcs, total points: {len(sdf_points)}")
    interpolator.fit(sdf_points, sdf_values)
    
    # visualize the interpolated SDF and gradients
    grid_x, grid_y = np.mgrid[0:1:100j, 0:1:100j]
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
    grid_values = interpolator.predict(grid_points).reshape(100, 100)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(grid_values.T, extent=(0, 1, 0, 1), origin='lower', cmap='viridis')
    plt.colorbar(im, ax=ax, label='Interpolated SDF Values')
    ax.contour(grid_x, grid_y, grid_values, levels=10, colors='yellow', linewidths=0.5, alpha=0.5)
    ax.scatter(sdf_points[:, 0], sdf_points[:, 1], c='r', s=3, label='Original Grid Points')
    ax.plot(points[:, 0], points[:, 1], 'b-', linewidth=2, label='Original Shape')
    ax.set_aspect('equal')
    ax.set_title('Drag a rectangle to select points for interpolation+gradient')
    # visualize visible arcs just like in test_visible_neighbors
    for i in range(len(visible_arcs)):
        # draw arc as a circle for simplicity
        center = sdf_points[i]
        radius = np.abs(sdf_values[i])
        color = '#FF6B9D' if sdf_values[i] < 0 else "#4ECDC5"
        circle = plt.Circle(center, radius, color=color, fill=True, alpha=1, linewidth=1)
        ax.add_patch(circle)
    # State for rectangle selection and drawn elements
    # mode: 'select' = drag rectangle, 'click' = click to inspect gradient
    state = {'press': None, 'rect': None, 'artists': [], 'click_artists': [],
             'local_interp': None, 'sel_idx': None, 'mode': 'select'}

    def clear_artists():
        for a in state['artists']:
            a.remove()
        state['artists'] = []

    def clear_click_artists():
        for a in state['click_artists']:
            a.remove()
        state['click_artists'] = []

    def on_press(event):
        if event.inaxes != ax:
            return
        # Right click: reset to selection mode
        if event.button == 3:
            clear_artists()
            clear_click_artists()
            if state['rect'] is not None:
                state['rect'].remove()
                state['rect'] = None
            state['local_interp'] = None
            state['sel_idx'] = None
            state['mode'] = 'select'
            ax.set_title('Drag a rectangle to select points for interpolation+gradient')
            fig.canvas.draw_idle()
            return
        if event.button != 1:
            return
        # In click mode: handle point click
        if state['mode'] == 'click':
            handle_click(event)
            return
        # In select mode: start rectangle drag
        state['press'] = (event.xdata, event.ydata)
        if state['rect'] is not None:
            state['rect'].remove()
            state['rect'] = None

    def handle_click(event):
        """Click a point to compute sample_best_gradient and show projection."""
        click_pt = np.array([event.xdata, event.ydata])
        distances_to_click = np.linalg.norm(sdf_points - click_pt, axis=1)
        idx = np.argmin(distances_to_click)

        local_interp = state['local_interp']
        pt = sdf_points[idx]
        sdf_val = sdf_values[idx]

        # Call sample_best_gradient
        best_grad = local_interp.sample_best_gradient(pt, sdf_val, num_samples=36)
        surface_pt = pt - sdf_val * best_grad

        # Also get the predict_gradient result for comparison
        pred_grad = local_interp.predict_gradient(pt.reshape(1, -1))[0]
        pred_grad /= np.linalg.norm(pred_grad) + 1e-8
        pred_surface = pt - sdf_val * pred_grad

        # Clear previous click highlights
        clear_click_artists()

        # Highlight the clicked point
        s = ax.scatter([pt[0]], [pt[1]], c='cyan', s=150, zorder=20, marker='*',
                       edgecolors='white', linewidths=2)
        state['click_artists'].append(s)

        # Draw sample_best_gradient result (yellow)
        s2 = ax.scatter([surface_pt[0]], [surface_pt[1]], c='yellow', s=80, zorder=19,
                        marker='o', edgecolors='white', linewidths=2, label='sample_best')
        state['click_artists'].append(s2)
        ln, = ax.plot([pt[0], surface_pt[0]], [pt[1], surface_pt[1]],
                      'y-', linewidth=2, zorder=18)
        state['click_artists'].append(ln)

        # Draw predict_gradient result (magenta) for comparison
        s3 = ax.scatter([pred_surface[0]], [pred_surface[1]], c='magenta', s=60, zorder=19,
                        marker='D', edgecolors='white', linewidths=1.5, label='predict_grad')
        state['click_artists'].append(s3)
        ln2, = ax.plot([pt[0], pred_surface[0]], [pt[1], pred_surface[1]],
                       'm--', linewidth=1.5, zorder=18)
        state['click_artists'].append(ln2)

        # Draw the sampled directions as thin lines
        angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)
        directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        samples = pt - sdf_val * directions
        preds = local_interp.predict(samples)
        # Color by |predicted SDF| — smaller = better
        cmap = plt.cm.RdYlGn_r
        abs_preds = np.abs(preds)
        norm = plt.Normalize(vmin=abs_preds.min(), vmax=abs_preds.max())
        for k in range(len(angles)):
            color = cmap(norm(abs_preds[k]))
            dot = ax.scatter([samples[k, 0]], [samples[k, 1]], c=[color], s=15, zorder=17,
                             edgecolors='gray', linewidths=0.5)
            state['click_artists'].append(dot)

        pred_at_surface = local_interp.predict(surface_pt.reshape(1, -1))[0]
        ax.set_title(f'Point {idx}: SDF={sdf_val:.4f} | best_grad={best_grad} | pred@surface={pred_at_surface:.6f}')
        ax.legend(loc='upper right', fontsize=8)
        fig.canvas.draw_idle()

    def on_motion(event):
        if state['press'] is None or event.inaxes != ax or state['mode'] != 'select':
            return
        x0, y0 = state['press']
        x1, y1 = event.xdata, event.ydata
        if state['rect'] is not None:
            state['rect'].remove()
        from matplotlib.patches import Rectangle
        w, h = x1 - x0, y1 - y0
        state['rect'] = ax.add_patch(Rectangle((x0, y0), w, h,
                                                linewidth=2, edgecolor='white',
                                                facecolor='white', alpha=0.15, linestyle='--'))
        fig.canvas.draw_idle()

    def on_release(event):
        if state['press'] is None or event.inaxes != ax or event.button != 1 or state['mode'] != 'select':
            state['press'] = None
            return
        x0, y0 = state['press']
        x1, y1 = event.xdata, event.ydata
        state['press'] = None

        # Compute bounding box
        xmin, xmax = min(x0, x1), max(x0, x1)
        ymin, ymax = min(y0, y1), max(y0, y1)

        # If too small, treat as click — skip
        if (xmax - xmin) < 1e-4 or (ymax - ymin) < 1e-4:
            return

        # Find points inside the rectangle
        mask = ((sdf_points[:, 0] >= xmin) & (sdf_points[:, 0] <= xmax) &
                (sdf_points[:, 1] >= ymin) & (sdf_points[:, 1] <= ymax))
        sel_idx = np.where(mask)[0]
        if len(sel_idx) < 2:
            ax.set_title(f'Only {len(sel_idx)} point(s) selected — need at least 2')
            fig.canvas.draw_idle()
            return

        # Clear previous results
        clear_artists()
        clear_click_artists()

        # Fit interpolator on selected points only
        sel_points = sdf_points[sel_idx]
        sel_values = sdf_values[sel_idx]
        local_interp = Interpolator(kernel='cubic')
        local_interp.fit(sel_points, sel_values)
        state['local_interp'] = local_interp
        state['sel_idx'] = sel_idx

        # Compute gradients and project to surface
        if use_sample_gradient:
            sel_gradients = np.array([
                local_interp.sample_best_gradient(sel_points[i], sel_values[i], num_samples=36)
                for i in range(len(sel_points))
            ])
        else:
            sel_gradients = local_interp.predict_gradient(sel_points)
            sel_gradients /= np.linalg.norm(sel_gradients, axis=1, keepdims=True) + 1e-8
        sel_surface = sel_points - sel_values[:, np.newaxis] * sel_gradients

        # Draw selected points highlighted
        s1 = ax.scatter(sel_points[:, 0], sel_points[:, 1], c='lime', s=30, zorder=12,
                        edgecolors='white', linewidths=1, label='Selected Points')
        state['artists'].append(s1)

        # Draw projected surface points
        s2 = ax.scatter(sel_surface[:, 0], sel_surface[:, 1], c='cyan', s=20, zorder=11,
                        label='Projected Surface')
        state['artists'].append(s2)

        # Draw correspondence lines
        for i in range(len(sel_idx)):
            ln, = ax.plot([sel_points[i, 0], sel_surface[i, 0]],
                          [sel_points[i, 1], sel_surface[i, 1]],
                          'w--', linewidth=0.8, zorder=10)
            state['artists'].append(ln)

        # Draw local interpolation contour over the full domain
        local_grid_x, local_grid_y = np.mgrid[0:1:200j, 0:1:200j]
        local_grid_pts = np.vstack([local_grid_x.ravel(), local_grid_y.ravel()]).T
        local_grid_vals = local_interp.predict(local_grid_pts).reshape(200, 200)
        cs = ax.contour(local_grid_x, local_grid_y, local_grid_vals, levels=[0.0],
                        colors='red', linewidths=2, zorder=13)
        # Track the entire ContourSet (works in all matplotlib versions)
        state['artists'].append(cs)

        # Switch to click mode
        state['mode'] = 'click'
        grad_method = 'sample_best' if use_sample_gradient else 'predict_grad'
        ax.set_title(f'{len(sel_idx)} points selected ({grad_method}) — click to inspect | right-click to reset')
        ax.legend(loc='upper right', fontsize=8)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)
    plt.show()

if __name__ == "__main__":
    # test_visible_neighbors(30, show_all=True)  # show all neighbor connections
    # test_visible_neighbors(30)  # interactive neighbor inspection
    # table = []
    # for n in [10, 20, 25, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50]:
    #     mc, psr, rfta = test_gradient_estimation(n, NeighborEstimation.SPATIAL, GradientEstimation.IRLS, see_arcs=True)
    #     table.append((n, mc, psr, rfta))
    # for n, mc, psr, rfta in table:
    #     print(f"{psr:.4f}")
    # test_single_gradient(30)
    # test_subdividing(30)
    # test_interpolation_gradients(30, use_sample_gradient=True)
    test_gradient_estimation(30, NeighborEstimation.SPATIAL, GradientEstimation.CurlFree_OPT, see_arcs=False, clamp_gradients=False)

