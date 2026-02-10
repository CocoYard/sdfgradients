from scipy.spatial import cKDTree
import gpytoolbox as gpy
import igl
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
import visible_arcs as va
import iterative_projection as ip
from enum import Enum

class NeighborEstimation(Enum):
    VISIBLE_CONNECTIVITY = 'visible_connectivity'
    SPATIAL = 'spatial'

class GradientEstimation(Enum):
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

class Interpolator:
    """
    A Duchon interpolator to fit and predict values based on input signed distance data.
    """
    def __init__(self, kernel):
        """
        Initialize the Duchon interpolation object with a specified radial basis function kernel.
        Parameters
        ----------
        kernel : str
            The type of radial basis function to use. Supported options are:
            - 'thin_plate': Uses r^2 * log(r) as the kernel function
            - Other values: Defaults to cubic kernel (r^3)
        Attributes
        ----------
        points : None
            Will store the interpolation points (initialized as None)
        values : None
            Will store the values at interpolation points (initialized as None)
        alpha : None
            Will store the interpolation coefficients (initialized as None)
        p : None
            Will store polynomial coefficients (initialized as None)
        q : None
            Will store additional coefficients (initialized as None)
        kernel : callable
            The radial basis function used for interpolation
        """
        self.points = None
        self.values = None
        self.alpha = None
        self.p = None
        self.q = None
        if kernel == 'thin_plate':
            self.kernel = lambda r: r**2 * np.log(r + 1e-10)  # Adding a small value to avoid log(0)
        else:
            self.kernel = lambda r: r**3  # Default to cubic kernel

    def fit(self, points, values):
        """
        Fit the interpolator with given points and their corresponding values.

        Parameters:
        points (np.ndarray): An array of shape (n_samples, m_dimensions) representing the input points.
        values (np.ndarray): An array of shape (n_samples,) representing the values at the input points.
        """
        self.points = points
        self.values = values
        self.alpha, self.p, self.q = self._compute_coefficients(points, values)

    def _compute_coefficients(self, points, values):
        """
        Compute the coefficients for the Duchon interpolation based on the input points and values.
        Parameters:
        points (np.ndarray): An array of shape (n_samples, m_dimensions) representing the input points.
        values (np.ndarray): An array of shape (n_samples,) representing the values at the input points.
        
        Returns:
        tuple: A tuple containing the coefficients for the radial basis functions and polynomial terms.
        """
        # construct the interpolation matrix
        n_samples = points.shape[0]
        m_dimensions = points.shape[1]
        K = np.zeros((n_samples + m_dimensions + 1, n_samples + m_dimensions + 1))
        for i in range(n_samples):
            for j in range(n_samples):
                r = np.linalg.norm(points[i] - points[j])
                if r == 0:
                    K[i, j] = 0
                else:
                    K[i, j] = self.kernel(r)
        # Add polynomial terms for Duchon interpolation
        P = np.ones((n_samples, m_dimensions + 1))
        P[:, :-1] = points
        K[:n_samples, n_samples:] = P
        K[n_samples:, :n_samples] = P.T
        y = np.zeros(n_samples + m_dimensions + 1)
        y[:n_samples] = values
        # Solve for coefficients
        coefficients = np.linalg.solve(K, y)
        return coefficients[:n_samples], coefficients[n_samples:-1], coefficients[-1]

    def predict(self, x_new):
        """
        Predict values at new input points using the fitted interpolator. Duchon interpolation multiplies all basis
        functions by a coefficient term. The basis functions are radial basis functions that depend on the distance
        between points.

        Parameters:
        x_new (np.ndarray): An array of shape (m_samples, dimensions) representing the new input points.

        Returns:
        np.ndarray: An array of shape (m_samples,) representing the predicted values at the new points.
        """
        n_samples = self.points.shape[0]
        m_samples = x_new.shape[0]
        distances = cdist(x_new, self.points, metric='euclidean')
        r = self.kernel(distances)  # Apply kernel to all distances at once
        return r @ self.alpha + x_new @ self.p + self.q
    
    def predict_gradient(self, x_new):
        """
        Predict gradients at new input points using the fitted interpolator.

        Parameters:
        x_new (np.ndarray): An array of shape (m_samples, dimensions) representing the new input points.

        Returns:
        np.ndarray: An array of shape (m_samples, dimensions) representing the predicted gradients at the new points.
        """
        n_samples = self.points.shape[0]
        m_samples = x_new.shape[0]
        dimensions = self.points.shape[1]
        gradients = np.zeros((m_samples, dimensions))
        for i in range(n_samples):
            diff = x_new - self.points[i]  # Shape (m_samples, dimensions)
            dist = np.linalg.norm(diff, axis=1, keepdims=True)  # Shape (m_samples, 1)
            # Avoid division by zero
            dist[dist == 0] = 1e-10
            if self.kernel == 'thin_plate':
                coeff = self.alpha[i] * (2 * np.log(dist) + 1) * diff  # Derivative of thin-plate spline kernel
            else:
                coeff = self.alpha[i] * 3 * dist * diff  # Derivative of cubic kernel
            gradients += coeff
        gradients += self.p  # Add polynomial term gradient
        return gradients
    
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
    image = io.imread(path_to_image)
    # image shape = (H, W, 4)
    gray_image = color.rgb2gray(image[:, :, :3])  # Convert to grayscale
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

def yongs_algorithm2( points, distances, vertices ):
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
    return new_points, gradients

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

def estimate_gradient_lstsq(points, sdf_values, neighbors=None):
    """ estimation of gradients from SDF values at given points. LSTSQ is the same as 
    to Prewitt finite difference (equal weights) if neighbors are the 8-connected grid 
    neighbors."""
    indices = neighbors
    gradients = np.zeros_like(points)
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
            grad, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)
        gradients[i] = grad
    # Normalize gradients
    norms = np.linalg.norm(gradients, axis=1, keepdims=True)
    gradients /= np.maximum(norms, 1e-8)
    return gradients

def estimate_gradient_finite_diff(points, sdf_values):
    """ estimation of gradients from SDF values at given points. This finite difference 
    uses central difference with Sobel operator (smaller weights on diagonals). The neighbors 
    are the 8-connected grid neighbors.
    """
    from scipy.ndimage import sobel
    # 1. Reshape to grid
    sdf_grid = sdf_values.reshape(int(points.shape[0]**0.5), int(points.shape[0]**0.5))
    sdf_padded = np.pad(sdf_grid, pad_width=1, mode='edge')
    
    # 2. Apply Sobel on each axis
    # sdf_grid shape: (n_x, n_y), where axis=0 is x, axis=1 is y
    # sobel(axis=0) differentiates along x, sobel(axis=1) differentiates along y
    # Assume uniform grid spacing
    dx = points[1, 1] - points[0, 1]
    grad_x_grid = sobel(sdf_padded, axis=0, mode='nearest') / (8 * dx)
    grad_y_grid = sobel(sdf_padded, axis=1, mode='nearest') / (8 * dx)
    # Remove padding
    grad_x_grid = grad_x_grid[1:-1, 1:-1]
    grad_y_grid = grad_y_grid[1:-1, 1:-1]
    
    # 3. Flatten back to (N, 2)
    grad_x = grad_x_grid.flatten()
    grad_y = grad_y_grid.flatten()
    
    # Combine
    gradients = np.stack([grad_x, grad_y], axis=1) # (N, 2)
    # Normalize gradients
    norms = np.linalg.norm(gradients, axis=1, keepdims=True)
    gradients /= np.maximum(norms, 1e-8)
    return gradients

def estimate_gradient_irls(points, sdf_values, neighbor_list=None, iters=5, sigma=0.05):
    """
    Estimate gradients using a pre-defined list of irregular neighbors.
    
    Args:
        points: (N, 2) coordinates.
        sdf_values: (N,) SDF values.
        neighbor_list: List of lists (your irregular indices).
        iters: IRLS iterations.
        sigma: Gaussian falloff parameter.
    """
    n = points.shape[0]
    if neighbor_list is None:
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=9, algorithm='auto').fit(points)
        distances, indices = nbrs.kneighbors(points)
        neighbor_list = [list(idx) for idx in indices]
    # 1. Padding: Find the max number of neighbors
    # Use 0 as a placeholder index for padding
    lengths = [len(sublist) for sublist in neighbor_list]
    k_max = max(lengths) if lengths else 0
    
    if k_max == 0:
        return np.zeros((n, 2))

    # Create a padded index matrix (N, k_max)
    # Default to 0, we will mask these out later
    padded_idx = np.zeros((n, k_max), dtype=int)
    mask = np.zeros((n, k_max), dtype=bool)
    
    for i, sublist in enumerate(neighbor_list):
        if len(sublist) > 0:
            padded_idx[i, :len(sublist)] = sublist
            mask[i, :len(sublist)] = True
            
    # 2. Extract neighbor data using the padded indices
    # pts_neighbors: (N, k_max, 2)
    pts_neighbors = points[padded_idx]
    # sdf_neighbors: (N, k_max)
    sdf_neighbors = sdf_values[padded_idx]
    
    # 3. System Matrix [x, y, 1]
    A = np.concatenate([pts_neighbors, np.ones((n, k_max, 1))], axis=2)
    
    # Initialize weights: 1.0 for real neighbors, 0.0 for padded ones
    weights = np.zeros((n, k_max))
    weights[mask] = 1.0
    
    best_grads = np.zeros((n, 2))

    for _ in range(iters):
        # Apply weights
        w_sqrt = np.sqrt(weights)[:, :, None]
        WA = A * w_sqrt
        Wb = (sdf_neighbors * w_sqrt[:, :, 0])[:, :, None]
        
        # Normal Equations
        ATWA = np.einsum('nki,nkj->nij', WA, WA)
        ATWb = np.einsum('nki,nkj->nij', WA, Wb)
        
        # To avoid singular matrices for points with 0 or 1 neighbors, 
        # add a small identity ridge to ATWA
        ridge = np.eye(3) * 1e-6
        ATWA += ridge[None, :, :]
        
        try:
            models = np.linalg.solve(ATWA, ATWb).squeeze(-1)
            grads = models[:, :2]
            biases = models[:, 2]
            
            # 4. Update Weights
            preds = np.einsum('nkj,nj->nk', pts_neighbors, grads) + biases[:, None]
            residuals = np.abs(preds - sdf_neighbors)
            
            # Only update weights for 'True' neighbors in the mask
            new_weights = np.exp(-(residuals**2) / (2 * (sigma**2)))
            weights = np.where(mask, new_weights, 0.0)
            
            best_grads = grads
            
        except np.linalg.LinAlgError:
            continue

    # Final Unit Normalization
    mags = np.linalg.norm(best_grads, axis=1, keepdims=True)
    mags = np.where(mags < 1e-8, 1.0, mags)
    return best_grads / mags

# def estimate_gradient_irls_single(points, sdf_values, ind, neighbors=None, iters=50, sigma=0.05):
#     """ estimation of gradients from SDF values at a given point using IRLS. Only for a single point at index ind."""
#     point = points[ind]
#     if neighbors is None:
#         from sklearn.neighbors import NearestNeighbors
#         nbrs = NearestNeighbors(n_neighbors=9, algorithm='auto').fit(points)
#         distance, indices = nbrs.kneighbors(point.reshape(1, -1))
#     else:
#         indices = neighbors
#     gradient = np.zeros_like(point)
#     neighbor_points = points[indices[0]]
#     neighbor_sdf = sdf_values[indices[0]]
#     diffs = np.c_[neighbor_points - point, np.ones(neighbor_points.shape[0])]
#     sdf_diffs = neighbor_sdf - sdf_values[ind]
#     A = diffs
#     b = sdf_diffs
#     print(diffs.shape, sdf_diffs.shape)
#     gradient = np.linalg.lstsq(A, b, rcond=None)[0]
#     for _ in range(iters):
#         preds = diffs @ gradient
#         residuals = np.abs(preds - sdf_diffs)
#         # print("residuals:", np.sum(residuals**2))
#         weights = np.exp(-(residuals**2) / (2 * (sigma**2)))
#         W = np.diag(weights)
#         # Weighted least squares
#         Aw = W @ A
#         bw = W @ b
#         gradient, residual, rank, s = np.linalg.lstsq(Aw, bw, rcond=None)
#         print("residual:", residual)
#     # gradient = _single_point_ransac_2d(neighbor_points, neighbor_sdf, 0.02)
#     # Normalize gradients
#     gradient = gradient[:2]
#     norm = np.linalg.norm(gradient)
#     gradient /= max(norm, 1e-8)
#     return gradient, weights, indices
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
import numpy as np
import itertools
from sklearn.neighbors import NearestNeighbors

def estimate_gradient_exhaustive(points, sdf_values, ind, neighbors=None):
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
        return grad, weights, indices
    # If not enough neighbors to form a pair, return zero gradient
    if n_neighbors < dim:
        return np.zeros(dim), np.zeros(n_neighbors), indices

    best_loss = float('inf')
    best_gradient = np.zeros(dim)
    best_subset_idx = [] # To store the indices of the "winning" pair
    
    # 2. Iterate through all combinations of 'dim' neighbors (Pairs in 2D)
    # subset_idx contains the local indices (0 to n_neighbors-1) of the chosen pair
    for subset_idx in itertools.combinations(range(n_neighbors), dim):
        # Extract the subset of points
        A_sub = A_local[list(subset_idx)]
        b_sub = b_local[list(subset_idx)]
        
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
                except np.linalg.LinAlgError:
                    continue
        # Check Gradient Validity (SDF property: norm should be approx 1)
        # We allow a loose tolerance to accept imperfect but reasonable gradients
        norm = np.linalg.norm(cand_grad)
        if not (0.5 < norm < 1.5):
            continue
            
        # 3. Validation: Evaluate this candidate gradient against ALL neighbors
        preds = A_local @ cand_grad
        residuals = np.abs(preds - b_local)
        
        # LMS Metric: Use Median of Residuals to be robust against 50% outliers
        loss = np.median(residuals)
        
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
    return best_gradient, weights, indices

def estimate_gradient_irls_single(points, sdf_values, ind, neighbors=None, iters=10, sigma=0.05):
    """
    Estimation of gradients from SDF values at a given point using IRLS.
    Only for a single point at index `ind`.
    
    Args:
        points: (N, D) array of points.
        sdf_values: (N,) array of SDF values.
        ind: Index of the point to estimate.
        neighbors: Optional list of neighbor indices. If None, KNN is used.
        iters: Number of IRLS iterations.
        sigma: Gaussian bandwidth for weighting.
        
    Returns:
        gradient: (D,) Estimated gradient vector.
        weights: (K,) Final weights of the neighbors.
        indices: (K,) Indices of the neighbors used.
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
        indices = np.array(neighbors)
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

    # 3. Initial Least Squares (OLS)
    # Solve Ax = b
    gradient, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    weights = np.ones(len(indices))

    # 4. Iterative Re-weighted Least Squares (IRLS)
    for i in range(iters):
        # Calculate residuals: |predicted_delta - actual_delta|
        preds = A @ gradient
        residuals = np.abs(preds - b)
        
        # Gaussian weighting: closer fits get higher weights
        # Note: We add a small epsilon to sigma to prevent division by zero if sigma is too small
        weights = np.exp(-(residuals**2) / (2 * (sigma**2)))
        
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

    return gradient, weights, indices

def estimate_gradient_RANSAC(points, sdf_values, neighbors=None):
    """ estimation of gradients from SDF values at given points.
    Inputs:
        points: (N, 2) coordinates of sample points
        sdf_values: (N,) corresponding SDF values
        k: number of neighbors to find for each point for RANSAC (default is 3×3 neighborhood)
    Returns:
        gradients: (N, 2) robust gradients for each point
    """
    gradients = np.zeros_like(points)
    for i in range(points.shape[0]):
        gradients[i], _, _ = estimate_gradient_exhaustive(points, sdf_values, i, neighbors=neighbors[i] if neighbors is not None else None)
    return gradients

def _single_point_ransac_2d(pts, sdfs, threshold, max_iters=100):
    n = pts.shape[0]
    best_grad = np.array([0.0, 0.0])
    max_inliers = -1
    min_gradient_bias = float('inf')
    # RANSAC core: random sampling to compute local plane
    for _ in range(max_iters):
        # 2D linear fitting requires at least 3 points
        idx = np.random.choice(n, 3, replace=False)
        A = np.c_[pts[idx], np.ones(3)]
        try:
            # Solve gx*x + gy*y + b = sdf
            m = np.linalg.lstsq(A, sdfs[idx], rcond=None)[0]
            g_hypo = m[:2]
            b_hypo = m[2]
            
            # Statistical consistency
            res = np.abs(pts @ g_hypo + b_hypo - sdfs)
            inliers = np.sum(res < threshold)
            
            # Add physical constraint: SDF gradient magnitude should not be too far from 1
            mag = np.linalg.norm(g_hypo)
            if abs(mag - 1) < min_gradient_bias:
                min_gradient_bias = abs(mag - 1)
                best_grad = g_hypo / mag # Normalize
        except:
            continue
            
    # If RANSAC doesn't find good results, fall back to ordinary least squares
    # if max_inliers < 0:
    #     A_all = np.c_[pts, np.ones(n)]
    #     m_all = np.linalg.lstsq(A_all, sdfs, rcond=None)[0]
    #     g_all = m_all[:2]
    #     mag_all = np.linalg.norm(g_all)
    #     best_grad = g_all / mag_all if mag_all > 1e-6 else g_all
    print("Best grad bias:", min_gradient_bias)
    return best_grad

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

def marching_cubes_2D(sdf_points, sdf_values, grid_size=100):
    """ Marching squares to extract 0-level contour from 2D SDF values."""
    from skimage import measure
    # Create a grid for marching squares
    grid_x, grid_y = np.mgrid[0:1:grid_size*1j, 0:1:grid_size*1j]
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
    # Interpolate SDF values on the grid
    from scipy.interpolate import griddata
    grid_values = griddata(sdf_points, sdf_values, grid_points, method='cubic').reshape(grid_size, grid_size)
    # Extract contours at level 0
    contours = measure.find_contours(grid_values, level=0.0)
    # Convert contour coordinates back to original scale
    contour_points = []
    for contour in contours:
        contour_points.append(np.column_stack((contour[:, 1] / (grid_size - 1), contour[:, 0] / (grid_size - 1))))
    # swap x and y
    contour_points = [cp[:, ::-1] for cp in contour_points]
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

def test_gradient_estimation(n, neighbor_estimation: NeighborEstimation, gradient_estimation: GradientEstimation, interpolator=None, on_gradient_neighbors=True):
    points, sdf_points, sdf_values = generate_2D_mesh(n=n, path_to_image='examples/horse.png')
    if interpolator is None:
        # Create and fit the interpolator
        interpolator = Interpolator(kernel='cubic')
        interpolator.fit(sdf_points, sdf_values)

    visible_arcs = va.compute_visible_arcs(sdf_points, sdf_values)
    radii = np.abs(sdf_values)
    if neighbor_estimation == NeighborEstimation.VISIBLE_CONNECTIVITY:
        neighbors = va.find_arcs_neighbors(sdf_points, radii, visible_arcs, 1e-3)
    elif neighbor_estimation == NeighborEstimation.SPATIAL:
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=9, algorithm='auto').fit(sdf_points)
        distances, neighbors = nbrs.kneighbors(sdf_points)
        neighbors = {i: list(neighbors[i]) for i in range(sdf_points.shape[0])}
    if on_gradient_neighbors:
        neighbors2 = neighbors_on_gradient(sdf_points, sdf_values, tol=1e-5)
        for k, v in neighbors2.items():
            neighbors[k] = v
        ids = set() # To plot the points in neighbors2 with different color, we need to collect their indices
        for k, v in neighbors2.items():
            ids.add(k)
            for idx in v:
                ids.add(idx)
        ids = list(ids)
        print(len(neighbors2))
    if gradient_estimation == GradientEstimation.IRLS:
        gradients = estimate_gradient_irls(sdf_points, sdf_values, neighbors)
    elif gradient_estimation == GradientEstimation.RANSAC:
        gradients = estimate_gradient_RANSAC(sdf_points, sdf_values, neighbors)
    elif gradient_estimation == GradientEstimation.FINITE:
        neighbor_estimation = NeighborEstimation.SPATIAL
        gradients = estimate_gradient_finite_diff(sdf_points, sdf_values)
    elif gradient_estimation == GradientEstimation.LSTSQ:
        gradients = estimate_gradient_lstsq(sdf_points, sdf_values, neighbors)

    print("Estimated gradients shape:", gradients.shape)
    points_on_surface = yongs_algorithm(sdf_points, sdf_values, gradients)
    # points_on_surface, gradients = yongs_algorithm2(sdf_points, sdf_values, points)
    wrong_count = rate_gradient_estimation(sdf_points, points_on_surface, sdf_values, tol=1e-3)
    mask = wrong_count <= 0
    mask = wrong_count >= 0
    good_sdf_points = sdf_points[mask]
    points_on_surface_wrong = points_on_surface[~mask]
    good_points_on_surface = points_on_surface[mask]
    good_gradients = gradients[mask]
    contour_points = marching_cubes_2D(sdf_points, sdf_values)[0]

    # Vr, Er = gpy.reach_for_the_arcs(sdf_points, sdf_values)
    # rfta_contour = obj_to_points(Vr, Er)[0]

    V, E = gpy.point_cloud_to_mesh( good_points_on_surface, good_gradients,
    method='PSR',
    psr_screening_weight=10.0,
    psr_outer_boundary_type="Neumann",
    )
    poisson_contour = obj_to_points(V, E)[0]

    # visualize results in 2D as heatmap
    # set the size of the figure to be just enough to hold the heatmap
    fig, ax = plt.subplots(figsize=(8, 7))
    grid_x, grid_y = np.mgrid[0:1:100j, 0:1:100j]
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
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
        # show all points in neighbors2 with white color
        plt.scatter(sdf_points[ids, 0], sdf_points[ids, 1], c='white', s=10, label='on-gradient Points')

    plt.plot(points[:, 0], points[:, 1], 'b-', linewidth=2, label='Original Shape')
    plt.plot(contour_points[:, 0], contour_points[:, 1], 'k', linewidth=2, label='MC Contour')
    plt.plot(poisson_contour[:, 0], poisson_contour[:, 1], 'm', linewidth=2, label='PSR Contour')

    plot_correspondence(good_sdf_points, good_points_on_surface, plt)
    # connect original points to projected points
    if on_gradient_neighbors:
        plot_correspondence(sdf_points[ids], points_on_surface[ids], plt, color='w')
    ax.set_aspect('equal')
    ax.set_xlim(-0.01, 1.1)
    ax.set_ylim(-0.01, 1.2)
    plt.title('Gradient Estimation and Surface Point Projection' + ' (' + str(n) + '^2 points ' + neighbor_estimation.value + ' + ' + gradient_estimation.value + ')')
    plt.legend(loc='upper right')
    plt.show()

def test_visible_neighbors(n=4, show_all=False):
    points, sdf_points, sdf_values = generate_2D_mesh(n=n, path_to_image='examples/horse.png')
    visible_arcs = va.compute_visible_arcs(sdf_points, sdf_values)
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
        print(f"Total neighbor connections: {total_connections}")
        print(f"Average neighbors per point: {total_connections / len(sdf_points):.2f}")
        
        ax.set_title(f'All neighbor connections shown ({len(sdf_points)} points, {points_with_neighbors} with neighbors)')
        plt.show()
        return
    
    # Storage for highlighted elements (now lists to accumulate)
    highlighted = {'lines': [], 'scatter': [], 'neighbor_scatter': []}
    selected_points = set()  # Track selected point indices
    
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
        
        # Get click coordinates
        click_x, click_y = event.xdata, event.ydata
        click_point = np.array([click_x, click_y])
        
        # Find nearest point
        distances = np.linalg.norm(sdf_points - click_point, axis=1)
        i = np.argmin(distances)
        
        # Skip if already selected
        if i in selected_points:
            print(f"Point {i} already selected")
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
        fig.canvas.draw_idle()
    
    # Connect the click event
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    ax.set_title('Click on any point to show its neighbors')
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

    import time
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

if __name__ == "__main__":
    # test_visible_neighbors(30, show_all=True)  # show all neighbor connections
    # test_visible_neighbors(30)  # interactive neighbor inspection
    test_gradient_estimation(30, neighbor_estimation=NeighborEstimation.SPATIAL, gradient_estimation=GradientEstimation.RANSAC)
    # test_single_gradient(30)
    # test_subdividing(30)
