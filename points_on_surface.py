# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "gpytoolbox",
#     "numpy",
#     "plot2gltf",
#     "trimesh[easy]",
# ]
# ///
import numpy as np
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

def generate_test_sphere( num_points=1000, radius=1.0, dimension=3 ):
    '''
    Generates random points around a sphere and computes their signed distances and gradients.
    Parameters:
    num_points: int
        The number of random points to generate.
    radius: float
        The radius of the sphere.
    Returns:
    points: (num_points, 3) array of point coordinates
        The generated random points in 3D space.
    distances: (num_points,) array of signed distance values
        The signed distance values for each point.
    gradients: (num_points, 3) array of gradient vectors
        The gradient vectors at each point.
    dimension: int
        The dimension of the space (default is 3 for 3D).
    '''
    
    # Generate random points in 3D space
    points = np.random.uniform(-2*radius, 2*radius, (num_points, dimension))
    
    # Compute signed distances from the sphere surface
    distances = np.linalg.norm(points, axis=1) - radius
    
    # Compute gradients (normalized position vectors)
    gradients = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    return points, distances, gradients

def generate_test_mesh_data( path_to_mesh, num_points=500 ):
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
    query = trimesh.proximity.ProximityQuery(mesh)
    closest, distances, _ = query.on_surface( points )
    gradients = points - closest
    # Normalize gradients
    norm_temp = np.linalg.norm(gradients, axis=1, keepdims=True)
    # Avoid division by zero
    norm_temp[np.abs(norm_temp) <= 1e-8] = 1.0
    gradients /= norm_temp

    # Filter out points that are too close to the surface (within 0.1 units), also remove respective gradients
    mask = np.abs(distances) > 0.1
    points = points[mask]
    distances = distances[mask]
    gradients = gradients[mask]

    distances[mesh.contains(points)] *= -1.0  # Invert distances for points inside the
    gradients[mesh.contains(points)] *= -1.0  # Invert gradients for points inside the mesh

    return points, distances, gradients

def save_to_gltf( points, surface_points, gradients, outbase, aux=False ):
    '''
    Saves the original points, surface points, and gradients to a GLTF file for visualization.
    Parameters:
    points: (N, 3) array of original point coordinates
        The input points in 3D space.
    surface_points: (N, 3) array of surface point coordinates
        The adjusted points on the surface.
    gradients: (N, 3) array of gradient vectors
        The gradient vectors at each point.
    outbase: str
        The base name for output files.
    '''
    # Plot the output using plot2gltf
    try:
        from plot2gltf import GLTFGeometryExporter
    except ImportError:
        print("plot2gltf not installed; skipping visualization.")
        return

    exporter = GLTFGeometryExporter()

    # Big spheres for the surface points
    exporter.add_spheres(surface_points, color=(0, 1, 0), radius = 0.01)  # Green points
    # Small arrows for the gradients
    exporter.add_normal_arrows(
        surface_points, .05*gradients, color=(0, 1, 1),
        shaft_radius=0.002, head_radius=0.004
    )
    if aux:
        # Add auxiliary visualizations if needed
        # Small spheres for the original points
        exporter.add_spheres(points, color=(1, 0, 0), radius = 0.005)  # Red points
        # Small arrows for the original gradients
        exporter.add_normal_arrows(
            points, .05*gradients, color=(1, 1, 0),
            shaft_radius=0.001, head_radius=0.002
        )

        # Add a very thin line from original points to surface points
        exporter.add_lines(
            np.concatenate([points, surface_points], axis=0),
            list(zip( np.arange(len(points)), np.arange(len(points), len(points)*2) )),
            color=(1, 1, 1)
        )

    outpath = "out/" + outbase + "_surface_points_" + str(args.num_points)
    exporter.save( outpath  + ".gltf")
    
    # # Also save the raw data for easy PSR reconstruction
    # np.savez(outpath + ".npz",
    #          surface_points=surface_points, 
    #          gradients=gradients,
    #          original_points=points)
    print("Saved surface points:", outpath)

def save_PSR_surface( points, normals, outbase, screening_weight = 10.0 ):
    '''
    Runs Poisson Surface Reconstruction to find the surface from the points and normals.
    Parameters:
    points: (N, 3) array of point coordinates
        The input points in 3D space.
    normals: (N, 3) array of normal vectors
        The normal vectors at each point.
    outbase: str
        The base name for output files.
    '''
    try:
        import gpytoolbox
    except ImportError:
        raise ImportError("gpytoolbox is required for PSR_surface function.")
        return
    
    V,F = gpytoolbox.point_cloud_to_mesh( points, normals,
        method='PSR',
        psr_screening_weight=screening_weight,
        psr_outer_boundary_type="Neumann",
        verbose=True
        )
    outpath = "out/" + outbase + "_PSR_surface_" + str(args.num_points) + ".obj"
    gpytoolbox.write_mesh( outpath, V, F )
    print( "Saved PSR surface:", outpath )

if __name__ == "__main__":
    from pathlib import Path
    # Command line arguments to load a mesh or create an n-D sphere
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--mesh", type=str, default=None,
                      help="Path to mesh file to load. If not provided, a sphere will be generated." )
    argparser.add_argument("--num-points", type=int, default=500,
                      help="Number of points to generate or sample." )
    argparser.add_argument("--sphere", type=int, default=3, choices=[2,3],
                      help="Dimension of sphere to generate if no mesh is provided (2 or 3)." )
    args = argparser.parse_args()

    if args.mesh:
        # Load mesh data from provided path
        points, distances, gradients = generate_test_mesh_data( args.mesh, num_points=args.num_points )
        outbase = Path(args.mesh).stem
    elif args.sphere:
        # Generate test data
        points, distances, gradients = generate_test_sphere( args.num_points, radius=1.0, dimension = args.sphere )
        outbase = f"sphere-{args.sphere}D"
        # Ensure points are 3D for visualization
        # If dimension is 2, add a zero z-coordinate
        if points.shape[1] == 2:
            points = np.hstack((points, np.zeros((points.shape[0], 1))))
            gradients = np.hstack((gradients, np.zeros((gradients.shape[0], 1))))
    
    """
    Apply reach-for-the-sphere algorithm to adjust points onto the surface
    based on their signed distances.
    """
    import gpytoolbox as gpy
    if 1:  # Use reach_for_the_spheres
        # some sdf data in numpy arrays SDF_POSITIONS, SDF_VALUES
        # construct initial mesh
        V0, F0 = gpy.icosphere(2)
        # call our algorithm
        # gpy.reach_for_the_spheres expects an SDF callable; build a fast nearest-neighbor
        # interpolant from the sampled `points`->`distances` so we can pass a function.
        from scipy.spatial import cKDTree as KDTree
        _tree = KDTree(points)
        def sdf_callable(q):
            # ensure shape (n,3)
            q = np.asarray(q)
            _, idx = _tree.query(q)
            return distances[idx]
        print("sum:", np.sum(sdf_callable(points) - distances))  # verify sdf_callable works

        Vr,Fr = gpy.reach_for_the_spheres(points, sdf_callable, V0, F0,
            min_h=0.01,
            remesh_iterations=3,
            verbose=False,
            tol=1e-4)
        outpath = "out/" + outbase + "_RFTS_surface_" + str(args.num_points) + ".obj"
    else:  # Use reach_for_the_arcs
        Vr, Fr = gpy.reach_for_the_arcs( points, distances)
        outpath = "out/" + outbase + "_RFTA_surface_" + str(args.num_points) + ".obj"

    # save the reconstructed mesh
    gpy.write_mesh( outpath, Vr, Fr )
    print("Saved RFTS/A surface mesh:", outpath)

    # Apply Yong's algorithm to find points on the surface
    surface_points = yongs_algorithm(points, distances, gradients)
    # filter out points that are too close to the surface (within 0.1 units), also remove respective gradients
    mask = np.abs(distances) > 0.1
    surface_points = surface_points[mask]
    gradients = gradients[mask]
    
    # Verify that the new points are on the sphere surface
    if args.sphere:
        new_distances = np.linalg.norm(surface_points, axis=1) - 1.0
        print("Max distance from surface after adjustment:", np.max(np.abs(new_distances)))

    save_to_gltf( points, surface_points, gradients, outbase )
    save_PSR_surface( surface_points, gradients, outbase )
