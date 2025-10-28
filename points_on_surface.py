# /// script
# requires-python = ">=3.10"
# dependencies = [
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
    norm_gradients = gradients / np.linalg.norm(gradients, axis=1, keepdims=True)
    # Compute the new points by moving along the gradient direction
    new_points = points - (distances[:, np.newaxis] * norm_gradients)
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

    # Generate random points in 3D space
    radius = 2*np.max( np.linalg.norm( mesh.vertices, axis=1 ) )
    points = np.random.uniform(-radius, radius, (num_points, 3))

    # Find the closest points on the mesh surface
    query = trimesh.proximity.ProximityQuery(mesh)
    closest, distances, _ = query.on_surface( points )
    gradients = points - closest
    # Normalize gradients
    gradients /= np.linalg.norm(gradients, axis=1, keepdims=True)

    return points, distances, gradients

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
    
    # Apply Yong's algorithm to find points on the surface
    surface_points = yongs_algorithm(points, distances, gradients)
    filtered_surface_points = surface_points[np.abs(distances) > .1]
    
    # Verify that the new points are on the sphere surface
    if args.sphere:
        new_distances = np.linalg.norm(surface_points, axis=1) - 1.0
        print("Max distance from surface after adjustment:", np.max(np.abs(new_distances)))

    # Plot the output using plot2gltf
    try:
        from plot2gltf import GLTFGeometryExporter
        exporter = GLTFGeometryExporter()

        # Big spheres for the surface points
        exporter.add_spheres(surface_points, color=(0, 1, 0), radius = 0.01)  # Green points
        # Small arrows for the gradients
        exporter.add_normal_arrows(
            surface_points, .05*gradients, color=(0, 1, 1),
            shaft_radius=0.002, head_radius=0.004
        )

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
        
        outpath = "surface_points " + outbase + ".gltf"
        exporter.save( outpath )
        print("Surface points saved:", outpath)
    except ImportError:
        print("plot2gltf not installed; skipping visualization.")
