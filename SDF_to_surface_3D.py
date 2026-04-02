from scipy.spatial import cKDTree
import gpytoolbox as gpy
import igl
from scipy.spatial.distance import cdist
import numpy as np
# import matplotlib
# matplotlib.use('Agg')  # Non-interactive backend, no window shown
import matplotlib.pyplot as plt
import iterative_projection as ip
from interpolation import Interpolator, CurlFree_Interpolator
from enum import Enum
import time
import optimization as opt
from util import mesh_distances, are_points_visible

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
    interpolator.fit(points, distances) # O(n^3) e.g. n=20^3=8000 points, 8000^3=512e9
    print(f"  ⏱  {'Interpolator fitted':<30} {time.perf_counter() - timer:>7.2f} s")

    # visualize results using marching cubes to extract isosurface
    timer = time.perf_counter()
    verts, faces = interpolator.extract_zero_level_set(bounds=((points[:, 0].min(), points[:, 0].max()),
                                                (points[:, 1].min(), points[:, 1].max()),
                                                (points[:, 2].min(), points[:, 2].max())),
                                                resolution=100) # O(100^3 n), e.g. n=20^3=8000 points, 1e6 * 8000=8e9
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
    out_dir = 'out/sdf_interp/' + path_to_obj.split('/')[-1].split('.')[0]
    os.makedirs(out_dir, exist_ok=True)
    recon = trimesh.Trimesh(vertices=verts, faces=faces)
    recon.export(f'{out_dir}/interpolant_{grid_len}.obj')
    trimesh.Trimesh(vertices=verts2, faces=faces2).export(f'{out_dir}/sample_points_{grid_len}.obj')
    print(f"Exported: {out_dir}/interpolant_{grid_len}.obj, {out_dir}/sample_points_{grid_len}.obj")
    mesh_distances(recon, mesh, verbose=True)
    return plt

def test_rfta(grid_len=20, path_to_sdf=None, path_to_obj=None, save_npz=False):
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
        mesh, points, distances, gt_gradients = generate_test_mesh_data(path_to_obj, base_name, grid_len=grid_len, save=save_npz)  # Generate new data with 4096 points
    # Export meshes to out/
    import trimesh, os
    out_dir = 'out/' + path_to_obj.split('/')[-1].split('.')[0]
    os.makedirs(out_dir, exist_ok=True)
    Vr, Fr = gpy.reach_for_the_arcs(points, distances)
    rfta = trimesh.Trimesh(vertices=Vr, faces=Fr)
    # only keep the largest connected component
    components = rfta.split(only_watertight=False)
    largest = max(components, key=lambda m: len(m.faces))
    largest.export(f'{out_dir}/rfta_{grid_len}.obj')
    print(f"Exported: {out_dir}/rfta_{grid_len}.obj  (kept largest component: {len(largest.faces)} faces out of {len(Fr)})")

def clamp_gradients_to_arcs(points, values, gradients, degenerate_arcs):
    pass

def clamp_gradient_to_colinear_neighbors(gradients, colinear_neighbors, degenerate_arcs, sdf_points, sdf_values, clamp_indices=None):
    """
    Use the colinear neighbors to estimate the gradient. For points inside clamp_indices, we will use that estimated gradient to clamp into visible arcs.
    """
    for i, neighbors in colinear_neighbors.items():
        if len(neighbors) == 0:
            continue
        if i in degenerate_arcs:
            continue    # skip clamping for points with degenerate arcs, since we will set their gradients directly toward the angle later
        if clamp_indices is not None and i not in clamp_indices:
            continue    # only clamp points that are outside visible arcs, for other points we keep their original gradients since they are already within visible arcs
        neighbor_points = sdf_points[neighbors]
        neighbor_sdf = sdf_values[neighbors]
        diffs = neighbor_points - sdf_points[i]
        sdf_diffs = neighbor_sdf - sdf_values[i]
        if len(diffs) >= 2 and np.linalg.matrix_rank(diffs, tol=1e-8) < 2:
            # Rank-deficient: project onto the available direction
            direction = diffs[0] / (np.linalg.norm(diffs[0]) + 1e-10)
            projections = diffs @ direction
            slope = np.linalg.lstsq(projections.reshape(-1, 1), sdf_diffs, rcond=None)[0][0]
            grad = slope * direction
        else:
            grad, _, _, _ = np.linalg.lstsq(diffs, sdf_diffs, rcond=None)
        gradients[i] = grad
    return gradients

def estimate_gradients_interp(sdf_points, sdf_values, interpolator : Interpolator, degenerate_arcs, colinear_neighbors=None):
    '''
    Estimate gradients by fitting a Duchon interpolator to the signed distance data and evaluating its gradient.

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
    # TODO: Add points for degenerate arcs.
    print(f"After adding points for degenerate arcs, total points: {len(to_train_points)}")
    interpolator.fit(to_train_points, to_train_sdf)
    print(f"first fit done with input {len(to_train_points)} points")
    gradients = interpolator.sample_best_gradients(sdf_points, sdf_values)
    # verts, faces = interpolator.extract_zero_level_set(bounds=((sdf_points[:, 0].min(), sdf_points[:, 0].max()), 
    #                                             (sdf_points[:, 1].min(), sdf_points[:, 1].max()), 
    #                                             (sdf_points[:, 2].min(), sdf_points[:, 2].max())), resolution=50)
    # # project sdf_points onto the zero level set mesh to get nearest surface points
    # faces_igl = np.asarray(faces, dtype=np.int32)
    # sq_dists, _, nearest = igl.point_mesh_squared_distance(sdf_points, verts, faces_igl)
    # valid = sq_dists > 1e-16  # skip points that landed exactly on surface
    # gradients = np.zeros_like(sdf_points)
    # # gradient = (point - nearest) / sdf_value  (sign is automatic via sdf_value)
    # gradients[valid] = (sdf_points[valid] - nearest[valid]) / sdf_values[valid, np.newaxis]
    # norms = np.linalg.norm(gradients, axis=1, keepdims=True)
    # norms[norms < 1e-10] = 1.0
    # gradients /= norms
    print("initial gradient estimation done")
    if colinear_neighbors is not None:
        # Clamp the gradients ONLY in clamp_indices to the directions defined by their colinear neighbors.
        # va.clamp_gradient_to_colinear_neighbors(gradients, colinear_neighbors, degenerate_arcs, sdf_points, sdf_values, clamp_indices)
        # Clamp the gradients to the directions defined by their colinear neighbors.
        clamp_gradient_to_colinear_neighbors(gradients, colinear_neighbors, degenerate_arcs, sdf_points, sdf_values)
    # TODO: Clamp gradients to visible arcs based on the lowest function value.
    clamp_indices = clamp_gradients_to_arcs(sdf_points, sdf_values, gradients, degenerate_arcs)
    for i, angle in degenerate_arcs.items():
        # For points with degenerate arcs, set gradient directly toward the angle
        gradients[i] = np.array([-np.cos(angle), -np.sin(angle)]) if sdf_values[i] > 0 else np.array([np.cos(angle), np.sin(angle)])
    return gradients

def yongs_algorithm(sdf_points, sdf_values, gt_gradients=None, max_iters=100):
    """ step 1: initial gradient estimation using an interpolation """
    degenerate_arcs = {}  #TODO: compute degenerate arcs
    interpolator = Interpolator('cubic')
    init_gradients = estimate_gradients_interp(sdf_points, sdf_values, interpolator, degenerate_arcs)
    # init_gradients = gt_gradients if gt_gradients is not None else init_gradients
    
    """ step 2: project onto surface, then filter invisible points """
    init_projections = sdf_points - sdf_values[:, np.newaxis] * init_gradients
    mask = are_points_visible(init_projections, sdf_points, sdf_values)
    # mask = np.ones(len(sdf_points), dtype=bool)
    num_visible_points = np.sum(mask)
    print(f"Number of visible projected points: {num_visible_points} out of {len(sdf_points)}. percent: {np.mean(mask) * 100:.2f}%")
    new_points = np.vstack([sdf_points, init_projections[mask]])
    new_values = np.concatenate([sdf_values, np.zeros(num_visible_points)])

    """ step 3: refit the interpolator with the original points + projected points """
    interpolator.fit(new_points, new_values, force_recompute=True) # it is the same as interpolator.fit(sdf_points, sdf_values, init_gradients, mask, force_recompute=True)

    # init_zero_contours = interpolator.extract_zero_level_set(bounds=((0, 1), (0, 1)), resolution=resolution)

    """ step 4: iterative optimization """
    # gradients = opt.iterative_gradient_alignment(sdf_points, sdf_values, init_gradients, interpolator, visible_arcs, degenerate_arcs, num_iter=iters, gt=points)
    gradients, interpolator = opt.iterative_projection_3d(sdf_points, sdf_values, init_gradients, interpolator=interpolator, num_iter=max_iters, gt_gradients=gt_gradients)

    # interpolator.fit(sdf_points, sdf_values, gradients, force_recompute=True, use_projection=True)
    return interpolator

def test_our_method(grid_len=20, path_to_sdf=None, path_to_obj=None, iters=10, save_npz=False):
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
        mesh, points, distances, gt_gradients = generate_test_mesh_data(path_to_obj, base_name, grid_len=grid_len, save=save_npz)  # Generate new data with 4096 points

    # Create and fit the interpolator
    timer = time.perf_counter()
    interpolator = yongs_algorithm(points, distances, gt_gradients=gt_gradients, max_iters=iters)
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
    recon = trimesh.Trimesh(vertices=verts, faces=faces)
    recon.export(f'{out_dir}/interpolant_{grid_len}_{iters}.obj')
    trimesh.Trimesh(vertices=verts2, faces=faces2).export(f'{out_dir}/sample_points_{grid_len}.obj')
    print(f"Exported: {out_dir}/interpolant_{grid_len}_{iters}.obj, {out_dir}/sample_points_{grid_len}.obj")
    mesh_distances(recon, mesh, verbose=True)
    return plt

def check_mesh_error(dir_to_meshes, path_to_gt):
    """ Compute the mesh distance (Hausdorff and Chamfer) between meshes in dir_to_meshes and the ground truth mesh at path_to_gt. """
    import trimesh, os
    gt_mesh = trimesh.load(path_to_gt)
    # Normalize the mesh to fit within a unit cube
    min = np.min( gt_mesh.vertices, axis=0 )
    max = np.max( gt_mesh.vertices, axis=0 )
    gt_mesh.vertices -= (min + max) / 2
    gt_mesh.vertices /= np.max( max - min )

    meshes = os.listdir(dir_to_meshes)
    meshes.sort()
    print(f"{os.path.basename(dir_to_meshes):<30}")
    for mesh_file in meshes:
        if mesh_file.endswith('.obj'):
            mesh = trimesh.load(os.path.join(dir_to_meshes, mesh_file))
            haus, chamfer = mesh_distances(mesh, gt_mesh)
            print(f"{mesh_file:<30} against ground truth...", end='')
            print(f"  Hausdorff: {haus:.4f}  Chamfer: {chamfer:.4f}")

if __name__ == "__main__":
    t0 = time.perf_counter()
    name = 'eiffel'
    # plt = test_mesh(grid_len=20, path_to_obj='examples/holes.obj')
    plt = test_our_method(grid_len=20, path_to_obj=f'examples/{name}.obj', iters=0)
    # check_mesh_error('out/bunny', 'examples/bunny.obj')
    # test_rfta(grid_len=20, path_to_obj=f'examples/{name}.obj')
    check_mesh_error(f'out/{name}', f'examples/{name}.obj')

    elapsed = time.perf_counter() - t0
    print(f"  ⏱  {'Total execution time':<30} {elapsed:>7.2f} s")
