import gpytoolbox as gpy
import igl
import numpy as np
import matplotlib.pyplot as plt
from interpolation import Interpolator, PUInterpolator, DuchonInterpolator
import time
import optimization as opt
from util import mesh_distances, are_points_visible
from bench import sphere_exposed_pybind as sep
from bench import sphere_intersect

class Options:
    def __init__(self, grid_len=20, gt_mesh=None, clamp=True, max_iters=10, name='horse', 
                 turn_off_short_arcs=False, export_short_arcs=True, export_projections=True, reg=1e-4,
                 use_gt_gradients=False, interpolator_type='PU', interp_partition='box', overlap=0.5, cpp_dc=True,
                 turn_off_projection=False, use_MES=False, post_processing=True, iter_gradient_finding='optimize', verbose=True):
        self.grid_len = grid_len
        self.max_iters = max_iters
        self.clamp = clamp
        self.turn_off_short_arcs = turn_off_short_arcs
        self.name = name
        self.path_to_obj = f'{data_dir}/{name}.obj'
        self.export_short_arcs = export_short_arcs  # whether to export short arcs .glb for visualization
        self.export_projections = export_projections  # export gradients .glb for visualization
        self.use_gt_gradients = use_gt_gradients
        self.interpolator_type = interpolator_type  # 'Duchon' or 'PU'
        self.interp_partition = interp_partition  # 'box' or 'fps' or 'sphere', only for PU interpolator
        self.interp_overlap = overlap
        self.use_MES = use_MES
        self.post_processing = post_processing
        self.reg = reg
        self.iter_gradient_finding = iter_gradient_finding  # 'optimize' or 'sample'
        self.cpp_dc = cpp_dc
        self.verbose = verbose

        self.gt_gradients = None  # set it manually if you want to use GT gradients for testing, e.g. from the intermediate output of generate_test_mesh_data
        self.gt_mesh = gt_mesh  # set it manually if you want to compute distances to GT mesh at the end, e.g. from the intermediate output of generate_test_mesh_data

        self.tolerance = Tolerance()  # set it manually if you want to adjust the tolerance for clamping, e.g. based on the mean spacing of the input points
        self.degenerate_pts = None  # set it after get_visible_arcs, which is a dictionary: point index -> list of degenerate points (the angle points for short arcs)
        self.batch = None
        self.ngbrs_list = None

        # never used
        self.path_to_sdf = None # set it manually to avoid accidentally loading old data, e.g. f'out/{name}_sdf_{grid_len**3}.npz'
        # self.turn_off_projection = turn_off_projection  # this means only interpolating SDF values without projection or optimization
    def print(self):
        print(f"Options: grid_len={self.grid_len}, name={self.name},"
              f" max_iters={self.max_iters}, clamp={self.clamp},"
              f" turn_off_short_arcs={self.turn_off_short_arcs},"
              f" export_short_arcs={self.export_short_arcs},"
              f" export_projections={self.export_projections},"
              f" use_gt_gradients={self.use_gt_gradients},"
              f" interpolator_type={self.interpolator_type},"
              f" interp_partition={self.interp_partition}",
              f" interp_overlap={self.interp_overlap}")

class Tolerance:
    def __init__(self, clamp_radius_ratio=0.2, clamp_sdf_tol=1e-3, angle_tol=np.radians(15)):
        # 0.2 means gradient rotates 11.5 degrees at most
        # 0.1 means gradient rotates 5.7 degrees at most
        self.clamp_radius_ratio = clamp_radius_ratio # for clamping to the nearest arc point
        self.clamp_sdf_tol = clamp_sdf_tol           # for clamping to the optimal point on visible boundary
        self.float_tol = 1e-8
        self.angle_tol = angle_tol

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
    print("bbox_min:", bbox_min, "bbox_max:", bbox_max)
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
        import os
        os.makedirs('normalized_examples', exist_ok=True)
        mesh.export(f'normalized_examples/{outbase}.obj')
        print(f"Saved normalized mesh to normalized_examples/{outbase}.obj")

    return mesh, points, distances, gradients

def test_mesh(grid_len=20, path_to_sdf=None, path_to_obj=None, save_gtmesh=True):
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
        mesh, points, distances, _ = generate_test_mesh_data(path_to_obj, base_name, grid_len=grid_len, save=save_gtmesh)  # Generate new data with 4096 points

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

def test_rfta(options, save_gtmesh=False, screening_weight=10, parallel=True):
    """
    Test function to demonstrate the process of loading SDF data, fitting an interpolator, and visualizing the results by Marching Cubes.
    e.g. path_to_sdf='out/bunny_sdf_1000.npz', path_to_obj='examples/bunny.obj')
    """
    grid_len, path_to_obj, path_to_sdf = options.grid_len, options.path_to_obj, options.path_to_sdf
    if path_to_sdf is not None:
        # read sdf data from file
        data = np.load(path_to_sdf)
        points = data['points']
        distances = data['sdf_values']
    else:
        base_name = path_to_obj.split('/')[-1].split('.')[0]
        mesh, points, distances, gt_gradients = generate_test_mesh_data(path_to_obj, base_name, grid_len=grid_len, save=save_gtmesh)  # Generate new data with 4096 points
    # Export meshes to out/
    import trimesh, os
    out_dir = 'out/' + path_to_obj.split('/')[-1].split('.')[0]
    os.makedirs(out_dir, exist_ok=True)
    Vr, Fr = gpy.reach_for_the_arcs(points, distances, screening_weight=screening_weight, parallel=parallel)
    rfta = trimesh.Trimesh(vertices=Vr, faces=Fr)
    # keep the largest component plus any components fully inside the points' bbox
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    components = rfta.split(only_watertight=False)
    largest = max(components, key=lambda m: len(m.faces))
    kept = [c for c in components
            if c is largest
            or (np.all(c.vertices >= bbox_min) and np.all(c.vertices <= bbox_max))]
    filtered = trimesh.util.concatenate(kept)
    filtered.export(f'{out_dir}/rfta_{grid_len}.obj')
    print(f"Exported: {out_dir}/rfta_{grid_len}.obj  (kept {len(kept)}/{len(components)} components, {len(filtered.faces)} faces out of {len(Fr)})")

def test_mes(options, save_gtmesh=False, screening_weight=10):
    """
    Test function to demonstrate the process of loading SDF data, fitting an interpolator, and visualizing the results by Marching Cubes.
    e.g. path_to_sdf='out/bunny_sdf_1000.npz', path_to_obj='examples/bunny.obj')
    """
    grid_len, path_to_obj, path_to_sdf = options.grid_len, options.path_to_obj, options.path_to_sdf
    if path_to_sdf is not None:
        # read sdf data from file
        data = np.load(path_to_sdf)
        points = data['points']
        distances = data['sdf_values']
    else:
        base_name = path_to_obj.split('/')[-1].split('.')[0]
        mesh, points, distances, gt_gradients = generate_test_mesh_data(path_to_obj, base_name, grid_len=grid_len, save=save_gtmesh)  # Generate new data with 4096 points
    import sys, os
    # Export meshes to out/
    out_dir = 'out/' + path_to_obj.split('/')[-1].split('.')[0]
    os.makedirs(out_dir, exist_ok=True)
    _here = os.path.dirname(__file__)
    sys.path.insert(0, os.path.join(_here, 'cpp', 'build', '_deps', 'mes_fork-src'))
    from cgal.EmptySpheresReconstruction import MESReconstruction
    R_cgal = MESReconstruction(points, distances,screening_weight=screening_weight)
    gpy.write_mesh(f"{out_dir}/mes_{grid_len}.obj", *R_cgal)

    print(f"Exported: {out_dir}/mes_{grid_len}.obj")

def filter_degenerate_pts(degenerate_pts, interpolator : Interpolator, dist_tol=1e-1):
    """ 
    Filter out degenerate points that are too far from the surface or more than 1 point, since 
    they may cause numerical issues in optimization. After filtering, every idx in degenerate_pts 
    should have exactly 1 point that is close to the surface.
    """
    to_remove = []
    for idx, pts in degenerate_pts.items():
        if len(pts) != 1:
            to_remove.append(idx)
            continue
        pt = pts[0]
        pred = interpolator.predict(pt[np.newaxis, :])
        if np.abs(pred) > dist_tol:
            print(f"Degenerate point {idx} is too far from the surface with predicted sdf {pred}, removing it. This may cause issues in later optimization.")
            to_remove.append(idx)
    for idx in to_remove:
        del degenerate_pts[idx]
    print(f"Filtered out {len(to_remove)} degenerate points that are too far from the surface or len!=1. Remaining degenerate points: {len(degenerate_pts)}")

def init_gradients_by_degenerate_pts(sdf_points, sdf_values, interpolator : Interpolator, degenerate_pts, batch, ngbrs_list, options : Options):
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
    degenerate_pts: a dictionary: point index -> list of degenerate points
        A collection of degenerate points that can be used to clamp the gradients.
    colinear_neighbors: a dictionary: point index -> list of colinear neighbors, optional
        A collection of colinear neighbors that can be used for gradient estimation. Default is None.
    Returns:
    -----------
    gradients: (N, d) array of estimated gradient vectors
        The estimated gradient vectors at each input point.
    '''
    # 1. fit, add degenerate points and refit 2. project and refit 3. clamp to MES points
    """ 1. Handle degenerate points """
    interpolator.fit(sdf_points, sdf_values)  # initial fit without gradients
    print(f"======== first fit done with input {len(sdf_points)} points ")

    filter_degenerate_pts(degenerate_pts, interpolator)
    if options.export_short_arcs and len(degenerate_pts) > 0:
        temp_mask = np.zeros(len(sdf_points), dtype=bool)
        temp_mask[list(degenerate_pts.keys())] = True
        origins = sdf_points[temp_mask]
        projections = np.array([pts[0] for pts in degenerate_pts.values()])
        temp_mask = np.ones(len(origins), dtype=bool)
        out_path = f'out/shortArcs_{options.name}_{options.grid_len}.glb'
        export_projection_visualization(origins, projections, mask=temp_mask, recon_mesh=options.gt_mesh, output_path=out_path)
    if options.use_gt_gradients and options.gt_gradients is not None:
        init_grads = options.gt_gradients
        interpolator.fit(sdf_points, sdf_values, init_grads)
        print(f"======== second fit done with GT gradients for all points =========")
    else:
        init_grads = np.nan*np.ones_like(sdf_points)
        to_train_points = sdf_points.copy()
        to_train_sdf = sdf_values.copy()
        # Add points for degenerate arcs.
        degenerate_pts_to_add = []
        for i, pts in degenerate_pts.items():
            degenerate_pts_to_add.append(pts[0])
            init_grads[i] = (sdf_points[i] - pts[0]) / (sdf_values[i] + 1e-10)
        if len(degenerate_pts_to_add) > 0:
            to_train_points = np.vstack([to_train_points, np.array(degenerate_pts_to_add)])
            to_train_sdf = np.append(to_train_sdf, np.zeros(len(degenerate_pts_to_add)))  # The SDF value at the projected point should be 0
        print(f"After adding points for degenerate arcs, total points: {len(to_train_points)}")
        interpolator.fit(to_train_points, to_train_sdf)
        print(f"======== second fit done with input {len(to_train_points)} points (including {len(degenerate_pts_to_add)} degenerate arc points)")
    print("initial gradient estimation done")

    # debug
    for i, pts in degenerate_pts.items():
        if len(pts) != 1:
            print(f"ERRRRRRRRRRR")
        # For points with degenerate arcs, set gradient directly toward the angle
        init_grads[i] = (sdf_points[i] - pts[0]) / (sdf_values[i] + 1e-10)
        proj = sdf_points[i] - sdf_values[i] * init_grads[i]
        pred_sdf = interpolator.predict(proj[np.newaxis, :])
        if pred_sdf > 0.6:
            print(f"Warning: For degenerate point {i}, the projected point is still outside with predicted sdf {pred_sdf}. This may cause issues in later optimization.")
    return init_grads

def find_all_neighbors_list(centers, radii):
    """ Find all neighbors for each point based on the sphere defined by centers and radii. """
    offsets, neighbors = sphere_intersect.find_intersections(centers, radii)
    return offsets, neighbors, [neighbors[offsets[i]:offsets[i+1]] for i in range(len(offsets)-1)]

def get_visible_arcs(sdf_points, sdf_values, epsilon=1e-8):
    nbr_offsets, nbr_indices, nbr_lists = find_all_neighbors_list(sdf_points, np.abs(sdf_values))
    # batch_res = compute_batch(sdf_points, np.abs(sdf_values), nbr_indices, nbr_offsets)
    batch_res = sep.compute_exposed_batch(sdf_points, np.abs(sdf_values), nbr_indices, nbr_offsets, tol=1e-4)
    n_arcs_arr  = batch_res['n_arcs']
    n_pts_arr   = batch_res['n_points']
    nbr_counts = np.array([len(v) for v in nbr_lists])
    fully_covered = int(np.sum(
        (n_arcs_arr == 0) & (n_pts_arr == 0) & (nbr_counts > 0)))
    print(fully_covered, "fully covered spheres (with neighbors but no arcs or points), which are likely to be completely invisible")
    # Extract degenerate points and organize by sphere
    degenerate_pts = {}
    if 'point_positions' in batch_res:
        pt_positions = batch_res['point_positions']  # shape: [total_pts, 3]
        pt_sphere_idx = batch_res['point_sphere_idx']  # shape: [total_pts]
        for idx, pos in zip(pt_sphere_idx, pt_positions):
            if idx not in degenerate_pts:
                degenerate_pts[idx] = []
            degenerate_pts[idx].append(pos)
    return batch_res, degenerate_pts, nbr_lists

def main_algorithm(sdf_points, sdf_values, options : Options):
    gt_gradients, max_iters, gt_mesh = options.gt_gradients, options.max_iters, options.gt_mesh
    """ step 1: initial gradient estimation using degenerate points """
    # collect visible arcs for each point, which will be used to clamp the gradients later
    batch, degenerate_pts, ngbrs_list = get_visible_arcs(sdf_points, sdf_values, epsilon=1e-8)
    if options.turn_off_short_arcs:
        degenerate_pts = {}
    interpolator = PUInterpolator('cubic', overlap=options.interp_overlap, partition=options.interp_partition)  # use PU for better extrapolation, which is important for the initial gradient estimation. We can switch to Duchon for faster fitting in the optimization loop since we only need to evaluate gradients at given points instead of sampling new points.
    if options.interpolator_type == 'Duchon':
        interpolator = DuchonInterpolator('cubic')
    init_grads = init_gradients_by_degenerate_pts(sdf_points, sdf_values, interpolator, degenerate_pts, batch, ngbrs_list, options)

    """ step 2: iterative optimization """
    # For iterative_projection_3d, we need to pass the degenerate points, batch, and neighbors list for clamping inside the optimization loop
    options.degenerate_pts = degenerate_pts
    options.batch = batch
    options.ngbrs_list = ngbrs_list
    # gradients = opt.iterative_gradient_alignment(sdf_points, sdf_values, init_gradients, interpolator, visible_arcs, degenerate_arcs, num_iter=iters, gt=points)
    gradients, interpolator = opt.iterative_projection_3d(sdf_points, sdf_values, init_grads, interpolator, options, num_iter=max_iters, gt_gradients=gt_gradients)

    # interpolator.fit(sdf_points, sdf_values, gradients, force_recompute=True, use_projection=True)
    projections = sdf_points - sdf_values[:, np.newaxis] * gradients
    return interpolator, projections, are_points_visible(projections, sdf_points, sdf_values)  # return whether there are invisible points in the initial projection (which may cause issues in later optimization)

def export_projection_visualization(sdf_points, projections, mask, recon_mesh, output_path='out/projection_debug.glb'):
    """
    Export a GLB visualization showing SDF points -> projections with visibility filtering.
    
    Visible projections: red SDF sphere -> blue projection sphere + green line
    Invisible projections: red SDF sphere -> gray projection sphere + red line
    """
    import trimesh
    from trimesh.visual.material import PBRMaterial
    import os
    
    scene = trimesh.Scene()
    
    # Add reconstructed mesh
    if recon_mesh is not None:
        mesh_copy = recon_mesh.copy()
        mesh_copy.visual.material = PBRMaterial(
            baseColorFactor=[200, 200, 200, 255], alphaMode='OPAQUE')
        scene.add_geometry(mesh_copy, "recon_mesh")
    
    num_visible = np.sum(mask != 0)
    num_invisible = np.sum(mask == 0)
    print(f"\nBuilding GLB scene: {num_visible} visible + {num_invisible} invisible projections...")
    
    for i in range(len(sdf_points)):
        sdf_pt = sdf_points[i]
        proj_pt = projections[i]
        is_visible = mask[i]
        
        # SDF point (always gray sphere)
        sdf_geom = trimesh.primitives.Sphere(radius=0.0008, center=sdf_pt)
        sdf_geom.visual.material = PBRMaterial(
            baseColorFactor=[100, 100, 100, 255],
            emissiveFactor=[0.3, 0.3, 0.3],
            alphaMode='OPAQUE')
        scene.add_geometry(sdf_geom)
        
        if is_visible:
            # Visible: blue projection sphere + green line
            proj_geom = trimesh.primitives.Sphere(radius=0.002, center=proj_pt)
            proj_geom.visual.material = PBRMaterial(
                baseColorFactor=[0, 0, 255, 255],
                emissiveFactor=[0.0, 0.0, 1.0],
                alphaMode='OPAQUE')
            scene.add_geometry(proj_geom)
            
            line_geom = trimesh.load_path([sdf_pt, proj_pt])
            for e in line_geom.entities:
                e.color = [0, 255, 0, 255]  # green
            scene.add_geometry(line_geom)
        else:
            # Invisible: light red projection sphere + red line
            proj_geom = trimesh.primitives.Sphere(radius=0.002, center=proj_pt)
            proj_geom.visual.material = PBRMaterial(
                baseColorFactor=[255, 50, 50, 255],
                emissiveFactor=[1, 0, 0],
                alphaMode='OPAQUE')
            scene.add_geometry(proj_geom)
            
            line_geom = trimesh.load_path([sdf_pt, proj_pt])
            for e in line_geom.entities:
                e.color = [255, 100, 100, 255]  # light red
            scene.add_geometry(line_geom)
    
    # Export
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else 'out', exist_ok=True)
    scene.export(output_path)
    print(f"Exported to {output_path}")

def test_our_method(options : Options, save_gtmesh=False):
    """
    Test function to demonstrate the process of loading SDF data, fitting an interpolator, and visualizing the results by Marching Cubes.
    e.g. path_to_sdf='out/bunny_sdf_1000.npz', path_to_obj='examples/bunny.obj')
    """
    grid_len = options.grid_len
    path_to_obj = options.path_to_obj
    path_to_sdf = options.path_to_sdf
    iters = options.max_iters
    options.print()
    if path_to_sdf is not None:
        # read sdf data from file
        data = np.load(path_to_sdf)
        points = data['points']
        distances = data['sdf_values']
    else:
        base_name = path_to_obj.split('/')[-1].split('.')[0]
        mesh, points, distances, gt_gradients = generate_test_mesh_data(path_to_obj, base_name, grid_len=grid_len, save=save_gtmesh)  # Generate new data with 4096 points
        options.gt_gradients = gt_gradients
        options.gt_mesh = mesh

    # Create and fit the interpolator
    timer = time.perf_counter()
    use_cpp = True
    if use_cpp:
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cpp', 'build'))
        import sdf_cpp
        cpp_opts = sdf_cpp.Options()
        cpp_opts.grid_len = options.grid_len
        cpp_opts.max_iters = options.max_iters
        cpp_opts.clamp = options.clamp
        cpp_opts.reg = options.reg
        cpp_opts.use_MES = options.use_MES
        cpp_opts.turn_off_short_arcs = options.turn_off_short_arcs
        cpp_opts.interpolator_type = options.interpolator_type
        cpp_opts.interp_partition = options.interp_partition
        cpp_opts.interp_overlap = options.interp_overlap
        cpp_opts.name = options.name
        cpp_opts.export_projections = options.export_projections
        cpp_opts.export_short_arcs  = options.export_short_arcs
        cpp_opts.iter_gradient_finding = options.iter_gradient_finding
        cpp_opts.verbose = options.verbose
        if options.use_gt_gradients:
            cpp_opts.gt_gradients = options.gt_gradients
        if options.tolerance is not None:
            cpp_opts.tolerance.clamp_radius_ratio = options.tolerance.clamp_radius_ratio
            cpp_opts.tolerance.clamp_sdf_tol = options.tolerance.clamp_sdf_tol
            cpp_opts.tolerance.angle_tol = options.tolerance.angle_tol
        result = sdf_cpp.main_algorithm(points, distances, cpp_opts)
        # Wrap C++ interpolator so it has extract_zero_level_set
        class _CppInterpolatorWrapper(Interpolator):
            def __init__(self, cpp_interp):
                self._cpp = cpp_interp
            def fit(self, points, values, gradients=None, **kwargs):
                pass
            def predict(self, x_new, chunk_size=5000):
                return self._cpp.predict(x_new, chunk_size)
            def predict_gradients(self, x_new, chunk_size=5000):
                return self._cpp.predict_gradients(x_new, chunk_size)
        interpolator = _CppInterpolatorWrapper(result.interpolator)
        projections = result.projections
        mask = result.visibility_mask
    else:
        interpolator, projections, mask = main_algorithm(points, distances, options)
    print(f"  ⏱  {'Interpolator fitted':<30} {time.perf_counter() - timer:>7.2f} s")

    # visualize results using marching cubes to extract isosurface
    timer = time.perf_counter()
    """ ========================= output post+dc ========================= """
    use_cpp = True
    bbox_min = np.array([points[:, 0].min(), points[:, 1].min(), points[:, 2].min()], dtype=np.float64)
    bbox_max = np.array([points[:, 0].max(), points[:, 1].max(), points[:, 2].max()], dtype=np.float64)
    extent = (bbox_max - bbox_min).max()
    hint_spacing = extent / max(options.grid_len - 1, 1)
    target_cells_per_hint = 4
    resolution = int(np.clip(np.ceil(extent / (hint_spacing / target_cells_per_hint)), 64, 512))

    if use_cpp:
        result.interpolator.verbose = options.verbose
        verts, faces = result.interpolator.extract_surface(
            bbox_min=bbox_min, bbox_max=bbox_max,
            nx=resolution, ny=resolution, nz=resolution, iso=0.0, chunk_size=5000,
            lipschitz_postfix=options.post_processing,
            use_dual_contouring=options.cpp_dc)
    else:
        verts, faces = interpolator.extract_zero_level_set(
            bounds=((points[:, 0].min(), points[:, 0].max()),
                    (points[:, 1].min(), points[:, 1].max()),
                    (points[:, 2].min(), points[:, 2].max())),
            resolution=resolution, use_odc=True)
    print(f"  ⏱  {'Grid evaluation':<30} {time.perf_counter() - timer:>7.2f} s")
    # Extract isosurface at value 0 using marching cubes
    # Export meshes to out/
    import trimesh, os
    out_dir = 'out/' + path_to_obj.split('/')[-1].split('.')[0]
    os.makedirs(out_dir, exist_ok=True)
    recon = trimesh.Trimesh(vertices=verts, faces=faces)
    # fname = f'interpolant_{grid_len}_{iters}_clamp_{options.interpolator_type}_{options.interp_partition}_ovlp{options.interp_overlap}_reg{options.reg}.obj' if options.clamp else f'interpolant_{grid_len}_{iters}_noclamp_{options.interpolator_type}_{options.interp_partition}_ovlp{options.interp_overlap}_reg{options.reg}.obj'
    clamp_str = 'clamp' if options.clamp else 'noclamp'
    mes_str = 'MES' if options.use_MES else 'noMES'
    post_str = 'post' if options.post_processing else 'nopost'
    if options.cpp_dc:
        post_str = post_str + '_dc'
    else:
        post_str = post_str + '_mc'
    short_arc_str = 'noShortArcs' if options.turn_off_short_arcs else 'shortArcs'
    if not use_cpp:
        post_str = 'odc'
    fname = f'interpolant_{grid_len}_{iters}_{short_arc_str}_{clamp_str}_{mes_str}_{post_str}_{options.interpolator_type}_reg{options.reg}.obj'
    if options.use_gt_gradients:
        fname = f'interpolant_{grid_len}_gtgrad_{post_str}_{options.interpolator_type}_{options.interp_partition}_ovlp{options.interp_overlap}_reg{options.reg}.obj'
    recon.export(f'{out_dir}/{fname}')
    '''
    # """ ========================= output post+mc ========================= """
    # use_cpp = True
    # resolution = 256
    # options.cpp_dc = False
    # if use_cpp:
    #     bbox_min = np.array([points[:, 0].min(), points[:, 1].min(), points[:, 2].min()], dtype=np.float64)
    #     bbox_max = np.array([points[:, 0].max(), points[:, 1].max(), points[:, 2].max()], dtype=np.float64)
    #     verts, faces = result.interpolator.extract_surface(
    #         bbox_min=bbox_min, bbox_max=bbox_max,
    #         nx=resolution, ny=resolution, nz=resolution, iso=0.0, chunk_size=5000,
    #         lipschitz_postfix=options.post_processing,
    #         use_dual_contouring=options.cpp_dc)
    # else:
    #     verts, faces = interpolator.extract_zero_level_set(
    #         bounds=((points[:, 0].min(), points[:, 0].max()),
    #                 (points[:, 1].min(), points[:, 1].max()),
    #                 (points[:, 2].min(), points[:, 2].max())),
    #         resolution=resolution, use_odc=True)
    # print(f"  ⏱  {'Grid evaluation':<30} {time.perf_counter() - timer:>7.2f} s")
    # # Extract isosurface at value 0 using marching cubes
    # # Export meshes to out/
    # import trimesh, os
    # out_dir = 'out/' + path_to_obj.split('/')[-1].split('.')[0]
    # os.makedirs(out_dir, exist_ok=True)
    # recon = trimesh.Trimesh(vertices=verts, faces=faces)
    # # fname = f'interpolant_{grid_len}_{iters}_clamp_{options.interpolator_type}_{options.interp_partition}_ovlp{options.interp_overlap}_reg{options.reg}.obj' if options.clamp else f'interpolant_{grid_len}_{iters}_noclamp_{options.interpolator_type}_{options.interp_partition}_ovlp{options.interp_overlap}_reg{options.reg}.obj'
    # clamp_str = 'clamp' if options.clamp else 'noclamp'
    # mes_str = 'MES' if options.use_MES else 'noMES'
    # post_str = 'post' if options.post_processing else 'nopost'
    # if options.cpp_dc:
    #     post_str = post_str + '_dc'
    # else:
    #     post_str = post_str + '_mc'
    # short_arc_str = 'noShortArcs' if options.turn_off_short_arcs else 'shortArcs'
    # if not use_cpp:
    #     post_str = 'odc'
    # fname = f'interpolant_{grid_len}_{iters}_{short_arc_str}_{clamp_str}_{mes_str}_{post_str}_{options.interpolator_type}_reg{options.reg}.obj'
    # if options.use_gt_gradients:
    #     fname = f'interpolant_{grid_len}_gtgrad_{post_str}_{options.interpolator_type}_{options.interp_partition}_ovlp{options.interp_overlap}_reg{options.reg}.obj'
    # recon.export(f'{out_dir}/{fname}')
    # """ ========================= output ODC ========================= """
    # use_cpp = False
    # resolution = 256
    # if use_cpp:
    #     bbox_min = np.array([points[:, 0].min(), points[:, 1].min(), points[:, 2].min()], dtype=np.float64)
    #     bbox_max = np.array([points[:, 0].max(), points[:, 1].max(), points[:, 2].max()], dtype=np.float64)
    #     verts, faces = result.interpolator.extract_surface(
    #         bbox_min=bbox_min, bbox_max=bbox_max,
    #         nx=resolution, ny=resolution, nz=resolution, iso=0.0, chunk_size=5000,
    #         lipschitz_postfix=options.post_processing,
    #         use_dual_contouring=options.cpp_dc)
    # else:
    #     verts, faces = interpolator.extract_zero_level_set(
    #         bounds=((points[:, 0].min(), points[:, 0].max()),
    #                 (points[:, 1].min(), points[:, 1].max()),
    #                 (points[:, 2].min(), points[:, 2].max())),
    #         resolution=resolution, use_odc=True)
    # print(f"  ⏱  {'Grid evaluation':<30} {time.perf_counter() - timer:>7.2f} s")
    # # Extract isosurface at value 0 using marching cubes
    # # Export meshes to out/
    # import trimesh, os
    # out_dir = 'out/' + path_to_obj.split('/')[-1].split('.')[0]
    # os.makedirs(out_dir, exist_ok=True)
    # recon = trimesh.Trimesh(vertices=verts, faces=faces)
    # # fname = f'interpolant_{grid_len}_{iters}_clamp_{options.interpolator_type}_{options.interp_partition}_ovlp{options.interp_overlap}_reg{options.reg}.obj' if options.clamp else f'interpolant_{grid_len}_{iters}_noclamp_{options.interpolator_type}_{options.interp_partition}_ovlp{options.interp_overlap}_reg{options.reg}.obj'
    # clamp_str = 'clamp' if options.clamp else 'noclamp'
    # mes_str = 'MES' if options.use_MES else 'noMES'
    # post_str = 'post' if options.post_processing else 'nopost'
    # if options.cpp_dc:
    #     post_str = post_str + '_dc'
    # else:
    #     post_str = post_str + '_mc'
    # short_arc_str = 'noShortArcs' if options.turn_off_short_arcs else 'shortArcs'
    # if not use_cpp:
    #     post_str = 'odc'
    # fname = f'interpolant_{grid_len}_{iters}_{short_arc_str}_{clamp_str}_{mes_str}_{post_str}_{options.interpolator_type}_reg{options.reg}.obj'
    # if options.use_gt_gradients:
    #     fname = f'interpolant_{grid_len}_gtgrad_{post_str}_{options.interpolator_type}_{options.interp_partition}_ovlp{options.interp_overlap}_reg{options.reg}.obj'
    # recon.export(f'{out_dir}/{fname}')
    '''

    # Export projection visualization GLB
    if not use_cpp and options.export_projections:
        out_path = f'out/projections_{options.name}_{options.grid_len}_{options.max_iters}.glb'
        if options.use_gt_gradients:
            out_path = f'out/projections_{options.name}_{options.grid_len}_gtmesh.glb'
            export_projection_visualization(points, projections, mask, mesh, output_path=out_path)
            out_path = f'out/projections_{options.name}_{options.grid_len}_gt.glb'
        export_projection_visualization(points, projections, mask, recon, output_path=out_path)
    
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
    trimesh.Trimesh(vertices=verts2, faces=faces2).export(f'{out_dir}/sample_points_{grid_len}.obj')    
    print(f"Exported: {out_dir}/{fname}, {out_dir}/sample_points_{grid_len}.obj")
    mesh_distances(recon, mesh, verbose=True)
    return plt

def check_mesh_error(dir_to_meshes, path_to_gt):
    """ Compute the mesh distance (Hausdorff and Chamfer) between meshes in dir_to_meshes and the ground truth mesh at path_to_gt. """
    import trimesh, os
    gt_mesh = trimesh.load(path_to_gt, force='mesh')
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
            mesh = trimesh.load(os.path.join(dir_to_meshes, mesh_file), force='mesh')
            haus, chamfer, f1 = mesh_distances(mesh, gt_mesh)
            print(f"{mesh_file:<50} against ground truth...", end='')
            print(f"  Hausdorff: {haus:.5f}  Chamfer: {chamfer:.7f}  F1: {f1:.4f}")

if __name__ == "__main__":
    t0 = time.perf_counter()
    batch = True
    data_dir = 'examples'
    if batch:
        for name in ['loewe']:
            for grid_len in [10, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100]:  # 20^3=8000 points, 30^3=27000 points
                # for turn_off_short_arcs in [True, False]:
                #     for clamp in [False, True]:
                #         for use_MES in [False, True]:
                #             for post_processing in [False, True]:
                #                 options = Options(name=name, grid_len=grid_len, max_iters=13, clamp=clamp, cpp_dc=True,
                #                     export_short_arcs=False, export_projections=False, turn_off_short_arcs=turn_off_short_arcs,
                #                     use_gt_gradients=False, interpolator_type='PU', interp_partition='sphere', 
                #                     overlap=0.2, reg=0, use_MES=use_MES, post_processing=post_processing, iter_gradient_finding='optimize')
                #                 options.tolerance = Tolerance(clamp_sdf_tol=1e-6)
                #                 plt = test_our_method(options, save_gtmesh=False)
                options = Options(name=name, grid_len=grid_len, max_iters=13, clamp=False, cpp_dc=True, verbose=True,
                        export_short_arcs=False, export_projections=False, turn_off_short_arcs=True,
                        use_gt_gradients=False, interpolator_type='PU', interp_partition='sphere', 
                        overlap=0.2, reg=0, use_MES=True, post_processing=False, iter_gradient_finding='optimize')
                # plt = test_our_method(options, save_gtmesh=False)
                # test_rfta(options, screening_weight=10, parallel=True)
                # test_mes(options, save_gtmesh=False, screening_weight=10)
            check_mesh_error(f'out/{name}', f'{data_dir}/{name}.obj')
    else:
        for length in [10, 15]:
            options = Options(name='eiffel', grid_len=length, max_iters=13, clamp=False, cpp_dc=True, verbose=True,
                            export_short_arcs=False, export_projections=False, turn_off_short_arcs=True,
                            use_gt_gradients=False, interpolator_type='PU', interp_partition='sphere', 
                            overlap=0.2, reg=0, use_MES=True, post_processing=False, iter_gradient_finding='optimize')
            options.tolerance = Tolerance(clamp_sdf_tol=1e-6)
            # # # plt = test_mesh(grid_len=20, path_to_obj='{data_dir}/holes.obj')
            # plt = test_our_method(options, save_gtmesh=False)
            test_rfta(options, screening_weight=10, parallel=True)
            # test_mes(options, save_gtmesh=False, screening_weight=10)
        check_mesh_error(f'out/{options.name}', f'{data_dir}/{options.name}.obj')

    elapsed = time.perf_counter() - t0
    print(f"  ⏱  {'Total execution time':<30} {elapsed:>7.2f} s")
