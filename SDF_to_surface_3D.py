import os
# torch (pulled in lazily via neural_sdf) and sdf_cpp each bundle their own
# libomp; without this the second one to initialize aborts (OMP Error #15). Set
# before anything can import torch so the neural-SDF source can run in-process.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import trimesh
import gpytoolbox as gpy
import igl
import numpy as np
import matplotlib.pyplot as plt
from interpolation import Interpolator, PUInterpolator, DuchonInterpolator
import time
from util import mesh_distances

# Seed for all randomness in generate_test_mesh_data (scatter sampling, noise).
# __main__ overrides this; importers can set it via `sdf3d.seed = ...` like data_dir.
# None = nondeterministic.
seed = None

class Options:
    def __init__(self, grid_len=20, gt_mesh=None, clamp=False, max_iters=10, name='horse', lr=0.2, optim_steps=5,
                 turn_off_short_arcs=False, export_short_arcs=False, export_projections=False, reg=0,
                 use_gt_gradients=False, interpolator_type='PU', interp_partition='sphere', overlap=0.2, cpp_dc=True,
                use_MES=0, post_processing=False, iter_gradient_finding='optimize', verbose=True,
                pair_local=False, noise=0, bound=1, scatter=False, neural_sdf=None,
                grad_optimizer='ascent'):
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
        self.pair_local = pair_local  # PU: pair each local RBF solve with missing input/projection partners
        self.use_MES = use_MES
        self.post_processing = post_processing
        self.reg = reg
        self.iter_gradient_finding = iter_gradient_finding  # 'optimize' or 'sample'
        self.cpp_dc = cpp_dc
        self.verbose = verbose
        self.lr = lr
        self.optim_steps = optim_steps  # quasi-Newton steps per outer iteration
        # solver behind iter_gradient_finding='optimize':
        # 'ascent' = fixed-step projected gradient ascent (lr is its step size),
        # 'lbfgspp' = one LBFGS++ solve per point
        self.grad_optimizer = grad_optimizer

        self.gt_gradients = None  # set it manually if you want to use GT gradients for testing, e.g. from the intermediate output of generate_test_mesh_data
        self.gt_mesh = gt_mesh  # set it manually if you want to compute distances to GT mesh at the end, e.g. from the intermediate output of generate_test_mesh_data

        self.tolerance = None  # set it manually if you want to adjust the tolerance for clamping, e.g. based on the mean spacing of the input points
        self.degenerate_pts = None  # set it after get_visible_arcs, which is a dictionary: point index -> list of degenerate points (the angle points for short arcs)
        self.batch = None
        self.ngbrs_list = None

        self.noise = noise
        self.bound = bound
        self.scatter = scatter

        # never used
        self.path_to_sdf = None # set it manually to avoid accidentally loading old data, e.g. f'out/{name}_sdf_{grid_len**3}.npz'
        # Source the SDF from a neural field trained on the obj, computed in-process
        # (no npz round-trip). None = exact mesh SDF; 'gt' / 'pc' / 'igr' = neural.
        self.neural_sdf = neural_sdf
        self.neural_retrain = False  # retrain the neural field instead of using the cached weights
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
              f" interp_overlap={self.interp_overlap}",
              f" pair_local={self.pair_local}",
              f" use_MES={self.use_MES}",
              f" grad_optimizer={self.grad_optimizer}",
              f" lr={self.lr}")

class Tolerance:
    def __init__(self, clamp_radius_ratio=0.2, clamp_sdf_tol=1e-3, angle_tol=np.radians(15)):
        # 0.2 means gradient rotates 11.5 degrees at most
        # 0.1 means gradient rotates 5.7 degrees at most
        self.clamp_radius_ratio = clamp_radius_ratio # for clamping to the nearest arc point
        self.clamp_sdf_tol = clamp_sdf_tol           # for clamping to the optimal point on visible boundary
        self.float_tol = 1e-8
        self.angle_tol = angle_tol

def generate_test_mesh_data( path_to_mesh, outbase, grid_len=10, save=False, noise=0.0, bound=1.0, scatter=False ):
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
    # All random draws below (scatter points, noise) come from this generator,
    # seeded by the module-level `seed` (set in __main__, like data_dir).
    # seed=None keeps the old nondeterministic behavior for importers.
    rng = np.random.default_rng(seed)

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
    if scatter:
        # totally random points in the bounding box
        points = rng.uniform(bbox_min, bbox_max, (grid_len**3, 3))
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

    # Add noise to the distances
    if noise > 0:
        distances += rng.normal(0, noise, distances.shape)

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

    if bound < 1.0:
        # Filter points based on SDF values to keep only those within the specified bound
        mask = np.abs(distances) <= bound
        points = points[mask]
        distances = distances[mask]
        gradients = gradients[mask]

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
    recon.export(f'{out_dir}/ours_{grid_len}.obj')
    trimesh.Trimesh(vertices=verts2, faces=faces2).export(f'{out_dir}/sample_points_{grid_len}.obj')
    print(f"Exported: {out_dir}/ours_{grid_len}.obj, {out_dir}/sample_points_{grid_len}.obj")
    mesh_distances(recon, mesh, verbose=True)
    return plt

def test_rfta(options, save_gtmesh=False, screening_weight=10, parallel=True, force_cpu=False, sdf=None):
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
    elif sdf is not None:
        points, distances = sdf
    elif options.neural_sdf is not None:
        # Source the SDF from a neural field trained on the obj, in-process — no
        # npz round-trip. torch and sdf_cpp coexist thanks to KMP_DUPLICATE_LIB_OK
        # (set at module import). options.neural_sdf selects the mode:
        #   'gt' / 'mesh' / 'pc' / 'igr'  (see neural_sdf.train_neural_sdf).
        # No artificial noise is injected: the neural field is itself the
        # imperfect (learned, smoothed) SDF, which is the point of the test. The
        # exact mesh is kept for error evaluation.
        from neural_sdf import generate_neural_sdf_data
        base_name = path_to_obj.split('/')[-1].split('.')[0]
        mesh, points, distances, _ = generate_neural_sdf_data(
            path_to_obj, base_name, grid_len=grid_len, mode=options.neural_sdf,
            bound=options.bound, scatter=options.scatter,
            retrain=options.neural_retrain, verbose=options.verbose)
        options.gt_mesh = mesh  # exact mesh kept for error evaluation
    else:
        base_name = path_to_obj.split('/')[-1].split('.')[0]
        mesh, points, distances, gt_gradients = generate_test_mesh_data(path_to_obj, base_name, grid_len=grid_len, save=save_gtmesh, noise=options.noise, bound=options.bound)  # Generate new data with 4096 points
    # Export meshes to out/
    import trimesh, os
    out_dir = 'out/' + path_to_obj.split('/')[-1].split('.')[0]
    os.makedirs(out_dir, exist_ok=True)
    timer = time.perf_counter()
    Vr, Fr = gpy.reach_for_the_arcs(points, distances, screening_weight=screening_weight, parallel=parallel, force_cpu=force_cpu)
    print(f"  ⏱  {'RFTA reconstruction':<30} {time.perf_counter() - timer:>7.2f} s")
    rfta = trimesh.Trimesh(vertices=Vr, faces=Fr)
    # Keep only components whose mean coordinates are fully inside the input bbox and 
    # percent of coordinates inside the input bbox is at least 50%, so PSR "bubble"
    # artifacts that wrap outside the sample region get dropped.
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    components = rfta.split(only_watertight=False)
    kept = [c for c in components
            if np.all(np.mean(c.vertices, axis=0) >= bbox_min)
                and np.all(np.mean(c.vertices, axis=0) <= bbox_max) and np.mean(np.all((c.vertices >= bbox_min) & (c.vertices <= bbox_max), axis=1)) > 0.5]
    if not kept:
        # First retry with a padded bbox — components that just barely poke
        # outside the sample region are usually still legitimate.
        pad = 0.1 * (bbox_max - bbox_min)
        pmin, pmax = bbox_min - pad, bbox_max + pad
        kept = [c for c in components
                if np.all(np.mean(c.vertices, axis=0) >= pmin)
                    and np.all(np.mean(c.vertices, axis=0) <= pmax) and np.mean(np.all((c.vertices >= pmin) & (c.vertices <= pmax), axis=1)) > 0.5]
    if not kept:
        # Last resort: every component crosses even the padded bbox. Keep the
        # largest so we still write a non-empty .obj.
        kept = [max(components, key=lambda m: len(m.faces))]
    filtered = trimesh.util.concatenate(kept)
    if options.noise > 0:
        fname = f'rfta_{grid_len}_noise{options.noise}.obj'
    elif options.bound < 1.0:
        fname = f'rfta_{grid_len}_bound{options.bound}.obj'
    else:
        fname = f'rfta_{grid_len}.obj'
    filtered.export(f'{out_dir}/' + fname)
    print(f"Exported: {out_dir}/" + fname + f"  (kept {len(kept)}/{len(components)} components, {len(filtered.faces)} faces out of {len(Fr)})")

def test_mes(options, save_gtmesh=False, screening_weight=10, sdf=None):
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
    elif sdf is not None:
        points, distances = sdf
    elif options.neural_sdf is not None:
        # Source the SDF from a neural field trained on the obj, in-process — no
        # npz round-trip. torch and sdf_cpp coexist thanks to KMP_DUPLICATE_LIB_OK
        # (set at module import). options.neural_sdf selects the mode:
        #   'gt' / 'mesh' / 'pc' / 'igr'  (see neural_sdf.train_neural_sdf).
        # No artificial noise is injected: the neural field is itself the
        # imperfect (learned, smoothed) SDF, which is the point of the test. The
        # exact mesh is kept for error evaluation.
        from neural_sdf import generate_neural_sdf_data
        base_name = path_to_obj.split('/')[-1].split('.')[0]
        mesh, points, distances, _ = generate_neural_sdf_data(
            path_to_obj, base_name, grid_len=grid_len, mode=options.neural_sdf,
            bound=options.bound, scatter=options.scatter,
            retrain=options.neural_retrain, verbose=options.verbose)
        options.gt_mesh = mesh  # exact mesh kept for error evaluation
    else:
        base_name = path_to_obj.split('/')[-1].split('.')[0]
        mesh, points, distances, gt_gradients = generate_test_mesh_data(path_to_obj, base_name, grid_len=grid_len, save=save_gtmesh, bound=options.bound)  # Generate new data with 4096 points
    import sys, os
    # Export meshes to out/
    out_dir = 'out/' + path_to_obj.split('/')[-1].split('.')[0]
    os.makedirs(out_dir, exist_ok=True)
    _here = os.path.dirname(__file__)
    sys.path.insert(0, os.path.join(_here, 'cpp', 'build', '_deps', 'mes_fork-src'))
    from cgal.EmptySpheresReconstruction import MESReconstruction
    timer = time.perf_counter()
    R_cgal = MESReconstruction(points, distances,screening_weight=screening_weight)
    print(f"  ⏱  {'MES reconstruction':<30} {time.perf_counter() - timer:>7.2f} s")

    fname = f'mes_{grid_len}.obj'
    if options.bound < 1.0:
        fname = f'mes_{grid_len}_bound{options.bound}.obj'
    gpy.write_mesh(f"{out_dir}/" + fname, *R_cgal)

    print(f"Exported: {out_dir}/" + fname)

def export_mes_spheres(options, radius_threshold=0.05, save_gtmesh=False):
    """
    Export the centers of the maximal empty spheres (MES) computed from the
    SDF samples as a point cloud, keeping only spheres whose |radius| is
    below `radius_threshold`. Useful to spot small/tight empty regions
    (thin features, noise, near-degenerate reconstructions).
    Uses the native sdf_cpp.contact_points_from_sdf binding (no subprocess).
    """
    grid_len, path_to_obj, path_to_sdf = options.grid_len, options.path_to_obj, options.path_to_sdf
    if path_to_sdf is not None:
        data = np.load(path_to_sdf)
        points = data['points']
        distances = data['sdf_values']
    else:
        base_name = path_to_obj.split('/')[-1].split('.')[0]
        mesh, points, distances, gt_gradients = generate_test_mesh_data(path_to_obj, base_name, grid_len=grid_len, save=save_gtmesh, bound=options.bound)
    import sys, os
    out_dir = 'out/' + path_to_obj.split('/')[-1].split('.')[0]
    os.makedirs(out_dir, exist_ok=True)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cpp', 'build'))
    import sdf_cpp

    _, _, spheres = sdf_cpp.contact_points_from_sdf(points, distances, True, 0)
    mask = np.abs(spheres[:, 3]) < radius_threshold
    centers, radii = spheres[mask, :3], spheres[mask, 3]
    fname = f'mes_spheres_{grid_len}_r{radius_threshold}'
    np.savez(f"{out_dir}/{fname}.npz", centers=centers, radii=radii)
    trimesh.PointCloud(centers).export(f"{out_dir}/{fname}.ply")

    print(f"Exported: {out_dir}/{fname}.ply  "
          f"({len(centers)}/{len(spheres)} spheres with |radius| < {radius_threshold})")

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
    import optimization as opt
    from bench import sphere_exposed_pybind as sep
    from bench import sphere_intersect
    from util import mesh_distances, are_points_visible
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

def _build_cpp_options(options : Options):
    """ Translate a Python Options into a C++ sdf_cpp.Options. Mirrors the block in
        test_our_method so the C++ pipeline (main_algorithm / interpolator) can be driven
        from anywhere. Returns the populated sdf_cpp.Options. """
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
    cpp_opts.pair_local = options.pair_local
    cpp_opts.name = options.name
    cpp_opts.export_projections = options.export_projections
    cpp_opts.export_short_arcs  = options.export_short_arcs
    cpp_opts.iter_gradient_finding = options.iter_gradient_finding
    cpp_opts.grad_optimizer = getattr(options, 'grad_optimizer', 'bfgs')
    cpp_opts.lr = options.lr
    cpp_opts.optim_steps = options.optim_steps
    cpp_opts.verbose = options.verbose
    if options.use_gt_gradients:
        cpp_opts.gt_gradients = options.gt_gradients
    if options.tolerance is not None:
        cpp_opts.tolerance.clamp_radius_ratio = options.tolerance.clamp_radius_ratio
        cpp_opts.tolerance.clamp_sdf_tol = options.tolerance.clamp_sdf_tol
        cpp_opts.tolerance.angle_tol = options.tolerance.angle_tol
    return cpp_opts

def _adaptive_resolution(points, grid_len):
    """ Dual-contouring grid resolution: ~target_cells_per_hint cells between adjacent SDF
        samples, clamped to [64, 512]. Shared by test_our_method and construct_mesh so the
        decoupled RBF reconstruction extracts at exactly the same resolution as the full
        pipeline. (The bbox extent cancels out, so this reduces to ~4*(grid_len-1).) """
    extent = (points.max(axis=0) - points.min(axis=0)).max()
    hint_spacing = extent / max(grid_len - 1, 1)
    target_cells_per_hint = 4
    return int(np.clip(np.ceil(extent / (hint_spacing / target_cells_per_hint)), 64, 512))

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
    elif options.neural_sdf is not None:
        # Source the SDF from a neural field trained on the obj, in-process — no
        # npz round-trip. torch and sdf_cpp coexist thanks to KMP_DUPLICATE_LIB_OK
        # (set at module import). options.neural_sdf selects the mode:
        #   'gt' / 'pc' / 'igr'  (see neural_sdf.train_neural_sdf).
        # No artificial noise is injected: the neural field is itself the
        # imperfect (learned, smoothed) SDF, which is the point of the test. The
        # exact mesh is kept for error evaluation.
        from neural_sdf import generate_neural_sdf_data
        base_name = path_to_obj.split('/')[-1].split('.')[0]
        mesh, points, distances, _ = generate_neural_sdf_data(
            path_to_obj, base_name, grid_len=grid_len, mode=options.neural_sdf,
            bound=options.bound, scatter=options.scatter,
            retrain=options.neural_retrain, verbose=options.verbose)
        options.gt_mesh = mesh  # exact mesh kept for error evaluation
    else:
        base_name = path_to_obj.split('/')[-1].split('.')[0]
        mesh, points, distances, gt_gradients = generate_test_mesh_data(path_to_obj, base_name, grid_len=grid_len, save=save_gtmesh, noise=options.noise, bound=options.bound, scatter=options.scatter)  # Generate new data with 4096 points
        options.gt_gradients = gt_gradients
        options.gt_mesh = mesh

    # Create and fit the interpolator
    timer = time.perf_counter()
    use_cpp = True
    if use_cpp:
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cpp', 'build'))
        import sdf_cpp
        cpp_opts = _build_cpp_options(options)
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
    _vis = np.asarray(mask).ravel()
    print(f"Final visibility: {int((_vis != 0).sum())}/{len(_vis)} ({100.0 * (_vis != 0).mean():.2f}%)")
    print(f"  ⏱  {'Interpolator fitted':<30} {time.perf_counter() - timer:>7.2f} s")

    # visualize results using marching cubes to extract isosurface
    timer = time.perf_counter()
    """ ========================= output post+dc ========================= """
    use_cpp = True
    bbox_min = np.array([points[:, 0].min(), points[:, 1].min(), points[:, 2].min()], dtype=np.float64)
    bbox_max = np.array([points[:, 0].max(), points[:, 1].max(), points[:, 2].max()], dtype=np.float64)
    resolution = _adaptive_resolution(points, options.grid_len)
    # resolution = 200
    print(f"Grid resolution for surface extraction: {resolution}")

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
    out_dir = 'out/' + path_to_obj.split('/')[-1].split('.')[0]
    os.makedirs(out_dir, exist_ok=True)
    recon = trimesh.Trimesh(vertices=verts, faces=faces)
    # fname = f'ours_{grid_len}_{iters}_clamp_{options.interpolator_type}_{options.interp_partition}_ovlp{options.interp_overlap}_reg{options.reg}.obj' if options.clamp else f'ours_{grid_len}_{iters}_noclamp_{options.interpolator_type}_{options.interp_partition}_ovlp{options.interp_overlap}_reg{options.reg}.obj'
    clamp_str = '_clamp' if options.clamp else ''
    mes_str = 'noMES' if options.use_MES == -1 else ('MESforce' if options.use_MES == 1 else 'MES')
    post_str = 'post' if options.post_processing else ''
    pair_str = '_pairLocal' if options.pair_local and options.interpolator_type == 'PU' else ''
    if options.cpp_dc:
        post_str = post_str + '_dc'
    else:
        post_str = post_str + '_mc'
    short_arc_str = 'noShortArcs' if options.turn_off_short_arcs else 'shortArcs'
    if not use_cpp:
        post_str = 'odc'
    # fname = f'ours_{grid_len}_{iters}_{short_arc_str}_{clamp_str}_{mes_str}_{post_str}_{options.interpolator_type}_reg{options.reg}_lr{options.lr}.obj'
    if options.noise > 0:
        fname = f'ours_{grid_len}_{iters}_{short_arc_str}_{mes_str}_{options.interpolator_type}{pair_str}{post_str}_noise{options.noise}_reg{options.reg}.obj'
    elif options.bound < 1.0:
        fname = f'ours_{grid_len}_{iters}_{short_arc_str}_{mes_str}_{options.interpolator_type}{pair_str}{post_str}_bound{options.bound}.obj'
    elif options.scatter:
        fname = f'ours_{grid_len}_{iters}_{short_arc_str}_{mes_str}_{options.interpolator_type}{pair_str}{post_str}_scatter.obj'
    else:
        fname = f'ours_{grid_len}_{iters}_{short_arc_str}_{mes_str}_{options.interpolator_type}{clamp_str}{pair_str}{post_str}_{options.grad_optimizer}.obj'
    # if options.use_gt_gradients:
    #     fname = f'ours_{grid_len}_gtgrad_{post_str}_{options.interpolator_type}_{options.interp_partition}_ovlp{options.interp_overlap}_reg{options.reg}.obj'
    recon.export(f'{out_dir}/{fname}')
    print(f"Exported: {out_dir}/{fname}")
    # `mesh` only exists when we generated the SDF from an obj (path_to_sdf is None);
    # when loading cached SDF (e.g. run_baseline) there is no GT mesh in scope.
    if path_to_sdf is None:
        mesh_distances(recon, mesh, verbose=True)
    return points, distances

def check_mesh_error(dir_to_meshes, path_to_gt, edge_chamfer=False):
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
    import edgeChamfer
    for mesh_file in meshes:
        if mesh_file.endswith('.obj'):
            mesh = trimesh.load(os.path.join(dir_to_meshes, mesh_file), force='mesh')
            haus, chamfer, f1 = mesh_distances(mesh, gt_mesh)
            if edge_chamfer:
                ecd, ef1 = edgeChamfer.compute_ecd(mesh, gt_mesh, sample_num=1000_000)
            print(f"{mesh_file:<50} against ground truth...", end='')
            if edge_chamfer:
                print(f"  Hausdorff: {haus:.5f}  Chamfer: {chamfer:.7f}  F1: {f1:.4f}  EdgeChamfer: {ecd:.5f}  EdgeF1: {ef1:.4f}")
            else:
                print(f"  Hausdorff: {haus:.5f}  Chamfer: {chamfer:.7f}  F1: {f1:.4f}")

def test_mc(options : Options, save_gtmesh=False, sdf=None):
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
    elif sdf is not None:
        points, distances = sdf
    elif options.neural_sdf is not None:
        # Source the SDF from a neural field trained on the obj, in-process — no
        # npz round-trip. torch and sdf_cpp coexist thanks to KMP_DUPLICATE_LIB_OK
        # (set at module import). options.neural_sdf selects the mode:
        #   'gt' / 'mesh' / 'pc' / 'igr'  (see neural_sdf.train_neural_sdf).
        # No artificial noise is injected: the neural field is itself the
        # imperfect (learned, smoothed) SDF, which is the point of the test. The
        # exact mesh is kept for error evaluation.
        from neural_sdf import generate_neural_sdf_data
        base_name = path_to_obj.split('/')[-1].split('.')[0]
        mesh, points, distances, _ = generate_neural_sdf_data(
            path_to_obj, base_name, grid_len=grid_len, mode=options.neural_sdf,
            bound=options.bound, scatter=options.scatter,
            retrain=options.neural_retrain, verbose=options.verbose)
        options.gt_mesh = mesh  # exact mesh kept for error evaluation
    else:
        base_name = path_to_obj.split('/')[-1].split('.')[0]
        mesh, points, distances, gt_gradients = generate_test_mesh_data(path_to_obj, base_name, grid_len=grid_len, save=save_gtmesh)  # Generate new data with 4096 points
        options.gt_gradients = gt_gradients
        options.gt_mesh = mesh
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
    out_dir = 'out/' + path_to_obj.split('/')[-1].split('.')[0]
    os.makedirs(out_dir, exist_ok=True)
    trimesh.Trimesh(vertices=verts2, faces=faces2).export(f'{out_dir}/sample_points_{grid_len}.obj')    
    print(f"Exported: {out_dir}/sample_points_{grid_len}.obj")
    # mesh_distances(recon, mesh, verbose=True)
    return plt

from enum import Enum
class TangentPoints(Enum):
    GT = 'gt'
    OURS = 'ours'
    RFTA = 'rfta'
    MES = 'mes'

def get_tangent_points(options : Options, method, save_gtmesh=False, screening_weight=10):
    """ Get the tangent points (surface contact points) for the given options and method.

        Returns
        -------
        tangent_pts: (M, 3) array of points lying on the reconstructed surface.
            For OURS the tangent points are the SDF-sample projections and are 1:1 with
            ``points``; for RFTA/MES they are the method's reconstructed point cloud and
            need not be 1:1 with the input samples.
        points:    (N, 3) input SDF sample coordinates.
        distances: (N,)   signed distances at ``points``.
    """
    grid_len = options.grid_len
    path_to_obj = options.path_to_obj
    path_to_sdf = options.path_to_sdf
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
    if method == TangentPoints.OURS:
        # Iterative projection (C++ pipeline, same as test_our_method): tangent points are
        # the SDF-sample projections onto the surface (points - distance * gradient). The
        # optimization's gradients exist only to produce these projections; some come out
        # non-visible / unreliable and the optimization excludes them via a mask when
        # fitting. We mirror that by marking the invalid projections NaN so the
        # reconstruction drops them. The valid tangent points stay index-aligned with
        # points/distances (1:1).
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cpp', 'build'))
        import sdf_cpp
        result = sdf_cpp.main_algorithm(points, distances, _build_cpp_options(options))
        tangent_pts = np.array(result.projections, dtype=np.float64)  # copy: pybind view is read-only
        vis = np.asarray(result.visibility_mask).reshape(-1).astype(bool)
        tangent_pts[~vis] = np.nan
    elif method == TangentPoints.RFTA:
        # Reach for the Arcs: use the reconstructed point cloud as tangent points.
        # NOTE: this point cloud is NOT 1:1 with the input samples (one sphere can
        # contribute several or zero points), so only the useRBF=True reconstruction
        # path (zero-level constraints) is valid for it.
        _, _, P, _ = gpy.reach_for_the_arcs(
            points, distances, return_point_cloud=True,
            screening_weight=screening_weight, parallel=True, force_cpu=False)
        tangent_pts = np.asarray(P, dtype=np.float64)
    elif method == TangentPoints.MES:
        # Maximal Empty Spheres: use the reconstructed oriented point cloud as tangent points.
        # MES does NOT emit one contact point per input sample (fully-covered / interior
        # samples produce none), so the count differs from len(points) and there is no 1:1
        # correspondence -- again only the useRBF=True path is valid for it.
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cpp', 'build', '_deps', 'mes_fork-src'))
        from cgal.EmptySpheresReconstruction import MESReconstruction
        *_, P, _N = MESReconstruction(points, distances, screening_weight=screening_weight, return_oriented_points=True)
        tangent_pts = np.asarray(P, dtype=np.float64)
        if tangent_pts.size == 0:
            raise RuntimeError("MES returned no contact points; cannot build a tangent-point set.")
    elif method == TangentPoints.GT:
        # Ground truth: the tangent point of each SDF sample is its closest point on the GT
        # mesh (the exact projection onto the true surface), 1:1 with points -- valid for
        # both reconstruction paths.
        gt = options.gt_mesh
        if gt is None:
            raise ValueError("GT tangent points require options.gt_mesh (set by generate_test_mesh_data).")
        V = np.asarray(gt.vertices, dtype=np.float64)
        F = np.asarray(gt.faces, dtype=np.int32)
        _, _, closest = igl.point_mesh_squared_distance(points, V, F)
        tangent_pts = np.asarray(closest, dtype=np.float64)
    else:
        raise ValueError(f"Unknown method: {method}")
    n_valid = int(np.isfinite(tangent_pts).all(axis=1).sum())
    print(f"  [{method.value}] tangent points: {n_valid} valid / {len(tangent_pts)} total  (from {len(points)} SDF samples)")
    return tangent_pts, points, distances

def construct_mesh(tangent_pts, points, distances, useRBF : bool, options : Options, screening_weight=10):
    """
        If useRBF is True, construct mesh using RBF interpolation, otherwise use sPSR on the input points.
        tangent_pts: (N, 3) array of tangent points corresponding to the input points. These
            can be used to compute normals for the sPSR method, or treated as 0 value constraints for RBF interpolation.
        points: (N, 3) array of point coordinates.
        distances: (N,) array of signed distance values corresponding to the input points.
        options: the RBF hyperparameters (reg, interp_overlap, interp_partition, pair_local)
            and grid_len (for the adaptive extraction resolution) are read from here so B2
            mirrors test_our_method's proposed RBF -- in particular reg must be passed through
            (the C++ ctor defaults to reg=1e-5, but Options.reg defaults to 0).
        screening_weight: PSR screening weight for the sPSR (useRBF=False) path.
        Returns a trimesh.Trimesh of the reconstructed surface.
    """
    # Invalid tangent points are flagged NaN by get_tangent_points (non-visible projections);
    # drop them before reconstruction.
    valid = np.isfinite(tangent_pts).all(axis=1)
    if useRBF:
        # Fit the C++ RBF (PU) interpolator to the SDF samples (value constraints) plus the
        # valid tangent points (zero-level constraints), then extract the surface with the
        # C++ dual-contouring path. Plain value RBF -- no gradient/Hermite constraints; the
        # tangent points carry the surface information. No 1:1 correspondence between
        # tangent_pts and points is required here.
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cpp', 'build'))
        import sdf_cpp
        resolution = _adaptive_resolution(points, options.grid_len)
        print(f"Grid resolution for surface extraction: {resolution}")
        tp = tangent_pts[valid]
        interp = sdf_cpp.PUInterpolator(kernel='cubic', overlap=options.interp_overlap, reg=options.reg,
                                        partition=options.interp_partition, pair_local=options.pair_local,
                                        verbose=False)
        fit_pts = np.vstack([points, tp])
        fit_vals = np.concatenate([distances, np.zeros(len(tp))])
        interp.fit(fit_pts, fit_vals)
        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)
        verts, faces = interp.extract_surface(
            bbox_min=bbox_min, bbox_max=bbox_max,
            nx=resolution, ny=resolution, nz=resolution, iso=0.0, chunk_size=5000,
            lipschitz_postfix=False, use_dual_contouring=True)
        return trimesh.Trimesh(vertices=verts, faces=faces)
    else:
        # Screened Poisson on the tangent points. Normals are derived from the SDF samples:
        # (point - tangent) points away from the surface for outside samples; flipping by the
        # sign of the distance orients every normal outward. Requires tangent_pts 1:1 with
        # points (true for OURS); restrict to the valid (index-aligned) subset.
        tp = tangent_pts[valid]
        pts = points[valid]
        dst = distances[valid]
        d = pts - tp
        n = np.linalg.norm(d, axis=1, keepdims=True)
        n[n < 1e-12] = 1.0
        normals = (d / n) * np.sign(dst)[:, np.newaxis]
        Vr, Fr = gpy.point_cloud_to_mesh(tp, normals, method='PSR', psr_screening_weight=screening_weight)
        recon = trimesh.Trimesh(vertices=Vr, faces=Fr)
        # sPSR can emit spurious closed "bubble" components floating outside the sampled
        # region; drop any component lying entirely outside the input (SDF sample) bbox.
        bmin, bmax = points.min(axis=0), points.max(axis=0)
        comps = recon.split(only_watertight=False)
        if len(comps) > 1:
            kept = [c for c in comps
                    if np.any(np.all((np.asarray(c.vertices) >= bmin)
                                     & (np.asarray(c.vertices) <= bmax), axis=1))]
            if not kept:  # never emit an empty mesh
                kept = [max(comps, key=lambda m: len(m.faces))]
            recon = trimesh.util.concatenate(kept)
        return recon

if __name__ == "__main__":
    t0 = time.perf_counter()
    seed = 1
    batch = False
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
        for length in [100]:
            for optimizer in [ 'lbfgspp', 'bfgs', 'ascent']:
                options = Options(name='eiffel', grid_len=length, use_MES=0, clamp=True, optim_steps=5)
                # options.path_to_sdf = 'out/bunny_neural_sdf_8000.npz'
                # tangent_pts, points, distances = get_tangent_points(options, TangentPoints.GT, save_gtmesh=False)
                # recon = construct_mesh(tangent_pts, points, distances, useRBF=True, options=options)
                # recon.export(f'out/{options.name}/ours_{length}_gtgrad.obj')
                # options.grad_optimizer = "lbfgspp"  # 每点独立跑 LBFGS++
                # options.grad_optimizer = "bfgs"
                # options.grad_optimizer = "ascent"    # 固定步长投影梯度上升(原方法)
                options.grad_optimizer = optimizer

                # export_mes_spheres(options, radius_threshold=0.001)
                points, distances = test_our_method(options, save_gtmesh=False)
                # test_rfta(options, screening_weight=10, parallel=True, sdf=(points, distances))
                # test_mc(options, save_gtmesh=False, sdf=(points, distances))
                # test_mes(options, save_gtmesh=False, screening_weight=10, sdf=(points, distances))
        # check_mesh_error(f'out/{options.name}', f'{data_dir}/{options.name}.obj', edge_chamfer=True)

    elapsed = time.perf_counter() - t0
    print(f"  ⏱  {'Total execution time':<30} {elapsed:>7.2f} s")
