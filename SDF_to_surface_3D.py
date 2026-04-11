import mes_contact
from scipy.spatial import cKDTree
import gpytoolbox as gpy
import igl
from scipy.spatial.distance import cdist
import numpy as np
# import matplotlib
# matplotlib.use('Agg')  # Non-interactive backend, no window shown
import matplotlib.pyplot as plt
import iterative_projection as ip
from interpolation import Interpolator, CurlFree_Interpolator, PUInterpolator, DuchonInterpolator
from enum import Enum
import time
import optimization as opt
from util import mesh_distances, are_points_visible
# from bench import sphere_exposed_cpp as va
from bench import sphere_exposed_pybind as sep
from bench import sphere_intersect
import util

class Options:
    def __init__(self, grid_len=20, gt_mesh=None, clamp=True, max_iters=10, name='horse', 
                 turn_off_short_arcs=False, export_short_arcs=True, export_projections=True,
                 use_gt_gradients=False, interpolator_type='PU', interp_partition='box', overlap=0.5, 
                 turn_off_projection=False):
        self.grid_len = grid_len
        self.max_iters = max_iters
        self.clamp = clamp
        self.turn_off_short_arcs = turn_off_short_arcs
        self.name = name
        self.path_to_obj = f'examples/{name}.obj'
        self.export_short_arcs = export_short_arcs  # whether to export short arcs .glb for visualization
        self.export_projections = export_projections  # export gradients .glb for visualization
        self.use_gt_gradients = use_gt_gradients
        self.interpolator_type = interpolator_type  # 'Duchon' or 'PU'
        self.interp_partition = interp_partition  # 'box' or 'fps' or 'sphere', only for PU interpolator
        self.interp_overlap = overlap
        
        self.gt_gradients = None  # set it manually if you want to use GT gradients for testing, e.g. from the intermediate output of generate_test_mesh_data
        self.gt_mesh = gt_mesh  # set it manually if you want to compute distances to GT mesh at the end, e.g. from the intermediate output of generate_test_mesh_data

        self.tolerance = None
        self.degenerate_pts = None  # set it after get_visible_arcs, which is a dictionary: point index -> list of degenerate points (the angle points for short arcs)
        self.batch = None
        self.ngbrs_list = None
        self.MES_normals = None  # set it after mes_contact

        # never used
        self.path_to_sdf = None # set it manually to avoid accidentally loading old data, e.g. f'out/{name}_sdf_{grid_len**3}.npz'
        # self.turn_off_projection = turn_off_projection  # this means only interpolating SDF values without projection or optimization
    def print(self):
        print(f"Options: grid_len={self.grid_len},"
              f" max_iters={self.max_iters}, clamp={self.clamp},"
              f" turn_off_short_arcs={self.turn_off_short_arcs},"
              f" export_short_arcs={self.export_short_arcs},"
              f" export_projections={self.export_projections},"
              f" use_gt_gradients={self.use_gt_gradients},"
              f" interpolator_type={self.interpolator_type},"
              f" interp_partition={self.interp_partition}",
              f" interp_overlap={self.interp_overlap}")

class Tolerance:
    def __init__(self, clamp_radius_ratio=0.2, clamp_sdf_tol=1e-2, angle_tol=np.radians(15)):
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

def test_rfta(options, save_npz=False):
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

def clamp_gradients_to_arcs(points, values, gradients, degenerate_pts, batch, ngbrs_list, interpolator : Interpolator, tolerance : Tolerance, MES_normals):
    projections = points - values[:, np.newaxis] * gradients
    sdf_tol = tolerance.clamp_sdf_tol
    ratio = tolerance.clamp_radius_ratio
    float_tol = tolerance.float_tol
    debug_cnt = 0
    clamp_precision_cnt = 0
    clamp_work_cnt = 0
    for i in range(len(points)):
        if i in degenerate_pts:
            continue    # skip clamping for points with degenerate arcs, since we will set their gradients directly toward the angle later
        ngbrs = points[ngbrs_list[i]]
        dists = np.linalg.norm(projections[i] - ngbrs, axis=1)
        inside = dists < (np.abs(values[ngbrs_list[i]])) - float_tol
        if not np.any(inside):
            continue    # no neighbor contains the projection, skip clamping
        # 1. if it is close to some arc, clamp to the closest point on that arc
        closest, distances = util.query_closest_on_arcs(projections[i:i+1], i, batch)
        if distances[0] < ratio * np.abs(values[i]):
            pt = closest[0]
            if (np.linalg.norm(ngbrs - pt, axis=1) < (np.abs(values[ngbrs_list[i]]) - float_tol)).any():
                clamp_precision_cnt += 1  # if the closest point on arc is too close to neighbors, it may cause more issues, skip clamping in this case
            else:
                clamp_work_cnt += 1
            gradients[i] = (points[i] - pt) / (values[i] + 1e-10) # clamp to the closest point on the arc, with a slightly relaxed denominator to avoid over-shooting
            continue

        # 2. otherwise, if some point on the arc has a low function value, clamp to that point
        # sample_pts = util.sample_arcs(i, batch, num_points=100)
        # if len(sample_pts) == 0:
        #     debug_cnt += 1
        # else:
        #     grads = interpolator.sample_best_gradients(points[i:i+1], values[i:i+1], given_samples=sample_pts[np.newaxis, :, :])  # given_samples shape: (1, num_points, 3)
        #     proj = points[i] - values[i] * grads[0]
        #     pred_sdf = interpolator.predict(proj[np.newaxis, :])
        #     if -sdf_tol < pred_sdf < sdf_tol:
        #         gradients[i] = grads[0]
        #         continue
            
        # # 3. if MES is available, directly use that normal
        # if not np.isnan(MES_normals[i]).any():
        #     proj = points[i] - values[i] * MES_normals[i]
        #     dists = np.linalg.norm(proj - ngbrs, axis=1)
        #     inside = dists < (np.abs(values[ngbrs_list[i]])) - float_tol
        #     if not inside.any():  # only clamp to MES normal if the projected point is visible, otherwise it may cause more issues
        #         gradients[i] = MES_normals[i]
        # 4. if no suitable arc point found, keep the original gradient (which is outside visible arcs, filtered out later)
        """ --- Visualization for debugging --- """
        """
        # 可视化采样点
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制当前球的点（加粗）
        ax.scatter(points[i, 0], points[i, 1], points[i, 2], 
                  c='red', marker='*', s=500, label=f'Sphere {i}')
        
        # 绘制该球的前5个邻居
        if i < len(ngbrs_list):
            ngbrs_indices = ngbrs_list[i][:5]  # 只取前5个邻居
            if len(ngbrs_indices) > 0:
                ax.scatter(points[ngbrs_indices, 0], points[ngbrs_indices, 1], points[ngbrs_indices, 2],
                          c='orange', marker='s', s=100, alpha=0.7, label=f'Neighbors (first {len(ngbrs_indices)})')
        
        # 绘制采样点
        if len(sample_pts) > 0:
            ax.scatter(sample_pts[:, 0], sample_pts[:, 1], sample_pts[:, 2], 
                      c='green', marker='.', s=50, alpha=0.8, label=f'Arc samples ({len(sample_pts)} pts)')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Sphere {i}: Arc Sampling Visualization')
        ax.legend()
        plt.tight_layout()
        plt.show()
        
        if len(sample_pts) == 0:
            continue
        """
    print(f"clamp_precision_cnt: {clamp_precision_cnt}")
    print(f"clamp_work_cnt: {clamp_work_cnt}")
    if debug_cnt > 0:
        print(f"\n there are {debug_cnt} samples without any arcs\n")

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

def init_gradients_interp(sdf_points, sdf_values, interpolator : Interpolator, degenerate_pts, batch, ngbrs_list, options : Options):
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
    
    to_train_points = sdf_points.copy()
    to_train_sdf = sdf_values.copy()
    # Add points for degenerate arcs.
    degenerate_pts_to_add = []
    for i, pts in degenerate_pts.items():
        degenerate_pts_to_add.append(pts[0])
    if len(degenerate_pts_to_add) > 0:
        to_train_points = np.vstack([to_train_points, np.array(degenerate_pts_to_add)])
        to_train_sdf = np.append(to_train_sdf, np.zeros(len(degenerate_pts_to_add)))  # The SDF value at the projected point should be 0
    print(f"After adding points for degenerate arcs, total points: {len(to_train_points)}")
    interpolator.fit(to_train_points, to_train_sdf)
    print(f"======== second fit done with input {len(to_train_points)} points (including {len(degenerate_pts_to_add)} degenerate arc points)")

    """ 2. project to the 0-level surface of the interpolator to get better gradient estimation for MES points """
    init_grads = interpolator.sample_best_gradients(sdf_points, sdf_values)
    for i, pts in degenerate_pts.items():
        init_grads[i] = (sdf_points[i] - pts[0]) / (sdf_values[i] + 1e-10)
    print("initial gradient estimation done")
    ## project to 0-level surface code
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

    if options.clamp:
        """ 3. MES normal estimation for non-degenerate points """
        contact_pts, normals = mes_contact.contact_points_from_sdf(sdf_points, sdf_values, debug_level=0)
        options.MES_normals = normals
        mask = ~np.isnan(contact_pts).any(axis=1)
        print(f"{mask.sum()} MES contact points, {len(degenerate_pts)} degenerate points")
        # check the distribution of valid MES normals
        # number of positive sdf_value points that have valid MES normals
        num_pos_valid = np.sum((sdf_values > 0) & mask)
        print(f"{num_pos_valid} positive sdf_value points have valid MES normals")
        num_neg_valid = np.sum((sdf_values < 0) & mask)
        print(f"{num_neg_valid} negative sdf_value points have valid MES normals")
        print(f"{np.sum(sdf_values > 0)} positive sdf_value points in total, {np.sum(sdf_values < 0)} negative sdf_value points in total")

        clamp_gradients_to_arcs(sdf_points, sdf_values, init_grads, degenerate_pts, batch, ngbrs_list, interpolator, options.tolerance, normals)

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

def yongs_algorithm(sdf_points, sdf_values, options : Options):
    gt_gradients, max_iters, gt_mesh = options.gt_gradients, options.max_iters, options.gt_mesh
    if options.tolerance is None:
        from sklearn.neighbors import KDTree
        tree = KDTree(sdf_points)
        dists, _ = tree.query(sdf_points, k=2)  # k=2, 0 distance for k=0 (the point itself)
        mean_spacing = np.median(dists[:, 1])
        clamp_sdf_tol = mean_spacing * 0.5
        tolerance = Tolerance(clamp_sdf_tol=clamp_sdf_tol)
        options.tolerance = tolerance

    """ step 1: initial gradient estimation using MES and degenerate points """
    # collect visible arcs for each point, which will be used to clamp the gradients later
    batch, degenerate_pts, ngbrs_list = get_visible_arcs(sdf_points, sdf_values, epsilon=1e-8)
    if options.turn_off_short_arcs:
        degenerate_pts = {}
    interpolator = PUInterpolator('cubic', overlap=options.interp_overlap, partition=options.interp_partition)  # use PU for better extrapolation, which is important for the initial gradient estimation. We can switch to Duchon for faster fitting in the optimization loop since we only need to evaluate gradients at given points instead of sampling new points.
    if options.interpolator_type == 'Duchon':
        interpolator = DuchonInterpolator('cubic')
    init_grads = init_gradients_interp(sdf_points, sdf_values, interpolator, degenerate_pts, batch, ngbrs_list, options)
    
    if options.use_gt_gradients and gt_gradients is not None:
        init_grads = gt_gradients
    
    # """ step 2: filter invalid points """
    # valid_mask = ~np.isnan(init_grads).any(axis=1)
    # num_valid_points = np.sum(valid_mask)
    """ step 2: project onto surface, then filter invisible points """
    init_projections = sdf_points - sdf_values[:, np.newaxis] * init_grads
    mask = are_points_visible(init_projections, sdf_points, sdf_values, epsilon=tolerance.float_tol)
    # mask = np.ones(len(sdf_points), dtype=bool)
    num_visible_points = np.sum(mask)
    # new_points = np.vstack([sdf_points, init_projections[mask]])
    # new_values = np.concatenate([sdf_values, np.zeros(num_visible_points)])

    """ step 3: fit the interpolator with the original points + projected points """
    interpolator.fit(sdf_points, sdf_values, init_grads, mask)  # refit the interpolator with the initial gradients to get better predictions for the degenerate points
    print(f"======== third fit done with input {num_visible_points + len(sdf_points)} points")
    print(f"Number of visible projected points: {num_visible_points} out of {len(sdf_points)}. percent: {np.mean(mask) * 100:.2f}%")

    # init_zero_contours = interpolator.extract_zero_level_set(bounds=((0, 1), (0, 1)), resolution=resolution)

    """ step 4: iterative optimization """
    # For iterative_projection_3d, we need to pass the degenerate points, batch, and neighbors list for clamping inside the optimization loop
    options.degenerate_pts = degenerate_pts
    options.batch = batch
    options.ngbrs_list = ngbrs_list
    # gradients = opt.iterative_gradient_alignment(sdf_points, sdf_values, init_gradients, interpolator, visible_arcs, degenerate_arcs, num_iter=iters, gt=points)
    gradients, interpolator = opt.iterative_projection_3d(sdf_points, sdf_values, init_grads, interpolator, options, num_iter=max_iters, gt_gradients=gt_gradients)

    # interpolator.fit(sdf_points, sdf_values, gradients, force_recompute=True, use_projection=True)
    init_projections = sdf_points - sdf_values[:, np.newaxis] * gradients
    return interpolator, init_projections, mask

def export_projection_visualization(sdf_points, init_projections, mask, recon_mesh, output_path='out/projection_debug.glb'):
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
        scene.add_geometry(mesh_copy)
    
    num_visible = np.sum(mask)
    num_invisible = np.sum(~mask)
    print(f"\nBuilding GLB scene: {num_visible} visible + {num_invisible} invisible projections...")
    
    for i in range(len(sdf_points)):
        sdf_pt = sdf_points[i]
        proj_pt = init_projections[i]
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

def test_our_method(options : Options, save_npz=False):
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
        mesh, points, distances, gt_gradients = generate_test_mesh_data(path_to_obj, base_name, grid_len=grid_len, save=save_npz)  # Generate new data with 4096 points
        options.gt_gradients = gt_gradients
        options.gt_mesh = mesh

    # Create and fit the interpolator
    timer = time.perf_counter()
    interpolator, init_projections, mask = yongs_algorithm(points, distances, options)
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
    fname = f'interpolant_{grid_len}_{iters}_clamp_{options.interpolator_type}_{options.interp_partition}_ovlp{options.interp_overlap}.obj' if options.clamp else f'interpolant_{grid_len}_{iters}_noclamp_{options.interpolator_type}_{options.interp_partition}_ovlp{options.interp_overlap}.obj'
    if options.use_gt_gradients:
        fname = f'interpolant_{grid_len}_gtgrad_{options.interpolator_type}_{options.interp_partition}_ovlp{options.interp_overlap}.obj'
    recon.export(f'{out_dir}/{fname}')
    
    # Export projection visualization GLB
    if options.export_projections:
        out_path = f'out/projections_{options.name}_{options.grid_len}_{options.max_iters}.glb'
        if options.use_gt_gradients:
            out_path = f'out/projections_{options.name}_{options.grid_len}_gtmesh.glb'
            export_projection_visualization(points, init_projections, mask, mesh, output_path=out_path)
            out_path = f'out/projections_{options.name}_{options.grid_len}_gt.glb'
        export_projection_visualization(points, init_projections, mask, recon, output_path=out_path)

    trimesh.Trimesh(vertices=verts2, faces=faces2).export(f'{out_dir}/sample_points_{grid_len}.obj')    
    print(f"Exported: {out_dir}/{fname}, {out_dir}/sample_points_{grid_len}.obj")
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
            print(f"{mesh_file:<50} against ground truth...", end='')
            print(f"  Hausdorff: {haus:.8f}  Chamfer: {chamfer:.4f}")

if __name__ == "__main__":
    t0 = time.perf_counter()
    batch = False
    if batch:
        for grid_len in [20, 25]:  # 20^3=8000 points, 30^3=27000 points
            for max_iters in [0, 10]:  # 0 means no optimization, only initial gradient estimation and projection
                for interp_partition in ['sphere', 'box']:
                    for overlap in [0.2, 0.5]:
                        ext = 1 if interp_partition == 'sphere' else 2
                        options = Options(name='horse', grid_len=grid_len, max_iters=max_iters, clamp=True, 
                                        export_short_arcs=True, export_projections=False, 
                                        use_gt_gradients=False, interpolator_type='PU', interp_partition=interp_partition, 
                                        overlap=overlap*ext)
                        # plt = test_mesh(grid_len=20, path_to_obj='examples/holes.obj')
                        plt = test_our_method(options)
            test_rfta(options)
    else:
        options = Options(name='horse', grid_len=20, max_iters=10, clamp=True, 
                        export_short_arcs=True, export_projections=False, 
                        use_gt_gradients=False, interpolator_type='PU', interp_partition='sphere', 
                        overlap=0.2)
        # # plt = test_mesh(grid_len=20, path_to_obj='examples/holes.obj')
        plt = test_our_method(options)
        # test_rfta(options)
    check_mesh_error(f'out/{options.name}', f'examples/{options.name}.obj')

    elapsed = time.perf_counter() - t0
    print(f"  ⏱  {'Total execution time':<30} {elapsed:>7.2f} s")
