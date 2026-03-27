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
    for i, angle in degenerate_arcs.items():
        # For points with degenerate arcs, set gradient directly toward the angle
        grad = np.array([-np.cos(angle), -np.sin(angle)]) if sdf_values[i] > 0 else np.array([np.cos(angle), np.sin(angle)])
        # Add a new point on the surface along this gradient direction
        new_point = sdf_points[i] - sdf_values[i] * grad
        to_train_points = np.vstack([to_train_points, new_point])
        to_train_sdf = np.append(to_train_sdf, 0)  # The SDF value at the projected point should be 0
    print(f"After adding points for degenerate arcs, total points: {len(to_train_points)}")
    if to_train_points.shape[0] > 10000:
        mask = np.abs(sdf_values) < 0.1
        to_train_points = to_train_points[mask]
        to_train_sdf = to_train_sdf[mask]
    interpolator.fit(to_train_points, to_train_sdf)
    print(f"first fit done with {len(to_train_points)} points")
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
    
    """ step 2: project onto surface, then filter visible points """
    init_projections = sdf_points - sdf_values[:, np.newaxis] * init_gradients
    mask = are_points_visible(init_projections, sdf_points, sdf_values)
    num_visible_points = np.sum(mask)
    print(f"Number of visible projected points: {num_visible_points} out of {len(sdf_points)}. percent: {np.mean(mask) * 100:.2f}%")
    new_points = init_projections[mask]
    
    """ step 3: refit the interpolator with the original points + projected points """
    mask2 = np.abs(sdf_values) < 0.1 if sdf_points.shape[0] > 10000 else np.ones(len(sdf_points), dtype=bool)
    to_train_points = sdf_points[mask2]
    to_train_sdf = sdf_values[mask2]
    to_train_points = np.vstack([to_train_points, new_points])
    to_train_sdf = np.append(to_train_sdf, np.zeros(len(new_points)))  # The SDF value at the projected point should be 0
    
    interpolator.fit(to_train_points, to_train_sdf, force_recompute=True, use_projection=False)

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
    for mesh_file in meshes:
        if mesh_file.endswith('.obj'):
            mesh = trimesh.load(os.path.join(dir_to_meshes, mesh_file))
            haus, chamfer = mesh_distances(mesh, gt_mesh)
            print(f"{mesh_file:<30} against ground truth...", end='')
            print(f"  Hausdorff: {haus:.4f}  Chamfer: {chamfer:.4f}")
    return haus, chamfer

if __name__ == "__main__":
    t0 = time.perf_counter()
    # plt = test_mesh(grid_len=20, path_to_obj='examples/holes.obj')
    plt = test_our_method(grid_len=10, path_to_obj='examples/horse.obj', iters=10)
    # check_mesh_error('out/eiffel', 'examples/eiffel.obj')

    elapsed = time.perf_counter() - t0
    print(f"  ⏱  {'Total execution time':<30} {elapsed:>7.2f} s")
