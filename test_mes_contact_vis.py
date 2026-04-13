"""
Visualize mes_contact results: input spheres and their contact points in 3D.
Exports mes_contact_vis.glb for viewing in any 3D viewer.
"""

import numpy as np
import trimesh
import mes_contact
import igl
import time

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

# ── Input spheres ─────────────────────────────────────────────────────────────
mesh, points, sdf_values, gradients = generate_test_mesh_data("examples/archer.obj", "archer", grid_len=35)
# points = np.array([
#     [ 1.0,  1.0,  0],
#     [ -1.0,  1.0,  0],
#     [ 1.0,  -1.0,  0],
#     [ -1.0,  -1.0,  0],
#     [ 3.0,  0,  0.0],
#     [ 3.0, -3,  0.0],
#     [ .0, 0,  10.0],

#     [ 1.0,  1.0,  0],
#     [ -1.0,  1.0,  0],
#     [ 1.0,  -1.0,  0],
#     [ -1.0,  -1.0,  0],
#     [ 3.0,  0,  0.0],
#     [ 3.0, -3,  0.0],
#     [ .0, 0,  10.0],
# ], dtype=np.float64)
# points[7:] += np.array([50, 0, 0])  # Shift last 7 points up by 0.5 in z
# sdf_values = np.array([0.8, 0.8, 0.8, 0.8, 3, 3, 7], dtype=np.float64)
# sdf_values = np.concatenate([sdf_values, -sdf_values])  # First 7 points outside, next 7 inside
N = len(points)

# Dump same input to CSV so the standalone reference binary / bench can read it
np.savetxt("/tmp/mes_bench_input.csv",
           np.concatenate([points, sdf_values[:, None]], axis=1),
           delimiter=",")
print(f"Dumped {len(points)} spheres to /tmp/mes_bench_input.csv")

# ── Run mes_contact ───────────────────────────────────────────────────────────
t0 = time.time()
contact_pts, normals = mes_contact.contact_points_from_sdf(points, sdf_values, debug_level=0)
mask = ~np.isnan(contact_pts).any(axis=1)
valid_idx = np.where(mask)[0]
valid_cp = contact_pts[mask]

print(f"Input points:        {N}")
print(f"Points with contact: {mask.sum()}")
print(f"mes_contact time:    {time.time() - t0:.2f} seconds")
if True:
    # ── Compare with reference MESReconstruction from maximal-empty-spheres fork ──
    import sys
    sys.path.insert(0, '/Users/yongcheng/Documents/phd/research/sdf/maximal-empty-spheres/cgal')
    from EmptySpheresReconstruction import MESReconstruction

    t0 = time.time()
    mes_verts, mes_faces, mes_pts, mes_normals = MESReconstruction(
        points, sdf_values,
        screening_weight=1.0,
        return_oriented_points=True,
        save_folder='/tmp',
    )
    print(f"MESReconstruction time (incl. PSR): {time.time() - t0:.2f} seconds")
    print(f"  oriented points from CGAL: {len(mes_pts)}")

# # Check: how many contact points are inside another sphere?
# radii = np.abs(sdf_values)
# n_contained = 0
# for k, i in enumerate(valid_idx):
#     cp = valid_cp[k]
#     dists = np.linalg.norm(points - cp[None, :], axis=1)
#     dists[i] = np.inf  # skip self
#     if (dists < radii).any():
#         n_contained += 1
# print(f"Contact points contained by another sphere: {n_contained}/{len(valid_cp)}")
# exit(0)
# # ── Build GLB scene ───────────────────────────────────────────────────────────
# scene = trimesh.Scene()
# radii = np.abs(sdf_values)

# # Input spheres (transparent)
# for i in range(N):
#     sphere = trimesh.creation.icosphere(subdivisions=3, radius=radii[i])
#     sphere.apply_translation(points[i])
#     base_color = [68, 136, 204] if sdf_values[i] >= 0 else [204, 136, 68]
#     mat = trimesh.visual.material.PBRMaterial(
#         baseColorFactor=base_color + [80],
#         alphaMode='BLEND',
#     )
#     sphere.visual = trimesh.visual.TextureVisuals(material=mat)
#     scene.add_geometry(sphere, node_name=f'sphere_{i}')

# # Sphere centers (small red spheres)
# for i in range(N):
#     center = trimesh.creation.icosphere(subdivisions=2, radius=0.005)
#     center.apply_translation(points[i])
#     center.visual.face_colors = [0, 0, 0, 255]
#     scene.add_geometry(center, node_name=f'center_{i}')

# # Contact points (red spheres)
# for k, i in enumerate(valid_idx):
#     cp = valid_cp[k]
#     pt = trimesh.creation.icosphere(subdivisions=2, radius=0.08)
#     pt.apply_translation(cp)
#     pt.visual.face_colors = [255, 0, 0, 255]
#     scene.add_geometry(pt, node_name=f'contact_{i}')

# out = 'test_mes_contact_vis.glb'
# scene.export(out)
# print(f"Saved: {out}")
