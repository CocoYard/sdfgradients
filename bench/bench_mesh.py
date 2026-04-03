"""
Benchmark: compute exposed arcs/points for spheres generated from
real mesh SDF data.

Usage:
    python bench_mesh.py path/to/mesh.obj [grid_len]

Requires: trimesh, igl (libigl python bindings), numpy, scipy
Also requires: sphere_exposed_pybind  (compile sphere_exposed_pybind.cpp first)

Compile:
    c++ -O3 -shared -fPIC -undefined dynamic_lookup -std=c++17 \
      $(python -m pybind11 --includes) \
      -o sphere_exposed_pybind$(python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))") \
      sphere_exposed_pybind.cpp -lm
"""

import numpy as np
import time
import sys
import sphere_intersect

# Try to import pybind11 extension
try:
    import sphere_exposed_pybind as ext
    _HAS_C_EXT = True
    print("Using sphere_exposed_pybind for exposed arc computation")
except ImportError:
    _HAS_C_EXT = False
    print("sphere_exposed_pybind not found, using pure Python (slower)")

from exposed_region_clip import (
    Sphere, compute_all_caps, compute_exposed_region,
)


# ═══════════════════════════════════════════════════════════════════
# Mesh SDF data generation
# ═══════════════════════════════════════════════════════════════════

def generate_test_mesh_data(path_to_mesh, grid_len=10):
    import trimesh
    import igl

    mesh = trimesh.load(path_to_mesh)

    vmin = np.min(mesh.vertices, axis=0)
    vmax = np.max(mesh.vertices, axis=0)
    mesh.vertices -= (vmin + vmax) / 2
    mesh.vertices /= np.max(vmax - vmin)

    bbox_min = np.min(mesh.vertices, axis=0) - 0.1
    bbox_max = np.max(mesh.vertices, axis=0) + 0.1
    x = np.linspace(bbox_min[0], bbox_max[0], grid_len)
    y = np.linspace(bbox_min[1], bbox_max[1], grid_len)
    z = np.linspace(bbox_min[2], bbox_max[2], grid_len)
    X, Y, Z = np.meshgrid(x, y, z)
    points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int32)
    sq_dists, face_ids, closest = igl.point_mesh_squared_distance(points, V, F)
    distances = np.sqrt(sq_dists)

    gradients = points - closest
    norm_temp = np.linalg.norm(gradients, axis=1, keepdims=True)
    norm_temp[np.abs(norm_temp) <= 1e-8] = 1.0
    gradients /= norm_temp

    mask = np.abs(distances) > 1e-8
    points    = points[mask]
    distances = distances[mask]
    gradients = gradients[mask]

    W = igl.winding_number(V, F, points)
    inside = W > 0.5
    distances[inside] *= -1.0
    gradients[inside] *= -1.0

    return mesh, points, distances, gradients


# ═══════════════════════════════════════════════════════════════════
# Neighbor finding
# ═══════════════════════════════════════════════════════════════════

def find_all_neighbors_cdist(centers, radii):
    from scipy.spatial.distance import cdist
    dists = cdist(centers, centers)
    sum_r = radii[:, None] + radii[None, :]
    mask = dists < sum_r
    np.fill_diagonal(mask, False)
    return [np.where(mask[i])[0] for i in range(len(centers))]


def find_all_neighbors_batch(centers, radii, batch_size=500):
    n = len(centers)
    neighbors = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        diff = centers[start:end, None, :] - centers[None, :, :]
        dists = np.linalg.norm(diff, axis=2)
        sum_r = radii[start:end, None] + radii[None, :]
        mask = dists < sum_r
        for li in range(end - start):
            gi = start + li
            mask[li, gi] = False
            neighbors.append(np.where(mask[li])[0])
    return neighbors


def find_all_neighbors(centers, radii):
    try:
        csr_offsets, csr_neighbors = sphere_intersect.find_intersections(centers, radii)
        nbr_list = [csr_neighbors[csr_offsets[i]:csr_offsets[i+1]]
                    for i in range(len(csr_offsets) - 1)]
        return nbr_list, csr_neighbors, csr_offsets
    except ImportError:
        if len(centers) < 30**3:
            nbr_list = find_all_neighbors_cdist(centers, radii)
        else:
            nbr_list = find_all_neighbors_batch(centers, radii)
        return nbr_list, None, None


# ═══════════════════════════════════════════════════════════════════
# Benchmark
# ═══════════════════════════════════════════════════════════════════

def benchmark(path_to_mesh=None, grid_len=20):
    # ── Generate data ──
    t0 = time.perf_counter()
    if path_to_mesh:
        mesh, points, sdf_values, gradients = generate_test_mesh_data(
            path_to_mesh, grid_len=grid_len)
    else:
        print("No mesh path provided, using synthetic torus data")
        from bench_compare import generate_spheres
        points, sdf_values = generate_spheres(grid_len ** 3, seed=42)
    t_gen = time.perf_counter() - t0

    centers  = np.ascontiguousarray(points,     dtype=np.float64)
    radii    = np.abs(np.ascontiguousarray(sdf_values, dtype=np.float64))
    n_spheres = len(centers)

    print("=" * 70)
    print(f"Benchmark: {n_spheres} spheres (grid_len={grid_len})")
    print("=" * 70)
    print(f"\n[0] Data generation: {t_gen:.3f}s")
    print(f"    Radius range: [{radii.min():.6f}, {radii.max():.6f}]")

    # ── Neighbors ──
    t0 = time.perf_counter()
    nbr_list, csr_neighbors, csr_offsets = find_all_neighbors(centers, radii)
    t_nbr = time.perf_counter() - t0
    nbr_counts = np.array([len(v) for v in nbr_list])
    print(f"[1] Neighbors:       {t_nbr:.3f}s  "
          f"(mean={nbr_counts.mean():.1f}, "
          f"median={np.median(nbr_counts):.0f}, "
          f"max={nbr_counts.max()})")

    # ── Compute exposed arcs ──
    t0 = time.perf_counter()
    pts_arr = [None] * n_spheres   # per-sphere exposed points (Python path only)

    if _HAS_C_EXT:
        # ── pybind11 batch ──
        if csr_neighbors is not None:
            indices = csr_neighbors.astype(np.int64)
            offsets = csr_offsets.astype(np.int64)
        else:
            indices = (np.concatenate([nb.astype(np.int64) for nb in nbr_list])
                       if any(len(nb) > 0 for nb in nbr_list)
                       else np.array([], dtype=np.int64))
            offsets = np.zeros(n_spheres + 1, dtype=np.int64)
            for i, nb in enumerate(nbr_list):
                offsets[i + 1] = offsets[i] + len(nb)

        results = ext.compute_exposed_batch(centers, radii, indices, offsets)
        t_compute = time.perf_counter() - t0

        # ── pybind 版 key 名 ──────────────────────────────────────
        # n_caps        (N,) int32   per-sphere compacted cap 数
        # n_arcs        (N,) int32   per-sphere 弧段数
        # n_points      (N,) int32   per-sphere 退化点数
        # total_arc     (N,) float64 per-sphere 弧度总和
        # arc_sphere_idx / arc_cap_idx / arc_start / arc_end  — 展平弧段
        # point_sphere_idx / point_positions                  — 展平退化点
        # cap_sphere_idx / cap_id                             — 展平 cap 索引
        # cap_normals (C,3) / cap_d (C,) / cap_centers (C,3)
        # cap_radii (C,)   / cap_u (C,3) / cap_v (C,3)
        # ─────────────────────────────────────────────────────────

        n_caps_arr  = results['n_caps']
        n_arcs_arr  = results['n_arcs']
        n_pts_arr   = results['n_points']
        arc_deg_arr = np.degrees(results['total_arc'])

        total_caps = len(results['cap_sphere_idx'])
        print(f"\n  Caps: {total_caps} total across {n_spheres} spheres")
        print(f"  Per-sphere cap counts: min={n_caps_arr.min()}, "
              f"max={n_caps_arr.max()}, "
              f"mean={n_caps_arr.mean():.1f}, "
              f"total={n_caps_arr.sum()}")

        # cap geometry（pybind 版字段名）
        print(f"  Cap arrays: "
              f"normals {results['cap_normals'].shape}, "
              f"centers {results['cap_centers'].shape}, "
              f"radii {results['cap_radii'].shape}")

        if total_caps > 0:
            print(f"  circle_radius range: "
                  f"[{results['cap_radii'].min():.4f}, "
                  f"{results['cap_radii'].max():.4f}]")
            # Show first few caps
            n_show = min(3, total_caps)
            for j in range(n_show):
                si  = results['cap_sphere_idx'][j]
                cid = results['cap_id'][j]
                cr  = results['cap_radii'][j]
                n   = results['cap_normals'][j]
                print(f"    cap[{j}]: sphere={si}, cap_id={cid}, "
                      f"circle_r={cr:.4f}, "
                      f"normal=[{n[0]:.3f},{n[1]:.3f},{n[2]:.3f}]")

        # Degen points
        if len(results['point_positions']) > 0:
            pt_positions  = results['point_positions']   # (P,3)
            pt_sphere_idx = results['point_sphere_idx']  # (P,)
            for i in range(n_spheres):
                mask = pt_sphere_idx == i
                if np.any(mask):
                    pts_arr[i] = pt_positions[mask]

    else:
        # ── Pure Python ──
        n_caps_arr  = np.zeros(n_spheres, dtype=int)
        n_arcs_arr  = np.zeros(n_spheres, dtype=int)
        n_pts_arr   = np.zeros(n_spheres, dtype=int)
        arc_deg_arr = np.zeros(n_spheres)
        timings     = np.zeros(n_spheres)

        report_every = max(n_spheres // 10, 1)

        for i in range(n_spheres):
            ts   = time.perf_counter()
            nbr  = nbr_list[i]
            if len(nbr) == 0:
                timings[i] = time.perf_counter() - ts
                continue

            main   = Sphere(centers[i], radii[i])
            others = [Sphere(centers[j], radii[j]) for j in nbr]
            caps   = compute_all_caps(main, others)
            arcs_by_cap, exposed_points = compute_exposed_region(main, caps)

            total_arc = sum(sum(arc.length() for arc in arcs)
                            for arcs in arcs_by_cap.values())
            n_arcs = sum(len(arcs) for arcs in arcs_by_cap.values())

            n_caps_arr[i]  = len(caps)
            n_arcs_arr[i]  = n_arcs
            n_pts_arr[i]   = len(exposed_points)
            pts_arr[i]     = exposed_points
            arc_deg_arr[i] = np.degrees(total_arc)
            timings[i]     = time.perf_counter() - ts

            if (i + 1) % report_every == 0:
                elapsed = time.perf_counter() - t0
                rate    = (i + 1) / elapsed
                eta     = (n_spheres - i - 1) / rate
                print(f"  [{i+1:6d}/{n_spheres}]  {rate:.0f} sph/s  ETA {eta:.1f}s")

        t_compute = time.perf_counter() - t0

    # ── GLTF export (only when mesh available + have degen points) ──
    if path_to_mesh:
        import trimesh
        from trimesh.visual.material import PBRMaterial
        import os

        scene = trimesh.Scene()
        ref_mesh = mesh.copy()
        ref_mesh.visual.material = PBRMaterial(
            baseColorFactor=[200, 200, 200, 255], alphaMode='OPAQUE')
        scene.add_geometry(ref_mesh)

        print("\nBuilding GLTF scene with exposed points...")
        for i in range(n_spheres):
            if n_pts_arr[i] > 0 and pts_arr[i] is not None:
                center = centers[i]
                for j in range(n_pts_arr[i]):
                    point = pts_arr[i][j]
                    pt_geom = trimesh.primitives.Sphere(radius=0.001, center=point)
                    pt_geom.visual.material = PBRMaterial(
                        baseColorFactor=[0, 0, 255, 255],
                        emissiveFactor=[0.0, 0.0, 1.0],
                        alphaMode='OPAQUE')
                    scene.add_geometry(pt_geom)
                    line_geom = trimesh.load_path([center, point])
                    for e in line_geom.entities:
                        e.color = [255, 0, 0, 255]
                    scene.add_geometry(line_geom)

        name = os.path.splitext(os.path.basename(path_to_mesh))[0]
        export_path = f'a_{name}_{grid_len}.glb'
        scene.export(export_path)
        print(f"Exported to {export_path}")

    # ── Stats ──
    fully_covered = int(np.sum(
        (n_arcs_arr == 0) & (n_pts_arr == 0) & (nbr_counts > 0)))
    has_arcs = int(np.sum(n_arcs_arr > 0))
    has_pts  = int(np.sum(n_pts_arr  > 0))
    no_nbr   = int(np.sum(nbr_counts == 0))
    t_all    = t_gen + t_nbr + t_compute

    method = "sphere_exposed_pybind" if _HAS_C_EXT else "Python"
    print(f"[2] Compute ({method}): {t_compute:.3f}s  "
          f"({t_compute / n_spheres * 1000:.3f} ms/sphere)")

    print(f"\n{'─' * 70}")
    print(f"RESULTS  ({n_spheres} spheres)")
    print(f"{'─' * 70}")
    print(f"[0] Generate:   {t_gen:.3f}s")
    print(f"[1] Neighbors:  {t_nbr:.3f}s")
    print(f"[2] Compute:    {t_compute:.3f}s")
    print(f"    Total:      {t_all:.3f}s")
    print(f"    Throughput: {n_spheres / t_all:.0f} spheres/s")

    if not _HAS_C_EXT:
        print(f"\nPer-sphere timing (ms):")
        print(f"  mean={timings.mean()*1000:.3f}  "
              f"median={np.median(timings)*1000:.3f}  "
              f"p95={np.percentile(timings, 95)*1000:.3f}  "
              f"max={timings.max()*1000:.3f}")

    print(f"\nCaps/sphere:  mean={n_caps_arr.mean():.1f}  "
          f"median={np.median(n_caps_arr):.0f}  max={n_caps_arr.max()}")
    print(f"Arcs/sphere:  mean={n_arcs_arr.mean():.1f}  "
          f"median={np.median(n_arcs_arr):.0f}  max={n_arcs_arr.max()}")

    has_arc_mask = n_arcs_arr > 0
    if has_arc_mask.any():
        print(f"Arc angle (°): mean={arc_deg_arr[has_arc_mask].mean():.1f}  "
              f"min={arc_deg_arr[has_arc_mask].min():.1f}  "
              f"max={arc_deg_arr[has_arc_mask].max():.1f}")

    print(f"\nClassification:")
    print(f"  no neighbors:    {no_nbr:5d}  ({no_nbr / n_spheres * 100:.1f}%)")
    print(f"  fully covered:   {fully_covered:5d}  ({fully_covered / n_spheres * 100:.1f}%)")
    print(f"  has arcs:        {has_arcs:5d}  ({has_arcs / n_spheres * 100:.1f}%)")
    print(f"  has points only: {has_pts:5d}  ({has_pts / n_spheres * 100:.1f}%)")

    if n_pts_arr.max() > 0:
        has_pt_mask = n_pts_arr > 0
        print(f"\nExposed points (degenerate):")
        print(f"  spheres with pts: {has_pts}")
        print(f"  mean count:       {n_pts_arr[has_pt_mask].mean():.1f}")
        print(f"  max count:        {n_pts_arr.max()}")

    print(f"\nTop 5 by cap count:")
    top5 = np.argsort(n_caps_arr)[-5:][::-1]
    for rank, idx in enumerate(top5):
        print(f"  #{rank+1}  sphere {idx:5d}: "
              f"nbrs={nbr_counts[idx]}  caps={n_caps_arr[idx]}  "
              f"arcs={n_arcs_arr[idx]}  pts={n_pts_arr[idx]}")

    results_dict = {
        'n_spheres': n_spheres,
        'centers':   centers,
        'radii':     radii,
        'nbr_list':  nbr_list,
        'n_caps':    n_caps_arr,
        'n_arcs':    n_arcs_arr,
        'n_pts':     n_pts_arr,
        'arc_deg':   arc_deg_arr,
    }
    if _HAS_C_EXT:
        results_dict['batch_results'] = results

    return results_dict


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} path/to/mesh.obj [grid_len]")
        print(f"       grid_len defaults to 20 ({20**3}=8000 spheres)")
        print()
        print("Example:")
        print(f"  python {sys.argv[0]} bunny.obj 20")
        print(f"  python {sys.argv[0]} dragon.obj 30")
        sys.exit(1)

    mesh_path = sys.argv[1]
    grid_len  = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    benchmark(path_to_mesh=mesh_path, grid_len=grid_len)