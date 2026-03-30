"""
Benchmark: compute exposed arcs/points for spheres generated from
real mesh SDF data.

Usage:
    python bench_mesh.py path/to/mesh.obj [grid_len]

Requires: trimesh, igl (libigl python bindings), numpy, scipy
Also requires: sphere_exposed_ext (compile sphere_exposed_ext.c first)

Compile the C extension:
    gcc -O3 -shared -fPIC \
      -o sphere_exposed_ext$(python3-config --extension-suffix) \
      sphere_exposed_ext.c \
      -I$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))") \
      -I$(python3 -c "import numpy; print(numpy.get_include())") \
      $(python3-config --ldflags) -lm
"""

import numpy as np
import time
import sys
import sphere_intersect

# Try to import C extension; fall back to Python
try:
    import sphere_exposed_ext as ext
    _HAS_C_EXT = True
    print("Using C extension for exposed arc computation")
except ImportError:
    _HAS_C_EXT = False
    print("C extension not found, using pure Python (slower)")

from robust_exposed import (
    Sphere, compute_all_caps, compute_exposed_arcs_robust,
)


# ═══════════════════════════════════════════════════════════════════
# Mesh SDF data generation (from user's code)
# ═══════════════════════════════════════════════════════════════════

def generate_test_mesh_data(path_to_mesh, grid_len=10):
    """
    Loads a mesh from the given path and computes signed distances
    and gradients on a regular grid around the mesh.

    Returns:
        mesh: trimesh.Trimesh
        points: (N, 3) array
        distances: (N,) array of signed distance values
        gradients: (N, 3) array
    """
    import trimesh
    import igl

    mesh = trimesh.load(path_to_mesh)

    # Normalize mesh to unit cube
    vmin = np.min(mesh.vertices, axis=0)
    vmax = np.max(mesh.vertices, axis=0)
    mesh.vertices -= (vmin + vmax) / 2
    mesh.vertices /= np.max(vmax - vmin)

    # Generate grid points around mesh bounding box
    bbox_min = np.min(mesh.vertices, axis=0) - 0.1
    bbox_max = np.max(mesh.vertices, axis=0) + 0.1
    x = np.linspace(bbox_min[0], bbox_max[0], grid_len)
    y = np.linspace(bbox_min[1], bbox_max[1], grid_len)
    z = np.linspace(bbox_min[2], bbox_max[2], grid_len)
    X, Y, Z = np.meshgrid(x, y, z)
    points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    # Compute distances to mesh
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int32)
    sq_dists, face_ids, closest = igl.point_mesh_squared_distance(points, V, F)
    distances = np.sqrt(sq_dists)

    gradients = points - closest
    norm_temp = np.linalg.norm(gradients, axis=1, keepdims=True)
    norm_temp[np.abs(norm_temp) <= 1e-8] = 1.0
    gradients /= norm_temp

    # Filter out points too close to surface
    mask = np.abs(distances) > 1e-8
    points = points[mask]
    distances = distances[mask]
    gradients = gradients[mask]

    # Use winding number for inside/outside
    W = igl.winding_number(V, F, points)
    inside = W > 0.5
    distances[inside] *= -1.0
    gradients[inside] *= -1.0

    return mesh, points, distances, gradients


# ═══════════════════════════════════════════════════════════════════
# Neighbor finding
# ═══════════════════════════════════════════════════════════════════

def find_all_neighbors_cdist(centers, radii):
    """Fast neighbor finding using scipy cdist."""
    from scipy.spatial.distance import cdist
    dists = cdist(centers, centers)
    sum_r = radii[:, None] + radii[None, :]
    mask = dists < sum_r
    np.fill_diagonal(mask, False)
    return [np.where(mask[i])[0] for i in range(len(centers))]


def find_all_neighbors_batch(centers, radii, batch_size=500):
    """Batched neighbor finding (no scipy dependency)."""
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
    """Use cdist if scipy available, else batched."""
    try:
        offsets, neighbors = sphere_intersect.find_intersections(centers, radii)
        return [neighbors[offsets[i]:offsets[i+1]] for i in range(len(offsets)-1)]
    except ImportError:
        if len(centers) < 30**3:
            return find_all_neighbors_cdist(centers, radii)
        return find_all_neighbors_batch(centers, radii)


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
        # Fallback: synthetic torus data
        print("No mesh path provided, using synthetic torus data")
        from bench_compare import generate_spheres
        points, sdf_values = generate_spheres(grid_len ** 3, seed=42)
    t_gen = time.perf_counter() - t0

    centers = np.ascontiguousarray(points, dtype=np.float64)
    radii = np.abs(np.ascontiguousarray(sdf_values, dtype=np.float64))
    n_spheres = len(centers)

    print("=" * 70)
    print(f"Benchmark: {n_spheres} spheres (grid_len={grid_len})")
    print("=" * 70)
    print(f"\n[0] Data generation: {t_gen:.3f}s")
    print(f"    Radius range: [{radii.min():.6f}, {radii.max():.6f}]")

    # ── Neighbors ──
    t0 = time.perf_counter()
    nbr_list = find_all_neighbors(centers, radii)
    t_nbr = time.perf_counter() - t0
    nbr_counts = np.array([len(v) for v in nbr_list])
    print(f"[1] Neighbors:       {t_nbr:.3f}s  "
          f"(mean={nbr_counts.mean():.1f}, "
          f"median={np.median(nbr_counts):.0f}, "
          f"max={nbr_counts.max()})")

    # ── Compute exposed arcs ──
    t0 = time.perf_counter()

    if _HAS_C_EXT:
        # ── C extension batch (v3 CSR API) ──
        # Convert neighbor list to CSR format
        if any(len(nb) > 0 for nb in nbr_list):
            indices = np.concatenate([nb.astype(np.int64) for nb in nbr_list])
        else:
            indices = np.array([], dtype=np.int64)
        offsets = np.zeros(n_spheres + 1, dtype=np.int64)
        for i, nb in enumerate(nbr_list):
            offsets[i + 1] = offsets[i] + len(nb)

        results = ext.compute_exposed_batch(centers, radii, indices, offsets)
        t_compute = time.perf_counter() - t0

        # v3 returns a dict of numpy arrays, not a list of dicts
        n_caps_arr = results['n_caps']
        n_arcs_arr = results['n_arcs']
        n_pts_arr  = results['n_points']
        arc_deg_arr = np.degrees(results['total_arc'])

    else:
        # ── Pure Python ──
        n_caps_arr = np.zeros(n_spheres, dtype=int)
        n_arcs_arr = np.zeros(n_spheres, dtype=int)
        n_pts_arr = np.zeros(n_spheres, dtype=int)
        arc_deg_arr = np.zeros(n_spheres)
        timings = np.zeros(n_spheres)

        report_every = max(n_spheres // 10, 1)

        for i in range(n_spheres):
            ts = time.perf_counter()
            nbr = nbr_list[i]
            if len(nbr) == 0:
                timings[i] = time.perf_counter() - ts
                continue

            main = Sphere(centers[i], radii[i])
            others = [Sphere(centers[j], radii[j]) for j in nbr]
            caps = compute_all_caps(main, others)
            arcs_by_cap, exposed_points = compute_exposed_arcs_robust(main, caps)

            total_arc = sum(sum(te - ts_ for ts_, te in arcs)
                            for arcs in arcs_by_cap.values())
            n_arcs = sum(len(arcs) for arcs in arcs_by_cap.values())

            n_caps_arr[i] = len(caps)
            n_arcs_arr[i] = n_arcs
            n_pts_arr[i] = len(exposed_points)
            arc_deg_arr[i] = np.degrees(total_arc)
            timings[i] = time.perf_counter() - ts

            if (i + 1) % report_every == 0:
                elapsed = time.perf_counter() - t0
                rate = (i + 1) / elapsed
                eta = (n_spheres - i - 1) / rate
                print(f"  [{i+1:6d}/{n_spheres}]  "
                      f"{rate:.0f} sph/s  ETA {eta:.1f}s")

        t_compute = time.perf_counter() - t0

    # ── Stats ──
    fully_covered = int(np.sum((n_arcs_arr == 0) & (n_pts_arr == 0)
                                & (nbr_counts > 0)))
    has_arcs = int(np.sum(n_arcs_arr > 0))
    has_pts = int(np.sum(n_pts_arr > 0))
    no_nbr = int(np.sum(nbr_counts == 0))
    t_all = t_gen + t_nbr + t_compute

    method = "C extension" if _HAS_C_EXT else "Python"
    print(f"[2] Compute ({method}): {t_compute:.3f}s  "
          f"({t_compute/n_spheres*1000:.3f} ms/sphere)")

    print(f"\n{'─' * 70}")
    print(f"RESULTS  ({n_spheres} spheres)")
    print(f"{'─' * 70}")
    print(f"[0] Generate:       {t_gen:.3f}s")
    print(f"[1] Neighbors:      {t_nbr:.3f}s")
    print(f"[2] Compute:        {t_compute:.3f}s")
    print(f"    Total:          {t_all:.3f}s")
    print(f"    Throughput:     {n_spheres / t_all:.0f} spheres/s")

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
        print(f"Arc angle (°):  mean={arc_deg_arr[has_arc_mask].mean():.1f}  "
              f"min={arc_deg_arr[has_arc_mask].min():.1f}  "
              f"max={arc_deg_arr[has_arc_mask].max():.1f}")

    print(f"\nClassification:")
    print(f"  no neighbors:     {no_nbr:5d}  ({no_nbr/n_spheres*100:.1f}%)")
    print(f"  fully covered:    {fully_covered:5d}  ({fully_covered/n_spheres*100:.1f}%)")
    print(f"  has arcs:         {has_arcs:5d}  ({has_arcs/n_spheres*100:.1f}%)")
    print(f"  has points only:  {has_pts:5d}  ({has_pts/n_spheres*100:.1f}%)")

    if n_pts_arr.max() > 0:
        has_pt_mask = n_pts_arr > 0
        print(f"\nExposed points (degenerate):")
        print(f"  spheres with pts: {has_pts}")
        print(f"  mean count:       {n_pts_arr[has_pt_mask].mean():.1f}")
        print(f"  max count:        {n_pts_arr.max()}")

    # Top 5 by cap count (proxy for slowest in C mode)
    print(f"\nTop 5 by cap count:")
    top5 = np.argsort(n_caps_arr)[-5:][::-1]
    for rank, idx in enumerate(top5):
        print(f"  #{rank+1}  sphere {idx:5d}: "
              f"nbrs={nbr_counts[idx]}  caps={n_caps_arr[idx]}  "
              f"arcs={n_arcs_arr[idx]}  pts={n_pts_arr[idx]}")

    return {
        'n_spheres': n_spheres,
        'centers': centers,
        'radii': radii,
        'nbr_list': nbr_list,
        'batch_results': results if _HAS_C_EXT else None,
        'n_caps': n_caps_arr,
        'n_arcs': n_arcs_arr,
        'n_pts': n_pts_arr,
        'arc_deg': arc_deg_arr,
    }


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
        print(f"  python {sys.argv[0]} dragon.obj 30  # 27000 spheres")
        sys.exit(1)

    mesh_path = sys.argv[1]
    grid_len = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    benchmark(path_to_mesh=mesh_path, grid_len=grid_len)