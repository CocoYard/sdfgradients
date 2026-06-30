"""
Build the solid-300 index used by the SLURM baseline batch.

Filter (intersection):
  manifold=True             (= vertex_manifold ∧ edge_manifold)
  closed=True               (no boundary edges)
  solid=True                (PWN with outward-facing normals — winding > 0 inside)
  self_intersecting=False
  num_vertices ∈ [500, 200_000]

After filtering ~3310 candidates remain. We sort by ``file_id`` ascending
and take the first 300 — fully deterministic, re-runs always produce the
same list (modulo thingi10k package version).

Why these criteria, vs. the original ``prepare_thingi10k.py`` (manifold+closed
only): some closed-manifold meshes have **inward-facing normals**, which makes
``igl.winding_number`` return negative values inside the mesh. The
``generate_test_mesh_data`` SDF computation uses ``mask = W > 0.5`` to flip
distances negative inside, so inward-normal meshes get all-positive SDF →
``skimage.marching_cubes(level=0)`` fails with "Surface level out of range".
``solid=True`` excludes such meshes at the dataset level.

Usage:
    python scripts/prepare_solid300.py
        # writes /scratch/ycheng27/new_solid/index.csv with 300 entries
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


CACHE_DIR = '/scratch/ycheng27/thingi10k'  # reused from prepare_thingi10k


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-dir', default=CACHE_DIR)
    p.add_argument('--index-path',
                   default='/scratch/ycheng27/new_solid/index.csv')
    p.add_argument('--num-keep', type=int, default=300)
    p.add_argument('--min-vertices', type=int, default=500)
    p.add_argument('--max-vertices', type=int, default=200_000)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.index_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    import thingi10k
    thingi10k.init(variant='raw', cache_dir=args.cache_dir)

    ds = thingi10k.dataset(manifold=True, closed=True, solid=True,
                           self_intersecting=False,
                           num_vertices=(args.min_vertices, args.max_vertices))
    print(f'Strict filter matches: {len(ds)} meshes')

    try:
        ds = ds.sort('file_id')
    except Exception as e:
        print(f'  (sort by file_id skipped: {e})')

    n = min(args.num_keep, len(ds))
    fields = ['file_id', 'thing_id', 'file_path', 'num_vertices', 'num_facets',
              'num_components', 'euler', 'closed', 'vertex_manifold',
              'edge_manifold', 'PWN', 'solid', 'oriented',
              'self_intersecting', 'author', 'license', 'category']
    with out.open('w') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i in range(n):
            e = ds[i]
            w.writerow({k: e.get(k, '') for k in fields})
    print(f'Wrote {n} entries → {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
