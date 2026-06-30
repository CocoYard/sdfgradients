"""
Post-process every ``rfta_<gl>.obj`` in the baseline output tree to drop
connected components that lie entirely outside the input mesh's bounding
box. Avoids re-running the full RFTA pipeline (~1100 CPU-h) when we only
want to tighten the artifact filter.

For each mesh we compare the rfta output against the GT mesh's bbox in
the same normalised coordinate frame as generate_test_mesh_data uses
(centred at origin, longest axis scaled to 1). A component is dropped iff
its own bounding box does NOT intersect the GT bbox — i.e. it is *fully
outside*. Components straddling the bbox edge are kept.

Usage:
    python scripts/cleanup_rfta_outliers.py
        # walks /scratch/.../baselines/out/*/ rfta_*.obj, overwrites in place

    python scripts/cleanup_rfta_outliers.py --dry-run
        # report what would be removed without modifying files

    python scripts/cleanup_rfta_outliers.py --workers 16
        # parallelise across N processes (default: cpu_count)
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import trimesh


DEFAULT_OUT_ROOT = Path('/scratch/ycheng27/sdfgradients/baselines/out')
DEFAULT_GT_ROOT = Path('/scratch/ycheng27/thingi10k/preview_obj')
DEFAULT_INDEX = Path('/scratch/ycheng27/thingi10k/index.csv')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-root', type=Path, default=DEFAULT_OUT_ROOT,
                   help='root holding {file_id}/rfta_<gl>.obj files')
    p.add_argument('--gt-root', type=Path, default=DEFAULT_GT_ROOT,
                   help='root holding {file_id}.obj GT meshes')
    p.add_argument('--index', type=Path, default=DEFAULT_INDEX,
                   help='index.csv to enumerate file_ids; otherwise walks out-root')
    p.add_argument('--workers', type=int, default=mp.cpu_count(),
                   help='parallel processes (default: %(default)s)')
    p.add_argument('--dry-run', action='store_true',
                   help='report without writing any file')
    return p.parse_args()


def _gt_bbox(gt_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load GT mesh, normalise to unit cube as generate_test_mesh_data does,
    then return the SDF sampling bbox (mesh bounds inflated by 0.1 on every
    axis — same padding that generate_test_mesh_data uses for the SDF grid).
    A reconstruction sticking exactly to that grid extent is legitimate and
    must not be filtered out as an "outlier"."""
    m = trimesh.load(str(gt_path), force='mesh')
    v = np.asarray(m.vertices, dtype=np.float64)
    mn = v.min(axis=0)
    mx = v.max(axis=0)
    v = v - (mn + mx) / 2
    v = v / np.max(mx - mn)
    return v.min(axis=0) - 0.1, v.max(axis=0) + 0.1


def _process_one(rfta_path: Path, gt_path: Path,
                 dry_run: bool) -> tuple[str, int, int, int, str]:
    """Return (file_id_gl, n_kept, n_dropped, n_components_total, err)."""
    tag = f'{rfta_path.parent.name}/{rfta_path.name}'
    try:
        bbox_min, bbox_max = _gt_bbox(gt_path)
        rfta = trimesh.load(str(rfta_path), force='mesh')
        components = rfta.split(only_watertight=False)
        if len(components) <= 1:
            return tag, 1, 0, len(components), ''  # nothing to filter

        kept = []
        for c in components:
            c_min = c.vertices.min(axis=0)
            c_max = c.vertices.max(axis=0)
            # AABB intersection test: outside iff c is strictly to one
            # side along any axis. We *keep* when the boxes overlap.
            if (np.all(c_max >= bbox_min)
                    and np.all(c_min <= bbox_max)):
                kept.append(c)

        if not kept:
            kept = [max(components, key=lambda m: len(m.faces))]

        if not dry_run and len(kept) < len(components):
            cleaned = trimesh.util.concatenate(kept)
            cleaned.export(str(rfta_path))
        return tag, len(kept), len(components) - len(kept), len(components), ''
    except Exception as e:
        return tag, 0, 0, 0, f'{type(e).__name__}: {e}'


def _enumerate_file_ids(out_root: Path, index: Path) -> list[str]:
    if index.exists():
        with index.open(newline='') as f:
            return [r['file_id'] for r in csv.DictReader(f)]
    # fallback: walk out_root
    return sorted(p.name for p in out_root.iterdir() if p.is_dir())


def main() -> int:
    args = parse_args()

    file_ids = _enumerate_file_ids(args.out_root, args.index)
    jobs: list[tuple[Path, Path]] = []
    for fid in file_ids:
        gt = args.gt_root / f'{fid}.obj'
        if not gt.exists():
            continue
        for rfta in (args.out_root / fid).glob('rfta_*.obj'):
            jobs.append((rfta, gt))

    print(f'Found {len(jobs)} rfta files across {len(file_ids)} meshes',
          flush=True)
    if args.dry_run:
        print('(dry run — files will not be modified)', flush=True)

    t0 = time.perf_counter()
    n_with_drops = 0
    n_total_dropped = 0
    n_with_err = 0

    with mp.Pool(args.workers) as pool:
        for i, (tag, kept, dropped, total, err) in enumerate(
                pool.starmap(_process_one,
                             [(p, g, args.dry_run) for p, g in jobs]), 1):
            if err:
                n_with_err += 1
                if n_with_err <= 5:
                    print(f'  ERROR {tag}: {err}', flush=True)
                continue
            if dropped > 0:
                n_with_drops += 1
                n_total_dropped += dropped
                if n_with_drops <= 20:
                    print(f'  {tag}: kept {kept}/{total} (dropped {dropped})',
                          flush=True)
            if i % 500 == 0:
                print(f'  ... {i}/{len(jobs)} done, '
                      f'elapsed {time.perf_counter() - t0:.1f}s',
                      flush=True)

    dt = time.perf_counter() - t0
    print(f'\nDone in {dt:.1f}s.')
    print(f'  files modified  : {n_with_drops}')
    print(f'  components dropped (total): {n_total_dropped}')
    print(f'  errors          : {n_with_err}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
