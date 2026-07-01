"""
Drop connected components that lie COMPLETELY outside the input SDF bounding
box -- sPSR (screened Poisson) reconstructions sometimes emit spurious closed
"bubble" components floating outside the sampled region.

For each processed mesh:
  - input bbox = bounding box of the SDF sample points (from the shared
    sdf_cache/sdf_{gl}.npz), i.e. the exact grid extent the mesh was built from.
  - a component is dropped iff NONE of its vertices fall inside that bbox
    (i.e. it is entirely outside). Straddling components are kept.
  - if anything is dropped, the ORIGINAL mesh is copied to the backup tree
    first, then overwritten with the pruned mesh.

Only sPSR outputs are processed by default (A2B1_ours, A3B1_gt); the RBF /
dual-contouring combos are extracted inside the bbox and cannot have external
components.

Usage:
    python scripts/prune_external_components.py --limit 50 --workers 16
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

DEFAULT_INDEX = '/scratch/ycheng27/new_solid/index.csv'
DEFAULT_ROOT = '/scratch/ycheng27/new_solid'
DEFAULT_COMBOS = ['A2B1_ours', 'A3B1_gt']
DEFAULT_GRIDS = [10, 20, 40, 60, 80]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', default=DEFAULT_INDEX)
    p.add_argument('--root', default=DEFAULT_ROOT,
                   help='holds out.AB/ (meshes), tasks/ (sdf caches); backup goes here too')
    p.add_argument('--out-subdir', default='out.AB')
    p.add_argument('--backup-subdir', default='out.AB.prebbox_bak')
    p.add_argument('--combos', default=','.join(DEFAULT_COMBOS))
    p.add_argument('--grids', default=','.join(str(g) for g in DEFAULT_GRIDS))
    p.add_argument('--limit', type=int, default=50)
    p.add_argument('--workers', type=int, default=16)
    return p.parse_args()


def _bbox_from_cache(root: Path, fid: str, gl: int):
    cache = root / 'tasks' / fid / 'sdf_cache' / f'sdf_{gl}.npz'
    if not cache.exists():
        return None
    pts = np.load(str(cache))['points']
    return pts.min(axis=0), pts.max(axis=0)


def _process_one(packed):
    root_s, fid, combo, gl, out_sub, bak_sub = packed
    root = Path(root_s)
    mesh_path = root / out_sub / fid / f'{combo}_{gl}.obj'
    if not mesh_path.exists() or mesh_path.stat().st_size == 0:
        return (fid, combo, gl, 'missing', 0, 0)
    bbox = _bbox_from_cache(root, fid, gl)
    if bbox is None:
        return (fid, combo, gl, 'no_bbox', 0, 0)
    bmin, bmax = bbox
    try:
        mesh = trimesh.load(str(mesh_path), force='mesh')
        comps = mesh.split(only_watertight=False)
        if len(comps) <= 1:
            return (fid, combo, gl, 'single', len(comps), len(comps))
        kept = []
        for c in comps:
            v = np.asarray(c.vertices)
            inside = np.all((v >= bmin) & (v <= bmax), axis=1)
            if inside.any():                 # keep if ANY vertex is inside the bbox
                kept.append(c)
        n_all, n_kept = len(comps), len(kept)
        if n_kept == n_all:
            return (fid, combo, gl, 'unchanged', n_all, n_kept)
        if n_kept == 0:                      # safety: never emit an empty mesh
            kept = [max(comps, key=lambda m: len(m.faces))]
            n_kept = 1
        # back up original, then overwrite with the pruned mesh
        bak_path = root / bak_sub / fid / f'{combo}_{gl}.obj'
        bak_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(mesh_path), str(bak_path))
        trimesh.util.concatenate(kept).export(str(mesh_path))
        return (fid, combo, gl, 'pruned', n_all, n_kept)
    except Exception as e:
        return (fid, combo, gl, f'err:{type(e).__name__}', 0, 0)


def main():
    args = parse_args()
    root = Path(args.root)
    combos = [c.strip() for c in args.combos.split(',') if c.strip()]
    grids = [int(g) for g in args.grids.split(',') if g.strip()]
    with open(args.index, newline='') as f:
        fids = [r['file_id'] for r in csv.DictReader(f)][: args.limit]

    jobs = [(str(root), fid, combo, gl, args.out_subdir, args.backup_subdir)
            for fid in fids for combo in combos for gl in grids]
    print(f'Scanning {len(jobs)} meshes ({len(fids)} models x {len(combos)} combos '
          f'x {len(grids)} grids) with {args.workers} workers', flush=True)

    t0 = time.perf_counter()
    from collections import Counter
    tally = Counter()
    pruned = []
    with mp.Pool(args.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(_process_one, jobs, chunksize=4), 1):
            fid, combo, gl, status, n_all, n_kept = row
            key = status if not status.startswith('err') else 'err'
            tally[key] += 1
            if status == 'pruned':
                pruned.append((fid, combo, gl, n_all, n_kept))
            if i % 200 == 0:
                print(f'  {i}/{len(jobs)} ...', flush=True)

    print(f'\nDone in {(time.perf_counter()-t0)/60:.1f} min')
    for k, v in sorted(tally.items()):
        print(f'  {k:>10}: {v}')
    if pruned:
        print(f'\nPruned {len(pruned)} meshes (backup -> {root/args.backup_subdir}); sample:')
        for fid, combo, gl, na, nk in pruned[:20]:
            print(f'  {fid} {combo} gl={gl}: {na} -> {nk} components')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
