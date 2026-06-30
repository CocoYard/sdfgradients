"""
Compare each reconstruction's topology against the input GT mesh and
write per-cell deltas to CSV. Tracks
  • connected components
  • open boundary edges (== 0 ⇔ watertight)
  • Euler characteristic V − E + F
  • genus (for watertight closed-orientable inputs)

Useful to flag failure modes that surface-distance metrics miss, e.g.
"reconstruction looks similar but has 50 spurious components" or
"the method closed up a torus into a sphere (lost a hole)".

Usage:
    python scripts/topology_check.py
        # all algos × all grids → /scratch/.../baselines/topology.csv

    python scripts/topology_check.py --algos ours,rfta --limit 20
"""

from __future__ import annotations

import argparse
import csv
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import trimesh


DEFAULT_INDEX = '/scratch/ycheng27/thingi10k/index.csv'
DEFAULT_GT_ROOT = '/scratch/ycheng27/thingi10k/preview_obj'
DEFAULT_OUT_ROOT = '/scratch/ycheng27/sdfgradients/baselines/out'
DEFAULT_GRIDS = [6, 10, 20, 30, 40, 50, 60, 80, 100]
DEFAULT_ALGOS = ['rfta', 'mes', 'ours', 'mc']
DEFAULT_OUTPUT = '/scratch/ycheng27/sdfgradients/baselines/topology.csv'


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', default=DEFAULT_INDEX)
    p.add_argument('--gt-root', default=DEFAULT_GT_ROOT)
    p.add_argument('--out-root', default=DEFAULT_OUT_ROOT)
    p.add_argument('--algos', default=','.join(DEFAULT_ALGOS))
    p.add_argument('--grids', default=','.join(str(g) for g in DEFAULT_GRIDS))
    p.add_argument('--output', default=DEFAULT_OUTPUT)
    p.add_argument('--workers', type=int, default=mp.cpu_count())
    p.add_argument('--limit', type=int, default=None)
    return p.parse_args()


def topology_of(mesh: trimesh.Trimesh) -> dict:
    """Compute component / boundary / Euler / genus stats for a mesh.

    ``genus`` is well-defined only for closed orientable surfaces; we set
    it to NaN otherwise (the caller can still compare ``euler`` and
    ``n_components`` to the GT)."""
    n_components = int(mesh.body_count)
    euler = int(mesh.euler_number)
    # Boundary edges = edges referenced by exactly one face. trimesh
    # exposes them via ``edges_unique`` + ``faces_unique_edges``; the
    # cheap way is to subtract the shared-edge count from total.
    edges = mesh.edges_sorted
    if len(edges):
        unique_e, counts = np.unique(edges, axis=0, return_counts=True)
        n_boundary = int((counts == 1).sum())
    else:
        n_boundary = 0
    is_watertight = bool(mesh.is_watertight)
    if is_watertight and n_boundary == 0:
        # χ = 2N − 2g  ⇒  g = N − χ/2  (integer for closed orientable)
        genus = float(n_components - euler / 2)
    else:
        genus = float('nan')
    return {
        'V': len(mesh.vertices),
        'F': len(mesh.faces),
        'n_components': n_components,
        'n_boundary': n_boundary,
        'is_watertight': is_watertight,
        'euler': euler,
        'genus': genus,
    }


def _normalize(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Same coord frame as everywhere else (centred, longest axis = 1)."""
    v = np.asarray(mesh.vertices, dtype=np.float64)
    mn = v.min(axis=0)
    mx = v.max(axis=0)
    v = v - (mn + mx) / 2
    v = v / np.max(mx - mn)
    mesh.vertices = v
    return mesh


def _process_one(packed) -> tuple:
    fid, algo, gl, pred_path, gt_top = packed
    if not pred_path.exists() or pred_path.stat().st_size == 0:
        return (fid, algo, gl,
                0, 0, 0, 0, 0,
                gt_top['V'], gt_top['F'], gt_top['n_components'],
                gt_top['n_boundary'], gt_top['euler'],
                _fmt(gt_top['genus']),
                '', '', '', '',
                'missing')
    try:
        m = trimesh.load(str(pred_path), force='mesh')
        if len(m.vertices) == 0 or len(m.faces) == 0:
            return (fid, algo, gl,
                    len(m.vertices), len(m.faces), 0, 0, 0,
                    gt_top['V'], gt_top['F'], gt_top['n_components'],
                    gt_top['n_boundary'], gt_top['euler'],
                    _fmt(gt_top['genus']),
                    '', '', '', '',
                    'empty')
        t = topology_of(m)
        d_comp = t['n_components'] - gt_top['n_components']
        d_bnd = t['n_boundary'] - gt_top['n_boundary']
        d_euler = t['euler'] - gt_top['euler']
        d_genus = (t['genus'] - gt_top['genus']
                   if not (math.isnan(t['genus']) or math.isnan(gt_top['genus']))
                   else float('nan'))
        return (fid, algo, gl,
                t['V'], t['F'], t['n_components'], t['n_boundary'], t['euler'],
                gt_top['V'], gt_top['F'], gt_top['n_components'],
                gt_top['n_boundary'], gt_top['euler'],
                _fmt(gt_top['genus']),
                d_comp, d_bnd, d_euler, _fmt(d_genus),
                '')
    except Exception as e:
        msg = f'{type(e).__name__}: {e}'.replace('\n', ' ')[:200]
        return (fid, algo, gl,
                0, 0, 0, 0, 0,
                gt_top['V'], gt_top['F'], gt_top['n_components'],
                gt_top['n_boundary'], gt_top['euler'],
                _fmt(gt_top['genus']),
                '', '', '', '',
                msg)


def _fmt(x):
    if isinstance(x, float) and math.isnan(x):
        return ''
    if isinstance(x, float):
        # Genus values should be effectively integer; render as int when so.
        if abs(x - round(x)) < 1e-6:
            return int(round(x))
        return f'{x:.4g}'
    return x


def main() -> int:
    args = parse_args()
    algos = [a.strip() for a in args.algos.split(',') if a.strip()]
    grids = [int(g) for g in args.grids.split(',') if g.strip()]

    with open(args.index, newline='') as f:
        file_ids = [row['file_id'] for row in csv.DictReader(f)]
    if args.limit:
        file_ids = file_ids[: args.limit]

    gt_root = Path(args.gt_root)
    out_root = Path(args.out_root)

    # Pre-compute GT topology once per mesh — avoids re-loading 9×4 = 36
    # times per file_id in the worker pool.
    print(f'Pre-computing GT topology for {len(file_ids)} meshes ...',
          flush=True)
    t0 = time.perf_counter()
    gt_cache: dict[str, dict] = {}
    for i, fid in enumerate(file_ids, 1):
        gt_path = gt_root / f'{fid}.obj'
        if not gt_path.exists():
            continue
        try:
            m = trimesh.load(str(gt_path), force='mesh')
            _normalize(m)
            gt_cache[fid] = topology_of(m)
        except Exception as e:
            print(f'  ERROR  GT {fid}: {e}', flush=True)
        if i % 100 == 0:
            print(f'  GT {i}/{len(file_ids)}  '
                  f'({time.perf_counter() - t0:.1f}s)',
                  flush=True)
    print(f'GT topology cached for {len(gt_cache)} meshes '
          f'in {time.perf_counter() - t0:.1f}s', flush=True)

    jobs = []
    for fid, gt_top in gt_cache.items():
        for algo in algos:
            for gl in grids:
                pred_path = out_root / fid / f'{algo}_{gl}.obj'
                jobs.append((fid, algo, gl, pred_path, gt_top))

    print(f'\nTopology check on {len(jobs)} cells with {args.workers} workers',
          flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fields = ['file_id', 'algo', 'grid_len',
              'recon_V', 'recon_F', 'recon_components',
              'recon_n_boundary', 'recon_euler',
              'gt_V', 'gt_F', 'gt_components',
              'gt_n_boundary', 'gt_euler', 'gt_genus',
              'delta_components', 'delta_n_boundary',
              'delta_euler', 'delta_genus',
              'err']

    t0 = time.perf_counter()
    n_ok = n_err = 0
    with out_path.open('w', newline='') as f, \
            mp.Pool(args.workers) as pool:
        w = csv.writer(f)
        w.writerow(fields)
        for i, row in enumerate(
                pool.imap_unordered(_process_one, jobs, chunksize=4), 1):
            w.writerow(row)
            if row[-1] == '':
                n_ok += 1
            else:
                n_err += 1
            if i % 1000 == 0:
                rate = i / (time.perf_counter() - t0)
                eta = (len(jobs) - i) / max(rate, 1e-6)
                print(f'  {i}/{len(jobs)}  ok={n_ok} err/missing={n_err}  '
                      f'eta={eta/60:.1f} min', flush=True)

    print(f'\nDone in {(time.perf_counter() - t0)/60:.1f} min '
          f'({len(jobs)} cells)')
    print(f'  ok            : {n_ok}')
    print(f'  missing/error : {n_err}')
    print(f'CSV → {out_path}')
    return 0 if n_err == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
