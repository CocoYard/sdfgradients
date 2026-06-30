"""
Compute Hausdorff / Chamfer / F1 between baseline reconstructions and
the GT meshes for every (file_id, algo, grid_len) cell, and append the
results to a single CSV.

Both meshes are placed in the same normalised frame as
``generate_test_mesh_data`` uses (centred at origin, longest axis scaled
to 1) before calling ``util.mesh_distances`` from this repo.

Usage:
    # all four algos (after every batch finishes)
    python scripts/compute_metrics.py

    # just the algos that are already done (rfta + mes)
    python scripts/compute_metrics.py --algos rfta,mes

    # smoke-test on the first 5 meshes
    python scripts/compute_metrics.py --algos rfta --limit 5

The script is idempotent: each run writes a fresh CSV at ``--output``,
overwriting whatever was there. Subsequent runs that include more algos
or more meshes simply re-overwrite — recomputation is fast (~1.5 h wall
on 16 cores for the full 18 000-cell sweep).
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

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from util import mesh_distances                                       # noqa: E402

DEFAULT_INDEX = '/scratch/ycheng27/thingi10k/index.csv'
DEFAULT_GT_ROOT = '/scratch/ycheng27/thingi10k/preview_obj'
DEFAULT_OUT_ROOT = '/scratch/ycheng27/sdfgradients/baselines/out'
DEFAULT_GRIDS = [6, 10, 20, 30, 40, 50, 60, 80, 100]
DEFAULT_ALGOS = ['rfta', 'mes', 'ours', 'mc']
DEFAULT_OUTPUT = '/scratch/ycheng27/sdfgradients/baselines/metrics.csv'


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', default=DEFAULT_INDEX)
    p.add_argument('--gt-root', default=DEFAULT_GT_ROOT)
    p.add_argument('--out-root', default=DEFAULT_OUT_ROOT)
    p.add_argument('--algos', default=','.join(DEFAULT_ALGOS))
    p.add_argument('--grids', default=','.join(str(g) for g in DEFAULT_GRIDS))
    p.add_argument('--output', default=DEFAULT_OUTPUT)
    p.add_argument('--workers', type=int, default=mp.cpu_count(),
                   help='parallel processes (default: %(default)s)')
    p.add_argument('--limit', type=int, default=None,
                   help='only process first N file_ids from the index')
    p.add_argument('--n-samples', type=int, default=100_000,
                   help='surface samples per mesh for Chamfer/F1')
    p.add_argument('--f1-tau', type=float, default=0.01,
                   help='F1 distance threshold (in normalised-cube units)')
    return p.parse_args()


def _normalize_in_place(mesh: trimesh.Trimesh) -> None:
    """Centre at origin, scale the longest axis to length 1.
    Matches the normalisation used in generate_test_mesh_data."""
    v = np.asarray(mesh.vertices, dtype=np.float64)
    mn = v.min(axis=0)
    mx = v.max(axis=0)
    v = v - (mn + mx) / 2
    v = v / np.max(mx - mn)
    mesh.vertices = v


def _compute_one(packed):
    file_id, algo, gl, gt_path, pred_path, n_samples, f1_tau = packed
    # Failure encoding: any case where we can't actually compare meshes
    # (file missing, empty mesh, or compute error) gets ``hausdorff=inf,
    # chamfer=inf, f1=0`` so the median aggregation in summarize_metrics
    # naturally penalises methods that fail to produce a valid output.
    fail = ('inf', 'inf', '0')
    if not pred_path.exists() or pred_path.stat().st_size == 0:
        return (file_id, algo, gl, *fail, 0, 0, 'missing')
    try:
        gt = trimesh.load(str(gt_path), force='mesh')
        _normalize_in_place(gt)
        pred = trimesh.load(str(pred_path), force='mesh')
        if len(pred.vertices) == 0 or len(pred.faces) == 0:
            return (file_id, algo, gl, *fail,
                    len(pred.vertices), len(pred.faces), 'empty')
        h, c, f1 = mesh_distances(pred, gt, verbose=False,
                                  n_samples=n_samples, f1_tau=f1_tau)
        return (file_id, algo, gl, f'{h:.6g}', f'{c:.6g}', f'{f1:.6g}',
                len(pred.vertices), len(pred.faces), '')
    except Exception as e:
        msg = f'{type(e).__name__}: {e}'.replace('\n', ' ')[:200]
        return (file_id, algo, gl, *fail, 0, 0, msg)


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

    jobs = []
    for fid in file_ids:
        gt_path = gt_root / f'{fid}.obj'
        if not gt_path.exists():
            continue
        for algo in algos:
            for gl in grids:
                pred_path = out_root / fid / f'{algo}_{gl}.obj'
                jobs.append((fid, algo, gl, gt_path, pred_path,
                             args.n_samples, args.f1_tau))

    print(f'Computing metrics for {len(jobs)} cells '
          f'({len(file_ids)} meshes × {len(algos)} algos × {len(grids)} grids) '
          f'with {args.workers} workers', flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    n_ok = n_missing = n_err = 0
    with out_path.open('w', newline='') as f, \
            mp.Pool(args.workers) as pool:
        w = csv.writer(f)
        w.writerow(['file_id', 'algo', 'grid_len', 'hausdorff', 'chamfer',
                    'f1', 'n_verts', 'n_faces', 'err'])
        for i, row in enumerate(
                pool.imap_unordered(_compute_one, jobs, chunksize=4), 1):
            w.writerow(row)
            err = row[8]
            if err == '':
                n_ok += 1
            elif err == 'missing':
                n_missing += 1
            else:
                n_err += 1
                if n_err <= 5:
                    print(f'  ERR  {row[0]} {row[1]} gl={row[2]}: {err}',
                          flush=True)
            if i % 500 == 0:
                rate = i / (time.perf_counter() - t0)
                eta = (len(jobs) - i) / max(rate, 1e-6)
                print(f'  {i}/{len(jobs)} done  ok={n_ok} '
                      f'miss={n_missing} err={n_err}  '
                      f'eta={eta/60:.1f} min', flush=True)

    dt = time.perf_counter() - t0
    print(f'\nDone in {dt/60:.1f} min '
          f'({len(jobs)} cells, {len(jobs)/dt:.1f} cells/s)')
    print(f'  ok      : {n_ok}')
    print(f'  missing : {n_missing}')
    print(f'  errors  : {n_err}')
    print(f'CSV → {out_path}')
    return 0 if n_err == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
