"""
SLURM array worker for the tangent-point × reconstruction experiment.

Combos are named ``tp<source>_<reconstruction>``: which method supplied the
tangent points, and which reconstruction turned them into a surface.

For ONE mesh, across all requested grid_lens, produce the decoupled
combinations that the baselines do not already cover:

    tprfta_rbf : RFTA tangent points  + RBF reconstruction
    tpmes_rbf  : MES  tangent points  + RBF reconstruction
    tpours_psr : OURS tangent points  + sPSR reconstruction
    tpours_rbf : OURS tangent points  + RBF reconstruction (our full method)
    tpgt_psr   : GT   tangent points  + sPSR reconstruction
    tpgt_rbf   : GT   tangent points  + RBF reconstruction

tpours_rbf duplicates what run_baseline.py's 'ours' produces, but it is
generated here rather than copied in: both ours cells then come out of one
script under one config, so they stay comparable even if the baseline run
used a different ablation cell. (tprfta_psr / tpmes_psr -- the native
rfta/mes sPSR reconstructions -- do come from the baseline runner.)

Outputs land at ``{out_root}/out/{file_id}/{combo}_{gl}.obj`` -- the same
tree the baselines use, so compute_metrics.py picks them up unchanged.

Idempotent: an existing non-empty output is skipped. SDF samples are
cached to ``{out_root}/tasks/{file_id}/sdf_cache/sdf_{gl}.npz`` and shared
with the baseline runner.

Usage:
    python scripts/run_experiment_AB.py --task-id $SLURM_ARRAY_TASK_ID
    python scripts/run_experiment_AB.py --file-id 32770 --mesh-path .../32770.stl
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# index_obj.csv, not index.csv: the latter still points at the thingi10k .stl
# extraction dir, which has since been partially cleaned up (e.g. 32770.stl is
# gone). index_obj.csv lists the same file_ids in the same order, pointing at
# the .obj conversions under new_solid/preview_obj/, which are all present.
DEFAULT_INDEX = '/scratch/ycheng27/new_solid/index_obj.csv'
DEFAULT_OUT = '/scratch/ycheng27/new_solid'
DEFAULT_GRIDS = [10, 20, 40, 60, 80]

# (combo_name, tangent method, useRBF)
COMBOS = [
    ('tprfta_rbf', 'RFTA', True),
    ('tpmes_rbf',  'MES',  True),
    ('tpours_psr', 'OURS', False),
    ('tpours_rbf', 'OURS', True),
    ('tpgt_psr',   'GT',   False),
    ('tpgt_rbf',   'GT',   True),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task-id', type=int, default=None)
    p.add_argument('--file-id', default=None)
    p.add_argument('--mesh-path', default=None)
    p.add_argument('--index', default=DEFAULT_INDEX)
    p.add_argument('--out-root', default=DEFAULT_OUT)
    p.add_argument('--out-subdir', default='out.AB',
                   help='mesh outputs land in {out_root}/{out_subdir}/{file_id}/')
    p.add_argument('--grid-lens', default=','.join(str(g) for g in DEFAULT_GRIDS))
    p.add_argument('--combos', default=','.join(c[0] for c in COMBOS),
                   help='comma-separated subset of combo names to run')
    p.add_argument('--screening-weight', type=float, default=10.0)
    p.add_argument('--max-iters', type=int, default=None,
                   help='OURS optimization iters (default: the Options default)')
    return p.parse_args()


def pick_mesh(args: argparse.Namespace):
    if args.mesh_path and args.file_id:
        return args.file_id, Path(args.mesh_path)
    if args.task_id is None:
        raise SystemExit('need either (--file-id + --mesh-path) or --task-id')
    with open(args.index, newline='') as f:
        rows = list(csv.DictReader(f))
    if not (0 <= args.task_id < len(rows)):
        raise SystemExit(f'task_id {args.task_id} out of range [0, {len(rows)})')
    row = rows[args.task_id]
    return row['file_id'], Path(row['file_path'])


def main() -> int:
    args = parse_args()
    file_id, mesh_path = pick_mesh(args)
    if not mesh_path.exists():
        raise SystemExit(f'mesh not found: {mesh_path}')

    grid_lens = [int(g) for g in args.grid_lens.split(',') if g.strip()]
    want = {c.strip() for c in args.combos.split(',') if c.strip()}
    combos = [c for c in COMBOS if c[0] in want]

    out_root = Path(args.out_root).resolve()
    # Per-task CWD so MES's transient ./pwn.csv doesn't collide across tasks.
    task_dir = out_root / 'tasks' / file_id
    (task_dir / 'sdf_cache').mkdir(parents=True, exist_ok=True)
    os.chdir(task_dir)
    sys.path.insert(0, str(REPO))

    import numpy as np
    import trimesh
    import igl
    import SDF_to_surface_3D as sdf3d
    sdf3d.data_dir = str(REPO / 'examples')  # fallback; path_to_obj is overridden
    from SDF_to_surface_3D import (                                    # noqa: E402
        Options, TangentPoints, generate_test_mesh_data,
        get_tangent_points, construct_mesh,
    )
    tp_enum = {m.name: m for m in TangentPoints}

    mesh_basename = mesh_path.stem
    out_dir = out_root / args.out_subdir / mesh_basename
    out_dir.mkdir(parents=True, exist_ok=True)
    sdf_cache_dir = task_dir / 'sdf_cache'
    log_dir = out_root / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    def _load_normalized_mesh(path):
        """Load raw mesh and apply the exact normalization used by
        generate_test_mesh_data (centre + scale by longest axis) so the GT
        closest-point queries live in the same frame as the cached SDF."""
        m = trimesh.load(str(path), force='mesh')
        v = np.asarray(m.vertices, dtype=np.float64)
        mn, mx = v.min(axis=0), v.max(axis=0)
        m.vertices = (v - (mn + mx) / 2) / np.max(mx - mn)
        return m

    rows: list[tuple] = []
    t_task = time.perf_counter()

    for gl in grid_lens:
        outs = {name: out_dir / f'{name}_{gl}.obj' for name, _, _ in combos}
        todo = [c for c in combos
                if not (outs[c[0]].exists() and outs[c[0]].stat().st_size > 0)]
        if not todo:
            for name, _, _ in combos:
                rows.append((name, gl, 'skipped', 0.0, outs[name].stat().st_size, ''))
            print(f'[{file_id}] gl={gl}: all present, skipping', flush=True)
            continue

        # --- SDF samples (shared with baseline runner's cache) ---
        sdf_cache = sdf_cache_dir / f'sdf_{gl}.npz'
        need_gt = any(m == 'GT' for _, m, _ in todo)
        gt_mesh = None
        try:
            if sdf_cache.exists():
                data = np.load(str(sdf_cache))
                points, distances = data['points'], data['sdf_values']
                if need_gt:
                    gt_mesh = _load_normalized_mesh(mesh_path)
            else:
                gt_mesh, points, distances, _ = generate_test_mesh_data(
                    str(mesh_path), mesh_basename, grid_len=gl, save=False)
                np.savez(sdf_cache, points=points, sdf_values=distances)
        except Exception:
            err = traceback.format_exc().strip().splitlines()[-1]
            print(f'[{file_id}] gl={gl}: SDF gen FAILED: {err}', flush=True)
            for name, _, _ in todo:
                rows.append((name, gl, 'sdf_fail', 0.0, 0, err))
            continue

        # Everything except the per-mesh identity stays at the Options default,
        # so this experiment and run_baseline.py cannot drift apart.
        base_opts = dict(name=mesh_basename, grid_len=gl, verbose=False)
        if args.max_iters is not None:
            base_opts['max_iters'] = args.max_iters

        # Compute each distinct tangent set once, reuse across its reconstructions.
        for tp_name in ['GT', 'OURS', 'RFTA', 'MES']:
            sub = [c for c in todo if c[1] == tp_name]
            if not sub:
                continue
            opts = Options(**base_opts)
            opts.path_to_obj = str(mesh_path)
            opts.path_to_sdf = str(sdf_cache)
            opts.gt_mesh = gt_mesh
            t_tp = time.perf_counter()
            try:
                tangent_pts, pts, dst = get_tangent_points(
                    opts, tp_enum[tp_name], screening_weight=args.screening_weight)
            except Exception:
                err = traceback.format_exc().strip().splitlines()[-1]
                dt = time.perf_counter() - t_tp
                for name, _, _ in sub:
                    rows.append((name, gl, 'tangent_fail', dt, 0, err))
                    print(f'[{file_id}] {name:>10} gl={gl:<3} tangent_fail {err}', flush=True)
                continue
            dt_tp = time.perf_counter() - t_tp

            for name, _, useRBF in sub:
                t0 = time.perf_counter()
                err = ''
                try:
                    recon = construct_mesh(tangent_pts, pts, dst, useRBF=useRBF,
                                           options=opts,
                                           screening_weight=args.screening_weight)
                    recon.export(str(outs[name]))
                except Exception:
                    err = traceback.format_exc().strip().splitlines()[-1]
                dt = dt_tp + (time.perf_counter() - t0)
                size = outs[name].stat().st_size if outs[name].exists() else 0
                status = 'ok' if size > 0 and not err else 'fail'
                print(f'[{file_id}] {name:>10} gl={gl:<3} {status:>4} '
                      f'{dt:>7.2f}s {size:>10}B {err}', flush=True)
                rows.append((name, gl, status, dt, size, err))

    timing_csv = log_dir / f'{file_id}.experiment_AB.csv'
    with timing_csv.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['combo', 'grid_len', 'status', 'wall_s', 'out_bytes', 'err'])
        for row in rows:
            w.writerow(row)

    n_ok = sum(1 for r in rows if r[2] in ('ok', 'skipped'))
    print(f'[{file_id}] DONE {n_ok}/{len(rows)} ok  '
          f'total {time.perf_counter() - t_task:.1f}s  -> {timing_csv}', flush=True)
    return 0 if n_ok == len(rows) else 1


if __name__ == '__main__':
    raise SystemExit(main())
