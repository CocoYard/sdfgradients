"""
SLURM array worker: run RFTA + MES baselines on ONE mesh across all
configured grid_lens, and write reconstructions + a timing CSV.

Idempotent: if ``{out_root}/out/{file_id}/{algo}_{gl}.obj`` already exists
and is non-empty, the (algo, gl) pair is skipped. SDF samples are
generated once per grid_len and cached to ``sdf_{gl}.npz``, so all algos
share the same input instead of recomputing it.

Each algo is independent and owns exactly one output file per grid_len:
running ``--algos ours`` produces only ours_<gl>.obj. Pass
``--algos ours,mc`` if you also want the marching-cubes baseline.

Typical usage from sbatch:
    python scripts/run_baseline.py --task-id $SLURM_ARRAY_TASK_ID

Direct usage (for debugging a single mesh):
    python scripts/run_baseline.py --file-id 32770 \
        --mesh-path /scratch/.../raw_meshes/32770.stl
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
DEFAULT_INDEX = '/scratch/ycheng27/thingi10k/index.csv'
DEFAULT_OUT = '/scratch/ycheng27/sdfgradients/baselines'
DEFAULT_GRIDS = [6, 10, 20, 30, 40, 50, 60, 80, 100]
DEFAULT_ALGOS = ['rfta', 'mes', 'ours']


def _compute_mc_only(sdf_cache_path: 'Path', out_path: 'Path') -> None:
    """Run the MC-on-samples baseline from a cached SDF .npz.

    Rebuilds the voxel grid from the sample coordinates and runs marching
    cubes on it — same construction as the block in ``test_mc``. Raises on
    degenerate SDFs (e.g. ``ValueError: Surface level must be within volume
    data range``), which is common at very coarse grids.
    """
    import numpy as np
    import trimesh
    from skimage.measure import marching_cubes

    data = np.load(str(sdf_cache_path))
    points = data['points']
    distances = data['sdf_values']
    xs = np.unique(np.round(points[:, 0], 8))
    ys = np.unique(np.round(points[:, 1], 8))
    zs = np.unique(np.round(points[:, 2], 8))
    nx, ny, nz = len(xs), len(ys), len(zs)
    ix = np.searchsorted(xs, np.round(points[:, 0], 8))
    iy = np.searchsorted(ys, np.round(points[:, 1], 8))
    iz = np.searchsorted(zs, np.round(points[:, 2], 8))
    grid_values = np.ones((nx, ny, nz))
    grid_values[ix, iy, iz] = distances
    sp = ((xs[-1] - xs[0]) / max(nx - 1, 1),
          (ys[-1] - ys[0]) / max(ny - 1, 1),
          (zs[-1] - zs[0]) / max(nz - 1, 1))
    verts, faces, _, _ = marching_cubes(grid_values, level=0.0, spacing=sp)
    verts += np.array([xs[0], ys[0], zs[0]])
    trimesh.Trimesh(vertices=verts, faces=faces).export(str(out_path))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task-id', type=int, default=None,
                   help='SLURM array task id (0-based). Reads row (task_id+1) '
                        'from the index CSV.')
    p.add_argument('--file-id', default=None,
                   help='explicit file_id (overrides --task-id)')
    p.add_argument('--mesh-path', default=None,
                   help='explicit mesh path (overrides index lookup)')
    p.add_argument('--index', default=DEFAULT_INDEX,
                   help='CSV written by prepare_thingi10k.py')
    p.add_argument('--out-root', default=DEFAULT_OUT,
                   help='output root; baselines land in {out_root}/out/{file_id}/')
    p.add_argument('--grid-lens', default=','.join(str(g) for g in DEFAULT_GRIDS))
    p.add_argument('--algos', default=','.join(DEFAULT_ALGOS),
                   help='comma-separated subset of rfta,mes')
    p.add_argument('--rfta-parallel', choices=['on', 'off'], default='on')
    p.add_argument('--rfta-force-cpu', action='store_true',
                   help='disable RFTA GPU path (for CPU-only timing benchmarks)')
    p.add_argument('--screening-weight', type=float, default=10.0)
    # --- ours ablation knobs (defaults = the pinned main config) ---
    p.add_argument('--ours-use-mes', type=int, default=-1,
                   choices=[-1, 0, 1],
                   help='use_MES setting for ours: -1=noMES, 0=default MES, '
                        '1=MESforce (paper timing variants)')
    p.add_argument('--ours-clamp', type=int, default=1, choices=[0, 1])
    p.add_argument('--ours-optimizer', default='bfgs',
                   choices=['bfgs', 'ascent', 'lbfgspp'])
    p.add_argument('--ours-post', type=int, default=0, choices=[0, 1],
                   help='post_processing (Lipschitz post-fix) for ours')
    return p.parse_args()


def pick_mesh(args: argparse.Namespace) -> tuple[str, Path]:
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
    algos = [a.strip() for a in args.algos.split(',') if a.strip()]

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    # MES's CPP binary writes ./pwn.csv to the current working directory,
    # which collides across concurrent array tasks. Give every task its
    # own CWD so those transient files are isolated.
    task_dir = out_root / 'tasks' / file_id
    task_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(task_dir)
    sys.path.insert(0, str(REPO))

    import numpy as np
    import SDF_to_surface_3D as sdf3d
    sdf3d.data_dir = str(REPO / 'examples')  # fallback; path_to_obj is overridden
    from SDF_to_surface_3D import (                                    # noqa: E402
        Options, generate_test_mesh_data, test_rfta, test_mes,
        test_our_method,
    )

    # test_rfta/test_mes write to "out/<basename>/<algo>_<gl>.obj" relative
    # to cwd — and we set cwd to task_dir for per-task isolation. That would
    # bury outputs at task_dir/out/<file_id>/..., doubling the file_id in the
    # path. Symlink task_dir/out/<file_id>  ->  out_root/out/<file_id>  so
    # the file actually lands at the clean canonical location.
    mesh_basename = mesh_path.stem
    canonical_out = out_root / 'out' / mesh_basename
    canonical_out.mkdir(parents=True, exist_ok=True)
    (task_dir / 'out').mkdir(exist_ok=True)
    link = task_dir / 'out' / mesh_basename
    if link.is_symlink():
        link.unlink()
    # Race-tolerant symlink creation: when two concurrent batches process
    # the same task_id (sharing this task_dir), both can race here. Treat
    # FileExistsError as a benign no-op since whatever the other process
    # created is also a symlink to the same canonical path.
    if not link.exists():
        try:
            link.symlink_to(canonical_out, target_is_directory=True)
        except FileExistsError:
            pass
    out_dir = canonical_out  # use the canonical path for existence checks
    sdf_cache_dir = task_dir / 'sdf_cache'
    sdf_cache_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_root / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    rows: list[tuple] = []
    t_task = time.perf_counter()

    for gl in grid_lens:
        # Check whether anything is still missing at this gl before we bother
        # with SDF generation. Every algo owns exactly one output file — 'mc'
        # is its own algo, so ask for it explicitly (--algos ours,mc) if you
        # want the marching-cubes baseline alongside ours.
        expected = {algo: out_dir / f'{algo}_{gl}.obj' for algo in algos}
        need_any = any(not (p.exists() and p.stat().st_size > 0)
                       for p in expected.values())
        if not need_any:
            for algo, p in expected.items():
                rows.append((algo, gl, 'skipped', 0.0, p.stat().st_size, ''))
            print(f'[{file_id}] gl={gl}: all outputs present, skipping',
                  flush=True)
            continue

        # Generate SDF samples once; share between algos.
        sdf_cache = sdf_cache_dir / f'sdf_{gl}.npz'
        if not sdf_cache.exists():
            print(f'[{file_id}] gl={gl}: generating SDF samples ...', flush=True)
            t0 = time.perf_counter()
            try:
                _, points, distances, _ = generate_test_mesh_data(
                    str(mesh_path), mesh_basename, grid_len=gl, save=False)
            except Exception:
                err = traceback.format_exc().strip().splitlines()[-1]
                print(f'[{file_id}] gl={gl}: SDF gen FAILED: {err}',
                      flush=True)
                for algo in algos:
                    rows.append((algo, gl, 'sdf_fail', 0.0, 0, err))
                continue
            np.savez(sdf_cache, points=points, sdf_values=distances)
            print(f'[{file_id}] gl={gl}: SDF ready in '
                  f'{time.perf_counter() - t0:.1f}s', flush=True)

        for algo in algos:
            out_file = expected[algo]

            if out_file.exists() and out_file.stat().st_size > 0:
                rows.append((algo, gl, 'skipped', 0.0,
                             out_file.stat().st_size, ''))
                continue

            if algo == 'mc':
                # Marching cubes straight on the cached SDF samples, timed like
                # any other algo. It shares the SDF cache with the other algos
                # but is otherwise independent of them.
                t0 = time.perf_counter()
                mc_err: str | None = None
                try:
                    _compute_mc_only(sdf_cache, out_file)
                except Exception:
                    mc_err = traceback.format_exc().strip().splitlines()[-1]
                dt = time.perf_counter() - t0
                size = out_file.stat().st_size if out_file.exists() else 0
                status = 'ok' if size > 0 and mc_err is None else 'fail'
                print(f'[{file_id}]   mc gl={gl:<3} '
                      f'{status:>7} {dt:>7.2f}s {size:>10}B {mc_err or ""}',
                      flush=True)
                rows.append(('mc', gl, status, dt, size, mc_err or ''))
                continue

            if algo == 'ours':
                # "Main config" for our method; the four --ours-* flags select
                # the ablation cell (defaults reproduce the pinned config).
                # verbose=True so per-step prints land in the SLURM stdout.
                opts = Options(name=mesh_basename, grid_len=gl,
                               max_iters=15, cpp_dc=True,
                               clamp=bool(args.ours_clamp),
                               export_short_arcs=False,
                               export_projections=False,
                               turn_off_short_arcs=False,
                               use_gt_gradients=False,
                               interpolator_type='PU',
                               interp_partition='sphere',
                               overlap=0.2, reg=0, lr=0.2,
                               use_MES=args.ours_use_mes,
                               optim_steps=5,
                               grad_optimizer=args.ours_optimizer,
                               post_processing=bool(args.ours_post),
                               iter_gradient_finding='optimize',
                               verbose=True)
            else:
                opts = Options(name=mesh_basename, grid_len=gl,
                               max_iters=5, clamp=False,
                               export_short_arcs=False, export_projections=False,
                               verbose=False)
            opts.path_to_obj = str(mesh_path)
            opts.path_to_sdf = str(sdf_cache)

            t0 = time.perf_counter()
            err: str | None = None
            try:
                if algo == 'rfta':
                    test_rfta(opts, save_gtmesh=False,
                              screening_weight=args.screening_weight,
                              parallel=(args.rfta_parallel == 'on'),
                              force_cpu=args.rfta_force_cpu)
                elif algo == 'mes':
                    test_mes(opts, save_gtmesh=False,
                             screening_weight=args.screening_weight)
                elif algo == 'ours':
                    test_our_method(opts, save_gtmesh=False)
                else:
                    raise ValueError(f'unknown algo: {algo}')
            except Exception:
                err = traceback.format_exc().strip().splitlines()[-1]

            if algo == 'ours':
                # test_our_method exports 'ours_<gl>_<iters>_..._<res>.obj'; collapse the
                # newest such file to the canonical 'ours_<gl>.obj'. (out_file is
                # 'ours_<gl>.obj', which this pattern does not match, so no self-rename.)
                interp = sorted(out_dir.glob(f'ours_{gl}_*.obj'),
                                key=lambda p: p.stat().st_mtime, reverse=True)
                if interp:
                    interp[0].rename(out_file)

            dt = time.perf_counter() - t0
            size = out_file.stat().st_size if out_file.exists() else 0
            status = 'ok' if size > 0 and err is None else 'fail'
            print(f'[{file_id}] {algo:>4} gl={gl:<3} '
                  f'{status:>7} {dt:>7.2f}s {size:>10}B {err or ""}',
                  flush=True)
            rows.append((algo, gl, status, dt, size, err or ''))

    # Per-task timing CSV. SLURM concurrency is fine since each task writes
    # its own file.
    timing_csv = log_dir / f'{file_id}.timing.csv'
    with timing_csv.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['algo', 'grid_len', 'status', 'wall_s', 'out_bytes', 'err'])
        for row in rows:
            w.writerow(row)

    n_ok = sum(1 for r in rows if r[2] in ('ok', 'skipped'))
    n_fail = len(rows) - n_ok
    print(f'[{file_id}] DONE  {n_ok}/{len(rows)} ok  '
          f'total {time.perf_counter() - t_task:.1f}s  '
          f'timing -> {timing_csv}', flush=True)
    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
