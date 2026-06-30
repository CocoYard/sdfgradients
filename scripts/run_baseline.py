"""
SLURM array worker: run RFTA + MES baselines on ONE mesh across all
configured grid_lens, and write reconstructions + a timing CSV.

Idempotent: if ``{out_root}/out/{file_id}/{algo}_{gl}.obj`` already exists
and is non-empty, the (algo, gl) pair is skipped. SDF samples are
generated once per grid_len and cached to ``sdf_{gl}.npz``, so RFTA
and MES share the same input instead of recomputing it.

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
    """Run only the MC-on-samples baseline from a cached SDF .npz.

    Mirrors the marching-cubes block of test_our_method (lines 670-685
    of SDF_to_surface_3D.py). Used as a fast path when ours_<gl>.obj is
    already present but mc_<gl>.obj is missing — saves us re-running the
    full ours pipeline (interpolator fit + surface extract) just to
    rebuild the cheap MC baseline. Raises on degenerate SDFs (e.g.
    ``ValueError: Surface level must be within volume data range``).
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
    p.add_argument('--ours-use-mes', type=int, default=0,
                   choices=[-1, 0, 1],
                   help='use_MES setting for ours: -1=noMES, 0=default MES, '
                        '1=MESforce (paper timing variants)')
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
        # with SDF generation. 'ours' implicitly produces 'mc' as a side
        # product, so include 'mc' in the expected set whenever 'ours' is
        # requested — otherwise an old ours_<gl>.obj could mask a missing
        # mc_<gl>.obj.
        expected = {algo: out_dir / f'{algo}_{gl}.obj' for algo in algos}
        if 'ours' in algos:
            expected['mc'] = out_dir / f'mc_{gl}.obj'
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

            # 'ours' produces ours_<gl>.obj AND mc_<gl>.obj as a pair (mc
            # is a marching-cubes baseline written near the end of
            # test_our_method). Only treat the work as done if BOTH files
            # exist; otherwise re-run test_our_method to backfill the
            # missing one. Note: simply checking ours alone misses the
            # case where a previous run was killed mid-way between the
            # interpolant export (line 578) and the sample_points export
            # (line 685).
            if algo == 'ours':
                mc_path = out_dir / f'mc_{gl}.obj'
                ours_present = (out_file.exists()
                                and out_file.stat().st_size > 0)
                mc_present = (mc_path.exists()
                              and mc_path.stat().st_size > 0)
                if ours_present and mc_present:
                    rows.append(('ours', gl, 'skipped', 0.0,
                                 out_file.stat().st_size, ''))
                    rows.append(('mc',   gl, 'skipped', 0.0,
                                 mc_path.stat().st_size, ''))
                    continue
                if ours_present and not mc_present:
                    # Fast path: only mc is missing. Don't redo the
                    # expensive test_our_method (interpolator fit + surface
                    # extract); just rebuild the marching-cubes baseline
                    # from the cached SDF samples directly.
                    t_mc = time.perf_counter()
                    mc_err = ''
                    try:
                        _compute_mc_only(sdf_cache, mc_path)
                    except Exception:
                        mc_err = traceback.format_exc().strip().splitlines()[-1]
                    dt_mc = time.perf_counter() - t_mc
                    mc_size = (mc_path.stat().st_size
                               if mc_path.exists() else 0)
                    mc_status = 'ok' if mc_size > 0 else 'fail'
                    print(f'[{file_id}]   mc gl={gl:<3} '
                          f'(only) {mc_status:>4} {dt_mc:>6.2f}s '
                          f'{mc_size:>10}B {mc_err}', flush=True)
                    rows.append(('ours', gl, 'skipped', 0.0,
                                 out_file.stat().st_size, ''))
                    rows.append(('mc', gl, mc_status, dt_mc, mc_size,
                                 mc_err if mc_status == 'fail'
                                 else ''))
                    continue
                # else: ours is missing — fall through and run the full
                # test_our_method (which writes both ours and mc).
            elif out_file.exists() and out_file.stat().st_size > 0:
                rows.append((algo, gl, 'skipped', 0.0,
                             out_file.stat().st_size, ''))
                continue

            if algo == 'mc':
                # Standalone marching-cubes timing: run _compute_mc_only on the
                # cached SDF and record real wall-time. Used for paper timing
                # benchmarks — the "free" mc-from-ours side-product reports
                # wall=0 which is correct for "ours total cost" but misleading
                # if the reader wants to know how expensive marching cubes
                # alone is.
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
                # Pinned "main config" for our method. Mirrors the __main__
                # block of SDF_to_surface_3D.py — change here if the canonical
                # config evolves. verbose=True so per-step prints land in the
                # SLURM stdout for later inspection.
                opts = Options(name=mesh_basename, grid_len=gl,
                               max_iters=15, clamp=False, cpp_dc=True,
                               export_short_arcs=False,
                               export_projections=False,
                               turn_off_short_arcs=False,
                               use_gt_gradients=False,
                               interpolator_type='PU',
                               interp_partition='sphere',
                               overlap=0.2, reg=0, lr=0.2,
                               use_MES=args.ours_use_mes,
                               post_processing=False,
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

            # 'ours' produces two outputs we care about, and may raise
            # *after* either is exported. Report status independently:
            # ours = ok if interpolant_*.obj got renamed (test_our_method
            # got past line 578), mc = ok if sample_points_<gl>.obj got
            # renamed (test_our_method got past line 685). Common failure
            # mode: skimage.marching_cubes throws "Surface level must be
            # within volume data range" on degenerate SDFs at small grids
            # — that aborts mc but ours has already been written.
            #   interpolant_<gl>_*.obj  -> ours_<gl>.obj   (our method)
            #   sample_points_<gl>.obj  -> mc_<gl>.obj     (MC-on-samples baseline)
            mc_size = 0
            mc_status = ''
            mc_err_msg = ''
            if algo == 'ours':
                interp = sorted(out_dir.glob(f'interpolant_{gl}_*.obj'),
                                key=lambda p: p.stat().st_mtime, reverse=True)
                if interp:
                    interp[0].rename(out_file)
                sp = out_dir / f'sample_points_{gl}.obj'
                mc_dst = out_dir / f'mc_{gl}.obj'
                # If 'mc' is a standalone algo in this run, mc_<gl>.obj was
                # already written with proper timing — don't overwrite it
                # with the side-product copy (which has wall=0 by definition).
                mc_standalone = 'mc' in algos
                if sp.exists():
                    if mc_standalone and mc_dst.exists() and mc_dst.stat().st_size > 0:
                        sp.unlink()
                    else:
                        sp.rename(mc_dst)
                elif not (mc_standalone and mc_dst.exists()
                          and mc_dst.stat().st_size > 0):
                    # test_our_method no longer exports sample_points_<gl>.obj —
                    # regenerate mc from the cached SDF (same path used when
                    # ours_<gl>.obj already exists but mc_<gl>.obj is missing).
                    try:
                        _compute_mc_only(sdf_cache, mc_dst)
                    except Exception:
                        pass  # falls through to mc_present=False below

                ours_present = (out_file.exists()
                                and out_file.stat().st_size > 0)
                mc_present = (mc_dst.exists()
                              and mc_dst.stat().st_size > 0)

                # If ours export succeeded, any captured exception was
                # raised AFTER ours landed — it is mc's problem, not ours'.
                # Move the err to mc and clear it for the ours row.
                if ours_present:
                    if not mc_present and err:
                        mc_err_msg = err
                    err = None

                mc_size = mc_dst.stat().st_size if mc_present else 0
                mc_status = 'ok' if mc_present else 'fail'
                if mc_status == 'fail' and not mc_err_msg:
                    mc_err_msg = 'mc not produced'

            dt = time.perf_counter() - t0
            size = out_file.stat().st_size if out_file.exists() else 0
            status = 'ok' if size > 0 and err is None else 'fail'
            print(f'[{file_id}] {algo:>4} gl={gl:<3} '
                  f'{status:>7} {dt:>7.2f}s {size:>10}B {err or ""}',
                  flush=True)
            rows.append((algo, gl, status, dt, size, err or ''))

            # 'mc' is a side-product of running 'ours': log a separate
            # row with wall=0 (we paid the cost under 'ours' already).
            # Skip if 'mc' was a standalone algo this run — its real timing
            # row is already in the CSV.
            if algo == 'ours' and 'mc' not in algos:
                rows.append(('mc', gl, mc_status, 0.0, mc_size, mc_err_msg))
                print(f'[{file_id}]   mc gl={gl:<3} '
                      f'{mc_status:>7}    0.00s {mc_size:>10}B {mc_err_msg}',
                      flush=True)

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
