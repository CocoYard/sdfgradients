"""
Smoke test for the two baseline reconstruction algorithms
(``test_rfta`` and ``test_mes`` from ``SDF_to_surface_3D``) before
committing to the 500-mesh Thingi10K batch.

Goals:
  1. Confirm each algorithm runs end-to-end and writes a non-empty .obj
     on a handful of small ``examples/*.obj`` meshes.
  2. Probe the worst-case wall-time at ``grid_len=100`` on one mesh so
     we can size the SLURM ``--time`` / ``--mem`` request.
  3. Surface silent failures (e.g. MES returning False when its CPP
     binary is missing, which normally raises inside ``gpy.write_mesh``
     with an opaque traceback).

Usage:
    cd /home/ycheng27/code/sdfgradients
    python scripts/smoke_test_baselines.py
    # or pick different meshes:
    python scripts/smoke_test_baselines.py --meshes bunny,eiffel --grids 20,50
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)                  # test_rfta/test_mes write to 'out/' relative to cwd
sys.path.insert(0, str(REPO))

import SDF_to_surface_3D as sdf3d                             # noqa: E402
# ``data_dir`` is only assigned inside SDF_to_surface_3D's ``__main__`` block,
# but ``Options.__init__`` reads it at construction time. Inject it here so we
# can build Options objects from an importer.
sdf3d.data_dir = 'examples'
from SDF_to_surface_3D import Options, test_rfta, test_mes    # noqa: E402

_RFTA_PARALLEL = True  # flipped by --rfta-parallel CLI flag


@dataclass
class RunResult:
    algo: str
    mesh: str
    grid_len: int
    ok: bool
    wall_s: float
    out_bytes: int
    err: str | None


def _make_options(mesh_name: str, grid_len: int) -> Options:
    opts = Options(name=mesh_name, grid_len=grid_len,
                   max_iters=5, clamp=False,
                   export_short_arcs=False, export_projections=False,
                   verbose=False)
    opts.path_to_obj = str(REPO / 'examples' / f'{mesh_name}.obj')
    opts.path_to_sdf = None
    return opts


def _expected_output(algo: str, mesh_name: str, grid_len: int) -> Path:
    # test_rfta / test_mes write to 'out/<basename>/<algo>_<grid_len>.obj'
    return REPO / 'out' / mesh_name / f'{algo}_{grid_len}.obj'


def run_one(algo: str, mesh_name: str, grid_len: int) -> RunResult:
    opts = _make_options(mesh_name, grid_len)
    out_path = _expected_output(algo, mesh_name, grid_len)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    t0 = time.perf_counter()
    err: str | None = None
    try:
        if algo == 'rfta':
            test_rfta(opts, save_gtmesh=False, screening_weight=10,
                      parallel=_RFTA_PARALLEL)
        else:
            test_mes(opts, save_gtmesh=False, screening_weight=10)
    except Exception:
        err = traceback.format_exc()
    dt = time.perf_counter() - t0

    size = out_path.stat().st_size if out_path.exists() else 0
    ok = err is None and size > 0
    err_line = err.strip().splitlines()[-1] if err else None
    return RunResult(algo, mesh_name, grid_len, ok, dt, size, err_line)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--meshes', default='bunny,chair',
                   help='comma-separated names (no .obj) under examples/')
    p.add_argument('--grids', default='20,50',
                   help='comma-separated grid_lens for the matrix')
    p.add_argument('--probe-grid', type=int, default=100,
                   help='grid_len for the single worst-case probe; 0 to skip')
    p.add_argument('--probe-mesh', default=None,
                   help='mesh for the probe; defaults to first --meshes entry')
    p.add_argument('--skip-mes', action='store_true',
                   help='skip test_mes (use if the CPP binary is not built yet)')
    p.add_argument('--skip-rfta', action='store_true')
    p.add_argument('--rfta-parallel', choices=['on', 'off'], default='on',
                   help='pass parallel=True/False to test_rfta; set off to '
                        'diagnose pthread oversubscription on cgroup-limited nodes')
    return p.parse_args()


def main() -> int:
    args = parse_args()
    global _RFTA_PARALLEL
    _RFTA_PARALLEL = (args.rfta_parallel == 'on')
    mesh_names = [m.strip() for m in args.meshes.split(',') if m.strip()]
    grid_lens = [int(g) for g in args.grids.split(',') if g.strip()]
    algos = [a for a, skip in [('rfta', args.skip_rfta), ('mes', args.skip_mes)] if not skip]

    # Check meshes exist
    for m in mesh_names:
        p = REPO / 'examples' / f'{m}.obj'
        if not p.exists():
            print(f'ERROR: {p} not found; pick from {sorted(x.stem for x in (REPO / "examples").glob("*.obj"))}')
            return 2

    rows: list[RunResult] = []
    print(f'{"algo":>4}  {"mesh":>10}  {"gl":>3}  {"status":>6}  {"wall":>8}  {"bytes":>10}  err')
    print('-' * 78)
    for mesh in mesh_names:
        for gl in grid_lens:
            for algo in algos:
                r = run_one(algo, mesh, gl)
                rows.append(r)
                print(f'{r.algo:>4}  {r.mesh:>10}  {r.grid_len:>3}  '
                      f'{"OK" if r.ok else "FAIL":>6}  '
                      f'{r.wall_s:>7.2f}s  {r.out_bytes:>10}  {r.err or ""}')

    # Probe run (worst-case sizing for SLURM --time)
    if args.probe_grid and args.probe_grid > 0:
        probe_mesh = args.probe_mesh or mesh_names[0]
        print(f'\n-- probe: grid_len={args.probe_grid} on {probe_mesh} --')
        for algo in algos:
            r = run_one(algo, probe_mesh, args.probe_grid)
            rows.append(r)
            print(f'{r.algo:>4}  {r.mesh:>10}  {r.grid_len:>3}  '
                  f'{"OK" if r.ok else "FAIL":>6}  '
                  f'{r.wall_s:>7.2f}s  {r.out_bytes:>10}  {r.err or ""}')

    # Summary
    fails = [r for r in rows if not r.ok]
    max_wall = max((r.wall_s for r in rows if r.ok), default=0.0)
    print('\n' + '=' * 78)
    print(f'Total runs: {len(rows)}   OK: {len(rows) - len(fails)}   FAIL: {len(fails)}')
    print(f'Max successful wall time: {max_wall:.2f} s')
    if fails:
        print('Failing runs:')
        for r in fails:
            print(f'  {r.algo} {r.mesh} gl={r.grid_len}: {r.err}')
    return 0 if not fails else 1


if __name__ == '__main__':
    raise SystemExit(main())
