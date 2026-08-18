"""Experiment 4 — convergence of the projection iteration.

Sweeps max_iters 0..15 per mesh, adds one max_iters=0 run without short arcs,
and runs RFTA, MES and marching cubes once each as a reference line. Writes
Hausdorff, Chamfer and F1 per run to results/convergence/metrics.csv.

    python additional_experiments/convergence.py
"""

import os
import re
import time
from pathlib import Path

import _common

# mesh -> grid_len. One resolution per mesh, so each curve is read against its
# own iteration count rather than against another mesh.
MESHES = {'denker': 20}

# 0 is not "nothing ran": the last fit before the iteration loop is value-only,
# so max_iters=0 extracts the plain RBF interpolant of the samples. Every
# further iteration re-estimates the gradients, reprojects and refits.
MAX_ITERS = list(range(16))

# Every run of this experiment lands in this one directory. `ours` already puts
# grid_len, max_iters and the short-arc flag in the name it exports
# (ours_<grid_len>_<max_iters>_<shortArcs|noShortArcs>_...), and the baselines
# put grid_len in theirs, so no two runs of a mesh can overwrite each other and
# there is nothing for a per-run directory to protect. The degraded-input
# experiments do need that split, which is why _common.run_cell still makes one
# -- test_mes writes mes_50.obj at every noise level, so its runs would collide.
RESULTS = _common.RESULTS / 'convergence'

# Read the swept fields back out of an exported filename, which is how the
# metrics below are keyed. The baselines have only their resolution to report.
NAME = re.compile(
    r'ours_(?P<grid_len>\d+)_(?P<iters>\d+)_(?P<arcs>noShortArcs|shortArcs)_')
BASELINE_NAME = re.compile(r'(?:rfta|mes|sample_points)_(?P<grid_len>\d+)')

# ours first, then the reference line, in the order the table reads.
ALGOS = ['ours', 'rfta', 'mes', 'mc']

FIELDS = ['mesh', 'grid_len', 'algo', 'max_iters', 'short_arcs',
          'hausdorff', 'chamfer', 'f1']


def run(mesh, grid_len, max_iters, short_arcs):
    """Reconstruct one run, unless its .obj is already there.

    We chdir into RESULTS because test_our_method exports to 'out/<mesh>/'
    relative to the working directory.
    """
    from SDF_to_surface_3D import Options, test_our_method

    arcs = 'shortArcs' if short_arcs else 'noShortArcs'
    stem = f'ours_{grid_len}_{max_iters}_{arcs}_'
    out = RESULTS / 'out' / mesh
    out.mkdir(parents=True, exist_ok=True)
    if list(out.glob(stem + '*.obj')):
        print(f'[skip] {mesh} {stem} — already reconstructed', flush=True)
        return

    print(f'\n=== {mesh} gl={grid_len} max_iters={max_iters} {arcs} ===', flush=True)
    cwd = Path.cwd()
    os.chdir(RESULTS)
    try:
        # Only the two swept knobs are passed; everything else stays at the
        # Options defaults, which are the single definition of the main config.
        _common._guard('ours', test_our_method,
                       Options(name=mesh, grid_len=grid_len, max_iters=max_iters,
                               turn_off_short_arcs=not short_arcs))
    finally:
        os.chdir(cwd)


def run_baselines(mesh, grid_len):
    """RFTA, MES and marching cubes on this mesh, once each.

    They have no iteration count, so they are a flat line under the curve
    rather than another point on it. They see exactly the samples every ours_
    run was given: generate_test_mesh_data draws from a generator seeded with
    _common.SEED, so regenerating the samples here reproduces them rather than
    drawing new ones, and the gap is the method and not its input.
    """
    from SDF_to_surface_3D import (Options, generate_test_mesh_data,
                                   test_rfta, test_mes, test_mc)

    out = RESULTS / 'out' / mesh
    out.mkdir(parents=True, exist_ok=True)
    todo = [(algo, fn) for algo, fn in (('rfta', test_rfta), ('mes', test_mes),
                                        ('mc', test_mc))
            if not list(out.glob(_common.PREFIX[algo] + '*.obj'))]
    if not todo:
        print(f'[skip] {mesh} baselines — already reconstructed', flush=True)
        return

    print(f'\n=== {mesh} gl={grid_len} → {", ".join(a for a, _ in todo)} ===', flush=True)
    options = Options(name=mesh, grid_len=grid_len)  # path_to_obj is absolute
    cwd = Path.cwd()
    os.chdir(RESULTS)
    try:
        _, points, distances, _ = generate_test_mesh_data(
            options.path_to_obj, mesh, grid_len=grid_len)
        for algo, fn in todo:
            _common._guard(algo, fn, options, sdf=(points, distances))
    finally:
        os.chdir(cwd)


def _mesh_rank(mesh):
    """MESHES order first, then whatever else is on disk, alphabetically."""
    order = list(MESHES)
    return (order.index(mesh) if mesh in order else len(order), mesh)


def metrics():
    """Hausdorff / Chamfer / F1 of every reconstruction on disk, vs the GT.

    Walks the output tree rather than MESHES, so a sweep run in several
    batches -- or one whose mesh list has moved on since -- still reports
    every reconstruction that is there.
    """
    import numpy as np
    import trimesh
    from util import mesh_distances

    rows = []
    root = RESULTS / 'out'
    for mesh_dir in sorted(p for p in root.glob('*') if p.is_dir()):
        mesh = mesh_dir.name
        for obj in sorted(mesh_dir.glob('*.obj')):
            algo = next((a for a in ALGOS
                         if obj.name.startswith(_common.PREFIX[a])), None)
            m = (NAME if algo == 'ours' else BASELINE_NAME).match(obj.name) \
                if algo else None
            if m is None:
                continue
            recon = trimesh.load(str(obj), force='mesh')
            if len(recon.faces) == 0:
                continue
            # mesh_distances samples both surfaces for Chamfer and F1; seed it
            # so a rerun reproduces the CSV digit for digit.
            np.random.seed(_common.SEED)
            haus, cham, f1 = mesh_distances(recon, _common.gt_mesh(mesh))
            # The baselines have no iteration count and no short arcs; those
            # two columns are left empty for them rather than filled with a
            # value that would read as a setting they were run at.
            rows.append({'mesh': mesh, 'grid_len': int(m['grid_len']),
                         'algo': algo,
                         'max_iters': int(m['iters']) if algo == 'ours' else None,
                         'short_arcs': m['arcs'] == 'shortArcs' if algo == 'ours' else None,
                         'hausdorff': round(haus, 6), 'chamfer': round(cham, 8),
                         'f1': round(f1, 4)})
    # Per mesh: the no-short-arc run, then the iteration curve, then the
    # baselines under it.
    rows.sort(key=lambda r: (_mesh_rank(r['mesh']), ALGOS.index(r['algo']),
                             bool(r['short_arcs']),
                             -1 if r['max_iters'] is None else r['max_iters']))
    return rows


def report():
    """Rewrite metrics.csv from the reconstructions on disk, running nothing.

    This is the whole of what collect.py does for the other experiments, which
    is why it calls this rather than _common.collect().
    """
    _common.load_sdf3d()
    rows = metrics()
    _common._write_csv(RESULTS / 'metrics.csv', FIELDS, rows)

    print(f'\n{"mesh":<12}{"gl":>4}{"algo":>6}{"iters":>7}{"arcs":>6}'
          f'{"hausdorff":>12}{"chamfer":>12}{"f1":>9}')
    for r in rows:
        iters = '-' if r['max_iters'] is None else r['max_iters']
        arcs = '-' if r['short_arcs'] is None else ('yes' if r['short_arcs'] else 'no')
        print(f'{r["mesh"]:<12}{r["grid_len"]:>4}{r["algo"]:>6}{iters:>7}{arcs:>6}'
              f'{r["hausdorff"]:>12.5f}{r["chamfer"]:>12.6f}{r["f1"]:>9.4f}')
    print(f'\n{len(rows)} rows -> {RESULTS / "metrics.csv"}')
    return rows


# Guarded so collect.py can import report() without setting a sweep going.
if __name__ == '__main__':
    _common.load_sdf3d()
    t0 = time.perf_counter()

    for iters in MAX_ITERS:
        for mesh, grid_len in MESHES.items():
            run(mesh, grid_len, iters, short_arcs=True)
    for mesh, grid_len in MESHES.items():
        # max_iters=0 with the short arcs off: no arcs means no degenerate-arc
        # points get added either, so this is the bare interpolant of the input
        # samples alone -- what both the arcs and the iteration start from.
        run(mesh, grid_len, 0, short_arcs=False)
        run_baselines(mesh, grid_len)

    report()
    print(f'Total: {time.perf_counter() - t0:.1f} s')
