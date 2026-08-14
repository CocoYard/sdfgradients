"""Shared plumbing for the three degraded-input experiments.

Each experiment script (scattered.py / truncation.py / noise.py) only declares
*what* to run at the top of the file; everything mechanical lives here.

Run any of them with no arguments from anywhere:

    python additional_experiments/scattered.py

Results land under ``additional_experiments/results/<experiment>/``:

    results/<experiment>/<param>_gl<N>/<mesh>/out/<mesh>/*.obj   reconstructions
    results/<experiment>/metrics.csv                             Hausdorff/Chamfer/F1

Reruns are cheap: a cell whose .obj already exists is skipped, so a sweep that
died half way (or that you extended with another parameter value) picks up
where it left off. Metrics are always recomputed from the .obj files on disk,
so the CSV is complete even when nothing was re-run.
"""

from __future__ import annotations

import csv
import os
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = Path(__file__).resolve().parent / 'results'

# Seeds the random draws in generate_test_mesh_data (scattered point positions
# and the noise added to the distances). Fixed so the whole sweep is
# reproducible, and so the baselines can regenerate byte-identical samples.
SEED = 1

# The .obj filename each runner exports starts with this. The runners build
# their own names (they encode grid_len and sometimes the degradation
# parameter), so we glob on the prefix rather than trying to predict the rest.
PREFIX = {'ours': 'ours', 'rfta': 'rfta', 'mes': 'mes', 'mc': 'sample_points'}


def load_sdf3d():
    """Import the main module and point it at this repo.

    `data_dir` must be absolute because run_cell() chdir's into the cell
    directory, and Options builds path_to_obj from it at construction time.
    """
    sys.path.insert(0, str(REPO))
    import SDF_to_surface_3D as sdf3d
    sdf3d.data_dir = str(REPO / 'examples')
    sdf3d.seed = SEED
    return sdf3d


def _cell_dir(exp, param, mesh, grid_len):
    # grid_len is part of the directory, not just the .obj name: the skip check
    # below globs on the algorithm prefix, so two grid_lens sharing a directory
    # would make the second one look already-done.
    return RESULTS / exp / f'{param}_gl{grid_len}' / mesh


def run_cell(exp, param, mesh, grid_len, algos, noise=0.0, bound=1.0, scatter=False):
    """Reconstruct one (mesh, grid_len, degradation) cell with every algo.

    All algorithms see the *same* SDF samples: `ours` generates them and hands
    them to the baselines through `sdf=`, so any difference in the output is a
    difference in the method and not in its input.

    The cell gets its own directory and we chdir into it. The runners export to
    'out/<mesh>/' relative to the CWD, and their filenames do not always encode
    the degradation parameter (test_mes writes mes_50.obj at every noise
    level), so without one directory per cell the sweep would overwrite itself.
    It also gives the MES binary a private place to drop its pwn.csv.
    """
    from SDF_to_surface_3D import (Options, generate_test_mesh_data,
                                   test_our_method, test_rfta, test_mes, test_mc)

    cell = _cell_dir(exp, param, mesh, grid_len)
    out = cell / 'out' / mesh
    out.mkdir(parents=True, exist_ok=True)

    todo = [a for a in algos if not list(out.glob(PREFIX[a] + '*.obj'))]
    if not todo:
        print(f'[skip] {exp}/{param}/{mesh} gl={grid_len} — already reconstructed', flush=True)
        return
    print(f'\n=== {exp}/{param}/{mesh} gl={grid_len} → {", ".join(todo)} ===', flush=True)

    cwd = Path.cwd()
    os.chdir(cell)
    try:
        # Per-run identity plus the one degradation knob, and nothing else:
        # the method configuration lives in Options' defaults alone, so this
        # sweep cannot silently drift away from the main experiments.
        options = Options(name=mesh, grid_len=grid_len,
                          noise=noise, bound=bound, scatter=scatter)

        points = distances = None
        if 'ours' in todo:
            points, distances = _guard('ours', test_our_method, options)
        if points is None and any(a in todo for a in ('rfta', 'mes', 'mc')):
            # 'ours' was skipped or failed, but the baselines still need the
            # samples. The seeded RNG makes this identical to what
            # test_our_method would have generated for the same arguments.
            _, points, distances, _ = generate_test_mesh_data(
                options.path_to_obj, mesh, grid_len=grid_len,
                noise=noise, bound=bound, scatter=scatter)
        if points is not None:
            print(f'SDF samples: {len(points)} points', flush=True)
            for algo, fn in (('rfta', test_rfta), ('mes', test_mes), ('mc', test_mc)):
                if algo in todo:
                    _guard(algo, fn, options, sdf=(points, distances))
    finally:
        os.chdir(cwd)


def _guard(algo, fn, *args, **kwargs):
    """Run one algorithm, timed; a failure loses that cell, not the sweep."""
    t = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
        print(f'  [{algo}] done in {time.perf_counter() - t:.1f} s', flush=True)
        return result
    except Exception:
        print(f'  [{algo}] FAILED after {time.perf_counter() - t:.1f} s', flush=True)
        traceback.print_exc()
        return None


_gt_cache = {}


def gt_mesh(mesh):
    """The ground-truth mesh, normalized exactly as generate_test_mesh_data does."""
    if mesh not in _gt_cache:
        import numpy as np
        import trimesh
        m = trimesh.load(str(REPO / 'examples' / f'{mesh}.obj'), force='mesh')
        lo = np.min(m.vertices, axis=0)
        hi = np.max(m.vertices, axis=0)
        m.vertices -= (lo + hi) / 2
        m.vertices /= np.max(hi - lo)
        _gt_cache[mesh] = m
    return _gt_cache[mesh]


def evaluate(exp, param, mesh, grid_len, algos, **_):
    """Hausdorff / Chamfer / F1 of every reconstruction in one cell vs the GT."""
    import numpy as np
    import trimesh
    from util import mesh_distances

    rows = []
    out = _cell_dir(exp, param, mesh, grid_len) / 'out' / mesh
    for algo in algos:
        for obj in sorted(out.glob(PREFIX[algo] + '*.obj')):
            recon = trimesh.load(str(obj), force='mesh')
            if len(recon.faces) == 0:
                continue
            # mesh_distances samples both surfaces to get Chamfer and F1; seed
            # it so re-running the sweep reproduces the CSV digit for digit.
            np.random.seed(SEED)
            haus, cham, f1 = mesh_distances(recon, gt_mesh(mesh))
            rows.append({'experiment': exp, 'param': param, 'mesh': mesh,
                         'grid_len': grid_len, 'algo': algo,
                         'hausdorff': round(haus, 6), 'chamfer': round(cham, 8),
                         'f1': round(f1, 4), 'faces': len(recon.faces),
                         'file': obj.name})
            print(f'  {param:>12s} {mesh:<12s} {algo:<5s} '
                  f'chamfer={cham:.6f} hausdorff={haus:.5f} f1={f1:.4f}', flush=True)
    return rows


def sweep(exp, cells):
    """Run every cell, then write results/<exp>/metrics.csv."""
    load_sdf3d()
    t0 = time.perf_counter()
    for cell in cells:
        run_cell(exp=exp, **cell)

    print(f'\n=== metrics: {exp} ===', flush=True)
    rows = []
    for cell in cells:
        rows += evaluate(exp=exp, **cell)

    csv_path = RESULTS / exp / 'metrics.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['experiment', 'param', 'mesh', 'grid_len',
                                          'algo', 'hausdorff', 'chamfer', 'f1',
                                          'faces', 'file'])
        w.writeheader()
        w.writerows(rows)
    print(f'\nWrote {len(rows)} rows to {csv_path}')
    print(f'Total: {time.perf_counter() - t0:.1f} s')
