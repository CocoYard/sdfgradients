"""Shared plumbing for the three degraded-input experiments.

Each script (scattered, truncation, noise) declares only *what* to run; the
mechanics live here. A cell whose .obj already exists is skipped, so an
interrupted sweep resumes, and metrics are recomputed from disk either way.
See README.md for the layout under results/.
"""

from __future__ import annotations

import csv
import os
import re
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
            mc_sdf = _mc_samples(points, distances, options, mesh, grid_len,
                                 noise, bound, scatter) if 'mc' in todo else None
            for algo, fn in (('rfta', test_rfta), ('mes', test_mes), ('mc', test_mc)):
                if algo in todo:
                    _guard(algo, fn, options,
                           sdf=mc_sdf if algo == 'mc' else (points, distances))
    finally:
        os.chdir(cwd)


def _mc_samples(points, distances, options, mesh, grid_len, noise, bound, scatter):
    """The samples marching cubes gets, which differ only under truncation.

    MC is the one method here that needs a dense grid. Handing it the truncated
    point set leaves it reconstructing on whatever sub-grid survived, with no
    sign at all for the cells that went missing -- test_mc has to assume +1
    ("unknown means outside") and hollows the model out as the band tightens.
    So truncation reaches MC the way a TSDF from range data actually stores it:
    every |d| > bound saturated to +/-bound, nothing missing. Inside the band
    the values are untouched, so this is the same degradation, in the only form
    a grid method can read it.

    Regenerating the untruncated samples reproduces the ones the truncated set
    was filtered out of, since generate_test_mesh_data draws from a generator
    seeded with SEED.
    """
    import numpy as np
    from SDF_to_surface_3D import generate_test_mesh_data

    if bound >= 1.0:
        return points, distances
    _, full_points, full_distances, _ = generate_test_mesh_data(
        options.path_to_obj, mesh, grid_len=grid_len,
        noise=noise, bound=1.0, scatter=scatter)
    print(f'  [mc] clamped to +/-{bound}: {len(full_points)} points, '
          f'{int((np.abs(full_distances) > bound).sum())} saturated', flush=True)
    return full_points, np.clip(full_distances, -bound, bound)


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


def evaluate(exp, param, mesh, grid_len, algos, verbose=True, **_):
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
            if verbose:
                print(f'  {param:>12s} {mesh:<12s} {algo:<5s} '
                      f'chamfer={cham:.6f} hausdorff={haus:.5f} f1={f1:.4f}', flush=True)
    return rows


def discover(exp):
    """Find every cell that has already been reconstructed under results/<exp>.

    Driven by what is on disk rather than by a script's cell list, so a sweep
    run in several batches -- or interrupted half way -- still reports
    everything that finished.
    """
    by_algo = {v: k for k, v in PREFIX.items()}
    cells = {}
    for obj in sorted((RESULTS / exp).glob('*/*/out/*/*.obj')):
        param_gl, mesh = obj.parts[-5], obj.parts[-4]
        m = re.match(r'(.+)_gl(\d+)$', param_gl)
        algo = next((a for pre, a in by_algo.items()
                     if obj.name.startswith(pre)), None)
        if not m or algo is None:
            continue
        key = (m.group(1), mesh, int(m.group(2)))
        cells.setdefault(key, set()).add(algo)
    return [dict(param=p, mesh=m, grid_len=g, algos=sorted(a))
            for (p, m, g), a in sorted(cells.items(), key=lambda kv: _order(kv[0]))]


def _order(key):
    """Sort params by the number in their name (noise0.005 before noise0.01)."""
    param, mesh, grid_len = key
    num = re.search(r'[\d.]+$', param)
    return (grid_len, float(num.group()) if num else 0.0, param, mesh)


def _write_csv(path, fieldnames, rows):
    """Write atomically: a stdout redirect aimed at this same path keeps its
    handle on the old inode instead of interleaving into the new file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def summarize(rows):
    """Mean / std / median of each metric across meshes, per (param, algo).

    std is the spread over the models, i.e. how much the model you picked
    matters at that setting -- not an uncertainty on any single number.
    """
    import numpy as np

    groups = {}
    for r in rows:
        groups.setdefault((r['grid_len'], r['param'], r['algo']), []).append(r)

    out = []
    for key in sorted(groups, key=lambda k: _order((k[1], '', k[0])) + (k[2],)):
        grid_len, param, algo = key
        g = groups[key]
        row = {'param': param, 'grid_len': grid_len, 'algo': algo, 'n_meshes': len(g)}
        for metric in ('chamfer', 'hausdorff', 'f1'):
            v = np.array([r[metric] for r in g], dtype=float)
            row[f'{metric}_mean'] = float(np.round(v.mean(), 8))
            row[f'{metric}_std'] = float(np.round(v.std(ddof=1), 8)) if len(v) > 1 else 0.0
            row[f'{metric}_median'] = float(np.round(np.median(v), 8))
        out.append(row)
    return out


def collect(exp, verbose=True):
    """Recompute metrics for everything on disk and rewrite the two CSVs."""
    load_sdf3d()
    cells = discover(exp)
    if not cells:
        print(f'nothing reconstructed yet under {RESULTS / exp}')
        return []

    rows = []
    for cell in cells:
        rows += evaluate(exp=exp, verbose=verbose, **cell)

    _write_csv(RESULTS / exp / 'metrics.csv',
               ['experiment', 'param', 'mesh', 'grid_len', 'algo', 'hausdorff',
                'chamfer', 'f1', 'faces', 'file'], rows)
    summary = summarize(rows)
    _write_csv(RESULTS / exp / 'summary.csv',
               ['param', 'grid_len', 'algo', 'n_meshes'] +
               [f'{m}_{s}' for m in ('chamfer', 'hausdorff', 'f1')
                for s in ('mean', 'std', 'median')], summary)

    print(f'\n=== {exp}: mean +/- std over meshes ===')
    print(f'{"param":<12}{"gl":>5}{"algo":>7}{"n":>4}'
          f'{"chamfer":>22}{"hausdorff":>20}{"f1":>16}')
    last = None
    for r in summary:
        if last is not None and r['param'] != last:
            print()
        last = r['param']
        print(f'{r["param"]:<12}{r["grid_len"]:>5}{r["algo"]:>7}{r["n_meshes"]:>4}'
              f'{r["chamfer_mean"]:>13.6f} +/-{r["chamfer_std"]:<8.6f}'
              f'{r["hausdorff_mean"]:>11.5f} +/-{r["hausdorff_std"]:<7.5f}'
              f'{r["f1_mean"]:>8.4f} +/-{r["f1_std"]:<7.4f}')
    print(f'\n{len(rows)} rows -> {RESULTS / exp / "metrics.csv"}')
    print(f'{len(summary)} rows -> {RESULTS / exp / "summary.csv"}')
    return rows


def sweep(exp, cells):
    """Run every cell, then recompute metrics over everything on disk."""
    load_sdf3d()
    t0 = time.perf_counter()
    for cell in cells:
        run_cell(exp=exp, **cell)
    collect(exp)
    print(f'Total: {time.perf_counter() - t0:.1f} s')
