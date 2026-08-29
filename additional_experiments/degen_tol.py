"""Experiment 5 — sensitivity to the short-arc threshold.

`degen_tol` is the total exposed-arc length below which a sphere's exposed
region is treated as collapsed to a tangent point, and the midpoint of its
short arc becomes a surface candidate — a zero-valued point handed to the RBF
fit, which is what pins the gradient of that sample. Lowering the threshold
admits fewer, better-conditioned candidates; raising it admits more, some of
which sit off the surface.

Two things are measured, and they answer different questions:

  Candidate quality  How far each candidate lies from the true surface. This
                     is a property of the threshold alone, measured before the
                     reconstruction sees it.
  Reconstruction     Hausdorff / Chamfer / F1 of the extracted mesh against
                     the ground truth, i.e. whether any of it reaches the
                     output.

    python additional_experiments/degen_tol.py

Only `ours` runs: the threshold is a knob of our method, so no baseline has an
answer to sweep. Everything else stays at the Options defaults.
"""

import os
import time
from pathlib import Path

import _common

MESH = 'eiffel'

# Two resolutions, because the candidate population changes with sample
# spacing: at 20^3 the short arcs come from a few genuinely thin features, at
# 50^3 from many more spheres whose exposed region is merely small.
GRID_LENS = [20, 50]

# The swept threshold, a length in units of the unit-cube-normalized mesh (not
# an angle). 1e-5 is the pipeline default, and the sweep runs an order of
# magnitude either side of it.
DEGEN_TOLS = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

# Surface-distance thresholds the candidate precision is reported at. The
# candidates a short arc yields are exact up to arithmetic, not up to sample
# spacing, so the interesting range sits well below the mesh scale: at 1e-4 and
# looser every column would read 100% and say nothing.
PRECISION_TAUS = [1e-8, 1e-7, 1e-6, 1e-5]

RESULTS = _common.RESULTS / 'degen_tol'

FIELDS = (['mesh', 'grid_len', 'degen_tol', 'n_cand_raw', 'n_cand',
           'cand_dist_mean', 'cand_dist_median', 'cand_dist_max']
          + [f'cand_within_{tau:.0e}' for tau in PRECISION_TAUS]
          + ['hausdorff', 'chamfer', 'f1', 'faces', 'file'])


def _cell(grid_len, degen_tol):
    # One directory per cell: the exported .obj name does not encode the
    # threshold (ours puts only grid_len, iters and the short-arc flag in it),
    # so cells sharing a directory would overwrite each other.
    return RESULTS / f'eps{degen_tol:.0e}_gl{grid_len}' / MESH


def run(grid_len, degen_tol):
    """Reconstruct one cell, unless it is already on disk.

    The candidates are saved next to the reconstruction because they cannot be
    recovered from it: they live in the C++ Options that main_algorithm filled
    in, and that object is gone once the run returns. `report()` then works
    from disk alone, like the other experiments.
    """
    import numpy as np
    from SDF_to_surface_3D import Options, test_our_method

    cell = _cell(grid_len, degen_tol)
    out = cell / 'out' / MESH
    out.mkdir(parents=True, exist_ok=True)
    cand_npz = cell / 'candidates.npz'
    if list(out.glob('ours*.obj')) and cand_npz.exists():
        print(f'[skip] gl={grid_len} degen_tol={degen_tol:.0e} — already reconstructed',
              flush=True)
        return

    print(f'\n=== {MESH} gl={grid_len} degen_tol={degen_tol:.0e} ===', flush=True)
    cwd = Path.cwd()
    os.chdir(cell)
    try:
        # The swept knob and the run identity, and nothing else: the method
        # configuration is the Options defaults, as in every other experiment.
        options = Options(name=MESH, grid_len=grid_len, degen_tol=degen_tol)
        if _common._guard('ours', test_our_method, options) is None:
            return
        # Set by _build_cpp_options; main_algorithm fills its degenerate_pts in
        # place, so these are the candidates that survived filter_degenerate_pts
        # — exactly the zero-valued points the second RBF fit was given.
        idx, pts = options.cpp_options.degenerate_points
        # The pre-filter midpoints as well: a sphere can own several, and how
        # many the filter had to choose between is part of what the threshold
        # controls. Kept so report() can tell the two populations apart.
        raw_idx, raw_pts = options.cpp_options.short_arc_candidates
        np.savez(cand_npz, idx=np.asarray(idx).ravel(),
                 pts=np.asarray(pts).reshape(-1, 3),
                 raw_idx=np.asarray(raw_idx).ravel(),
                 raw_pts=np.asarray(raw_pts).reshape(-1, 3),
                 grid_len=grid_len, degen_tol=degen_tol)
        print(f'  [cand] {len(raw_pts)} short-arc midpoints, {len(idx)} kept '
              f'-> {cand_npz.name}', flush=True)
    finally:
        os.chdir(cwd)


def _candidate_stats(pts, gt):
    """Distance from each candidate to the GT surface, summarized.

    Point-to-surface, not point-to-vertex: a candidate is a point in space and
    the GT is a triangulated surface, so the nearest point on a face is the
    honest distance.
    """
    import igl
    import numpy as np

    row = {'n_cand': len(pts)}
    if len(pts) == 0:
        return row | {'cand_dist_mean': None, 'cand_dist_median': None,
                      'cand_dist_max': None} | {
            f'cand_within_{tau:.0e}': None for tau in PRECISION_TAUS}
    V = np.asarray(gt.vertices, dtype=np.float64)
    F = np.asarray(gt.faces, dtype=np.int32)
    sqrD, _, _ = igl.point_mesh_squared_distance(
        np.asarray(pts, dtype=np.float64), V, F)
    d = np.sqrt(sqrD)
    row |= {'cand_dist_mean': float(d.mean()),
            'cand_dist_median': float(np.median(d)),
            'cand_dist_max': float(d.max())}
    for tau in PRECISION_TAUS:
        row[f'cand_within_{tau:.0e}'] = round(float((d <= tau).mean()), 4)
    return row


def metrics():
    """Candidate quality and reconstruction quality for every cell on disk.

    Driven by the directory tree rather than by the lists above, so a sweep run
    in batches -- or one whose lists have moved on -- still reports everything
    that finished.
    """
    import numpy as np
    import trimesh
    from util import mesh_distances

    rows = []
    for cand_npz in sorted(RESULTS.glob(f'*/{MESH}/candidates.npz')):
        cell = cand_npz.parent
        data = np.load(cand_npz)
        grid_len, degen_tol = int(data['grid_len']), float(data['degen_tol'])
        gt = _common.gt_mesh(MESH)
        row = {'mesh': MESH, 'grid_len': grid_len, 'degen_tol': degen_tol,
               'n_cand_raw': len(data['raw_pts'])}
        # The stats are of the kept candidates: those are the zero-valued
        # points the fit was given, so they are the ones the reconstruction
        # can be blamed on. n_cand_raw only says how many the filter saw.
        row |= _candidate_stats(data['pts'], gt)

        objs = sorted((cell / 'out' / MESH).glob('ours*.obj'))
        recon = trimesh.load(str(objs[0]), force='mesh') if objs else None
        if recon is None or len(recon.faces) == 0:
            rows.append(row | {'hausdorff': None, 'chamfer': None, 'f1': None,
                               'faces': 0, 'file': ''})
            continue
        # mesh_distances samples both surfaces for Chamfer and F1; seed it so a
        # rerun reproduces the CSV digit for digit.
        np.random.seed(_common.SEED)
        haus, cham, f1 = mesh_distances(recon, gt)
        rows.append(row | {'hausdorff': round(haus, 6), 'chamfer': round(cham, 8),
                           'f1': round(f1, 4), 'faces': len(recon.faces),
                           'file': objs[0].name})
    rows.sort(key=lambda r: (r['grid_len'], -r['degen_tol']))
    return rows


def _fmt(v, spec):
    return '-' if v is None else format(v, spec)


def report(latex=False):
    """Rewrite metrics.csv from what is on disk, running nothing."""
    import numpy as np

    _common.load_sdf3d()
    rows = metrics()
    if not rows:
        print(f'nothing reconstructed yet under {RESULTS}')
        return []
    _common._write_csv(RESULTS / 'metrics.csv', FIELDS, rows)

    taus = ''.join(f'{"<=" + format(tau, ".0e"):>10}' for tau in PRECISION_TAUS)
    print(f'\n{"gl":>4}{"degen_tol":>11}{"#raw":>8}{"#cand":>8}{"d_surf":>11}{taus}'
          f'{"hausdorff":>12}{"chamfer":>11}{"f1":>9}')
    last = None
    for r in rows:
        if last is not None and r['grid_len'] != last:
            print()
        last = r['grid_len']
        prec = ''.join(_fmt(r[f'cand_within_{tau:.0e}'], '10.1%')
                       for tau in PRECISION_TAUS)
        print(f'{r["grid_len"]:>4}{r["degen_tol"]:>11.0e}{r["n_cand_raw"]:>8}'
              f'{r["n_cand"]:>8}'
              f'{_fmt(r["cand_dist_mean"], "11.3e")}{prec}'
              f'{_fmt(r["hausdorff"], "12.5f")}{_fmt(r["chamfer"], "11.6f")}'
              f'{_fmt(r["f1"], "9.4f")}')
    print(f'\n{len(rows)} rows -> {RESULTS / "metrics.csv"}')
    if latex:
        print('\n% body rows for tab:degen-tol')
        for r in rows:
            m, e = f'{r["cand_dist_mean"]:.1e}'.split('e')
            prec = ' & '.join(f'{r[f"cand_within_{tau:.0e}"]:.1%}'.replace('%', r'\%')
                              for tau in PRECISION_TAUS)
            print(f'{r["grid_len"]} & $10^{{{int(round(np.log10(r["degen_tol"])))}}}$ '
                  f'& {r["n_cand"]} & ${m}\\!\\times\\!10^{{{int(e)}}}$ & {prec} '
                  f'& {r["hausdorff"]:.4f} & {r["chamfer"]:.5f} & {r["f1"]:.4f} \\\\')
    return rows


if __name__ == '__main__':
    _common.load_sdf3d()
    t0 = time.perf_counter()
    for grid_len in GRID_LENS:
        for degen_tol in DEGEN_TOLS:
            run(grid_len, degen_tol)
    report(latex=True)
    print(f'Total: {time.perf_counter() - t0:.1f} s')
