"""
Edge Chamfer Distance (ECD) and Edge F1 (EF1) for every (file_id, algo,
grid_len) cell, on the sharp-feature subset produced by rank_sharpness.py.

ECD/EF1 only look at points near creases, so they are the metric that actually
sees sharp features -- plain Chamfer is dominated by the large smooth regions.
Restricting the report to sharp meshes keeps the metric from being averaged
into noise by models that have no creases at all.

Normalisation matches compute_metrics.py: the GT is centred/scaled to a unit
cube, the prediction is left alone because reconstructions are already produced
in that normalised frame. compute_ecd is therefore called with normalize=False
-- passing normalize=True would re-scale the (already normalised) prediction by
the raw GT bounding box and silently corrupt every distance.

    python scripts/compute_edge_metrics.py --sharp-csv .../sharpness.csv \
        --n-sharp 50 --output .../edge_metrics.csv
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
import edgeChamfer                                                    # noqa: E402

DEFAULT_GT_ROOT = '/scratch/ycheng27/new_solid/preview_obj'
DEFAULT_GRIDS = [6, 10, 20, 30, 40, 50, 60, 80, 100]
# algo -> (run dir, filename stem). The stem is what run_baseline/compute_metrics
# wrote: '<stem>_<gl>.obj' under <run dir>/out/<file_id>/.
DEFAULT_SOURCES = {
    'rfta_sw10':         ('/scratch/ycheng27/new_solid', 'rfta_sw10'),
    'mes_sw10':          ('/scratch/ycheng27/new_solid', 'mes_sw10'),
    'mc':                ('/scratch/ycheng27/new_solid', 'mc'),
    # clamp on/off, everything else identical (bfgs, no post-processing, same
    # post-fix binary) — so the gap between these two is the clamp effect.
    'ours_clamp_bfgs':   ('/scratch/ycheng27/new_solid_clamp_bfgs', 'ours'),
    'ours_noclamp_bfgs': ('/scratch/ycheng27/new_solid_noclamp_bfgs', 'ours'),
}


def _normalize_in_place(mesh: trimesh.Trimesh) -> None:
    """Centre at origin, scale longest axis to 1 — same as compute_metrics.py."""
    v = np.asarray(mesh.vertices, dtype=np.float64)
    mn, mx = v.min(axis=0), v.max(axis=0)
    mesh.vertices = (v - (mn + mx) / 2) / np.max(mx - mn)


def _est_pairs_per_point(area: float, sample_num: int, radius: float) -> float:
    """Expected neighbours within `radius` for one of `sample_num` samples spread
    over a surface of `area`. Edge detection calls cKDTree.query_pairs, whose
    result array is O(n_points * this) -- so a collapsed reconstruction (area
    ~1e-4 while the GT spans a unit cube) asks for billions of pairs and tens of
    GB before anything can be measured."""
    return sample_num * np.pi * radius ** 2 / max(area, 1e-12)


def _one(packed):
    file_id, algo, gl, gt_path, pred_path, sample_num, max_ppp = packed
    # Failure encoding mirrors compute_metrics.py: ecd=inf, ef1=0 so a method
    # that fails to produce output is penalised rather than silently dropped.
    fail = ('inf', '0')
    if not pred_path.exists() or pred_path.stat().st_size == 0:
        return (file_id, algo, gl, *fail, 'missing')
    try:
        gt = trimesh.load(str(gt_path), force='mesh')
        _normalize_in_place(gt)
        pred = trimesh.load(str(pred_path), force='mesh')
        if len(pred.vertices) == 0 or len(pred.faces) == 0:
            return (file_id, algo, gl, *fail, 'empty')
        # A reconstruction this collapsed is a failed reconstruction, not a
        # measurable one: record it as a failure instead of hanging the pool.
        ppp = _est_pairs_per_point(float(pred.area), sample_num,
                                   edgeChamfer.EF1_RADIUS)
        if ppp > max_ppp:
            return (file_id, algo, gl, *fail,
                    f'degenerate: area={pred.area:.4g} est_pairs/pt={ppp:.0f}')
        ecd, ef1 = edgeChamfer.compute_ecd(gt, pred, sample_num=sample_num,
                                           normalize=False, seed=0)
        return (file_id, algo, gl, f'{ecd:.6g}', f'{ef1:.6g}', '')
    except Exception as e:
        return (file_id, algo, gl, *fail,
                f'{type(e).__name__}: {e}'.replace('\n', ' ')[:150])


def pick_sharp(sharp_csv: str, n: int) -> list[str]:
    """Top-n meshes by edge_ratio, returned sorted by file_id."""
    rows = [r for r in csv.DictReader(open(sharp_csv)) if not r['err']]
    rows.sort(key=lambda r: -float(r['edge_ratio']))
    top = rows[:n]
    lo = min(float(r['edge_ratio']) for r in top)
    hi = max(float(r['edge_ratio']) for r in top)
    print(f'sharp subset: {len(top)} meshes, edge_ratio {lo:.4f} .. {hi:.4f}')
    return sorted((r['file_id'] for r in top), key=int)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sharp-csv', required=True,
                   help='output of rank_sharpness.py')
    p.add_argument('--n-sharp', type=int, default=50)
    p.add_argument('--gt-root', default=DEFAULT_GT_ROOT)
    p.add_argument('--grids', default=','.join(str(g) for g in DEFAULT_GRIDS))
    p.add_argument('--sample-num', type=int, default=edgeChamfer.SAMPLE_NUM)
    p.add_argument('--max-pairs-per-point', type=float, default=1000.0,
                   help='reject a prediction whose sampling density would make '
                        'edge detection allocate an unbounded pair array '
                        '(healthy meshes sit around 5)')
    p.add_argument('--output', required=True)
    p.add_argument('--subset-output', default=None,
                   help='also write the chosen file_ids here')
    p.add_argument('--workers', type=int, default=mp.cpu_count())
    args = p.parse_args()

    grids = [int(g) for g in args.grids.split(',') if g.strip()]
    fids = pick_sharp(args.sharp_csv, args.n_sharp)
    if args.subset_output:
        Path(args.subset_output).write_text('\n'.join(fids) + '\n')
        print(f'subset -> {args.subset_output}')

    jobs = []
    for fid in fids:
        gt = Path(args.gt_root) / f'{fid}.obj'
        for algo, (root, stem) in DEFAULT_SOURCES.items():
            for gl in grids:
                jobs.append((fid, algo, gl, gt,
                             Path(root) / 'out' / fid / f'{stem}_{gl}.obj',
                             args.sample_num, args.max_pairs_per_point))
    print(f'{len(jobs)} cells ({len(fids)} meshes x {len(DEFAULT_SOURCES)} algos '
          f'x {len(grids)} grids) with {args.workers} workers', flush=True)

    # Stream rows to disk as they finish. Buffering everything until the end
    # means one pathological cell (or a SLURM timeout) throws away every
    # completed measurement; the partial CSV is always usable.
    t0 = time.perf_counter()
    out, done = [], 0
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(args.output) + '.partial')
    with open(partial, 'w', newline='') as pf:
        pw = csv.writer(pf)
        pw.writerow(['file_id', 'algo', 'grid_len', 'ecd', 'ef1', 'err'])
        with mp.Pool(args.workers) as pool:
            for r in pool.imap_unordered(_one, jobs, chunksize=1):
                out.append(r)
                pw.writerow(r)
                done += 1
                if done % 100 == 0:
                    pf.flush()
                    el = time.perf_counter() - t0
                    print(f'  {done}/{len(jobs)}  '
                          f'eta={el / done * (len(jobs) - done) / 60:.1f} min',
                          flush=True)

    out.sort(key=lambda r: (int(r[0]), r[1], r[2]))
    with open(args.output, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['file_id', 'algo', 'grid_len', 'ecd', 'ef1', 'err'])
        w.writerows(out)
    partial.unlink(missing_ok=True)

    bad = [r for r in out if r[5]]
    print(f'\ndone in {(time.perf_counter() - t0) / 60:.1f} min  '
          f'({len(out)} cells, {len(bad)} failed)  -> {args.output}')
    for r in bad[:20]:
        print(f'  {r[0]} {r[1]} gl={r[2]}: {r[5]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
