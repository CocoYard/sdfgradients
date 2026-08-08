"""
Rank the dataset meshes by how much sharp feature they carry.

"Sharpness" is measured with the *same* criterion the ECD/EF1 metric uses
(edgeChamfer._extract_edge_points): the fraction of surface samples whose
neighbourhood within EF1_RADIUS contains a near-perpendicular normal. Using
the metric's own definition avoids inventing a second, inconsistent notion
of "sharp" (e.g. a dihedral-angle heuristic) for selecting the subset the
metric will then be reported on.

Meshes are normalised exactly like compute_metrics.py does (centre at origin,
longest axis to 1), so EF1_RADIUS -- an absolute distance -- means the same
thing on every mesh.

    python scripts/rank_sharpness.py --output /scratch/.../sharpness.csv
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import trimesh

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import edgeChamfer                                                    # noqa: E402

DEFAULT_INDEX = '/scratch/ycheng27/new_solid/index_obj.csv'
DEFAULT_GT_ROOT = '/scratch/ycheng27/new_solid/preview_obj'


def _normalize_in_place(mesh: trimesh.Trimesh) -> None:
    """Centre at origin, scale longest axis to 1 — same as compute_metrics.py."""
    v = np.asarray(mesh.vertices, dtype=np.float64)
    mn, mx = v.min(axis=0), v.max(axis=0)
    mesh.vertices = (v - (mn + mx) / 2) / np.max(mx - mn)


def _one(packed):
    file_id, gt_path, sample_num = packed
    try:
        gt = trimesh.load(str(gt_path), force='mesh')
        _normalize_in_place(gt)
        pts, _ = edgeChamfer.extract_edge_points(
            gt, sample_num=sample_num, normalize_to_unit=False, seed=0)
        return (file_id, len(pts), len(pts) / sample_num, len(gt.faces), '')
    except Exception as e:
        return (file_id, 0, 0.0, 0, f'{type(e).__name__}: {e}'[:150])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', default=DEFAULT_INDEX)
    p.add_argument('--gt-root', default=DEFAULT_GT_ROOT)
    p.add_argument('--sample-num', type=int, default=edgeChamfer.SAMPLE_NUM)
    p.add_argument('--output', required=True)
    p.add_argument('--workers', type=int, default=mp.cpu_count())
    args = p.parse_args()

    rows = list(csv.DictReader(open(args.index)))
    jobs = [(r['file_id'], Path(args.gt_root) / f"{r['file_id']}.obj",
             args.sample_num) for r in rows]
    print(f'ranking {len(jobs)} meshes with {args.workers} workers ...', flush=True)

    with mp.Pool(args.workers) as pool:
        out = pool.map(_one, jobs, chunksize=4)

    out.sort(key=lambda r: -r[2])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['file_id', 'n_edge_pts', 'edge_ratio', 'n_faces', 'err'])
        w.writerows(out)
    ok = [r for r in out if not r[4]]
    print(f'wrote {args.output}  ({len(ok)} ok, {len(out) - len(ok)} failed)')
    ratios = np.array([r[2] for r in ok])
    for q in (100, 90, 75, 50, 25, 0):
        print(f'  p{q:<3} edge_ratio = {np.percentile(ratios, q):.4f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
