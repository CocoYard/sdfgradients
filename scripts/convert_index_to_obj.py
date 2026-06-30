"""
Convert every mesh listed in index.csv from its source .stl (as cached by
thingi10k) to .obj, so the filtered subset can be browsed with standard
mesh viewers / the IDE's file explorer.

Output layout:
    {out_dir}/{file_id}.obj    — one file per row in index.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--index', default='/scratch/ycheng27/thingi10k/index.csv')
    p.add_argument('--out-dir', default='/scratch/ycheng27/thingi10k/preview_obj')
    p.add_argument('--force', action='store_true',
                   help='re-convert even if the .obj already exists')
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import trimesh

    with open(args.index, newline='') as f:
        rows = list(csv.DictReader(f))
    print(f'{len(rows)} meshes in {args.index}', flush=True)

    t0 = time.perf_counter()
    n_ok = n_skip = n_fail = 0
    for i, row in enumerate(rows):
        file_id = row['file_id']
        src = Path(row['file_path'])
        dst = out_dir / f'{file_id}.obj'
        if dst.exists() and dst.stat().st_size > 0 and not args.force:
            n_skip += 1
            continue
        try:
            m = trimesh.load(str(src), force='mesh')
            m.export(str(dst))
            n_ok += 1
        except Exception as e:
            n_fail += 1
            print(f'[{i+1}/{len(rows)}] {file_id}: FAIL {e}', flush=True)
            continue
        if (i + 1) % 50 == 0 or i == len(rows) - 1:
            print(f'[{i+1}/{len(rows)}] ok={n_ok} skip={n_skip} fail={n_fail} '
                  f'elapsed={time.perf_counter() - t0:.1f}s', flush=True)

    print(f'DONE ok={n_ok} skip={n_skip} fail={n_fail}  '
          f'total {time.perf_counter() - t0:.1f}s  '
          f'out -> {out_dir}', flush=True)
    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
