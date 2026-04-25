"""
Download Thingi10K and build an index of meshes for the baseline batch.

Uses the official ``thingi10k`` Python package:
  1. ``thingi10k.init(variant='raw', cache_dir=...)`` downloads ~8 GB of
     mesh files into the cache (one-time cost, cached on subsequent runs).
  2. ``thingi10k.dataset(manifold=True, closed=True, num_vertices=(lo, hi))``
     returns a HuggingFace ``Dataset`` filtered locally from the cached
     manifest. No extra download.

The script writes an ``index.csv`` with the first N matching entries
(default 500) that the SLURM batch will iterate over.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path


DEFAULT_CACHE_DIR = '/scratch/ycheng27/sdfgradients/thingi10k'


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-dir', default=DEFAULT_CACHE_DIR,
                   help='where thingi10k caches its download')
    p.add_argument('--variant', default='raw', choices=['raw', 'npz', 'tetwild'],
                   help="'raw' keeps original .obj/.stl files on disk which "
                        "test_rfta / test_mes need")
    p.add_argument('--min-vertices', type=int, default=500)
    p.add_argument('--max-vertices', type=int, default=200_000)
    p.add_argument('--num-entries', type=int, default=500,
                   help='how many meshes to keep in the index')
    p.add_argument('--index-path', default=None,
                   help='output index.csv; defaults to {cache-dir}/index.csv')
    p.add_argument('--skip-init', action='store_true',
                   help='skip thingi10k.init(); assumes cache is already populated')
    p.add_argument('--force-redownload', action='store_true')
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = Path(args.index_path) if args.index_path else cache_dir / 'index.csv'

    import thingi10k

    if not args.skip_init:
        print(f'[{args.variant}] thingi10k.init(cache_dir={cache_dir}) ...',
              flush=True)
        t0 = time.perf_counter()
        thingi10k.init(variant=args.variant, cache_dir=str(cache_dir),
                       force_redownload=args.force_redownload)
        print(f'  init() done in {time.perf_counter() - t0:.1f} s', flush=True)

    print(f'Filtering manifold=True closed=True '
          f'num_vertices=[{args.min_vertices}, {args.max_vertices}] ...',
          flush=True)
    ds = thingi10k.dataset(manifold=True, closed=True,
                           num_vertices=(args.min_vertices, args.max_vertices))
    print(f'  matches: {len(ds)} meshes', flush=True)
    if len(ds) == 0:
        print('ERROR: zero matches; loosen the filter', file=sys.stderr)
        return 2

    # Inspect schema on the first row
    schema = list(ds[0].keys())
    print(f'  entry schema ({len(schema)} fields): {schema}', flush=True)

    # Deterministic ordering so re-runs pick the same 500
    try:
        ds = ds.sort('file_id')
    except Exception as e:
        print(f'  (sort by file_id skipped: {e})', flush=True)

    n_keep = min(args.num_entries, len(ds))
    print(f'  keeping first {n_keep} → {index_path}', flush=True)

    # Columns we care about. ``num_components`` and ``genus`` may be absent
    # in some thingi10k versions — we still write them with blanks.
    fields = ['file_id', 'thing_id', 'file_path', 'num_vertices', 'num_facets',
              'num_components', 'genus', 'closed', 'manifold', 'author',
              'license', 'category']

    # Sanity-check extensions
    ext_counts: dict[str, int] = {}
    with index_path.open('w') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i in range(n_keep):
            e = ds[i]
            row = {k: e.get(k, '') for k in fields}
            w.writerow(row)
            ext = Path(str(e.get('file_path', ''))).suffix.lower()
            ext_counts[ext] = ext_counts.get(ext, 0) + 1

    print('  extensions in index:', ext_counts, flush=True)
    print(f'DONE. Index written to {index_path}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
