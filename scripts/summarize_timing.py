"""
Aggregate per-task timing CSVs into a paper-ready (algo, grid_len) table.

Two modes:

(1) Single source (default for the timing_4algo benchmark) — pass --csv-glob
    pointing at one batch's logs. All algos/file_ids in those CSVs are kept
    and (optionally) intersected with --index.

(2) Multi-source mode (the legacy 4-group merge from earlier batches) —
    pass --multi-source. Pulls (ours, mc) from new_solid/logs and (rfta, mes)
    from the older 500-mesh sw=10 archive, intersected with new_solid index.

Outputs:
  - prints n / mean / median / P25 / P75 / max per (algo, gl)
  - writes timing_summary.tsv (and timing_summary.raw.tsv) next to --output
"""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import numpy as np
import pandas as pd


LEGACY_SOURCES = [
    # (algo_keep, src_glob)
    ('ours', '/scratch/ycheng27/new_solid/logs/*.timing.csv'),
    ('mc',   '/scratch/ycheng27/new_solid/logs/*.timing.csv'),
    ('rfta', '/scratch/ycheng27/sdfgradients/baselines/logs.run1-20260424-1520/*.timing.csv'),
    ('mes',  '/scratch/ycheng27/sdfgradients/baselines/logs.run1-20260424-1520/*.timing.csv'),
]


def load_csv_glob(src_glob: str, algo_filter: str | None = None) -> pd.DataFrame:
    """Load every <fid>.timing.csv matching ``src_glob``. If ``algo_filter``
    is set, keep only rows where algo == algo_filter (used by the legacy
    multi-source mode where each glob contributes one algo). Otherwise keep
    all algos."""
    paths = sorted(glob.glob(src_glob))
    rows: list[dict] = []
    for p in paths:
        fid = Path(p).stem.replace('.timing', '')
        with open(p, newline='') as f:
            for r in csv.DictReader(f):
                if algo_filter is not None and r['algo'] != algo_filter:
                    continue
                if r['status'] != 'ok':
                    continue
                try:
                    wall = float(r['wall_s'])
                except ValueError:
                    continue
                rows.append({
                    'file_id': fid,
                    'algo': r['algo'],
                    'grid_len': int(r['grid_len']),
                    'wall_s': wall,
                })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--csv-glob',
                    default='/scratch/ycheng27/timing_4algo/logs/*.timing.csv',
                    help='glob of <fid>.timing.csv files (single-source mode)')
    ap.add_argument('--multi-source', action='store_true',
                    help='legacy: merge ours/mc from new_solid + rfta/mes '
                         'from logs.run1 archive')
    ap.add_argument('--index', default=None,
                    help='optional index.csv to filter file_ids; default: '
                         'no filter for single-source, new_solid for multi-source')
    ap.add_argument('--output',
                    default='/scratch/ycheng27/timing_4algo/timing_summary.tsv')
    args = ap.parse_args()

    if args.multi_source:
        if args.index is None:
            args.index = '/scratch/ycheng27/new_solid/index.csv'
        keep_ids = set()
        with open(args.index, newline='') as f:
            for r in csv.DictReader(f):
                keep_ids.add(str(r['file_id']))
        print(f'index: {args.index}  →  {len(keep_ids)} mesh ids')
        frames = [load_csv_glob(g, algo_filter=a) for a, g in LEGACY_SOURCES]
        df = pd.concat(frames, ignore_index=True)
        n_total = len(df)
        df = df[df['file_id'].isin(keep_ids)].copy()
        print(f'rows kept: {len(df):>6} / {n_total} (filtered to index)')
    else:
        print(f'csv-glob: {args.csv_glob}')
        df = load_csv_glob(args.csv_glob)
        if df.empty:
            print('no rows found — nothing to summarise')
            return 1
        if args.index is not None:
            keep_ids = set()
            with open(args.index, newline='') as f:
                for r in csv.DictReader(f):
                    keep_ids.add(str(r['file_id']))
            n_total = len(df)
            df = df[df['file_id'].isin(keep_ids)].copy()
            print(f'rows kept: {len(df):>6} / {n_total} (filtered to {args.index})')
        else:
            print(f'rows kept: {len(df):>6}')

    grids = sorted(df['grid_len'].unique())
    algos = ['rfta', 'mes', 'mc', 'ours']

    # ── per (algo, gl) summary ──────────────────────────────────────────
    summary = (df.groupby(['algo', 'grid_len'])['wall_s']
                 .agg(n='count',
                      mean='mean',
                      median='median',
                      p25=lambda s: np.percentile(s, 25),
                      p75=lambda s: np.percentile(s, 75),
                      p95=lambda s: np.percentile(s, 95),
                      max='max')
                 .reset_index())
    order = {a: i for i, a in enumerate(algos)}
    summary['_o'] = summary['algo'].map(order).fillna(99)
    summary = summary.sort_values(['_o', 'grid_len']).drop(columns='_o')

    print()
    print(summary.to_string(index=False,
                            float_format=lambda x: f'{x:>9.3f}'))

    # ── compact (algo × gl) matrix views ────────────────────────────────
    for label, agg in [('mean', 'mean'), ('median', 'median')]:
        print(f'\n=== wall_s {label} (algo × grid_len) ===')
        piv = (df.groupby(['algo', 'grid_len'])['wall_s']
                 .agg(agg).unstack().reindex(algos).dropna(how='all'))
        print(piv.to_string(float_format=lambda x: f'{x:>10.2f}'))

    print('\n=== n (samples per cell) ===')
    piv = (df.groupby(['algo', 'grid_len'])['wall_s']
             .count().unstack().reindex(algos).dropna(how='all').astype('Int64'))
    print(piv.to_string())

    # ── write tidy TSV ───────────────────────────────────────────────────
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, sep='\t', index=False, float_format='%.4f')
    print(f'\nwrote {out}')

    # also dump the raw wide table (every per-mesh wall_s) for boxplots
    raw = out.with_suffix('.raw.tsv')
    df.sort_values(['algo', 'grid_len', 'file_id']).to_csv(
        raw, sep='\t', index=False, float_format='%.4f')
    print(f'wrote {raw}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
