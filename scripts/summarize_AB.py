"""
Summarize the A×B experiment metrics: per (combo, grid_len) mean & median
Hausdorff / Chamfer / F1, plus survival counts. Unlike summarize_metrics.py
this does not filter to a hardcoded algo list, so it prints whatever combos
are present in the CSV.

Usage:
    python scripts/summarize_AB.py --csv /scratch/.../baselines/metrics_AB.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv', default='/scratch/ycheng27/sdfgradients/baselines/metrics_AB.csv')
    p.add_argument('--output', default=None, help='optional TSV dump of the summary table')
    p.add_argument('--report', default=None,
                   help='write the printed tables (incl. the per-metric pivots) '
                        'to this text file as well as stdout')
    args = p.parse_args()

    lines: list[str] = []

    def emit(text=''):
        """Print and remember, so --report saves exactly what you see."""
        print(text)
        lines.append(str(text))

    df = pd.read_csv(args.csv)
    for col in ('hausdorff', 'chamfer', 'f1'):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # Failures (err set) are penalised: haus/cham=inf, f1=0.
    bad = df['err'].fillna('').astype(str).str.len() > 0
    df.loc[bad, ['hausdorff', 'chamfer']] = np.inf
    df.loc[bad, 'f1'] = 0.0
    df['_real'] = np.isfinite(df['hausdorff'])

    g = df.groupby(['algo', 'grid_len'])
    summary = g.agg(
        n=('hausdorff', 'size'),
        n_ok=('_real', 'sum'),
        haus_mean=('hausdorff', lambda s: s[np.isfinite(s)].mean()),
        haus_med=('hausdorff', 'median'),
        cham_mean=('chamfer', lambda s: s[np.isfinite(s)].mean()),
        cham_med=('chamfer', 'median'),
        f1_mean=('f1', 'mean'),
        f1_med=('f1', 'median'),
    ).reset_index()

    pd.set_option('display.width', 200)
    emit(summary.to_string(index=False, float_format=lambda x: f'{x:.5g}'))

    real = df[df['_real']]
    for metric in ('f1', 'chamfer', 'hausdorff'):
        for stat in ('mean', 'median'):
            emit(f'\n=== {stat.capitalize()} {metric} by (combo, grid_len) '
                 f'— successful cells only ===')
            piv = real.groupby(['algo', 'grid_len'])[metric].agg(stat).unstack()
            emit(piv.to_string(float_format=lambda x: f'{x:.6g}'))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.output, sep='\t', index=False)
        print(f'\nTSV -> {args.output}')
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text('\n'.join(lines) + '\n')
        print(f'report -> {args.report}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
