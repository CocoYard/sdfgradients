"""
Summarize a metrics CSV: a written summary first, then per (combo, grid_len)
mean & median tables and survival counts. Unlike summarize_metrics.py this does
not filter to a hardcoded algo list, so it prints whatever combos are present.

Works on either metric family, picked up from the CSV's columns:
    hausdorff / chamfer / f1   (compute_metrics.py)
    ecd / ef1                  (compute_edge_metrics.py)

Usage:
    python scripts/summarize_AB.py --csv /scratch/.../metrics_AB.csv
    python scripts/summarize_AB.py --csv /scratch/.../edge_metrics_sharp50.csv \
        --report /scratch/.../edge_report.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# metric -> (lower_is_better, penalty value applied to failed cells)
METRICS = {
    'hausdorff': (True, np.inf),
    'chamfer': (True, np.inf),
    'ecd': (True, np.inf),
    'f1': (False, 0.0),
    'ef1': (False, 0.0),
}


def _fmt(v: float) -> str:
    return f'{v:.4g}'


def _summarize(df, real, metrics, grids) -> list[str]:
    """Lead with the finding: per metric, who wins at the finest grid and by how
    much, whether that holds across grids, and what the failures qualify."""
    out = ['=' * 78, 'SUMMARY', '=' * 78]
    fine = grids[-1]

    for m in metrics:
        lower_better, _ = METRICS[m]
        piv = real.groupby(['algo', 'grid_len'])[m].mean().unstack()
        if fine not in piv.columns:
            continue
        col = piv[fine].dropna().sort_values(ascending=lower_better)
        if len(col) < 2:
            continue
        best, second = col.index[0], col.index[1]
        ratio = (col.iloc[1] / col.iloc[0]) if lower_better else (col.iloc[0] / col.iloc[1])
        wins = sum(1 for g in piv.columns
                   if piv[g].notna().any()
                   and (piv[g].idxmin() if lower_better else piv[g].idxmax()) == best)
        arrow = 'lower' if lower_better else 'higher'
        out.append(f'- {m.upper():<9} ({arrow} is better): best at gl={fine} is '
                   f'{best} = {_fmt(col.iloc[0])}, '
                   f'{ratio:.2f}x better than {second} ({_fmt(col.iloc[1])}); '
                   f'best in {wins}/{len(piv.columns)} grid sizes.')

    # Monotonicity: does each combo keep improving as the grid refines? A metric
    # that turns around is worth naming — it means resolution stops helping.
    m0 = metrics[0]
    lower_better, _ = METRICS[m0]
    piv = real.groupby(['algo', 'grid_len'])[m0].mean().unstack()
    turned = []
    for a in piv.index:
        s = piv.loc[a].dropna()
        if len(s) < 3:
            continue
        best_g = s.idxmin() if lower_better else s.idxmax()
        if best_g != s.index[-1]:
            turned.append(f'{a} (best at gl={best_g}, worse after)')
    if turned:
        out.append(f'- {m0.upper()} stops improving with resolution for: '
                   + '; '.join(turned))

    n_fail = int((~df['_real']).sum())
    if n_fail:
        per = df[~df['_real']].groupby('algo').size().sort_values(ascending=False)
        out.append(f'- Data completeness: {n_fail}/{len(df)} cells failed '
                   f'({", ".join(f"{a}:{n}" for a, n in per.items())}). '
                   'Tables below cover successful cells only, so combos with '
                   'failures are flattered by exactly the cases they lost.')
    else:
        out.append(f'- Data completeness: all {len(df)} cells succeeded.')

    out.append('=' * 78)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv', default='/scratch/ycheng27/sdfgradients/baselines/metrics_AB.csv')
    p.add_argument('--output', default=None, help='optional TSV dump of the summary table')
    p.add_argument('--report', default=None,
                   help='write the printed summary + tables to this text file '
                        'as well as stdout')
    args = p.parse_args()

    lines: list[str] = []

    def emit(text=''):
        """Print and remember, so --report saves exactly what you see."""
        print(text)
        lines.append(str(text))

    df = pd.read_csv(args.csv)
    metrics = [m for m in METRICS if m in df.columns]
    if not metrics:
        raise SystemExit(f'no known metric columns in {args.csv}; '
                         f'expected some of {list(METRICS)}')

    for m in metrics:
        df[m] = pd.to_numeric(df[m], errors='coerce')
    # Failures (err set) are penalised so they cannot be silently dropped.
    bad = df['err'].fillna('').astype(str).str.len() > 0
    for m in metrics:
        df.loc[bad, m] = METRICS[m][1]
    # A cell is real if its lower-is-better metric is finite (or, for a
    # higher-is-better-only CSV, if it wasn't flagged as failed).
    lower = [m for m in metrics if METRICS[m][0]]
    df['_real'] = np.isfinite(df[lower[0]]) if lower else ~bad

    real = df[df['_real']]
    grids = sorted(df.grid_len.unique())

    pd.set_option('display.width', 240)
    for ln in _summarize(df, real, metrics, grids):
        emit(ln)
    emit()

    agg = {'n': (metrics[0], 'size'), 'n_ok': ('_real', 'sum')}
    for m in metrics:
        agg[f'{m}_mean'] = (m, lambda s: s[np.isfinite(s)].mean())
        agg[f'{m}_med'] = (m, 'median')
    summary = df.groupby(['algo', 'grid_len']).agg(**agg).reset_index()
    emit(summary.to_string(index=False, float_format=lambda x: f'{x:.5g}'))

    for metric in metrics:
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
