"""
Summarise topology consistency from ``topology.csv``: per (algo, gl),
report the median of ``|Δcomponents| + |Δgenus|`` across all 500 meshes,
plus the fraction of reconstructions that came out non-watertight (where
genus is undefined and we couldn't fold them into the genus-error term).

Usage:
    python scripts/summarize_topology.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = '/scratch/ycheng27/sdfgradients/baselines/topology.csv'


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', default=DEFAULT_INPUT)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.input)

    for col in ('delta_components', 'delta_genus',
                'recon_components', 'recon_n_boundary'):
        df[col] = pd.to_numeric(df[col], errors='coerce')

    err = df['err'].fillna('').astype(str)
    failed = err.str.len() > 0

    # Headline scalar: |Δcomponents| + |Δgenus|, taking the absolute
    # difference between recon and GT topology. Δgenus is NaN when the
    # recon is non-watertight (genus undefined); those rows propagate
    # NaN into topo_err and drop out of the median below — see the
    # separate watertight-rate table for that bias.
    df['topo_err'] = df['delta_components'].abs() + df['delta_genus'].abs()
    # Failures (missing file / load error): infinite penalty so the
    # median surfaces them.
    df.loc[failed, 'topo_err'] = np.inf

    df['_is_watertight'] = ~df['delta_genus'].isna() & ~failed
    df['_is_failed'] = failed

    grouped = df.groupby(['algo', 'grid_len'])
    summary = grouped.agg(
        n_total=('topo_err', 'size'),
        n_watertight=('_is_watertight', 'sum'),
        n_failed=('_is_failed', 'sum'),
        topo_err_med=('topo_err', 'median'),
        d_comp_abs_med=('delta_components',
                        lambda s: s.abs().median()),
        d_genus_abs_med=('delta_genus',
                         lambda s: s.abs().median()),
    ).reset_index()

    algo_order = {a: i for i, a in enumerate(['rfta', 'mes', 'ours', 'mc'])}
    summary['_o'] = summary['algo'].map(lambda a: algo_order.get(a, 99))
    summary = summary.sort_values(['_o', 'grid_len']).drop(columns='_o')

    print(summary.to_string(index=False,
                            float_format=lambda x: f'{x:.4g}'))

    print('\n=== Median |Δcomponents| by (algo, grid_len) ===')
    piv = grouped['delta_components'].apply(
        lambda s: s.abs().median()).unstack()
    piv = piv.reindex(['rfta', 'mes', 'ours', 'mc']).dropna(how='all')
    print(piv.to_string(float_format=lambda x: f'{x:.2f}'))

    print('\n=== Median |Δgenus| by (algo, grid_len) (watertight only) ===')
    piv2 = grouped['delta_genus'].apply(
        lambda s: s.abs().median()).unstack()
    piv2 = piv2.reindex(['rfta', 'mes', 'ours', 'mc']).dropna(how='all')
    print(piv2.to_string(float_format=lambda x: f'{x:.2f}'))

    print('\n=== Median (|Δcomp|+|Δgenus|), failures→inf ===')
    piv3 = grouped['topo_err'].median().unstack()
    piv3 = piv3.reindex(['rfta', 'mes', 'ours', 'mc']).dropna(how='all')
    print(piv3.to_string(float_format=lambda x: f'{x:.3g}'))

    print('\n=== Watertight rate (n_watertight / n_total) ===')
    rate = grouped['_is_watertight'].mean().unstack()
    rate = rate.reindex(['rfta', 'mes', 'ours', 'mc']).dropna(how='all')
    print(rate.to_string(float_format=lambda x: f'{x:.3f}'))

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
