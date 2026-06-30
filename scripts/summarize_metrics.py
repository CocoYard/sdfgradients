"""
Aggregate metrics CSVs into a (algo, grid_len) median table.

Reads any metrics_*.csv file under the baselines folder (default
/scratch/.../baselines/), concatenates them, and prints a per-cell
median Hausdorff / Chamfer / F1 along with the surviving sample count.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv-glob', default='/scratch/ycheng27/sdfgradients/baselines/metrics_*.csv')
    p.add_argument('--output', default=None,
                   help='write summary as TSV to this path (otherwise stdout only)')
    args = p.parse_args()

    paths = sorted(glob.glob(args.csv_glob))
    if not paths:
        print(f'No CSVs match {args.csv_glob}')
        return 1
    print('Loading:')
    for p_ in paths:
        print(f'  {p_}')

    dfs = [pd.read_csv(p_) for p_ in paths]
    df = pd.concat(dfs, ignore_index=True)

    # Failures (missing file / empty mesh / compute error) are encoded as
    # hausdorff=inf, chamfer=inf, f1=0 in the CSV — so they sort to the
    # tail of any sample and properly penalise methods that didn't produce
    # an output. Older CSVs may still have empty strings for those cells;
    # we coerce those to the same penalty values for backward compatibility.
    ok = df.copy()
    for col in ('hausdorff', 'chamfer', 'f1'):
        ok[col] = pd.to_numeric(ok[col], errors='coerce')
    err_col = ok['err'].fillna('').astype(str)
    bad = err_col.str.len() > 0
    ok.loc[bad, 'hausdorff'] = np.inf
    ok.loc[bad, 'chamfer'] = np.inf
    ok.loc[bad, 'f1'] = 0.0
    ok = ok.dropna(subset=['hausdorff', 'chamfer', 'f1'])

    # n_total counts every cell (including failures). n_ok is the count of
    # cells with a real measurement (haus < inf), so we can see which
    # method/gl combos had a high failure rate driving up the median.
    ok['_is_real'] = np.isfinite(ok['hausdorff'])
    grouped = ok.groupby(['algo', 'grid_len'])
    summary = grouped.agg(
        n_total=('hausdorff', 'size'),
        n_ok=('_is_real', 'sum'),
        haus_med=('hausdorff', 'median'),
        cham_med=('chamfer', 'median'),
        f1_med=('f1', 'median'),
    ).reset_index()

    # nice ordering
    algo_order = {a: i for i, a in enumerate(['rfta_sw1', 'rfta_sw10', 'rfta_sw100', 'mes_sw1', 'mes_sw10', 'mes_sw100', 'ours', 'mc'])}
    summary['_o'] = summary['algo'].map(lambda a: algo_order.get(a, 99))
    summary = summary.sort_values(['_o', 'grid_len']).drop(columns='_o')

    print()
    print(summary.to_string(index=False,
                            float_format=lambda x: f'{x:.5g}'))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.output, sep='\t', index=False)
        print(f'\nTSV → {args.output}')

    # Compact (algo, gl) matrices for quick visual scan. Survival count
    # (n_ok) is reported once in its own table rather than appended to
    # every metric cell — keeps the metric tables narrow and copy-pastable.
    algo_idx = ['rfta_sw1', 'rfta_sw10', 'rfta_sw100', 'mes_sw1', 'mes_sw10', 'mes_sw100', 'ours', 'mc']

    print('\n=== Median F1 by (algo, grid_len) ===')
    pivot_f1 = ok.groupby(['algo', 'grid_len'])['f1'].median().unstack()
    pivot_f1 = pivot_f1.reindex(algo_idx).dropna(how='all')
    print(pivot_f1.to_string(float_format=lambda x: f'{x:.6f}'))

    print('\n=== Median Chamfer by (algo, grid_len) ===')
    pivot_c = ok.groupby(['algo', 'grid_len'])['chamfer'].median().unstack()
    pivot_c = pivot_c.reindex(algo_idx).dropna(how='all')
    print(pivot_c.to_string(float_format=lambda x: f'{x:.5f}'))

    print('\n=== Median Hausdorff by (algo, grid_len) ===')
    pivot_h = ok.groupby(['algo', 'grid_len'])['hausdorff'].median().unstack()
    pivot_h = pivot_h.reindex(algo_idx).dropna(how='all')
    print(pivot_h.to_string(float_format=lambda x: f'{x:.5f}'))

    print('\n=== Surviving count n_ok / 500 by (algo, grid_len) ===')
    pivot_n = ok.groupby(['algo', 'grid_len'])['_is_real'].sum().unstack()
    pivot_n = pivot_n.reindex(algo_idx).dropna(how='all').astype(int)
    print(pivot_n.to_string())

    # ── Mean over successful cases only (not penalised by failures) ──
    # This is the complement to median — useful for paper "Avg" rows.
    # We strictly filter rows with non-empty err (real failures), so f1=0
    # from a real failure is excluded too (not just inf for haus/cham).
    ok_success = ok[~bad].copy()
    print('\n=== Mean F1 by (algo, grid_len) — successful cases only ===')
    piv = ok_success.groupby(['algo', 'grid_len'])['f1'].mean().unstack().reindex(algo_idx).dropna(how='all')
    print(piv.to_string(float_format=lambda x: f'{x:.6f}'))
    print('\n=== Mean Chamfer by (algo, grid_len) — successful cases only ===')
    piv = ok_success.groupby(['algo', 'grid_len'])['chamfer'].mean().unstack().reindex(algo_idx).dropna(how='all')
    print(piv.to_string(float_format=lambda x: f'{x:.5g}'))
    print('\n=== Mean Hausdorff by (algo, grid_len) — successful cases only ===')
    piv = ok_success.groupby(['algo', 'grid_len'])['hausdorff'].mean().unstack().reindex(algo_idx).dropna(how='all')
    print(piv.to_string(float_format=lambda x: f'{x:.5g}'))

    # ── Boxplot percentile tables (0/25/50/75/100 = min/Q1/median/Q3/max) ──
    # Useful for paper boxplots: gives the five-number summary per (algo, gl).
    # Replace inf with NaN for percentile calc so failed rows don't dominate
    # the max — the survival count above already shows how many of those
    # there are.
    ok_finite = ok.copy()
    for col in ('hausdorff', 'chamfer', 'f1'):
        ok_finite[col] = ok_finite[col].replace([np.inf, -np.inf], np.nan)
    pcts = [0, 25, 50, 75, 100]
    for metric, fmt in [('f1', '{:.6f}'), ('chamfer', '{:.5g}'), ('hausdorff', '{:.5g}')]:
        for p in pcts:
            print(f'\n=== {metric.capitalize()} P{p:02d} by (algo, grid_len) ===')
            piv = ok_finite.groupby(['algo', 'grid_len'])[metric].quantile(p/100).unstack()
            piv = piv.reindex(algo_idx).dropna(how='all')
            print(piv.to_string(float_format=fmt.format))

    # ── Boxplot PDF (one figure per metric, all algos × gls) ──
    out_dir = Path(args.output).parent if args.output else Path('.')
    pdf_path = out_dir / 'boxplots.pdf'
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        gls = sorted(ok_finite['grid_len'].unique())
        algos_present = [a for a in algo_idx if a in ok_finite['algo'].unique()]
        with PdfPages(str(pdf_path)) as pdf:
            for metric in ('f1', 'chamfer', 'hausdorff'):
                fig, axes = plt.subplots(len(gls), 1, figsize=(10, 2.5*len(gls)),
                                         squeeze=False)
                for gi, gl in enumerate(gls):
                    ax = axes[gi, 0]
                    data = []
                    labels = []
                    for a in algos_present:
                        vals = ok_finite[(ok_finite['algo']==a) &
                                         (ok_finite['grid_len']==gl)][metric].dropna()
                        data.append(vals.values)
                        labels.append(a)
                    ax.boxplot(data, labels=labels, showfliers=False)
                    ax.set_title(f'{metric}  gl={gl}')
                    ax.set_yscale('log' if metric != 'f1' else 'linear')
                    ax.tick_params(axis='x', rotation=30)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
        print(f'\nBoxplot PDF → {pdf_path}')
    except Exception as e:
        print(f'\n(boxplot generation failed: {e})')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
