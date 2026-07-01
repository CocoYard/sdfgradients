"""Plot ours & mc mean error vs grid_len from a metrics CSV (temporary, for eyeballing)."""
import argparse
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument('--csv', default='/scratch/ycheng27/new_solid/metrics_highres.csv')
p.add_argument('--out', default='/home/ycheng27/code/sdfgradients/tmp_ours_mc_curves.png')
args = p.parse_args()

df = pd.read_csv(args.csv)
for c in ('hausdorff', 'chamfer', 'f1'):
    df[c] = pd.to_numeric(df[c], errors='coerce')
df = df[df['err'].fillna('').astype(str).str.len() == 0]
df = df[df.algo.isin(['ours', 'mc'])]

col = {'ours': 'tab:blue', 'mc': 'tab:gray'}
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, metric, title in zip(axes, ['chamfer', 'hausdorff', 'f1'],
                             ['Mean Chamfer', 'Mean Hausdorff', 'Mean F1']):
    piv = df.groupby(['algo', 'grid_len'])[metric].mean().unstack()
    for a in ['ours', 'mc']:
        if a in piv.index:
            s = piv.loc[a].dropna()
            ax.plot(s.index, s.values, '-o', color=col[a], label=a, lw=1.8, ms=6)
    ax.set_xlabel('grid_len'); ax.set_title(title)
    if metric != 'f1':
        ax.set_yscale('log')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
fig.suptitle('ours vs mc — mean over 50 models (new_solid)', y=1.02)
fig.tight_layout()
fig.savefig(args.out, dpi=110, bbox_inches='tight')
print(f'saved -> {args.out}')
