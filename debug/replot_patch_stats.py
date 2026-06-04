"""
Regenerate the combined plot from out/patch_stats_summary.json without
rerunning the full sweep. Mirrors the plotting section of
plot_patch_stats.py — keep them in sync if you change one.
"""
import json
import numpy as np
import matplotlib.pyplot as plt

SUMMARY = 'out/patch_stats_summary.json'

data = json.load(open(SUMMARY))
data.sort(key=lambda r: r['n_constraints'])
grid_sorted  = np.array([r['grid_len']      for r in data])
nc_sorted    = np.array([r['n_constraints'] for r in data])
np_sorted    = np.array([r['n_patches']     for r in data])
sizes_sorted = [r['patch_sizes'] for r in data]

# Single axes, shared x = n_constraints (log). Boxplot on left y (linear,
# "points per patch", 16-675). n_patches line on right y (log, 3-9054).
fig, ax_box = plt.subplots(figsize=(10, 5.0))
ax_line = ax_box.twinx()

positions = nc_sorted.astype(float)
# widths ∝ position so every box has the same visual width on log x.
widths = positions * 0.15
bp = ax_box.boxplot(sizes_sorted, positions=positions, widths=widths,
                    whis=(0, 100), showfliers=False, patch_artist=True,
                    zorder=2)
for box in bp['boxes']:
    box.set(facecolor='#cfe2f3', edgecolor='C0')
for med in bp['medians']:
    med.set(color='C0', lw=1.5)
means = [np.mean(s) for s in sizes_sorted]
mean_h, = ax_box.plot(positions, means, 's--', color='C0', lw=1, ms=4,
                      alpha=0.8, label='points/patch mean', zorder=3)

# n_patches on twin (log) y axis. Slope-1 reference through largest point.
patch_h, = ax_line.plot(positions, np_sorted, 'o-', color='C3',
                        lw=1.5, ms=5, label='n_patches', zorder=4)
ref_y = np_sorted[-1] * (positions / positions[-1])
ref_h, = ax_line.plot(positions, ref_y, ':', color='C3', lw=1, alpha=0.5,
                      label='n_patches slope 1', zorder=3)

ax_box.set_xscale('log')
ax_line.set_yscale('log')
ax_box.set_xticks(positions)
ax_box.set_xticklabels([f'g={gl}\nN={nc:,}' for gl, nc in
                        zip(grid_sorted, nc_sorted)], fontsize=8)
ax_box.tick_params(axis='x', which='minor', length=0)

ax_box.set_xlabel('n_constraints (log scale)')
ax_box.set_ylabel('points per patch (boxplot, linear)', color='C0')
ax_line.set_ylabel('n_patches (log)', color='C3')
ax_box.tick_params(axis='y', colors='C0')
ax_line.tick_params(axis='y', colors='C3')

ax_box.set_title('PU patch count and per-patch size vs n_constraints (last iter fit)')
ax_box.grid(True, axis='y', alpha=0.3)

# Combined legend.
boxes_proxy = plt.Rectangle((0, 0), 1, 1, facecolor='#cfe2f3', edgecolor='C0',
                            label='points/patch (Q1/median/Q3, whiskers = min/max)')
ax_box.legend(handles=[boxes_proxy, mean_h, patch_h, ref_h],
              loc='upper left', fontsize=8)

fig.tight_layout()
fig.savefig('out/patch_stats.png', dpi=140)
print('Wrote out/patch_stats.png')
