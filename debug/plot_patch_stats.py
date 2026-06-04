"""
Collect (n_constraints, n_patches, patch_sizes) of the LAST fit() call for a
sweep over grid_len, then plot:
  1. line: x=n_constraints, y=n_patches
  2. boxplot: x=n_constraints, y=patch_sizes distribution

Mechanism: PUInterpolator::fit() in cpp/src/pu_interpolator.cpp appends one
JSONL record per call when env PU_FIT_LOG is set. Iterative optimization runs
fit() many times, so we take the LAST line of the file as "last iteration's
fit" for that grid_len.
"""
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt

import SDF_to_surface_3D as S
from SDF_to_surface_3D import Options, test_our_method


def run_one(grid_len, log_path, name='chair'):
    # Clear the log file so the run's last line == last fit of this grid_len.
    open(log_path, 'w').close()

    os.environ['PU_FIT_LOG'] = log_path
    # Mirror the args used in SDF_to_surface_3D.py:749-750.
    options = Options(name=name, grid_len=grid_len, max_iters=15, verbose=True,
                      use_MES=-1, export_short_arcs=False)
    test_our_method(options, save_gtmesh=False)

    # Read last JSONL line.
    with open(log_path) as f:
        lines = [ln for ln in f if ln.strip()]
    if not lines:
        return None
    return json.loads(lines[-1])


def main():
    # data_dir is referenced by Options.__init__ via SDF_to_surface_3D's globals.
    S.data_dir = 'examples'

    grid_lens = [6, 10, 20, 30, 40, 50, 60, 80, 100]
    log_path = os.path.abspath('out/pu_fit_log.jsonl')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    records = []  # (grid_len, n_constraints, n_patches, patch_sizes[])
    for gl in grid_lens:
        t0 = time.perf_counter()
        rec = run_one(gl, log_path)
        dt = time.perf_counter() - t0
        if rec is None:
            print(f"[grid_len={gl}] no fit log captured (skipped)")
            continue
        print(f"[grid_len={gl}] N={rec['n_constraints']}  P={rec['n_patches']}  "
              f"size_min/mean/max={min(rec['patch_sizes'])}/"
              f"{int(np.mean(rec['patch_sizes']))}/{max(rec['patch_sizes'])}  "
              f"({dt:.1f}s)")
        records.append((gl, rec['n_constraints'], rec['n_patches'],
                        rec['patch_sizes']))

    if not records:
        print("Nothing collected; aborting plots.")
        return

    # Save raw stats next to the plots so we never have to rerun to inspect.
    summary_path = 'out/patch_stats_summary.json'
    with open(summary_path, 'w') as f:
        json.dump([{'grid_len': gl, 'n_constraints': nc,
                    'n_patches': np_, 'patch_sizes': ps}
                   for (gl, nc, np_, ps) in records], f)
    print(f"Wrote {summary_path}")

    grid_arr = np.array([r[0] for r in records])
    n_constraints = np.array([r[1] for r in records])
    n_patches = np.array([r[2] for r in records])
    sizes_list = [r[3] for r in records]

    # Sort by n_constraints so the line is monotone in x.
    order = np.argsort(n_constraints)
    grid_sorted = grid_arr[order]
    nc_sorted = n_constraints[order]
    np_sorted = n_patches[order]
    sizes_sorted = [sizes_list[i] for i in order]

    # ── Single-axes overlay: boxplot on left y, n_patches line on twin ──
    fig, ax_box = plt.subplots(figsize=(10, 5.0))
    ax_line = ax_box.twinx()

    positions = nc_sorted.astype(float)
    # widths ∝ position → every box has the same visual width on log x.
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

    boxes_proxy = plt.Rectangle((0, 0), 1, 1, facecolor='#cfe2f3', edgecolor='C0',
                                label='points/patch (Q1/median/Q3, whiskers = min/max)')
    ax_box.legend(handles=[boxes_proxy, mean_h, patch_h, ref_h],
                  loc='upper left', fontsize=8)

    fig.tight_layout()
    fig.savefig('out/patch_stats.png', dpi=140)
    print('Wrote out/patch_stats.png')


if __name__ == '__main__':
    main()
