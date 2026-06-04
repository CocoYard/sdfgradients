"""
Boxplot of per-sphere "max arcs per cap" distribution at varying grid
resolution (eiffel mesh).

For each sphere we record max_i arc_count[cap_i] over its surviving caps,
then plot the distribution of that scalar across all spheres, one box per gl.

Data comes from `[MAX_ARC_PER_CAP_HIST]` lines emitted by `compute_exposed_batch`
in `cpp/src/existing_modules_adapter.cpp`, as compact `value:count` pairs.

Re-run with `DEBUG_ARC_HIST` defined (see the gated `#define` near the top
of that file) to regenerate these histograms.

Captured AFTER the `intersect_intervals` wrap-seam dedup fix.  Before the
fix, an iv degenerate point landing on (or in the tiny FP gap between) the
two pieces of a wrap-around `con` produced one fake-hit + one legit push
at near-identical locations, doubling `niv` per step until it saturated
`MAX_INTERVALS = 2048`; the eiffel mesh's pre-fix tail at ~2034 was that
saturation, *not* legitimate geometry (the old `2^n_active`
interpretation was a wrong guess).  Post-fix the theoretical bound
arcs/cap ≤ MAX_CAPS = 512 is respected with plenty of headroom — the
observed max for gl=20 is only 13.
"""
import numpy as np
import matplotlib.pyplot as plt


# {gl: "value:count value:count ..."} — verbatim from the trailing
# `pairs(value:count): ...` section of each `[MAX_ARC_PER_CAP_HIST]` line.
# Captured on eiffel, post wrap-seam dedup fix. observed_max stays in the
# single-digit / low-double-digit range across all gls; in particular the
# old pre-fix tail at ~2034 is gone (used to be doubling artifacts).
RAW = {
    6:   "1:18 2:76 3:106 4:12",
    10:  "1:181 2:599 3:188",
    20:  "1:3404 2:3582 3:554 4:149 5:19 6:11 7:8 8:2 9:3 10:2 11:3 12:1 13:2",
    30:  "1:13168 2:10965 3:1482 4:451 5:121 6:81 7:59 8:25 9:10 10:5 11:4 12:2 13:2 14:1 16:2 17:1",
    40:  "0:8 1:32441 2:25255 3:3603 4:850 5:383 6:255 7:162 8:84 9:72 10:27 11:11 12:8 13:5 14:3 15:9",
    50:  "0:29 1:65887 2:47230 3:6466 4:1728 5:716 6:491 7:292 8:277 9:179 10:58 11:51 12:11 13:3",
    60:  "0:253 1:115996 2:77581 3:11102 4:2625 5:1158 6:773 7:575 8:388 9:269 10:253 11:137 12:74 13:43 14:32 15:16 16:15 17:6 18:1 19:2",
    80:  "0:135 1:284992 2:178925 3:25231 4:5805 5:2052 6:1661 7:1186 8:960 9:674 10:372 11:221 12:90 13:40 14:30 15:21 16:8 17:5 18:1 19:1",
    100: "0:92 1:569554 2:343043 3:44103 4:10435 5:4167 6:3126 7:2362 8:1734 9:1175 10:586 11:424 12:285 13:89 14:60 15:38 16:21 17:7 18:8 19:4 20:4 22:1",
}


def parse(s):
    """'1:18 2:74 3:108' -> np.array([1]*18 + [2]*74 + [3]*108) (samples).

    Drops 0 entries: a sphere with max-arcs-per-cap == 0 has no surviving cap
    (fully covered by a neighbor or no neighbors at all), so it never enters
    the per-cap arc bookkeeping we care about.
    """
    vals, cnts = [], []
    for tok in s.split():
        v, c = tok.split(':')
        v, c = int(v), int(c)
        if v == 0:
            continue
        vals.append(v)
        cnts.append(c)
    return np.repeat(vals, cnts) if vals else np.array([], dtype=int)


def main():
    gls = sorted(RAW.keys())
    samples = [parse(RAW[gl]) for gl in gls]
    # Drop any gls whose RAW string is empty (placeholder).
    keep = [(gl, s) for gl, s in zip(gls, samples) if s.size > 0]
    if not keep:
        print("No data in RAW yet — fill in `value:count` strings and re-run.")
        return
    gls, samples = [k[0] for k in keep], [k[1] for k in keep]

    # Custom 5-number summary boxplot: whiskers go to actual min/max
    # (whis=(0,100) → no fliers, no IQR-based clipping). Median drawn as a line.
    # p99 is overlaid as a tick mark on top of each box.
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(samples,
                    positions=range(len(gls)),
                    widths=0.6,
                    whis=(0, 100),
                    showfliers=False,
                    patch_artist=True)
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(gls), 1)))
    for patch, color in zip(bp['boxes'], cmap):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for med in bp['medians']:
        med.set_color('black')
        med.set_linewidth(1.8)

    # Overlay p99 as a short horizontal red tick across each box.
    p99s = [np.percentile(s, 99) for s in samples]
    for x, p in enumerate(p99s):
        ax.hlines(p, x - 0.3, x + 0.3, colors='red', linewidth=1.6,
                  zorder=5, label='p99' if x == 0 else None)

    # Annotate observed maximum above each box.
    obs_max = [int(s.max()) for s in samples]
    ymax = max(obs_max)
    for x, m in enumerate(obs_max):
        ax.text(x, m + 0.5, str(m),
                ha='center', va='bottom', fontsize=8, color='black')

    ax.set_xticks(range(len(gls)))
    ax.set_xticklabels([str(g) for g in gls])
    ax.set_xlabel('grid resolution (gl)')
    ax.set_ylabel('max arcs per cap (per host sphere)')
    ax.set_yscale('linear')
    ax.set_ylim(0, ymax * 1.2 + 1)
    ax.grid(True, axis='y', which='both', alpha=0.3)
    ax.set_title('eiffel — max arcs/cap per sphere (post wrap-seam dedup fix)')
    ax.legend(loc='upper right', fontsize=9)

    fig.tight_layout()
    out = 'out/max_arcs_per_cap.png'
    fig.savefig(out, dpi=150)
    print(f'wrote {out}')

    print('\n  gl |  min | median |  q3 |  p99 |  max')
    for gl, s in zip(gls, samples):
        print(f' {gl:3d} | {int(s.min()):>4d} | {int(np.median(s)):>6d} '
              f'| {int(np.percentile(s, 75)):>3d} '
              f'| {int(np.percentile(s, 99)):>4d} | {int(s.max()):>4d}')


if __name__ == '__main__':
    import os
    os.makedirs('out', exist_ok=True)
    main()
