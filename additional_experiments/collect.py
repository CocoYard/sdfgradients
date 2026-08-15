"""Recompute metrics from the reconstructions already on disk.

Reads whatever finished, without re-running or regenerating anything, and
rewrites results/<experiment>/metrics.csv (one row per reconstruction) and
results/<experiment>/summary.csv (mean / std / median across meshes). Use it
when a sweep was run in several batches, or was interrupted.

    python additional_experiments/collect.py            # every experiment
    python additional_experiments/collect.py noise      # just one
"""

import sys

import _common

names = sys.argv[1:] or sorted(
    p.name for p in _common.RESULTS.iterdir() if p.is_dir())

for name in names:
    _common.collect(name)
