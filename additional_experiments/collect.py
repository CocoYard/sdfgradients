"""Recompute metrics from the reconstructions already on disk.

Rewrites each experiment's CSVs from whatever finished, re-running and
regenerating nothing. Use it after a sweep that ran in several batches or was
interrupted.

    python additional_experiments/collect.py            # every experiment
    python additional_experiments/collect.py noise      # just one
"""

import sys

import _common

names = sys.argv[1:] or sorted(
    p.name for p in _common.RESULTS.iterdir() if p.is_dir())

for name in names:
    if name in ('convergence', 'degen_tol'):
        # Not driven by sweep(): each has a layout and a CSV shape of its own
        # (convergence shares one directory, degen_tol also reports candidate
        # quality), so the recompute lives in its own script.
        module = __import__(name)
        module.report()
        continue
    _common.collect(name)
