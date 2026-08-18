"""Experiment 1 — scattered SDF samples.

Draws the same number of samples uniformly at random inside the bounding box
instead of on a regular grid, pairing every scattered cell with a grid cell at
the same budget. Grid methods lose their grid; ours never used one.

    python additional_experiments/scattered.py
"""

import _common

MESHES = ['bunny', 'horse', 'eiffel', 'rossignol', 'fandisk',
          'chair', 'metratron', 'armadillo', '32770', 'denker']

GRID_LENS = [50]

# Marching cubes needs a regular grid to run on at all, so it only appears in
# the control. On scattered points every coordinate is unique, which would ask
# for an N^3 voxel grid from N samples.
ALGOS_SCATTERED = ['ours', 'rfta', 'mes']
ALGOS_GRID = ['ours', 'rfta', 'mes', 'mc']

cells = []
for grid_len in GRID_LENS:
    for mesh in MESHES:
        cells.append(dict(param='scattered', mesh=mesh, grid_len=grid_len,
                          algos=ALGOS_SCATTERED, scatter=True))
        cells.append(dict(param='grid', mesh=mesh, grid_len=grid_len,
                          algos=ALGOS_GRID, scatter=False))

_common.sweep('scattered', cells)
