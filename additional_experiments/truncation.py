"""Experiment 2 — truncated SDF samples.

Keeps only the samples whose |SDF| falls inside a shrinking band around the
surface, the way a TSDF from range data carries nothing far from it. The
tightest band leaves a thin shell on a unit-cube-normalized mesh.

    python additional_experiments/truncation.py
"""

import _common

MESH = 'rossignol'
GRID_LEN = 50  # 50^3 samples before truncation

# The band half-width. 1.0 is the untruncated control: generate_test_mesh_data
# only filters when bound < 1.0, and no sample of a unit-cube-normalized mesh
# is further than that from the surface anyway.
BOUNDS = [1.0, 0.1, 0.05, 0.005, 0.002]

# Marching cubes still has its grid here -- truncation removes samples from it
# rather than moving them -- but note test_mc fills every removed cell with +1,
# i.e. "unknown means outside". That is the usual TSDF convention and it is what
# makes MC hollow out the interior as the band tightens.
ALGOS = ['ours', 'rfta', 'mes', 'mc']

cells = [dict(param=f'bound{bound}', mesh=MESH, grid_len=GRID_LEN,
              algos=ALGOS, bound=bound)
         for bound in BOUNDS]

_common.sweep('truncation', cells)
