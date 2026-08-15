"""Experiment 3 — noisy SDF samples.

Perturb every sampled distance by Gaussian noise before reconstruction, so the
input is no longer a consistent distance field. Sigma is in the units of the
unit-cube-normalized mesh; at 50^3 the sample spacing is about 0.024, so the
sweep goes from noise well below the spacing up to noise larger than it.

    python additional_experiments/noise.py
"""

import _common

MESHES = ['bunny', 'horse', 'eiffel', 'rossignol', 'fandisk',
          'chair', 'metratron', 'armadillo', '32770', 'denker']

GRID_LEN = 50

# 0.0 is the noise-free control.
NOISE_LEVELS = [0.001, 0.005, 0.01, 0]

ALGOS_NOISY = ['ours']

cells = []
for sigma in NOISE_LEVELS:
    algos = ALGOS_NOISY
    for mesh in MESHES:
        cells.append(dict(param=f'noise{sigma}', mesh=mesh, grid_len=GRID_LEN,
                          algos=algos, noise=sigma))

_common.sweep('noise', cells)
