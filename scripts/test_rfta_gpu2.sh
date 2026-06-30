#!/bin/bash
set -uo pipefail
module purge 2>/dev/null || true
module load hosts/hopper gnu12 python/3.12.1-33 cuda 2>/dev/null || true
module load vulkan 2>/dev/null || true   # may not exist
echo "host: $(hostname)"
nvidia-smi -L 2>&1 | head -3
echo "--- vulkan probe ---"
which vulkaninfo 2>&1 | head; vulkaninfo --summary 2>&1 | head -25
echo "--- ls /dev/dri ---"
ls -la /dev/dri 2>&1
echo "--- A/B test ---"
cd /home/ycheng27/code/sdfgradients
/home/ycheng27/envs/sdf/bin/python - <<'PY'
import time, numpy as np, gpytoolbox as gpy
n=8000
rng=np.random.default_rng(0)
U=rng.random((n,3))
S=np.linalg.norm(U-0.5,axis=1)-0.3
for fc in (True, False):
    t=time.perf_counter()
    V,F=gpy.reach_for_the_arcs(U, S, force_cpu=fc, verbose=False, parallel=True, screening_weight=10)
    print(f'  force_cpu={fc}: {time.perf_counter()-t:.2f}s  V={len(V)} F={len(F)}')
# Also try a real cluster mesh at gl=50 (medium)
print('--- realistic mesh gl=50 ---')
import sys
sys.path.insert(0, '/home/ycheng27/code/sdfgradients')
from SDF_to_surface_3D import generate_test_mesh_data, test_rfta, Options
mp = '/scratch/ycheng27/thingi10k/extracted/25f0e335c0c10988086ffa152169449e6f5dc2328598c8806dc8eb1a0c5993f8/Thingi10K/raw_meshes/32770.stl'
import os
os.makedirs('/tmp/sdf_gpu_realtest/out/32770', exist_ok=True)
os.chdir('/tmp/sdf_gpu_realtest')
_, U2, S2, _ = generate_test_mesh_data(mp, '32770', grid_len=50, save=False)
import numpy as np
np.savez('/tmp/sdf_gpu_realtest/sdf50.npz', points=U2, sdf_values=S2)
for fc in (True, False):
    o = Options(name='32770', grid_len=50, max_iters=5, clamp=False,
                export_short_arcs=False, export_projections=False, verbose=False)
    o.path_to_obj = mp
    o.path_to_sdf = '/tmp/sdf_gpu_realtest/sdf50.npz'
    t=time.perf_counter()
    test_rfta(o, save_gtmesh=False, screening_weight=10, parallel=True, force_cpu=fc)
    print(f'  realmesh gl=50 force_cpu={fc}: {time.perf_counter()-t:.2f}s')
PY
