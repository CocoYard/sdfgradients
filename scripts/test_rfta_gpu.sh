#!/bin/bash
set -uo pipefail
module purge 2>/dev/null || true
module load hosts/hopper gnu12 python/3.12.1-33 cuda 2>/dev/null || true
echo "=== nvidia-smi ==="
nvidia-smi 2>&1 | head -15
echo "=== node ==="; hostname
echo "=== gpu deps ==="
/home/ycheng27/envs/sdf/bin/python - <<'PY'
import sys
print('python:', sys.executable)
try:
    import torch
    print('torch:', torch.__version__, 'cuda?', torch.cuda.is_available(),
          'devs:', torch.cuda.device_count())
except Exception as e:
    print('torch import failed:', e)
try:
    import cupy
    print('cupy:', cupy.__version__)
    print('  cupy.cuda available:', cupy.cuda.is_available())
except Exception as e:
    print('cupy import failed:', e)
import gpytoolbox
print('gpytoolbox:', gpytoolbox.__version__ if hasattr(gpytoolbox,'__version__') else gpytoolbox.__file__)
PY
echo "=== run rfta with force_cpu=False on small mesh ==="
cd /home/ycheng27/code/sdfgradients
/home/ycheng27/envs/sdf/bin/python scripts/run_baseline.py \
    --task-id 0 \
    --index /scratch/ycheng27/new_solid/index.csv \
    --out-root /tmp/sdf_smoketest_rfta_gpu \
    --algos rfta \
    --grid-lens 30 2>&1 | tail -15
