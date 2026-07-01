#!/bin/bash
# High-resolution ours+mc run on new_solid. grid_len passed via env GL.
# partition / mem / array / output are set by the orchestrator via sbatch CLI.
#SBATCH --job-name=sdf-hr
#SBATCH --account=ygingold
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --requeue

set -uo pipefail
module purge 2>/dev/null || true
module load hosts/hopper gnu12 python/3.12.1-33 2>/dev/null || true
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

cd /home/ycheng27/code/sdfgradients
exec /home/ycheng27/envs/sdf/bin/python scripts/run_baseline.py \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --index /scratch/ycheng27/new_solid/index.csv \
    --out-root /scratch/ycheng27/new_solid \
    --algos ours,mc --grid-lens "${GL}"
