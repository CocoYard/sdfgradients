#!/bin/bash
# Non-exclusive variant of the timing-decomposition run: same intel (Xeon Gold
# 6240R) node + 48 threads, but WITHOUT --exclusive so it schedules without
# waiting for a fully-drained node. Wall-times carry mild co-tenancy noise; the
# per-step % breakdown is essentially unaffected.
#SBATCH --job-name=sdf-tdecomp
#SBATCH --partition=normal
#SBATCH --account=ygingold
#SBATCH --constraint=intel
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/ycheng27/timing_decomp/logs/slurm_%A_%a.out
#SBATCH --error=/scratch/ycheng27/timing_decomp/logs/slurm_%A_%a.err
#SBATCH --requeue

set -uo pipefail
module purge 2>/dev/null || true
module load hosts/hopper gnu12 python/3.12.1-33 2>/dev/null || true

NCPU=${SLURM_CPUS_PER_TASK:-48}   # follows --cpus-per-task so it can fit a partially-used node
export OMP_NUM_THREADS=${NCPU}
export OPENBLAS_NUM_THREADS=${NCPU}
export MKL_NUM_THREADS=${NCPU}

REPO=/home/ycheng27/code/sdfgradients
PYTHON=/home/ycheng27/envs/sdf/bin/python
INDEX=/scratch/ycheng27/new_solid/index.csv
OUT_ROOT=/scratch/ycheng27/timing_decomp

mkdir -p "${OUT_ROOT}/logs"
echo "NodeList: ${SLURM_JOB_NODELIST}  Start: $(date -Iseconds)"
cd "${REPO}"
exec "${PYTHON}" scripts/run_baseline.py \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --index "${INDEX}" \
    --out-root "${OUT_ROOT}" \
    --algos ours \
    --grid-lens 6,10,20,30,40,50,60,80,100
