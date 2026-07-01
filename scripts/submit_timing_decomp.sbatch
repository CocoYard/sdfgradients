#!/bin/bash
#
# Runtime-decomposition timing: rerun OURS on the first 10 new_solid meshes
# (index rows 0-9, == timing_4algo tasks 0-9) across the same 9 grid_lens,
# EXCLUSIVE on a hop (Xeon Gold 6240R, 48c, HT off) node so per-step wall
# times are clean and directly comparable to the earlier timing_4algo run.
#
# Submit:  sbatch scripts/submit_timing_decomp.sbatch
#
#SBATCH --job-name=sdf-tdecomp
#SBATCH --partition=normal
#SBATCH --account=ygingold
#SBATCH --constraint=intel
#SBATCH --array=0-9
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/ycheng27/timing_decomp/logs/slurm_%A_%a.out
#SBATCH --error=/scratch/ycheng27/timing_decomp/logs/slurm_%A_%a.err
#SBATCH --requeue

set -uo pipefail
module purge 2>/dev/null || true
module load hosts/hopper gnu12 python/3.12.1-33 2>/dev/null || true

NCPU=48
export OMP_NUM_THREADS=${NCPU}
export OPENBLAS_NUM_THREADS=${NCPU}
export MKL_NUM_THREADS=${NCPU}

REPO=/home/ycheng27/code/sdfgradients
PYTHON=/home/ycheng27/envs/sdf/bin/python
INDEX=/scratch/ycheng27/new_solid/index.csv
OUT_ROOT=/scratch/ycheng27/timing_decomp

mkdir -p "${OUT_ROOT}/logs"

echo "=== SLURM ==="
echo "ArrayTaskID : ${SLURM_ARRAY_TASK_ID}"
echo "NodeList    : ${SLURM_JOB_NODELIST}"
echo "CPUs        : ${NCPU} (exclusive)"
echo "StartTime   : $(date -Iseconds)"
echo

cd "${REPO}"
exec "${PYTHON}" scripts/run_baseline.py \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --index "${INDEX}" \
    --out-root "${OUT_ROOT}" \
    --algos ours \
    --grid-lens 6,10,20,30,40,50,60,80,100
