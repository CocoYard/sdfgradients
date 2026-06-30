#!/bin/bash
#
# Paper-grade timing benchmark: 4 default-config algorithms × 9 grid_lens
# on 40 representative meshes from the new_solid index, each task running
# EXCLUSIVE on a hop (Intel Xeon 48c) node so co-tenancy noise can't bias
# wall-times.
#
# Submit:
#   sbatch scripts/submit_timing_4algo.sbatch
#
# Resubmit a single failed task:
#   sbatch --array=17 scripts/submit_timing_4algo.sbatch
#
#SBATCH --job-name=sdf-timing
#SBATCH --partition=normal
#SBATCH --account=ygingold
#SBATCH --constraint=intel
#SBATCH --array=0-39%16
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/ycheng27/timing_4algo/logs/slurm_%A_%a.out
#SBATCH --error=/scratch/ycheng27/timing_4algo/logs/slurm_%A_%a.err
#SBATCH --requeue

set -uo pipefail

module purge 2>/dev/null || true
module load hosts/hopper gnu12 python/3.12.1-33 2>/dev/null || true

# --exclusive gives us the whole 48-core hop node. Pin all thread pools to
# 48 so rfta pthreads / scipy-OpenBLAS / MKL agree on a consistent core
# count — otherwise oversubscription introduces 2-3x wall-time noise.
NCPU=48
export OMP_NUM_THREADS=${NCPU}
export OPENBLAS_NUM_THREADS=${NCPU}
export MKL_NUM_THREADS=${NCPU}

REPO=/home/ycheng27/code/sdfgradients
PYTHON=/home/ycheng27/envs/sdf/bin/python
INDEX=/scratch/ycheng27/new_solid/index.csv
OUT_ROOT=/scratch/ycheng27/timing_4algo

mkdir -p "${OUT_ROOT}/logs"

echo "=== SLURM ==="
echo "JobID       : ${SLURM_JOB_ID}"
echo "ArrayTaskID : ${SLURM_ARRAY_TASK_ID}"
echo "NodeList    : ${SLURM_JOB_NODELIST}"
echo "CPUs        : ${NCPU} (exclusive)"
echo "StartTime   : $(date -Iseconds)"
echo

cd "${REPO}"

# 4 algos in this exact order: rfta -> mes -> mc -> ours.
# mc must run before ours so that ours' side-product mc rename doesn't
# overwrite the mc file that already has a real timing row attached.
exec "${PYTHON}" scripts/run_baseline.py \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --index "${INDEX}" \
    --out-root "${OUT_ROOT}" \
    --algos rfta,mes,mc,ours \
    --grid-lens 6,10,20,30,40,50,60,80,100
