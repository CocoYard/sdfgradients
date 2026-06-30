#!/bin/bash
#SBATCH --job-name=sdf-perftest
#SBATCH --partition=normal
#SBATCH --account=ygingold
#SBATCH --array=428
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/ycheng27/sdfgradients/baselines_perf_test/slurm_%A_%a.out
#SBATCH --error=/scratch/ycheng27/sdfgradients/baselines_perf_test/slurm_%A_%a.err

set -uo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mkdir -p /scratch/ycheng27/sdfgradients/baselines_perf_test

echo "=== SLURM ==="
echo "JobID       : ${SLURM_JOB_ID}"
echo "NodeList    : ${SLURM_JOB_NODELIST}"
echo "CPUs        : ${SLURM_CPUS_PER_TASK}"
echo "Mem         : ${SLURM_MEM_PER_NODE:-?} MB"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "StartTime   : $(date -Iseconds)"
echo

cd /home/ycheng27/code/sdfgradients

# Single task, gl=20 only, ours-only — fast smoke test.
# Separate out-root so we don't race with live batch on tasks/49546.
/home/ycheng27/envs/sdf/bin/python scripts/run_baseline.py \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --algos ours \
    --grid-lens 50 \
    --out-root /scratch/ycheng27/sdfgradients/baselines_perf_test
