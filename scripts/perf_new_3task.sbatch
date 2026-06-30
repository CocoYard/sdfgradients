#!/bin/bash
#SBATCH --job-name=perf-new-3
#SBATCH --partition=bigmem
#SBATCH --account=ygingold
#SBATCH --array=1-3
#SBATCH --nodelist=amd084
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=0:30:00
#SBATCH --output=/scratch/ycheng27/sdfgradients/baselines_perf_test/new_%A_%a.out
#SBATCH --error=/scratch/ycheng27/sdfgradients/baselines_perf_test/new_%A_%a.err

set -uo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

OUT_ROOT=/scratch/ycheng27/sdfgradients/baselines_perf_test/new/${SLURM_ARRAY_TASK_ID}
mkdir -p "${OUT_ROOT}"

echo "=== array=${SLURM_ARRAY_TASK_ID} on ${SLURM_JOB_NODELIST} cpus=${SLURM_CPUS_PER_TASK} ==="
date -Iseconds

cd /home/ycheng27/code/sdfgradients
# All three sibling tasks process the same mesh (49546 = task-id 428) into
# disjoint out-roots so we measure pure resource contention, not file races.
/home/ycheng27/envs/sdf/bin/python scripts/run_baseline.py \
    --task-id 428 \
    --algos ours \
    --grid-lens 20 \
    --out-root "${OUT_ROOT}"
