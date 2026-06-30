#!/bin/bash
#SBATCH --job-name=perf-solo
#SBATCH --partition=bigmem
#SBATCH --account=ygingold
#SBATCH --nodelist=amd084
#SBATCH --nodes=1
#SBATCH --time=0:10:00
#SBATCH --output=/scratch/ycheng27/sdfgradients/baselines_perf_test/solo_%A.out
#SBATCH --error=/scratch/ycheng27/sdfgradients/baselines_perf_test/solo_%A.err

# Pass --cpus-per-task and --mem on CLI when submitting:
#   sbatch --cpus-per-task=16 --mem=128G --job-name=solo-new perf_baseline_solo.sbatch
#   sbatch --cpus-per-task=4  --mem=96G  --job-name=solo-old perf_baseline_solo.sbatch

set -uo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

OUT_ROOT=/scratch/ycheng27/sdfgradients/baselines_perf_test/solo/${SLURM_JOB_ID}
mkdir -p "${OUT_ROOT}"

echo "=== solo on ${SLURM_JOB_NODELIST} cpus=${SLURM_CPUS_PER_TASK} mem=${SLURM_MEM_PER_NODE}MB ==="
date -Iseconds

cd /home/ycheng27/code/sdfgradients
/home/ycheng27/envs/sdf/bin/python scripts/run_baseline.py \
    --task-id 428 \
    --algos ours \
    --grid-lens 20 \
    --out-root "${OUT_ROOT}"
