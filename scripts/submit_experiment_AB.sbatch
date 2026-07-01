#!/bin/bash
#
# SLURM array job: A×B tangent/reconstruction experiment on the first 50
# new_solid (solid300) meshes (index rows 0-49), grid_lens 10,20,40,60,80.
#
# Submit:
#   sbatch scripts/submit_experiment_AB.sbatch
# Pilot one mesh:
#   sbatch --array=0 scripts/submit_experiment_AB.sbatch
# Resubmit failed indices (idempotent):
#   sbatch --array=3,17 scripts/submit_experiment_AB.sbatch
#
#SBATCH --job-name=sdf-expAB
#SBATCH --partition=normal
#SBATCH --account=ygingold
#SBATCH --array=0-49
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/ycheng27/new_solid/logs/expAB_%A_%a.out
#SBATCH --error=/scratch/ycheng27/new_solid/logs/expAB_%A_%a.err
#SBATCH --requeue

set -uo pipefail

# Match the interactive environment (sbatch does NOT source ~/.bashrc).
# gnu12 is required: gnu9 libgomp/libstdc++ causes a 10-100x memory blowup
# (cgroup OOM) on some meshes.
module purge 2>/dev/null || true
module load hosts/hopper gnu12 python/3.12.1-33 2>/dev/null || true

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

REPO=/home/ycheng27/code/sdfgradients
PYTHON=/home/ycheng27/envs/sdf/bin/python

mkdir -p /scratch/ycheng27/new_solid/logs

echo "=== SLURM ==="
echo "ArrayTaskID : ${SLURM_ARRAY_TASK_ID}"
echo "NodeList    : ${SLURM_JOB_NODELIST}"
echo "StartTime   : $(date -Iseconds)"
echo

cd "${REPO}"

GRID_LENS_ARG=()
if [[ -n "${GRID_LENS:-}" ]]; then
    GRID_LENS_ARG=(--grid-lens "${GRID_LENS}")
fi
COMBOS_ARG=()
if [[ -n "${COMBOS:-}" ]]; then
    COMBOS_ARG=(--combos "${COMBOS}")
fi

exec "${PYTHON}" scripts/run_experiment_AB.py --task-id "${SLURM_ARRAY_TASK_ID}" \
    "${GRID_LENS_ARG[@]}" \
    "${COMBOS_ARG[@]}"
