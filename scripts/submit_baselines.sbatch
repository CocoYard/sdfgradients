#!/bin/bash
#
# SLURM array job: run RFTA + MES baselines on 500 Thingi10K meshes.
#
# Submit with:
#   sbatch scripts/submit_baselines.sbatch
#
# Pilot a subset (e.g. first 5 meshes) before the full run:
#   sbatch --array=0-4 scripts/submit_baselines.sbatch
#
# Resubmit only failed tasks (idempotent thanks to run_baseline.py's
# existence check):
#   sbatch --array=17,42,99 scripts/submit_baselines.sbatch
#
#SBATCH --job-name=sdf-baseline
#SBATCH --partition=normal
#SBATCH --account=ygingold
#SBATCH --array=0-299
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
# Per-task runtime is ~14 min mean / ~29 min max on the solid-300 set, so 2h
# is 4x headroom. Keep it tight: the backfill scheduler can only slot a job
# into a gap it fits in, and a 24h request never fits.
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/ycheng27/sdfgradients/baselines/logs/slurm_%A_%a.out
#SBATCH --error=/scratch/ycheng27/sdfgradients/baselines/logs/slurm_%A_%a.err
#SBATCH --requeue

set -uo pipefail

# Match the interactive environment (sbatch does NOT source ~/.bashrc).
# Without these, the batch task gets gnu9 libgomp/libstdc++ instead of gnu12,
# which on at least one mesh (49546 gl=40) causes a 10-100x memory blowup
# (cgroup OOM at 64-256 GB while login-node baseline is 8 GB).
module purge 2>/dev/null || true
module load hosts/hopper gnu12 python/3.12.1-33 2>/dev/null || true

# gpytoolbox's RFTA uses pthreads (std::thread::hardware_concurrency), but
# scipy-openblas uses OMP_NUM_THREADS. Cap both to what SLURM allocated
# so we don't oversubscribe.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

REPO=/home/ycheng27/code/sdfgradients
PYTHON=/home/ycheng27/envs/sdf/bin/python

# Ensure log dir exists (SLURM won't create it)
mkdir -p /scratch/ycheng27/sdfgradients/baselines/logs

echo "=== SLURM ==="
echo "JobID       : ${SLURM_JOB_ID}"
echo "ArrayJobID  : ${SLURM_ARRAY_JOB_ID}"
echo "ArrayTaskID : ${SLURM_ARRAY_TASK_ID}"
echo "NodeList    : ${SLURM_JOB_NODELIST}"
echo "CPUs        : ${SLURM_CPUS_PER_TASK}"
echo "Mem         : ${SLURM_MEM_PER_NODE:-?} MB"
echo "TimeLimit   : ${SLURM_TIME_LIMIT_MIN:-?} min"
echo "StartTime   : $(date -Iseconds)"
echo

cd "${REPO}"
# RUN_ALGOS lets you submit a single-algo run via the env, e.g.:
#   sbatch --export=ALL,RUN_ALGOS=ours scripts/submit_baselines.sbatch
# Default keeps all three algorithms (existing rfta/mes outputs are
# idempotently skipped, so re-running with the default after baselines
# are done only adds 'ours' work).
RUN_ALGOS_ARG=()
if [[ -n "${RUN_ALGOS:-}" ]]; then
    RUN_ALGOS_ARG=(--algos "${RUN_ALGOS}")
    echo "RunAlgos    : ${RUN_ALGOS}"
fi
GRID_LENS_ARG=()
if [[ -n "${GRID_LENS:-}" ]]; then
    GRID_LENS_ARG=(--grid-lens "${GRID_LENS}")
    echo "GridLens    : ${GRID_LENS}"
fi
INDEX_ARG=()
if [[ -n "${INDEX:-}" ]]; then
    INDEX_ARG=(--index "${INDEX}")
    echo "Index       : ${INDEX}"
fi
OUT_ROOT_ARG=()
if [[ -n "${OUT_ROOT:-}" ]]; then
    OUT_ROOT_ARG=(--out-root "${OUT_ROOT}")
    echo "OutRoot     : ${OUT_ROOT}"
fi
SCREENING_ARG=()
if [[ -n "${SCREENING_WEIGHT:-}" ]]; then
    SCREENING_ARG=(--screening-weight "${SCREENING_WEIGHT}")
    echo "ScreenWt    : ${SCREENING_WEIGHT}"
fi
RFTA_CPU_ARG=()
if [[ -n "${RFTA_FORCE_CPU:-}" ]]; then
    RFTA_CPU_ARG=(--rfta-force-cpu)
    echo "RftaForceCPU: yes"
fi
OURS_USE_MES_ARG=()
if [[ -n "${OURS_USE_MES:-}" ]]; then
    OURS_USE_MES_ARG=(--ours-use-mes "${OURS_USE_MES}")
    echo "OursUseMES  : ${OURS_USE_MES}"
fi
# ours ablation cell: clamp / gradient optimizer / Lipschitz post-fix.
#   sbatch --export=ALL,OURS_CLAMP=0,OURS_OPTIMIZER=ascent ...
OURS_CLAMP_ARG=()
if [[ -n "${OURS_CLAMP:-}" ]]; then
    OURS_CLAMP_ARG=(--ours-clamp "${OURS_CLAMP}")
    echo "OursClamp   : ${OURS_CLAMP}"
fi
OURS_OPT_ARG=()
if [[ -n "${OURS_OPTIMIZER:-}" ]]; then
    OURS_OPT_ARG=(--ours-optimizer "${OURS_OPTIMIZER}")
    echo "OursOptim   : ${OURS_OPTIMIZER}"
fi
OURS_POST_ARG=()
if [[ -n "${OURS_POST:-}" ]]; then
    OURS_POST_ARG=(--ours-post "${OURS_POST}")
    echo "OursPost    : ${OURS_POST}"
fi

# Hard time limit: SLURM kills cleanly when time is up (no trap, no retry,
# no resubmit). If a task didn't finish, just resubmit that array index
# manually after the run.
exec "${PYTHON}" scripts/run_baseline.py --task-id "${SLURM_ARRAY_TASK_ID}" \
    "${INDEX_ARG[@]}" \
    "${OUT_ROOT_ARG[@]}" \
    "${RUN_ALGOS_ARG[@]}" \
    "${GRID_LENS_ARG[@]}" \
    "${SCREENING_ARG[@]}" \
    "${RFTA_CPU_ARG[@]}" \
    "${OURS_USE_MES_ARG[@]}" \
    "${OURS_CLAMP_ARG[@]}" \
    "${OURS_OPT_ARG[@]}" \
    "${OURS_POST_ARG[@]}"
