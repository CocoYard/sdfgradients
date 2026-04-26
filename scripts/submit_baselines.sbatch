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
#SBATCH --array=0-499%200
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/ycheng27/sdfgradients/baselines/logs/slurm_%A_%a.out
#SBATCH --error=/scratch/ycheng27/sdfgradients/baselines/logs/slurm_%A_%a.err
#SBATCH --requeue
#SBATCH --signal=B:TERM@180

set -uo pipefail

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

# ── Auto-retry logic ─────────────────────────────────────────────────
# On timeout, SLURM sends SIGTERM to this bash script 180 s before the
# hard kill (configured via #SBATCH --signal=B:TERM@180). We catch the
# signal, resubmit just this task_id with ~2x time budget, and exit
# cleanly. RETRY_COUNT caps the depth so a truly stuck task can't spin
# forever — bail after 2 retries.
RETRY_COUNT=${RETRY_COUNT:-0}
MAX_RETRIES=2
echo "RetryCount  : ${RETRY_COUNT} / ${MAX_RETRIES}"

on_term() {
    echo "[sbatch] SIGTERM caught at $(date -Iseconds) — resubmitting task ${SLURM_ARRAY_TASK_ID}"
    # Kill the python worker so it releases files before the new task
    # starts on (potentially) another node.
    if [[ -n "${PY_PID:-}" ]]; then
        kill -TERM "${PY_PID}" 2>/dev/null || true
        wait "${PY_PID}" 2>/dev/null || true
    fi
    if (( RETRY_COUNT < MAX_RETRIES )); then
        NEXT=$((RETRY_COUNT + 1))
        # Double the time budget each retry, cap at 24 h
        NEW_HOURS=$(( (${SLURM_TIME_LIMIT_MIN:-360} * 2) / 60 ))
        (( NEW_HOURS > 24 )) && NEW_HOURS=24
        NEW_TIME=$(printf "%02d:00:00" "${NEW_HOURS}")
        echo "[sbatch] submitting retry #${NEXT} with --time=${NEW_TIME}"
        sbatch \
            --array="${SLURM_ARRAY_TASK_ID}" \
            --time="${NEW_TIME}" \
            --mem="${SLURM_MEM_PER_NODE:-96000}M" \
            --export=ALL,RETRY_COUNT="${NEXT}" \
            "${REPO}/scripts/submit_baselines.sbatch" \
            || echo "[sbatch] retry submit failed!"
    else
        echo "[sbatch] hit MAX_RETRIES=${MAX_RETRIES}, giving up on task ${SLURM_ARRAY_TASK_ID}"
    fi
    exit 0
}
trap on_term TERM

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

# Run python as a background child so bash keeps a live signal handler.
# 'wait' is interruptible; on SIGTERM the trap fires and does the resubmit.
"${PYTHON}" scripts/run_baseline.py --task-id "${SLURM_ARRAY_TASK_ID}" \
    "${RUN_ALGOS_ARG[@]}" &
PY_PID=$!
wait "${PY_PID}"
RC=$?
exit ${RC}
