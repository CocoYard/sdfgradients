#!/bin/bash
#
# Companion to submit_metrics.sbatch — but only re-computes the 'ours' rows
# (for the use_MES=-1 batch), then merges them into a copy of the existing
# metrics.csv so rfta/mes/mc rows from the previous full sweep are preserved
# verbatim and ours rows are replaced with the noMES variant.
#
# Submit chained after the ours batch:
#   sbatch --dependency=afterany:<JOBID> scripts/submit_metrics_useMES_neg1.sbatch
#
#SBATCH --job-name=sdf-metrics-noMES
#SBATCH --partition=normal
#SBATCH --account=ygingold
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/ycheng27/new_solid_useMES_neg1/metrics_%j.out
#SBATCH --error=/scratch/ycheng27/new_solid_useMES_neg1/metrics_%j.err

set -uo pipefail
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2

REPO=/home/ycheng27/code/sdfgradients
PYTHON=/home/ycheng27/envs/sdf/bin/python

INDEX=/scratch/ycheng27/new_solid/index.csv
GT_ROOT=/scratch/ycheng27/thingi10k/preview_obj
NEW_OUT_ROOT=/scratch/ycheng27/new_solid_useMES_neg1/out
EXISTING_METRICS=/scratch/ycheng27/new_solid/metrics.csv
EXISTING_TOPOLOGY=/scratch/ycheng27/new_solid/topology.csv

OUT_DIR=/scratch/ycheng27/new_solid_useMES_neg1
NEW_OURS_METRICS="${OUT_DIR}/metrics_ours_only.csv"
NEW_OURS_TOPOLOGY="${OUT_DIR}/topology_ours_only.csv"
COMBINED_METRICS="${OUT_DIR}/metrics.csv"
COMBINED_TOPOLOGY="${OUT_DIR}/topology.csv"

mkdir -p "${OUT_DIR}"
cd "${REPO}"

echo "=== ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST} cpus=${SLURM_CPUS_PER_TASK} ==="
date -Iseconds
echo

# ── 1/4 compute_metrics — ours only, new out_root ─────────────────────────
echo "── 1/4 compute_metrics.py (ours only, use_MES=-1) ──"
"${PYTHON}" scripts/compute_metrics.py \
    --index "${INDEX}" \
    --gt-root "${GT_ROOT}" \
    --out-root "${NEW_OUT_ROOT}" \
    --algos ours \
    --output "${NEW_OURS_METRICS}" \
    --workers ${SLURM_CPUS_PER_TASK}
echo

# ── 2/4 topology_check — ours only ────────────────────────────────────────
echo "── 2/4 topology_check.py (ours only) ──"
"${PYTHON}" scripts/topology_check.py \
    --index "${INDEX}" \
    --gt-root "${GT_ROOT}" \
    --out-root "${NEW_OUT_ROOT}" \
    --algos ours \
    --output "${NEW_OURS_TOPOLOGY}" \
    --workers ${SLURM_CPUS_PER_TASK}
echo

# ── 3/4 merge: keep existing non-ours rows, replace ours rows with new ───
# Safety: writes to a NEW path (combined_metrics.csv) — never overwrites the
# original new_solid/metrics.csv. If you want to promote it, copy manually.
echo "── 3/4 merge ours rows into combined CSVs ──"
"${PYTHON}" - <<PYEOF
import pandas as pd
from pathlib import Path

def merge(existing_path: str, new_path: str, out_path: str, label: str):
    existing = pd.read_csv(existing_path)
    new = pd.read_csv(new_path)
    n_before = (existing['algo'] == 'ours').sum()
    n_new = (new['algo'] == 'ours').sum()
    other = existing[existing['algo'] != 'ours']
    new_ours = new[new['algo'] == 'ours']
    merged = pd.concat([other, new_ours], ignore_index=True)
    # Stable sort so diffs against the old CSV are clean
    sort_cols = [c for c in ('file_id', 'algo', 'grid_len') if c in merged.columns]
    merged = merged.sort_values(sort_cols).reset_index(drop=True)
    merged.to_csv(out_path, index=False)
    print(f'  {label}: existing ours rows={n_before}  new ours rows={n_new}  '
          f'merged total rows={len(merged)} -> {out_path}')

merge('${EXISTING_METRICS}',  '${NEW_OURS_METRICS}',  '${COMBINED_METRICS}',  'metrics')
merge('${EXISTING_TOPOLOGY}', '${NEW_OURS_TOPOLOGY}', '${COMBINED_TOPOLOGY}', 'topology')
PYEOF
echo

# ── 4/4 summaries on the merged CSVs ─────────────────────────────────────
echo "── 4/4 summarize_metrics + summarize_topology on merged ──"
"${PYTHON}" scripts/summarize_metrics.py \
    --csv-glob "${COMBINED_METRICS}" \
    --output "${OUT_DIR}/summary_metrics.tsv"
echo
"${PYTHON}" scripts/summarize_topology.py \
    --input "${COMBINED_TOPOLOGY}"

echo
echo "=== ALL DONE $(date -Iseconds) ==="
echo "Outputs:"
echo "  merged metrics : ${COMBINED_METRICS}"
echo "  merged topology: ${COMBINED_TOPOLOGY}"
echo "  summary tsv    : ${OUT_DIR}/summary_metrics.tsv"
echo "  boxplots pdf   : ${OUT_DIR}/boxplots.pdf"
