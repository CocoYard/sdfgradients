#!/bin/bash
#
# Metrics + summaries for one run directory.
#
#   sbatch scripts/submit_metrics.sbatch <RUN_DIR> [COMPARE ...]
#
#   RUN_DIR   the run to score, e.g. /scratch/ycheng27/new_solid_bfgs_post
#             (reconstructions are read from <RUN_DIR>/out/)
#   COMPARE   zero or more other runs (dir or metrics CSV) to line up
#             beside it in the summary tables
#
# Examples:
#   # three-way: this run vs bfgs vs the original
#   sbatch scripts/submit_metrics.sbatch /scratch/ycheng27/new_solid_bfgs_post \
#          /scratch/ycheng27/new_solid_bfgs /scratch/ycheng27/new_solid
#   sbatch scripts/submit_metrics.sbatch /scratch/ycheng27/new_solid   # score only
#
# Labels, algo list and output paths are all derived from RUN_DIR. The
# comparison tables show 'ours' only; everything is overridable:
#   ALGOS=ours,mc  COMPARE_ALGOS=ours,mc  INDEX=...  RUN_TOPOLOGY=1  sbatch ...
#
#SBATCH --job-name=sdf-metrics
#SBATCH --partition=normal
#SBATCH --account=ygingold
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/ycheng27/metrics_logs/metrics_%j.out
#SBATCH --error=/scratch/ycheng27/metrics_logs/metrics_%j.err

set -uo pipefail
export OMP_NUM_THREADS=2  # multiprocessing.Pool spawns workers; keep per-worker threads small
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2

REPO=/home/ycheng27/code/sdfgradients
PYTHON=/home/ycheng27/envs/sdf/bin/python
mkdir -p /scratch/ycheng27/metrics_logs

RUN_DIR=${1:-${RUN_DIR:-/scratch/ycheng27/new_solid}}
shift || true
COMPARE=${COMPARE:-$(printf '%s\n' "$@")}   # zero or more runs to line up against
COMPARE_ALGOS=${COMPARE_ALGOS:-ours}        # comparison tables show these algos only

# Short label for the summary tables: new_solid_bfgs_post -> bfgs_post
label_of() { basename "$1" | sed 's/^new_solid_*//; s/^$/base/'; }

INDEX=${INDEX:-/scratch/ycheng27/new_solid/index.csv}   # shared mesh list
GT_ROOT=${GT_ROOT:-/scratch/ycheng27/thingi10k/preview_obj}
GRIDS=${GRIDS:-6,10,20,30,40,50,60,80,100}
# Reconstructions normally live in <RUN_DIR>/out; override for side trees such
# as the tangent-point experiment's out.AB (see run_experiment_AB.py).
OUT_ROOT=${OUT_ROOT:-${RUN_DIR}/out}
METRICS_CSV=${METRICS_CSV:-${RUN_DIR}/metrics.csv}
RUN_TOPOLOGY=${RUN_TOPOLOGY:-0}

# Which algos are actually in this run? Read them off the .obj filenames
# instead of making the caller spell them out.
ALGOS=${ALGOS:-$(OUT_ROOT="${OUT_ROOT}" "${PYTHON}" - <<'PYEOF'
import os, re, glob
out = os.environ['OUT_ROOT']
algos = set()
for d in sorted(glob.glob(f'{out}/*'))[:20]:      # a few meshes is enough
    for f in glob.glob(f'{d}/*.obj'):
        m = re.fullmatch(r'(.+)_\d+\.obj', os.path.basename(f))
        if m:
            algos.add(m.group(1))
print(','.join(sorted(algos)))
PYEOF
)}

if [[ -z "${ALGOS}" ]]; then
    echo "ERROR: no <algo>_<gl>.obj files found under ${OUT_ROOT}" >&2
    exit 1
fi

cd "${REPO}"
echo "=== ${SLURM_JOB_ID:-local} on ${SLURM_JOB_NODELIST:-$(hostname)} ──"
date -Iseconds
echo "RUN_DIR : ${RUN_DIR}"
echo "ALGOS   : ${ALGOS}   (auto-detected unless overridden)"
echo "GRIDS   : ${GRIDS}"
[[ -n "${COMPARE}" ]] && echo "COMPARE : ${COMPARE}"
echo

echo "── compute_metrics.py ──"
"${PYTHON}" scripts/compute_metrics.py \
    --index "${INDEX}" --gt-root "${GT_ROOT}" --out-root "${OUT_ROOT}" \
    --algos "${ALGOS}" --grids "${GRIDS}" --output "${METRICS_CSV}" \
    --workers "${SLURM_CPUS_PER_TASK:-8}"

if [[ "${RUN_TOPOLOGY}" == "1" ]]; then
    echo
    echo "── topology_check.py ──"
    "${PYTHON}" scripts/topology_check.py \
        --index "${INDEX}" --gt-root "${GT_ROOT}" --out-root "${OUT_ROOT}" \
        --algos "${ALGOS}" --output "${RUN_DIR}/topology.csv" \
        --workers "${SLURM_CPUS_PER_TASK:-8}"
    "${PYTHON}" scripts/summarize_topology.py --input "${RUN_DIR}/topology.csv"
fi

echo
echo "── summarize_metrics.py (percentile tables + boxplots.pdf) ──"
"${PYTHON}" scripts/summarize_metrics.py \
    --csv-glob "${METRICS_CSV}" --output "${RUN_DIR}/summary_metrics.tsv"

echo
echo "── summary tables for this run -> summary_metrics.txt ──"
"${PYTHON}" scripts/summarize_AB.py --csv "${METRICS_CSV}" \
    --report "${RUN_DIR}/summary_metrics.txt"

if [[ -n "${COMPARE}" ]]; then
    MERGED="${RUN_DIR}/metrics_compare.csv"
    echo
    echo "── comparison (${COMPARE_ALGOS}): $(label_of "${RUN_DIR}") vs others ──"
    # Runs to line up: this one first, then every COMPARE entry. Each may be a
    # run dir or a metrics CSV; resolution + labelling happens in python.
    RUNS="${METRICS_CSV}
${COMPARE}"
    RUNS="${RUNS}" CUR_DIR="${RUN_DIR}" KEEP_ALGOS="${COMPARE_ALGOS}" \
    GRIDS="${GRIDS}" MERGED="${MERGED}" \
    "${PYTHON}" - <<'PYEOF'
import glob
import os
import sys

import pandas as pd

keep = [a.strip() for a in os.environ['KEEP_ALGOS'].split(',') if a.strip()]
grids = [int(g) for g in os.environ['GRIDS'].split(',') if g.strip()]
entries = [e for e in os.environ['RUNS'].split('\n') if e.strip()]


def label_of(path):
    name = os.path.basename(path.rstrip('/'))
    name = name[len('new_solid'):].lstrip('_') if name.startswith('new_solid') else name
    return name or 'base'


def load(entry, fallback_dir=None):
    """entry -> (label, rows). A dir picks whichever metrics*.csv actually
    holds the algos we want — 'newest' is not a safe rule when a run dir has
    several CSVs (metrics.csv, metrics_AB.csv, metrics_highres.csv ...)."""
    if os.path.isdir(entry):
        cands, label = sorted(glob.glob(f'{entry}/metrics*.csv')), label_of(entry)
    else:
        cands, label = [entry], label_of(fallback_dir or os.path.dirname(entry))
    for c in cands:
        try:
            d = pd.read_csv(c)
        except Exception:
            continue
        d = d[d.algo.isin(keep) & d.grid_len.isin(grids)]
        if len(d):
            return label, d.copy(), c
    return label, None, None


parts = []
for i, entry in enumerate(entries):
    label, d, src = load(entry, fallback_dir=os.environ['CUR_DIR'] if i == 0 else None)
    if d is None:
        print(f'  WARN: no rows for algos={keep} in {entry} — skipped', file=sys.stderr)
        continue
    d['algo'] = d['algo'] + '_' + label
    parts.append(d)
    print(f'  {label:<12} {len(d):>5} rows  <- {src}')

if len(parts) < 2:
    print('  need at least 2 runs with data; skipping comparison', file=sys.stderr)
    raise SystemExit(1)

merged = pd.concat(parts, ignore_index=True)
merged.to_csv(os.environ['MERGED'], index=False)
print(f'  merged {len(merged)} rows -> {os.environ["MERGED"]}')
PYEOF
    [[ -f "${MERGED}" ]] && "${PYTHON}" scripts/summarize_AB.py --csv "${MERGED}" \
        --report "${RUN_DIR}/summary_compare.txt"
fi

echo
echo "=== ALL DONE $(date -Iseconds) ==="
