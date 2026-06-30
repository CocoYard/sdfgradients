#!/bin/bash
#
# Run metrics + topology + summaries on the solid-300 batch outputs.
# Submit AFTER the main array job:
#   sbatch --dependency=afterany:<JOBID> scripts/submit_metrics.sbatch
#
#SBATCH --job-name=sdf-metrics
#SBATCH --partition=normal
#SBATCH --account=ygingold
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/ycheng27/new_solid/metrics_%j.out
#SBATCH --error=/scratch/ycheng27/new_solid/metrics_%j.err

set -uo pipefail
export OMP_NUM_THREADS=2  # multiprocessing.Pool will spawn workers; keep per-worker threads small
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2

REPO=/home/ycheng27/code/sdfgradients
PYTHON=/home/ycheng27/envs/sdf/bin/python

INDEX=/scratch/ycheng27/new_solid/index.csv
OUT_ROOT=/scratch/ycheng27/new_solid/out
GT_ROOT=/scratch/ycheng27/thingi10k/preview_obj

cd "${REPO}"
echo "=== ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST} cpus=${SLURM_CPUS_PER_TASK} ──"
date -Iseconds
echo

echo "── 0/5 collect rfta/mes from 3 sources (sw=1, sw=10, sw=100) into out/ ──"
# Pre-step: gather all screening_weight variants under out/{fid}/ with
# explicit suffix names so compute_metrics can iterate them as separate
# algos:
#   - out/{fid}/{rfta,mes}_X.obj                                 (in-place rename) → {algo}_sw1_X.obj
#   - screen10/{fid}/{rfta,mes}_X.obj                           (mv + rename)     → {algo}_sw10_X.obj
#   - /scratch/ycheng27/new_solid_sw100/out/{fid}/{rfta,mes}_X.obj (mv + rename) → {algo}_sw100_X.obj
"${PYTHON}" - <<'PYEOF'
import csv, os, glob, shutil
INDEX = '/scratch/ycheng27/new_solid/index.csv'
OUT = '/scratch/ycheng27/new_solid/out'
SOURCES = [
    ('sw1',   None,                                       'in-place'),
    ('sw10',  '/scratch/ycheng27/new_solid/screen10',     'move'),
    ('sw100', '/scratch/ycheng27/new_solid_sw100/out',    'move'),
]

with open(INDEX) as f:
    fids = [r['file_id'] for r in csv.DictReader(f)]

stats = {tag: 0 for tag, _, _ in SOURCES}
for fid in fids:
    od = f'{OUT}/{fid}'
    os.makedirs(od, exist_ok=True)
    for tag, src_root, mode in SOURCES:
        sd = od if mode == 'in-place' else (f'{src_root}/{fid}' if src_root else None)
        if not sd or not os.path.isdir(sd): continue
        for algo in ('rfta', 'mes'):
            for p in glob.glob(f'{sd}/{algo}_*.obj'):
                base = os.path.basename(p)
                if '_sw' in base: continue  # already renamed (idempotent)
                gl = base[len(algo)+1:-4]
                dst = f'{od}/{algo}_{tag}_{gl}.obj'
                if os.path.exists(dst): continue
                shutil.move(p, dst)
                stats[tag] += 1
for tag, _, _ in SOURCES:
    print(f'  collected {tag}: {stats[tag]} files')
PYEOF
echo

echo "── 1/5 compute_metrics.py ──"
"${PYTHON}" scripts/compute_metrics.py \
    --index "${INDEX}" \
    --gt-root "${GT_ROOT}" \
    --out-root "${OUT_ROOT}" \
    --algos rfta_sw1,rfta_sw10,rfta_sw100,mes_sw1,mes_sw10,mes_sw100,ours,mc \
    --output /scratch/ycheng27/new_solid/metrics.csv \
    --workers ${SLURM_CPUS_PER_TASK}

echo
echo "── 2/5 topology_check.py ──"
"${PYTHON}" scripts/topology_check.py \
    --index "${INDEX}" \
    --gt-root "${GT_ROOT}" \
    --out-root "${OUT_ROOT}" \
    --algos rfta_sw1,rfta_sw10,rfta_sw100,mes_sw1,mes_sw10,mes_sw100,ours,mc \
    --output /scratch/ycheng27/new_solid/topology.csv \
    --workers ${SLURM_CPUS_PER_TASK}

echo
echo "── 3/5 summarize_metrics.py ──"
"${PYTHON}" scripts/summarize_metrics.py \
    --csv-glob /scratch/ycheng27/new_solid/metrics.csv \
    --output /scratch/ycheng27/new_solid/summary_metrics.tsv

echo
echo "── 4/5 summarize_topology.py ──"
"${PYTHON}" scripts/summarize_topology.py \
    --input /scratch/ycheng27/new_solid/topology.csv

echo
echo "=== ALL DONE $(date -Iseconds) ==="
