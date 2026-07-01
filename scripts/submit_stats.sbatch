#!/bin/bash
# Final stats + plot for the high-res ours/mc run. GRIDS passed via env.
#SBATCH --job-name=sdf-hr-stats
#SBATCH --account=ygingold
#SBATCH --qos=normal
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1:00:00

set -uo pipefail
module purge 2>/dev/null || true
module load hosts/hopper gnu12 python/3.12.1-33 2>/dev/null || true

REPO=/home/ycheng27/code/sdfgradients
ROOT=/scratch/ycheng27/new_solid
PY=/home/ycheng27/envs/sdf/bin/python
# grids come as the first positional arg (env vars can't hold commas cleanly through
# sbatch --export, which splits the value on commas).
GRIDS="${1:-40,60,80,100}"

"$PY" "$REPO/scripts/compute_metrics.py" --algos ours,mc --grids "$GRIDS" --limit 50 --workers 16 \
    --index "$ROOT/index.csv" --out-root "$ROOT/out" --output "$ROOT/metrics_highres.csv"

echo "===== TABLE (mean) ====="
"$PY" "$REPO/scripts/summarize_AB.py" --csv "$ROOT/metrics_highres.csv" \
    | tee "$ROOT/metrics_highres_summary.txt"

"$PY" "$REPO/scripts/plot_ours_mc.py" --csv "$ROOT/metrics_highres.csv" \
    --out "$REPO/tmp_ours_mc_curves.png"
echo "STATS DONE"
