#!/bin/bash
# Sequentially run ours+mc at gl=150, then 200, then 300 on the first 50
# new_solid models. Escalate to the next gl only if the current one produced
# results for >=30/50 models; stop at the first failure. Then compute metrics
# (including the existing gl=40,60,80,100) and plot.
set -uo pipefail
cd /home/ycheng27/code/sdfgradients
REPO=/home/ycheng27/code/sdfgradients
ROOT=/scratch/ycheng27/new_solid
IDX=$ROOT/index.csv
LOG=$ROOT/logs
mkdir -p "$LOG"
PY=/home/ycheng27/envs/sdf/bin/python

GLS=(200 300)   # gl=150 already completed (recovered) in a prior round
declare -A PART=( [150]=normal [200]=bigmem [300]=bigmem )
declare -A MEM=(  [150]=160G   [200]=400G    [300]=1000G )
FIDS=$(tail -n +2 "$IDX" | head -50 | cut -d, -f1)

SUCC=()
for gl in "${GLS[@]}"; do
    echo "=== submit gl=$gl (part=${PART[$gl]} mem=${MEM[$gl]})  $(date -Iseconds) ==="
    jid=$(sbatch --parsable --array=0-49 --partition="${PART[$gl]}" --mem="${MEM[$gl]}" \
          --output="$LOG/hr_${gl}_%A_%a.out" --error="$LOG/hr_${gl}_%A_%a.err" \
          --export=ALL,GL=$gl "$REPO/scripts/submit_highres.sbatch")
    echo "  job=$jid"
    while squeue -j "$jid" -h -o "%t" 2>/dev/null | grep -qE "PD|R|CG|CF"; do sleep 60; done
    n=0
    for fid in $FIDS; do [ -s "$ROOT/out/$fid/ours_$gl.obj" ] && n=$((n+1)); done
    echo "  gl=$gl produced ours: $n/50  $(date -Iseconds)"
    if [ "$n" -ge 30 ]; then
        SUCC+=("$gl"); echo "  gl=$gl OK -> escalate"
    else
        echo "  gl=$gl FAILED ($n/50) -> stop escalation"; break
    fi
done

NEWGLS=$(IFS=,; echo "${SUCC[*]:-}")
GRIDS="40,60,80,100,150"; [ -n "$NEWGLS" ] && GRIDS="$GRIDS,$NEWGLS"
echo "=== submit STATS job: grids=$GRIDS  $(date -Iseconds) ==="
sbatch --partition=normal --account=ygingold --qos=normal --time=1:00:00 \
       --cpus-per-task=16 --mem=64G --nodes=1 \
       --output="$LOG/stats_%A.out" --error="$LOG/stats_%A.err" \
       "$REPO/scripts/submit_stats.sbatch" "$GRIDS"
echo "=== ORCHESTRATION DONE (stats job submitted)  $(date -Iseconds) ==="
