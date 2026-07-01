#!/bin/bash
# Durable pipeline controller: runs orchestrate_highres.sh as a SLURM job so the
# 150->200->300 escalation + stats survive independently of any interactive session.
# Tiny footprint (mostly sleeps polling squeue); submits the gl arrays and the
# final stats job itself.
#SBATCH --job-name=hr-ctl
#SBATCH --account=ygingold
#SBATCH --qos=normal
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/ycheng27/new_solid/logs/orchestrator_%A.out
#SBATCH --error=/scratch/ycheng27/new_solid/logs/orchestrator_%A.err

set -uo pipefail
cd /home/ycheng27/code/sdfgradients
exec bash scripts/orchestrate_highres.sh
