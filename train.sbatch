#!/bin/bash
#SBATCH -J sarsae_train
#SBATCH -p long
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 3-00:00:00
#SBATCH --array=0-3
#SBATCH -o logs/%x_%A_%a.out
source "$SLURM_SUBMIT_DIR/env.sh"; cd "$SARSAE_ROOT"
python -u worker.py "$@"
