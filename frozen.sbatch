#!/bin/bash
#SBATCH -J sarsae_frozen
#SBATCH -p long
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x_%j.out
source "$SLURM_SUBMIT_DIR/env.sh"; cd "$SARSAE_ROOT"
python -u frozen.py