#!/bin/bash
# =============================================================================
# GenAD Training on Bench2Drive data
# Run this script INSIDE WSL2 Ubuntu:
#   wsl -d Ubuntu-22.04
#   cd ~/Bench2DriveZoo
#   bash ~/GenAD_scripts/run_training.sh
# =============================================================================

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate b2d_zoo

cd $HOME/Bench2DriveZoo

echo "Starting GenAD training..."
sh ./adzoo/genad/dist_train.sh ./adzoo/genad/configs/VAD/GenAD_config_b2d.py 1
