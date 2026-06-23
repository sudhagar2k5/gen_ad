#!/bin/bash
# =============================================================================
# GenAD Open-Loop Evaluation
# Run this script INSIDE WSL2 Ubuntu:
#   wsl -d Ubuntu-22.04
#   cd ~/Bench2DriveZoo
#   bash ~/GenAD_scripts/run_openloop_eval.sh
# =============================================================================

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate b2d_zoo

cd $HOME/Bench2DriveZoo

# Update the epoch number and checkpoint path
CKPT_PATH=./work_dirs/GenAD_config_b2d/epoch_1.pth

echo "Starting GenAD open-loop evaluation..."
sh ./adzoo/genad/dist_test.sh ./adzoo/genad/configs/VAD/GenAD_config_b2d.py $CKPT_PATH 1
