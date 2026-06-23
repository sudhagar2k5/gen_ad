#!/bin/bash
source /home/sudhagar/miniconda3/etc/profile.d/conda.sh
conda activate b2d_zoo
cd /home/sudhagar/Bench2DriveZoo

export CUDA_VISIBLE_DEVICES=0

echo "Starting GenAD training on Bench2Drive mini dataset..."
echo "Config: adzoo/genad/configs/VAD/GenAD_config_b2d.py"
echo "GPU: RTX 4070 Laptop (8GB VRAM)"

sh ./adzoo/genad/dist_train.sh ./adzoo/genad/configs/VAD/GenAD_config_b2d.py 1 \
  --work-dir ./work_dirs/GenAD_config_b2d
