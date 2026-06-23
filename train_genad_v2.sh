#!/bin/bash
source /home/sudhagar/miniconda3/etc/profile.d/conda.sh
conda activate b2d_zoo
cd /home/sudhagar/Bench2DriveZoo

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/home/sudhagar/Bench2DriveZoo/adzoo/genad/..":$PYTHONPATH

echo "Starting GenAD training..."

python -m torch.distributed.launch --nproc_per_node=1 --master_port=28509 \
    /home/sudhagar/Bench2DriveZoo/adzoo/genad/train.py \
    /home/sudhagar/Bench2DriveZoo/adzoo/genad/configs/VAD/GenAD_config_b2d.py \
    --launcher pytorch --deterministic \
    --work-dir /home/sudhagar/Bench2DriveZoo/work_dirs/GenAD_config_b2d
