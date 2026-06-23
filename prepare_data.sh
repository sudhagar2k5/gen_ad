#!/bin/bash
source /home/sudhagar/miniconda3/etc/profile.d/conda.sh
conda activate b2d_zoo
cd /home/sudhagar/Bench2DriveZoo/mmcv/datasets
python prepare_B2D.py --workers 4
echo "DATA_PREP_DONE"
ls -lh /home/sudhagar/Bench2DriveZoo/data/infos/
