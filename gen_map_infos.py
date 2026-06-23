#!/usr/bin/env python
"""Generate b2d_map_infos.pkl from HD maps."""
import os, sys
sys.path.insert(0, '/home/sudhagar/Bench2DriveZoo/mmcv/datasets')
os.chdir('/home/sudhagar/Bench2DriveZoo/mmcv/datasets')

from prepare_B2D import gengrate_map

MAP_ROOT = '/home/sudhagar/Bench2DriveZoo/data/bench2drive/maps'

# Override the paths
import prepare_B2D
prepare_B2D.MAP_ROOT = MAP_ROOT
prepare_B2D.OUT_DIR = '/home/sudhagar/Bench2DriveZoo/data/infos'

gengrate_map(MAP_ROOT)
print("Map infos generated!")
