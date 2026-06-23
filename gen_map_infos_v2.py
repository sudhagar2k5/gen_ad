#!/usr/bin/env python
"""Generate b2d_map_infos.pkl processing one town at a time to save memory."""
import os, pickle, gc
import numpy as np

MAP_ROOT = '/home/sudhagar/Bench2DriveZoo/data/bench2drive/maps'
OUT_DIR = '/home/sudhagar/Bench2DriveZoo/data/infos'

# Only process towns we actually need (from our mini dataset clips)
needed_towns = set()
v1_dir = '/home/sudhagar/Bench2DriveZoo/data/bench2drive/v1'
for d in os.listdir(v1_dir):
    if os.path.isdir(os.path.join(v1_dir, d)):
        parts = d.split('_')
        for i, p in enumerate(parts):
            if p.startswith('Town'):
                needed_towns.add(p)
                break

print(f"Towns needed: {needed_towns}")

map_infos = {}
for file_name in sorted(os.listdir(MAP_ROOT)):
    if '.npz' not in file_name:
        continue
    town_name = file_name.split('_')[0]
    if town_name not in needed_towns:
        print(f"  Skipping {town_name} (not needed)")
        continue

    print(f"Processing {file_name} ({town_name})...")
    map_info = dict(np.load(os.path.join(MAP_ROOT, file_name), allow_pickle=True)['arr'])
    map_infos[town_name] = {}
    lane_points = []
    lane_types = []
    lane_sample_points = []
    trigger_volumes_points = []
    trigger_volumes_types = []
    trigger_volumes_sample_points = []

    for road_id, road in map_info.items():
        for lane_id, lane in road.items():
            if lane_id == 'Trigger_Volumes':
                for stv in lane:
                    points = np.array(stv['Points'])
                    points[:, 1] *= -1
                    trigger_volumes_points.append(points)
                    trigger_volumes_sample_points.append(points.mean(axis=0))
                    trigger_volumes_types.append(stv['Type'])
            else:
                for single_lane in lane:
                    points = np.array([rp[0] for rp in single_lane['Points']])
                    points[:, 1] *= -1
                    lane_points.append(points)
                    lane_types.append(single_lane['Type'])
                    ll = points.shape[0]
                    dp = [50*i for i in range(ll//50 + (1 if ll%50 != 0 else 0))]
                    dp.append(ll - 1)
                    lane_sample_points.append(points[dp])

    map_infos[town_name]['lane_points'] = lane_points
    map_infos[town_name]['lane_sample_points'] = lane_sample_points
    map_infos[town_name]['lane_types'] = lane_types
    map_infos[town_name]['trigger_volumes_points'] = trigger_volumes_points
    map_infos[town_name]['trigger_volumes_sample_points'] = trigger_volumes_sample_points
    map_infos[town_name]['trigger_volumes_types'] = trigger_volumes_types

    del map_info
    gc.collect()
    print(f"  Done: {len(lane_points)} lanes, {len(trigger_volumes_points)} triggers")

os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, 'b2d_map_infos.pkl'), 'wb') as f:
    pickle.dump(map_infos, f)
print(f"\nSaved map infos with {len(map_infos)} towns: {list(map_infos.keys())}")
