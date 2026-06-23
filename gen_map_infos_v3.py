#!/usr/bin/env python
"""Generate b2d_map_infos.pkl - process one town at a time, saving each separately."""
import os, pickle, gc, sys
import numpy as np

MAP_ROOT = '/home/sudhagar/Bench2DriveZoo/data/bench2drive/maps'
OUT_DIR = '/home/sudhagar/Bench2DriveZoo/data/infos'

# Only the towns in our mini dataset, process small ones first
needed_towns = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05',
                'Town10HD', 'Town15', 'Town11', 'Town12', 'Town13']

os.makedirs(OUT_DIR, exist_ok=True)

def process_town(town_name):
    file_name = f"{town_name}_HD_map.npz"
    fpath = os.path.join(MAP_ROOT, file_name)
    if not os.path.exists(fpath):
        print(f"  {file_name} not found, skipping")
        return None

    print(f"Processing {file_name}...", flush=True)
    fsize = os.path.getsize(fpath) / 1024 / 1024
    print(f"  File size: {fsize:.1f} MB", flush=True)

    data = np.load(fpath, allow_pickle=True)
    map_info = dict(data['arr'])
    data.close()
    del data

    result = {}
    lane_points = []
    lane_types = []
    lane_sample_points = []
    tvp = []
    tvt = []
    tvsp = []

    for road_id, road in map_info.items():
        for lane_id, lane in road.items():
            if lane_id == 'Trigger_Volumes':
                for stv in lane:
                    points = np.array(stv['Points'], dtype=np.float32)
                    points[:, 1] *= -1
                    tvp.append(points)
                    tvsp.append(points.mean(axis=0))
                    tvt.append(stv['Type'])
            else:
                for single_lane in lane:
                    points = np.array([rp[0] for rp in single_lane['Points']], dtype=np.float32)
                    points[:, 1] *= -1
                    lane_points.append(points)
                    lane_types.append(single_lane['Type'])
                    ll = points.shape[0]
                    dp = [50*i for i in range(ll//50 + (1 if ll%50 != 0 else 0))]
                    dp.append(ll - 1)
                    lane_sample_points.append(points[dp])

    result['lane_points'] = lane_points
    result['lane_sample_points'] = lane_sample_points
    result['lane_types'] = lane_types
    result['trigger_volumes_points'] = tvp
    result['trigger_volumes_sample_points'] = tvsp
    result['trigger_volumes_types'] = tvt

    del map_info
    gc.collect()
    print(f"  Done: {len(lane_points)} lanes, {len(tvp)} triggers", flush=True)
    return result

map_infos = {}
for town in needed_towns:
    result = process_town(town)
    if result is not None:
        map_infos[town] = result
    gc.collect()

with open(os.path.join(OUT_DIR, 'b2d_map_infos.pkl'), 'wb') as f:
    pickle.dump(map_infos, f)
print(f"\nSaved: {list(map_infos.keys())}")
