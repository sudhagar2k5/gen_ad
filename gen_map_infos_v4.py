#!/usr/bin/env python
"""Generate b2d_map_infos.pkl - skip oversized towns (11, 12, 13)."""
import os, pickle, gc
import numpy as np

MAP_ROOT = '/home/sudhagar/Bench2DriveZoo/data/bench2drive/maps'
OUT_DIR = '/home/sudhagar/Bench2DriveZoo/data/infos'

# Skip towns that are too large for 16GB RAM
skip_towns = {'Town11', 'Town12', 'Town13'}
needed_towns = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town10HD', 'Town15']

os.makedirs(OUT_DIR, exist_ok=True)
map_infos = {}

for town in needed_towns:
    fname = f"{town}_HD_map.npz"
    fpath = os.path.join(MAP_ROOT, fname)
    if not os.path.exists(fpath):
        continue
    print(f"Processing {fname}...", flush=True)
    data = np.load(fpath, allow_pickle=True)
    mi = dict(data['arr'])
    data.close(); del data

    result = {}
    lp, lt, lsp, tvp, tvt, tvsp = [], [], [], [], [], []
    for road_id, road in mi.items():
        for lane_id, lane in road.items():
            if lane_id == 'Trigger_Volumes':
                for stv in lane:
                    pts = np.array(stv['Points'], dtype=np.float32)
                    pts[:, 1] *= -1
                    tvp.append(pts); tvsp.append(pts.mean(axis=0)); tvt.append(stv['Type'])
            else:
                for sl in lane:
                    pts = np.array([rp[0] for rp in sl['Points']], dtype=np.float32)
                    pts[:, 1] *= -1
                    lp.append(pts); lt.append(sl['Type'])
                    ll = pts.shape[0]
                    dp = [50*i for i in range(ll//50 + (1 if ll%50 else 0))]
                    dp.append(ll-1)
                    lsp.append(pts[dp])
    result['lane_points'] = lp; result['lane_sample_points'] = lsp; result['lane_types'] = lt
    result['trigger_volumes_points'] = tvp; result['trigger_volumes_sample_points'] = tvsp; result['trigger_volumes_types'] = tvt
    map_infos[town] = result
    del mi; gc.collect()
    print(f"  {len(lp)} lanes, {len(tvp)} triggers")

with open(os.path.join(OUT_DIR, 'b2d_map_infos.pkl'), 'wb') as f:
    pickle.dump(map_infos, f)
print(f"\nSaved: {list(map_infos.keys())}")

# Also filter train/val pkl to exclude clips from skipped towns
train_path = os.path.join(OUT_DIR, 'b2d_infos_train.pkl')
val_path = os.path.join(OUT_DIR, 'b2d_infos_val.pkl')

with open(train_path, 'rb') as f:
    train_data = pickle.load(f)

def has_skip_town(info):
    scene = info.get('scene_name', '') if isinstance(info, dict) else ''
    for t in skip_towns:
        if t in scene:
            return True
    return False

if isinstance(train_data, list):
    # Check what keys are in the data
    if train_data:
        sample = train_data[0]
        if isinstance(sample, dict):
            print(f"Sample keys: {list(sample.keys())[:10]}")
            # Try to find the scene/town name
            for k in ['scene_name', 'scene_token', 'folder']:
                if k in sample:
                    print(f"  {k}: {sample[k]}")

    original_len = len(train_data)
    filtered = []
    for info in train_data:
        skip = False
        if isinstance(info, dict):
            for k in ['scene_name', 'folder', 'scene_token']:
                val = info.get(k, '')
                if isinstance(val, str):
                    for t in skip_towns:
                        if t in val:
                            skip = True
                            break
                if skip:
                    break
        if not skip:
            filtered.append(info)

    n_val = max(1, len(filtered) // 10)
    val_data = filtered[-n_val:]
    train_filtered = filtered[:-n_val]

    with open(train_path, 'wb') as f:
        pickle.dump(train_filtered, f)
    with open(val_path, 'wb') as f:
        pickle.dump(val_data, f)
    print(f"\nFiltered train: {original_len} -> {len(train_filtered)}")
    print(f"Val: {len(val_data)}")
