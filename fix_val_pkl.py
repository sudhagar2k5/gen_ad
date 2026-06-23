#!/usr/bin/env python
"""Create a minimal val pkl from the train pkl (use a subset for validation)."""
import pickle
import os

infos_dir = '/home/sudhagar/Bench2DriveZoo/data/infos'
train_path = os.path.join(infos_dir, 'b2d_infos_train.pkl')
val_path = os.path.join(infos_dir, 'b2d_infos_val.pkl')
map_path = os.path.join(infos_dir, 'b2d_map_infos.pkl')

# Load train infos
with open(train_path, 'rb') as f:
    train_data = pickle.load(f)

print(f"Train data type: {type(train_data)}")
if isinstance(train_data, dict):
    print(f"Train keys: {list(train_data.keys())}")
    for k, v in train_data.items():
        if isinstance(v, list):
            print(f"  {k}: list of {len(v)} items")
        else:
            print(f"  {k}: {type(v)}")
elif isinstance(train_data, list):
    print(f"Train data: list of {len(train_data)} items")

# Create val pkl as a small subset of train (last 10% or at least 1 sample)
if isinstance(train_data, dict) and 'infos' in train_data:
    infos = train_data['infos']
    n_val = max(1, len(infos) // 10)
    val_data = dict(train_data)
    val_data['infos'] = infos[-n_val:]
    train_data['infos'] = infos[:-n_val] if n_val < len(infos) else infos
elif isinstance(train_data, list):
    n_val = max(1, len(train_data) // 10)
    val_data = train_data[-n_val:]
    train_data = train_data[:-n_val] if n_val < len(train_data) else train_data
else:
    val_data = train_data

with open(val_path, 'wb') as f:
    pickle.dump(val_data, f)
print(f"Val pkl created at {val_path}")

# Create empty map infos if not exists
if not os.path.exists(map_path):
    with open(map_path, 'wb') as f:
        pickle.dump({}, f)
    print(f"Empty map infos created at {map_path}")
else:
    print(f"Map infos already exists at {map_path}")
