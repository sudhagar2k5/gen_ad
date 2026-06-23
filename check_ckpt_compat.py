#!/usr/bin/env python
"""Check if GenAD open-loop checkpoint is compatible with Bench2Drive config."""
import torch
import sys
sys.path.insert(0, '/home/sudhagar/Bench2DriveZoo')
sys.path.insert(0, '/home/sudhagar/Bench2DriveZoo/adzoo/genad/..')

ckpt_path = '/home/sudhagar/Bench2DriveZoo/ckpts/GenAD/checkpoints.pth'
print(f"Loading checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location='cpu')

print(f"\nCheckpoint keys: {list(ckpt.keys())}")
if 'state_dict' in ckpt:
    state_dict = ckpt['state_dict']
elif 'model' in ckpt:
    state_dict = ckpt['model']
else:
    state_dict = ckpt

print(f"State dict has {len(state_dict)} keys")
print(f"\nFirst 30 keys:")
for i, k in enumerate(sorted(state_dict.keys())):
    if i < 30:
        print(f"  {k}: {state_dict[k].shape}")

# Check for key patterns
patterns = ['img_backbone', 'img_neck', 'pts_bbox_head', 'plan', 'vae', 'traj', 'map']
print(f"\nKey pattern analysis:")
for p in patterns:
    matching = [k for k in state_dict.keys() if p in k.lower()]
    print(f"  '{p}': {len(matching)} keys")

print(f"\nMetadata:")
if 'meta' in ckpt:
    print(f"  meta: {ckpt['meta']}")
if 'epoch' in ckpt:
    print(f"  epoch: {ckpt['epoch']}")
