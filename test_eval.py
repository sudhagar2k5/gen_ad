import sys, traceback
try:
    print("Step 1: imports...")
    import carla
    import torch
    import numpy as np
    print("Step 2: CARLA connect...")
    c = carla.Client("localhost", 2000)
    c.set_timeout(10)
    w = c.get_world()
    print(f"  Map: {w.get_map().name}")
    print("Step 3: Load checkpoint...")
    ckpt = torch.load(r"C:\GenAD\genad_ckpt.pth", map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    print(f"  Keys: {len(sd)}, Epoch: {ckpt.get('epoch', 'N/A')}")
    print("ALL OK")
except Exception as e:
    traceback.print_exc()
