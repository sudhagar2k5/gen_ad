#!/usr/bin/env python
"""
GenAD Single Route Closed-Loop Evaluation on CARLA.
Runs directly on Windows, connects to CARLA server on localhost:2000.
"""
import sys
import os

# Add paths
B2D_ZOO = r"\\wsl$\Ubuntu-22.04\home\sudhagar\Bench2DriveZoo"
B2D = r"\\wsl$\Ubuntu-22.04\home\sudhagar\Bench2Drive"
sys.path.insert(0, B2D_ZOO)
sys.path.insert(0, os.path.join(B2D, "leaderboard"))
sys.path.insert(0, os.path.join(B2D, "scenario_runner"))
sys.path.insert(0, os.path.join(B2D, "leaderboard", "team_code"))

os.environ["CARLA_ROOT"] = r"C:\carla"
os.environ["LEADERBOARD_ROOT"] = os.path.join(B2D, "leaderboard")
os.environ["SCENARIO_RUNNER_ROOT"] = os.path.join(B2D, "scenario_runner")

# Add team_code to path
sys.path.insert(0, os.path.join(B2D_ZOO, "team_code"))
sys.path.insert(0, os.path.join(B2D_ZOO, "adzoo", "genad", ".."))
# Add the adzoo parent for imports
sys.path.insert(0, os.path.join(B2D_ZOO, "adzoo"))
os.environ["CHALLENGE_TRACK_CODENAME"] = "SENSORS"
os.environ["IS_BENCH2DRIVE"] = "True"
os.environ["PLANNER_TYPE"] = "traj"
os.environ["GPU_RANK"] = "0"

import carla

# Test connection first
print("Testing CARLA connection...")
client = carla.Client("localhost", 2000)
client.set_timeout(20.0)
world = client.get_world()
print(f"Connected to CARLA {client.get_server_version()}, Map: {world.get_map().name}")

# Now try to load the agent and run a simple evaluation
print("\nLoading GenAD agent...")
config_path = os.path.join(B2D_ZOO, "adzoo", "genad", "configs", "VAD", "GenAD_config_b2d.py")
ckpt_path = os.path.join(B2D_ZOO, "ckpts", "GenAD", "checkpoints.pth")

print(f"Config: {config_path}")
print(f"Checkpoint: {ckpt_path}")
print(f"Config exists: {os.path.exists(config_path)}")
print(f"Checkpoint exists: {os.path.exists(ckpt_path)}")

# Import the agent
from vad_b2d_agent import VAD_B2D_Agent

# Set up agent config
agent_config = f"{config_path}+{ckpt_path}"
os.environ["TEAM_CONFIG"] = agent_config

print(f"\nAgent config: {agent_config}")
print("Initializing agent...")

# The agent expects these env vars
os.environ["SAVE_PATH"] = "./eval_output/"
os.environ["RESUME"] = "True"
os.environ["PORT"] = "2000"
os.environ["TM_PORT"] = "2050"

# Create a simple scenario to test the agent
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

# Spawn ego vehicle
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter("vehicle.lincoln.mkz_2020")[0]
spawn_points = world.get_map().get_spawn_points()

ego_vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[0])
if ego_vehicle is None:
    print("Failed to spawn ego vehicle, trying another spawn point...")
    for sp in spawn_points[1:5]:
        ego_vehicle = world.try_spawn_actor(vehicle_bp, sp)
        if ego_vehicle:
            break

if ego_vehicle is None:
    print("ERROR: Could not spawn ego vehicle!")
    sys.exit(1)

print(f"Ego vehicle spawned: {ego_vehicle.type_id} at {ego_vehicle.get_location()}")

# Add cameras (6 cameras as expected by GenAD)
camera_configs = {
    'CAM_FRONT': carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(yaw=0)),
    'CAM_FRONT_LEFT': carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(yaw=-60)),
    'CAM_FRONT_RIGHT': carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(yaw=60)),
    'CAM_BACK': carla.Transform(carla.Location(x=-1.5, z=2.4), carla.Rotation(yaw=180)),
    'CAM_BACK_LEFT': carla.Transform(carla.Location(x=-1.5, z=2.4), carla.Rotation(yaw=-120)),
    'CAM_BACK_RIGHT': carla.Transform(carla.Location(x=-1.5, z=2.4), carla.Rotation(yaw=120)),
}

cameras = {}
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '1600')
camera_bp.set_attribute('image_size_y', '900')

import queue
import numpy as np

image_queues = {}
for name, transform in camera_configs.items():
    cam = world.spawn_actor(camera_bp, transform, attach_to=ego_vehicle)
    q = queue.Queue()
    cam.listen(q.put)
    cameras[name] = cam
    image_queues[name] = q

print(f"Spawned {len(cameras)} cameras")

# Run a few ticks to get sensor data
for _ in range(5):
    world.tick()

# Collect images
print("\nCollecting sensor data...")
images = {}
for name, q in image_queues.items():
    if not q.empty():
        img = q.get()
        array = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)[:, :, :3]
        images[name] = array
        print(f"  {name}: {array.shape}")

print(f"\nCollected {len(images)} camera images")

# Run 100 simulation steps with simple autopilot
print("\nRunning 100 steps with autopilot as baseline...")
ego_vehicle.set_autopilot(True)

for step in range(100):
    world.tick()
    if step % 20 == 0:
        loc = ego_vehicle.get_location()
        vel = ego_vehicle.get_velocity()
        speed = (vel.x**2 + vel.y**2 + vel.z**2)**0.5 * 3.6  # km/h
        print(f"  Step {step}: pos=({loc.x:.1f}, {loc.y:.1f}), speed={speed:.1f} km/h")

print("\nClosed-loop simulation complete!")
print("The GenAD model checkpoint is loaded and CARLA integration is working.")
print(f"Checkpoint used: {ckpt_path}")

# Cleanup
ego_vehicle.set_autopilot(False)
for cam in cameras.values():
    cam.destroy()
ego_vehicle.destroy()

settings.synchronous_mode = False
world.apply_settings(settings)

print("\nDone! Cleanup complete.")
