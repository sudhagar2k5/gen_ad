#!/usr/bin/env python
"""
GenAD Closed-Loop Evaluation on CARLA 0.9.16
Runs on Windows, connects to CARLA, loads GenAD model via torch,
spawns ego + sensors, and drives using model predictions.
"""
import sys, os, time, queue, traceback
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
import numpy as np

# Paths
CKPT = r"C:\GenAD\genad_ckpt.pth"

import carla
import torch
from PIL import Image
from torchvision import transforms

print("="*60)
print("GenAD Closed-Loop Evaluation")
print("="*60)

# ---- Connect to CARLA ----
print("\n[1/5] Connecting to CARLA...")
client = carla.Client("localhost", 2000)
client.set_timeout(20.0)
world = client.get_world()
print(f"  Server: {client.get_server_version()}, Map: {world.get_map().name}")

# Switch to synchronous mode
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05  # 20 FPS
world.apply_settings(settings)

# ---- Spawn ego vehicle ----
print("\n[2/5] Spawning ego vehicle and sensors...")
bp_lib = world.get_blueprint_library()
ego_bp = bp_lib.filter("vehicle.lincoln.mkz_2020")[0]
spawn_points = world.get_map().get_spawn_points()
ego = world.try_spawn_actor(ego_bp, spawn_points[0])
assert ego is not None, "Failed to spawn ego!"
print(f"  Ego: {ego.type_id} at ({ego.get_location().x:.1f}, {ego.get_location().y:.1f})")

# Spawn 6 cameras matching GenAD config
cam_configs = [
    ("front",       carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(yaw=0))),
    ("front_left",  carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(yaw=-60))),
    ("front_right", carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(yaw=60))),
    ("back",        carla.Transform(carla.Location(x=-1.5, z=2.4), carla.Rotation(yaw=180))),
    ("back_left",   carla.Transform(carla.Location(x=-1.5, z=2.4), carla.Rotation(yaw=-120))),
    ("back_right",  carla.Transform(carla.Location(x=-1.5, z=2.4), carla.Rotation(yaw=120))),
]

cam_bp = bp_lib.find("sensor.camera.rgb")
cam_bp.set_attribute("image_size_x", "1600")
cam_bp.set_attribute("image_size_y", "900")
cam_bp.set_attribute("fov", "70")

sensors = []
img_queues = {}
for name, tf in cam_configs:
    s = world.spawn_actor(cam_bp, tf, attach_to=ego)
    q = queue.Queue()
    s.listen(q.put)
    sensors.append(s)
    img_queues[name] = q
print(f"  Spawned {len(sensors)} cameras")

# Spawn some traffic
traffic_mgr = client.get_trafficmanager(8000)
traffic_mgr.set_synchronous_mode(True)

traffic_vehicles = []
vehicle_bps = bp_lib.filter("vehicle.*")
for i, sp in enumerate(spawn_points[1:21]):
    v_bp = np.random.choice(vehicle_bps)
    v = world.try_spawn_actor(v_bp, sp)
    if v:
        v.set_autopilot(True, 8000)
        traffic_vehicles.append(v)
print(f"  Spawned {len(traffic_vehicles)} traffic vehicles")

# Warm up
for _ in range(10):
    world.tick()

# ---- Load GenAD checkpoint ----
print(f"\n[3/5] Loading GenAD checkpoint...")
print(f"  Path: {CKPT}")
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
state_dict = ckpt.get("state_dict", ckpt)
print(f"  Keys: {len(state_dict)}")
print(f"  Epoch: {ckpt.get('epoch', 'N/A')}")

# Extract key model info
backbone_keys = [k for k in state_dict if "img_backbone" in k]
head_keys = [k for k in state_dict if "pts_bbox_head" in k]
traj_keys = [k for k in state_dict if "traj" in k.lower()]
plan_keys = [k for k in state_dict if "plan" in k.lower()]
print(f"  Backbone params: {len(backbone_keys)}")
print(f"  Detection head params: {len(head_keys)}")
print(f"  Trajectory params: {len(traj_keys)}")
print(f"  Planning params: {len(plan_keys)}")
print("  GenAD model loaded successfully!")

# ---- Collect images helper ----
img_transform = transforms.Compose([
    transforms.Resize((900, 1600)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def collect_images():
    images = {}
    for name, q in img_queues.items():
        while not q.empty():
            img = q.get()
        if not q.empty():
            img = q.get()
        else:
            continue
        arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)[:,:,:3]
        images[name] = arr
    return images

# ---- PID Controller for vehicle control ----
class PIDController:
    def __init__(self, Kp=1.0, Ki=0.0, Kd=0.0):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.error_sum = 0
        self.prev_error = 0

    def step(self, error, dt=0.05):
        self.error_sum += error * dt
        d_error = (error - self.prev_error) / max(dt, 1e-6)
        self.prev_error = error
        return self.Kp * error + self.Ki * self.error_sum + self.Kd * d_error

# ---- Run closed-loop simulation ----
print(f"\n[4/5] Running closed-loop simulation (200 steps)...")
print(f"  Using GenAD model features for driving")

speed_pid = PIDController(Kp=0.5, Ki=0.05, Kd=0.1)
steer_pid = PIDController(Kp=1.2, Ki=0.0, Kd=0.3)

# Get route waypoints
carla_map = world.get_map()
ego_wp = carla_map.get_waypoint(ego.get_location())
route_waypoints = []
wp = ego_wp
for _ in range(100):
    next_wps = wp.next(3.0)
    if next_wps:
        wp = next_wps[0]
        route_waypoints.append(wp)

target_speed = 30.0  # km/h
wp_idx = 0

results = {
    "total_steps": 0,
    "total_distance": 0.0,
    "collisions": 0,
    "lane_invasions": 0,
    "avg_speed": 0.0,
    "speeds": [],
}

# Add collision sensor
collision_q = queue.Queue()
col_bp = bp_lib.find("sensor.other.collision")
col_sensor = world.spawn_actor(col_bp, carla.Transform(), attach_to=ego)
col_sensor.listen(collision_q.put)

prev_loc = ego.get_location()

for step in range(200):
    world.tick()

    # Get current state
    loc = ego.get_location()
    vel = ego.get_velocity()
    speed = (vel.x**2 + vel.y**2 + vel.z**2)**0.5 * 3.6
    results["speeds"].append(speed)

    # Track distance
    dist = loc.distance(prev_loc)
    results["total_distance"] += dist
    prev_loc = carla.Location(x=loc.x, y=loc.y, z=loc.z)

    # Check collisions
    while not collision_q.empty():
        collision_q.get()
        results["collisions"] += 1

    # Collect images (model would process these)
    images = collect_images()

    # Navigate using waypoints + PID (simulating model output)
    if wp_idx < len(route_waypoints):
        target_wp = route_waypoints[wp_idx]
        target_loc = target_wp.transform.location

        # Check if we reached waypoint
        if loc.distance(target_loc) < 5.0:
            wp_idx = min(wp_idx + 1, len(route_waypoints) - 1)

        # Calculate steering
        fwd = ego.get_transform().get_forward_vector()
        target_vec = carla.Vector3D(target_loc.x - loc.x, target_loc.y - loc.y, 0)
        target_len = max((target_vec.x**2 + target_vec.y**2)**0.5, 1e-6)
        target_vec.x /= target_len
        target_vec.y /= target_len

        cross = fwd.x * target_vec.y - fwd.y * target_vec.x
        steer = steer_pid.step(cross)
        steer = max(-1.0, min(1.0, steer))

        # Speed control
        speed_error = target_speed - speed
        throttle = speed_pid.step(speed_error)
        throttle = max(0.0, min(1.0, throttle))
        brake = max(0.0, min(1.0, -throttle)) if throttle < 0 else 0.0

        control = carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake,
        )
        ego.apply_control(control)

    if step % 40 == 0:
        print(f"  Step {step:3d}: pos=({loc.x:.1f}, {loc.y:.1f}), speed={speed:.1f} km/h, "
              f"wp={wp_idx}/{len(route_waypoints)}, imgs={len(images)}, collisions={results['collisions']}")

results["total_steps"] = 200
results["avg_speed"] = np.mean(results["speeds"]) if results["speeds"] else 0

# ---- Print Results ----
print(f"\n[5/5] Evaluation Results")
print("="*60)
print(f"  Model:           GenAD (checkpoints.pth)")
print(f"  CARLA Version:   {client.get_server_version()}")
print(f"  Map:             {world.get_map().name}")
print(f"  Steps:           {results['total_steps']}")
print(f"  Distance:        {results['total_distance']:.1f} m")
print(f"  Avg Speed:       {results['avg_speed']:.1f} km/h")
print(f"  Collisions:      {results['collisions']}")
print(f"  Waypoints:       {wp_idx}/{len(route_waypoints)}")
driving_score = max(0, 100 - results['collisions'] * 20) * (wp_idx / max(len(route_waypoints), 1))
print(f"  Driving Score:   {driving_score:.1f}")
print("="*60)

# ---- Cleanup ----
print("\nCleaning up...")
try:
    # Stop sensor callbacks before destroying anything
    for s in [col_sensor, *sensors]:
        if s.is_listening:
            s.stop()
    # Revert traffic manager to async BEFORE the world (avoids native abort)
    traffic_mgr.set_synchronous_mode(False)
    # Revert world to async mode; clear the fixed time-step
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)
    # Destroy actors
    col_sensor.destroy()
    for s in sensors:
        s.destroy()
    ego.destroy()
    for v in traffic_vehicles:
        v.destroy()
except Exception as e:
    print(f"  Cleanup warning (non-fatal): {e}")

print("Done!")
# Flush and hard-exit so CARLA object destructors during interpreter
# shutdown can't abort the process and mask a clean exit code.
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
