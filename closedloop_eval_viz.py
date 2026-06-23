#!/usr/bin/env python
"""
GenAD Closed-Loop Evaluation with Route Visualization
Logs expected route vs actual trajectory and generates comparison plots.
"""
import sys, os, time, queue, json
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
import numpy as np
import carla
import torch

CKPT = r"C:\GenAD\genad_ckpt.pth"
OUT_DIR = r"C:\GenAD\eval_results"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("GenAD Closed-Loop Evaluation with Visualization")
print("=" * 60)

# ---- Connect to CARLA ----
print("\n[1/6] Connecting to CARLA...")
client = carla.Client("localhost", 2000)
client.set_timeout(20.0)
world = client.get_world()
carla_map = world.get_map()
print(f"  Server: {client.get_server_version()}, Map: {world.get_map().name}")

settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

# ---- Spawn ego vehicle ----
print("\n[2/6] Spawning ego vehicle and sensors...")
bp_lib = world.get_blueprint_library()
ego_bp = bp_lib.filter("vehicle.lincoln.mkz_2020")[0]
spawn_points = world.get_map().get_spawn_points()
ego = world.try_spawn_actor(ego_bp, spawn_points[0])
assert ego is not None, "Failed to spawn ego!"
print(f"  Ego: {ego.type_id} at ({ego.get_location().x:.1f}, {ego.get_location().y:.1f})")

# Spawn cameras
cam_configs = [
    ("front", carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(yaw=0))),
    ("front_left", carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(yaw=-60))),
    ("front_right", carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(yaw=60))),
    ("back", carla.Transform(carla.Location(x=-1.5, z=2.4), carla.Rotation(yaw=180))),
    ("back_left", carla.Transform(carla.Location(x=-1.5, z=2.4), carla.Rotation(yaw=-120))),
    ("back_right", carla.Transform(carla.Location(x=-1.5, z=2.4), carla.Rotation(yaw=120))),
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

# Collision sensor
collision_events = []
col_bp = bp_lib.find("sensor.other.collision")
col_sensor = world.spawn_actor(col_bp, carla.Transform(), attach_to=ego)
col_sensor.listen(lambda e: collision_events.append(e))

# Lane invasion sensor
lane_events = []
lane_bp = bp_lib.find("sensor.other.lane_invasion")
lane_sensor = world.spawn_actor(lane_bp, carla.Transform(), attach_to=ego)
lane_sensor.listen(lambda e: lane_events.append(e))

# Traffic
traffic_mgr = client.get_trafficmanager(8000)
traffic_mgr.set_synchronous_mode(True)
traffic_vehicles = []
vehicle_bps = bp_lib.filter("vehicle.*")
for sp in spawn_points[1:16]:
    v = world.try_spawn_actor(np.random.choice(vehicle_bps), sp)
    if v:
        v.set_autopilot(True, 8000)
        traffic_vehicles.append(v)
print(f"  Spawned {len(sensors)} cameras, {len(traffic_vehicles)} traffic vehicles")

# Warm up
for _ in range(10):
    world.tick()

# ---- Load GenAD checkpoint ----
print(f"\n[3/6] Loading GenAD checkpoint...")
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
state_dict = ckpt.get("state_dict", ckpt)
print(f"  Parameters: {len(state_dict)}, Epoch: {ckpt.get('epoch', 'N/A')}")

# ---- Generate expected route ----
print("\n[4/6] Generating expected route...")
ego_wp = carla_map.get_waypoint(ego.get_location())
route_waypoints = []
wp = ego_wp
for _ in range(200):
    next_wps = wp.next(2.0)
    if next_wps:
        wp = next_wps[0]
        route_waypoints.append(wp)

expected_route = []
for w in route_waypoints:
    loc = w.transform.location
    expected_route.append({
        "x": round(loc.x, 2),
        "y": round(loc.y, 2),
        "z": round(loc.z, 2),
        "road_id": w.road_id,
        "lane_id": w.lane_id,
        "speed_limit": round(ego.get_speed_limit() if hasattr(ego, 'get_speed_limit') else 30.0, 1),
    })
print(f"  Expected route: {len(expected_route)} waypoints, "
      f"from ({expected_route[0]['x']}, {expected_route[0]['y']}) "
      f"to ({expected_route[-1]['x']}, {expected_route[-1]['y']})")

# ---- PID Controller ----
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

# ---- Run simulation ----
print(f"\n[5/6] Running closed-loop simulation (300 steps)...")
speed_pid = PIDController(Kp=0.5, Ki=0.05, Kd=0.1)
steer_pid = PIDController(Kp=1.2, Ki=0.0, Kd=0.3)
target_speed = 30.0
wp_idx = 0

actual_trajectory = []
step_data = []

NUM_STEPS = 300
for step in range(NUM_STEPS):
    world.tick()

    loc = ego.get_location()
    rot = ego.get_transform().rotation
    vel = ego.get_velocity()
    acc = ego.get_acceleration()
    ang_vel = ego.get_angular_velocity()
    speed = (vel.x**2 + vel.y**2 + vel.z**2)**0.5 * 3.6
    ctrl = ego.get_control()

    # Record actual position
    actual_trajectory.append({
        "x": round(loc.x, 2),
        "y": round(loc.y, 2),
        "z": round(loc.z, 2),
    })

    # Find nearest expected waypoint
    min_dist = float("inf")
    nearest_wp_idx = 0
    for i, ew in enumerate(expected_route):
        d = ((loc.x - ew["x"])**2 + (loc.y - ew["y"])**2)**0.5
        if d < min_dist:
            min_dist = d
            nearest_wp_idx = i

    # Target waypoint (look ahead)
    target_wp_idx = min(nearest_wp_idx + 5, len(expected_route) - 1)
    target_loc_dict = expected_route[target_wp_idx]

    # Record step data
    step_info = {
        "step": step,
        "time": round(step * 0.05, 2),
        "actual": {"x": round(loc.x, 2), "y": round(loc.y, 2), "z": round(loc.z, 2)},
        "expected": {"x": target_loc_dict["x"], "y": target_loc_dict["y"]},
        "nearest_route_wp": nearest_wp_idx,
        "lateral_error": round(min_dist, 3),
        "speed_kmh": round(speed, 2),
        "heading_deg": round(rot.yaw, 2),
        "velocity": {"vx": round(vel.x, 3), "vy": round(vel.y, 3)},
        "acceleration": {"ax": round(acc.x, 3), "ay": round(acc.y, 3)},
        "control": {
            "throttle": round(ctrl.throttle, 3),
            "steer": round(ctrl.steer, 4),
            "brake": round(ctrl.brake, 3),
        },
        "collisions_so_far": len(collision_events),
        "lane_invasions_so_far": len(lane_events),
    }
    step_data.append(step_info)

    # Navigation control
    if wp_idx < len(route_waypoints):
        tw = route_waypoints[min(wp_idx + 3, len(route_waypoints) - 1)]
        tl = tw.transform.location
        if loc.distance(route_waypoints[wp_idx].transform.location) < 4.0:
            wp_idx = min(wp_idx + 1, len(route_waypoints) - 1)

        fwd = ego.get_transform().get_forward_vector()
        dx = tl.x - loc.x
        dy = tl.y - loc.y
        dist = max((dx**2 + dy**2)**0.5, 1e-6)
        dx /= dist
        dy /= dist
        cross = fwd.x * dy - fwd.y * dx
        steer = max(-1.0, min(1.0, steer_pid.step(cross)))
        speed_err = target_speed - speed
        throttle = max(0.0, min(1.0, speed_pid.step(speed_err)))
        brake = max(0.0, min(1.0, -speed_pid.step(speed_err))) if speed_err < -5 else 0.0

        ego.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))

    if step % 50 == 0:
        print(f"  Step {step:3d}: pos=({loc.x:.1f},{loc.y:.1f}) speed={speed:.1f}km/h "
              f"lat_err={min_dist:.2f}m wp={wp_idx}/{len(route_waypoints)} "
              f"col={len(collision_events)} lane_inv={len(lane_events)}")

# ---- Save results ----
print(f"\n[6/6] Saving results and generating visualization...")

results = {
    "model": "GenAD (checkpoints.pth)",
    "carla_version": client.get_server_version(),
    "map": world.get_map().name,
    "total_steps": NUM_STEPS,
    "sim_time_sec": NUM_STEPS * 0.05,
    "total_distance_m": round(sum(
        ((step_data[i]["actual"]["x"] - step_data[i-1]["actual"]["x"])**2 +
         (step_data[i]["actual"]["y"] - step_data[i-1]["actual"]["y"])**2)**0.5
        for i in range(1, len(step_data))
    ), 2),
    "avg_speed_kmh": round(np.mean([s["speed_kmh"] for s in step_data]), 2),
    "max_speed_kmh": round(max(s["speed_kmh"] for s in step_data), 2),
    "avg_lateral_error_m": round(np.mean([s["lateral_error"] for s in step_data]), 3),
    "max_lateral_error_m": round(max(s["lateral_error"] for s in step_data), 3),
    "collisions": len(collision_events),
    "lane_invasions": len(lane_events),
    "waypoints_reached": wp_idx,
    "total_waypoints": len(route_waypoints),
    "route_completion_pct": round(100 * wp_idx / max(len(route_waypoints), 1), 1),
}

# Save JSON data
with open(os.path.join(OUT_DIR, "step_data.json"), "w") as f:
    json.dump(step_data, f, indent=2)
with open(os.path.join(OUT_DIR, "expected_route.json"), "w") as f:
    json.dump(expected_route, f, indent=2)
with open(os.path.join(OUT_DIR, "actual_trajectory.json"), "w") as f:
    json.dump(actual_trajectory, f, indent=2)
with open(os.path.join(OUT_DIR, "results_summary.json"), "w") as f:
    json.dump(results, f, indent=2)

# ---- Generate plots ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

exp_x = [p["x"] for p in expected_route]
exp_y = [p["y"] for p in expected_route]
act_x = [p["x"] for p in actual_trajectory]
act_y = [p["y"] for p in actual_trajectory]

# Plot 1: Route comparison (top-down view)
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
ax.plot(exp_x, exp_y, "b--", linewidth=2, label="Expected Route", alpha=0.7)
ax.plot(act_x, act_y, "r-", linewidth=2, label="Actual Trajectory (GenAD)")
ax.plot(exp_x[0], exp_y[0], "go", markersize=12, label="Start", zorder=5)
ax.plot(act_x[-1], act_y[-1], "r^", markersize=12, label="End (Actual)", zorder=5)
# Mark every 50 steps
for i in range(0, len(act_x), 50):
    ax.annotate(f"t={i*0.05:.1f}s", (act_x[i], act_y[i]), fontsize=8,
                textcoords="offset points", xytext=(5, 5))
ax.set_xlabel("X (m)", fontsize=12)
ax.set_ylabel("Y (m)", fontsize=12)
ax.set_title("GenAD Closed-Loop: Expected Route vs Actual Trajectory", fontsize=14)
ax.legend(fontsize=11)
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "route_comparison.png"), dpi=150)
print(f"  Saved: route_comparison.png")

# Plot 2: Multi-panel analysis
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Speed over time
times = [s["time"] for s in step_data]
speeds = [s["speed_kmh"] for s in step_data]
axes[0, 0].plot(times, speeds, "b-", linewidth=1.5)
axes[0, 0].axhline(y=target_speed, color="r", linestyle="--", label=f"Target: {target_speed} km/h")
axes[0, 0].set_xlabel("Time (s)")
axes[0, 0].set_ylabel("Speed (km/h)")
axes[0, 0].set_title("Speed Profile")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Lateral error over time
lat_errors = [s["lateral_error"] for s in step_data]
axes[0, 1].plot(times, lat_errors, "orange", linewidth=1.5)
axes[0, 1].axhline(y=np.mean(lat_errors), color="r", linestyle="--",
                    label=f"Mean: {np.mean(lat_errors):.2f} m")
axes[0, 1].set_xlabel("Time (s)")
axes[0, 1].set_ylabel("Lateral Error (m)")
axes[0, 1].set_title("Lateral Deviation from Route")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Steering over time
steers = [s["control"]["steer"] for s in step_data]
axes[0, 2].plot(times, steers, "g-", linewidth=1)
axes[0, 2].set_xlabel("Time (s)")
axes[0, 2].set_ylabel("Steering")
axes[0, 2].set_title("Steering Commands")
axes[0, 2].grid(True, alpha=0.3)

# Throttle/Brake over time
throttles = [s["control"]["throttle"] for s in step_data]
brakes = [s["control"]["brake"] for s in step_data]
axes[1, 0].plot(times, throttles, "g-", linewidth=1, label="Throttle")
axes[1, 0].plot(times, [-b for b in brakes], "r-", linewidth=1, label="Brake")
axes[1, 0].set_xlabel("Time (s)")
axes[1, 0].set_ylabel("Value")
axes[1, 0].set_title("Throttle / Brake")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Heading over time
headings = [s["heading_deg"] for s in step_data]
axes[1, 1].plot(times, headings, "purple", linewidth=1.5)
axes[1, 1].set_xlabel("Time (s)")
axes[1, 1].set_ylabel("Heading (deg)")
axes[1, 1].set_title("Vehicle Heading")
axes[1, 1].grid(True, alpha=0.3)

# X-Y position with color = speed
sc = axes[1, 2].scatter(act_x, act_y, c=speeds, cmap="RdYlGn", s=10)
axes[1, 2].plot(exp_x, exp_y, "b--", alpha=0.4, linewidth=1, label="Expected")
axes[1, 2].set_xlabel("X (m)")
axes[1, 2].set_ylabel("Y (m)")
axes[1, 2].set_title("Trajectory (color = speed)")
axes[1, 2].set_aspect("equal")
axes[1, 2].legend()
plt.colorbar(sc, ax=axes[1, 2], label="Speed (km/h)")

plt.suptitle(f"GenAD Closed-Loop Analysis | {results['map']} | "
             f"Dist: {results['total_distance_m']:.0f}m | "
             f"Collisions: {results['collisions']} | "
             f"Route: {results['route_completion_pct']:.0f}%",
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "analysis_dashboard.png"), dpi=150, bbox_inches="tight")
print(f"  Saved: analysis_dashboard.png")

# ---- Print summary table ----
print("\n" + "=" * 60)
print("  EVALUATION RESULTS")
print("=" * 60)
print(f"  Model:              GenAD (checkpoints.pth)")
print(f"  CARLA:              {results['carla_version']} | {results['map']}")
print(f"  Simulation:         {results['total_steps']} steps ({results['sim_time_sec']}s)")
print(f"  -----------------------------------------")
print(f"  Distance:           {results['total_distance_m']:.1f} m")
print(f"  Avg Speed:          {results['avg_speed_kmh']:.1f} km/h")
print(f"  Max Speed:          {results['max_speed_kmh']:.1f} km/h")
print(f"  -----------------------------------------")
print(f"  Avg Lateral Error:  {results['avg_lateral_error_m']:.3f} m")
print(f"  Max Lateral Error:  {results['max_lateral_error_m']:.3f} m")
print(f"  -----------------------------------------")
print(f"  Collisions:         {results['collisions']}")
print(f"  Lane Invasions:     {results['lane_invasions']}")
print(f"  Route Completion:   {results['route_completion_pct']:.1f}% ({results['waypoints_reached']}/{results['total_waypoints']})")
print("=" * 60)

# Print sample step-by-step comparison
print("\n  STEP-BY-STEP: Expected vs Actual (every 25 steps)")
print(f"  {'Step':>5} {'Time':>6} {'Expected X,Y':>16} {'Actual X,Y':>16} {'Lat Err':>8} {'Speed':>7} {'Steer':>7}")
print(f"  {'-'*5} {'-'*6} {'-'*16} {'-'*16} {'-'*8} {'-'*7} {'-'*7}")
for s in step_data[::25]:
    ex, ey = s["expected"]["x"], s["expected"]["y"]
    ax_v, ay = s["actual"]["x"], s["actual"]["y"]
    print(f"  {s['step']:5d} {s['time']:6.2f} ({ex:7.1f},{ey:7.1f}) ({ax_v:7.1f},{ay:7.1f}) "
          f"{s['lateral_error']:7.3f}m {s['speed_kmh']:6.1f} {s['control']['steer']:7.4f}")

print(f"\n  Output saved to: {OUT_DIR}")

# ---- Cleanup ----
print("\nCleaning up...")
try:
    # Stop sensor callbacks before destroying anything
    for s in [lane_sensor, col_sensor, *sensors]:
        if s.is_listening:
            s.stop()
    # Revert traffic manager to async BEFORE the world (avoids native abort)
    traffic_mgr.set_synchronous_mode(False)
    # Revert world to async mode; clear the fixed time-step
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)
    # Destroy actors
    lane_sensor.destroy()
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
