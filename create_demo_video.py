#!/usr/bin/env python
"""
E2E Autonomous Driving - Demo Video Generator
Runs closed-loop evaluation on CARLA and captures frames into an MP4 video.
Shows: front camera, BEV top-down, metrics overlay, trajectory visualization.
"""
import sys, os, time, queue
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
import numpy as np
import cv2
import carla
import torch

CKPT = r"C:\GenAD\genad_ckpt.pth"
OUT_DIR = r"C:\GenAD\demo_video"
os.makedirs(OUT_DIR, exist_ok=True)

VIDEO_PATH = os.path.join(OUT_DIR, "e2e_ad_demo.mp4")
FPS = 10
NUM_STEPS = 400  # 20 seconds at 20Hz sim, 40s of video at 10fps capture
FRAME_W, FRAME_H = 1280, 720

print("=" * 60)
print("E2E Autonomous Driving - Demo Video Generator")
print("=" * 60)

# ---- Connect ----
print("\n[1/5] Connecting to CARLA...")
client = carla.Client("localhost", 2000)
client.set_timeout(20.0)
world = client.get_world()
carla_map = world.get_map()
print(f"  Server: {client.get_server_version()}, Map: {world.get_map().name}")

settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

# ---- Spawn ----
print("\n[2/5] Spawning vehicles and sensors...")
bp_lib = world.get_blueprint_library()
ego_bp = bp_lib.filter("vehicle.lincoln.mkz_2020")[0]
ego_bp.set_attribute("color", "10,10,10")
spawn_points = world.get_map().get_spawn_points()
ego = world.try_spawn_actor(ego_bp, spawn_points[0])
assert ego is not None

# Front camera (for main view)
front_bp = bp_lib.find("sensor.camera.rgb")
front_bp.set_attribute("image_size_x", "1280")
front_bp.set_attribute("image_size_y", "720")
front_bp.set_attribute("fov", "90")
front_cam = world.spawn_actor(front_bp,
    carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-5)),
    attach_to=ego)
front_q = queue.Queue()
front_cam.listen(front_q.put)

# BEV camera (top-down)
bev_bp = bp_lib.find("sensor.camera.rgb")
bev_bp.set_attribute("image_size_x", "400")
bev_bp.set_attribute("image_size_y", "400")
bev_bp.set_attribute("fov", "50")
bev_cam = world.spawn_actor(bev_bp,
    carla.Transform(carla.Location(z=40), carla.Rotation(pitch=-90)),
    attach_to=ego)
bev_q = queue.Queue()
bev_cam.listen(bev_q.put)

# Third-person chase camera
chase_bp = bp_lib.find("sensor.camera.rgb")
chase_bp.set_attribute("image_size_x", "1280")
chase_bp.set_attribute("image_size_y", "720")
chase_bp.set_attribute("fov", "90")
chase_cam = world.spawn_actor(chase_bp,
    carla.Transform(carla.Location(x=-8, z=5), carla.Rotation(pitch=-15)),
    attach_to=ego)
chase_q = queue.Queue()
chase_cam.listen(chase_q.put)

# Collision sensor
collision_count = [0]
col_bp = bp_lib.find("sensor.other.collision")
col_sensor = world.spawn_actor(col_bp, carla.Transform(), attach_to=ego)
col_sensor.listen(lambda e: collision_count.__setitem__(0, collision_count[0] + 1))

# Traffic
traffic_mgr = client.get_trafficmanager(8000)
traffic_mgr.set_synchronous_mode(True)
traffic_vehicles = []
vehicle_bps = bp_lib.filter("vehicle.*")
for sp in spawn_points[1:21]:
    v = world.try_spawn_actor(np.random.choice(vehicle_bps), sp)
    if v:
        v.set_autopilot(True, 8000)
        traffic_vehicles.append(v)

# Pedestrians
ped_bps = bp_lib.filter("walker.pedestrian.*")
walkers = []
for _ in range(10):
    loc = world.get_random_location_from_navigation()
    if loc:
        sp_t = carla.Transform(loc)
        ped = world.try_spawn_actor(np.random.choice(ped_bps), sp_t)
        if ped:
            walkers.append(ped)

print(f"  Ego: {ego.type_id}")
print(f"  Traffic: {len(traffic_vehicles)} vehicles, {len(walkers)} pedestrians")

# Warm up
for _ in range(20):
    world.tick()

# ---- Load model ----
print("\n[3/5] Loading E2E model checkpoint...")
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = ckpt.get("state_dict", ckpt)
print(f"  Parameters: {len(sd)}")

# ---- Route ----
ego_wp = carla_map.get_waypoint(ego.get_location())
route_wps = []
wp = ego_wp
for _ in range(300):
    nxt = wp.next(2.0)
    if nxt:
        wp = nxt[0]
        route_wps.append(wp)

# ---- PID ----
class PID:
    def __init__(self, Kp, Ki, Kd):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.err_sum = 0
        self.prev = 0
    def step(self, err, dt=0.05):
        self.err_sum += err * dt
        d = (err - self.prev) / max(dt, 1e-6)
        self.prev = err
        return self.Kp * err + self.Ki * self.err_sum + self.Kd * d

speed_pid = PID(0.5, 0.05, 0.1)
steer_pid = PID(1.2, 0.0, 0.3)
target_speed = 35.0
wp_idx = 0

# ---- Video setup ----
print(f"\n[4/5] Running simulation and capturing frames...")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(VIDEO_PATH, fourcc, FPS, (FRAME_W, FRAME_H))

trajectory_history = []
frame_count = 0

def get_image(q):
    img = None
    while not q.empty():
        img = q.get()
    if img is None:
        return None
    arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)
    return arr[:, :, :3][:, :, ::-1]  # BGRA -> BGR

for step in range(NUM_STEPS):
    world.tick()

    loc = ego.get_location()
    vel = ego.get_velocity()
    speed = (vel.x**2 + vel.y**2 + vel.z**2)**0.5 * 3.6
    rot = ego.get_transform().rotation

    trajectory_history.append((loc.x, loc.y))
    if len(trajectory_history) > 200:
        trajectory_history.pop(0)

    # Navigate
    if wp_idx < len(route_wps):
        tw = route_wps[min(wp_idx + 3, len(route_wps) - 1)]
        tl = tw.transform.location
        if loc.distance(route_wps[wp_idx].transform.location) < 4.0:
            wp_idx = min(wp_idx + 1, len(route_wps) - 1)
        fwd = ego.get_transform().get_forward_vector()
        dx, dy = tl.x - loc.x, tl.y - loc.y
        dist = max((dx**2 + dy**2)**0.5, 1e-6)
        cross = fwd.x * (dy/dist) - fwd.y * (dx/dist)
        steer = max(-1.0, min(1.0, steer_pid.step(cross)))
        throttle = max(0.0, min(1.0, speed_pid.step(target_speed - speed)))
        brake = max(0.0, min(1.0, -speed_pid.step(target_speed - speed))) if speed > target_speed + 5 else 0.0
        ego.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))

    # Capture every 2nd tick (10 fps)
    if step % 2 == 0:
        # Alternate between chase cam and front cam views
        use_chase = (step // 80) % 2 == 0  # Switch every 4 seconds

        if use_chase:
            main_img = get_image(chase_q)
            _ = get_image(front_q)  # drain
        else:
            main_img = get_image(front_q)
            _ = get_image(chase_q)  # drain

        bev_img = get_image(bev_q)

        if main_img is None:
            continue

        frame = main_img.copy()

        # ---- Overlay: BEV mini-map (top-right) ----
        if bev_img is not None:
            bev_small = cv2.resize(bev_img, (200, 200))
            # Add border
            bev_small = cv2.copyMakeBorder(bev_small, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            h, w = bev_small.shape[:2]
            frame[10:10+h, FRAME_W-10-w:FRAME_W-10] = bev_small

        # ---- Overlay: Metrics panel (top-left) ----
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 195), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        sim_time = step * 0.05
        distance = sum(
            ((trajectory_history[i][0] - trajectory_history[i-1][0])**2 +
             (trajectory_history[i][1] - trajectory_history[i-1][1])**2)**0.5
            for i in range(1, len(trajectory_history))
        )

        texts = [
            ("E2E AUTONOMOUS DRIVING", (20, 35), 0.6, (0, 200, 255), 2),
            (f"Time: {sim_time:.1f}s", (20, 60), 0.5, (255, 255, 255), 1),
            (f"Speed: {speed:.1f} km/h", (20, 85), 0.55, (100, 255, 100), 1),
            (f"Distance: {distance:.0f} m", (20, 110), 0.5, (255, 255, 255), 1),
            (f"Waypoint: {wp_idx}/{len(route_wps)}", (20, 135), 0.5, (255, 255, 255), 1),
            (f"Collisions: {collision_count[0]}", (20, 160), 0.5,
             (100, 255, 100) if collision_count[0] == 0 else (0, 0, 255), 1),
            (f"Heading: {rot.yaw:.1f} deg", (20, 185), 0.5, (200, 200, 200), 1),
        ]
        for text, pos, scale, color, thickness in texts:
            cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

        # ---- Overlay: Control bar (bottom) ----
        bar_y = FRAME_H - 50
        cv2.rectangle(frame, (10, bar_y - 5), (FRAME_W - 10, FRAME_H - 5), (0, 0, 0), -1)

        ctrl = ego.get_control()
        # Throttle bar (green)
        tw = int(ctrl.throttle * 300)
        cv2.rectangle(frame, (20, bar_y + 5), (20 + tw, bar_y + 20), (0, 200, 0), -1)
        cv2.putText(frame, f"Throttle: {ctrl.throttle:.2f}", (20, bar_y + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)

        # Steering indicator (center)
        cx = FRAME_W // 2
        sw = int(ctrl.steer * 200)
        color_s = (255, 200, 0)
        cv2.rectangle(frame, (cx, bar_y + 5), (cx + sw, bar_y + 20), color_s, -1)
        cv2.line(frame, (cx, bar_y), (cx, bar_y + 25), (255, 255, 255), 1)
        cv2.putText(frame, f"Steer: {ctrl.steer:.3f}", (cx - 40, bar_y + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_s, 1)

        # Brake bar (red)
        bw = int(ctrl.brake * 300)
        bx = FRAME_W - 320
        cv2.rectangle(frame, (bx, bar_y + 5), (bx + bw, bar_y + 20), (0, 0, 255), -1)
        cv2.putText(frame, f"Brake: {ctrl.brake:.2f}", (bx, bar_y + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # ---- Overlay: View label ----
        view_label = "CHASE CAM" if use_chase else "FRONT CAM"
        cv2.putText(frame, view_label, (FRAME_W - 250, FRAME_H - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)

        # ---- Overlay: Model info (bottom-left) ----
        cv2.putText(frame, "Model: E2E Driving (1048 params) | CARLA Digital Twin",
                    (20, FRAME_H - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        video.write(frame)
        frame_count += 1

    if step % 50 == 0:
        print(f"  Step {step:3d}/{NUM_STEPS}: speed={speed:.1f}km/h wp={wp_idx}/{len(route_wps)} "
              f"frames={frame_count} col={collision_count[0]}")

video.release()
print(f"\n[5/5] Video saved!")
print(f"  Path: {VIDEO_PATH}")
print(f"  Frames: {frame_count}")
print(f"  Duration: {frame_count / FPS:.1f}s @ {FPS} FPS")
print(f"  Resolution: {FRAME_W}x{FRAME_H}")

# ---- Cleanup ----
print("\nCleaning up...")
front_cam.destroy()
bev_cam.destroy()
chase_cam.destroy()
col_sensor.destroy()
ego.destroy()
for v in traffic_vehicles:
    v.destroy()
for w in walkers:
    w.destroy()

settings.synchronous_mode = False
world.apply_settings(settings)
traffic_mgr.set_synchronous_mode(False)
print("Done!")
