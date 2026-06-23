#!/usr/bin/env python
"""
End-to-End Autonomous Driving - MLOps Architecture Diagram
Generates comprehensive block diagrams for the closed-loop pipeline.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ============================================================
# DIAGRAM 1: High-Level System Architecture
# ============================================================
def draw_main_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(24, 16))
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 16)
    ax.axis("off")

    # Color scheme
    COLORS = {
        "carla": "#4A90D9",
        "model": "#E74C3C",
        "data": "#27AE60",
        "mlops": "#8E44AD",
        "deploy": "#F39C12",
        "monitor": "#1ABC9C",
        "header": "#2C3E50",
        "bg_light": "#F8F9FA",
        "arrow": "#34495E",
    }

    def draw_box(x, y, w, h, label, sublabel="", color="#4A90D9", fontsize=11):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                             facecolor=color, edgecolor="white", linewidth=2, alpha=0.9)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2 + (0.15 if sublabel else 0), label,
                ha="center", va="center", fontsize=fontsize, fontweight="bold", color="white")
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.25, sublabel,
                    ha="center", va="center", fontsize=8, color="white", alpha=0.9)

    def draw_arrow(x1, y1, x2, y2, label="", color="#34495E", style="->"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=2.5))
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx, my + 0.2, label, ha="center", va="center",
                    fontsize=7.5, color=color, style="italic",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=color, alpha=0.8))

    def draw_section_bg(x, y, w, h, label, color):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2",
                              facecolor=color, edgecolor="grey", linewidth=1, alpha=0.08)
        ax.add_patch(rect)
        ax.text(x + 0.3, y + h - 0.3, label, fontsize=10, fontweight="bold",
                color=color, alpha=0.7)

    # Title
    ax.text(12, 15.5, "End-to-End Autonomous Driving - MLOps Architecture",
            ha="center", va="center", fontsize=18, fontweight="bold", color=COLORS["header"])
    ax.text(12, 15.0, "Closed-Loop Evaluation | Retraining | Digital Twin | CI/CD Pipeline",
            ha="center", va="center", fontsize=11, color="grey")

    # ---- Section backgrounds ----
    draw_section_bg(0.3, 10.2, 7.0, 4.3, "DIGITAL TWIN (CARLA)", COLORS["carla"])
    draw_section_bg(7.8, 10.2, 8.5, 4.3, "E2E DRIVING MODEL", COLORS["model"])
    draw_section_bg(16.8, 10.2, 6.8, 4.3, "DATA PIPELINE", COLORS["data"])
    draw_section_bg(0.3, 5.0, 11.0, 4.7, "MLOps / CI/CD PIPELINE", COLORS["mlops"])
    draw_section_bg(11.8, 5.0, 11.8, 4.7, "MONITORING & FEEDBACK", COLORS["monitor"])
    draw_section_bg(0.3, 0.5, 23.3, 4.0, "DEPLOYMENT & VEHICLE INTEGRATION", COLORS["deploy"])

    # ======== ROW 1: Digital Twin + Model + Data ========

    # CARLA Digital Twin
    draw_box(0.8, 13.0, 2.8, 1.0, "CARLA 0.9.16", "Simulator Engine", COLORS["carla"])
    draw_box(0.8, 11.5, 2.8, 1.0, "Route Manager", "220 Bench2Drive Routes", COLORS["carla"])
    draw_box(4.0, 13.0, 2.8, 1.0, "Scenario Engine", "Weather/Traffic/Events", COLORS["carla"])
    draw_box(4.0, 11.5, 2.8, 1.0, "Sensor Suite", "6 Cameras + LiDAR", COLORS["carla"])

    # E2E Driving Model
    draw_box(8.3, 13.0, 3.5, 1.0, "Image Backbone", "ResNet-50 (318 params)", COLORS["model"])
    draw_box(12.0, 13.0, 3.8, 1.0, "BEV Encoder", "Deformable Attention", COLORS["model"])
    draw_box(8.3, 11.5, 3.5, 1.0, "Trajectory Decoder", "VAE Generator (32 params)", COLORS["model"])
    draw_box(12.0, 11.5, 3.8, 1.0, "Planning Head", "Collision-Aware Planner", COLORS["model"])

    # Data Pipeline
    draw_box(17.3, 13.0, 2.8, 1.0, "Data Lake", "S3/Azure Blob", COLORS["data"])
    draw_box(20.5, 13.0, 2.8, 1.0, "Feature Store", "Embeddings + Labels", COLORS["data"])
    draw_box(17.3, 11.5, 2.8, 1.0, "Data Versioning", "DVC / LakeFS", COLORS["data"])
    draw_box(20.5, 11.5, 2.8, 1.0, "Annotation", "Auto-Label Pipeline", COLORS["data"])

    # ======== ROW 2: MLOps + Monitoring ========

    # MLOps Pipeline
    draw_box(0.8, 8.2, 2.5, 1.0, "Git Repository", "GitHub + DVC", COLORS["mlops"])
    draw_box(3.7, 8.2, 2.5, 1.0, "CI/CD Pipeline", "GitHub Actions", COLORS["mlops"])
    draw_box(6.6, 8.2, 2.5, 1.0, "Training Infra", "GPU Cluster / Cloud", COLORS["mlops"])
    draw_box(0.8, 6.5, 2.5, 1.0, "Experiment Track", "MLflow / W&B", COLORS["mlops"])
    draw_box(3.7, 6.5, 2.5, 1.0, "Model Registry", "Version + Stage", COLORS["mlops"])
    draw_box(6.6, 6.5, 2.5, 1.0, "Validation Gate", "Auto Eval + Approve", COLORS["mlops"])
    draw_box(9.5, 6.5, 1.5, 2.7, "Retrain\nTrigger", COLORS["mlops"], fontsize=9)

    # Monitoring
    draw_box(12.3, 8.2, 2.8, 1.0, "Performance KPIs", "Driving Score/Collision", COLORS["monitor"])
    draw_box(15.5, 8.2, 2.8, 1.0, "Model Drift", "Distribution Monitor", COLORS["monitor"])
    draw_box(18.7, 8.2, 2.8, 1.0, "A/B Testing", "Shadow Mode Eval", COLORS["monitor"])
    draw_box(12.3, 6.5, 2.8, 1.0, "Dashboard", "Grafana / Custom", COLORS["monitor"])
    draw_box(15.5, 6.5, 2.8, 1.0, "Alert System", "Threshold Triggers", COLORS["monitor"])
    draw_box(18.7, 6.5, 2.8, 1.0, "Log Analytics", "ELK / CloudWatch", COLORS["monitor"])
    draw_box(21.9, 6.5, 1.5, 2.7, "Feedback\nLoop", COLORS["monitor"], fontsize=9)

    # ======== ROW 3: Deployment ========
    draw_box(0.8, 2.8, 3.0, 1.0, "Model Optimization", "TensorRT / ONNX", COLORS["deploy"])
    draw_box(4.2, 2.8, 3.0, 1.0, "Edge Deployment", "NVIDIA Orin / Xavier", COLORS["deploy"])
    draw_box(7.6, 2.8, 3.0, 1.0, "Vehicle ECU", "Real-time Inference", COLORS["deploy"])
    draw_box(11.0, 2.8, 3.0, 1.0, "CAN Bus Interface", "Control Commands", COLORS["deploy"])
    draw_box(14.4, 2.8, 3.0, 1.0, "Vehicle Sensors", "Camera/LiDAR/Radar", COLORS["deploy"])
    draw_box(17.8, 2.8, 3.0, 1.0, "Data Recorder", "Drive Logs + Events", COLORS["deploy"])
    draw_box(21.2, 2.8, 2.2, 1.0, "OTA Update", "Fleet Mgmt", COLORS["deploy"])

    draw_box(0.8, 1.0, 5.5, 1.2, "Physical Vehicle Fleet", "Lincoln MKZ / Test Vehicles", "#7F8C8D", fontsize=12)
    draw_box(6.8, 1.0, 5.5, 1.2, "Road Testing", "Geo-fenced ODD Zones", "#7F8C8D", fontsize=12)
    draw_box(12.8, 1.0, 5.5, 1.2, "V2X Communication", "Traffic Infra Integration", "#7F8C8D", fontsize=12)
    draw_box(18.8, 1.0, 4.7, 1.2, "Regulatory Compliance", "Safety Standards", "#7F8C8D", fontsize=12)

    # ======== ARROWS (Data Flow) ========
    # CARLA -> Model
    draw_arrow(6.8, 13.5, 8.3, 13.5, "6x RGB Images", COLORS["carla"])
    draw_arrow(6.8, 12.0, 8.3, 12.0, "Ego State + Route", COLORS["carla"])

    # Model -> CARLA (control)
    draw_arrow(8.3, 11.3, 4.0, 11.3, "Trajectory + Control", COLORS["model"])

    # Model -> Data
    draw_arrow(15.8, 13.5, 17.3, 13.5, "Predictions", COLORS["model"])
    draw_arrow(15.8, 12.0, 17.3, 12.0, "Embeddings", COLORS["model"])

    # Data -> MLOps
    draw_arrow(17.3, 11.3, 9.5, 9.5, "Training Data", COLORS["data"])

    # MLOps internal flow
    draw_arrow(3.3, 8.7, 3.7, 8.7, "", COLORS["mlops"])
    draw_arrow(6.2, 8.7, 6.6, 8.7, "", COLORS["mlops"])
    draw_arrow(3.3, 7.0, 3.7, 7.0, "", COLORS["mlops"])
    draw_arrow(6.2, 7.0, 6.6, 7.0, "", COLORS["mlops"])

    # Validation -> Model (retrain)
    draw_arrow(9.5, 9.5, 10.0, 11.3, "Updated Model", COLORS["mlops"])

    # Monitoring -> Retrain trigger
    draw_arrow(15.5, 6.3, 11.0, 6.5, "Drift Alert", COLORS["monitor"])

    # Feedback loop
    draw_arrow(22.6, 6.5, 22.6, 5.0, "", COLORS["monitor"])
    draw_arrow(22.6, 5.0, 17.8, 3.8, "New Data", COLORS["monitor"])

    # MLOps -> Deployment
    draw_arrow(6.6, 6.3, 3.0, 3.8, "Approved Model", COLORS["mlops"])

    # Deployment flow
    draw_arrow(3.8, 3.3, 4.2, 3.3, "", COLORS["deploy"])
    draw_arrow(7.2, 3.3, 7.6, 3.3, "", COLORS["deploy"])
    draw_arrow(10.6, 3.3, 11.0, 3.3, "", COLORS["deploy"])
    draw_arrow(14.0, 3.3, 14.4, 3.3, "", COLORS["deploy"])
    draw_arrow(17.4, 3.3, 17.8, 3.3, "", COLORS["deploy"])

    # Vehicle data back to data lake
    draw_arrow(20.8, 3.3, 20.8, 11.5, "Drive Logs Upload", COLORS["data"])

    plt.tight_layout()
    plt.savefig(r"C:\GenAD\architecture\01_system_architecture.png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print("  Saved: 01_system_architecture.png")
    plt.close()


# ============================================================
# DIAGRAM 2: Closed-Loop Evaluation Pipeline Detail
# ============================================================
def draw_closedloop_pipeline():
    fig, ax = plt.subplots(1, 1, figsize=(22, 14))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 14)
    ax.axis("off")

    C = {
        "input": "#3498DB",
        "process": "#E74C3C",
        "output": "#27AE60",
        "feedback": "#F39C12",
        "infra": "#8E44AD",
    }

    def box(x, y, w, h, label, sub="", color="#3498DB", fs=10):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                           facecolor=color, edgecolor="white", linewidth=2, alpha=0.9)
        ax.add_patch(b)
        ax.text(x+w/2, y+h/2+(0.12 if sub else 0), label, ha="center", va="center",
                fontsize=fs, fontweight="bold", color="white")
        if sub:
            ax.text(x+w/2, y+h/2-0.2, sub, ha="center", va="center",
                    fontsize=7.5, color="white", alpha=0.9)

    def arr(x1, y1, x2, y2, label="", color="#34495E"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2))
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx, my+0.2, label, ha="center", va="center", fontsize=7,
                    color=color, style="italic",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, alpha=0.8))

    ax.text(11, 13.5, "E2E Autonomous Driving - Closed-Loop Evaluation Pipeline", ha="center",
            fontsize=16, fontweight="bold", color="#2C3E50")

    # ---- Stage 1: CARLA Environment Setup ----
    ax.text(1.5, 12.7, "STAGE 1: Environment Setup", fontsize=11, fontweight="bold", color=C["input"])
    box(0.5, 11.5, 2.5, 0.9, "Load Route", "bench2drive220.xml", C["input"])
    box(3.3, 11.5, 2.5, 0.9, "Init CARLA", "Town/Weather/Traffic", C["input"])
    box(6.1, 11.5, 2.5, 0.9, "Spawn Ego", "Lincoln MKZ 2020", C["input"])
    box(8.9, 11.5, 2.5, 0.9, "Mount Sensors", "6 Cam + LiDAR + IMU", C["input"])
    box(11.7, 11.5, 2.5, 0.9, "Spawn Traffic", "NPC Vehicles + Peds", C["input"])
    box(14.5, 11.5, 2.5, 0.9, "Set Scenario", "Accidents/Crossings", C["input"])

    arr(3.0, 11.95, 3.3, 11.95)
    arr(5.8, 11.95, 6.1, 11.95)
    arr(8.6, 11.95, 8.9, 11.95)
    arr(11.4, 11.95, 11.7, 11.95)
    arr(14.2, 11.95, 14.5, 11.95)

    # ---- Stage 2: Perception (per tick) ----
    ax.text(1.5, 10.7, "STAGE 2: Sensor Data Capture (each tick @ 20Hz)", fontsize=11,
            fontweight="bold", color=C["input"])
    box(0.5, 9.3, 2.0, 1.0, "Front Cam", "1600x900 RGB", C["input"])
    box(2.7, 9.3, 2.0, 1.0, "Front-L/R", "1600x900 RGB", C["input"])
    box(4.9, 9.3, 2.0, 1.0, "Back Cam", "1600x900 RGB", C["input"])
    box(7.1, 9.3, 2.0, 1.0, "Back-L/R", "1600x900 RGB", C["input"])
    box(9.3, 9.3, 2.0, 1.0, "LiDAR", "Point Cloud", C["input"])
    box(11.5, 9.3, 2.5, 1.0, "Vehicle State", "Pos/Vel/Acc/Yaw", C["input"])
    box(14.3, 9.3, 2.5, 1.0, "Route Info", "Waypoints + Cmd", C["input"])

    # ---- Stage 3: E2E Model Inference ----
    ax.text(1.5, 8.3, "STAGE 3: E2E Model Inference", fontsize=11,
            fontweight="bold", color=C["process"])

    box(0.5, 6.5, 3.0, 1.3, "Image Backbone\n(ResNet-50)", "Feature Extraction", C["process"])
    box(3.8, 6.5, 3.0, 1.3, "BEV Transform\n(Deformable Attn)", "2D->3D Projection", C["process"])
    box(7.1, 6.5, 2.5, 1.3, "3D Detection\nHead", "Bbox + Class", C["process"])
    box(9.9, 6.5, 2.5, 1.3, "Map Prediction\nHead", "Lane + Boundary", C["process"])
    box(12.7, 6.5, 2.8, 1.3, "Trajectory VAE\nDecoder", "Multi-modal Traj", C["process"])
    box(15.8, 6.5, 2.8, 1.3, "Planning Head\n(Collision-Aware)", "Optimal Trajectory", C["process"])

    # Arrows from sensors to model
    for x in [1.5, 3.7, 5.9, 8.1]:
        arr(x, 9.3, 2.0, 7.8, "", C["input"])
    arr(10.3, 9.3, 7.5, 7.8, "", C["input"])
    arr(12.75, 9.3, 14.0, 7.8, "", C["input"])
    arr(15.5, 9.3, 17.0, 7.8, "", C["input"])

    # Model internal flow
    arr(3.5, 7.15, 3.8, 7.15)
    arr(6.8, 7.15, 7.1, 7.15)
    arr(9.6, 7.15, 9.9, 7.15)
    arr(12.4, 7.15, 12.7, 7.15)
    arr(15.5, 7.15, 15.8, 7.15)

    # ---- Stage 4: Control Output ----
    ax.text(1.5, 5.7, "STAGE 4: Control & Actuation", fontsize=11,
            fontweight="bold", color=C["output"])

    box(0.5, 4.3, 2.8, 1.0, "Trajectory Select", "Score + Collision Check", C["output"])
    box(3.6, 4.3, 2.5, 1.0, "PID Controller", "Lateral + Longitudinal", C["output"])
    box(6.4, 4.3, 2.5, 1.0, "Control Output", "Throttle/Steer/Brake", C["output"])
    box(9.2, 4.3, 2.5, 1.0, "Apply to Ego", "CARLA Vehicle API", C["output"])
    box(12.0, 4.3, 2.5, 1.0, "World Tick", "Step Simulation", C["output"])

    arr(16.0, 6.5, 1.9, 5.3, "Best Trajectory", C["process"])
    arr(3.3, 4.8, 3.6, 4.8)
    arr(6.1, 4.8, 6.4, 4.8)
    arr(8.9, 4.8, 9.2, 4.8)
    arr(11.7, 4.8, 12.0, 4.8)

    # ---- Stage 5: Evaluation Metrics ----
    ax.text(1.5, 3.5, "STAGE 5: Metrics & Logging", fontsize=11,
            fontweight="bold", color=C["feedback"])

    box(0.5, 2.0, 2.5, 1.0, "Driving Score", "Route x Infractions", C["feedback"])
    box(3.3, 2.0, 2.5, 1.0, "Collision Rate", "Per km / Per route", C["feedback"])
    box(6.1, 2.0, 2.5, 1.0, "Lateral Error", "Avg + Max deviation", C["feedback"])
    box(8.9, 2.0, 2.5, 1.0, "Route Complete", "% Waypoints reached", C["feedback"])
    box(11.7, 2.0, 2.5, 1.0, "Speed Profile", "Avg/Max + Violations", C["feedback"])
    box(14.5, 2.0, 2.5, 1.0, "Lane Invasions", "Count + Duration", C["feedback"])

    # Loop back arrow
    ax.annotate("", xy=(14.5, 12.0), xytext=(17.5, 4.8),
                arrowprops=dict(arrowstyle="->", color=C["feedback"], lw=3,
                                connectionstyle="arc3,rad=-0.3"))
    ax.text(18.5, 8.5, "LOOP\nBACK\n(next tick)", ha="center", fontsize=10,
            fontweight="bold", color=C["feedback"],
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFF3E0", ec=C["feedback"]))

    # Feedback to retrain
    box(17.5, 2.0, 4.0, 1.0, "Log to Data Lake", "Trigger Retrain Pipeline", C["infra"])
    arr(17.0, 2.5, 17.5, 2.5, "", C["feedback"])

    ax.text(19.5, 1.2, "-> MLOps Pipeline\n-> Model Retraining\n-> Digital Twin Re-eval",
            ha="center", fontsize=9, color=C["infra"],
            bbox=dict(boxstyle="round,pad=0.3", fc="#F3E5F5", ec=C["infra"], alpha=0.5))

    plt.tight_layout()
    plt.savefig(r"C:\GenAD\architecture\02_closedloop_pipeline.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    print("  Saved: 02_closedloop_pipeline.png")
    plt.close()


# ============================================================
# DIAGRAM 3: CI/CD + MLOps Pipeline
# ============================================================
def draw_cicd_pipeline():
    fig, ax = plt.subplots(1, 1, figsize=(22, 12))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 12)
    ax.axis("off")

    C = {
        "trigger": "#E74C3C",
        "build": "#3498DB",
        "train": "#8E44AD",
        "eval": "#27AE60",
        "deploy": "#F39C12",
        "gate": "#1ABC9C",
    }

    def box(x, y, w, h, label, sub="", color="#3498DB", fs=10):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                           facecolor=color, edgecolor="white", linewidth=2, alpha=0.9)
        ax.add_patch(b)
        ax.text(x+w/2, y+h/2+(0.12 if sub else 0), label, ha="center", va="center",
                fontsize=fs, fontweight="bold", color="white")
        if sub:
            ax.text(x+w/2, y+h/2-0.2, sub, ha="center", va="center",
                    fontsize=7.5, color="white", alpha=0.9)

    def arr(x1, y1, x2, y2, label="", color="#34495E"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2.5))
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx, my+0.25, label, ha="center", fontsize=7.5, color=color,
                    style="italic", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, alpha=0.8))

    def diamond(x, y, s, label, color):
        pts = np.array([[x, y-s], [x+s, y], [x, y+s], [x-s, y]])
        ax.fill(pts[:,0], pts[:,1], color=color, alpha=0.9)
        ax.plot(pts[[0,1,2,3,0],0], pts[[0,1,2,3,0],1], color="white", lw=2)
        ax.text(x, y, label, ha="center", va="center", fontsize=8, fontweight="bold", color="white")

    ax.text(11, 11.5, "CI/CD + MLOps Pipeline for E2E Autonomous Driving", ha="center",
            fontsize=16, fontweight="bold", color="#2C3E50")
    ax.text(11, 11.0, "Automated Retraining, Validation, and Deployment", ha="center",
            fontsize=11, color="grey")

    # Row 1: Triggers
    ax.text(1.0, 10.2, "TRIGGERS", fontsize=10, fontweight="bold", color=C["trigger"])
    box(0.5, 9.0, 3.0, 0.9, "Git Push / PR", "Code Change", C["trigger"])
    box(3.8, 9.0, 3.0, 0.9, "New Data Batch", "Drive Logs Uploaded", C["trigger"])
    box(7.1, 9.0, 3.0, 0.9, "Model Drift Alert", "KPI Below Threshold", C["trigger"])
    box(10.4, 9.0, 3.0, 0.9, "Scheduled Retrain", "Weekly / Bi-weekly", C["trigger"])

    # Row 2: Build & Test
    ax.text(1.0, 8.0, "BUILD & VALIDATE", fontsize=10, fontweight="bold", color=C["build"])
    box(0.5, 6.7, 2.5, 0.9, "Lint + Format", "flake8 / black", C["build"])
    box(3.3, 6.7, 2.5, 0.9, "Unit Tests", "pytest + coverage", C["build"])
    box(6.1, 6.7, 2.5, 0.9, "Data Validation", "Schema + Quality", C["build"])
    box(8.9, 6.7, 2.5, 0.9, "Docker Build", "GPU Container Image", C["build"])

    arr(2.0, 9.0, 1.75, 7.6, "", C["trigger"])
    arr(5.3, 9.0, 4.55, 7.6, "", C["trigger"])
    arr(3.0, 7.15, 3.3, 7.15)
    arr(5.8, 7.15, 6.1, 7.15)
    arr(8.6, 7.15, 8.9, 7.15)

    # Gate 1
    diamond(12.0, 7.15, 0.5, "Pass?", C["gate"])
    arr(11.4, 7.15, 11.5, 7.15)

    # Row 3: Training
    ax.text(1.0, 5.7, "TRAIN & EXPERIMENT", fontsize=10, fontweight="bold", color=C["train"])
    box(0.5, 4.4, 2.5, 0.9, "Data Prep", "Augment + Split", C["train"])
    box(3.3, 4.4, 2.8, 0.9, "Distributed Train", "Multi-GPU / Cloud", C["train"])
    box(6.4, 4.4, 2.5, 0.9, "Hyperparameter", "Optuna / Ray Tune", C["train"])
    box(9.2, 4.4, 2.5, 0.9, "MLflow Log", "Metrics + Artifacts", C["train"])

    arr(12.0, 6.65, 1.75, 5.3, "Build OK", C["gate"])
    arr(3.0, 4.85, 3.3, 4.85)
    arr(6.1, 4.85, 6.4, 4.85)
    arr(8.9, 4.85, 9.2, 4.85)

    # Gate 2
    diamond(12.5, 4.85, 0.5, "Better?", C["gate"])
    arr(11.7, 4.85, 12.0, 4.85)

    # Row 4: Eval
    ax.text(14.0, 5.7, "DIGITAL TWIN EVALUATION", fontsize=10, fontweight="bold", color=C["eval"])
    box(14.0, 4.4, 2.5, 0.9, "CARLA Eval", "220 Routes / Scenarios", C["eval"])
    box(16.8, 4.4, 2.5, 0.9, "Metrics Compute", "Driving Score + Safety", C["eval"])
    box(19.6, 4.4, 2.0, 0.9, "Compare", "vs Baseline", C["eval"])

    arr(13.0, 4.85, 14.0, 4.85, "Model OK", C["gate"])
    arr(16.5, 4.85, 16.8, 4.85)
    arr(19.3, 4.85, 19.6, 4.85)

    # Gate 3
    diamond(12.5, 2.85, 0.5, "Safe?", C["gate"])
    arr(20.6, 4.4, 12.5, 3.35, "Results", C["eval"])

    # Row 5: Deploy
    ax.text(1.0, 3.2, "DEPLOY", fontsize=10, fontweight="bold", color=C["deploy"])
    box(0.5, 1.8, 2.5, 0.9, "Model Export", "TensorRT / ONNX", C["deploy"])
    box(3.3, 1.8, 2.5, 0.9, "Canary Deploy", "1% Fleet Rollout", C["deploy"])
    box(6.1, 1.8, 2.5, 0.9, "Shadow Mode", "Compare w/ Human", C["deploy"])
    box(8.9, 1.8, 2.5, 0.9, "Full Rollout", "OTA to Fleet", C["deploy"])
    box(11.7, 1.8, 2.5, 0.9, "Monitor", "Production KPIs", C["deploy"])

    arr(12.5, 2.35, 1.75, 2.7, "Approved", C["gate"])
    arr(3.0, 2.25, 3.3, 2.25)
    arr(5.8, 2.25, 6.1, 2.25)
    arr(8.6, 2.25, 8.9, 2.25)
    arr(11.4, 2.25, 11.7, 2.25)

    # Feedback loop
    arr(12.95, 2.25, 14.5, 2.25, "", C["deploy"])
    box(14.5, 1.8, 3.5, 0.9, "Feedback -> Data Lake", "New Training Data", C["trigger"])
    ax.annotate("", xy=(8.6, 9.0), xytext=(16.25, 2.7),
                arrowprops=dict(arrowstyle="->", color=C["trigger"], lw=2.5,
                                connectionstyle="arc3,rad=0.4", linestyle="dashed"))

    # Reject paths
    ax.text(13.0, 7.6, "Fail -> Fix", fontsize=8, color="red", style="italic")
    ax.text(13.2, 5.3, "Worse -> Tune", fontsize=8, color="red", style="italic")
    ax.text(13.2, 2.3, "Unsafe -> Retrain", fontsize=8, color="red", style="italic")

    plt.tight_layout()
    plt.savefig(r"C:\GenAD\architecture\03_cicd_pipeline.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    print("  Saved: 03_cicd_pipeline.png")
    plt.close()


# ============================================================
# Generate all diagrams
# ============================================================
if __name__ == "__main__":
    import os
    os.makedirs(r"C:\GenAD\architecture", exist_ok=True)
    print("Generating architecture diagrams...")
    draw_main_architecture()
    draw_closedloop_pipeline()
    draw_cicd_pipeline()
    print("\nAll diagrams saved to C:\\GenAD\\architecture\\")
