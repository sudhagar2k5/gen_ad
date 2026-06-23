import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(18, 10))
ax.set_xlim(0, 18)
ax.set_ylim(0, 10)
ax.axis("off")

def box(x, y, w, h, title, items, color, fs_title=13, fs_items=9):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                       facecolor=color, edgecolor="white", linewidth=2.5, alpha=0.92)
    ax.add_patch(b)
    ax.text(x + w/2, y + h - 0.35, title, ha="center", va="center",
            fontsize=fs_title, fontweight="bold", color="white")
    for i, item in enumerate(items):
        ax.text(x + w/2, y + h - 0.75 - i*0.3, item, ha="center", va="center",
                fontsize=fs_items, color="white", alpha=0.9)

def arrow(x1, y1, x2, y2, label="", color="#333", lw=2.5, style="-"):
    ls = style if style != "-" else "-"
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                linestyle="dashed" if style == "--" else "solid"))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my + 0.22, label, ha="center", va="center", fontsize=8,
                color=color, style="italic",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, alpha=0.85, lw=0.8))

# Title
ax.text(9, 9.6, "E2E Autonomous Driving - Closed-Loop MLOps Pipeline",
        ha="center", fontsize=17, fontweight="bold", color="#212121")

# Section labels
ax.text(6.5, 8.9, "CLOSED-LOOP SIMULATION", ha="center", fontsize=11,
        fontweight="bold", color="#1565C0", alpha=0.7)
ax.text(6.5, 4.65, "CI/CD & MLOps PIPELINE", ha="center", fontsize=11,
        fontweight="bold", color="#6A1B9A", alpha=0.7)

# ---- Row 1: Simulation Loop ----
box(0.3, 6.5, 2.8, 2.2, "Digital Twin", ["(CARLA Simulator)", "", "6x RGB Cameras", "GPS / IMU / Speed", "Route & Scenarios", "Traffic & Weather"], "#1565C0")
box(4.0, 6.5, 3.2, 2.2, "E2E Driving Model", ["", "Image Backbone (CNN)", "BEV Encoder", "3D Detection + Map", "Trajectory VAE", "Collision-Aware Planner"], "#C62828")
box(8.1, 6.5, 2.8, 2.2, "Vehicle Control", ["", "PID Controller", "", "Throttle: 0.0-0.75", "Steering: -1.0 to +1.0", "Brake: 0.0-1.0"], "#1B5E20")
box(11.8, 6.5, 2.8, 2.2, "Evaluation", ["Metrics", "", "Driving Score", "Collisions", "Route Completion", "Lateral Error"], "#E65100")

# Row 1 arrows
arrow(3.1, 7.6, 4.0, 7.6, "Sensor Data", "#1565C0")
arrow(7.2, 7.6, 8.1, 7.6, "Trajectory", "#C62828")
arrow(10.9, 7.6, 11.8, 7.6, "Actions", "#1B5E20")

# Closed-loop feedback
ax.annotate("", xy=(1.7, 6.5), xytext=(9.5, 6.5),
            arrowprops=dict(arrowstyle="-|>", color="#1B5E20", lw=2,
                            linestyle="dashed", connectionstyle="arc3,rad=0.35"))
ax.text(5.6, 5.7, "Control Commands (Closed Loop)", ha="center", fontsize=8,
        color="#1B5E20", style="italic")

# ---- Row 2: MLOps Pipeline ----
box(0.3, 2.2, 2.8, 2.0, "Data Lake", ["", "Drive Logs Storage", "Versioned Datasets", "Auto Annotation"], "#2E7D32")
box(4.0, 2.2, 3.2, 2.0, "Model Training", ["", "Distributed GPU", "Experiment Tracking", "Hyperparameter Tuning"], "#6A1B9A")
box(8.1, 2.2, 2.8, 2.0, "Digital Twin", ["Validation", "", "220 Route Evaluation", "Compare vs Baseline", "Safety Gate"], "#00695C")
box(11.8, 2.2, 2.8, 2.0, "Fleet", ["Deployment", "", "Model Optimization", "OTA Update", "Canary Rollout"], "#E65100")

# Row 2 arrows
arrow(3.1, 3.2, 4.0, 3.2, "Training Data", "#2E7D32")
arrow(7.2, 3.2, 8.1, 3.2, "Trained Model", "#6A1B9A")
arrow(10.9, 3.2, 11.8, 3.2, "Approved", "#00695C")

# ---- Vertical connections ----
# Metrics -> Data Lake
ax.annotate("", xy=(1.7, 4.2), xytext=(13.2, 6.5),
            arrowprops=dict(arrowstyle="-|>", color="#E65100", lw=2, linestyle="dashed",
                            connectionstyle="arc3,rad=0.3"))
ax.text(8.5, 5.15, "Logged Data", ha="center", fontsize=8, color="#E65100", style="italic")

# Validation -> Model (update)
ax.annotate("", xy=(5.6, 8.7), xytext=(9.5, 4.2),
            arrowprops=dict(arrowstyle="-|>", color="#00695C", lw=2, linestyle="dashed",
                            connectionstyle="arc3,rad=-0.3"))
ax.text(8.8, 6.1, "Updated Model", ha="center", fontsize=8, color="#00695C", style="italic")

# Deploy -> Data Lake (feedback)
ax.annotate("", xy=(1.7, 2.2), xytext=(13.2, 2.2),
            arrowprops=dict(arrowstyle="-|>", color="#C62828", lw=3, linestyle="dashed"))
ax.text(7.5, 1.7, "Fleet Drive Logs (Continuous Feedback Loop)", ha="center",
        fontsize=9, fontweight="bold", color="#C62828",
        bbox=dict(boxstyle="round,pad=0.2", fc="#FFEBEE", ec="#C62828", lw=1))

# Monitoring box
box(15.2, 4.5, 2.5, 2.0, "Monitoring", ["", "KPI Dashboard", "Drift Detection", "Alert System", "Retrain Trigger"], "#37474F")
ax.annotate("", xy=(15.2, 5.5), xytext=(14.6, 7.0),
            arrowprops=dict(arrowstyle="-|>", color="#37474F", lw=1.5, linestyle="dashed"))
ax.annotate("", xy=(15.2, 4.8), xytext=(14.6, 3.2),
            arrowprops=dict(arrowstyle="-|>", color="#37474F", lw=1.5, linestyle="dashed"))
ax.annotate("", xy=(5.6, 2.0), xytext=(15.2, 4.5),
            arrowprops=dict(arrowstyle="-|>", color="#C62828", lw=1.5, linestyle="dashed",
                            connectionstyle="arc3,rad=0.3"))
ax.text(11.5, 1.15, "Drift Alert -> Retrain", ha="center", fontsize=8, color="#C62828", style="italic")

plt.tight_layout()
plt.savefig(r"C:\GenAD\architecture\04_minimal_architecture.png", dpi=150,
            bbox_inches="tight", facecolor="white")
print("Saved: 04_minimal_architecture.png")
