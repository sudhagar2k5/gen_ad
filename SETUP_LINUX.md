# Closed-Loop Setup (Linux)

This repo holds the **GenAD code + Windows demo scripts + the Bench2Drive
closed-loop customizations** (`bench2drive_custom/`) + a one-shot driver script
(`run_full_evaluation.sh`). The heavy external dependencies (CARLA, checkpoints,
the upstream repos) are NOT in git and are fetched separately — mostly automated.

## Quick start (new PC)

```bash
# 0. prerequisites (once): miniconda, git, vulkan-tools, NVIDIA driver
sudo apt install -y git vulkan-tools
nvidia-smi                                   # confirm GPU driver

# 1. clone this repo
git clone https://github.com/sudhagar2k5/gen_ad.git
cd gen_ad && chmod +x run_full_evaluation.sh

# 2. bootstrap: clone Bench2Drive(+Zoo), build conda env, overlay our
#    customizations + launchers, then run the preflight checks
./run_full_evaluation.sh --clone-repos --setup-env --check-only

# 3. print the two downloads it cannot auto-fetch (CARLA + checkpoints)
./run_full_evaluation.sh --deps-help
#    -> download CARLA 0.9.15 to ~/carla
#    -> download vad_b2d_base.pth + resnet50-19c8e357.pth to ~/Bench2DriveZoo/ckpts/

# 4. re-check (should be all green on a native-Linux NVIDIA GPU)
./run_full_evaluation.sh --check-only

# 5. run the evaluation
cd ~/Bench2Drive
./run_full_evaluation.sh            # single Town01 route (quick)
./run_full_evaluation.sh --full     # all 220 routes
```

Default paths are `$HOME`-based (`~/Bench2Drive`, `~/Bench2DriveZoo`, `~/carla`,
`~/miniconda3`). Override any of them with env vars, e.g.:
`CARLA_ROOT=/opt/carla BENCH2DRIVE=/data/Bench2Drive ./run_full_evaluation.sh ...`

## run_full_evaluation.sh

| Flag | Effect |
|---|---|
| (none) | preflight checks, then run single Town01 route |
| `--check-only` | only the preflight checks (no run) |
| `--full` | run all 220 routes |
| `--model genad` | use GenAD config+ckpt instead of VAD (needs matching ckpt) |
| `--clone-repos` | `git clone` Bench2Drive + Bench2DriveZoo (`uniad/vad`), overlay customizations |
| `--setup-env` | `conda env create` from `env/b2d_zoo_environment.yml` |
| `--deps-help` | print CARLA + checkpoint download steps |
| `--force` | run even if a check FAILs |

The preflight verifies: conda env, python/torch/mmcv/carla, GPU, **Vulkan**,
CARLA install, repos, checkpoints, route files, and RAM/disk.

## External dependencies (what `--clone-repos` / `--deps-help` handle)

| Dependency | How | Auto? |
|---|---|---|
| Bench2Drive | `git clone https://github.com/Thinklab-SJTU/Bench2Drive.git` | ✅ `--clone-repos` |
| Bench2DriveZoo | `git clone -b uniad/vad https://github.com/Thinklab-SJTU/Bench2DriveZoo.git` | ✅ `--clone-repos` |
| conda env `b2d_zoo` | `conda env create -f env/b2d_zoo_environment.yml` | ✅ `--setup-env` |
| customizations | overlay `bench2drive_custom/` onto Bench2Drive | ✅ `--clone-repos` |
| CARLA 0.9.15 | download + extract to `~/carla` (~20 GB) | ❌ manual (`--deps-help`) |
| Checkpoints | `vad_b2d_base.pth`, `resnet50-19c8e357.pth` (+ `GenAD/checkpoints.pth` for genad) → `Bench2DriveZoo/ckpts/` | ❌ manual (`--deps-help`) |

## Environment details

`env/` holds the exact `b2d_zoo` spec:
- `b2d_zoo_environment.yml` / `b2d_zoo_environment_nobuilds.yml` (conda)
- `b2d_zoo_pip_freeze.txt` (pip)

Key versions: **Python 3.8.20, torch 2.4.1+cu118, carla 0.9.15, numpy 1.20.0,
nuscenes-devkit 1.1.10**. `mmcv`/`mmdet3d` are **vendored** inside
`Bench2DriveZoo/mmcv/` (not pip packages).

## Status / known issue

- The **VAD** model builds + loads (`strict=True`, 58.4M params) and is wired
  for closed-loop. The GenAD nuScenes checkpoint does NOT match the Bench2Drive
  head (no GenAD-b2d checkpoint exists), so VAD is the runnable closed-loop model.
- On **WSL** the run is blocked: CARLA has no hardware Vulkan
  (`ERROR_INCOMPATIBLE_DRIVER`, software `llvmpipe` only). A **native Linux GPU**
  resolves this — the preflight's Vulkan check will then pass.
