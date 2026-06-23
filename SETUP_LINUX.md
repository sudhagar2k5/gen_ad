# Closed-Loop Setup (Linux)

This repo holds the **GenAD code + Windows demo scripts + the Bench2Drive
closed-loop customizations** (`bench2drive_custom/`). The heavy external
dependencies are NOT in git (size limits) and must be obtained separately.

## External dependencies (clone / download on the Linux machine)

| Dependency | How to get it | Notes |
|---|---|---|
| Bench2Drive | `git clone https://github.com/Thinklab-SJTU/Bench2Drive.git` | leaderboard + scenario_runner |
| Bench2DriveZoo | `git clone https://github.com/Thinklab-SJTU/Bench2DriveZoo.git` (branch `uniad/vad`) | vendored mmcv + configs |
| CARLA 0.9.15 | download from carla.org releases | ~20 GB; needs hardware Vulkan (native Linux GPU) |
| Checkpoints | download `vad_b2d_base.pth`, `resnet50-19c8e357.pth`, `r101_dcn_fcos3d_pretrain.pth` into `Bench2DriveZoo/ckpts/` | >100 MB each, not in git |

## Conda env

Recreate `b2d_zoo` from the upstream requirements:
`Bench2DriveZoo/requirements.txt`, `Bench2Drive/leaderboard/requirements.txt`,
`Bench2Drive/scenario_runner/requirements.txt`.

## Apply the customizations

Copy the files from `bench2drive_custom/` over a fresh Bench2Drive clone:
- `leaderboard/scripts/run_evaluation.sh`  (tweaked)
- `leaderboard/team_code/`                  (vad/uniad agents, pid, planner)
- `leaderboard/data/route_single.xml`       (single Town01 route for a quick test)

Then copy `run_closedloop_vad.sh` into the Bench2Drive root and run:

```bash
cd Bench2Drive
conda activate b2d_zoo
bash run_closedloop_vad.sh        # single Town01 route, VAD model
```

For the full benchmark, set `ROUTES=leaderboard/data/bench2drive220.xml` inside
`run_closedloop_vad.sh`.

## Status / known issue

- The **VAD** model builds + loads (`strict=True`, 58.4M params) and is wired
  for closed-loop. The GenAD nuScenes checkpoint does NOT match the Bench2Drive
  head (no GenAD-b2d checkpoint exists), so VAD is the runnable closed-loop model.
- On **WSL** the run is blocked: CARLA has no hardware Vulkan
  (`ERROR_INCOMPATIBLE_DRIVER`, software `llvmpipe` only). A **native Linux GPU**
  resolves this.
