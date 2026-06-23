#!/bin/bash
# Real closed-loop CARLA evaluation with the VAD Bench2Drive model.
# The leaderboard evaluator self-starts CARLA (see leaderboard_evaluator.py),
# so we do NOT launch CARLA here (avoids a second instance / OOM).
source /home/sudhagar/miniconda3/etc/profile.d/conda.sh
conda activate b2d_zoo

export CARLA_ROOT=/home/sudhagar/carla
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg

cd /home/sudhagar/Bench2Drive

echo "--- free RAM before run ---"
free -h | head -2

PORT=30000
TM_PORT=50000
IS_BENCH2DRIVE=True
ROUTES=leaderboard/data/route_single.xml
TEAM_AGENT=leaderboard/team_code/vad_b2d_agent.py
TEAM_CONFIG=Bench2DriveZoo/adzoo/vad/configs/VAD/VAD_base_e2e_b2d.py+Bench2DriveZoo/ckpts/vad_b2d_base.pth
CHECKPOINT_ENDPOINT=eval_vad.json
SAVE_PATH=./eval_vad_closedloop/
PLANNER_TYPE=traj
GPU_RANK=0

echo "Running VAD closed-loop evaluation (single route, Town01)..."
bash leaderboard/scripts/run_evaluation.sh \
    $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES \
    $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT \
    $SAVE_PATH $PLANNER_TYPE $GPU_RANK

echo "Cleaning up any stray CARLA..."
pkill -9 -f CarlaUE4 2>/dev/null
echo "Done! Results: $CHECKPOINT_ENDPOINT"
