#!/bin/bash
# =============================================================================
# GenAD Closed-Loop Evaluation on CARLA via Bench2Drive
# Run this script INSIDE WSL2 Ubuntu:
#   wsl -d Ubuntu-22.04
#   cd ~/Bench2Drive
#   bash ~/GenAD_scripts/run_closedloop_eval.sh
# =============================================================================

# ---- Paths ----
export CARLA_ROOT=$HOME/carla
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg

# ---- Conda ----
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate b2d_zoo

# ---- Config ----
BASE_PORT=30000
BASE_TM_PORT=50000
IS_BENCH2DRIVE=True
BASE_ROUTES=leaderboard/data/bench2drive220
TEAM_AGENT=leaderboard/team_code/vad_b2d_agent.py
# Update the checkpoint path below after training or downloading a pretrained checkpoint
TEAM_CONFIG=Bench2DriveZoo/adzoo/genad/configs/VAD/GenAD_config_b2d.py+YOUR_CKPT_PATH
BASE_CHECKPOINT_ENDPOINT=eval_genad
PLANNER_TYPE=traj
SAVE_PATH=./eval_genad_closedloop/
GPU_RANK=0

# ---- Step 1: Start CARLA server in background ----
echo "Starting CARLA server..."
DISPLAY= ${CARLA_SERVER} -RenderOffScreen -nosound -carla-rpc-port=${BASE_PORT} &
CARLA_PID=$!
sleep 15  # Wait for CARLA to initialize

# ---- Step 2: Run evaluation ----
PORT=$BASE_PORT
TM_PORT=$BASE_TM_PORT
ROUTES="${BASE_ROUTES}.xml"
CHECKPOINT_ENDPOINT="${BASE_CHECKPOINT_ENDPOINT}.json"

echo "Running GenAD closed-loop evaluation..."
echo "  PORT: $PORT"
echo "  TM_PORT: $TM_PORT"
echo "  GPU: $GPU_RANK"
echo "  ROUTES: $ROUTES"
echo "  CONFIG: $TEAM_CONFIG"

cd $HOME/Bench2Drive

bash leaderboard/scripts/run_evaluation.sh \
    $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES \
    $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT \
    $SAVE_PATH $PLANNER_TYPE $GPU_RANK

# ---- Cleanup ----
echo "Stopping CARLA server..."
kill $CARLA_PID 2>/dev/null
echo "Done! Results saved to: $SAVE_PATH"
echo "Checkpoint endpoint: $CHECKPOINT_ENDPOINT"
