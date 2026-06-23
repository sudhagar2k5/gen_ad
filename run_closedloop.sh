#!/bin/bash
source /home/sudhagar/miniconda3/etc/profile.d/conda.sh
conda activate b2d_zoo

export CARLA_ROOT=/home/sudhagar/carla
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg

cd /home/sudhagar/Bench2Drive

# Config
BASE_PORT=30000
BASE_TM_PORT=50000
IS_BENCH2DRIVE=True
BASE_ROUTES=leaderboard/data/bench2drive220
TEAM_AGENT=leaderboard/team_code/vad_b2d_agent.py
TEAM_CONFIG=Bench2DriveZoo/adzoo/genad/configs/VAD/GenAD_config_b2d.py+Bench2DriveZoo/ckpts/vad_b2d_base.pth
BASE_CHECKPOINT_ENDPOINT=eval_genad
PLANNER_TYPE=traj
SAVE_PATH=./eval_genad_closedloop/
GPU_RANK=0

# Start CARLA server (headless)
echo "Starting CARLA server on port $BASE_PORT..."
DISPLAY= ${CARLA_SERVER} -RenderOffScreen -nosound -carla-rpc-port=${BASE_PORT} &
CARLA_PID=$!
echo "CARLA PID: $CARLA_PID"
sleep 20

# Run evaluation
PORT=$BASE_PORT
TM_PORT=$BASE_TM_PORT
ROUTES="${BASE_ROUTES}.xml"
CHECKPOINT_ENDPOINT="${BASE_CHECKPOINT_ENDPOINT}.json"

echo "Running closed-loop evaluation..."
bash leaderboard/scripts/run_evaluation.sh \
    $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES \
    $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT \
    $SAVE_PATH $PLANNER_TYPE $GPU_RANK

echo "Stopping CARLA..."
kill $CARLA_PID 2>/dev/null
echo "Done! Results: $CHECKPOINT_ENDPOINT"
