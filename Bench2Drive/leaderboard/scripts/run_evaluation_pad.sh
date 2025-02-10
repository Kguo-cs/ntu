#!/bin/bash
BASE_PORT=30000
BASE_TM_PORT=50000
IS_BENCH2DRIVE=True
BASE_ROUTES=leaderboard/data/bench2drive220
TEAM_AGENT=leaderboard/pad_team_code/pad_b2d_agent.py
# Must set YOUR_CKPT_PATH
TEAM_CONFIG=leaderboard/pad_team_code/pad_config.py+$1
BASE_CHECKPOINT_ENDPOINT=eval_bench2drive220
PLANNER_TYPE=traj
ALGO=pad
SAVE_PATH=$3

bash leaderboard/scripts/clean_carla.sh $2
pkill -9 -f run_evaluation.sh

PORT=$((BASE_PORT + i * 150))
TM_PORT=$((BASE_TM_PORT + i * 150))
ROUTES="${BASE_ROUTES}_$2_${ALGO}_${PLANNER_TYPE}.xml"
CHECKPOINT_ENDPOINT="$3/${BASE_CHECKPOINT_ENDPOINT}_$2.json"
GPU_RANK=$2
echo -e "\033[32m ALGO: $ALGO \033[0m"
echo -e "\033[32m PLANNER_TYPE: $PLANNER_TYPE \033[0m"
echo -e "\033[32m TASK_ID: $i \033[0m"
echo -e "\033[32m PORT: $PORT \033[0m"
echo -e "\033[32m TM_PORT: $TM_PORT \033[0m"
echo -e "\033[32m CHECKPOINT_ENDPOINT: $CHECKPOINT_ENDPOINT \033[0m"
echo -e "\033[32m GPU_RANK: $GPU_RANK \033[0m"
echo -e "\033[32m bash leaderboard/scripts/run_evaluation.sh $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK \033[0m"
echo -e "***********************************************************************************"
bash -e leaderboard/scripts/run_evaluation.sh $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK  2>&1 > $3/$2.log &
sleep 100