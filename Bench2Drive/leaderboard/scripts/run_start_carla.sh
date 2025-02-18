CUDA_VISIBLE_DEVICES=${GPU_RANK} python leaderboard/leaderboard/start_carla.py --gpu_rank=$1 > $1.log  2>&1 &
