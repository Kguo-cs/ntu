
lyuchen@aspire2pntu.nscc.sg
Lyuchen2018!
qsub -I -l select=1:ngpus=1 -l walltime=24:00:00 -P 12002486
ssh ke.guo@aspire2antu.nscc.sg

sudo apt install gcc g++ -y
export no_proxy=localhost,127.0.0.1,10.104.0.0/21
export https_proxy=http://10.104.4.124:10104
export http_proxy=http://10.104.4.124:10104
pip install -U openmim
mim install mmcv-full==1.7.2

enroot create --name carla /home/users/ntu/lyuchen/scratch/keguo_projects/ntu/exp/carla2.sqsh
enroot start --mount /home/users/ntu/lyuchen/scratch:/home/users/ntu/lyuchen/scratch  --mount /home/users/ntu/lyuchen/miniconda3:/home/users/ntu/lyuchen/miniconda3 -w carla bash
enroot start --rw --mount /home/users/ntu/lyuchen/scratch/keguo_projects/ntu/exp/carla:/home/users/ntu/lyuchen/scratch/keguo_projects/ntu/exp/carla --mount /tmp/.X11-unix:/tmp/.X11-unix carla /bin/bash -c '/home/users/ntu/lyuchen/scratch/keguo_projects/ntu/exp/carla/CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=20004 -graphicsadapter=4'

enroot create --name carla /home/users/ntu/lyuchen/scratch/keguo_projects/ntu/exp/carla2.sqsh
source "/home/users/ntu/lyuchen/miniconda3/bin/activate"
cd /home/users/ntu/lyuchen/scratch/keguo_projects/ntu
conda activate pad

source "/home/users/ntu/lyuchen/miniconda3/bin/activate"
cd /home/users/ntu/lyuchen/scratch/keguo_projects/ntu/sim
conda activate catk

python -m torch.distributed.run --nproc_per_node=8 navsim/planning/script/run_b2d_training.py > B2d32_acc_cuda121.log 2>&1 & tail -f B2d32_acc_cuda121.log

bash leaderboard/scripts/run_evaluation_pad.sh /home/users/ntu/lyuchen/scratch/keguo_projects/ntu/exp/ke/B2d32_acclocal0_2000_d0_pip_cuda122_qsub/02.15_12.22/lightning_logs/version_0/checkpoints/epoch=0-step=768.ckpt 3 /home/users/ntu/lyuchen/scratch/keguo_projects/ntu/exp/ke/B2d32_acclocal0_2000_d0_pip_cuda122_qsub/02.15_12.22/res_epoch=0-step=768


qstat -ans
export PBS_JOBID=29877.pbs111  
echo "$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg" >> /home/users/ntu/lyuchen/miniconda3/envs/pad/lib/python3.8/site-packages/carla.pth # python 3.8 also works well, please set YOUR_CONDA_PATH and YOUR_CONDA_ENV_NAME

sh cuda_12.1.1_530.30.02_linux.run  --toolkitpath=/home/users/ntu/lyuchen/scratch/keguo_projects/cuda --toolkit --silent

# Install
```bash
conda create -n pad python=3.8
conda activate pad
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install -e ./Bench2DriveZoo
pip install -e ./nuplan-devkit
pip install -e .
```

# Set environment variable
set the environment variable based on where you place the PAD directory. 
```bash
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$HOME/pad_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="$HOME/pad_workspace/exp"
export NAVSIM_DEVKIT_ROOT="$HOME/pad_workspace/navsim"
export OPENSCENE_DATA_ROOT="$HOME/pad_workspace/dataset"
export Bench2Drive_ROOT="$HOME/pad_workspace/Bench2Drive"
```


# Navsim
1. download the navtrain dataset and map as [Navsim](https://github.com/autonomousvision/navsim) 
```bash
bash download/download_maps.sh
bash download/download_navtrain.sh
bash download/download_navtest.sh
```
Put the downloaded maps in "dataset/maps", and dataset in "dataset/navsim_logs" and "dataset/sensor_blobs" 

2. cache training data and metric
```bash
python navsim/planning/script/run_training_metric_caching.py
python navsim/planning/script/run_dataset_caching.py
```
3. train navsim model
```bash
python navsim/planing/script/run_training.py
```
4. test navsim model

Change the checkpoint path in [agent_config](navsim/planning/script/config/common/agent/navsim_agent.yaml)
```bash
python navsim/planing/script/run_create_submission_pickle.py
```
Then, submit the created "submission.pkl" to the [official leaderboard](https://huggingface.co/spaces/AGC2024-P/e2e-driving-navsim) on HuggingFace.


# Bench2drive
1. download the base dataset as [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive) 
```bash
huggingface-cli download --repo-type dataset --resume-download rethinklab/Bench2Drive --local-dir Bench2Drive-Base
```
2. download and setup CARLA 0.9.15
```bash
    mkdir carla
    cd carla
    wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz
    tar -xvf CARLA_0.9.15.tar.gz
    cd Import && wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_0.9.15.tar.gz
    cd .. && bash ImportAssets.sh
    export CARLA_ROOT=YOUR_CARLA_PATH
    echo "$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg" >> YOUR_CONDA_PATH/envs/YOUR_CONDA_ENV_NAME/lib/python3.7/site-packages/carla.pth # python 3.8 also works well, please set YOUR_CONDA_PATH and YOUR_CONDA_ENV_NAME
```
3. [Prepare Dataset](Bench2DriveZoo/docs/DATA_PREP.md)

4. cache training data and metric
```bash
python Bench2Drive/leaderboard/pad_team_code/b2d_datacache.py
python Bench2Drive/leaderboard/pad_team_code/gen_mapinfo.py
```

5. train Bench2drive model
```bash
python -m torch.distributed.run --nproc_per_node=8 navsim/planning/script/run_b2d_training.py 
# python navsim/planning/script/run_b2d_training.py
```

6. closeloop evaluation
```bash
cd Bench2Drive
python leaderboard/leaderboard/pad_eval.py
python tools/merge_route_json.py
python tools/ability_benchmark.py
python tools/efficiency_smoothness_benchmark.py
```