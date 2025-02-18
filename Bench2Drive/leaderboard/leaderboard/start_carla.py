
import traceback
import argparse
from argparse import RawTextHelpFormatter
from distutils.version import LooseVersion
import importlib
import os
import pkg_resources
import sys
import carla
import signal

CARLA_ROOT = os.environ.get("CARLA_ROOT")
Bench2Drive_ROOT = os.environ.get("Bench2Drive_ROOT")

sys.path.append(CARLA_ROOT + "/PythonAPI")
sys.path.append(CARLA_ROOT + "/PythonAPI/carla")
sys.path.append(CARLA_ROOT + "/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg")

sys.path.append(Bench2Drive_ROOT + '/leaderboard')
sys.path.append(Bench2Drive_ROOT + '/leaderboard/pad_team_code')
sys.path.append(Bench2Drive_ROOT + '/scenario_runner')

os.environ["IS_BENCH2DRIVE"] = "True"
os.environ["SCENARIO_RUNNER_ROOT"] = "scenario_runner"
os.environ["LEADERBOARD_ROOT"] = "leaderboard"

from srunner.scenariomanager.carla_data_provider import *
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog

from leaderboard.scenarios.scenario_manager import ScenarioManager
from leaderboard.scenarios.route_scenario import RouteScenario
from leaderboard.envs.sensor_interface import SensorConfigurationInvalid
from leaderboard.autoagents.agent_wrapper import AgentError, validate_sensor_configuration, TickRuntimeError
from leaderboard.utils.statistics_manager import StatisticsManager, FAILURE_MESSAGES
from leaderboard.utils.route_indexer import RouteIndexer
import atexit
import subprocess
import time
import random
from datetime import datetime

carla_path = os.environ["CARLA_ROOT"]
frame_rate=20.0
import socket

def find_free_port(starting_port):
    port = starting_port
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", port))
                return port
        except OSError:
            port += 1


parser = argparse.ArgumentParser()
parser.add_argument('--gpu_rank', default=0,
                    help='IP of the host server (default: localhost)')
arguments = parser.parse_args()
gpu_rank=int(arguments.gpu_rank)
time.sleep(10 * gpu_rank)
port=30000+150*gpu_rank
attempts = 0
num_max_restarts = 20
host="localhost"
while attempts < num_max_restarts:
    try:
        # cmd1 = f"{os.path.join(self.carla_path, 'CarlaUE4.sh')} -RenderOffScreen -nosound -carla-rpc-port={args.port} -graphicsadapter={gpu_rank}"
        cmd1 = f"enroot start --rw --mount {carla_path}:{carla_path} --mount /tmp/.X11-unix:/tmp/.X11-unix carla /bin/bash -c '{os.path.join(carla_path, 'CarlaUE4.sh')} -RenderOffScreen -nosound -carla-rpc-port={port} -graphicsadapter={gpu_rank}'"
        server = subprocess.Popen(cmd1, shell=True, preexec_fn=os.setsid)
        print(cmd1, server.returncode, flush=True)
        atexit.register(os.killpg, server.pid, signal.SIGKILL)
        time.sleep(30)
        print('start')

        client = carla.Client(host, port)
        client.set_timeout(100)
        print('seting', port, host)

        settings = carla.WorldSettings(
            synchronous_mode=True,
            fixed_delta_seconds=1.0 / frame_rate,
            deterministic_ragdolls=True,
            spectator_as_ego=False
        )
        client.get_world().apply_settings(settings)
        print(f"load_world success , attempts={attempts}", flush=True)
        break
    except Exception as e:
        print(f"load_world failed , attempts={attempts}", flush=True)
        print(e, flush=True)
        attempts += 1
        time.sleep(5)

