import logging
import os
import sys

from gym.envs.registration import register

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "yes please"
LOG_DIR = os.path.join(os.getcwd(), "logs")
if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)

# Init and setup the root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(module)s %(funcName)s: "%(message)s"',
    datefmt="%d-%m-%y %H:%M:%S",
    handlers=[
        logging.FileHandler(f"{LOG_DIR}/cmad-gym.log", "w"),
        logging.StreamHandler(sys.stdout),
    ],
)

# Fix path issues with included CARLA API
sys.path.append(os.path.join(os.path.dirname(__file__), "carla/PythonAPI/carla"))

# Declare available environments with a brief description
_AVAILABLE_ENVS = {
    "HomoNcomIndePOIntrxMASS3CTWN3-v0": {
        "entry_point": "cmad.envs.macad:HomoNcomIndePOIntrxMASS3CTWN3",
        "description": "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0",
    },
    "HeteNcomIndePOIntrxMATLS1B2C1PTWN3-v0": {
        "entry_point": "cmad.envs.macad:HeteNcomIndePOIntrxMATLS1B2C1PTWN3",
        "description": "Heterogeneous, Non-communicating, Independent,"
        "Partially-Observable Intersection Multi-Agent"
        " scenario with Traffic-Light Signal, 1-Bike, 2-Car,"
        "1-Pedestrian in Town3, version 0",
    },
    "Town01-v0": {
        "entry_point": "cmad.envs.example:Town01Sim",
        "description": "Homogeneous, Non-communicating, Cooperative, Partially-"
        "Observable Navigation scenario with Ego in Town01"
        "version 0",
    },
    "Town03-v0": {
        "entry_point": "cmad.envs.example:Town03Sim",
        "description": "Homogeneous, Non-communicating, Cooperative, Partially-"
        "Observable Navigation scenario with Ego in Town03"
        "version 0",
    },
    "Town05-v0": {
        "entry_point": "cmad.envs.example:Town05Sim",
        "description": "Homogeneous, Non-communicating, Cooperative, Partially-"
        "Observable Navigation scenario with Ego in Town05"
        "version 0",
    },
}

for env_id, val in _AVAILABLE_ENVS.items():
    register(id=env_id, entry_point=val.get("entry_point"))


def list_available_envs():
    print("Environment-ID: Short-description")
    import pprint

    available_envs = {}
    for env_id, val in _AVAILABLE_ENVS.items():
        available_envs[env_id] = val.get("description")
    pprint.pprint(available_envs)


# Expose necessary classes and methods
from cmad.agent.action import AbstractAction, ActionInterface
from cmad.agent.done import Done
from cmad.agent.reward import Reward, RewardLogger, RewardState
from cmad.envs.multi_env import MultiCarlaEnv
from cmad.envs.static_asset import ENV_ASSETS, SYS_ASSETS
from cmad.simulation.data.simulator import Simulator

__all__ = [
    "AbstractAction",
    "ActionInterface",
    "ENV_ASSETS",
    "SYS_ASSETS",
    "LOG_DIR",
    "Reward",
    "RewardLogger",
    "RewardState",
    "Done",
    "MultiCarlaEnv",
    "Simulator",
    "list_available_envs",
]
