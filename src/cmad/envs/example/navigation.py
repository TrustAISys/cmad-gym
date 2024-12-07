import os
import random
from collections import deque
from copy import deepcopy

import carla

from cmad import SYS_ASSETS, MultiCarlaEnv, Reward, RewardState

BASE_CONFIG = {
    "scenarios": [],
    "env": {
        "server_ip": "localhost",
        "server_port": 2000,
        "server_map": "/Game/Carla/Maps/Town01",
        "reload": True,
        "spectator_loc": None,
        "sync_server": False,
        "fixed_delta_seconds": 0.05,
        "render": True,
        "render_x_res": 800,
        "render_y_res": 600,
        "obs": {
            "obs_x_res": 168,
            "obs_y_res": 168,
            "framestack": 1,
            "use_depth_camera": False,
            "send_measurements": True,
            "measurement_type": ["all"],
        },
        "action": {
            "type": "low_level_action",
            "action_range": {
                "throttle": [0, 1.0],
                "brake": [-1.0, 0],
                "steer": [-0.5, 0.5],
            },
            "use_discrete": True,
        },
        "global_observation": {
            "camera_type": "semseg",
            "x_res": 512,
            "y_res": 512,
            "camera_position": {
                "x": 201.897842,
                "y": 169.607925,
                "z": 240.176193,
                "pitch": -90,
                "yaw": -90,
                "roll": 0,
            },
            "render": True,
        },
        "verbose": False,
    },
    "actors": {},
}


def gen_pairs(
    map: carla.Map, num: int, distance_lower_bound: float, distance_upper_bound: float
):
    """Generate random start and end points for navigation.

    Args:
        map (carla.Map): The map object.
        num (int): The number of pairs to generate.
        distance_lower_bound (float): The lower bound of the distance.
        distance_upper_bound (float): The upper bound of the distance.

    Returns:
        list: A list of start and end points.
    """
    pairs = []
    waypoints = map.generate_waypoints(2.0)

    max_tries = 1000
    while len(pairs) < num and max_tries > 0:
        start = random.choice(waypoints)
        end = random.choice(waypoints)
        distance = start.transform.location.distance(end.transform.location)
        # Check if the distance is around the given limit (within 10% margin)
        if distance_lower_bound <= distance <= distance_upper_bound:
            pairs.append(
                (
                    [
                        start.transform.location.x,
                        start.transform.location.y,
                        start.transform.location.z + 0.5,
                    ],
                    [
                        end.transform.location.x,
                        end.transform.location.y,
                        end.transform.location.z + 0.5,
                    ],
                )
            )

            # remove from the spawn list so we don"t get the same pair again
            waypoints.remove(start)
            waypoints.remove(end)

        max_tries -= 1

    return pairs


def update_actor_configs(base_config: dict, num_agent: int):
    actor_configs = {}
    for i in range(1, num_agent + 1):
        actor_configs[f"car{i}"] = {
            "spawn": True,
            "type": "vehicle_4W",
            "blueprint": "vehicle.tesla.model3",
            "reward_function": "custom",
            "done_criteria": ["timeout", "reach_goal", "collision", "offroad"],
            "collision_sensor": "on",
            "lane_sensor": "on",
            # Camera can drop the FPS significantly, set to None and False to disable
            "camera_type": "rgb",
            "render": True,
            # Enable autocontrol for debug only
            "auto_control": False,
            "enable_planner": True,
            "opt_dict": {
                "ignore_traffic_lights": True,
                "ignore_vehicles": False,
                "ignore_stop_signs": True,
                "sampling_resolution": 2.0,
                "base_vehicle_threshold": 5.0,
                "base_tlight_threshold": 5.0,
                "max_brake": 0.5,
            },
            "initial_speed": 20,
            "target_speed": 20,
        }
    base_config["actors"] = actor_configs


def update_scenario_configs(
    base_config: dict, map_name: str, pairs: list, num_agents: int, num_scenarios: int
):
    scenarios = []

    for _ in range(num_scenarios):
        tmp_pairs = deepcopy(pairs)
        scenario = {
            "actors": {},
            "map": map_name,
            "max_steps": 550,
            "weather_distribution": [0],
        }
        for i in range(num_agents + 1):
            pair = random.choice(tmp_pairs)
            scenario["actors"][f"car{i}"] = {"start": pair[0], "end": pair[1]}
            tmp_pairs.remove(pair)
        scenarios.append(scenario)

    base_config["scenarios"] = scenarios


def gen_config(
    map_name: str,
    num_agent: int = 2,
    num_scenarios: int = 1,
    distance_lower_bound: float = 100,
    distance_upper_bound: float = None,
):
    """Generate a random navigation config.

    Args:
        map_name (str): The map name.
        num_agent (int): The number of vehicles.
        num_scenarios (int): The number of scenarios.
        distance_lower_bound (float): The lower bound of the distance.
        distance_upper_bound (float): The upper bound of the distance. If None, it will be set double the lower bound.
    """
    xodr_path = os.path.join(
        SYS_ASSETS.paths.directory,
        "CarlaUE4/Content/Carla/Maps/OpenDrive",
        map_name + ".xodr",
    )
    if not os.path.exists(xodr_path):
        raise ValueError(f"Map {map_name} does not exist.")

    with open(xodr_path, "r") as f:
        map = carla.Map(map_name, f.read())

    if distance_upper_bound is None:
        distance_upper_bound = distance_lower_bound * 2

    possible_start_ends = gen_pairs(
        map, num_agent * num_scenarios, distance_lower_bound, distance_upper_bound
    )
    base_config = deepcopy(BASE_CONFIG)
    base_config["env"]["server_map"] = f"/Game/Carla/Maps/{map_name}"
    update_actor_configs(base_config, num_agent)
    update_scenario_configs(
        base_config, map_name, possible_start_ends, num_agent, num_scenarios
    )
    return base_config


@Reward.reward_signature
def steer_reward(actor_id: str, reward_state: RewardState):
    """A reward function that only cares about steering"""
    reward = 0.0
    is_active = reward_state.cache[actor_id]["active_state"]
    curr_m = reward_state.current_measurements[actor_id]

    if is_active:
        # If the vehicle kept switching from left to right, then there will be penalty
        if curr_m.control is not None:
            if reward_state.cache[actor_id].get("steer_cache", None) is None:
                reward_state.cache[actor_id]["steer_cache"] = deque(maxlen=3)

            # append 1, 0, -1 according to the steering angle
            steering = curr_m.control.get("steer", 0)
            reward_state.cache[actor_id]["steer_cache"].append(steering)

        steer_sign = all(
            steer <= 0 for steer in reward_state.cache[actor_id]["steer_cache"]
        ) or all(steer >= 0 for steer in reward_state.cache[actor_id]["steer_cache"])
        steer_reward = -0.3 if not steer_sign else 0
        reward += steer_reward

        # Potential functions for waypoint
        waypoint_reward = 0
        distance_to_goal = curr_m.exp_info.distance_to_goal

        if "distance_to_goal_cache" not in reward_state.cache[actor_id]:
            reward_state.cache[actor_id]["distance_to_goal_cache"] = distance_to_goal
            reward_state.cache[actor_id][
                "min_distance_to_goal_cache"
            ] = distance_to_goal

        if (
            distance_to_goal
            < reward_state.cache[actor_id]["min_distance_to_goal_cache"]
        ):
            waypoint_reward += (
                200
                * (
                    reward_state.cache[actor_id]["min_distance_to_goal_cache"]
                    - distance_to_goal
                )
                / reward_state.cache[actor_id]["distance_to_goal_cache"]
            )
            reward_state.cache[actor_id][
                "min_distance_to_goal_cache"
            ] = distance_to_goal
        reward += waypoint_reward

    reward_state.cache[actor_id]["active_state"] = not curr_m.exp_info.done
    return reward


class Navigation(MultiCarlaEnv):
    def __init__(
        self,
        map_name="Town01",
        num_agent=2,
        num_scenarios=1,
        distance_lower_bound=100,
        distance_upper_bound=None,
    ):
        Reward.register_reward("custom", steer_reward)
        configs = gen_config(
            map_name,
            num_agent,
            num_scenarios,
            distance_lower_bound,
            distance_upper_bound,
        )
        super(Navigation, self).__init__(configs)


if __name__ == "__main__":
    from cmad.misc import test_run

    env = Navigation("Town01", 4, 5, 100, 300)
    test_run(env, 5)
