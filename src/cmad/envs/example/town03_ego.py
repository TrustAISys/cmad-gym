import random
from copy import deepcopy

import numpy as np
from gym.spaces import Dict
from cmad import MultiCarlaEnv
from cmad.agent.reward import Reward, RewardState

configs = {
    "scenarios": [
        {  # Turn left
            "actors": {
                "ego": {
                    "start": {"road_id": 42, "lane_id": 1, "s": 6.5, "random_s": 2},
                    "end": {"road_id": 18, "lane_id": -1, "s": 6.0, "random_s": 2},
                },
                "car1": {
                    "start": {"road_id": 17, "lane_id": -1, "s": 35, "random_s": 5}
                },
                "car2": {
                    "start": {"road_id": 41, "lane_id": -1, "s": 30, "random_s": 5}
                },
                "car3": {"start": {"road_id": 18, "lane_id": 1, "s": 6, "random_s": 5}},
                "walker1": {"start": [93.123566, -125.217232, 10.0, -180]},
                "walker2": {"start": [74.081169, -145.164047, 10.0, 90]},
            },
            "map": "Town03",
            "max_steps": 300,
            "weather_distribution": [0],
        },
        {  # Straight
            "actors": {
                "ego": {
                    "start": {"road_id": 42, "lane_id": 1, "s": 6.5, "random_s": 2},
                    "end": {"road_id": 41, "lane_id": 1, "s": 28.0, "random_s": 2},
                },
                "car1": {
                    "start": {"road_id": 17, "lane_id": -1, "s": 35, "random_s": 5}
                },
                "car2": {
                    "start": {"road_id": 18, "lane_id": -1, "s": 6.0, "random_s": 2}
                },
                "car3": {
                    "start": {"road_id": 41, "lane_id": -1, "s": 30, "random_s": 5}
                },
                "walker1": {"start": [71.849365, -125.367477, 10.0, 2]},
                "walker2": {"start": [94.984146, -144.989029, 10.0, -180]},
            },
            "map": "Town03",
            "max_steps": 300,
            "weather_distribution": [0],
        },
        {  # Turn right
            "actors": {
                "ego": {
                    "start": {"road_id": 42, "lane_id": 1, "s": 6.5, "random_s": 2},
                    "end": {"road_id": 17, "lane_id": 1, "s": 36.0, "random_s": 2},
                },
                "car1": {
                    "start": {"road_id": 17, "lane_id": -1, "s": 35, "random_s": 5}
                },
                "car2": {
                    "start": {"road_id": 18, "lane_id": -1, "s": 6.0, "random_s": 2}
                },
                "car3": {
                    "start": {"road_id": 41, "lane_id": -1, "s": 30, "random_s": 5}
                },
                "walker1": {"start": [93.214615, -123.411385, 10.0, -90]},
                "walker2": {"start": [93.250069, -145.132080, 10.0, 90]},
            },
            "map": "Town03",
            "max_steps": 300,
            "weather_distribution": [0],
        },
        {  # Turn left
            "actors": {
                "ego": {
                    "start": {"road_id": 17, "lane_id": -1, "s": 38.0, "random_s": 2},
                    "end": {"road_id": 42, "lane_id": -1, "s": 5.0, "random_s": 2},
                },
                "car1": {
                    "start": {"road_id": 42, "lane_id": 1, "s": 6.0, "random_s": 3}
                },
                "car2": {
                    "start": {"road_id": 18, "lane_id": 1, "s": 6.0, "random_s": 2}
                },
                "car3": {
                    "start": {"road_id": 41, "lane_id": -1, "s": 30.0, "random_s": 2}
                },
                "walker1": {"start": [94.145180, -125.150757, 10.0, -180]},
                "walker2": {"start": [93.732811, -146.050751, 10.0, 90]},
            },
            "map": "Town03",
            "max_steps": 300,
            "weather_distribution": [0],
        },
        {  # Straight
            "actors": {
                "ego": {
                    "start": {"road_id": 17, "lane_id": -1, "s": 38.0, "random_s": 2},
                    "end": {"road_id": 18, "lane_id": -1, "s": 6.0, "random_s": 2},
                },
                "car1": {
                    "start": {"road_id": 42, "lane_id": 1, "s": 6.0, "random_s": 3}
                },
                "car2": {
                    "start": {"road_id": 18, "lane_id": 1, "s": 6.0, "random_s": 2}
                },
                "car3": {
                    "start": {"road_id": 41, "lane_id": -1, "s": 30.0, "random_s": 2}
                },
                "walker1": {"start": [73.170181, -123.649193, 10.0, -90]},
                "walker2": {"start": [93.732811, -146.050751, 10.0, 90]},
            },
            "map": "Town03",
            "max_steps": 300,
            "weather_distribution": [0],
        },
        {  # Turn right
            "actors": {
                "ego": {
                    "start": {"road_id": 17, "lane_id": -1, "s": 38.0, "random_s": 2},
                    "end": {"road_id": 41, "lane_id": 1, "s": 28.0, "random_s": 2},
                },
                "car1": {
                    "start": {"road_id": 42, "lane_id": 1, "s": 6.0, "random_s": 3}
                },
                "car2": {
                    "start": {"road_id": 18, "lane_id": 1, "s": 6.0, "random_s": 2}
                },
                "car3": {
                    "start": {"road_id": 41, "lane_id": -1, "s": 30.0, "random_s": 2}
                },
                "walker1": {"start": [93.797920, -145.137634, 10.0, -180]},
                "walker2": {"start": [93.926491, -124.023216, 10.0, -90]},
            },
            "map": "Town03",
            "max_steps": 300,
            "weather_distribution": [0],
        },
    ],
    "env": {
        "server_ip": "localhost",
        "server_port": 2000,
        "server_map": "/Game/Carla/Maps/Town03",
        "sync_server": True,
        "fixed_delta_seconds": 0.05,
        "reload": True,
        "record": False,
        "render": True,
        "spectator_loc": [91, -104, 23, -18, -111, 0],
        "render_x_res": 800,
        "render_y_res": 600,
        "obs": {
            "obs_x_res": 224,
            "obs_y_res": 224,
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
        "done_criteria": [
            "ego_collision",
            "ego_reach_goal",
            "ego_timeout",
        ],
        "global_observation": {
            "camera_type": "bev",
            "attach_to": "ego",
            "x_res": 512,
            "y_res": 512,
            "camera_position": {
                "x": 0,
                "y": 0,
                "z": 50,
                "pitch": -90,
                "yaw": 0,
                "roll": 0,
            },
            "render": True,
        },
        "verbose": False,
    },
    "actors": {
        "ego": {
            "spawn": True,
            "type": "vehicle_4W",
            "camera_type": "bev",
            "manual_control": False,
            "auto_control": False,
            "enable_planner": True,
            "reward_function": "ego_custom",
            "done_criteria": ["timeout", "reach_goal", "collision", "offroad"],
            "lane_sensor": "on",
            "collision_sensor": "on",
            "initial_speed": 0,
            "target_speed": 20,
            "blueprint": "vehicle.tesla.model3",
        },
        "car1": {
            "auto_control": True,
            "action": {"type": "pseudo_action"},
        },
        "car2": {
            "auto_control": True,
            "action": {"type": "pseudo_action"},
        },
        "car3": {
            "auto_control": True,
            "action": {"type": "pseudo_action"},
        },
        "walker1": {
            "type": "walker",
            "action": {"type": "walker_speed_action"},
        },
        "walker2": {
            "type": "walker",
            "action": {"type": "walker_speed_action"},
        },
    },
}


@Reward.reward_signature
def compute_custom_reward(actor_id: str, reward_state: RewardState, **kwargs):
    step_reward = 0.0
    actor_cache = reward_state.cache[actor_id]
    is_active = actor_cache["active_state"]
    curr_m = reward_state.current_measurements[actor_id]
    prev_m = reward_state.prev_measurements[actor_id]

    if is_active:
        # Lane keeping reward
        cur_dist = curr_m.exp_info.distance_to_goal
        prev_dist = prev_m.exp_info.distance_to_goal
        step_reward += np.clip(prev_dist - cur_dist, -10.0, 10.0)

        # Collision penalty
        collision_reward = 0
        if curr_m.collision.diff(prev_m.collision) > 0:
            collision_reward -= 100.0
        step_reward += collision_reward

        # New sidewalk intersection
        step_reward -= 2 * (curr_m.lane_invasion.offroad - prev_m.lane_invasion.offroad)

        # New opposite lane intersection
        step_reward -= 2 * (
            curr_m.lane_invasion.otherlane - prev_m.lane_invasion.otherlane
        )

    actor_cache["active_state"] = not curr_m.exp_info.done
    return step_reward


class TrafficNearEgo(MultiCarlaEnv):
    """Single-Agent Carla environment with traffic near ego vehicle."""

    def __init__(self, ego_only: bool = True, semantic_only: bool = False):
        Reward.register_reward("ego_custom", compute_custom_reward)
        if ego_only:
            obs_config = configs["env"]["obs"]
            obs_config["focus_actors"] = ["ego"]
            obs_config["measurement_type"] = ["heading", "speed", "waypoints"]
        super(TrafficNearEgo, self).__init__(deepcopy(configs))

        self.semantic_only = semantic_only
        self.original_space = self.observation_space
        if semantic_only:
            new_space = {}
            for actor in self.actor_configs:
                orig_space = self.original_space[actor].spaces.copy()
                orig_space.pop("camera", None)
                new_space[actor] = Dict(orig_space)

            self.observation_space = Dict(new_space)

    def reset(self) -> dict:
        # random speed for the walker this episode
        self.walker1_speed = random.randint(1, 3)
        self.walker2_speed = random.randint(1, 3)
        obs = super().reset()

        if self.semantic_only:
            for actor in self.actor_configs:
                obs[actor].pop("camera", None)
        return obs

    def step(self, action):
        try:
            action["walker1"] = self.walker1_speed
            action["walker2"] = self.walker2_speed
            obs, reward, done, info = super().step(action)
            done["__all__"] = done["ego"]
        except RuntimeError:
            obs = self.observation_space.sample()
            reward = {actor: 0 for actor in self.actor_configs}
            done = {actor: True for actor in self.actor_configs}
            info = {actor: {} for actor in self.actor_configs}
            done["__all__"] = True

        if self.semantic_only:
            for actor in self.actor_configs:
                obs[actor].pop("camera", None)
        return obs, reward, done, info


if __name__ == "__main__":
    import argparse

    from cmad.misc import SingleAgentWrapper, test_run

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--single_agent", action="store_true")
    argparser.add_argument("--num_epoch", type=int, default=10)
    args = argparser.parse_args()

    env = TrafficNearEgo(semantic_only=True)
    if args.single_agent:
        env = SingleAgentWrapper(env)
        for _ in range(args.num_epoch):
            obs = env.reset()
            done = False
            while not done:
                obs, reward, done, info = env.step(env.action_space.sample())
        env.close()
    else:
        test_run(env, args.num_epoch)
