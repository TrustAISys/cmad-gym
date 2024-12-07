from copy import deepcopy

from cmad import MultiCarlaEnv, ENV_ASSETS

configs = {
    "scenarios": [
        {
            "actors": {
                "ego": {
                    "start": {"road_id": 20, "lane_id": 1, "s": 70.0, "random_s": 5},
                    "end": {"road_id": 19, "lane_id": 1, "s": 135.0, "random_s": 5},
                    "ego_end": [126.12, -139.10, 0.5],
                },
                "car1": {
                    "start": {"road_id": 20, "lane_id": 1, "s": 35.0, "random_s": 5},
                    "end": {"road_id": 19, "lane_id": 1, "s": 155.0, "random_s": 5},
                },
                "car2": {
                    "start": {"road_id": 19, "lane_id": -1, "s": 135.0, "random_s": 5},
                    "end": {"road_id": 20, "lane_id": -1, "s": 30.0, "random_s": 5},
                },
                "car3": {
                    "start": {"road_id": 19, "lane_id": -1, "s": 155.0, "random_s": 5},
                    "end": {"road_id": 20, "lane_id": -1, "s": 50.0, "random_s": 5},
                },
                "car4": {
                    "start": {"road_id": 20, "lane_id": 1, "s": 55.0, "random_s": 5},
                },
            },
            "map": "Town05",
            "max_steps": 550,
            "weather_distribution": [0],
        },
        {
            "actors": {
                "ego": {
                    "start": {"road_id": 19, "lane_id": -1, "s": 130.0, "random_s": 5},
                    "end": {"road_id": 20, "lane_id": -1, "s": 75.0, "random_s": 5},
                    "ego_end": [130.85, 126.08, 0.5],
                },
                "car1": {
                    "start": {"road_id": 19, "lane_id": -1, "s": 160.0, "random_s": 5},
                    "end": {"road_id": 20, "lane_id": -1, "s": 55.0, "random_s": 5},
                },
                "car2": {
                    "start": {"road_id": 20, "lane_id": 1, "s": 60.0, "random_s": 5},
                    "end": {"road_id": 19, "lane_id": 1, "s": 175.0, "random_s": 5},
                },
                "car3": {
                    "start": {"road_id": 20, "lane_id": 1, "s": 45.0, "random_s": 5},
                    "end": {"road_id": 19, "lane_id": 1, "s": 140.0, "random_s": 5},
                },
                "car4": {
                    "start": {"road_id": 19, "lane_id": -1, "s": 145.0, "random_s": 5},
                },
            },
            "map": "Town05",
            "max_steps": 550,
            "weather_distribution": [0],
        },
        {
            "actors": {
                "ego": {
                    "start": {"road_id": 20, "lane_id": 1, "s": 205.0, "random_s": 5},
                    "end": {"road_id": 20, "lane_id": 1, "s": 5.0, "random_s": 5},
                    "ego_end": [155.27, -24.64, 0.5],
                },
                "car1": {
                    "start": {"road_id": 20, "lane_id": 1, "s": 165.0, "random_s": 5},
                    "end": {"road_id": 20, "lane_id": 1, "s": 45.0, "random_s": 5},
                },
                "car2": {
                    "start": {"road_id": 20, "lane_id": -1, "s": 10.0, "random_s": 5},
                    "end": {"road_id": 20, "lane_id": -1, "s": 170.0, "random_s": 5},
                },
                "car3": {
                    "start": {"road_id": 20, "lane_id": -1, "s": 30.0, "random_s": 5},
                },
                "car4": {
                    "start": {"road_id": 20, "lane_id": 1, "s": 180.0, "random_s": 5},
                },
            },
            "map": "Town05",
            "max_steps": 550,
            "weather_distribution": [0],
        },
        {
            "actors": {
                "ego": {
                    "start": {"road_id": 19, "lane_id": 1, "s": 205.0, "random_s": 5},
                    "end": {"road_id": 19, "lane_id": 1, "s": 10.0, "random_s": 5},
                    "ego_end": [38.44, -164.53, 0.5],
                },
                "car1": {
                    "start": {"road_id": 19, "lane_id": 1, "s": 175.0, "random_s": 5},
                    "end": {"road_id": 19, "lane_id": 1, "s": 45.0, "random_s": 5},
                },
                "car2": {
                    "start": {"road_id": 19, "lane_id": -1, "s": 10.0, "random_s": 5},
                    "end": {"road_id": 19, "lane_id": -1, "s": 170.0, "random_s": 5},
                },
                "car3": {
                    "start": {"road_id": 19, "lane_id": -1, "s": 30.0, "random_s": 5},
                    "end": {"road_id": 19, "lane_id": -1, "s": 190.0, "random_s": 5},
                },
                "car4": {
                    "start": {"road_id": 19, "lane_id": 1, "s": 190.0, "random_s": 5},
                },
            },
            "map": "Town05",
            "max_steps": 550,
            "weather_distribution": [0],
        },
    ],
    "env": {
        "server_ip": "localhost",
        "server_port": 2000,
        "server_map": "/Game/Carla/Maps/Town05",
        "sync_server": True,
        "fixed_delta_seconds": 0.05,
        "reload": True,
        "record": False,
        "render": True,
        "render_x_res": 800,
        "render_y_res": 600,
        "spectator_loc": [
            47.268875,
            156.649460,
            44.786396,
            -23.775635,
            -103.873795,
            0.0,
        ],
        "obs": {
            "obs_x_res": 168,
            "obs_y_res": 168,
            "framestack": 1,
            "send_measurements": True,
            "measurement_type": ["all"],
            "add_action_mask": True,
        },
        "action": {
            "type": "vehicle_atomic_action",
            "use_discrete": True,
            "discrete_action_set": ENV_ASSETS.default_vehicle_atomic_discrete_actions,
        },
        "global_observation": {
            "camera_type": "rgb",
            "attach_to": "ego",
            "x_res": 512,
            "y_res": 512,
            "camera_position": {
                "x": 0,
                "y": 0,
                "z": 120,
                "pitch": -90,
                "yaw": 0,
                "roll": 0,
            },
            "render": True,
        },
        # "use_redis": True,
        "redis_host": "127.0.0.1",
        "redis_port": 6379,
        "redis_db": 0,
        "verbose": False,
    },
    "actors": {
        "car1": {
            "spawn": True,
            "type": "vehicle_4W",
            "camera_type": "rgb",
            "render": True,
            "manual_control": False,
            "auto_control": False,
            "enable_planner": True,
            "reward_function": "npc",
            "done_criteria": ["timeout", "reach_goal", "collision", "offroad"],
            "lane_sensor": "on",
            "collision_sensor": "on",
            "initial_speed": 20,
            "target_speed": 20,
            "blueprint": "vehicle.tesla.cybertruck",
        },
        "car2": {
            "spawn": True,
            "type": "vehicle_4W",
            "camera_type": "rgb",
            "render": True,
            "manual_control": False,
            "auto_control": False,
            "enable_planner": True,
            "reward_function": "npc",
            "done_criteria": ["timeout", "reach_goal", "collision", "offroad"],
            "lane_sensor": "on",
            "collision_sensor": "on",
            "initial_speed": 20,
            "target_speed": 20,
            "blueprint": "vehicle.tesla.cybertruck",
        },
        "car3": {
            "spawn": True,
            "type": "vehicle_4W",
            "camera_type": "rgb",
            "render": True,
            "manual_control": False,
            "auto_control": False,
            "enable_planner": True,
            "reward_function": "npc",
            "done_criteria": ["timeout", "reach_goal", "collision", "offroad"],
            "lane_sensor": "on",
            "collision_sensor": "on",
            "initial_speed": 20,
            "target_speed": 20,
            "blueprint": "vehicle.tesla.cybertruck",
        },
        "car4": {
            "spawn": True,
            "type": "static_vehicle",
            "blueprint": "vehicle.tesla.cybertruck",
        },
        "ego": {
            "spawn": True,
            "type": "vehicle_4W",
            "rolename": "hero",
            "camera_type": "rgb",
            "render": True,
            "manual_control": False,
            # Toggle manual control for ego to enable a pygame window.
            # You cannot control the ego vehicle with the keyboard.
            "auto_control": False,
            "enable_planner": True,
            "reward_function": "ego",
            "done_criteria": ["timeout", "reach_goal", "collision", "offroad"],
            "action": {"type": "pseudo_action"},
            "lane_sensor": "on",
            "collision_sensor": "on",
            "blueprint": "vehicle.tesla.model3",
        },
    },
}


class Town05Sim(MultiCarlaEnv):
    """Multi-Agent Carla-Gym environment"""

    def __init__(self):
        ENV_ASSETS.step_ticks = 10
        ENV_ASSETS.action_ticks = 1
        super(Town05Sim, self).__init__(deepcopy(configs))


if __name__ == "__main__":
    from cmad.misc import test_run

    env = Town05Sim()
    test_run(env, 5)
