from copy import deepcopy

from cmad import ENV_ASSETS, MultiCarlaEnv

custom_atomic_action_set = (
    {
        0: "stop",
        1: "lane_follow",
        2: "left_lane_change",
        3: "right_lane_change",
    },
    {
        0: 0,
        1: 4,
        2: 8,
        3: 12,
        4: 16,
    },
)

configs = {
    "scenarios": [
        # {
        #     # Example of OpenScenario scenario
        #     "xosc": "srunner/examples/Town01Sim.xosc",
        #     "actors": {
        #         "ego": {
        #             "end": {"road_id": 15, "lane_id": -1, "s": 260.0},
        #             "ego_end": [-2.0, 322.48, 0.5]
        #         },
        #         "car1": {
        #             "end": {"road_id": 15, "lane_id": -1, "s": 225}
        #         },
        #         "car2": {
        #             "end": {"road_id": 15, "lane_id": 1, "s": 60}
        #         },
        #         "car3": {
        #             "end": {"road_id": 15, "lane_id": 1, "s": 30}
        #         },
        #         "car4": {
        #             "end": {"road_id": 15, "lane_id": -1, "s": 20.0}
        #         }
        #     },
        #     "max_steps": 200,
        #     "random_range": 5,
        # },
        {
            "actors": {
                "ego": {
                    "start": {"road_id": 21, "lane_id": -1, "s": 5.0, "random_s": 5},
                    "end": {"road_id": 24, "lane_id": -1, "s": 83.0, "random_s": 5},
                    "ego_end": [88.78, 315.15, 0.5],
                },
                "car1": {
                    "start": {"road_id": 22, "lane_id": -1, "s": 30.0, "random_s": 5},
                    "end": {"road_id": 24, "lane_id": -1, "s": 63.0, "random_s": 5},
                },
                "car2": {
                    "start": {"road_id": 24, "lane_id": 1, "s": 70.0, "random_s": 5},
                    "end": {"road_id": 22, "lane_id": 1, "s": 25.0, "random_s": 5},
                },
                "car3": {
                    "start": {"road_id": 24, "lane_id": 1, "s": 45.0, "random_s": 5},
                    "end": {"road_id": 21, "lane_id": 1, "s": 28.0, "random_s": 5},
                },
                "car4": {
                    "start": {"road_id": 21, "lane_id": -1, "s": 30.0, "random_s": 5},
                },
            },
            "map": "Town01",
            "max_steps": 200,
            "weather_distribution": [0],
        },
        {
            "actors": {
                "ego": {
                    "start": {"road_id": 16, "lane_id": -1, "s": 5.0, "random_s": 5},
                    "end": {"road_id": 19, "lane_id": -1, "s": 82.5, "random_s": 5},
                    "ego_end": [335.85, 320.0, 0.5],
                },
                "car1": {
                    "start": {"road_id": 17, "lane_id": -1, "s": 15.0, "random_s": 5},
                    "end": {"road_id": 19, "lane_id": -1, "s": 47, "random_s": 5},
                },
                "car2": {
                    "start": {"road_id": 19, "lane_id": 1, "s": 70.0, "random_s": 5},
                    "end": {"road_id": 17, "lane_id": 1, "s": 33.0, "random_s": 5},
                },
                "car3": {
                    "start": {"road_id": 19, "lane_id": 1, "s": 45.0, "random_s": 5},
                    "end": {"road_id": 16, "lane_id": 1, "s": 25, "random_s": 5},
                },
                "car4": {
                    "start": {"road_id": 16, "lane_id": -1, "s": 20.0, "random_s": 5},
                },
            },
            "map": "Town01",
            "max_steps": 200,
            "weather_distribution": [0],
        },
        {
            "actors": {
                "ego": {
                    "start": {"road_id": 15, "lane_id": -1, "s": 5.0, "random_s": 5},
                    "end": {"road_id": 15, "lane_id": -1, "s": 260.0, "random_s": 5},
                    "ego_end": [-2.0, 322.48, 0.5],
                },
                "car1": {
                    "start": {"road_id": 15, "lane_id": -1, "s": 50.0, "random_s": 5},
                    "end": {"road_id": 15, "lane_id": -1, "s": 225, "random_s": 5},
                },
                "car2": {
                    "start": {"road_id": 15, "lane_id": 1, "s": 240.0, "random_s": 5},
                    "end": {"road_id": 15, "lane_id": 1, "s": 60, "random_s": 5},
                },
                "car3": {
                    "start": {"road_id": 15, "lane_id": 1, "s": 200.0, "random_s": 5},
                    "end": {"road_id": 15, "lane_id": 1, "s": 30, "random_s": 5},
                },
                "car4": {
                    "start": {"road_id": 15, "lane_id": -1, "s": 30.0, "random_s": 5},
                },
            },
            "map": "Town01",
            "max_steps": 200,
            "weather_distribution": [0],
        },
        {
            "actors": {
                "ego": {
                    "start": {"road_id": 8, "lane_id": -1, "s": 5.0, "random_s": 5},
                    "end": {"road_id": 8, "lane_id": -1, "s": 260.0, "random_s": 5},
                    "ego_end": [396.63, 6.63, 0.5],
                },
                "car1": {
                    "start": {"road_id": 8, "lane_id": -1, "s": 50.0, "random_s": 5},
                    "end": {"road_id": 8, "lane_id": -1, "s": 240.0, "random_s": 5},
                },
                "car2": {
                    "start": {"road_id": 8, "lane_id": 1, "s": 255.0, "random_s": 5},
                    "end": {"road_id": 8, "lane_id": 1, "s": 60.0, "random_s": 5},
                },
                "car3": {
                    "start": {"road_id": 8, "lane_id": 1, "s": 235.0, "random_s": 5},
                    "end": {"road_id": 8, "lane_id": 1, "s": 20, "random_s": 5},
                },
                "car4": {
                    "start": {"road_id": 8, "lane_id": -1, "s": 30.0, "random_s": 5},
                },
            },
            "map": "Town01",
            "max_steps": 200,
            "weather_distribution": [0],
        },
        {
            "actors": {
                "ego": {
                    "start": {"road_id": 5, "lane_id": -1, "s": 5.0, "random_s": 5},
                    "end": {"road_id": 7, "lane_id": -1, "s": 15, "random_s": 5},
                    "ego_end": [388.66, 330.33, 0.5],
                },
                "car1": {
                    "start": {"road_id": 6, "lane_id": -1, "s": 5.0, "random_s": 5},
                    "end": {"road_id": 6, "lane_id": -1, "s": 215, "random_s": 5},
                },
                "car2": {
                    "start": {"road_id": 7, "lane_id": 1, "s": 15.0, "random_s": 5},
                    "end": {"road_id": 6, "lane_id": 1, "s": 15.0, "random_s": 5},
                },
                "car3": {
                    "start": {"road_id": 6, "lane_id": 1, "s": 215.0, "random_s": 5},
                    "end": {"road_id": 5, "lane_id": 1, "s": 45.0, "random_s": 5},
                },
                "car4": {
                    "start": {"road_id": 5, "lane_id": -1, "s": 40.0, "random_s": 5},
                },
            },
            "map": "Town01",
            "max_steps": 200,
            "weather_distribution": [0],
        },
        {
            "actors": {
                "ego": {
                    "start": {"road_id": 0, "lane_id": -1, "s": 5.0, "random_s": 5},
                    "end": {"road_id": 3, "lane_id": -1, "s": 55.0, "random_s": 5},
                    "ego_end": [8.53, -2.47, 0.5],
                },
                "car1": {
                    "start": {"road_id": 1, "lane_id": -1, "s": 10.0, "random_s": 5},
                    "end": {"road_id": 3, "lane_id": -1, "s": 10.0, "random_s": 5},
                },
                "car2": {
                    "start": {"road_id": 3, "lane_id": 1, "s": 15.0, "random_s": 5},
                    "end": {"road_id": 1, "lane_id": 1, "s": 45.0, "random_s": 5},
                },
                "car3": {
                    "start": {"road_id": 2, "lane_id": 1, "s": 30.0, "random_s": 5},
                    "end": {"road_id": 0, "lane_id": 1, "s": 25.0, "random_s": 5},
                },
                "car4": {
                    "start": {"road_id": 0, "lane_id": -1, "s": 30.0, "random_s": 5},
                },
            },
            "map": "Town01",
            "max_steps": 200,
            "weather_distribution": [0],
        },
    ],
    "env": {
        "server_ip": "localhost",
        "server_port": 2000,
        "server_map": "/Game/Carla/Maps/Town01",
        "sync_server": True,
        "fixed_delta_seconds": 0.05,
        "reload": True,
        "record": False,
        "render": True,
        "render_x_res": 800,
        "render_y_res": 600,
        "spectator_loc": [
            -12.160260,
            20.710268,
            10.342386,
            -11.998654,
            65.999794,
            0.000121,
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
            "discrete_action_set": custom_atomic_action_set,
        },
        "global_observation": {
            "camera_type": "rgb",
            "attach_to": "car2",
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
            "opt_dict": {
                "ignore_traffic_lights": True,
                "ignore_vehicles": False,
                "ignore_stop_signs": True,
                "sampling_resolution": 2.0,
                "base_vehicle_threshold": 5.0,
                "base_tlight_threshold": 5.0,
                "max_brake": 0.5,
            },
            "reward_function": "npc",
            "done_criteria": ["timeout", "reach_goal", "collision", "offroad"],
            "lane_sensor": "on",
            "collision_sensor": "on",
            "initial_speed": 20,
            "target_speed": 20,
            "blueprint": "vehicle.lincoln.mkz_2020",
        },
        "car2": {
            "spawn": True,
            "type": "vehicle_4W",
            "camera_type": "rgb",
            "render": True,
            "manual_control": False,
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
            "reward_function": "npc",
            "done_criteria": ["timeout", "reach_goal", "collision", "offroad"],
            "lane_sensor": "on",
            "collision_sensor": "on",
            "initial_speed": 20,
            "target_speed": 20,
            "blueprint": "vehicle.lincoln.mkz_2020",
        },
        "car3": {
            "spawn": True,
            "type": "vehicle_4W",
            "camera_type": "rgb",
            "render": True,
            "manual_control": False,
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
            "reward_function": "npc",
            "done_criteria": ["timeout", "reach_goal", "collision", "offroad"],
            "lane_sensor": "on",
            "collision_sensor": "on",
            "initial_speed": 20,
            "target_speed": 20,
            "blueprint": "vehicle.lincoln.mkz_2020",
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
            "reward_function": "ego",
            "done_criteria": ["timeout", "reach_goal", "collision", "offroad"],
            "action": {"type": "pseudo_action"},
            "lane_sensor": "on",
            "collision_sensor": "on",
            "blueprint": "vehicle.tesla.model3",
        },
    },
}


class Town01Sim(MultiCarlaEnv):
    """A multi car Multi-Agent Carla-Gym environment"""

    def __init__(self):
        ENV_ASSETS.step_ticks = 10
        ENV_ASSETS.action_ticks = 1
        super(Town01Sim, self).__init__(deepcopy(configs))


if __name__ == "__main__":
    from cmad.misc import test_run

    env = Town01Sim()
    test_run(env, 6)
