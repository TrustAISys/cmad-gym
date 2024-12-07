from cmad.envs.multi_env import MultiCarlaEnv
from cmad.envs.static_asset import ENV_ASSETS

# These scenarios start right at the junction, suitable for low-level control
ONLY_JUNCTION = [
    # "SSUI3C_TOWN3",   # This scenario has some problem, we recommend to skip it
    {
        "actors": {
            "car1": {
                "start": [-104.015350, -136.909698, 0.574561],
                "end": [-77.895226, -153.209488, 0.500000],
            },
            "car2": {
                "start": [-78.187149, -118.133545, 0.545893],
                "end": [85.9, -149.6, 7.5],
            },
            "car3": {
                "start": [-84.836906, -154.872681, 0.500000],
                "end": [-101.056969, -140.437302, 0.516044],
            },
        },
        "map": "Town03",
        "max_steps": 500,
        "weather_distribution": [0],
    },
    {
        "actors": {
            "car1": {
                "start": [82.334099, -151.948761, 7.5],
                "end": [99.286003, -132.644531, 8.000000],
            },
            "car2": {"start": [65.3, -133.2, 7.5], "end": [85.9, -149.6, 7.5]},
            "car3": {
                "start": [100.815964, -136.112564, 8.500000],
                "end": [65.168404, -136.876236, 7.543454],
            },
        },
        "map": "Town03",
        "max_steps": 500,
        "weather_distribution": [0],
    },
    {
        "actors": {
            "car1": {
                "start": [-6.0551, 115.694, 1.25572],
                "end": [21.368252, 134.385666, 1.113119],
            },
            "car2": {
                "start": [-27.288765, 134.913040, 1.452596],
                "end": [1.986925, 115.451508, 1.108423],
            },
            "car3": {
                "start": [21.999359, 130.259109, 0.786458],
                "end": [-25.747421, 130.731583, 0.683603],
            },
        },
        "map": "Town03",
        "max_steps": 500,
        "weather_distribution": [0],
    },
    {
        "actors": {
            "car1": {"start": [-17.7, -135, 0.5], "end": [7.1, -153.6, 0.5]},
            "car2": {"start": [6.2, -118.7, 0.5], "end": [-15.0, -138.6, 0.5]},
            "car3": {"start": [-0.4, -154.0, 0.5], "end": [-1.2, -120.8, 0.5]},
        },
        "map": "Town03",
        "max_steps": 500,
        "weather_distribution": [0],
    },
]


# These scenarios start some distance before the junction, the agents have to drive to the junction first
EXTEND_JUNCTION = [
    {
        "actors": {
            "car1": {
                "start": {"road_id": 20, "lane_id": 1, "s": 20.0},
                "end": {"road_id": 2, "lane_id": 2, "s": 90.0},
            },
            "car2": {
                "start": {"road_id": 3, "lane_id": 2, "s": 20.0},
                "end": {"road_id": 20, "lane_id": -1, "s": 15.0},
            },
            "car3": {
                "start": {"road_id": 2, "lane_id": -1, "s": 85.0},
                "end": {"road_id": 3, "lane_id": -1, "s": 15.0},
            },
        },
        "map": "Town03",
        "max_steps": 250,
        "weather_distribution": [0],
    },
    {
        "actors": {
            "car1": {
                "start": {"road_id": 19, "lane_id": 1, "s": 15.0},
                "end": {"road_id": 29, "lane_id": -1, "s": 10.0},
            },
            "car2": {
                "start": {"road_id": 28, "lane_id": -1, "s": 60.0},
                "end": {"road_id": 19, "lane_id": -1, "s": 10.0},
            },
            "car3": {
                "start": {"road_id": 29, "lane_id": 2, "s": 10.0},
                "end": {"road_id": 28, "lane_id": 2, "s": 60.0},
            },
        },
        "map": "Town03",
        "max_steps": 250,
        "weather_distribution": [0],
    },
    {
        "actors": {
            "car1": {
                "start": {"road_id": 29, "lane_id": -1, "s": 20.0},
                "end": {"road_id": 2, "lane_id": -1, "s": 10.0},
            },
            "car2": {
                "start": {"road_id": 1, "lane_id": -1, "s": 35.0},
                "end": {"road_id": 29, "lane_id": 2, "s": 25.0},
            },
            "car3": {
                "start": {"road_id": 2, "lane_id": 2, "s": 15.0},
                "end": {"road_id": 1, "lane_id": 2, "s": 35.0},
            },
        },
        "map": "Town03",
        "max_steps": 250,
        "weather_distribution": [0],
    },
    {
        "actors": {
            "car1": {
                "start": {"road_id": 69, "lane_id": -1, "s": 35.0},
                "end": {"road_id": 23, "lane_id": 1, "s": 45.0},
            },
            "car2": {
                "start": {"road_id": 24, "lane_id": 1, "s": 15.0},
                "end": {"road_id": 69, "lane_id": 2, "s": 35.0},
            },
            "car3": {
                "start": {"road_id": 23, "lane_id": -1, "s": 45.0},
                "end": {"road_id": 24, "lane_id": -1, "s": 15.0},
            },
        },
        "map": "Town03",
        "max_steps": 250,
        "weather_distribution": [0],
    },
    {
        "actors": {
            "car1": {
                "start": {"road_id": 31, "lane_id": 4, "s": 5.0},
                "end": {"road_id": 24, "lane_id": 1, "s": 30.0},
            },
            "car2": {
                "start": {"road_id": 76, "lane_id": 1, "s": 20.0},
                "end": {"road_id": 31, "lane_id": -1, "s": 2.0},
            },
            "car3": {
                "start": {"road_id": 24, "lane_id": -1, "s": 30.0},
                "end": {"road_id": 76, "lane_id": -1, "s": 15.0},
            },
        },
        "map": "Town03",
        "max_steps": 250,
        "weather_distribution": [0],
    },
    {
        "actors": {
            "car1": {
                "start": {"road_id": 28, "lane_id": -1, "s": 60.0},
                "end": {"road_id": 19, "lane_id": -1, "s": 15.0},
            },
            "car2": {
                "start": {"road_id": 18, "lane_id": -1, "s": 30.0},
                "end": {"road_id": 28, "lane_id": 2, "s": 60.0},
            },
            "car3": {
                "start": {"road_id": 29, "lane_id": 2, "s": 10.0},
                "end": {"road_id": 28, "lane_id": 2, "s": 60.0},
            },
        },
        "map": "Town03",
        "max_steps": 250,
        "weather_distribution": [0],
    },
    {
        "actors": {
            "car1": {
                "start": {"road_id": 42, "lane_id": 1, "s": 20.0},
                "end": {"road_id": 18, "lane_id": -1, "s": 15.0},
            },
            "car2": {
                "start": {"road_id": 17, "lane_id": -1, "s": 25.0},
                "end": {"road_id": 42, "lane_id": -1, "s": 15.0},
            },
            "car3": {
                "start": {"road_id": 18, "lane_id": 1, "s": 25.0},
                "end": {"road_id": 17, "lane_id": 1, "s": 25.0},
            },
        },
        "map": "Town03",
        "max_steps": 250,
        "weather_distribution": [0],
    },
]


configs = {
    "scenarios": EXTEND_JUNCTION,
    "env": {
        "server_map": "/Game/Carla/Maps/Town03",
        "sync_server": True,
        "fixed_delta_seconds": 0.05,
        "render": True,
        "render_x_res": 800,
        "render_y_res": 600,
        "spectator_loc": [140, 68, 9],
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
            "attach_to": "car2",
            "x_res": 512,
            "y_res": 512,
            "camera_position": {
                "x": 0,
                "y": 0,
                "z": 100,
                "pitch": -90,
                "yaw": 0,
                "roll": 0,
            },
            "render": True,
        },
        "verbose": False,
    },
    "actors": {
        "car1": {
            "type": "vehicle_4W",
            # "camera_type": "rgb",
            "render": False,
            "auto_control": False,
            "enable_planner": True,
            "manual_control": False,
            "reward_function": "custom",
            "done_criteria": ["timeout", "reach_goal", "collision", "offroad"],
            "collision_sensor": "on",
            "lane_sensor": "on",
            "blueprint": "vehicle.tesla.model3",
        },
        "car2": {
            "type": "vehicle_4W",
            # "camera_type": "rgb",
            "render": False,
            "auto_control": False,
            "enable_planner": True,
            "manual_control": False,
            "reward_function": "custom",
            "done_criteria": ["timeout", "reach_goal", "collision", "offroad"],
            "collision_sensor": "on",
            "lane_sensor": "on",
            "blueprint": "vehicle.tesla.model3",
        },
        "car3": {
            "type": "vehicle_4W",
            # "camera_type": "rgb",
            "render": False,
            "auto_control": False,
            "enable_planner": True,
            "manual_control": False,
            "reward_function": "custom",
            "done_criteria": ["timeout", "reach_goal", "collision", "offroad"],
            "collision_sensor": "on",
            "lane_sensor": "on",
            "blueprint": "vehicle.tesla.model3",
        },
    },
}


class StopSign3CarTown03(MultiCarlaEnv):
    """A 4-way signalized intersection Multi-Agent Carla-Gym environment"""

    def __init__(self):
        ENV_ASSETS.step_ticks = 10
        ENV_ASSETS.action_ticks = 1
        super(StopSign3CarTown03, self).__init__(configs)


if __name__ == "__main__":
    from cmad.misc import test_run

    env = StopSign3CarTown03()
    test_run(env, 7)
