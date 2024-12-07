from copy import deepcopy

from cmad import MultiCarlaEnv

configs = {
    "scenarios": [
        {
            "actors": {
                "ego": {
                    "start": [155.034836, -4474.476074, 180.506729],
                    "end": [-139.998123, -4198.502930, 179.019943],
                },
                "car1": {
                    "start": [79.150467, -4403.563477, 181.855484],
                    "end": [-85.907188, -4249.091797, 180.534561],
                },
                "car2": {
                    "start": [-41.688984, -4284.516602, 182.483902],
                    "end": [113.954681, -4430.784180, 181.115463],
                },
                "car3": {
                    "start": [9.966719, -4333.604980, 183.313889],
                    "end": [180.772568, -4492.918457, 180.034256],
                },
                "car4": {
                    "start": [96.249916, -4419.940430, 180.970154],
                    "end": [96.249916, -4419.940430, 180.970154],
                },
            },
            "map": "Town11",
            "max_steps": 650,
            "weather_distribution": [0],
        },
        {
            "actors": {
                "ego": {
                    "start": [166.806168, -4485.056641, 180.367538],
                    "end": [-139.998123, -4198.502930, 179.019943],
                },
                "car1": {
                    "start": [70.493126, -4395.570801, 181.760696],
                    "end": [-85.907188, -4249.091797, 180.534561],
                },
                "car2": {
                    "start": [-55.197186, -4272.839355, 181.785095],
                    "end": [113.954681, -4430.784180, 181.115463],
                },
                "car3": {
                    "start": [1.191797, -4325.604492, 183.038589],
                    "end": [180.772568, -4492.918457, 180.034256],
                },
                "car4": {
                    "start": [96.249916, -4419.940430, 180.970154],
                    "end": [96.249916, -4419.940430, 180.970154],
                },
            },
            "map": "Town11",
            "max_steps": 650,
            "weather_distribution": [0],
        },
        {
            "actors": {
                "ego": {
                    "start": [175.597504, -4493.744629, 180.716873],
                    "end": [-139.998123, -4198.502930, 179.019943],
                },
                "car1": {
                    "start": [61.833904, -4387.397949, 181.990433],
                    "end": [-85.907188, -4249.091797, 180.534561],
                },
                "car2": {
                    "start": [-30.941952, -4295.816406, 182.994720],
                    "end": [113.954681, -4430.784180, 181.115463],
                },
                "car3": {
                    "start": [19.519297, -4342.761230, 182.957581],
                    "end": [180.772568, -4492.918457, 180.034256],
                },
                "car4": {
                    "start": [96.249916, -4419.940430, 180.970154],
                    "end": [96.249916, -4419.940430, 180.970154],
                },
            },
            "map": "Town11",
            "max_steps": 650,
            "weather_distribution": [0],
        },
    ],
    "env": {
        "server_ip": "localhost",
        "server_port": 2000,
        "server_map": "/Game/Carla/Maps/Town11/Town11",
        "sync_server": False,
        "fixed_delta_seconds": 0.05,
        "reload": True,
        "record": False,
        "render": True,
        "render_x_res": 800,
        "render_y_res": 600,
        "spectator_loc": [
            159.006485,
            -4533.056152,
            202.240601,
            -8.594543,
            110.603889,
            0.000002,
        ],
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
            "camera_type": "rgb",
            "x_res": 512,
            "y_res": 512,
            "camera_position": {
                "x": 43.233513,
                "y": -4305.487305,
                "z": 487.095306,
                "pitch": -90,
                "yaw": -90,
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
            "type": "vehicle_4W",
            "camera_type": "rgb",
            "render": False,
            "manual_control": False,
            "auto_control": False,
            "enable_planner": True,
            "collision_sensor": "on",
            "lane_sensor": "on",
            "reward_function": "npc",
            "done_criteria": ["timeout", "reach_goal", "collision", "offroad"],
            "initial_speed": 25,
            "target_speed": 30,
            "blueprint": "vehicle.tesla.model3",
        },
        "car2": {
            "type": "vehicle_4W",
            "camera_type": "rgb",
            "render": False,
            "manual_control": False,
            "auto_control": False,
            "enable_planner": True,
            "collision_sensor": "on",
            "lane_sensor": "on",
            "reward_function": "npc",
            "done_criteria": ["timeout", "reach_goal", "collision", "offroad"],
            "initial_speed": 25,
            "target_speed": 30,
            "blueprint": "vehicle.toyota.prius",
        },
        "car3": {
            "type": "vehicle_4W",
            "camera_type": "rgb",
            "render": False,
            "manual_control": False,
            "auto_control": False,
            "enable_planner": True,
            "collision_sensor": "on",
            "lane_sensor": "on",
            "reward_function": "npc",
            "done_criteria": ["timeout", "reach_goal", "collision", "offroad"],
            "initial_speed": 25,
            "target_speed": 30,
            "blueprint": "vehicle.lincoln.mkz_2020",
        },
        "car4": {"type": "static_vehicle", "blueprint": "vehicle.tesla.cybertruck"},
        "ego": {
            "type": "vehicle_4W",
            "rolename": "hero",
            "camera_type": "rgb",
            "auto_control": False,
            # Toggle manual control for ego to enable a pygame window.
            # You cannot control the ego vehicle with the keyboard.
            "manual_control": False,
            "enable_planner": True,
            "reward_function": "ego",
            "done_criteria": ["timeout", "reach_goal", "collision", "offroad"],
            "action": {"type": "pseudo_action"},
            "collision_sensor": "on",
            "lane_sensor": "on",
            "blueprint": "vehicle.nissan.patrol",
        },
    },
}


class Town11Sim(MultiCarlaEnv):
    """A multi car Multi-Agent Carla-Gym environment"""

    def __init__(self):
        super(Town11Sim, self).__init__(deepcopy(configs))


if __name__ == "__main__":
    from cmad.misc import test_run

    env = Town11Sim()
    test_run(env, 3)
