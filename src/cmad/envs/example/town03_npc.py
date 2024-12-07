from copy import deepcopy

from cmad.envs.multi_env import MultiCarlaEnv

configs = {
    "scenarios": [
        {
            "actors": {
                "car1": {
                    "start": [-47.297966, 135.398590, 0.696029],
                    "end": [96.847588, 133.347061, 0.682112],
                },
                "car2": {
                    "start": [54.655403, 129.625046, 0.699949],
                    "end": [-62.608490, 131.693207, 0.653706],
                },
                "car3": {
                    "start": [16.249294, 130.738220, 0.679025],
                    "end": [-116.235016, 132.949692, 0.615929],
                },
                "car4": {
                    "start": [-57.674652, 138.941025, 0.511058],
                    "end": [-57.674652, 138.941025, 0.511058],
                },
                "ego": {
                    "start": [-120.902733, 136.943878, 0.605727],
                    "end": [108.098511, 133.061005, 0.769964],
                },
            },
            "map": "Town03",
            "max_steps": 550,
            "weather_distribution": [0],
        },
        {
            "actors": {
                "car1": {
                    "start": [-32.837158, 134.854843, 0.526696],
                    "end": [96.847588, 133.347061, 0.682112],
                },
                "car2": {
                    "start": [20.029617, 130.685181, 1.012778],
                    "end": [-62.608490, 131.693207, 0.653706],
                },
                "car3": {
                    "start": [-20.055809, 131.134567, 0.549508],
                    "end": [-116.235016, 132.949692, 0.615929],
                },
                "car4": {
                    "start": [-53.825386, 135.572891, 0.617443],
                    "end": [-53.825386, 135.572891, 0.617443],
                },
                "ego": {
                    "start": [-120.902733, 136.943878, 0.605727],
                    "end": [108.098511, 133.061005, 0.769964],
                },
            },
            "map": "Town03",
            "max_steps": 550,
            "weather_distribution": [0],
        },
        {
            "actors": {
                "car1": {
                    "start": [45.001991, 130.133163, 0.596007],
                    "end": [-60.346596, 131.634476, 0.427079],
                },
                "car2": {
                    "start": [-62.123028, 135.664871, 0.593592],
                    "end": [47.052082, 133.959534, 0.567074],
                },
                "car3": {
                    "start": [30.100903, 135.298798, 0.868551],
                    "end": [120.407631, 132.674515, 0.548566],
                },
                "car4": {
                    "start": [88.607559, 129.707520, 0.426985],
                    "end": [88.607559, 129.707520, 0.426985],
                },
                "ego": {
                    "start": [128.412796, 128.966171, 0.525449],
                    "end": [-120.411568, 133.320694, 0.689567],
                    "ego_end": [-144.030014, 114.963707, 0.559743],
                },
            },
            "map": "Town03",
            "max_steps": 550,
            "weather_distribution": [0],
        },
    ],
    "env": {
        "server_ip": "localhost",
        "server_port": 2000,
        "server_map": "/Game/Carla/Maps/Town03",
        "sync_server": False,
        "fixed_delta_seconds": 0.05,
        "reload": True,
        "record": False,
        "render": True,
        "render_x_res": 800,
        "render_y_res": 600,
        "spectator_loc": [-129.206268, 159.879318, 37.855957],
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
                "x": -4.207656,
                "y": 141.529907,
                "z": 140.138519,
                "pitch": -90,
                "yaw": -90,
                "roll": 0,
            },
            "render": True,
        },
        # 'use_redis': True,
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
            "reward_function": "npc",
            "done_criteria": ["timeout", "reach_goal", "collision", "offroad"],
            "collision_sensor": "on",
            "lane_sensor": "on",
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
            "reward_function": "npc",
            "done_criteria": ["timeout", "reach_goal", "collision", "offroad"],
            "collision_sensor": "on",
            "lane_sensor": "on",
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
            "reward_function": "npc",
            "done_criteria": ["timeout", "reach_goal", "collision", "offroad"],
            "collision_sensor": "on",
            "lane_sensor": "on",
            "initial_speed": 25,
            "target_speed": 30,
            "blueprint": "vehicle.lincoln.mkz_2020",
        },
        "car4": {"type": "static_vehicle", "blueprint": "vehicle.tesla.cybertruck"},
        "ego": {
            "type": "vehicle_4W",
            "rolename": "hero",
            "auto_control": False,
            "enable_planner": False,
            "reward_function": "ego",
            "done_criteria": ["timeout", "reach_goal", "collision", "offroad"],
            "action": {"type": "pseudo_action"},
            "collision_sensor": "on",
            "lane_sensor": "on",
            "blueprint": "vehicle.nissan.patrol",
        },
    },
}


class Town03Sim(MultiCarlaEnv):
    """A multi car Multi-Agent Carla-Gym environment"""

    def __init__(self):
        super(Town03Sim, self).__init__(deepcopy(configs))


if __name__ == "__main__":
    from cmad.misc import test_run

    env = Town03Sim()
    test_run(env, 3)
