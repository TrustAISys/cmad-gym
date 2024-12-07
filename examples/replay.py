"""
Author: Morphlng
Date: 2023-07-10 10:28:33
LastEditTime: 2023-09-04 10:33:31
LastEditors: Morphlng
Description: Demonstration of how to use the replay function
"""
from copy import deepcopy

from cmad.envs.multi_env import MultiCarlaEnv

configs = {
    "scenarios": [
        {
            "actors": {
                # The start and end position is not important during replay
                # We will update the position to the one in the log file
                "ego": {
                    "start": [-2.077557, 37.381592, 0.306523],
                    "end": [-2.153992, 250.168594, 0.341896]
                    # ego_end: [-2.128490, 317.256927, 0.556630]
                },
                "car1": {
                    "start": [-2.005636, 71.842804, 0.445798],
                    "end": [-1.913923, 194.361740, 0.524820],
                },
                "car2": {
                    "start": [2.075526, 236.833450, 0.401698],
                    "end": [2.075225, 120.532562, 0.527723],
                },
                "car3": {
                    "start": [2.009022, 221.231964, 0.351630],
                    "end": [2.080578, 96.304008, 0.411035],
                },
                "car4": {
                    "start": [-2.075735, 63.077972, 0.577694],
                    "end": [-2.075735, 63.077972, 0.577694],
                },
            },
            "map": "Town01",
            "max_steps": 550,
            "weather_distribution": [0],
        },
    ],
    "env": {
        "server_ip": "localhost",
        "server_port": 2000,
        "server_map": "/Game/Carla/Maps/Town01",
        "reload": True,
        "sync_server": False,
        "fixed_delta_seconds": 0.05,
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
            "use_depth_camera": False,
            "send_measurements": True,
            "measurement_type": ["all"],
        },
        "global_observation": {
            "camera_type": "rgb",
            "x_res": 512,
            "y_res": 512,
            "camera_position": {
                "x": -2.971099,
                "y": 161.967834,
                "z": 118.703270,
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
            "type": "vehicle_4W",
            "rolename": "hero",
            "camera_type": "rgb",
            "render": True,
            "auto_control": False,
            # Toggle manual control for ego to enable a pygame window.
            # You cannot control the ego vehicle with the keyboard.
            "manual_control": False,
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
            "scenarios": "TOWN01_EGO",
            "reward_function": "ego",
            "early_terminate_on_collision": True,
            "collision_sensor": "off",
            "lane_sensor": "on",
            "use_depth_camera": False,
            "blueprint": "vehicle.tesla.model3",
        },
        "car1": {
            "auto_control": True,
            "camera_type": "rgb",
            "render": True,
            "collision_sensor": "off",
            "convert_images_to_video": False,
            "early_terminate_on_collision": True,
            "enable_planner": True,
            "lane_sensor": "on",
            "manual_control": False,
            "reward_function": "npc",
            "scenarios": "TOWN01_CAR1",
            "type": "vehicle_4W",
            "use_depth_camera": False,
            "initial_speed": 20,
            "target_speed": 20,
            "blueprint": "vehicle.tesla.model3",
        },
        "car2": {
            "auto_control": True,
            "camera_type": "rgb",
            "render": True,
            "collision_sensor": "off",
            "convert_images_to_video": False,
            "early_terminate_on_collision": True,
            "enable_planner": True,
            "lane_sensor": "on",
            "manual_control": False,
            "reward_function": "npc",
            "scenarios": "TOWN01_CAR2",
            "type": "vehicle_4W",
            "use_depth_camera": False,
            "initial_speed": 20,
            "target_speed": 20,
            "blueprint": "vehicle.toyota.prius",
        },
        "car3": {
            "auto_control": True,
            "camera_type": "rgb",
            "render": True,
            "collision_sensor": "off",
            "convert_images_to_video": False,
            "early_terminate_on_collision": True,
            "enable_planner": True,
            "lane_sensor": "on",
            "manual_control": False,
            "reward_function": "npc",
            "scenarios": "TOWN01_CAR3",
            "type": "vehicle_4W",
            "use_depth_camera": False,
            "initial_speed": 20,
            "target_speed": 20,
            "blueprint": "vehicle.lincoln.mkz_2020",
        },
        "car4": {
            "type": "static_vehicle",
            "scenarios": "TOWN11_CAR4",
            "blueprint": "vehicle.tesla.cybertruck",
        },
    },
}


class ReplayEnv(MultiCarlaEnv):
    """A multi car Multi-Agent Carla-Gym environment"""

    def __init__(self):
        super(ReplayEnv, self).__init__(deepcopy(configs))


if __name__ == "__main__":
    import os

    env = ReplayEnv()

    replay_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "./replays"))
    replay_files = [
        os.path.join(replay_root, f)
        for f in os.listdir(replay_root)
        if f.endswith(".dat")
    ]

    for replay in replay_files:
        obs, steps = env.replay(replay, follow_vehicle="ego")

        i = 0
        for ep in range(steps + 1):
            i += 1
            action_dict = env.action_space.sample()
            obs, reward, done, info = env.step(action_dict, True)

            for actor_id in info:
                print(f"{actor_id}'s action at step {i}: {info[actor_id]['control']}")
                print(f"{actor_id}'s reward at step {i}: {reward[actor_id]}")
                print(f"{actor_id}'s distance to goal: {info[actor_id]['exp_info']['distance_to_goal']}")

            if done["__all__"]:
                print("All done, episode end...")
                break

        env._clean_world()

    env.close()
