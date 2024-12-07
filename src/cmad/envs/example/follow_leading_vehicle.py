from cmad.envs.multi_env import MultiCarlaEnv

configs = {
    "scenarios": {
        "map": "Town01",
        "actors": {
            "car1": {
                "start": [107, 133, 0.5],
                "end": [300, 133, 0.5],
            },
            "car2": {
                "start": [115, 133, 0.5],
                "end": [310, 133, 0.5],
            },
        },
        "num_vehicles": 0,
        "num_pedestrians": 0,
        "weather_distribution": [0],
        "max_steps": 500,
    },
    "env": {
        "server_map": "/Game/Carla/Maps/Town01",
        "fixed_delta_seconds": 0.05,
        "sync_server": True,
        "render": True,
        "render_x_res": 800,  # For both Carla-Server and Manual-Control
        "render_y_res": 600,
        "spectator_loc": [100, 133, 9],
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
        "verbose": False,
    },
    "actors": {
        "car1": {
            "type": "vehicle_4W",
            "camera_type": "rgb",
            "render": True,
            # When "auto_control" is True,
            # starts the actor using auto-pilot.
            # Allows manual control take-over on
            # pressing Key `p` on the PyGame window
            # if manual_control is also True
            "manual_control": True,
            "auto_control": False,
            "enable_planner": True,
            "reward_function": "corl2017",
            "done_criteria": ["timeout", "reach_goal", "collision", "offroad"],
            "collision_sensor": "on",
            "lane_sensor": "on",
        },
        "car2": {
            "type": "vehicle_4W",
            "camera_type": "rgb",
            "render": True,
            "manual_control": False,
            "auto_control": True,
            "enable_planner": True,
            "reward_function": "corl2017",
            "done_criteria": ["timeout", "reach_goal", "collision", "offroad"],
            "collision_sensor": "on",
            "lane_sensor": "on",
        },
    },
}


class FollowLeadingVehicle(MultiCarlaEnv):
    """A two car Multi-Agent Carla-Gym environment

    Example of creating a custom environment, also a demonstration of how to use the manual_control.

    This is a scenario extracted from Carla/scenario_runner (https://github.com/carla-simulator/scenario_runner/blob/master/srunner/scenarios/follow_leading_vehicle.py).
    """

    def __init__(self):
        self.configs = configs
        super(FollowLeadingVehicle, self).__init__(self.configs)


if __name__ == "__main__":
    from cmad.misc import test_run

    env = FollowLeadingVehicle()
    test_run(env, 2)
