from cmad.envs.multi_env import MultiCarlaEnv

configs = {
    "scenarios": "SUI1B2C1P_TOWN3",
    "env": {
        "server_map": "/Game/Carla/Maps/Town03",
        "sync_server": True,
        "fixed_delta_seconds": 0.05,
        "render": True,
        "render_x_res": 800,
        "render_y_res": 600,
        "spectator_loc": [70, -125, 9],
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
            "render": False,
            "auto_control": False,
            "enable_planner": True,
            "manual_control": False,
            "reward_function": "corl2017",
            "done_criteria": ["timeout", "reach_goal", "collision", "offroad"],
            "collision_sensor": "on",
            "lane_sensor": "on",
        },
        "car2": {
            "type": "vehicle_4W",
            "camera_type": "rgb",
            "render": False,
            "auto_control": False,
            "enable_planner": True,
            "manual_control": False,
            "reward_function": "corl2017",
            "done_criteria": ["timeout", "reach_goal", "collision", "offroad"],
            "collision_sensor": "on",
            "lane_sensor": "on",
        },
        "pedestrian1": {
            "type": "pedestrian",
            "camera_type": "rgb",
            "render": False,
            "auto_control": False,
            "enable_planner": False,
            "manual_control": False,
            "reward_function": "corl2017",
            "done_criteria": ["timeout", "reach_goal", "collision", "offroad"],
            "collision_sensor": "on",
            "lane_sensor": "off",
        },
        "bike1": {
            "type": "vehicle_2W",
            "camera_type": "rgb",
            "render": False,
            "auto_control": False,
            "enable_planner": True,
            "manual_control": False,
            "reward_function": "corl2017",
            "done_criteria": ["timeout", "reach_goal", "collision", "offroad"],
            "collision_sensor": "on",
            "lane_sensor": "on",
        },
    },
}


class TrafficLightSignal1B2C1PTown03(MultiCarlaEnv):
    """A 4-way signalized intersection with 1 Bike, 2 Cars, 1 Pedestrian"""

    def __init__(self):
        super(TrafficLightSignal1B2C1PTown03, self).__init__(configs)


if __name__ == "__main__":
    from cmad.misc import test_run

    env = TrafficLightSignal1B2C1PTown03()
    test_run(env, 5)
