def gen_actor_config(
    actor_id,
    actor_type="vehicle_4W",
    role_name=None,
    blueprint=None,
    spawn=True,
    done_criteria=None,
    camera_type=None,
    collision_sensor="on",
    lane_sensor="on",
    render=True,
    reward="custom",
    auto_control=False,
    enable_planner=False,
    opt_dict=None,
    initial_speed=20,
    target_speed=20,
):
    """Generate actor config.

    Args:
        actor_id (str): actor id.
        actor_type (str): actor type.
        role_name (str): role name.
        blueprint (str): blueprint.
        spawn (bool): whether to spawn the actor.
        done_criteria (list[str]): done criteria.
        camera_type (str | list[str]): camera type.
        collision_sensor ("on" | "off"): whether attach collision sensor or not.
        lane_sensor ("on" | "off"): whether attach lane_invasion sensor or not.
        render (bool): whether to render the actor.
        reward (str): reward function.
        auto_control (bool): whether to enable auto control.
        enable_planner (bool): whether to enable planner.
        opt_dict (dict): optional dict.
        initial_speed (float): initial speed.
        target_speed (float): target speed.

    Returns:
        dict: actor config.
    """
    if actor_id in ["ego", "hero"]:
        actor_id = "ego"

    actor_config = {
        "actor_id": actor_id,
        "spawn": spawn,
        "type": actor_type,
        "rolename": role_name or actor_id,
        "reward_function": reward,
        "done_criteria": done_criteria or ["timeout", "reach_goal", "collision", "offroad"],
        "collision_sensor": collision_sensor,
        "lane_sensor": lane_sensor,
        # Enable autocontrol for debug only
        "auto_control": auto_control,
        "enable_planner": enable_planner,
        "opt_dict": opt_dict or {
            "ignore_traffic_lights": True,
            "ignore_vehicles": False,
            "ignore_stop_signs": True,
            "sampling_resolution": 2.0,
            "base_vehicle_threshold": 5.0,
            "base_tlight_threshold": 5.0,
            "max_brake": 0.5,
        },
        "initial_speed": initial_speed if "static" not in actor_type else 0,
        "target_speed": target_speed if "static" not in actor_type else 0,
    }

    # Optional config
    if camera_type is not None:
        actor_config["camera_type"] = camera_type
        actor_config["render"] = render
    if blueprint is not None:
        actor_config["blueprint"] = blueprint

    return actor_config


def gen_env_config(
    server_ip=None,
    server_port=None,
    server_map="/Game/Carla/Maps/Town01",
    reload=True,
    hard_reset=False,
    sync_server=True,
    fixed_delta_seconds=0.05,
    render=True,
    render_x_res=800,
    render_y_res=600,
    spectator_loc=None,
    obs_x_res=168,
    obs_y_res=168,
    framestack=1,
    send_measurements=True,
    measurement_type=["all"],
    action_type="low_level_action",
    discrete_actions=True,
    global_observation=None,
    use_redis=False,
    redis_host="localhost",
    redis_port=6379,
    redis_db=0,
    verbose=False,
):
    """Generate environment config.

    Args:
        server_ip (str): Existed server ip. If None, Cmad will start a new server.
        server_port (int): Existed server port. Use it only when server_ip is not None.
        server_map (str): Server map.
        reload (bool): Whether to reload the world. This is useful during unstable training.
        hard_reset (bool): Whether to cleanup the world each episode.
        sync_server (bool): Whether enable sync mode.
        fixed_delta_seconds (float): Fixed delta seconds.
        render (bool): Whether to show the Carla window, only works on Cmad server.
        render_x_res (int): Carla window render x resolution.
        render_y_res (int): Carla window render y resolution.
        spectator_loc (list): Spectator location. i.e. [x, y, z, pitch, yaw, roll]
        obs_x_res (int): Observation x resolution for vehicle"s rgb.
        obs_y_res (int): Observation y resolution for vehicle"s rgb.
        framestack (int): Framestack. (Better use 1)
        send_measurements (bool): Whether to send measurements.
        measurement_type (list[str]): Measurement type.
        action_type (str): Action type.
        discrete_actions (bool): Whether to use discrete actions.
        global_observation (dict): Global observation info.
        use_redis (bool): Whether to use redis. This is used to sync with Ego Vehicle outside of CMAD.
        redis_host (str): Redis host.
        redis_port (int): Redis port.
        redis_db (int): Redis db.
        verbose (bool): Whether to save debug info.
    """

    env_config = {
        # Server info
        "server_map": server_map,
        "reload": reload,
        "hard_reset": hard_reset,
        "sync_server": sync_server,
        "fixed_delta_seconds": fixed_delta_seconds,
        "render": render,
        "render_x_res": render_x_res,
        "render_y_res": render_y_res,
        "spectator_loc": spectator_loc,
        "global_observation": global_observation,
        # observation info
        "obs": {
            "obs_x_res": obs_x_res,
            "obs_y_res": obs_y_res,
            "framestack": framestack,
            "send_measurements": send_measurements,
            "measurement_type": measurement_type,
        },
        # Action info
        "action": {
            "type": action_type,
            "use_discrete": discrete_actions,
        },
        # Debug info
        "verbose": verbose,
    }

    # optional config
    if server_ip is not None:
        env_config["server_ip"] = server_ip
        env_config["server_port"] = server_port or 2000
    if use_redis:
        env_config["use_redis"] = use_redis
        env_config["redis_host"] = redis_host
        env_config["redis_port"] = redis_port
        env_config["redis_db"] = redis_db

    return env_config


def gen_scenario_config(
    map_name, actors, max_steps=600, weather_distribution=None, position_dict=None
):
    """Generate a pseudo scenario config.

    Args:
        map_name (str): Map name.
        actors (iterable[str]): List of actors ids.
        max_steps (int): Max steps.
        weather_distribution (list[int]): Weather distribution.

    Returns:
        dict: Scenario config.
    """
    scenario_config = {
        "actors": {
            actor_id: {"start": [0, 0, 0], "end": [0, 0, 0]} for actor_id in actors
        },
        "map": map_name,
        "max_steps": max_steps,
        "weather_distribution": weather_distribution or [0],
    }

    if position_dict is not None:
        for actor_id, position in position_dict.items():
            scenario_config["actors"][actor_id]["start"] = position["start"]
            scenario_config["actors"][actor_id]["end"] = position["end"]

    return scenario_config
