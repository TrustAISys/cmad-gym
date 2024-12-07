"""
Author: Morphlng
Date: 2023-07-17 10:45:47
LastEditTime: 2023-12-16 22:03:36
LastEditors: Morphlng
Description: This script demonstate how to replay a collision scenario, and let the env run from jsut before the collision point.
"""

import os
import time
from collections import defaultdict

import carla

from cmad import MultiCarlaEnv, Simulator
from cmad.simulation.data.replay_parser import Replay
from cmad.misc.config import (
    gen_actor_config,
    gen_env_config,
    gen_scenario_config,
)


def get_first_collision(replay: Replay, focus_actor: str = "hero"):
    focus_actor_id = -1
    for actor in replay.actors:
        if actor.attributes["role_name"] == focus_actor:
            focus_actor_id = actor.id
            break

    if focus_actor_id == -1:
        raise ValueError(f"Cannot find actor {focus_actor} in the replay file")

    for frame in replay.get_collision_frames():
        for colli in frame.collision:
            if focus_actor_id in colli:
                return frame.frame_id

    print(f"No collision related to the focus actor: {focus_actor} is found")
    return -1


if __name__ == "__main__":
    replay_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "./replays"))
    replay_files = [
        os.path.join(replay_root, f)
        for f in os.listdir(replay_root)
        if f.endswith(".dat")
    ]

    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    replay_file = replay_files[-1]
    replay = Replay(client, replay_file, lazy_init=False)

    world = client.load_world(replay.map_name)
    map = world.get_map()

    # You have to wait for the world to fully loaded before you can use it
    # The time might need to be adjusted based on your machine
    time.sleep(3)

    # find the first collision related to our interested actor
    first_colli_id = get_first_collision(replay, "hero")
    start_frame_id = max(0, first_colli_id - 20)
    client.replay_file(
        replay_file,
        replay.get_frame(start_frame_id).frame_time,
        0.1,
        replay.get_actor_by_rolename("hero").id,
        False,
    )

    # Generate env_config
    config = {
        "scenarios": [],
        "env": {},
        "actors": {},
    }

    # Generate actor_config based on actors
    for idx, actor in enumerate(replay.actors):
        id = actor.id
        blueprint = actor.blueprint
        attributes = actor.attributes

        role_name = attributes.get("role_name", None)
        actor_id = "actor_{}".format(idx) if role_name is None else role_name
        if actor_id in ["ego", "hero"]:
            actor_id = "ego"

        number_of_wheels = attributes.get("number_of_wheels", None)
        end_location = replay.frame_info[-1].transform[id].location
        is_static = (actor.spawn_point - end_location).length() < 0.5
        init_speed = replay.get_frame(start_frame_id).velocity[id] * 3.6  # km/h

        if number_of_wheels is not None:
            actor_type = f"vehicle_{number_of_wheels}w"
        elif "walker" in blueprint:
            actor_type = "pedestrian"

        if is_static:
            actor_type = "static_" + actor_type

        actor_config = gen_actor_config(
            actor_id,
            actor_type,
            role_name,
            blueprint,
            spawn=False,
            camera_type=None if is_static else "rgb",
            render=True,
            reward="npc" if actor_id != "ego" else "ego",
            enable_planner=False if is_static else True,
            initial_speed=init_speed,
        )
        config["actors"][actor_id] = actor_config

    # Generate env_config based on map_name
    spectator_loc = world.get_spectator().get_transform().location
    spectator_rot = world.get_spectator().get_transform().rotation
    env_config = gen_env_config(
        "localhost",
        2000,
        f"/Game/Carla/Maps/{replay.map_name}",
        reload=False,
        spectator_loc=[
            spectator_loc.x,
            spectator_loc.y,
            spectator_loc.z,
            spectator_rot.pitch,
            spectator_rot.yaw,
            spectator_rot.roll,
        ],
    )
    config["env"] = env_config

    # Generate a pseudo scenario config
    position_dict = defaultdict(lambda: defaultdict(dict))
    for actor in replay.actors:
        id = actor.id
        attributes = actor.attributes
        role_name = attributes.get("role_name", None)
        actor_id = "actor_{}".format(idx) if role_name is None else role_name
        if actor_id in ["ego", "hero"]:
            actor_id = "ego"

        actor_type = config["actors"].get(actor_id, {}).get("type", "vehicle_4W")

        spawn_transform = Simulator.generate_spawn_point(map, actor.spawn_point)
        position_dict[actor_id]["start"] = (
            spawn_transform.location.x,
            spawn_transform.location.y,
            spawn_transform.location.z,
        )

        if "static" in actor_type:
            goal_transform = spawn_transform
        else:
            goal_transform = Simulator.get_lane_end(map, actor.spawn_point).transform
        position_dict[actor_id]["end"] = (
            goal_transform.location.x,
            goal_transform.location.y,
            goal_transform.location.z,
        )

    config["scenarios"].append(
        gen_scenario_config(
            replay.map_name, config["actors"].keys(), 500, [0], position_dict
        )
    )

    env = MultiCarlaEnv(config)
    env.reset()

    done = {"__all__": False}
    while not done["__all__"]:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

    env.close()
