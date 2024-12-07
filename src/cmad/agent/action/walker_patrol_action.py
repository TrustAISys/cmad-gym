from __future__ import annotations

import random
from typing import TYPE_CHECKING, Tuple

import carla
import numpy as np

if TYPE_CHECKING:
    from cmad.envs.multi_env import MultiCarlaEnv

from cmad.agent.action.action_interface import AbstractAction, ActionInterface


class DirectAction(AbstractAction):
    def __init__(self, action, duration=20, waypoints=None):
        super().__init__(action, duration)
        self.waypoints = waypoints
        self.idx = 0

        if self.waypoints is not None:
            self.target_point = np.array(self.waypoints[self.idx])

    def run_step(self, actor: carla.Walker):
        self.duration -= 1

        direction = carla.Vector3D()
        direction_str = self.action["direction"]
        speed = self.action["speed"]
        actor_transform = actor.get_transform()

        if self.waypoints is not None:
            location = actor_transform.location
            diff = self.target_point - np.array([location.x, location.y])

            # Check if actor has reached the current waypoint
            while np.linalg.norm(diff) < 1.0:
                self.move_to_next_waypoint()
                diff = self.target_point - np.array([location.x, location.y])

            diff = diff / (np.linalg.norm(diff) + 1e-6)
            perturbed = self.get_perturbed(direction_str, actor_transform)
            direction.x, direction.y = (
                float(diff[0] + perturbed[0]),
                float(diff[1] + perturbed[1]),
            )

        actor_control = carla.WalkerControl(direction, speed)
        return actor_control

    def move_to_next_waypoint(self):
        self.idx += 1
        if len(self.waypoints) > 1:
            self.idx %= len(self.waypoints)
        elif self.idx >= len(self.waypoints):
            self.idx -= 1
        self.target_point = np.array(self.waypoints[self.idx])

    def update_action(self, action):
        self.action = action
        self.duration = 20

    def reset_waypoint_index(self):
        self.idx = 0

        if self.waypoints is not None:
            self.target_point = np.array(self.waypoints[self.idx])

    @staticmethod
    def get_perturbed(direction_str: str, actor_transform: carla.Transform):
        right_vec = actor_transform.get_right_vector()
        if direction_str == "perturbed":
            x, y = random.uniform(-1, 1), random.uniform(-1, 1)
        elif direction_str == "left":
            x, y = -right_vec.x, -right_vec.y
        elif direction_str == "right":
            x, y = right_vec.x, right_vec.y
        else:
            x, y = 0, 0
        return np.array([x, y])


class WalkerPatrolAction(ActionInterface):
    def __init__(self, action_config: dict):
        super().__init__(action_config)
        if "waypoints" not in action_config:
            raise ValueError("Waypoints must be provided for walker patrol action")

        self.current_episode = -1
        self.env: "MultiCarlaEnv" = action_config["env"]
        self.waypoints = np.array(action_config["waypoints"])
        self.reset = action_config.get("reset", True)

        self.direct_action = DirectAction(
            action={"direction": "stable", "speed": 0.0}, waypoints=self.waypoints
        )

    def convert_single_action(
        self, action: Tuple[str, float], done_state: bool = False
    ):
        if self.reset and self.current_episode != self.env.episode_id:
            self.direct_action.reset_waypoint_index()
            self.current_episode = self.env.episode_id

        if done_state:
            return self.stop_action(env_action=False)
        else:
            self.direct_action.update_action(
                action={"direction": action[0], "speed": action[1]}
            )
            return self.direct_action

    def get_action_mask(self, actor, action_space):
        return True

    def stop_action(self, env_action: bool = True, use_discrete: bool = False):
        if not env_action:
            self.direct_action.update_action(
                action={"direction": "stable", "speed": 0.0}
            )
            return self.direct_action
        return (0, 0)
