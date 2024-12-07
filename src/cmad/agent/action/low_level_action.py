from __future__ import annotations

import logging
import math
from typing import Tuple

import carla
import numpy as np

from cmad.agent.action.action_interface import AbstractAction, ActionInterface

logger = logging.getLogger(__name__)


def sigmoid(x: float):
    x = float(x)
    return math.exp(x) / (1 + math.exp(x))


def default_preprocess_fn(action: Tuple[float, float]):
    throttle = float(np.clip(action[0], 0, 1))
    brake = float(np.abs(np.clip(action[0], -1, 1)))
    steer = float(np.clip(action[1], -1, 1))
    return {
        "throttle": throttle,
        "brake": brake,
        "steer": steer,
        "reverse": False,
        "hand_brake": False,
    }


def squash_action_logits(action: Tuple[float, float]):
    forward = 2 * float(sigmoid(action[0]) - 0.5)
    throttle = float(np.clip(forward, 0, 1))
    brake = float(np.abs(np.clip(forward, -1, 0)))
    steer = 2 * float(sigmoid(action[1]) - 0.5)
    return {
        "throttle": throttle,
        "brake": brake,
        "steer": steer,
        "reverse": False,
        "hand_brake": False,
    }


class DirectAction(AbstractAction):
    def __init__(self, action, duration=1):
        super().__init__(action, duration)

    def run_step(self, actor: carla.Actor):
        """Return a carla control signal based on the actor type and the action

        Args:
            actor (carla.Actor): The actor to run the action

        Returns:
            carla.VehicleControl or carla.WalkerControl: The control signal
        """
        self.duration -= 1

        if isinstance(actor, carla.Walker):
            rotation = actor.get_transform().rotation
            rotation.yaw += self.action["steer"] * 10.0
            x_dir = math.cos(math.radians(rotation.yaw))
            y_dir = math.sin(math.radians(rotation.yaw))

            return carla.WalkerControl(
                speed=3.0 * self.action["throttle"],
                direction=carla.Vector3D(x_dir, y_dir, 0.0),
            )
        elif isinstance(actor, carla.Vehicle):
            control = carla.VehicleControl()
            control.throttle = self.action["throttle"]
            control.steer = self.action["steer"]
            control.brake = self.action["brake"]
            control.hand_brake = self.action["hand_brake"]
            control.reverse = self.action["reverse"]
        else:
            logger.warning(
                "Unknown actor type %s, returning pseudo action", actor.type_id
            )
            control = actor.get_control()

        return control

    def to_dict(self):
        return self.action


class LowLevelAction(ActionInterface):
    def __init__(self, action_config: dict):
        """Initialize the action converter for low-level action space

        Args:
            action_config (dict): A dictionary of action config
        """
        super().__init__(action_config)

        self._action_range = self._action_config.get(
            "action_range",
            {
                "throttle": [0, 1.0],
                "brake": [-1.0, 0],
                "steer": [-1.0, 1.0],
            },
        )

        self._preprocess_fn = self._action_config.get("preprocess_fn", None)
        if self._preprocess_fn is None or self._preprocess_fn == "default":
            self._preprocess_fn = default_preprocess_fn
        elif self._preprocess_fn == "squash":
            self._preprocess_fn = squash_action_logits
        elif not callable(self._preprocess_fn):
            raise ValueError(
                "preprocess_fn should be either None, 'default', 'squash' or a callable function"
            )

    def convert_single_action(self, action: Tuple[int, int], done_state: bool = False):
        """Convert the action of a model output to an AbstractAction instance

        Args:
            action: Action for a single actor
            done_state (bool): Whether the actor is done. If done, return a stop action

        Returns:
            DirectAction: A direct action instance
        """
        if done_state:
            return self.stop_action(env_action=False)
        else:
            action_dict = self._clip_action(self._preprocess_fn(action))
            return DirectAction(action_dict)

    def _clip_action(self, action_dict: dict):
        """Clip the action to make sure the action is within the range of action space"""
        action_dict["throttle"] = np.clip(
            action_dict["throttle"],
            self._action_range["throttle"][0],
            self._action_range["throttle"][1],
        )
        action_dict["brake"] = np.clip(
            action_dict["brake"],
            self._action_range["brake"][0],
            self._action_range["brake"][1],
        )
        action_dict["steer"] = np.clip(
            action_dict["steer"],
            self._action_range["steer"][0],
            self._action_range["steer"][1],
        )
        return action_dict

    def get_action_mask(self, actor, action_space):
        """Low-level action is always applicable"""
        return True

    def stop_action(self, env_action: bool = True, use_discrete: bool = False):
        """Return the stop action representation in low-level action space

        Args:
            env_action (bool): Whether using env action space
            use_discrete (bool): Whether using discrete action space

        Returns:
            list[int, int]: if env_action is True and use_discrete is False, return the stop action in the continuous action space.
            int: if env_action is True and use_discrete is True, return the index of the stop action in the discrete action space.
            DirectAction: if env_action is False, return the stop action in the action space of the action handler.
        """
        if not env_action:
            return DirectAction(
                {
                    "throttle": 0.0,
                    "brake": 1.0,
                    "steer": 0.0,
                    "reverse": False,
                    "hand_brake": True,
                }
            )

        return 0 if use_discrete else [-1, 0]

    @staticmethod
    def sort_action_set(action_set: dict):
        """Re-arrange the action_set to make sure the order of actions is consistent

        This function can help to make sure that order of low-level action_space is ordered by "throttle/brake" (from low to high) and then "steer" (from low to high)

        By using this, we make sure the brake action is always the first action (index 0)
        """
        sorted_items = sorted(action_set.items(), key=lambda item: item[1][0])
        return {new_key: value for new_key, (old_key, value) in enumerate(sorted_items)}
