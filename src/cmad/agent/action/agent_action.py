"""
This module defines the action used in Agent.
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import Any, Dict, Union, Tuple

import numpy as np
from gym.spaces import Space, Box, Discrete, MultiDiscrete

from cmad.agent.action.action_interface import AbstractAction, ActionInterface
from cmad.envs.static_asset import ENV_ASSETS


class AgentAction:
    def __init__(self, action_config: dict):
        """Initialize action space for a single agent

        Args:
            action_config (dict): The action configuration
        """
        self.action_config: dict = action_config
        self.action_type: str = action_config["type"]
        self.is_discrete: bool = action_config.get("use_discrete", True)
        self.discrete_action_set: Union[Dict[int, Any], Tuple[dict]] = (
            action_config.get("discrete_action_set", None)
        )

        if self.is_discrete and self.discrete_action_set is None:
            attr_name = (
                f"default_{self.action_type.split('_action')[0]}_discrete_actions"
            )
            self.discrete_action_set = getattr(ENV_ASSETS, attr_name, None)
            if self.discrete_action_set is None:
                raise ValueError(
                    f"Please provide a discrete action set for {self.action_type}"
                )
            else:
                action_config["discrete_action_set"] = self.discrete_action_set

        self.space_type = self.get_space_type()
        self.action_space = self.get_action_space()
        self._action_handler = self._init_action_handler()

    def _init_action_handler(self) -> ActionInterface:
        """Initilize the action handler based on the action type

        This can either be a path to a python file or a python module. For example:

        1. "cmad/agent/action/custom_action.py".
        2. "custom_action". By default, the module will be searched in the current directory.
        """
        if ".py" in self.action_type:
            module_name = os.path.basename(self.action_type).split(".")[0]
            sys.path.append(os.path.dirname(self.action_type))
            module_control = importlib.import_module(module_name)
            action_handler_name = module_control.__name__.title().replace("_", "")
        else:
            sys.path.append(os.path.dirname(__file__))
            module_control = importlib.import_module(self.action_type)
            action_handler_name = (
                self.action_type.split(".")[-1].title().replace("_", "")
            )

        # Initialize and return the class
        return getattr(module_control, action_handler_name)(self.action_config)

    def convert_single_action(
        self, action: Union[int, Any], done_state=False
    ) -> AbstractAction:
        """Convert a model output action to an AbstractAction

        Args:
            action (int | Any): The action to be converted
            done_state (bool, optional): Whether the actor is done. Defaults to False.

        Returns:
            AbstractAction: An action instance
        """
        if self.space_type == "Discrete":
            action = self.discrete_action_set[int(action)]
        elif self.space_type == "MultiDiscrete":
            action = [
                action_set[int(action[i])]
                for i, action_set in enumerate(self.discrete_action_set)
            ]

        return self._action_handler.convert_single_action(action, done_state)

    def get_space_type(self) -> str:
        """Return the space type of the action space"""
        if self.is_discrete:
            if isinstance(self.discrete_action_set, dict):
                return "Discrete"
            else:
                return "MultiDiscrete"
        else:
            return "Box"

    def get_stop_action(self, abstrct_action: bool = False) -> Union[int, Any]:
        """Return a stop action to patch the action input to env

        Returns:
            int | Any: A valid value in action space, representing stop behavior
        """
        return self._action_handler.stop_action(
            env_action=(not abstrct_action), use_discrete=self.is_discrete
        )

    def get_action_space(self) -> Space:
        """Get the action space for this agent"""
        if self.space_type == "Discrete":
            action_space = Discrete(len(self.discrete_action_set))
        elif self.space_type == "MultiDiscrete":
            action_space = MultiDiscrete(
                [len(action_set) for action_set in self.discrete_action_set]
            )
        else:
            # TODO: configure the action space
            action_space = Box(-np.inf, np.inf, shape=(2,))

        return action_space

    def get_action_mask(self, actor) -> np.ndarray:
        """Get the action mask for a given actor

        Args:
            actor (carla.Actor): The actor

        Returns:
            np.ndarray: A numpy array of action mask
        """
        action_mask = self._action_handler.get_action_mask(
            actor, self.discrete_action_set
        )

        if action_mask == True:
            if self.space_type == "MultiDiscrete":
                return np.ones(
                    sum([len(action_set) for action_set in self.discrete_action_set]),
                    dtype=np.float32,
                )
            else:
                return np.ones(len(self.discrete_action_set), dtype=np.float32)
        elif isinstance(action_mask, dict):
            # Discrete
            return np.array(
                [action_mask[action] for action in self.discrete_action_set.values()],
                dtype=np.float32,
            )
        elif isinstance(action_mask, tuple):
            # MultiDiscrete
            return np.concatenate(
                [
                    [action_mask[idx][action] for action in space.values()]
                    for idx, space in enumerate(self.discrete_action_set)
                ],
                dtype=np.float32,
            )
        else:
            raise ValueError(f"Invalid action mask type: {type(action_mask)}")
