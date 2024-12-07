from __future__ import annotations

import abc

from typing import Union


class AbstractAction(abc.ABC):
    def __init__(self, action, duration: int):
        self.action = action
        self.duration = duration

    @abc.abstractmethod
    def run_step(self, actor):
        """Run the action for one step and return a valid control signal in Carla."""
        raise NotImplementedError

    def done(self) -> bool:
        """Signal whether the action is done."""
        return self.duration <= 0

    def to_dict(self) -> dict:
        """Return a dictionary representation of the action."""
        return {"action_info": self.action, "duration": self.duration}


class ActionInterface(abc.ABC):
    def __init__(self, config: dict):
        """Initialize the action handler with a configuration dictionary."""
        self._action_config = config

    @abc.abstractmethod
    def convert_single_action(
        self, action, done_state: bool = False, **kwargs
    ) -> AbstractAction:
        """Convert a single action to a AbstactAction object.

        Args:
            action: model output action
            done_state (bool, optional): Whether the actor is in done state. Defaults to False.

        Returns:
            AbstactAction: The converted action. Call run_step with the actor to get the control signal.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_action_mask(self, actor, action_space=None) -> Union[bool, dict]:
        """Check if all the action in the action space is valid for the actor.

        Args:
            actor (carla.Actor): actor to be checked
            action_space (dict): candidate action to be checked

        Returns:
            bool: True if all the actions are valid
            dict: A dict of 0/1 for each action in the action space. 1 means the action is valid.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def stop_action(
        self, env_action: bool = True, use_discrete: bool = False
    ) -> Union[int, AbstractAction]:
        """Return a stop action representing in the action space

        Args:
            env_action (bool, optional): Whether to return the stop action in the env action space. Defaults to True.
            use_discrete (bool, optional): Whether to use discrete action space. Defaults to False.

        Returns:
            int: if env_action is True and use_discrete is True, return the index of the stop action in the discrete action space.
            Any: if env_action is True and use_discrete is False, return the stop action in the continuous action space.
            AbstractAction: if env_action is False, return the stop action in the action space of the action handler.
        """
        raise NotImplementedError
