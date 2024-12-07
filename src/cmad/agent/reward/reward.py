from __future__ import annotations

import inspect
from collections import defaultdict
from functools import wraps
from typing import Any, Dict, List

import numpy as np

from cmad.simulation.data.measurement import Measurements


class RewardState:
    """This class should be instantiated per Environment, and passed to the Reward function"""

    def __init__(self, measurements_history: List[Measurements] = None):
        self.episode_reward = 0.0
        self.measurement_hist = measurements_history
        self.prev_measurements: Measurements = None
        self.current_measurements: Measurements = None
        # expose a cache to store intermediate results
        self.cache = defaultdict(
            lambda: defaultdict(lambda: None, {"active_state": True})
        )

    def update(
        self, prev_measurements: Measurements, current_measurements: Measurements
    ):
        self.prev_measurements = prev_measurements
        self.current_measurements = current_measurements

    def reset(self):
        self.episode_reward = 0.0
        self.prev_measurements = None
        self.current_measurements = None
        self.cache.clear()


class Reward:
    """
    The reward class holds all registered reward computation functions, and provides a unified interface to compute the reward.

    Properties:
    - episode_reward (float): The accumulated reward for the current episode. (All-agent sum)
    - prev (dict[str, Measurement]): The previous measurements of all actors.
    - curr (dict[str, Measurement]): The current measurement of all actors.
    - cache (defaultdict): A cache to store intermediate results.

        Recommended usage: The best way to use cache is to store Agent-wise.

        E.g. self.cache[actor_id] = {"active_state": True, "distance_to_goal_cache": 0.0}

        Note that this cache will be reset after each episode.
    """

    _registered_rewards = {}

    @classmethod
    def register_reward(cls, name, func):
        """
        Register a new reward computation function.

        Args:
        - name: The name to associate with the reward computation function.
        - func: The function to compute the reward.
        """
        cls._registered_rewards[name] = func

    @classmethod
    def compute_reward(
        cls,
        actor_id: str,
        reward_state: RewardState,
        flag: str,
        *args,
        **kwargs,
    ):
        # Use the 'flag' to look up the reward computation function in the dictionary,
        # and call it.
        if flag in cls._registered_rewards:
            step_reward = cls._registered_rewards[flag](
                actor_id, reward_state, *args, **kwargs
            )
            reward_state.episode_reward += step_reward
            return step_reward
        else:
            raise ValueError(f"Unrecognized reward computation method: {flag}")

    @staticmethod
    def reward_signature(func):
        """This decorator is used to wrap the reward computation function, and make sure it has the correct signature.

        Args:
            func (Callable): The user-defined reward computation function.

        Returns:
            func (Callable): The wrapped reward computation function.
        """

        # Get the signature of the function
        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(first_arg: str, second_arg: RewardState, *args, **kwargs):
            # Start with the first two positional arguments
            bound_args = [first_arg, second_arg]

            # Add *args and **kwargs if the function can accept them
            if any(p.kind == p.VAR_POSITIONAL for p in sig.parameters.values()):
                bound_args.extend(args)
            if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
                bound_kwargs = kwargs
            else:
                bound_kwargs = {}

            # Call the function with the prepared arguments
            return func(*bound_args, **bound_kwargs)

        return wrapper

    @staticmethod
    def capture_reward_detail(
        local_vars: Dict[str, Any], prefix="reward_", suffix="_reward", only_scalar=True
    ):
        """This function is used to capture the reward detail, and return a dictionary.

        Args:
            local_vars (dict): The local variables of the reward computation function.
            prefix (str, optional): The prefix of the reward detail. Defaults to "reward_".
            suffix (str, optional): The suffix of the reward detail. Defaults to "_reward".
            only_scalar (bool, optional): Whether to only capture scalar values. Defaults to True.

        Returns:
            dict: The reward detail.
        """
        reward_detail = {}
        for key, value in local_vars.items():
            if key.startswith(prefix) or key.endswith(suffix):
                if only_scalar and isinstance(
                    value, (float, int, np.integer, np.floating)
                ):
                    reward_detail[key] = value
        return reward_detail
