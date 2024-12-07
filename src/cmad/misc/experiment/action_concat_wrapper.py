from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Dict

import gym
import numpy as np
from gym.spaces import Box
from gym.spaces import Dict as GymDict
from gym.spaces import Discrete, MultiDiscrete

if TYPE_CHECKING:
    from cmad.envs.multi_env import MultiCarlaEnv


def concat_action_spaces(spaces: Dict[str, gym.Space]):
    """Pad action spaces to be a single Discrete space.

    Args:
        spaces (dict[str, gym.Space]): A dict of action_type and action_space

    Raises:
        ValueError: We only support Discrete and MultiDiscrete action spaces

    Returns:
        Discrete: The padded action space

    Example:

        ```python
        action_space = pad_action_spaces({
            "low_level_action": Discrete(9),
            "vehicle_atomic_action": MultiDiscrete([6, 5])
        })
        ```
    """
    total_size = 0
    space_mapping = {}

    for space_name, space in spaces.items():
        if isinstance(space, Discrete):
            space_size = space.n
        elif isinstance(space, MultiDiscrete):
            space_size = np.prod(space.nvec)
        else:
            raise ValueError("Unsupported action space type")

        space_mapping[space_name] = (total_size, total_size + space_size - 1)
        total_size += space_size

    return Discrete(total_size), space_mapping


class ActionConcatWrapper(gym.Wrapper):
    """This wrapper is used to concatenate action spaces for each type of agent into a single Discrete space."""

    def __init__(self, env: MultiCarlaEnv):
        super(ActionConcatWrapper, self).__init__(env)

        self.env = env
        self.agent_actions = env.agent_actions

        spaces = {}
        for agent_action in self.agent_actions.values():
            action_type = agent_action.action_type
            if action_type not in spaces:
                spaces[action_type] = agent_action.action_space
        self.spaces = dict(sorted(spaces.items()))

        # Modify observation and action space to be the padded version
        self.single_action_space, self.space_mapping = concat_action_spaces(self.spaces)
        self.action_space = GymDict(
            {actor_id: self.single_action_space for actor_id in self.env.action_space}
        )

        self.observation_space = GymDict(
            {
                actor_id: GymDict(
                    {
                        **self.env.observation_space[actor_id],
                        "action_mask": Box(
                            0.0,
                            1.0,
                            shape=(self.single_action_space.n,),
                            dtype=np.float32,
                        ),
                    }
                )
                for actor_id in self.env.observation_space
            }
        )

    def reset(self):
        obs_dict = self.env.reset()
        for actor_id, obs in obs_dict.items():
            obs["action_mask"] = self.get_mask_for_agent(
                actor_id, obs.get("action_mask", None)
            )
        return obs_dict

    def step(self, action_dict: dict):
        # Convert actions for each agent back to their original space
        for actor_id, action in action_dict.items():
            original_space = self.env.action_space.spaces[actor_id]
            action_dict[actor_id] = self.convert_action_to_original(
                action, actor_id, original_space
            )

        obs_dict, reward, done, info = self.env.step(action_dict)

        for actor_id, obs in obs_dict.items():
            obs["action_mask"] = self.get_mask_for_agent(
                actor_id, obs.get("action_mask", None)
            )

        return obs_dict, reward, done, info

    def get_mask_for_agent(self, agent: str, origin_mask: np.ndarray = None):
        action_type = self.agent_actions[agent].action_type
        origin_space = self.spaces[action_type]

        total_size = self.single_action_space.n
        start_idx, end_idx = self.space_mapping[action_type]
        mask = np.zeros(total_size, dtype=np.float32)

        if origin_mask is not None:
            mask[start_idx : end_idx + 1] = self.convert_agent_mask_to_padded(
                origin_mask, origin_space
            )
        else:
            mask[start_idx : end_idx + 1] = 1
        return mask

    def convert_action_to_original(
        self, action: int, agent: str, original_space: gym.Space
    ):
        action_type = self.agent_actions[agent].action_type
        start_idx, _ = self.space_mapping[action_type]
        if isinstance(original_space, Discrete):
            return np.clip(action - start_idx, 0, original_space.n - 1)
        elif isinstance(original_space, MultiDiscrete):
            local_action = action - start_idx
            multi_discrete_values = []
            for n in reversed(original_space.nvec):
                multi_discrete_values.append(local_action % n)
                local_action //= n
            return list(reversed(multi_discrete_values))

    def convert_agent_mask_to_padded(self, mask: np.ndarray, original_space: gym.Space):
        """
        Convert an agent's mask from the original action space to the concatenated action space.

        Parameters:
        - mask: A list containing the mask for the original action space.
        - original_space: The original gym action space.

        Returns:
        - A list containing the mask for the concatenated action space.
        """
        if not isinstance(original_space, MultiDiscrete):
            return mask  # For Discrete spaces, the mask remains unchanged

        # Split the mask into sections based on the MultiDiscrete dimensions
        sections = []
        start_idx = 0
        for dim in original_space.nvec:
            sections.append(mask[start_idx : start_idx + dim])
            start_idx += dim

        # Generate valid combinations
        new_mask = []
        for combination in itertools.product(*sections):
            # 1 if all values in the combination are valid (i.e., 1 in the original mask), 0 otherwise
            new_mask.append(int(all(combination)))

        return new_mask
