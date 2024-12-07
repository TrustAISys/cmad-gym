from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import gym
import numpy as np
from gym.spaces import Box, Dict, Discrete, MultiDiscrete

if TYPE_CHECKING:
    from cmad.envs.multi_env import MultiCarlaEnv


class ActionPaddingWrapper(gym.Wrapper):
    """This wrapper is used to pad action spaces to be a single Discrete space with largest action space size."""

    def __init__(self, env: MultiCarlaEnv):
        super(ActionPaddingWrapper, self).__init__(env)

        self.env = env
        self.agent_actions = env.agent_actions

        spaces = {}
        for agent_action in self.agent_actions.values():
            action_type = agent_action.action_type
            if action_type not in spaces:
                spaces[action_type] = agent_action.action_space
        self.spaces = dict(sorted(spaces.items()))

        self.valid_actions = {
            actor_id: self.get_valid_actions_for_agent(actor_id)
            for actor_id in self.env.action_space
        }

        max_action_space = max(self.valid_actions.values(), key=len)
        self.max_action_space_size = len(max_action_space)
        self.action_space = Dict(
            {
                actor_id: Discrete(self.max_action_space_size)
                for actor_id in self.env.action_space
            }
        )

        self.observation_space = Dict(
            {
                actor_id: Dict(
                    {
                        **self.env.observation_space[actor_id],
                        "action_mask": Box(
                            0.0,
                            1.0,
                            shape=(self.max_action_space_size,),
                            dtype=np.float32,
                        ),
                    }
                )
                for actor_id in self.env.observation_space
            }
        )

    def get_valid_actions_for_agent(self, agent: str):
        action_type = self.agent_actions[agent].action_type
        space = self.spaces[action_type]
        if isinstance(space, Discrete):
            return list(range(space.n))
        elif isinstance(space, MultiDiscrete):
            return list(range(np.prod(space.nvec)))

    def reset(self):
        obs_dict = self.env.reset()
        for actor_id, obs in obs_dict.items():
            obs["action_mask"] = self.get_mask_for_agent(
                actor_id, obs.get("action_mask", None)
            )
        return obs_dict

    def step(self, action_dict: dict):
        for actor_id, action in action_dict.items():
            original_space = self.env.action_space.spaces[actor_id]
            action_dict[actor_id] = self.convert_action_to_original(
                action, original_space
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

        start_idx, end_idx = self.valid_actions[agent][0], self.valid_actions[agent][-1]
        mask = np.zeros(self.max_action_space_size, dtype=np.float32)

        if origin_mask is None:
            mask[start_idx : end_idx + 1] = 1
        else:
            mask[start_idx : end_idx + 1] = self.convert_agent_mask_to_padded(
                origin_mask, origin_space
            )
        return mask

    def convert_action_to_original(self, action: int, original_space: gym.Space):
        if isinstance(original_space, Discrete):
            return np.clip(action, 0, original_space.n - 1)
        elif isinstance(original_space, MultiDiscrete):
            multi_discrete_values = []
            for n in reversed(original_space.nvec):
                multi_discrete_values.append(action % n)
                action //= n
            return list(reversed(multi_discrete_values))

    def convert_agent_mask_to_padded(
        self, mask: np.ndarray, original_space: "gym.Space | str"
    ):
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
