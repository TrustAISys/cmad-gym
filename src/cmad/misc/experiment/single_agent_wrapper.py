from __future__ import annotations

from typing import TYPE_CHECKING

import gym

if TYPE_CHECKING:
    from cmad.envs.multi_env import MultiCarlaEnv


class SingleAgentWrapper(gym.Wrapper):
    def __init__(self, env: MultiCarlaEnv, target_actor: str = "ego"):
        if target_actor not in env.actor_configs:
            raise ValueError(f"Target actor {target_actor} is not in the environment.")

        super(SingleAgentWrapper, self).__init__(env)
        self.actor = target_actor
        self.observation_space = env.observation_space[self.actor]
        self.action_space = env.action_space[self.actor]

    def reset(self):
        return self.env.reset()[self.actor]

    def step(self, action):
        actions = self.env.action_space.sample()
        if isinstance(action, dict):
            actions.update(action)
        else:
            actions[self.actor] = action

        obs, reward, done, info = self.env.step(actions)
        return obs[self.actor], reward[self.actor], done[self.actor], info[self.actor]
