from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Dict

from gym.spaces import Dict as GymDict

if TYPE_CHECKING:
    from cmad.envs.multi_env import MultiCarlaEnv

from cmad.agent.action.agent_action import AgentAction
from cmad.envs.static_asset import ENV_ASSETS


class EnvAction:
    def __init__(self, env: "MultiCarlaEnv"):
        """Initialize the env action with a configuration dictionary. The configuration dictionary is stored in env.env_config

        Args:
            env (MultiCarlaEnv): The environment
        """
        self.env_ref = weakref.proxy(env)
        self._actor_configs = env.actor_configs
        self._background_actor_ids = env.background_actor_ids

        # If the action space is set in Env, it is a fallback for each actor
        if env.env_config.get("action", None):
            for config in self._actor_configs.values():
                config["action"] = (
                    config.get("action", None) or env.env_config["action"]
                )

        self.agent_actions = self.parse_agent_action(self._actor_configs)

    def parse_agent_action(
        self, actor_configs: Dict[str, dict]
    ) -> Dict[str, AgentAction]:
        """Parse actor configs and return a dictionary of agent actions

        Args:
            actor_configs (dict): A dictionary of actor configs

        Returns:
            dict[str, AgentAction]: A dictionary of agent actions
        """
        agent_actions = {}

        for actor_id, config in actor_configs.items():
            if actor_id in self._background_actor_ids:
                continue

            action_config = config.get("action", ENV_ASSETS.default_action_conf)
            action_config["env"] = self.env_ref
            agent_actions[actor_id] = AgentAction(action_config)

        return agent_actions

    def get_action_space(self) -> GymDict:
        """Get the action space of the environment"""
        action_space = {}

        for actor_id, agent_action in self.agent_actions.items():
            if actor_id in self._background_actor_ids:
                continue

            action_space[actor_id] = agent_action.action_space

        return GymDict(action_space)

    def get_agent_actions(self) -> Dict[str, AgentAction]:
        """Return the agent actions

        Returns:
            dict[str, AgentAction]: A dictionary of agent actions
        """
        return self.agent_actions

    def check_validity(self, action_dict: dict, done_states: Dict[str, bool]):
        """Check if the action given by `env.step(action)` is valid. If valid, complement pseudo actions for already done actors

        Args:
            action_dict (dict): A dictionary of actions
            done_states (dict): A dictionary of done states

        Raises:
            ValueError: If action_dict is not a dict
            ValueError: If action_dict contains actions for non-existent actors

        Returns:
            actions (dict): A dictionary of actions (including pseudo actions for already done actors)
        """

        if not isinstance(action_dict, dict):
            raise ValueError(
                "`step(action_dict)` expected dict of actions. "
                "Got {}".format(type(action_dict))
            )

        # Make sure the action_dict contains actions only for actors that
        # exist in the environment
        input_actor = set(action_dict)
        exist_actor = set(self._actor_configs)
        if not input_actor.issubset(exist_actor):
            raise ValueError(
                "Cannot execute actions for non-existent actors."
                " Received unexpected actor ids:{}".format(
                    input_actor.difference(exist_actor)
                )
            )

        actions = action_dict.copy()
        for actor_id, done_state in done_states.items():
            if (actor_id in self._actor_configs) and done_state:
                actions[actor_id] = self.agent_actions[actor_id].get_stop_action()

        return actions
