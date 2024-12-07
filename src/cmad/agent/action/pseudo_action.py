from __future__ import annotations

import logging

import carla
import math

from cmad.agent.action.action_interface import AbstractAction, ActionInterface

logger = logging.getLogger(__name__)


class NullAction(AbstractAction):
    def __init__(self, action, duration=math.inf):
        super().__init__(action, duration)

    def run_step(self, actor: carla.Actor):
        return None


class PseudoAction(ActionInterface):
    def __init__(self, action_config: dict):
        super().__init__(action_config)

    def convert_single_action(self, action, done_state: bool = False):
        return NullAction(action)

    def get_action_mask(self, actor, action_space):
        return True

    def stop_action(self, env_action: bool = True, use_discrete: bool = False):
        return NullAction("null") if not env_action else 0
