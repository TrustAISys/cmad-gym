from __future__ import annotations

import carla

from cmad.agent.action.action_interface import AbstractAction, ActionInterface


class DirectAction(AbstractAction):
    def __init__(self, action, duration: int = 10):
        super().__init__(action, duration)

    def run_step(self, actor: carla.Walker):
        """Return a carla control signal based on the actor type and the action

        Args:
            actor (carla.Walker): The actor to run the action

        Returns:
            carla.WalkerControl: The control signal
        """
        self.duration -= 1

        # Always go straight ahead (cross road)
        direction = actor.get_transform().get_forward_vector()
        speed = self.action["speed"]
        actor_control = carla.WalkerControl(direction, speed)
        return actor_control


class WalkerSpeedAction(ActionInterface):
    def __init__(self, action_config: dict):
        """Initialize the action converter for low-level action space

        Args:
            action_config (dict): A dictionary of action config
        """
        super().__init__(action_config)

    def convert_single_action(self, action: int, done_state: bool = False):
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
            return DirectAction({"speed": action})

    def get_action_mask(self, actor, action_space):
        """Low-level action is always applicable"""
        return True

    def stop_action(self, env_action: bool = True, use_discrete: bool = False):
        """Return the stop action representation in low-level action space

        Args:
            env_action (bool): Whether using env action space
            use_discrete (bool): Whether using discrete action space

        Returns:
            DirectAction: if env_action is False, return the stop action in the action space of the action handler.
            EnvAction: a valid action in the env action space
        """

        if not env_action:
            return DirectAction({"speed": 0})

        return 0
