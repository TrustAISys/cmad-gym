from __future__ import annotations

import copy
from typing import Callable, Dict, List, Optional, Union

from cmad.simulation.data.measurement import Measurements


class DoneContext:
    def __init__(self, actor_id: str, measurements: Measurements, extra_args: dict):
        self.actor_id = actor_id
        self.measurements = measurements
        self.extra_args = extra_args or {}


class Done:
    _registered_done: Dict[str, dict] = {}

    _registered_done_chain: Dict[str, List[str]] = {
        "ego_done_chain": [
            "ego_collision",
            "ego_offroad",
            "ego_rollover",
            "ego_goal",
            "ego_timeout",
        ],
        "npc_done_chain": ["collision", "offroad", "rollover", "timeout", "goal"],
        "episode_done_chain": ["npc_done", "ego_done_chain"],
    }

    EGO_PREFIX = "ego_"
    ANY_PREFIX = "any_"
    CHAIN_POSTFIX = "_chain"

    # register functions
    @staticmethod
    def _register_strategy(
        done_name: str, strategy_fn: Callable, force: bool = False, **default_args
    ):
        """
        Private method to handle the core registration logic.

        Args:
            done_name (str): The name of the strategy to register.
            strategy_fn (function): The function implementing the strategy.
            force (bool): Whether to force register the strategy if it is already registered.
            default_args (dict): Default arguments for the strategy function.

        Raises:
            ValueError: If the strategy with the given name is already registered.
        """
        if done_name in Done._registered_done and not force:
            raise ValueError(
                f"Strategy '{done_name}' is already registered. Duplicate registration is not allowed."
            )
        Done._registered_done[done_name] = {
            "function": strategy_fn,
            "default_args": default_args,
        }

    @staticmethod
    def register_done(
        done_name: str, strategy_fn: Callable, force: bool = False, **default_args
    ):
        """
        Method to register a strategy dynamically at runtime.

        Args:
            done_name (str): The name of the strategy to register.
            strategy_fn (function): The function implementing the strategy.
            force (bool): Whether to force register the strategy if it is already registered.
            default_args (dict): Default arguments for the strategy function.
        """
        Done._register_strategy(done_name, strategy_fn, force, **default_args)

    # apply functions

    @staticmethod
    def _parse_strategy_name(actor_id: str, strategy_name: str):
        """
        Parses the strategy name and returns actor_id type and actual strategy name.
        """
        if strategy_name.startswith(Done.EGO_PREFIX):
            return "ego", strategy_name[len(Done.EGO_PREFIX) :]
        elif strategy_name.startswith(Done.ANY_PREFIX):
            return "any", strategy_name[len(Done.ANY_PREFIX) :]
        else:
            return actor_id, strategy_name

    @staticmethod
    def _apply_done(done_name: str, context: DoneContext):
        """
        Apply the done strategy with the given name.

        Args:
            done_name (str): The name of the strategy.
            context (DoneContext): The context containing actor_id, measurements, and extra_args.

        Returns:
            bool: The result of the strategy.
        """
        actor_id, strategy_name = Done._parse_strategy_name(context.actor_id, done_name)
        if strategy_name not in Done._registered_done:
            raise ValueError(f"Cannot find strategy ({strategy_name})")

        strategy = Done._registered_done[strategy_name]
        kwargs = copy.deepcopy(strategy.get("default_args", {}))
        if done_name in context.extra_args:
            kwargs.update(context.extra_args[done_name])

        if actor_id == "any":
            return any(
                [
                    strategy["function"](k, context.measurements, **kwargs)
                    for k in context.measurements
                ]
            )
        else:
            return strategy["function"](actor_id, context.measurements, **kwargs)

    @staticmethod
    def _apply_done_chain(
        done_chain: List[str], context: DoneContext, return_reason: bool = False
    ):
        """
        Apply a chain of done strategies.

        Args:
            done_chain (list): A list of strategy names.
            context (DoneContext): The context containing actor_id, measurements, and extra_args.
            return_reason (bool): Whether to return the reason for the done.

        Returns:
            bool: The combined result of the strategies.
            str: The reason for the done.
        """
        result = False
        for done_name in done_chain:
            if Done._apply_done(done_name, context):
                result = True
                break

        if return_reason:
            return result, done_name.upper() if result else "NOT_DONE"
        else:
            return result

    @staticmethod
    def is_done(
        actor_id: str,
        measurements: Measurements,
        done_config: Union[str, list],
        return_reason: bool = False,
        extra_args: Optional[dict] = None,
    ):
        """
        Determine if an actor_id is done based on the provided config.

        Args:
            actor_id: The actor_id object.
            measurements: The measurements data for all actor.
            done_config (str or list): The strategy name or a list of strategy names.
            return_reason (bool): Whether to return the reason for the done.
            extra_args (dict, optional): Additional arguments for the strategies.

        Returns:
            bool: The result based on the strategy or strategies.
            str: The reason for the done.

        Examples:
        >>> done_config = ["timeout", "offroad"]
        >>> extra_args = {"offroad": {"threshold": 10}}
        >>> Done.is_done(actor_id, measurements, done_config, extra_args=extra_args)
        """
        if done_config is None:
            return False

        context = DoneContext(actor_id, measurements, extra_args)
        done_config = Done._parse_done_config(done_config)
        return Done._apply_done_chain(done_config, context, return_reason)

    @staticmethod
    def _parse_done_config(done_config: Union[str, list]):
        """
        Recursively parse the done_config to get a flattened list of strategy names.
        """
        # Base case: if done_config is a simple string strategy
        if isinstance(done_config, str) and not done_config.endswith("_chain"):
            return [done_config]

        # Case: if done_config is a chain strategy
        if isinstance(done_config, str) and done_config.endswith("_chain"):
            if done_config not in Done._registered_done_chain:
                raise ValueError(f"Chain strategy {done_config} not registered.")
            return Done._parse_done_config(Done._registered_done_chain[done_config])

        # Case: if done_config is a list of strategies
        if isinstance(done_config, list):
            strategies = []
            for strategy in done_config:
                strategies.extend(Done._parse_done_config(strategy))
            return strategies

        raise ValueError(f"Invalid done_config {done_config}")

    @staticmethod
    def list_available_strategies():
        """
        List all available strategies.
        """
        from tabulate import tabulate

        strategies = []
        for name, strategy in Done._registered_done.items():
            strategies.append([name, strategy["function"].__doc__])
        print(tabulate(strategies, headers=["Name", "Description"], tablefmt="github"))
