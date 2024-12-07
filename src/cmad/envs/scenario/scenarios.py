from __future__ import annotations

import json
import random
from functools import lru_cache
from typing import Tuple, Union

import py_trees

from cmad.envs.scenario.macad_scenario import MacadScenario
from cmad.envs.scenario.scenario_helper import ScenarioHelper
from cmad.envs.static_asset import ENV_ASSETS
from cmad.simulation.data.simulator import Simulator
from cmad.srunner.scenario_runner import ScenarioRunner


class Scenarios:
    """Scenario interface for cmad-gym"""

    def __init__(self, scenario_config: Union[list, dict], env_config: dict = None):
        """Initialize Scenarios

        Args:
            scenario_config (list | dict): The scenario config
            env_config (dict, optional): Env related config. If None, will use default config. Defaults to None.
        """
        self._env_config = env_config or {}

        if not isinstance(scenario_config, list):
            scenario_config = [scenario_config]

        self.scenario_config = MacadScenario.resolve_scenarios_parameter(
            scenario_config
        )
        self.scenario_runner = ScenarioRunner()
        self._scenario_iter = iter(self.scenario_config)
        self._scenario_map = {}

        self._fixed_delta_seconds = self._env_config.get("fixed_delta_seconds", 0.05)
        self._start_pos = {}
        self._end_pos = {}

    def get_start_pos(self, actor_id: str = None) -> Union[Tuple[float, ...], dict]:
        """Get start position

        Args:
            actor_id (str, optional): Provide actor id to get specific actor's start position. Otherwise,
            will return all actors' start position. Defaults to None.

        Returns:
            position (tuple[float, ...] | dict)
        """
        return self._start_pos[actor_id] if actor_id else self._start_pos

    def get_end_pos(self, actor_id: str = None) -> Union[Tuple[float, ...], dict]:
        """Get end position

        Args:
            actor_id (str, optional): Provide actor id to get specific actor's end position. Otherwise,
            will return all actors' end position. Defaults to None.

        Returns:
            position (tuple[float, ...] | dict)
        """
        if actor_id is not None:
            # 1. Return end position if it exists
            # 2. Return start position if no end position specify (static actor)
            # 3. Return (0, 0, 0) if no start position specify (Not an active actor)
            return self._end_pos.get(actor_id, self._start_pos.get(actor_id, (0, 0, 0)))

        return self._end_pos

    def get_scenario_map(self) -> dict:
        """Return current scenario map"""
        return self._scenario_map

    def get_weather_distribution(self) -> list:
        """Return the weather distribution of current scenario"""
        return self._scenario_map.get("weather_distribution", -1)

    def get_weather_spec(self) -> dict:
        """Return the extra weather spec of current scenario"""
        return self._scenario_map.get("weather_spec", {})

    def get_max_steps(self) -> int:
        """Return the max steps of current scenario"""
        return self._scenario_map.get("max_steps", float("inf"))

    def get_num_npcs(self, type: str = "vehicle") -> int:
        """Return the number of NPCs of current scenario"""
        if type in ["vehicle", "car", "vehicles"]:
            return self._scenario_map.get("num_vehicles", 0)
        elif type in ["walker", "pedestrian", "pedestrians"]:
            return self._scenario_map.get("num_walkers", 0)

        return 0

    def get_traffic_pivot(self) -> Tuple[float, ...]:
        """Return the traffic pivot of current scenario"""
        pivot = self._scenario_map.get("traffic_pivot", "ego")
        if isinstance(pivot, str):
            return self.get_start_pos(pivot)

        return pivot

    def get_traffic_spawn_range(self) -> float:
        """Return the traffic spawn range of current scenario"""
        return self._scenario_map.get("traffic_spawn_range", float("inf"))

    def set_start_pos(self, actor_id: str, pos: Union[tuple, list, dict]):
        """Set start position for an actor

        Args:
            actor_id (str): Actor id
            pos: Start position
        """
        self._start_pos[actor_id] = ScenarioHelper.parse_location(
            pos, Simulator.get_map()
        )

    def set_end_pos(self, actor_id: str, pos: "tuple | list | dict"):
        """Set end position for an actor

        Args:
            actor_id (str): Actor id
            pos: End position
        """
        self._end_pos[actor_id] = ScenarioHelper.parse_location(
            pos, Simulator.get_map()
        )

    def tick_scenario(self):
        """Tick the behavior tree to let background actors move"""
        scenario_tree = self.scenario_runner.manager.scenario_tree
        if scenario_tree and scenario_tree.status != py_trees.common.Status.FAILURE:
            scenario_tree.tick_once()

    def load_next_scenario(self) -> dict:
        """Get next scenario

        Returns:
            MacadScenario (dict): Scenario map
        """
        scenario = next(self._scenario_iter, random.choice(self.scenario_config))

        if not isinstance(scenario, dict):
            raise ValueError("Unsupported scenario type")

        if "xosc" in scenario:
            sceanrio_map = self.load_scenario_runner(scenario)
        else:
            sceanrio_map = self.load_macad_scenario(scenario)

        self._scenario_map = sceanrio_map

        map = Simulator.get_map()
        for actor_id, conf in sceanrio_map["actors"].items():
            self._start_pos[actor_id] = ScenarioHelper.parse_location(
                conf["start"], map
            )
            self._end_pos[actor_id] = (
                ScenarioHelper.parse_location(conf["end"], map)
                if "end" in conf
                else self._start_pos[actor_id]
            )

        if (
            "ego" in sceanrio_map["actors"]
            and "ego_end" in sceanrio_map["actors"]["ego"]
            and Simulator._redis
        ):
            Simulator._redis.set(
                "ego_end",
                ",".join(
                    [
                        str(n)
                        for n in ScenarioHelper.parse_location(
                            sceanrio_map["actors"]["ego"]["ego_end"], map
                        )
                    ]
                ),
            )

        return sceanrio_map

    def load_macad_scenario(self, scenario: dict) -> dict:
        """Load macad scenario

        Args:
            scenario (dict): Original MACAD style scenario config

        Returns:
            scenario (dict): The updated scenario
        """
        if scenario.get("max_time", None):
            scenario.update(
                {
                    "max_steps": int(
                        scenario["max_time"]
                        / (self._fixed_delta_seconds * ENV_ASSETS.step_ticks)
                    )
                }
            )

        return scenario

    def load_scenario_runner(self, scenario: dict):
        """Load scenario runner

        Args:
            scenario (dict): Scenario config with xosc file path

        Returns:
            ScenarioManager: Scenario manager
        """
        xosc_file = scenario["xosc"]
        params = scenario.get("params", None)
        if params is not None:
            params = self._parse_xosc_params(params)

        osc_conf = self._get_osc_conf(xosc_file, params)
        self.scenario_runner._load_scenario(osc_conf, clean_on_failed=False)

        scenario_map = {
            "actors": {},
            "map": osc_conf.town,
            "max_steps": scenario.get("max_steps", None) or 200,
            "weather_distribution": -1,  # Do not set weather again
        }

        for actor in osc_conf.ego_vehicles + osc_conf.other_actors:
            rolename = actor.rolename.lower()
            if (
                "static" in rolename
                or "static" in actor.model
                or "misc" in actor.category
            ):
                continue

            actor_id = rolename if rolename not in ["hero", "ego"] else "ego"
            start_loc = actor.transform.location
            end_loc = scenario.get("actors", {}).get(actor_id, {}).get("end", None)
            if end_loc is None:
                end_wpt = Simulator.get_lane_end(Simulator.get_map(), start_loc)
                end_loc = (
                    end_wpt.transform.location.x,
                    end_wpt.transform.location.y,
                    end_wpt.transform.location.z,
                )

            scenario_map["actors"][actor_id] = {
                "start": (start_loc.x, start_loc.y, start_loc.z),
                "end": end_loc,
            }

            if actor_id == "ego":
                ego_end = (
                    scenario.get("actors", {}).get(actor_id, {}).get("ego_end", None)
                )
                scenario_map["actors"]["ego"]["ego_end"] = ego_end or end_loc

        return scenario_map

    @lru_cache
    def _get_osc_conf(self, scenario: str, params: str = None):
        """Parse OSC config from xosc file
        Args:
            scenario (str): Path to xosc file
            params (str, optional): OSC params, in format: "param1: value1, param2: value2". Defaults to None.

        Returns:
            OpenScenarioConfiguration: OSC config
        """
        if not scenario.endswith(".xosc"):
            raise ValueError("Scenario file must be a .xosc file")

        self.scenario_runner._args.openscenario = scenario
        self.scenario_runner._args.openscenarioparams = params
        osc_conf = self.scenario_runner._load_openscenario(clean_on_failed=False)
        if not osc_conf:
            raise ValueError("Scenario file is not valid")

        return osc_conf

    def _parse_xosc_params(self, params: Union[dict, str]):
        """Parse OSC params from string

        Args:
            params (dict | str): OSC params dict or path to json file

        Returns:
            str: OSC params
        """
        if isinstance(params, str) and params.endswith(".json"):
            with open(params, "r") as f:
                params = json.load(f)

        if not isinstance(params, dict):
            raise ValueError("OSC params must be a dict or a json file")

        res = {}
        for k, v in params.items():
            if isinstance(v, (list, tuple)):
                res[k] = random.uniform(v[0], v[-1])

        res = ",".join([f"{k}:{v}" for k, v in res.items()])
        return res
