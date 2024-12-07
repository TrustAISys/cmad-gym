from __future__ import absolute_import, annotations, division

import logging
import math
import os
import random
import time
from collections import Counter, defaultdict, deque
from copy import deepcopy
from typing import Deque, Dict, Optional, Set, Tuple

import gym
import numpy as np
from tabulate import tabulate

from cmad.agent.action import AbstractAction, EnvAction
from cmad.agent.done import Done
from cmad.agent.obs import EnvObs
from cmad.agent.reward import Reward, RewardState
from cmad.envs import Analyzer, Scenarios, ENV_ASSETS, SYS_ASSETS
from cmad.misc import change_logging_level
from cmad.simulation.agents import AgentWrapper, HumanAgent, CmadAgent, RLAgent
from cmad.simulation.data import (
    CollisionRecord,
    DataCollector,
    ExpInfo,
    Measurements,
    Replay,
    Simulator,
)
from cmad.simulation.data.local_carla_api import Vector3D, Waypoint
from cmad.simulation.maps.nav_utils import PathTracker
from cmad.viz import Render

logger = logging.getLogger(__name__)


Base = gym.Env
try:
    from ray.rllib.env import MultiAgentEnv

    Base = MultiAgentEnv
except ImportError:
    logger.warning("\n Disabling RLlib support.")


class MultiCarlaEnv(Base):
    _gym_disable_underscore_compat = True

    def __init__(self, configs: Optional[dict] = None):
        """CMAD environment implementation.

        Args:
            configs (dict): Configuration for environment specified under the
                `env` key and configurations for each actor specified as dict
                under `actor`.
        """

        if configs is None:
            configs = deepcopy(ENV_ASSETS.default_multienv_config)

        self.configs = deepcopy(configs)
        self.env_config: dict = self.configs["env"]
        self.actor_configs: Dict[str, dict] = self.configs["actors"]
        self.background_actor_ids: Set[str] = self._get_background_actors()

        # Unify units, E.g. convert km/h to m/s
        for actor_config in self.actor_configs.values():
            if "target_speed" in actor_config:
                actor_config["target_speed"] = actor_config["target_speed"] / 3.6
            if "initial_speed" in actor_config:
                init_speed = actor_config["initial_speed"]
                if isinstance(init_speed, (tuple, list)):
                    actor_config["initial_speed"] = (
                        init_speed[0] / 3.6,
                        init_speed[1] / 3.6,
                    )
                else:
                    actor_config["initial_speed"] = init_speed / 3.6

        # At most one actor can be manual controlled
        manual_control_count = 0
        for _, actor_config in self.actor_configs.items():
            if actor_config.get("manual_control", False):
                if "vehicle" not in actor_config.get("type", "vehicle_4W"):
                    raise ValueError("Only vehicles can be manual controlled.")
                manual_control_count += 1

        assert manual_control_count <= 1, (
            "At most one actor can be manually controlled. "
            "Found %d actors with manual_control=True" % manual_control_count
        )

        # Camera position is problematic for certain vehicles and even in
        # autopilot they are prone to error
        self.exclude_hard_vehicles = True
        # list of str: Supported values for `type` filed in `actor_configs`
        # for actors than can be actively controlled
        self._supported_active_actor_types = [
            "vehicle",
            "vehicle_4W",
            "vehicle_2W",
            "pedestrian",
            "walker",
            "traffic_light",
        ]
        # list of str: Supported values for `type` field in `actor_configs`
        # for actors that are passive. Example: A camera mounted on a pole
        self._supported_passive_actor_types = [
            "camera",
            "obstacle",
            "static_vehicle",
            "static_walker",
            "static_pedestrian",
            "static_object",
            "static_obstacle",
        ]

        # Set attributes as in gym's specs
        self.reward_range = (-float("inf"), float("inf"))
        self.metadata = {"render.modes": "human"}

        # Belongs to env_config. Optional parameters are retrieved with .get()
        self._server_port: Optional[int] = self.env_config.get("server_port", None)
        self._server_ip: str = self.env_config.get("server_ip", "localhost")
        self._server_map: str = self.env_config["server_map"]
        self._map_name: str = self._server_map.split("/")[-1]
        self._reload: bool = self.env_config.get("reload", False)
        self._use_hard_reset: bool = self.env_config.get("hard_reset", False)
        self._render_simulator: bool = self.env_config.get("render", False)
        self._sync_server: bool = self.env_config.get("sync_server", True)
        self._fixed_delta_seconds: float = self.env_config.get(
            "fixed_delta_seconds", 0.05
        )
        self._enable_traffic_light: bool = self.env_config.get(
            "enable_traffic_light", True
        )
        self._global_obs_conf: dict = self.env_config.get("global_observation", None)
        self._verbose: bool = self.env_config.get("verbose", False)
        if self._verbose:
            change_logging_level(logging.DEBUG)

        self._spec = lambda: None
        self._spec.id = "Carla-v0"
        self._frame_cnt: int = None
        self._weather_spec: list = None
        self._replay: Replay = None
        self._last_done_reason: str = "UNKNOWN"
        self._prev_measurements: Measurements = None
        self._measurement_hist: Deque[Measurements] = deque(
            [{}], maxlen=max(10, ENV_ASSETS.step_ticks)
        )

        # Actions Space
        self.env_action = EnvAction(self)
        self.action_space = self.env_action.get_action_space()
        self.agent_actions = self.env_action.get_agent_actions()
        self._action_map: Dict[str, AbstractAction] = {}

        # Observation Space
        self.env_obs = EnvObs(self)
        self.observation_space = self.env_obs.get_observation_space()

        # Reward function
        self.reward_state = RewardState(self._measurement_hist)

        # Scenario
        self.scenarios = Scenarios(self.configs["scenarios"], self.env_config)
        self.episode_id: int = 0

        # Experiment data collection
        self._obs_dict = defaultdict(lambda: None)
        self._done_dict = defaultdict(bool)
        self._previous_actions = defaultdict(lambda: None)
        self._previous_rewards = defaultdict(float)
        self._num_steps = defaultdict(int)
        self._total_reward = defaultdict(float)

        self._agents: Dict[str, AgentWrapper] = {}
        """Dictionary of agents with actor_id as key"""
        self._actors: Dict[str, int] = {}
        """Dictionary of actor.id with actor_id as key"""
        self._path_trackers: Dict[str, PathTracker] = {}
        """Dictionary of PathTrackers with actor_id as key"""
        self._cause_counter: Counter = Counter()
        """Counter for the reason of episode end"""

        # Render related
        self._human_agent: dict = None
        render_x_res: int = self.env_config.get("render_x_res", 800)
        render_y_res: int = self.env_config.get("render_y_res", 600)
        obs_x_res: int = self.env_obs._obs_x_res
        obs_y_res: int = self.env_obs._obs_y_res
        for config in self.actor_configs.values():
            config.update({"x_res": obs_x_res, "y_res": obs_y_res})

        Render.resize_screen(render_x_res, render_y_res)
        self._camera_poses, window_dim = Render.get_surface_poses(
            [obs_x_res, obs_y_res], self.actor_configs
        )

        if manual_control_count == 0:
            Render.resize_screen(window_dim[0], window_dim[1])
            self._manual_control_render_pose = None
        else:
            self._manual_control_render_pose = (0, window_dim[1])
            Render.resize_screen(
                max(render_x_res, window_dim[0]),
                render_y_res + window_dim[1],
            )

        if self._global_obs_conf and self._global_obs_conf.get("render", False):
            self._global_view_pos = (
                (Render.resX, window_dim[1])
                if manual_control_count != 0
                else (0, window_dim[1])
            )

            Render.resize_screen(
                (
                    max(Render.resX, self._global_obs_conf["x_res"])
                    if manual_control_count == 0
                    else Render.resX + self._global_obs_conf["x_res"]
                ),
                max(Render.resY, self._global_obs_conf["y_res"] + window_dim[1]),
            )
        else:
            self._global_view_pos = None
        self._global_sensor = None

        # A list is False if empty
        self.render_required = (
            (
                True
                if [
                    k
                    for k, v in self.actor_configs.items()
                    if v.get("render", False) and v.get("camera_type", False)
                ]
                else False
            )
            or (
                self._global_obs_conf is not None
                and self._global_obs_conf.get("render", False)
            )
            or manual_control_count != 0
        )

    def reset(self) -> Dict[str, dict]:
        """Reset the carla world, call _init_server()

        Returns:
            obs (dict): Initial observation
        """
        if self.env_config.get("record", False) and Simulator.ready:
            Simulator.stop_recorder(add_prefix=self._last_done_reason)

        # Cleanup episode related data
        self.reward_state.reset()
        self._done_dict["__all__"] = False

        for retry in range(ENV_ASSETS.retries_on_error):
            try:
                if not Simulator.ready:
                    Simulator.setup(self)
                    self.scenarios.load_next_scenario()

                    if Simulator._redis:
                        Simulator._redis.set("reset", "yes")
                    self._hard_reset(clean_world=False)
                else:
                    self.scenarios.load_next_scenario()

                    if Simulator._redis:
                        # The ego's camera must be removed before Pylot end
                        Simulator.toggle_sync_mode(False)
                        self._reset_ego()
                        Simulator._redis.set("reset", "yes")
                        time.sleep(2)

                    if not self._check_actors() or self._use_hard_reset:
                        self._hard_reset(clean_world=True)
                    else:
                        self._soft_reset()
                break
            except RuntimeError as e:
                logger.exception("Error during MultiCarlaEnv.reset")
                logger.warning(
                    "reset(): Retry #: %d/%d", retry + 1, ENV_ASSETS.retries_on_error
                )

                if len(self._actors) < len(self.actor_configs):
                    Simulator.cleanup()
                    Simulator.ready = False

                if retry + 1 == ENV_ASSETS.retries_on_error:
                    raise e

        # Prepare initial speed command batch
        set_speed_batch = []
        for actor_id, id in self._actors.items():
            init_speed = self.actor_configs[actor_id].get("initial_speed", None)
            if init_speed:
                init_speed = (
                    random.uniform(*init_speed)
                    if isinstance(init_speed, tuple)
                    else init_speed
                )
                set_speed_batch.append(
                    Simulator.set_actor_speed(id, init_speed, ret_command=True)
                )

        if Simulator._redis:
            wait_time = 60
            while wait_time:
                if Simulator._redis.get("START_EGO") == "1":
                    break
                time.sleep(0.05)
                wait_time -= 0.05
                if wait_time <= 0:
                    logger.warning("No response from Pylot, re-raising signal...")
                    self.scenarios.load_next_scenario()
                    self._soft_reset()
                    self._reset_ego()
                    Simulator._redis.set("reset", "yes")
                    wait_time = 60

        if self.env_config.get("record", False):
            Simulator.start_recorder(
                os.path.join(SYS_ASSETS.paths.output, f"{self.episode_id}.log")
            )

        if not Simulator._redis:
            for _ in range(10):
                Simulator.tick()

        Simulator.send_batch(set_speed_batch)
        self._frame_cnt = Simulator.tick()

        self._prev_measurements = py_measurements = self._read_observation()
        self._measurement_hist.append(py_measurements)

        self._total_reward.clear()
        self._num_steps.clear()
        self._done_dict.clear()

        # Get initial observation
        for actor_id, actor_config in self.actor_configs.items():
            if actor_id in self.background_actor_ids:
                continue

            action_mask = self.agent_actions[actor_id].get_action_mask(
                Simulator.get_actor_by_id(self._actors[actor_id])
            )
            obs = self.env_obs.encode_obs(actor_id, py_measurements, action_mask)
            self._obs_dict[actor_id] = obs

        return self._obs_dict

    def replay(self, file: str, follow_vehicle: Optional[str] = None):
        """Replay a recorded file

        Args:
            file (str): Path to the file to replay
            follow_vehicle (str): role_name of the vehicle to follow

        Returns:
            obs (dict): Initial observation
            steps (int): how many steps the replay will took based on configured fixed_delta_seconds

            Note: env.step() will tick STEP_TICKS times each time it is called.
        """
        if not Simulator.ready:
            Simulator.setup(self)

        self.scenarios.load_next_scenario()

        # This will toggle world to sync mode and replay the file
        replay, carla_actors, start_time = Simulator.replay_file(file, follow_vehicle)
        self._replay = replay

        self._weather_spec = Simulator.set_weather(
            self.scenarios.get_weather_distribution(),
            self.scenarios.get_weather_spec(),
        )

        # Use "role_name" to map correct actor_id
        for actor in replay.actors:
            role_name = actor.attributes.get("role_name", None)
            for actor_id in self.actor_configs:
                actor_type = self.actor_configs[actor_id].get("type", "vehicle_4W")
                if (role_name == actor_id) or (role_name in ["hero", "ego"]):
                    self._actors[actor_id] = actor.id
                    self.scenarios.set_start_pos(actor_id, actor.spawn_point)

                    if "static" not in actor_type:
                        lane_end_waypoint = Simulator.get_lane_end(
                            Simulator.get_map(), self.scenarios.get_start_pos(actor_id)
                        )
                        self.scenarios.set_end_pos(
                            actor_id,
                            (
                                lane_end_waypoint.transform.location.x,
                                lane_end_waypoint.transform.location.y,
                                lane_end_waypoint.transform.location.z,
                            ),
                        )
                    else:
                        self.scenarios.set_end_pos(actor_id, actor.spawn_point)
                    break

        if len(self._actors) != len(self.actor_configs):
            raise ValueError(
                "Not all actors in the scenario are found in the replay file"
            )

        id_to_name = {v: k for k, v in self._actors.items()}
        for actor in carla_actors:
            actor_id = id_to_name[actor.id]
            actor_config = self.actor_configs[actor_id]
            Simulator.register_actor(actor, add_to_pool=True)

            if actor_id not in self.background_actor_ids:
                actor_config.update(
                    {
                        "actor_id": actor_id,
                        "id": actor.id,
                        "action_handler": self.agent_actions[actor_id],
                    }
                )

                if actor_config.get("enable_planner", False):
                    try:
                        self._path_trackers[actor_id] = PathTracker(
                            actor,
                            origin=self.scenarios.get_start_pos(actor_id),
                            destination=self.scenarios.get_end_pos(actor_id),
                            target_speed=actor_config.get("target_speed", 10),
                            opt_dict=actor_config.get("opt_dict", None),
                            planned_route=actor_config.get("planned_route_path", None),
                        )
                    except:
                        logger.warning("Unable to plan path for actor: %s", actor_id)
                        actor_config["enable_planner"] = False

                # Spawn collision and lane sensors if necessary
                if actor_config.get("camera_type", None) == "bev":
                    Simulator.register_birdeye_sensor(
                        (actor_config["x_res"], actor_config["y_res"])
                    )
                if actor_config.get("collision_sensor", "off") == "on":
                    Simulator.register_collision_sensor(actor_id, actor)
                if actor_config.get("lane_sensor", "off") == "on":
                    Simulator.register_lane_invasion_sensor(actor_id, actor)

                agent = AgentWrapper(RLAgent(actor_config))
                agent.setup_sensors(actor)
                self._agents[actor_id] = agent

        # Global observation if necessary
        if self._global_obs_conf:
            self._global_obs_conf.update({"actor_id": "global"})
            global_agent = AgentWrapper(CmadAgent(self._global_obs_conf))

            if self._global_obs_conf.get("camera_type", "rgb") == "bev":
                Simulator.register_birdeye_sensor(
                    (self._global_obs_conf["x_res"], self._global_obs_conf["y_res"])
                )
                global_agent.setup_sensors(
                    Simulator.get_actor_by_id(
                        self._actors[self._global_obs_conf.get("attach_to", "ego")]
                    )
                )
            elif "attach_to" in self._global_obs_conf:
                global_agent.setup_sensors(
                    Simulator.get_actor_by_id(
                        self._actors[self._global_obs_conf["attach_to"]]
                    )
                )
            else:
                global_agent.setup_sensors(None)

            self._agents["global"] = global_agent
            self._global_sensor = global_agent.sensors()[0]

        # Wait for the first frame to be ready
        for actor_id in self._actors:
            actor_camera_id = self.actor_configs[actor_id].get("camera_type", "bev")
            if actor_camera_id and actor_camera_id != "bev":
                while Simulator.get_actor_camera_data(actor_id) is None:
                    self._frame_cnt = Simulator.tick()
            else:
                self._frame_cnt = Simulator.tick()

        self._prev_measurements = py_measurements = self._read_observation()
        self._measurement_hist.append(py_measurements)

        for actor_id, actor_config in self.actor_configs.items():
            if actor_id in self.background_actor_ids:
                continue

            # Reset attributes
            self._total_reward[actor_id] = 0
            self._num_steps[actor_id] = 0
            self._done_dict[actor_id] = False

            # Get initial observation
            action_mask = self.agent_actions[actor_id].get_action_mask(
                Simulator.get_actor_by_id(self._actors[actor_id])
            )
            obs = self.env_obs.encode_obs(actor_id, py_measurements, action_mask)
            self._obs_dict[actor_id] = obs

        ticks = (replay.duration - start_time) / self._fixed_delta_seconds
        self.episode_id += 1
        return self._obs_dict, int(ticks / ENV_ASSETS.step_ticks)

    def step(
        self, action_dict: dict, replay: bool = False
    ) -> Tuple[Dict[str, dict], Dict[str, float], Dict[str, bool], Dict[str, dict]]:
        """Execute one environment step for the specified actors.

        Executes the provided action for the corresponding actors in the
        environment and returns the resulting environment observation, reward,
        done and info (measurements) for each of the actors.

        Args:
            action_dict (dict): Actions to be executed for each actor. Keys are
                agent_id strings, values are corresponding actions.
            replay (bool): If True, then the action is ignored. Will retreive the last action from the simulator.

        Returns
            obs (dict): Observations for each actor.
            rewards (dict): Reward values for each actor. None for first step
            dones (dict): Done values for each actor. Special key "__all__" is
            set when all actors are done and the env terminates
            info (dict): Info for each actor.

        Raises
            RuntimeError: If `step(...)` is called before calling `reset()`
            ValueError: If `action_dict` is not a dictionary of actions
            ValueError: If `action_dict` contains actions for nonexistent actor
        """

        if not Simulator.ready:
            raise RuntimeError("Cannot call step(...) before calling reset()")

        assert len(self._actors), (
            "No actors exist in the environment. Either"
            " the environment was not properly "
            "initialized using`reset()` or all the "
            "actors have exited. Cannot execute `step()`"
        )

        if self._verbose:
            logger.debug(
                "\n%s",
                tabulate(
                    action_dict.items(), headers=["Actor", "Action"], tablefmt="pretty"
                ),
            )

        actions = self.env_action.check_validity(action_dict, self._done_dict)
        try:
            obs_dict = {}
            reward_dict = {}
            done_dict = {actor_id: False for actor_id in action_dict}
            info_dict = {}
            applied_action = {}

            # Convert action to abstract action
            for actor_id, action in actions.items():
                self._action_map[actor_id] = self.agent_actions[
                    actor_id
                ].convert_single_action(action, self._done_dict[actor_id])

            # Take action
            if not replay:
                current_frame = self._frame_cnt
                while self._frame_cnt - current_frame < ENV_ASSETS.step_ticks:
                    # Send control command
                    applied_action = self._step_before_tick()
                    # Tick to take effect
                    self._tick_until(ENV_ASSETS.action_ticks)
                    # Collect data and render
                    measurements = self._step_after_tick(action_dict, applied_action)

                    if all(act.done() for act in self._action_map.values()):
                        break
            else:
                # Get action after tick if replay (so its the latest action)
                self._tick_until(ENV_ASSETS.action_ticks)
                for actor_id, action in actions.items():
                    applied_action[actor_id] = Simulator.get_actor_control(
                        self._actors[actor_id]
                    )
                measurements = self._step_after_tick(action_dict, applied_action)

            if self._verbose:
                table = [
                    [
                        self.episode_id,
                        actor_id,
                        measurement.transform,
                        measurement.velocity,
                        measurement.acceleration,
                        measurement.planned_waypoint,
                        measurement.collision,
                        measurement.lane_invasion,
                        measurement.control,
                    ]
                    for actor_id, measurement in measurements.items()
                ]
                logger.debug(
                    "\n%s",
                    tabulate(
                        table, headers=ENV_ASSETS.verbose_info_header, tablefmt="pretty"
                    ),
                )

            # Get observations
            for actor_id, action in applied_action.items():
                if actor_id not in action_dict:
                    continue

                obs, reward, done, info = self._get_state(actor_id, measurements)
                obs_dict[actor_id] = obs
                reward_dict[actor_id] = reward
                info_dict[actor_id] = info
                done_dict[actor_id] = done

                # Note: self._done_dict is an activation flag for each actor
                # we will return done=True for an actor only when all actors
                # are done. This is to ensure that the environment does not
                # terminate prematurely when some actors are done and others don't
                if not self._done_dict.get(actor_id, False):
                    self._done_dict[actor_id] = done

            ego_reward = reward_dict.get("ego", 0)
            for actor_id in reward_dict:
                reward_dict[actor_id] += ego_reward

            self._done_dict["__all__"], done_type = Done.is_done(
                "episode",
                measurements,
                self.env_config.get(
                    "done_criteria", ENV_ASSETS.default_episode_done_criteria
                ),
                return_reason=True,
            )

            # Actors will done together
            if self._done_dict["__all__"]:
                done_dict = self._done_dict
                self._last_done_reason = Analyzer.causal_analysis(
                    measurements,
                    self._measurement_hist,
                    {v: k for k, v in self._actors.items()},
                    done_type,
                )
                self._cause_counter[self._last_done_reason] += 1

                if self.episode_id % 100 == 0:
                    causes = tabulate(
                        self._cause_counter.most_common(),
                        headers=["Cause", "Count"],
                        tablefmt="pretty",
                    )
                    logger.info("\n%s", causes)
                    self._cause_counter.clear()
            else:
                done_dict["__all__"] = False

            self._prev_measurements = measurements

            return obs_dict, reward_dict, done_dict, info_dict
        except Exception as e:
            if Simulator.ready:
                self._clean_world()
                Simulator.clear_server_state()
            raise RuntimeError("Error during step, terminating episode early.") from e

    def render(self, mode: str = "human"):
        logger.warning(
            "Render is toggled through config file. Ignoring `render()` call."
        )

    def _render(self, image_dict: Dict[str, np.ndarray]):
        """Render the environment."""

        images = {
            k: self.env_obs.decode_image(k, v)
            for k, v in image_dict.items()
            if k != "global"
            and (
                self.actor_configs[k].get("render", False)
                and self.actor_configs[k].get("camera_type", False)
            )
        }
        Render.multi_view_render(images, self._camera_poses)

        if self._global_view_pos:
            Render.draw(image_dict["global"], self._global_view_pos, flip=True)

        if self._human_agent is None:
            Render.dummy_event_handler()
        else:
            agent: HumanAgent = self._human_agent["agent"]
            agent()  # render camera
            agent._hic._hud.tick(
                Simulator.get_actor_by_id(self._human_agent["id"]),
                Simulator.get_actor_collision_sensor(self._human_agent["actor_id"]),
                agent._hic._clock,
            )

    def close(self):
        """Clean-up the world, clear server state & close the Env"""

        if len(self._cause_counter) > 0:
            causes = tabulate(
                self._cause_counter.most_common(),
                headers=["Cause", "Count"],
                tablefmt="pretty",
            )
            logger.info("\n%s", causes)
            self._cause_counter.clear()

        if Simulator.ready:
            if self.env_config.get("record", False):
                Simulator.stop_recorder(add_prefix=self._last_done_reason)

            self._clean_world()
            Simulator.clear_server_state()

        import pygame

        if pygame.get_init():
            pygame.quit()

    def _soft_reset(self):
        """Soft reset by moving existing actors to their initial positions."""

        self._weather_spec = Simulator.set_weather(
            self.scenarios.get_weather_distribution(),
            self.scenarios.get_weather_spec(),
        )

        # teleport to start position
        carla_map = Simulator.get_map()
        batches = []
        for actor_id, id in self._actors.items():
            transform = Simulator.generate_spawn_point(
                carla_map, self.scenarios.get_start_pos(actor_id)
            )
            batches.extend(Simulator.teleport_actor(id, transform, ret_command=True))
        Simulator.send_batch(batches, tick=10)

        # plan path
        for actor_id, id in self._actors.items():
            actor_config = self.actor_configs[actor_id]
            start_pos = self.scenarios.get_start_pos(actor_id)
            end_pos = self.scenarios.get_end_pos(actor_id)

            if actor_config.get("enable_planner", False):
                self._path_trackers[actor_id].plan_route(start_pos, end_pos)

            logger.info(
                "Actor: %s start_pos_xyz: %s, end_pos_xyz: %s",
                actor_id,
                start_pos,
                end_pos,
            )

        # Reset properties of the actors
        Simulator.cleanup(soft_reset=True)

        self.episode_id += 1
        logger.info("New episode initialized with actors: %s", self._actors.keys())

    def _hard_reset(self, clean_world=True):
        """Hard reset episode by re-spawning all actors (including sensor)

        Args:
            clean_world (bool): If True, clean the world before spawning actors

        Raises:
            RuntimeError: If spawning an actor at its initial state as per its'
            configuration fails (eg.: Due to collision with an existing object
            on that spot). This Error will be handled by the caller
            `self.reset()` which will perform a "hard" reset by creating
            a new server instance
        """
        if clean_world:
            self._clean_world()

        self._weather_spec = Simulator.set_weather(
            self.scenarios.get_weather_distribution(),
            self.scenarios.get_weather_spec(),
        )

        for actor_id, actor_config in self.actor_configs.items():
            actor_spawned = Simulator.get_actor_by_rolename(
                actor_config.get("rolename", actor_id), from_world=False
            )
            if not actor_config.get("spawn", True):
                # Vehicle is already spawned in the world
                while actor_spawned is None:
                    actor_spawned = Simulator.get_actor_by_rolename(
                        actor_config.get("rolename", actor_id)
                    )
                    if actor_spawned is None:
                        self._frame_cnt = Simulator.tick()

                Simulator.register_actor(actor_spawned, add_to_pool=True)
                self._actors[actor_id] = actor_spawned.id
            elif not actor_spawned:
                # Try to spawn actor or fail and reinitialize the server before get back here
                try:
                    actor_spawned = self._spawn_new_actor(actor_id)
                    self._actors[actor_id] = actor_spawned.id
                except RuntimeError as spawn_err:
                    # Chain the exception & re-raise to be handled by the caller `self.reset()`
                    raise spawn_err from RuntimeError(
                        f"Unable to spawn actor: {actor_id}"
                    )
            else:
                self._actors[actor_id] = actor_spawned.id

            start_pos = self.scenarios.get_start_pos(actor_id)
            end_pos = self.scenarios.get_end_pos(actor_id)
            if actor_id not in self.background_actor_ids:
                actor_config.update(
                    {
                        "actor_id": actor_id,
                        "id": self._actors[actor_id],
                        "action_handler": self.agent_actions[actor_id],
                    }
                )

                if actor_config.get("enable_planner", False):
                    self._path_trackers[actor_id] = PathTracker(
                        actor_spawned,
                        origin=start_pos,
                        destination=end_pos,
                        target_speed=actor_config.get("target_speed", 10),
                        opt_dict=actor_config.get("opt_dict", None),
                        planned_route=actor_config.get("planned_route_path", None),
                    )

                # Spawn collision and lane sensors if necessary
                if actor_config.get("camera_type", None) == "bev":
                    Simulator.register_birdeye_sensor(
                        (actor_config["x_res"], actor_config["y_res"])
                    )
                if actor_config.get("collision_sensor", "off") == "on":
                    Simulator.register_collision_sensor(actor_id, actor_spawned)
                if actor_config.get("lane_sensor", "off") == "on":
                    Simulator.register_lane_invasion_sensor(actor_id, actor_spawned)

                if not actor_config.get("manual_control", False):
                    agent = AgentWrapper(RLAgent(actor_config))
                    agent.setup_sensors(actor_spawned)
                else:
                    actor_config.update(
                        {
                            "render_config": {
                                "width": self.env_config["render_x_res"],
                                "height": self.env_config["render_y_res"],
                                "render_pos": self._manual_control_render_pose,
                            }
                        }
                    )
                    agent = AgentWrapper(HumanAgent(actor_config))
                    agent.setup_sensors(actor_spawned)
                    self._human_agent = {
                        "agent": agent._agent,
                        "actor_id": actor_id,
                        "id": actor_spawned.id,
                    }
                self._agents[actor_id] = agent

            logger.info(
                "Actor: %s start_pos_xyz: %s, end_pos_xyz: %s",
                actor_id,
                start_pos,
                end_pos,
            )

        # Initialize global observation if defined
        if self._global_obs_conf:
            self._global_obs_conf.update({"actor_id": "global"})
            global_agent = AgentWrapper(CmadAgent(self._global_obs_conf))

            if self._global_obs_conf.get("camera_type", "rgb") == "bev":
                Simulator.register_birdeye_sensor(
                    (self._global_obs_conf["x_res"], self._global_obs_conf["y_res"])
                )
                global_agent.setup_sensors(
                    Simulator.get_actor_by_id(
                        self._actors[self._global_obs_conf.get("attach_to", "ego")]
                    )
                )
            elif "attach_to" in self._global_obs_conf:
                global_agent.setup_sensors(
                    Simulator.get_actor_by_id(
                        self._actors[self._global_obs_conf["attach_to"]]
                    )
                )
            else:
                global_agent.setup_sensors(None)

            self._agents["global"] = global_agent
            self._global_sensor = global_agent.sensors()[0]

        npc_vehicles = self.scenarios.get_num_npcs("vehicle")
        npc_pedestrians = self.scenarios.get_num_npcs("pedestrian")
        if npc_pedestrians + npc_vehicles > 0:
            Simulator.apply_traffic(
                npc_vehicles,
                npc_pedestrians,
                ref_position=self.scenarios.get_traffic_pivot(),
                spawn_range=self.scenarios.get_traffic_spawn_range(),
                safe=self.exclude_hard_vehicles,
            )

        self.episode_id += 1
        logger.info("New episode initialized with actors: %s", self._actors.keys())

    def _spawn_new_actor(self, actor_id: str):
        """Spawn an agent as per the blueprint at the given pose

        Args:
            blueprint: Blueprint of the actor. Can be a Vehicle or Pedestrian
            pose: carla.Transform object with location and rotation

        Returns:
            An instance of a subclass of carla.Actor. carla.Vehicle in the case
            of a Vehicle agent.

        """
        actor_config = self.actor_configs[actor_id]
        actor_type = actor_config.get("type", "vehicle_4W")
        if (actor_type not in self._supported_active_actor_types) and (
            actor_type not in self._supported_passive_actor_types
        ):
            logger.info(
                "Unsupported actor type: %s. Using vehicle_4W as the type", actor_type
            )
            actor_type = "vehicle_4W"

        transform = Simulator.generate_spawn_point(
            Simulator.get_map(), self.scenarios.get_start_pos(actor_id)
        )
        model = actor_config.get("blueprint", None)

        if actor_type == "traffic_light":
            # Traffic lights already exist in the world & can't be spawned.
            # Find closest traffic light actor in world.actor_list and return
            from cmad.simulation.maps import get_tls

            tls = get_tls(Simulator.get_world(), transform, sort=True)
            #: Return the key (carla.TrafficLight object) of closest match
            return tls[0][0]

        if "vehicle" in actor_type:
            model = "vehicle.*" if model is None else model
            actor_type = "car"
        elif "bike" in actor_type or "bicycle" in actor_type:
            actor_type = "bike"
        elif "motor" in actor_type or "motorcycle" in actor_type:
            actor_type = "motorbike"
        elif "pedestrian" in actor_type or "walker" in actor_type:
            model = "walker.pedestrian.*" if model is None else model
            actor_type = "pedestrian"

        vehicle = None
        for retry in range(ENV_ASSETS.retries_on_error):
            vehicle = Simulator.request_new_actor(
                model,
                transform,
                rolename=actor_config.get("rolename", actor_id),
                autopilot=False,
                actor_category=actor_type,
                safe_blueprint=self.exclude_hard_vehicles,
                immortal=(not self._use_hard_reset),
            )

            if vehicle is not None:
                break

            logger.warning(
                "spawn_actor: Retry#: %d/%d", retry + 1, ENV_ASSETS.retries_on_error
            )

        if vehicle is None:
            raise RuntimeError(f"Failed to spawn actor: {actor_id}")

        return vehicle

    def _take_action(
        self, actor_id: str, action: AbstractAction, ret_command: bool = True
    ):
        """Perform the actual step in the CARLA environment

        Applies control to `actor_id` based on `action`, world.tick should be called afterwards.

        Args:
            actor_id (str): Actor identifier
            action (AbstractAction): Action to be executed for the actor
            ret_command (bool): If True, return the command to be executed.

        Returns:
            action_dict (dict): Action to be executed for the actor in dict.
            action_command (carla.command.ApplyVehicleControl): Action command to be executed.
        """
        cur_id = self._actors[actor_id]
        config = self.actor_configs[actor_id]
        convert_action = action.run_step(Simulator.get_actor_by_id(cur_id))
        carla_action = None

        if config.get("enable_planner", False):
            # update planned route, this will affect _read_observation()
            planned_action = self._path_trackers[actor_id].run_step()

        if config.get("manual_control", False):
            agent: HumanAgent = self._human_agent["agent"]
            if agent.use_autopilot:
                Simulator.toggle_actor_autopilot(self._human_agent["id"], True)
                carla_action = None
            else:
                Simulator.toggle_actor_autopilot(self._human_agent["id"], False)
                carla_action = agent()
        elif config.get("auto_control", False):
            if config.get("enable_planner", False):
                carla_action = planned_action
            else:
                Simulator.toggle_actor_autopilot(cur_id, True)
        else:
            carla_action = convert_action

        action_command = Simulator.apply_actor_control(
            cur_id, carla_action, ret_command=ret_command
        )
        return action.to_dict(), action_command

    def _read_observation(self, action_dict: dict = None, applied_action: dict = None):
        """This function will call DataCollector to read all observations.

        Args:
            action_dict (dict): Action to be exectued in _take_action.
            applied_action (dict): Action applied in _take_action. if None, ignored

        Returns:
            py_measurements (dict): Measurements of All actors.
        """

        # This contains raw data for each actor from CarlaDataProvider and SensorDataProvider
        py_measurements = DataCollector.get_data(self._actors, self.actor_configs)

        # If replaying, we need to update some info manually
        if self._replay is not None:
            self._update_info_from_replay(py_measurements)

        # We need to update some loacl data that cannot be directly read from DataCollector
        for actor_id, measurement in py_measurements.items():
            config = self.actor_configs[actor_id]
            start_pos = self.scenarios.get_start_pos(actor_id)
            end_pos = self.scenarios.get_end_pos(actor_id)

            if end_pos == start_pos:
                # This actor has no target in reaching a destination
                distance_to_goal_euclidean = -1
            else:
                distance_to_goal_euclidean = float(
                    np.linalg.norm(
                        [
                            measurement.transform.location.x - end_pos[0],
                            measurement.transform.location.y - end_pos[1],
                        ]
                    )
                )

            path_tracker = self._path_trackers.get(actor_id, None)
            if path_tracker:
                distance_to_goal = path_tracker.get_distance_to_end(False)
                orientation_diff = path_tracker.get_orientation_diff_to_end()
                planning = path_tracker.get_planner_path()

                if len(planning) > 1:
                    next_command = ENV_ASSETS.road_option_to_commands.get(
                        planning[0], "LANE_FOLLOW"
                    )
                elif (
                    0 <= distance_to_goal <= ENV_ASSETS.distance_to_goal_threshold
                    and orientation_diff <= ENV_ASSETS.orientation_to_goal_threshold
                ):
                    next_command = "REACH_GOAL"
                elif len(planning) == 0 or "rear" != PathTracker.relative_position(
                    planning[0][0].transform, measurement.transform
                ):
                    next_command = "PASS_GOAL"
                else:
                    next_command = "LANE_FOLLOW"

                distance_threshold = 2.0 if actor_id != "ego" else math.inf
                planned_waypoints = path_tracker.get_nearest_waypoints(
                    nums=7, interval=2.0, distance_threshold=distance_threshold
                )
            else:
                try:
                    wp = Simulator.get_actor_waypoint(self._actors[actor_id])
                    plan, _ = PathTracker.generate_target_waypoint_list(
                        wp, 0, 20, measurement.transform.rotation.yaw
                    )
                    planned_waypoints = [p[0] for p in plan][:7] if plan else [wp]
                    while len(planned_waypoints) < 7:
                        last_wpt = planned_waypoints[-1]
                        next_wpt = last_wpt.next(2.0)
                        planned_waypoints.append(next_wpt[0] if next_wpt else last_wpt)
                except:
                    planned_waypoints = [None] * 7

                distance_to_goal = distance_to_goal_euclidean
                if 0 <= distance_to_goal <= ENV_ASSETS.distance_to_goal_threshold:
                    next_command = "REACH_GOAL"
                else:
                    next_command = "LANE_FOLLOW"

            measurement.planned_waypoint = Waypoint.from_simulator_waypoint(
                planned_waypoints[0]
            )
            measurement.planned_waypoints = [
                Waypoint.from_simulator_waypoint(wpt) for wpt in planned_waypoints[2:]
            ]

            if planned_waypoints[0] is not None:
                measurement.orientation_diff = DataCollector.calculate_orientation_diff(
                    measurement.transform, measurement.planned_waypoint.transform
                )
                measurement.road_offset = (
                    measurement.planned_waypoint.transform.inverse_transform_location(
                        measurement.transform.location
                    ).y
                )

            # Local info
            measurement.exp_info = ExpInfo(
                distance_to_goal=distance_to_goal,
                distance_to_goal_euclidean=distance_to_goal_euclidean,
                next_command=next_command,
                target_speed=config.get("target_speed", 20 / 3.6),
                actor_in_scene=self._actors,
                episode_id=self.episode_id,
                step=self._num_steps[actor_id],
                max_steps=self.scenarios.get_max_steps(),
                step_time=ENV_ASSETS.step_ticks * self._fixed_delta_seconds,
                start_pos=self.scenarios.get_start_pos(actor_id),
                end_pos=self.scenarios.get_end_pos(actor_id),
                previous_action=self._previous_actions[actor_id],
                previous_reward=self._previous_rewards[actor_id],
            )

            if applied_action is not None:
                self._previous_actions[actor_id] = action_dict.get(actor_id, 0)
                measurement.control = applied_action.get(actor_id, None)

        return py_measurements

    def _get_state(self, actor_id: str, py_measurements: Measurements):
        """Process measurements and return 4-tuple of (obs, reward, done, info)

        Args:
            actor_id(str): Actor identifier
            action_dict (dict): Action exectued in _take_action.
            py_measurements (dict): Measurements of All actors.

        Returns
            obs (obs_space): Observation for the actor whose id is actor_id.
            reward (float): Reward for actor. None for first step
            done (bool): Done value for actor.
            info (dict): Info for actor.
        """

        # Process observations
        measurement = py_measurements[actor_id]

        # Compute done
        config = self.actor_configs[actor_id]
        done, reason = Done.is_done(
            actor_id,
            py_measurements,
            config.get("done_criteria", ENV_ASSETS.default_actor_done_criteria),
            return_reason=True,
        )
        measurement.exp_info.done = done
        if done and not self._done_dict[actor_id]:
            logger.info(
                "Episode %s, Actor %s done due to %s",
                self.episode_id,
                actor_id,
                reason,
            )

        # Compute reward
        reward_flag = config.get("reward_function", "corl2017")
        reward_params = config.get("reward_params", {})
        self.reward_state.update(self._prev_measurements, py_measurements)
        reward = Reward.compute_reward(
            actor_id,
            self.reward_state,
            reward_flag,
            **reward_params,
        )

        self._previous_rewards[actor_id] = reward
        self._total_reward[actor_id] += reward

        measurement.exp_info.reward = reward
        measurement.exp_info.total_reward = self._total_reward[actor_id]

        action_mask = self.agent_actions[actor_id].get_action_mask(
            Simulator.get_actor_by_id(self._actors[actor_id])
        )
        encode_obs = self.env_obs.encode_obs(actor_id, py_measurements, action_mask)

        self._num_steps[actor_id] += 1
        return (
            encode_obs,
            reward,
            done,
            measurement.as_dict(),
        )

    def _step_before_tick(self) -> Dict[str, dict]:
        """Action conversion and execution before tick.

        Returns:
            applied_action (dict): Action applied in _take_action().
        """
        # Tick backgroud world
        self.scenarios.tick_scenario()

        applied_action = {}

        # Take action
        control_batch = []
        for actor_id, action in self._action_map.items():
            applied_action[actor_id], action_command = self._take_action(
                actor_id, action
            )
            control_batch.append(action_command)

        # Tick to make the action take effect
        Simulator.send_batch(control_batch)

        return applied_action

    def _tick_until(self, tick_cnt: int):
        """Helper function to tick the world until the specified tick count is reached."""
        while True:
            current_frame = Simulator.tick()
            if current_frame >= self._frame_cnt + tick_cnt:
                self._frame_cnt = current_frame
                break

    def _step_after_tick(self, action_dict: dict, applied_action: dict) -> Measurements:
        """Collect measurements, then handle render, logging, etc.

        Args:
            action_dict (dict): Action passed into env.step().
            applied_action (dict): Action applied in _take_action().

        Returns:
            measurements (dict): Measurements of All actors.
        """

        # Get measurements from CarlaDataProvider and SensorDataProvider
        measurements = self._read_observation(action_dict, applied_action)
        self._measurement_hist.append(measurements)

        image_dict = {}
        for actor_id, measurement in measurements.items():
            image = measurement.camera_dict
            if image:
                image = image[self.actor_configs[actor_id]["camera_type"]][1]
                image = EnvObs.preprocess_image(image)
                image_dict[actor_id] = image

        if self._global_obs_conf:
            global_image = Simulator.get_actor_camera_data("global")[
                self._global_obs_conf["camera_type"]
            ][1]
            global_image = global_image.swapaxes(0, 1)
            image_dict["global"] = global_image

        if self.render_required:
            self._render(image_dict)

        return measurements

    def _update_info_from_replay(self, py_measurements: Measurements):
        """Update collision and speed info from replay data.

        Args:
            py_measurements (dict): Measurements of All actors.
        """
        # fmt: off
        if self._replay:
            collision_info = defaultdict(CollisionRecord)
            framd_max_idx = len(self._replay.frame_info) - 1

            # Collision info
            for i in range(ENV_ASSETS.step_ticks, 0, -1):
                frame_index = min(Simulator.replay_tick - i, framd_max_idx)
                frame_info = self._replay.frame_info[frame_index]
                collision_data = frame_info.collision
                actor_id_to_name = {v: k for k, v in self._actors.items()}
                for colli in collision_data:
                    actor1 = actor_id_to_name.get(colli.actor, f"other_{colli.actor}")
                    actor2 = actor_id_to_name.get(colli.other_actor, f"other_{colli.other_actor}")
                    EnvObs.update_collision_info(collision_info, self.actor_configs, actor1, actor2, colli.other_actor)
                    EnvObs.update_collision_info(collision_info, self.actor_configs, actor2, actor1, colli.actor)

            curr_frame = self._replay.get_frame(Simulator.replay_tick)
            for actor_id, id in self._actors.items():
                m = py_measurements[actor_id]
                m.velocity = Vector3D(*curr_frame.get_velocity_by_id(id))
                m.acceleration = Vector3D(*curr_frame.get_acceleration_by_id(id))

                # Only update if collision sensor callback does not work
                if self._prev_measurements and actor_id in self._prev_measurements:
                    prev_collision = self._prev_measurements[actor_id].collision
                    curr_collision = m.collision
                    if prev_collision == curr_collision:
                        curr_collision.update(collision_info[actor_id])
        # fmt: on

    def _clean_world(self):
        """Destroy all actors cleanly before exiting

        Returns:
            N/A
        """
        for agent in self._agents.values():
            agent.cleanup()
        Simulator.cleanup()
        Simulator.tick()

        self._actors.clear()
        self._agents.clear()
        self._path_trackers.clear()
        self._done_dict.clear()
        self._human_agent = None
        self._global_sensor = None

        logger.info("Cleaned-up the world...")

    def _reset_ego(self):
        """Reset ego vehicle to its initial state."""
        if "ego" not in self._actors:
            return

        while True:
            ignores = [actor.id for actor in self._agents["ego"]._sensors_list]
            if self._global_sensor:
                ignores.append(self._global_sensor.id)

            attachments = [
                Simulator.get_actor_by_id(id, from_world=True)
                for id in Simulator.clean_ego_attachment(self._actors["ego"], ignores)
            ]
            Simulator.tick()

            if len(attachments) == 0 or all(
                (actor is None) or (not actor.is_alive) or (not actor.is_active)
                for actor in attachments
            ):
                break

    def _get_background_actors(self):
        """Actor ids to be ignored in the environment during running. Typically static actors as obstacles

        Returns:
            ignores (set): set of actor ids should be ignored
        """
        ignores = set(["global"])

        for actor_id in self.actor_configs.keys():
            actor_type = self.actor_configs[actor_id].get("type", "vehicle_4W")
            blueprint = self.actor_configs[actor_id].get("blueprint", "")

            if "static" in actor_type or "misc" in actor_type or "prop" in blueprint:
                ignores.add(actor_id)

        return ignores

    def _check_actors(self):
        """Check if all actors are alive and active.

        Returns:
            bool: Whether all actors is still valid
        """
        if not Simulator.ready:
            return False

        for id in self._actors.values():
            actor = Simulator.get_actor_by_id(id)
            if actor is None or not actor.is_alive or not actor.is_active:
                return False
        return True
