from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field

from cmad.simulation.PythonAPI.carla.agents.navigation import RoadOption


@dataclass
class CarlaPaths:
    output: str = os.environ.get("CARLA_OUT", os.path.expanduser("~/carla_out"))
    """The directory to store the output files."""

    directory: str = os.environ.get(
        "CARLA_ROOT", os.path.expanduser("~/software/CARLA_0.9.13")
    )
    """The directory of the Carla root."""

    executable: str = os.environ.get(
        "CARLA_SERVER", os.path.expanduser("~/software/CARLA_0.9.13/CarlaUE4.sh")
    )
    """The path to the Carla executable."""

    def __post_init__(self):
        if self.output and not os.path.exists(self.output):
            os.makedirs(self.output)


@dataclass
class SystemAssets:
    is_windows_platform: bool = "win" in sys.platform
    """Whether the current platform is Windows."""

    paths: CarlaPaths = field(default_factory=CarlaPaths)
    """Holds some paths to Carla binaries and output directories."""


@dataclass
class EnvAssets:
    distance_to_goal_threshold: float = 1.0
    """Threshold for reaching the goal."""

    orientation_to_goal_threshold: float = math.pi / 4.0
    """Threshold for considering the orientation to be aligned with the goal."""

    ground_z: float = 10.0
    """Dummy z value when we don't care about the z value."""

    action_ticks: int = 2
    """How many ticks each actor.apply_control() lasts."""

    step_ticks: int = 2
    """How many ticks each env.step() takes."""

    retries_on_error: int = 3
    """How many times to retry when an error occurs."""

    @property
    def commands_enum(self):
        """Number index to string like RoadOption mapping"""
        return {
            0.0: "REACH_GOAL",
            5.0: "GO_STRAIGHT",
            4.0: "TURN_RIGHT",
            3.0: "TURN_LEFT",
            2.0: "LANE_FOLLOW",
        }

    @property
    def command_ordinal(self):
        """String like RoadOption mapping to number index"""
        return {
            "REACH_GOAL": 0,
            "GO_STRAIGHT": 1,
            "TURN_RIGHT": 2,
            "TURN_LEFT": 3,
            "LANE_FOLLOW": 4,
            "CHANGE_LANE_LEFT": 5,
            "CHANGE_LANE_RIGHT": 6,
        }

    @property
    def road_option_to_commands(self):
        """RoadOption to string like RoadOption mapping"""
        return {
            RoadOption.VOID: "REACH_GOAL",
            RoadOption.STRAIGHT: "GO_STRAIGHT",
            RoadOption.RIGHT: "TURN_RIGHT",
            RoadOption.LEFT: "TURN_LEFT",
            RoadOption.LANEFOLLOW: "LANE_FOLLOW",
            RoadOption.CHANGELANELEFT: "CHANGE_LANE_LEFT",
            RoadOption.CHANGELANERIGHT: "CHANGE_LANE_RIGHT",
        }

    @property
    def verbose_info_header(self):
        return [
            "Episode ID",
            "Actor ID",
            "Transform",
            "Velocity",
            "Acceleration",
            "Planned Waypoint",
            "Collision",
            "Lane Invasion",
            "Action",
        ]

    @property
    def default_obs_conf(self):
        """The default observation config for the env."""
        return {
            "obs_x_res": 168,
            "obs_y_res": 168,
            "framestack": 1,
            "use_depth_camera": False,
            "send_measurements": True,
            "measurement_type": ["all"],
            "add_action_mask": True,
        }

    @property
    def default_actor_done_criteria(self):
        """The default done criteria for the actor."""
        return ["timeout"]

    @property
    def default_episode_done_criteria(self):
        """The default done criteria for the env."""
        return [
            "ego_collision",
            "ego_offroad",
            "ego_rollover",
            "ego_reach_goal",
            "ego_timeout",
            "npc_done",
        ]

    @property
    def default_low_level_discrete_actions(self):
        """The default discrete action set for low_level (throttle/steer) control"""
        return {
            0: [-0.5, 0.0],
            1: [-0.15, -0.15],
            2: [-0.15, 0.15],
            3: [0.0, 0.0],
            4: [0.25, -0.3],
            5: [0.25, 0.3],
            6: [0.75, -0.15],
            7: [0.75, 0.15],
            8: [1.0, 0.0],
        }

    @property
    def default_vehicle_atomic_discrete_actions(self):
        """The default discrete action set for atomic control"""
        return (
            # Planning actions
            {
                0: "stop",
                1: "lane_follow",
                2: "left_lane_change",
                3: "right_lane_change",
                4: "turn_left",
                5: "turn_right",
            },
            # Target speed
            {
                0: 0,
                1: 6,
                2: 12,
                3: 18,
                4: 24,
            },
        )

    @property
    def default_vehicle_route_discrete_actions(self):
        """The default discrete action set for vehicle follow control"""
        return {0: 0, 1: 6, 2: 12, 3: 18, 4: 24}

    @property
    def default_walker_discrete_actions(self):
        """The default discrete action set for walker control"""
        return (
            # direction
            {
                0: "stay",
                1: "front",
                2: "left",
                3: "right",
            },
            # target speed
            {i: i for i in range(4)},
        )

    @property
    def default_walker_speed_discrete_actions(self):
        """The default discrete action set for walker speed only control"""
        return {i: i for i in range(4)}

    @property
    def default_walker_patrol_discrete_actions(self):
        """The default discrete action set for walker patrol control"""
        return (
            # direction
            {
                0: "stable",
                1: "left",
                2: "right",
            },
            # target speed
            {0: 0, 1: 2, 2: 4, 3: 6},
        )

    @property
    def default_pseudo_discrete_actions(self):
        """The default discrete action set for pseudo control"""
        return {0: "null"}

    @property
    def default_action_conf(self):
        """Default action space config for an agent."""
        return {
            "type": "low_level_action",
            "use_discrete": True,
            "discrete_action_set": self.default_low_level_discrete_actions,
            "action_range": {
                "throttle": [0, 1.0],
                "brake": [-1.0, 0],
                "steer": [-0.5, 0.5],
            },
            "preprocess_fn": None,
        }

    @property
    def default_multienv_config(self):
        """Default multienv config for cmad."""
        return {
            "scenarios": "DEFAULT_SCENARIO_TOWN1",
            "env": {
                # server config
                "server_ip": "localhost",
                "server_port": 2000,
                "server_map": "/Game/Carla/Maps/Town01",
                "reload": True,
                "render": True,
                "render_x_res": 800,
                "render_y_res": 600,
                "sync_server": True,
                "fixed_delta_seconds": 0.05,
                "spectator_loc": [
                    -12.160260,
                    20.710268,
                    10.342386,
                    -11.998654,
                    65.999794,
                    0.000121,
                ],
                "global_observation": {
                    "camera_type": "rgb",
                    "x_res": 512,
                    "y_res": 512,
                    "camera_position": {
                        "x": -2.971099,
                        "y": 161.967834,
                        "z": 118.703270,
                        "pitch": -90,
                        "yaw": 0,
                        "roll": 0,
                    },
                    "render": True,
                },
                # experiment config
                "obs": self.default_obs_conf,
                "action": self.default_action_conf,
                # logging
                "record": False,
                "verbose": True,
                # external component
                # 'use_redis': True,
                "redis_host": "127.0.0.1",
                "redis_port": 6379,
                "redis_db": 0,
            },
            "actors": {
                "vehicle1": {
                    # actor info
                    "type": "vehicle_4W",
                    # control type
                    "auto_control": True,
                    "enable_planner": True,
                    "manual_control": False,
                    # obs
                    "camera_type": "rgb",
                    "render": False,
                    "lane_sensor": "on",
                    "collision_sensor": "on",
                    # done
                    "done_criteria": ["timeout", "reach_goal", "collision", "offroad"],
                    # reward
                    "reward_function": "npc",
                    # attributes
                    "initial_speed": 25,
                    "target_speed": 30,
                    "blueprint": "vehicle.tesla.model3",
                }
            },
        }


SYS_ASSETS = SystemAssets()
ENV_ASSETS = EnvAssets()

__all__ = ["SYS_ASSETS", "ENV_ASSETS"]
