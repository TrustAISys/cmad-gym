from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Tuple

import numpy as np

from cmad.simulation.data.local_carla_api import *


@dataclass
class ExpInfo:
    """The `ExpInfo` class is a data class that stores various information related to the current state of
    an experiment or episode.
    """

    distance_to_goal: float = None
    """Distance to goal in meters. This is calculated from vehicle location to the destination `along the planned route`"""

    distance_to_goal_euclidean: float = None
    """Distance to goal directly calculated from vehicle location to the end point"""

    next_command: str = None
    """Recommended next_command given by PathTracker (if Planner is disabled, always set to LaneFollow)"""

    target_speed: float = None
    """Target speed set by user in m/s"""

    actor_in_scene: "dict[str, int]" = None
    """Actor map from actor_id to carla.Actor.id in the current scene"""

    episode_id: int = None
    """Unique identifier for the current episode"""

    step: int = None
    """Current step or iteration in the episode"""

    max_steps: int = None
    """Maximum number of steps allowed in the episode"""

    step_time: float = None
    """How long of a simulation time each `env.step` takes"""

    start_pos: "tuple[float, float, float]" = None
    """Starting position coordinates (x, y, z) for the actor"""

    end_pos: "tuple[float, float, float]" = None
    """Ending or target position coordinates (x, y, z) for the actor"""

    previous_action: dict = None
    """Action taken in the previous step"""

    previous_reward: float = None
    """Reward obtained in the previous step"""

    done: bool = True
    """Indicates if the actor is finished or not in current episode"""

    reward: float = None
    """Reward obtained in the current step"""

    total_reward: float = None
    """Reward obtained in the current step"""


@dataclass
class ExtraInfo:
    """The `ExtraInfo` class holds additional information related to the actor."""

    def __getattr__(self, key):
        """Temporary solution to allow any extra info to be accessed as an attribute"""
        return self.__dict__.get(key, None)


@dataclass
class CollisionRecord:
    """This class is a wrapper for the collision information"""

    vehicles: int = 0
    """Number of collisions with other vehicles"""

    pedestrians: int = 0
    """Number of collisions with pedestrians"""

    others: int = 0
    """Number of collisions with other objects"""

    id_set: "set[int]" = field(default_factory=set)
    """Set of unique identifiers for collided objects"""

    def update(self, other: CollisionRecord):
        """Update the current collision record with another collision record

        Args:
            other (CollisionRecord): Another collision record
        """
        self.vehicles += other.vehicles
        self.pedestrians += other.pedestrians
        self.others += other.others
        self.id_set.update(other.id_set)

    def diff(
        self,
        other: CollisionRecord,
        check_vehicle: bool = True,
        check_pedestrain: bool = True,
        check_other: bool = True,
    ) -> int:
        """Calculate the difference between two collision records

        Args:
            other (CollisionRecord): Another collision record
            check_vehicle (bool): Whether to check the difference in collisions with vehicles
            check_pedestrain (bool): Whether to check the difference in collisions with pedestrians
            check_other (bool): Whether to check the difference in collisions with other objects

        Returns:
            int: The difference between the two collision records
        """
        diff_count = 0

        if check_vehicle:
            diff_count += abs(self.vehicles - other.vehicles)
        if check_pedestrain:
            diff_count += abs(self.pedestrians - other.pedestrians)
        if check_other:
            diff_count += abs(self.others - other.others)

        return diff_count


@dataclass
class LaneInvasionRecord:
    """This class is a wrapper for the lane invasion information"""

    offroad: int = 0
    """Number of times the actor went off-road"""

    otherlane: int = 0
    """Number of times the actor entered another lane"""


@dataclass
class Measurement:
    """This `Measurement` class is meant to be used for generating observation, calculating reward, and deciding done state
    for an actor. This class is serializable due to the usage of local carla api.
    """

    actor_id: str = None
    """Unique identifier for the actor or agent"""

    type: str = None
    """Type or category of the actor"""

    transform: Transform = None
    """Spatial transformation of the actor in the environment (Global reference frame)"""

    forward_vector: Vector3D = None
    """Forward direction vector of the actor (Global reference frame, unit vector)"""

    velocity: Vector3D = None
    """Velocity vector of the actor"""

    speed: float = 0.0
    """Speed of the actor in m/s"""

    acceleration: Vector3D = None
    """Acceleration vector of the actor"""

    bounding_box: BoundingBox = None
    """Bounding box of the actor"""

    waypoint: Waypoint = None
    """Waypoint right under the actor"""

    road_curvature: float = 0.0
    """Road curvature at the actor's location"""

    collision: CollisionRecord = field(default_factory=CollisionRecord)
    """Collision record for the actor"""

    lane_invasion: LaneInvasionRecord = field(default_factory=LaneInvasionRecord)
    """Lane invasion record for the actor"""

    camera_dict: Dict[str, Tuple[int, np.ndarray]] = None
    """Dictionary containing camera data and images."""

    control: dict = None
    """Dictionary containing detailed action info."""

    planned_waypoint: Waypoint = None
    """Nearest planned waypoint for the actor (could be inside the actor)"""

    planned_waypoints: List[Waypoint] = None
    """List of subsequent planned waypoints for the actor (starting from at least 2m ahead)"""

    orientation_diff: float = 0.0
    """Difference in orientation from the nearest planned waypoint"""

    road_offset: float = 0.0
    """Offset from the nearest planned waypoint (could be considered as the offset from the center of the road)"""

    extra_info: ExtraInfo = field(default_factory=ExtraInfo)
    """Additional information related to the actor"""

    exp_info: ExpInfo = field(default_factory=ExpInfo)
    """Experiment-related information for the actor"""

    def as_dict(self):
        return asdict(self)

    def get(self, key: str):
        """Get the value of a specific key in the measurement

        Args:
            key (str): The key to be accessed

        Returns:
            Any: The value of the key
        """
        if hasattr(self, key):
            return getattr(self, key)
        elif hasattr(self.exp_info, key):
            return getattr(self.exp_info, key)
        elif hasattr(self.extra_info, key):
            return getattr(self.extra_info, key)
        else:
            raise KeyError(f"Key {key} not found in measurement")


Measurements = Dict[str, Measurement]
"""Typically measurement of all actors in the scene"""
