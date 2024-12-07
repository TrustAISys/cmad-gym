from __future__ import annotations

import math
import weakref
from typing import TYPE_CHECKING, Dict, Optional, Set, Union

import numpy as np
from gym.spaces import Box
from gym.spaces import Dict as GymDict

if TYPE_CHECKING:
    from cmad.envs.multi_env import MultiCarlaEnv

from cmad.envs.static_asset import ENV_ASSETS
from cmad.misc import flatten_dict
from cmad.simulation.data.local_carla_api import Transform, Vector3D
from cmad.simulation.data.measurement import CollisionRecord, Measurements


class EnvObs:
    def __init__(self, env: "MultiCarlaEnv"):
        self.env_ref: MultiCarlaEnv = weakref.proxy(env)
        self.env_config: dict = env.env_config
        self.obs_config: dict = self.env_config.get("obs", ENV_ASSETS.default_obs_conf)
        self.actor_configs: dict = env.actor_configs
        self._background_actor_ids: set = env.background_actor_ids

        self._focus_actors: Set[str] = set(
            self.obs_config.get("focus_actors", self.actor_configs.keys())
        )
        self._ignore_actors: Set[str] = set(self.obs_config.get("ignore_actors", []))
        self._valid_actors: Set[str] = self._focus_actors - self._ignore_actors

        self._use_depth_camera: bool = self.obs_config.get("use_depth_camera", False)
        self._obs_x_res: int = self.obs_config.get("obs_x_res", 168)
        self._obs_y_res: int = self.obs_config.get("obs_y_res", 168)
        self._norm_image: bool = self.obs_config.get("norm_image", True)
        self._framestack: int = self.obs_config.get("framestack", 1)
        self._reference_frame: str = self.obs_config.get("reference_frame", "self")
        assert self._framestack in [1, 2]

        self.send_measurements: bool = self.obs_config.get("send_measurements", False)
        self.measurement_type: Set[str] = set(
            self.obs_config.get("measurement_type", ["all"])
        )
        self.add_action_mask: bool = self.obs_config.get("add_action_mask", False)

        self._prev_image: dict = {}

    def encode_obs(
        self,
        actor_id: str,
        py_measurements: Measurements,
        action_mask: np.ndarray = None,
    ):
        """Encode sensor and measurements into obs based on state-space config.

        Args:
            actor_id (str): Actor identifier
            py_measurements (dict): measurement file of ALL actors
            action_mask (np.ndarray): action mask for the actor

        Returns:
            obs (dict): properly encoded observation data for each actor
        """
        image = py_measurements[actor_id].camera_dict
        if image is not None and len(image) > 0:
            image = image[self.actor_configs[actor_id]["camera_type"]][1]
            if self._norm_image:
                image = EnvObs.preprocess_image(image)

        # Stack frames
        prev_image = self._prev_image.get(actor_id, None)
        self._prev_image[actor_id] = image
        if prev_image is None:
            prev_image = image
        if self._framestack == 2:
            image = np.concatenate([prev_image, image])

        # Structure the observation
        obs = {"camera": image}

        # Add semantic info if required
        if self.send_measurements:
            semantic_info = self.get_semantic_info(
                actor_id, py_measurements, reference=self._reference_frame
            )
            obs["state"] = flatten_dict(semantic_info)

        # Add action_mask if required
        if self.add_action_mask:
            obs["action_mask"] = action_mask

        return obs

    def decode_image(self, actor_id: str, img: np.ndarray):
        """Decode actor observation into original image reversing the pre_process() operation.
        Args:
            actor_id (str): Actor identifier
            img (np.ndarray): Encoded observation image of an actor

        Returns:
            image (np.ndarray): Original actor camera view
        """
        # Reverse the processing operation
        if self.actor_configs[actor_id].get("use_depth_camera", False):
            img = np.tile(img.swapaxes(0, 1), 3)
        else:
            img = img.swapaxes(0, 1) * 128 + 128
        return img

    def get_semantic_info(
        self,
        target_id: str,
        py_measurements: Measurements,
        reference: str = "self",
    ):
        """Get semantic information from the current frame.

        Args:
            target_id (str): Actor we focus on
            py_measurements (dict): measurement file containing all actors' information
            reference (str): reference frame for the semantic information. "global" | "self" | "world"

        Returns:
            semantic_info (dict): semantic information
        """

        semantic_info = {"self": {}, "ego": {}, "others": {}}

        # Iterate through each actor
        for actor_id in py_measurements:
            if actor_id not in self._valid_actors:
                continue

            # Get all information in world frame
            measurement = py_measurements[actor_id]

            # Location in world frame
            transform = measurement.transform
            location = transform.location
            heading = measurement.transform.rotation.yaw

            # Project bounding box to world frame
            bbox = measurement.bounding_box
            # bb_cords = bbox.transform_to_world_frame(transform, bottom_only=True)

            # Velocity in world frame
            velocity = measurement.velocity

            # Nearest waypoint in world frame
            planned_wpts = measurement.planned_waypoints
            if measurement.planned_waypoints[0] is not None:
                waypoint_locations = [wp.transform.location for wp in planned_wpts]
            else:
                waypoint_locations = [location] * len(planned_wpts)
            waypoints = [(wp.x, wp.y) for wp in waypoint_locations]

            if reference == "global":
                target_transform = Transform.from_simulator_transform(
                    self.env_ref._global_sensor.get_transform()
                )
                axis_mapping = target_transform.get_axis_mapping()
            elif reference == "self":
                target_transform = py_measurements[target_id].transform
                axis_mapping = None
            else:
                target_transform = None

            if target_transform is not None:
                location = self.transform_vector(
                    location, target_transform, axis_mapping
                )
                velocity = (
                    self.transform_vector(
                        velocity + transform.location,
                        target_transform,
                        axis_mapping,
                    )
                    - location
                )
                # bb_cords = self.transform_points(bb_cords, target_transform, axis_mapping)
                heading = (
                    -heading
                    if reference == "global"
                    else heading - target_transform.rotation.yaw
                )

                # Provide planned waypoints only for the actor itself
                if actor_id == target_id:
                    waypoint_locations = [
                        self.transform_vector(wp_loc, target_transform, axis_mapping)
                        for wp_loc in waypoint_locations
                    ]
                    waypoints = [(wp.x, wp.y) for wp in waypoint_locations]
                else:
                    waypoints = [(0, 0)] * len(waypoints)

            zero_check = lambda x: 0 if abs(x) < 1e-5 else x
            info = self.filter_semantic_info(
                {
                    "active": 0 if self.env_ref._done_dict.get(actor_id, True) else 1,
                    "is_ego": 1 if actor_id == "ego" else 0,
                    "x": zero_check(location.x),
                    "y": zero_check(location.y),
                    "heading": math.cos(math.radians(zero_check(heading))),
                    "vx": zero_check(velocity.x),
                    "vy": zero_check(velocity.y),
                    "speed": zero_check(measurement.speed),
                    "road_offset": zero_check(measurement.road_offset),
                    # for bounding box, [:, :2] (bottom 4 points), [::2, :2] (diagonal points)
                    # "bounding_box": bb_cords[::2, :2],
                    "extent": bbox.extent.as_numpy_array(),
                    "waypoints": [
                        [zero_check(waypoint[0]), zero_check(waypoint[1])]
                        for waypoint in waypoints
                    ],
                },
                self.measurement_type,
            )

            # Update dict
            if actor_id == target_id:
                semantic_info["self"] = info
            elif actor_id == "ego":
                semantic_info["ego"] = info
            else:
                semantic_info["others"][actor_id] = info

        return semantic_info

    @staticmethod
    def transform_points(
        points: np.ndarray,
        reference_transform: Transform,
        axis_mapping: Optional[Transform.AxisMap] = None,
        inverse_transform: bool = True,
    ) -> np.ndarray:
        """Transform a set of points from one frame to another.

        Args:
            points (np.ndarray): The points to be transformed.
            reference_transform (Transform): The reference frame.
            axis_mapping (Transform.AxisMap, optional): The axis mapping between reference frame.
            inverse_transform (bool): Whether to transform into global frame or actor frame. Defaults to actor frame.

        Returns:
            np.ndarray: _description_
        """
        transformed_points = (
            reference_transform.inverse_transform_points(points)
            if inverse_transform
            else reference_transform.transform_points(points)
        )

        if axis_mapping is not None:
            return EnvObs.points_axis_map(transformed_points, axis_mapping)
        else:
            return transformed_points

    @staticmethod
    def points_axis_map(points: np.ndarray, axis_mapping: Transform.AxisMap = None):
        """Map the axis of a point based on given axis mapping.

        Args:
            point (np.ndarray): The point to be mapped.
            axis_mapping (Transform.AxisMap): The axis mapping between reference frame.

        Returns:
            point (np.ndarray): Mapped point. This will always be a 2D array.
        """
        if axis_mapping is not None:
            if len(points.shape) == 1:
                points = points.reshape(1, -1)

            return np.array(
                [
                    [
                        point[axis_mapping.x.index],
                        point[axis_mapping.y.index],
                        point[axis_mapping.z.index],
                    ]
                    for point in points
                ]
            )
        else:
            return points

    @staticmethod
    def transform_vector(
        vector: Vector3D,
        reference_transform: Transform,
        axis_mapping: Optional[Transform.AxisMap] = None,
        inverse_transform: bool = True,
    ) -> Vector3D:
        """Transform a vector from one frame to another.

        Args:
            vector (Vector3D): The vector to be transformed.
            reference_transform (Transform): The reference frame.
            axis_mapping (Transform.AxisMap, optional): The axis mapping between reference frame.
            inverse_transform (bool): Whether to transform into global frame or actor frame. Defaults to actor frame.

        Returns:
            Vector3D: The transformed vector.
        """
        transformed_vector = (
            reference_transform.inverse_transform_location(vector)
            if inverse_transform
            else reference_transform.transform_location(vector)
        )

        if axis_mapping is not None:
            return EnvObs.vector_axis_map(transformed_vector, axis_mapping)
        else:
            return transformed_vector

    @staticmethod
    def vector_axis_map(vector: Vector3D, axis_mapping: Transform.AxisMap = None):
        """Map the axis of a vector based on given axis mapping.

        Args:
            vector (Vector3D): The vector to be mapped.
            axis_mapping (Transform.AxisMap): The axis mapping between reference frame.

        Returns:
            vector (Vector3D): Mapped vector.
        """
        if axis_mapping is not None:
            return Vector3D(
                getattr(vector, axis_mapping.x.axis),
                getattr(vector, axis_mapping.y.axis),
                getattr(vector, axis_mapping.z.axis),
            )
        else:
            return vector

    @staticmethod
    def filter_semantic_info(semantic_info: dict, filter_set: Set[str]):
        """Filter out the semantic information that is not needed.

        Args:
            semantic_info (dict): semantic information containing all information
            filter_set (set): set of semantic information (keys) to be kept

        Returns:
            semantic_info (dict): filtered semantic information
        """
        if "all" in filter_set:
            return semantic_info

        filtered_info = {}
        for key in semantic_info:
            if key in filter_set:
                filtered_info[key] = semantic_info[key]
        return filtered_info

    @staticmethod
    def preprocess_image(image: np.ndarray):
        """Process image raw data to array data.

        Args:
            image (np.ndarray): image data from Callback.

        Returns:
            np.ndarray: Image array.
        """
        image = (image.astype(np.float32) - 128) / 128
        return image

    @staticmethod
    def get_next_actions(measurements: Measurements, is_discrete_actions: bool):
        """Get/Update next action, work with way_point based planner.

        Args:
            measurements (dict): measurement data.
            is_discrete_actions (bool): whether use discrete actions

        Returns:
            dict: action_dict, dict of len-two integer lists.
        """
        action_dict = {}
        for actor_id, m in measurements.items():
            command = m.exp_info.next_command
            if command in ["PASS_GOAL", "REACH_GOAL"]:
                action_dict[actor_id] = 0 if is_discrete_actions else (-1, 0)
            elif command == "GO_STRAIGHT":
                action_dict[actor_id] = 3 if is_discrete_actions else (1, 0)
            elif command == "TURN_RIGHT":
                action_dict[actor_id] = 6 if is_discrete_actions else (0.75, 0.15)
            elif command == "TURN_LEFT":
                action_dict[actor_id] = 5 if is_discrete_actions else (0.75, -0.15)
            elif command == "LANE_FOLLOW":
                action_dict[actor_id] = 0 if is_discrete_actions else (0, 0)

        return action_dict

    @staticmethod
    def update_collision_info(
        collision_info: Dict[str, CollisionRecord],
        actor_configs: Dict[str, dict],
        actor_id: str,
        collided_with: str,
        collided_id: int,
    ):
        """Update collision information in place

        Args:
            collision_info (dict): collision information to update
            actor_configs (dict): actor configurations
            actor_id (str): The ego actor's name
            collided_with (str): The collided actor's name
            collided_id (int): The collided actor id
        """

        actor_type = actor_configs.get(collided_with, {}).get("type", "other")

        if "vehicle" in actor_type:
            collision_info[actor_id].vehicles += 1
        elif "pedestrian" in actor_type or "walker" in actor_type:
            collision_info[actor_id].pedestrians += 1
        else:
            collision_info[actor_id].others += 1

        collision_info[actor_id].id_set.add(collided_id)

    @staticmethod
    def semantic_info_dimension():
        """Return the dimension of each semantic information after flatten."""

        return {
            "active": 1,
            "is_ego": 1,
            "x": 1,
            "y": 1,
            "vx": 1,
            "vy": 1,
            "speed": 1,
            "heading": 1,
            "road_offset": 1,
            # "bounding_box": 4,
            "extent": 3,
            "waypoints": 10,
        }

    def get_valid_actors(self):
        """Return the valid actors that will be count in the semantic information."""
        return self._valid_actors

    def get_observation_space(self):
        """Returns the observation space for a reinforcement learning environment, which may include image data, semantic information, and action masks if required.

        Returns:
            observation_space (gym.spaces.Dict): observation space of the environment.
        """

        # Output space of images after preprocessing
        image_space = Box(
            0.0,
            255.0,
            shape=(
                self._obs_y_res,
                self._obs_x_res,
                self._framestack * (1 if self._use_depth_camera else 3),
            ),
        )

        obs_dict = {"camera": image_space}

        # Add semantic info space if required
        if self.send_measurements:
            actor_semantic_dim = sum(
                self.filter_semantic_info(
                    self.semantic_info_dimension(), self.measurement_type
                ).values()
            )
            semantic_length = actor_semantic_dim * (len(self._valid_actors))
            obs_dict["state"] = Box(
                -np.inf, np.inf, shape=(semantic_length,), dtype=np.float32
            )

        # Action mask is agent-wise
        agent_wise_obs_dict = {}
        for actor_id in self.actor_configs:
            if actor_id in self._background_actor_ids:
                continue

            agent_obs = obs_dict.copy()
            if self.add_action_mask:
                space_type = self.env_ref.agent_actions[actor_id].get_space_type()
                if space_type == "Discrete":
                    agent_obs["action_mask"] = Box(
                        0.0,
                        1.0,
                        (
                            len(
                                self.env_ref.agent_actions[actor_id].discrete_action_set
                            ),
                        ),
                    )
                elif space_type == "MultiDiscrete":
                    agent_obs["action_mask"] = Box(
                        0.0,
                        1.0,
                        (
                            sum(
                                [
                                    len(action_set)
                                    for action_set in self.env_ref.agent_actions[
                                        actor_id
                                    ].discrete_action_set
                                ]
                            ),
                        ),
                    )
                else:
                    raise NotImplementedError(
                        f"Action mask for space type {space_type} is not supported."
                    )
            agent_wise_obs_dict[actor_id] = GymDict(agent_obs)

        observation_space = GymDict(agent_wise_obs_dict)
        return observation_space
