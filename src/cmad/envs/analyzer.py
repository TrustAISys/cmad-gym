from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, List

import numpy as np

from cmad.simulation.data.measurement import Measurement

Measurements = Dict[str, Measurement]
"""Measurement of all actors in the latest step"""

History = List[Measurements]
"""History of measurements"""

IdMap = Dict[int, str]
"""A mapping between carla id and actor id"""


class Analyzer:
    """Class provides a set of utility functions to analyze one episode"""

    _registered_analyzer: Dict[str, Callable] = {}

    @staticmethod
    def causal_analysis(
        py_measurements: Measurements,
        history: History,
        id_actor_map: IdMap,
        end_type: str,
    ) -> str:
        """Analyze the causal of the end of the episode

        Args:
            py_measurements (dict): The measurements of the latest step
            history (list): The history of measurements
            id_actor_map (dict): The mapping from carla id to actor id
            end_type (str): The type of the end of the episode

        Returns:
            Top1_reason (str): The most likely cause of the end of the episode
        """
        analyzer = f"{end_type.lower()}_analysis"
        if analyzer in Analyzer._registered_analyzer:
            reason, _ = Analyzer._registered_analyzer[analyzer](
                py_measurements, history, id_actor_map
            )
        else:
            reason = end_type

        return reason

    @staticmethod
    def register_analyzer(identifier: str, func: Callable):
        """Register a new analysis function

        Args:
            identifier (str): The identifier of the analysis function
            func (Callable): The analysis function
        """
        full_name = f"{identifier.lower()}_analysis"
        Analyzer._registered_analyzer[full_name] = func

    @staticmethod
    def ego_collision_analysis(
        py_measurements: Measurements,
        history: History,
        actor_map: IdMap,
        focus_actor: str = "ego",
    ) -> tuple[str, dict]:
        """Analyze the collision and return the most likely cause of the collision

        Args:
            py_measurements (dict): The measurements of the latest step
            history (list): The history of measurements
            actor_map (dict): The mapping from carla id to actor id
            focus_actor (str): The actor to focus on, default to "ego"

        Returns:
            Top1_cause (str): The most likely cause of the collision
            possible_cause (dict): The likelihood of each possible cause
        """

        if Analyzer.is_standing_still(focus_actor, history):
            return "NPC_COLLIDE_STILL_EGO", {}

        ego_measurement = py_measurements[focus_actor]
        ego_forward = ego_measurement.forward_vector.as_numpy_array()
        ego_cut_in = Analyzer.is_cutting_in(ego_measurement, history)

        # extract the actors involved in the collision (apart from the ego)
        collision_actors = [
            actor_id
            for id, actor_id in actor_map.items()
            if id in ego_measurement.collision.id_set and actor_id != "ego"
        ]
        colli_actor_measurements = {
            k: v for k, v in py_measurements.items() if k in collision_actors
        }

        possible_cause = defaultdict(float, {"EGO_COLLI_UNKNOWN": 0.5})

        if len(colli_actor_measurements) == 0:
            possible_cause["EGO_COLLI_OBSTACLE"] += 1

        for actor_id, measurement in colli_actor_measurements.items():
            npc_forward = measurement.forward_vector.as_numpy_array()
            relative_position = Analyzer.relative_position(ego_measurement, measurement)
            npc_cut_in = Analyzer.is_cutting_in(measurement, history)
            angle_diff = Analyzer.angle_diff(ego_forward, npc_forward)

            # Classify the collision based on the relative position, cut-in and lane change
            if angle_diff < np.pi / 2:  # both vehicles are moving in the same direction
                if relative_position == "rear":
                    if ego_cut_in:
                        possible_cause["EGO_CUT_IN_NPC"] += 1
                    else:
                        possible_cause["NPC_REAR_END_EGO"] += 1
                elif relative_position == "front":
                    if npc_cut_in:
                        possible_cause["NPC_CUT_IN_EGO"] += 1
                    else:
                        possible_cause["EGO_REAR_END_NPC"] += 1
                else:  # side
                    if ego_cut_in:
                        possible_cause["EGO_CUT_IN_NPC"] += 1
                    else:
                        possible_cause["NPC_PINCER_EGO"] += 0.5
            else:  # vehicles are moving in the opposite directions
                if relative_position == "rear":
                    possible_cause["EGO_BACK_COLLISION"] += 1
                elif relative_position == "front":
                    possible_cause["EGO_HEAD_ON_COLLISION"] += 1
                else:  # side
                    possible_cause["EGO_SIDE_COLLISION"] += 1

        possible_cause["NPC_PINCER_EGO"] += Analyzer.calculate_pincer_probability(
            py_measurements
        )

        # Return the most likely cause of the collision
        likelihood = sorted(possible_cause.items(), key=lambda x: x[1], reverse=True)
        return likelihood[0][0], possible_cause

    @staticmethod
    def ego_offroad_analysis(
        py_measurements: Measurements,
        history: History,
        actor_map: IdMap,
        focus_actor: str = "ego",
    ) -> tuple[str, dict]:
        """Analyze the offroad scenario and return the most likely cause.

        Args:
            py_measurements (dict): The measurements of the latest step of all actors
            history (list): The history of measurements
            actor_map (dict): The mapping from carla id to actor id
            focus_actor (str): The actor to focus on, default to "ego"

        Returns:
            Top1_cause (str): The most likely cause of offroad
            possible_cause (dict): The likelihood of each possible cause
        """

        ego_measurement = py_measurements[focus_actor]
        ego_loc = ego_measurement.transform.location.as_numpy_array()
        ego_speed = ego_measurement.speed
        orientation_diff = ego_measurement.orientation_diff
        road_curvature = ego_measurement.road_curvature
        target_speed = ego_measurement.exp_info.target_speed

        possible_cause = defaultdict(float, {"EGO_LOSE_CONTROL": 0.5})

        # Avoiding Collision
        for actor_name, measurement in py_measurements.items():
            if actor_name == focus_actor:
                continue
            vehicle_loc = measurement.transform.location.as_numpy_array()
            distance = np.linalg.norm(vehicle_loc - ego_loc)

            if (
                distance <= 5
            ):  # Threshold distance to consider a vehicle as being too close
                possible_cause["EGO_AVOID_COLLISION"] += 1

        # High Speed
        if (
            ego_speed > target_speed + 5
        ):  # Threshold of 5 m/s over the target speed as "high speed"
            possible_cause["EGO_HIGH_SPEED"] += 1

        # Misjudgment
        if (
            orientation_diff > np.pi / 4 and road_curvature > 0.1
        ):  # Thresholds can be adjusted based on empirical evidence
            possible_cause["EGO_MISJUDGEMENT"] += 1

        # Check the speed and acceleration history for sudden changes
        ego_speed_history = [measurement[focus_actor].speed for measurement in history]
        ego_speed_change = np.diff(ego_speed_history)
        if np.any(ego_speed_change > 5):  # Threshold of 5 m/s^2 for sudden speed change
            possible_cause["EGO_SUDDEN_SPEED_CHANGE"] += 1

        # Return the most likely reason
        likelihood = sorted(possible_cause.items(), key=lambda x: x[1], reverse=True)
        return likelihood[0][0], possible_cause

    @staticmethod
    def ego_timeout_analysis(
        py_measurements: Measurements, history: History, actor_map: IdMap
    ) -> tuple[str, dict]:
        """Analyze the timeout scenario and return the most likely cause.

        Args:
            py_measurements (dict): The measurements of the latest step of all actors
            history (list): The history of measurements
            actor_map (dict): The mapping from carla id to actor id

        Returns:
            Top1_cause (str): The most likely cause of timeout
            possible_cause (dict): The likelihood of each possible cause
        """

        ego_measurement = py_measurements["ego"]
        ego_loc = ego_measurement.transform.location
        ego_speed = ego_measurement.speed

        possible_cause = defaultdict(float, {"EGO_TIMEOUT_UNKNOWN": 0.5})

        # Check if the ego vehicle is stuck
        if Analyzer.is_standing_still("ego", history):
            possible_cause["EGO_STUCK"] += 0.8

            # Check if stuck due to npc blocking
            for actor_id, measurement in py_measurements.items():
                if actor_id in ["ego", "hero"]:
                    continue

                npc_loc = measurement.transform.location
                if ego_loc.distance(npc_loc) < 5:
                    possible_cause["NPC_BLOCK_EGO"] += 1

        # Check if the ego vehicle is going too slow
        if 0 <= ego_speed <= 5:
            possible_cause["EGO_TOO_SLOW"] += 1

        # Return the most likely reason
        likelihood = sorted(possible_cause.items(), key=lambda x: x[1], reverse=True)
        return likelihood[0][0], possible_cause

    @staticmethod
    def npc_done_analysis(
        py_measurements: Measurements, history: History, actor_map: IdMap
    ) -> tuple[str, dict]:
        """Analyze the NPC done scenario and return the most likely cause.

        Args:
            py_measurements (dict): The measurements of the latest step of all actors
            history (list): The history of measurements
            actor_map (dict): The mapping from carla id to actor id

        Returns:
            Top1_cause (str): The most likely cause of NPC done
            possible_cause (dict): The likelihood of each possible cause
        """

        reached_goal_count = 0
        collided_count = 0

        # Iterate over all NPCs
        num_of_npc = 0
        for npc_id, npc_measurement in py_measurements.items():
            if npc_id in ["ego", "hero"] or "static" in npc_measurement.type:
                continue

            # Check if the NPC reached its goal
            if (
                npc_measurement.exp_info.next_command in ["PASS_GOAL", "REACH_GOAL"]
                or 0 <= npc_measurement.exp_info.distance_to_goal <= 5
            ):
                reached_goal_count += 1
            # Check if the NPC collided
            elif len(npc_measurement.collision.id_set) > 0:
                collided_count += 1

            num_of_npc += 1

        if reached_goal_count + collided_count == 0:
            reason = "NPC_TASK_FAILED"
        elif reached_goal_count >= num_of_npc:
            reason = "ALL_NPCS_REACHED_GOAL"
        elif collided_count >= num_of_npc:
            reason = "ALL_NPCS_COLLIDED"
        else:
            reason = "NPC_DONE_UNKNOWN"

        return reason, {}

    @staticmethod
    def is_cutting_in(
        curr_measurement: Measurement,
        history: History,
        reference_vector: np.ndarray = None,
    ) -> bool:
        """Check if the vehicle is performing a cutting in behavior

        Args:
            curr_measurement (Measurement): The measurement of an actor at current step
            history (list): The history of measurements
            reference_vector (np.array): The reference vector to check against, if None, the waypoint vector is used

        Returns:
            is_cutting_in (bool): Whether the vehicle is performing a cutting in behavior
        """
        actor_id = curr_measurement.actor_id

        # Calculate the displacement of the vehicle
        vehicle_forward = curr_measurement.forward_vector.as_numpy_array()
        vehicle_loc = curr_measurement.transform.location.as_numpy_array()
        previous_loc = history[0][actor_id].transform.location.as_numpy_array()
        displacement = vehicle_loc - previous_loc

        # Check for lateral movement by taking the cross product of the forward vector and displacement
        if reference_vector is None:
            waypoint_rot = curr_measurement.waypoint.transform.rotation
            cp = np.cos(np.radians(waypoint_rot.pitch))
            sp = np.sin(np.radians(waypoint_rot.pitch))
            cy = np.cos(np.radians(waypoint_rot.yaw))
            sy = np.sin(np.radians(waypoint_rot.yaw))
            reference_vector = np.array([cp * cy, cp * sy, sp])

        lateral_movement = np.cross(reference_vector, displacement)
        angle_diff = Analyzer.angle_diff(reference_vector, vehicle_forward)
        if angle_diff > np.pi / 2:
            angle_diff = np.pi - angle_diff

        # If the magnitude of the cross product is larger than a threshold, this is a cut-in
        if np.linalg.norm(lateral_movement) > 2 or angle_diff > np.pi / 9:
            cut_in = True
        else:
            cut_in = False

        # Check if the vehicle's lane has changed
        if curr_measurement.waypoint.lane_id != history[0][actor_id].waypoint.lane_id:
            lane_change = True
        else:
            lane_change = False

        return cut_in or lane_change

    @staticmethod
    def relative_position(
        ego: Measurement, vehicle: Measurement, side_range: float = 3
    ) -> str:
        """Calculate the relative position of the other vehicle to the ego vehicle

        Args:
            ego (Measurement): The measurements of the ego vehicle
            vehicle (Measurement): The measurements of the other vehicle
            side_range (float): The range to consider a vehicle to be side by side

        Returns:
            relative_pos (str): The relative position of the other vehicle to the ego vehicle
        """

        # Get the forward vector and location for the ego and other vehicle
        ego_forward = ego.forward_vector.as_numpy_array()
        ego_loc = ego.transform.location.as_numpy_array()
        vehicle_loc = vehicle.transform.location.as_numpy_array()

        # Calculate the vector from the ego vehicle to the other vehicle
        ego_to_vehicle = vehicle_loc - ego_loc

        # Project this vector onto the ego vehicle's forward vector
        # to determine if the other vehicle is in front of or behind the ego vehicle
        forward_projection = np.dot(ego_to_vehicle, ego_forward)

        # Calculate a vector orthogonal (perpendicular) to the ego vehicle's forward vector
        # This could be done in a few ways, but one simple method is to rotate the forward vector by 90 degrees
        # Since we're in 3D, we'll assume the rotation is around the Z axis (upward direction)
        ego_side = np.array([-ego_forward[1], ego_forward[0], 0])

        # Project the vector from the ego vehicle to the other vehicle onto the ego vehicle's side vector
        side_projection = np.dot(ego_to_vehicle, ego_side)

        # If the absolute value of the forward projection is small and the side projection is within a certain range,
        # then the NPC vehicle is side by side with the ego vehicle
        if abs(forward_projection) <= side_range and abs(side_projection) <= side_range:
            return "left_side" if side_projection < 0 else "right_side"
        elif forward_projection > 0:
            return "front"
        elif forward_projection < 0:
            return "rear"

    @staticmethod
    def angle_diff(
        reference_direction: np.ndarray,
        forward_direction: np.ndarray,
        in_degree: bool = False,
    ) -> float:
        """Calculate the angle difference between two forward vectors

        Args:
            reference_direction (np.array): The reference direction, unit vector
            forward_direction (np.array): The forward direction, unit vector
            in_degree (bool): Whether to return the angle difference in degree

        Returns:
            angle_diff (float): The angle difference between the ego vehicle and the other vehicle
        """

        # Calculate the dot product of the two forward vectors
        dot_product = np.clip(np.dot(reference_direction, forward_direction), -1, 1)

        # Calculate the angle between the two forward vectors
        angle = np.arccos(dot_product)

        return angle if not in_degree else np.rad2deg(angle)

    @staticmethod
    def calculate_pincer_probability(
        py_measurements: Measurements,
        consider_range: float = 10,
        max_npc_vehicles: int = 2,
    ) -> float:
        """Calculate the probability of a pincer collision

        Args:
            py_measurements (dict): The measurements of all actors in the latest step
            consider_range (float): The range to consider a vehicle to be side by side
            max_npc_vehicles (int): The maximum number of NPC vehicles to consider

        Returns:
            probability (float): The probability of a pincer collision
        """

        ego_measurement = py_measurements["ego"]

        # Get the ego vehicle's location and forward vector
        ego_loc = ego_measurement.transform.location.as_numpy_array()
        ego_forward = ego_measurement.forward_vector.as_numpy_array()

        # Initialize lists to hold the distances to the NPC vehicles and the angles between the forward vectors
        distances = []
        angles = []

        # Iterate over the NPC vehicles
        for actor_name, measurement in py_measurements.items():
            if actor_name in ["ego", "hero"]:
                continue

            # Calculate the distance to the ego vehicle
            vehicle_loc = measurement.transform.location.as_numpy_array()
            distance = np.linalg.norm(vehicle_loc - ego_loc)

            # If the vehicle is within the consider range, calculate the angle between the forward vectors
            if distance <= consider_range:
                vehicle_forward = measurement.forward_vector.as_numpy_array()

                distances.append(distance)
                angles.append(Analyzer.angle_diff(ego_forward, vehicle_forward))

        if len(distances) == 0:
            return 0.0

        # Normalize the distances and the number of NPC vehicles
        normalized_distances = 1 - np.array(distances) / consider_range
        normalized_npc_vehicles = len(distances) / max_npc_vehicles

        # Check if the NPC vehicles are not all going in the same direction
        different_directions = max(angles) > np.pi / 4

        # Calculate the pincer probability as the product of the normalized distances, the normalized number of NPC vehicles, and the direction factor
        pincer_probability = (
            np.mean(normalized_distances)
            * normalized_npc_vehicles
            * int(different_directions)
        )

        return pincer_probability

    @staticmethod
    def is_standing_still(
        actor_id: str,
        history: History,
        speed_threshold: float = 0.1,
        acc_threshold: float = 0.3,
        distance_threshold: float = 0.5,
    ) -> bool:
        """Check if the actor has been standing still

        Args:
            actor_id (str): The actor id
            history (History): The previous measurements history
            speed_threshold (float): The speed threshold to consider the actor as standing still
            acc_threshold (float): The acceleration threshold to consider the actor as standing still
            distance_threshold (float): The distance threshold to consider the actor as stuck
        """

        # check both average speed and acceleration
        avg_speed = np.mean([measurement[actor_id].speed for measurement in history])
        avg_acc = np.mean(
            [measurement[actor_id].acceleration.length() for measurement in history]
        )
        if avg_speed < speed_threshold and avg_acc < acc_threshold:
            return True

        # check if the actor is stuck
        hist_loc = history[0][actor_id].transform.location
        curr_loc = history[-1][actor_id].transform.location
        if hist_loc.distance(curr_loc) < distance_threshold:
            return True

        return False


Analyzer.register_analyzer("ego_collision", Analyzer.ego_collision_analysis)
Analyzer.register_analyzer("ego_offroad", Analyzer.ego_offroad_analysis)
Analyzer.register_analyzer("ego_timeout", Analyzer.ego_timeout_analysis)
Analyzer.register_analyzer("npc_done", Analyzer.npc_done_analysis)
