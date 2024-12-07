from __future__ import annotations

import carla
import numpy as np

from typing import Dict, Tuple
from cmad.simulation.data.local_carla_api import Waypoint
from cmad.simulation.data.measurement import (
    CollisionRecord,
    LaneInvasionRecord,
    Measurement,
    Measurements,
)
from cmad.simulation.data.simulator import Simulator


class DataCollector:
    """
    This class is used to collect all data from CarlaDataProvider and SensorDataProvider after tick in each step.
    """

    @staticmethod
    def get_data(
        id_map: Dict[str, int],
        configs: Dict[str, dict],
    ) -> Measurements:
        """Get all data from CarlaDataProvider and SensorDataProvider

        Args:
            id_map (dict[str, int]): A mapping from actor.id to actor_id.
            configs (dict[str, dict]): A mapping from actor_id to config.
        """
        measurements = {}
        for actor_id, id in id_map.items():
            config = configs[actor_id]

            # Carla Data
            transform = Simulator.get_actor_transform(id, use_local_api=True)
            forward_vector = Simulator.get_actor_forward(id, use_local_api=True)
            velocity = Simulator.get_actor_velocity(id, use_local_api=True)
            acceleration = Simulator.get_actor_acceleration(id, use_local_api=True)
            bounding_box = Simulator.get_actor_bounding_box(id, use_local_api=True)
            waypoint = Simulator.get_actor_waypoint(id, use_local_api=False)

            road_curvature = DataCollector.calculate_road_curvature(waypoint)
            waypoint = Waypoint.from_simulator_waypoint(waypoint)

            # Sensor Data
            colli_record = CollisionRecord()
            collision_sensor = Simulator.get_actor_collision_sensor(actor_id)
            if collision_sensor:
                colli_record.vehicles = collision_sensor.collision_vehicles
                colli_record.pedestrians = collision_sensor.collision_pedestrians
                colli_record.others = collision_sensor.collision_other
                colli_record.id_set = collision_sensor.collision_id_set

            lane_record = LaneInvasionRecord()
            lane_sensor = Simulator.get_actor_lane_invasion_sensor(actor_id)
            if lane_sensor:
                lane_record.otherlane = lane_sensor.offlane
                lane_record.offroad = lane_sensor.offroad

            # camera data
            camera_dict = Simulator.get_actor_camera_data(actor_id)

            measurements[actor_id] = Measurement(
                actor_id=actor_id,
                type=config.get("type", "vehicle_4W"),
                transform=transform,
                forward_vector=forward_vector,
                velocity=velocity,
                speed=velocity.length(),
                acceleration=acceleration,
                bounding_box=bounding_box,
                waypoint=waypoint,
                road_curvature=road_curvature,
                collision=colli_record,
                lane_invasion=lane_record,
                camera_dict=camera_dict,
            )
        return measurements

    @staticmethod
    def calculate_ttc(
        measurements: Measurements, lead_actor_id: str, follow_actor_id: str
    ):
        """
        Calculate the time to collision between two actors. Note, this function assume linear motion.

        Args:
            lead_actor_id (str): The actor id of the lead vehicle.
            follow_actor_id (str): The actor id of the following vehicle.

        Returns:
            float: The time to collision between the two actors.
        """
        ego = measurements[lead_actor_id]
        npc = measurements[follow_actor_id]

        # Calculate relative velocity
        rel_vel = (ego.velocity - npc.velocity).as_numpy_array()

        # Calculate relative distance
        rel_dist = (ego.transform.location - npc.transform.location).as_numpy_array()

        # Calculate relative speed (dot product of relative velocity and distance normalized)
        rel_speed = np.dot(rel_vel, rel_dist) / np.linalg.norm(rel_dist)

        # If the relative speed is not positive, the actors are not moving towards each other, so TTC is infinite
        if rel_speed <= 0:
            return float("inf")

        # Calculate TTC as relative distance divided by relative speed
        ttc = np.linalg.norm(rel_dist) / rel_speed

        return ttc

    @staticmethod
    def calculate_distance(
        measurements: Measurements, ego_actor_id: str, other_actor_id: str
    ) -> Tuple[float, float, float]:
        """
        Calculate the elucidean, lateral and longitudinal distance between two actors.

        Args:
            measurements (dict): A dictionary containing all measurements.
            ego_actor_id (str): The actor id of the ego vehicle.
            other_actor_id (str): The actor id of the other vehicle.

        Returns:
            elucidean distance (float): The elucidean distance between the two actors.
            lateral distance (float): The lateral distance between the two actors.
            longitudinal distance (float): The longitudinal distance between the two actors.
        """
        ego_loc_np = measurements[ego_actor_id].transform.location.as_numpy_array()
        ego_forward_np = measurements[ego_actor_id].forward_vector.as_numpy_array()
        other_loc_np = measurements[other_actor_id].transform.location.as_numpy_array()

        location_diff_np = other_loc_np - ego_loc_np
        longitudinal_distance = np.dot(location_diff_np, ego_forward_np)
        lateral_distance = np.linalg.norm(np.cross(location_diff_np, ego_forward_np))

        return np.linalg.norm(location_diff_np), lateral_distance, longitudinal_distance

    @staticmethod
    def calculate_bb_distance(
        measurements: Measurements, ego_id: str, actor_id: str
    ) -> float:
        """Calculate distance between the nearest point of two actor's bounding box

        Args:
            measurements (dict): A dictionary containing all measurements.
            ego_id (str): The actor id of the ego vehicle.
            actor_id (str): The actor id of the other vehicle.

        Returns:
            distance (float): The distance between the two actor's bounding box.
        """
        from shapely.geometry import Polygon

        ego_m, actor_m = measurements[ego_id], measurements[actor_id]
        ego_bb, actor_bb = ego_m.bounding_box, actor_m.bounding_box

        ego_world_bb = ego_bb.transform_to_world_frame(ego_m.transform, True)
        actor_world_bb = actor_bb.transform_to_world_frame(actor_m.transform, True)
        return Polygon(ego_world_bb).distance(Polygon(actor_world_bb))

    @staticmethod
    def calculate_road_curvature(
        host_waypoint: carla.Waypoint, route_distance: float = 2.0
    ):
        """Calculate the road curvature for a given waypoint. (Approximated)

        Args:
            host_waypoint (carla.Waypoint): The waypoint to calculate the road curvature for.
            route_distance (float): The distance along the route to the waypoint to calculate the curvature.

        Returns:
            curvature (float): The road curvature for the given waypoint.
        """
        if host_waypoint is None:
            return 0.0

        try:
            previous_waypoint = host_waypoint.previous(route_distance)[0]
            next_waypoint = host_waypoint.next(route_distance)[0]
        except IndexError:
            # If no previous or next waypoint can be found, return 0.0
            return 0.0

        _transform = next_waypoint.transform
        _location, _rotation = _transform.location, _transform.rotation
        x1, y1 = _location.x, _location.y
        yaw1 = _rotation.yaw

        _transform = previous_waypoint.transform
        _location, _rotation = _transform.location, _transform.rotation
        x2, y2 = _location.x, _location.y
        yaw2 = _rotation.yaw

        c = (
            2
            * np.sin(np.radians((yaw1 - yaw2) / 2))
            / np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        )
        return c

    @staticmethod
    def calculate_orientation_diff(
        vehicle_transform: carla.Transform, waypoint_transform: carla.Transform
    ):
        """Calculate the orientation difference between a vehicle and a waypoint.

        Args:
            vehicle_transform (carla.Transform): The transform of the vehicle.
            waypoint_transform (carla.Transform): The transform of nearest waypoint on the route.

        Returns:
            float: The orientation difference between the vehicle and the waypoint in degree.
        """
        vehicle_yaw = vehicle_transform.rotation.yaw
        waypoint_yaw = waypoint_transform.rotation.yaw
        orientation_diff = (vehicle_yaw - waypoint_yaw + 180) % 360 - 180
        return orientation_diff

    @staticmethod
    def get_lane_point(waypoint: carla.Waypoint):
        """Get the lane point of a waypoint.

        Args:
            waypoint (carla.Waypoint): The waypoint to get the lane point from.

        Returns:
            left_lane_point(carla.Location): The left lane point of the waypoint.
            right_lane_point(carla.Location): The right lane point of the waypoint.
        """
        rvec = waypoint.transform.get_right_vector() * 0.5
        right_loc = waypoint.transform.location + rvec * waypoint.lane_width
        left_loc = waypoint.transform.location - rvec * waypoint.lane_width
        return left_loc, right_loc
