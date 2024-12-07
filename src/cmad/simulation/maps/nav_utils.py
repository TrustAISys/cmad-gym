from __future__ import annotations

import math
from collections import deque
from typing import Deque, List, Optional, Tuple, Union

import carla
import numpy as np
import shapely

from cmad.simulation.PythonAPI.carla.agents.navigation import (
    BasicAgent,
    GlobalRoutePlanner,
    RoadOption,
)
from cmad.simulation.PythonAPI.carla.agents.tools import vector

default_opt_dict = {
    "ignore_traffic_lights": True,
    "ignore_vehicles": False,
    "ignore_stop_signs": True,
    "sampling_resolution": 2.0,
    "base_vehicle_threshold": 5.0,
    "base_tlight_threshold": 5.0,
    "max_brake": 0.5,
}


def route_complement(
    route: List[carla.Waypoint],
) -> List[Tuple[carla.Waypoint, RoadOption]]:
    """Complement the route with the lane follow option.

    Args:
        route (List[carla.Waypoint]): The route to complement.
    """
    new_route = []
    for i in range(len(route) - 1):
        new_route.append((route[i], RoadOption.LANEFOLLOW))

    new_route.append((route[-1], RoadOption.VOID))
    return new_route


class RotatedRectangle(object):
    """
    This class contains method to draw rectangle and find intersection point.
    """

    def __init__(self, c_x, c_y, width, height, angle):
        self.c_x = c_x
        self.c_y = c_y
        self.w = width  # pylint: disable=invalid-name
        self.h = height  # pylint: disable=invalid-name
        self.angle = angle

    def get_contour(self):
        """
        create contour
        """
        w = self.w
        h = self.h
        c = shapely.geometry.box(-w / 2.0, -h / 2.0, w / 2.0, h / 2.0)
        rc = shapely.affinity.rotate(c, self.angle)
        return shapely.affinity.translate(rc, self.c_x, self.c_y)

    def intersection(self, other):
        """
        Obtain a intersection point between two contour.
        """
        return self.get_contour().intersection(other.get_contour())


class PathTracker:
    """Path tracker for the agent to follow the path"""

    def __init__(
        self,
        actor: carla.Actor,
        origin: Tuple[float, float, float],
        destination: Tuple[float, float, float],
        target_speed: float = 20,
        opt_dict: Optional[dict] = None,
        planned_route: Optional[List[Tuple[carla.Waypoint, RoadOption]]] = None,
    ):
        """
        Args:
            origin (Tuple(float, float, float)): The origin of the path.
            destination (Tuple(float, float, float)): The destination of the path.
            actor (carla.Actor): The actor to be controlled.
            planned_route (List[Tuple(carla.Waypoint, RoadOption)] | None): The route to follow. If None, the route will be planned from the origin to the destination.
        """
        if opt_dict is None:
            opt_dict = default_opt_dict

        self.agent = BasicAgent(actor, target_speed=target_speed, opt_dict=opt_dict)
        self.origin = origin
        self.destination = destination

        self.planned_path: deque[tuple[carla.Waypoint, RoadOption]] = None
        self.planned_wpts: list[carla.Waypoint] = None
        self.last_location: carla.Location = None
        self.nearest_waypoint_idx: int = 0
        self.distance_cache: float = 0.0

        if planned_route is None or len(planned_route) == 0:
            self.plan_route(self.origin, self.destination)
        else:
            if not isinstance(planned_route[0], tuple):
                planned_route = route_complement(planned_route)

            self.agent.set_global_plan(planned_route)
            self.planned_path = self.get_planner_path().copy()
            self.planned_wpts = [wp[0] for wp in self.planned_path]
            self.origin = (
                planned_route[0][0].transform.location.x,
                planned_route[0][0].transform.location.y,
                planned_route[0][0].transform.location.z,
            )
            self.destination = (
                planned_route[-1][0].transform.location.x,
                planned_route[-1][0].transform.location.y,
                planned_route[-1][0].transform.location.z,
            )
            self.get_nearest_waypoints()

    def plan_route(
        self,
        origin: Tuple[float, float, float],
        destination: Tuple[float, float, float],
    ):
        """Plan the route from the origin to the destination.

        Args:
            origin (Tuple(float, float, float)): The origin of the path.
            destination (Tuple(float, float, float)): The destination of the path.
        """
        self.nearest_waypoint_idx = 0
        self.get_planner_path().clear()

        if self.planned_path is not None and (
            self.origin == origin and self.destination == destination
        ):
            self.agent.set_global_plan(self.planned_path)
        else:
            self.agent.set_destination(
                carla.Location(*destination[:3]),
                carla.Location(*origin[:3]),
                start_from_vehicle=False,
            )
            self.origin = origin
            self.destination = destination
            self.planned_path = self.get_planner_path().copy()
            self.planned_wpts = [wp[0] for wp in self.planned_path]

        self.get_nearest_waypoints()
        self.last_location = None
        self.distance_cache = 0.0

    def get_orientation_diff_to_end(self, in_radians: bool = True) -> float:
        yaw_diff = math.fabs(
            self.agent._vehicle.get_transform().rotation.yaw % 360
            - self.planned_wpts[-1].transform.rotation.yaw % 360
        )

        return math.radians(yaw_diff) if in_radians else yaw_diff

    def get_euclidean_distance_to_end(self) -> float:
        """Get the euclidean distance to the end of the planned path."""
        return self.planned_wpts[-1].transform.location.distance(
            self.agent._vehicle.get_location()
        )

    def get_distance_to_end(self, distance_to_node: bool = True) -> float:
        """Get the distance to the end of the planned path.

        By default, the distance is calculated from the vehicle to the nearest waypoint in the planned route
        and then to the second waypoint and so on. (That is, the distance along the path)

        If distance_to_node is set to False, the distance is calculated only from the nearest waypoint to the end.

        Args:
            distance_to_node (bool, optional): If True, we will include the distance to the first node in the path. Defaults to False.
        """

        # use cache to accelerate the calculation
        last_loc = self.agent._vehicle.get_location()
        if self.last_location is None or self.last_location.distance(last_loc) >= 0.5:
            self.last_location = last_loc
        else:
            return self.distance_cache

        dist = 0
        if self.nearest_waypoint_idx < len(self.planned_wpts):
            if distance_to_node:
                # actor distance to the first node
                dist += self.planned_wpts[
                    self.nearest_waypoint_idx
                ].transform.location.distance(last_loc)

            # distance between nodes
            for i in range(self.nearest_waypoint_idx, len(self.planned_wpts) - 1):
                dist += self.planned_wpts[i].transform.location.distance(
                    self.planned_wpts[i + 1].transform.location
                )
        else:
            return self.planned_wpts[
                self.nearest_waypoint_idx
            ].transform.location.distance(last_loc)

        self.distance_cache = dist
        return dist

    def run_step(self) -> carla.VehicleControl:
        """Run one step of navigation.

        Returns:
            carla.VehicleControl
        """
        behavior = self.agent.run_step()
        return behavior

    def draw(
        self,
        color: Union[str, Tuple[int, int, int]] = (255, 0, 0),
        z: float = 1.0,
        life_time: float = 1.0,
    ):
        """Draw the planned path on the simulator.

        Args:
            color (str | Tuple(int, int, int)): The color of the path. Defaults to "red".
            z (float, optional): The z coordinate of the path. Defaults to 1.0.
            life_time (float, optional): The life time of the path. Defaults to 1.0.
        """

        if isinstance(color, (tuple, list)):
            color = carla.Color(*color)
        elif color == "red":
            color = carla.Color(255, 0, 0)
        elif color == "green":
            color = carla.Color(0, 255, 0)
        elif color == "blue":
            color = carla.Color(0, 0, 255)
        elif color == "yellow":
            color = carla.Color(255, 255, 0)
        elif color == "random":
            import random

            color = carla.Color(
                random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            )
        else:
            color = carla.Color(255, 0, 0)

        world = self.agent._vehicle.get_world()
        waypoints = [wp[0] for wp in self.get_planner_path()]

        for wpt in waypoints:
            wpt_t = wpt.transform
            begin = wpt_t.location + carla.Location(z=z)
            angle = math.radians(wpt_t.rotation.yaw)
            end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
            world.debug.draw_arrow(
                begin, end, arrow_size=0.3, color=color, life_time=life_time
            )

    def get_planner_path(self) -> Deque[Tuple[carla.Waypoint, RoadOption]]:
        """Get the path planned by the local planner.

        This path is dynamically updated by the local planner after each call to run_step().

        Returns:
            deque[Tuple(carla.Waypoint, RoadOption)]
        """
        path = self.agent.get_local_planner().get_plan()
        return path

    def get_nearest_waypoints(
        self, nums: int = 2, interval: float = 2.0, distance_threshold: float = math.inf
    ) -> List[carla.Waypoint]:
        """Get the nearest waypoint along the planned path (that haven't been through) to the vehicle.

        This function by default will return the nearest waypoint and its successors if any.

        Args:
            nums (int, optional): The number of waypoints to return. Defaults to 2.
            interval (float, optional): The interval between two waypoints. Defaults to 2.0.
            distance_threshold (float, optional): The distance limit that considered as "reached" nearest waypoint. Defaults to math.inf, meaning no limit.

        Returns:
            list[carla.Waypoint] -- The nearest waypoint and its successors if any.
        """

        # The interval can't be smaller than the sampling resolution
        interval = max(interval, self.agent._sampling_resolution)

        if self.nearest_waypoint_idx >= len(self.planned_wpts):
            nearest_wpts = [self.planned_wpts[-1]]
        else:
            actor_loc = self.agent._vehicle.get_location()
            min_distance = math.inf
            min_index = -1

            for i in range(self.nearest_waypoint_idx, len(self.planned_wpts)):
                node_loc = self.planned_wpts[i].transform.location
                dist = actor_loc.distance(node_loc)
                if dist <= min_distance:
                    min_distance = dist
                    min_index = i

            if min_distance <= distance_threshold:
                self.nearest_waypoint_idx = min_index

            slicing_interval = int(interval / self.agent._sampling_resolution)
            nearest_wpts = self.planned_wpts[
                self.nearest_waypoint_idx : self.nearest_waypoint_idx
                + nums * slicing_interval : slicing_interval
            ]

        while len(nearest_wpts) < nums:
            wpt_next = nearest_wpts[-1].next(interval)
            if len(wpt_next) > 0:
                nearest_wpts.append(wpt_next[0])
            else:
                nearest_wpts.append(nearest_wpts[-1])

        return nearest_wpts

    def get_waypoint_in_distance(
        self, distance: float, stop_at_junction: bool = True
    ) -> Tuple[carla.Waypoint, float]:
        """
        Obtain a waypoint in a given distance from the current actor's location.
        Note: Search is stopped on first intersection.
        @return obtained waypoint and the traveled distance
        """
        waypoint = self.agent._map.get_waypoint(self.agent._vehicle.get_location())
        traveled_distance = 0
        while (
            not (waypoint.is_intersection and stop_at_junction)
            and traveled_distance < distance
        ):
            wp_next = waypoint.next(1.0)
            if wp_next:
                waypoint_new = wp_next[-1]
                traveled_distance += waypoint_new.transform.location.distance(
                    waypoint.transform.location
                )
                waypoint = waypoint_new
            else:
                break

        return waypoint, traveled_distance

    def detect_lane_obstacle(
        self, extension_factor: float = 3, margin: float = 1.02
    ) -> bool:
        """
        This function identifies if an obstacle is present in front of the reference actor
        """
        world_actors = self.agent._world.get_actors().filter("vehicle.*")
        actor_bbox = self.agent._vehicle.bounding_box
        actor_transform = self.agent._vehicle.get_transform()
        actor_location = actor_transform.location
        actor_vector = actor_transform.rotation.get_forward_vector()
        actor_vector = np.array([actor_vector.x, actor_vector.y])
        actor_vector = actor_vector / np.linalg.norm(actor_vector)
        actor_vector = actor_vector * (extension_factor - 1) * actor_bbox.extent.x
        actor_location = actor_location + carla.Location(
            actor_vector[0], actor_vector[1]
        )
        actor_yaw = actor_transform.rotation.yaw

        is_hazard = False
        for adversary in world_actors:
            if (
                adversary.id != self.agent._vehicle.id
                and actor_transform.location.distance(adversary.get_location()) < 50
            ):
                adversary_bbox = adversary.bounding_box
                adversary_transform = adversary.get_transform()
                adversary_loc = adversary_transform.location
                adversary_yaw = adversary_transform.rotation.yaw
                overlap_adversary = RotatedRectangle(
                    adversary_loc.x,
                    adversary_loc.y,
                    2 * margin * adversary_bbox.extent.x,
                    2 * margin * adversary_bbox.extent.y,
                    adversary_yaw,
                )
                overlap_actor = RotatedRectangle(
                    actor_location.x,
                    actor_location.y,
                    2 * margin * actor_bbox.extent.x * extension_factor,
                    2 * margin * actor_bbox.extent.y,
                    actor_yaw,
                )
                overlap_area = overlap_adversary.intersection(overlap_actor).area
                if overlap_area > 0:
                    is_hazard = True
                    break

        return is_hazard

    def get_closest_traffic_light(
        self, traffic_lights: Optional[List[carla.TrafficLight]] = None
    ) -> carla.TrafficLight:
        """
        Returns the traffic light closest to the vehicle. The distance is computed between the
        waypoint and the traffic light's bounding box.
        Checks all traffic lights part of 'traffic_lights', or all the town ones, if None are passed.
        """
        if not traffic_lights:
            traffic_lights = self.agent._world.get_actors().filter("*traffic_light*")

        closest_dist = float("inf")
        closest_tl = None

        waypoint = self.agent._map.get_waypoint(self.agent._vehicle.get_location())
        wp_location = waypoint.transform.location
        for tl in traffic_lights:
            tl_waypoints = tl.get_stop_waypoints()
            for tl_waypoint in tl_waypoints:
                distance = wp_location.distance(tl_waypoint.transform.location)
                if distance < closest_dist:
                    closest_dist = distance
                    closest_tl = tl

        return closest_tl

    @staticmethod
    def choose_at_junction(
        current_waypoint: carla.Waypoint,
        next_choices: List[carla.Waypoint],
        direction: int = 0,
    ) -> carla.Waypoint:
        """
        This function chooses the appropriate waypoint from next_choices based on direction
        """
        current_transform = current_waypoint.transform
        current_location = current_transform.location
        projected_location = current_location + carla.Location(
            x=math.cos(math.radians(current_transform.rotation.yaw)),
            y=math.sin(math.radians(current_transform.rotation.yaw)),
        )
        current_vector = vector(current_location, projected_location)
        cross_list = []
        cross_to_waypoint = {}
        for waypoint in next_choices:
            waypoint = waypoint.next(10)[0]
            select_vector = vector(current_location, waypoint.transform.location)
            cross = np.cross(current_vector, select_vector)[2]
            cross_list.append(cross)
            cross_to_waypoint[cross] = waypoint
        select_cross = None
        if direction > 0:
            select_cross = max(cross_list)
        elif direction < 0:
            select_cross = min(cross_list)
        else:
            select_cross = min(cross_list, key=abs)

        return cross_to_waypoint[select_cross]

    @staticmethod
    def generate_target_waypoint_list(
        waypoint: carla.Waypoint,
        turn: int = 0,
        max_waypoint: int = 50,
        vehicle_yaw: Optional[float] = None,
    ) -> Tuple[
        Optional[Deque[Tuple[carla.Waypoint, RoadOption]]],
        Optional[carla.Waypoint],
    ]:
        """
        This method follows waypoints to a junction and chooses a path based on turn input.
        Turn input: LEFT -> -1, RIGHT -> 1, STRAIGHT -> 0
        @returns a waypoint list from the starting point to the end point according to turn input
        """
        reached_junction = False
        threshold = math.radians(0.1)
        plan: Deque[Tuple[carla.Waypoint, RoadOption]] = deque()

        def is_opposite_direction(yaw1: float, yaw2: float) -> bool:
            """Check if two yaws are generally in opposite directions."""

            # Normalize yaws to [0, 360)
            yaw1 = (yaw1 + 360) % 360
            yaw2 = (yaw2 + 360) % 360

            # Calculate the absolute difference and ensure it's in [0, 180)
            diff = abs(yaw1 - yaw2)
            diff = min(360 - diff, diff)

            return diff >= 90

        def get_next_waypoint(waypoint: carla.Waypoint, distance: float):
            """Get the next waypoint based on vehicle's heading."""
            if vehicle_yaw is not None and is_opposite_direction(
                vehicle_yaw, waypoint.transform.rotation.yaw
            ):
                return waypoint.previous(distance)
            return waypoint.next(distance)

        while len(plan) < max_waypoint:
            wp_choice = get_next_waypoint(waypoint, 2)

            if not wp_choice:
                break
            elif len(wp_choice) > 1:
                reached_junction = True
                waypoint = PathTracker.choose_at_junction(waypoint, wp_choice, turn)
            else:
                waypoint = wp_choice[0]

            plan.append((waypoint, RoadOption.LANEFOLLOW))

            # End condition for the behavior
            if turn != 0 and reached_junction and len(plan) >= 3:
                v_1 = vector(
                    plan[-2][0].transform.location, plan[-1][0].transform.location
                )
                v_2 = vector(
                    plan[-3][0].transform.location, plan[-2][0].transform.location
                )
                angle_wp = math.acos(
                    np.clip(
                        np.dot(v_1, v_2)
                        / abs((np.linalg.norm(v_1) * np.linalg.norm(v_2))),
                        -1,
                        1,
                    )
                )
                if angle_wp < threshold:
                    break
            elif reached_junction and not plan[-1][0].is_intersection:
                break

        return (plan, plan[-1][0]) if len(plan) > 0 else (None, None)

    @staticmethod
    def generate_target_waypoint_list_multilane(
        waypoint: carla.Waypoint,
        change: str = "left",
        distance_same_lane: float = 1,
        distance_other_lane: float = 3,
        total_lane_change_distance: float = 5,
        check: bool = True,
        lane_changes: float = 1,
        step_distance: float = 2,
        vehicle_yaw: Optional[float] = None,
    ) -> Tuple[Optional[Deque[Tuple[carla.Waypoint, RoadOption]]], Optional[int]]:
        """
        This methods generates a waypoint list which leads the vehicle to a parallel lane.
        The change input must be 'left' or 'right', depending on which lane you want to change.

        The default step distance between waypoints on the same lane is 2m.
        The default step distance between the lane change is set to 5m.

        @returns a waypoint list from the starting point to the end point on a right or left parallel lane.
        The function might break before reaching the end point, if the asked behavior is impossible.
        """

        def is_opposite_direction(yaw1: float, yaw2: float):
            """Check if two yaws are generally in opposite directions."""

            # Normalize yaws to [0, 360)
            yaw1 = (yaw1 + 360) % 360
            yaw2 = (yaw2 + 360) % 360

            # Calculate the absolute difference and ensure it's in [0, 180)
            diff = abs(yaw1 - yaw2)
            diff = min(360 - diff, diff)

            return diff >= 90

        def get_next_waypoint(waypoint: carla.Waypoint, distance: float):
            """Get the next waypoint based on vehicle's heading."""
            if vehicle_yaw is not None and is_opposite_direction(
                vehicle_yaw, waypoint.transform.rotation.yaw
            ):
                return waypoint.previous(distance)
            return waypoint.next(distance)

        plan = []
        plan.append((waypoint, RoadOption.LANEFOLLOW))  # start position

        option = RoadOption.LANEFOLLOW

        # Same lane
        distance = 0
        while distance < distance_same_lane:
            next_wps = get_next_waypoint(plan[-1][0], step_distance)
            if not next_wps:
                return None, None
            next_wp = next_wps[0]
            distance += next_wp.transform.location.distance(
                plan[-1][0].transform.location
            )
            plan.append((next_wp, RoadOption.LANEFOLLOW))

        if change == "left":
            option = RoadOption.CHANGELANELEFT
        elif change == "right":
            option = RoadOption.CHANGELANERIGHT
        else:
            # ERROR, input value for change must be 'left' or 'right'
            return None, None

        lane_changes_done = 0
        lane_change_distance = total_lane_change_distance / lane_changes

        # Lane change
        while lane_changes_done < lane_changes:
            # Move forward
            next_wps = get_next_waypoint(plan[-1][0], lane_change_distance)
            if not next_wps:
                return None, None
            next_wp = next_wps[0]

            # Get the side lane
            if vehicle_yaw is not None and is_opposite_direction(
                vehicle_yaw, next_wp.transform.rotation.yaw
            ):
                left_lane = next_wp.get_right_lane
                right_lane = next_wp.get_left_lane
            else:
                left_lane = next_wp.get_left_lane
                right_lane = next_wp.get_right_lane

            if change == "left":
                if check and str(next_wp.lane_change) not in ["Left", "Both"]:
                    return None, None
                side_wp = left_lane()
            else:
                if check and str(next_wp.lane_change) not in ["Right", "Both"]:
                    return None, None
                side_wp = right_lane()

            if not side_wp or side_wp.lane_type != carla.LaneType.Driving:
                return None, None

            # Update the plan
            plan.append((side_wp, option))
            lane_changes_done += 1

        # Other lane
        distance = 0
        while distance < distance_other_lane:
            next_wps = get_next_waypoint(plan[-1][0], step_distance)
            if not next_wps:
                return None, None
            next_wp = next_wps[0]
            distance += next_wp.transform.location.distance(
                plan[-1][0].transform.location
            )
            plan.append((next_wp, RoadOption.LANEFOLLOW))

        target_lane_id = plan[-1][0].lane_id

        return plan, target_lane_id

    @staticmethod
    def get_offset_transform(
        transform: carla.Transform, offset: float
    ) -> carla.Transform:
        """
        This function adjusts the give transform by offset and returns the new transform.
        """
        if offset != 0:
            forward_vector = transform.rotation.get_forward_vector()
            orthogonal_vector = carla.Vector3D(
                x=-forward_vector.y, y=forward_vector.x, z=forward_vector.z
            )
            transform.location.x = transform.location.x + offset * orthogonal_vector.x
            transform.location.y = transform.location.y + offset * orthogonal_vector.y
        return transform

    @staticmethod
    def get_troad_from_transform(
        map: carla.Map, actor_transform: carla.Transform
    ) -> float:
        """
        This function finds the lateral road position (t) from actor_transform
        """
        actor_loc = actor_transform.location
        c_wp = map.get_waypoint(actor_loc)
        left_lanes, right_lanes = [], []
        # opendrive standard: (left ==> +ve lane_id) and (right ==> -ve lane_id)
        ref_lane = map.get_waypoint_xodr(c_wp.road_id, 0, c_wp.s)
        for i in range(-50, 50):
            _wp = map.get_waypoint_xodr(c_wp.road_id, i, c_wp.s)
            if _wp:
                if i < 0:
                    left_lanes.append(_wp)
                elif i > 0:
                    right_lanes.append(_wp)

        if left_lanes:
            left_lane_ids = [ln.lane_id for ln in left_lanes]
            lm_id = min(left_lane_ids)
            lm_lane = left_lanes[left_lane_ids.index(lm_id)]
            lm_lane_offset = lm_lane.lane_width / 2
        else:
            lm_lane, lm_lane_offset = ref_lane, 0
        lm_tr = PathTracker.get_offset_transform(
            carla.Transform(lm_lane.transform.location, lm_lane.transform.rotation),
            lm_lane_offset,
        )
        distance_from_lm_lane_edge = lm_tr.location.distance(actor_loc)
        distance_from_lm_lane_ref_lane = lm_tr.location.distance(
            ref_lane.transform.location
        )
        if right_lanes:
            right_lane_ids = [ln.lane_id for ln in right_lanes]
            rm_id = max(right_lane_ids)
            rm_lane = right_lanes[right_lane_ids.index(rm_id)]
            rm_lane_offset = -rm_lane.lane_width / 2
        else:
            rm_lane, rm_lane_offset = ref_lane, -distance_from_lm_lane_ref_lane
        distance_from_rm_lane_edge = PathTracker.get_offset_transform(
            carla.Transform(rm_lane.transform.location, rm_lane.transform.rotation),
            rm_lane_offset,
        ).location.distance(actor_loc)
        t_road = ref_lane.transform.location.distance(actor_loc)
        if not right_lanes or not left_lanes:
            closest_road_edge = min(
                distance_from_lm_lane_edge, distance_from_rm_lane_edge
            )
            if closest_road_edge == distance_from_lm_lane_edge:
                t_road = -1 * t_road
        else:
            if c_wp.lane_id < 0:
                t_road = -1 * t_road

        return t_road

    @staticmethod
    def get_distance_between_actors(
        map: carla.Map,
        current: carla.Actor,
        target: carla.Actor,
        distance_type: str = "euclidianDistance",
        freespace: bool = False,
        global_planner: Optional[GlobalRoutePlanner] = None,
    ) -> float:
        """
        This function finds the distance between actors for different use cases described by distance_type and freespace
        attributes
        """

        target_transform = target.get_transform()
        current_transform = current.get_transform()
        target_wp = map.get_waypoint(target_transform.location)
        current_wp = map.get_waypoint(current_transform.location)

        extent_sum_x, extent_sum_y = 0, 0
        if freespace:
            if isinstance(target, (carla.Vehicle, carla.Walker)):
                extent_sum_x = (
                    target.bounding_box.extent.x + current.bounding_box.extent.x
                )
                extent_sum_y = (
                    target.bounding_box.extent.y + current.bounding_box.extent.y
                )
        if distance_type == "longitudinal":
            if not current_wp.road_id == target_wp.road_id:
                distance = 0
                # Get the route
                route = global_planner.trace_route(
                    current_transform.location, target_transform.location
                )
                # Get the distance of the route
                for i in range(1, len(route)):
                    curr_loc = route[i][0].transform.location
                    prev_loc = route[i - 1][0].transform.location
                    distance += curr_loc.distance(prev_loc)
            else:
                distance = abs(current_wp.s - target_wp.s)
            if freespace:
                distance = distance - extent_sum_x
        elif distance_type == "lateral":
            target_t = PathTracker.get_troad_from_transform(target_transform)
            current_t = PathTracker.get_troad_from_transform(current_transform)
            distance = abs(target_t - current_t)
            if freespace:
                distance = distance - extent_sum_y
        elif distance_type in ["cartesianDistance", "euclidianDistance"]:
            distance = target_transform.location.distance(current_transform.location)
            if freespace:
                distance = distance - extent_sum_x
        else:
            raise TypeError(f"unknown distance_type: {distance_type}")

        # distance will be negative for feeespace when there is overlap condition
        # truncate to 0.0 when this happens
        distance = 0.0 if distance < 0.0 else distance

        return distance

    @staticmethod
    def is_valid_turn(
        start_waypoint: carla.Waypoint, end_waypoint: carla.Waypoint, direction: str
    ) -> bool:
        """Check if the given turn planning is valid.

        Args:
            start_waypoint (carla.Waypoint): the starting waypoint
            end_waypoint (carla.Waypoint): the ending waypoint
            direction (str): left or right

        Returns:
            bool: Whether it is a valid turn planning in the given direction
        """
        # Get the yaw values for both the starting and ending waypoints
        start_yaw = start_waypoint.transform.rotation.yaw
        end_yaw = end_waypoint.transform.rotation.yaw

        # Calculate the difference in yaw
        yaw_difference = end_yaw - start_yaw

        # Normalize the yaw_difference to be between -180 and 180
        yaw_difference = (yaw_difference + 180) % 360 - 180

        # Determine if the yaw difference corresponds to the desired direction
        if direction == "left" and yaw_difference < 0:
            return True
        elif direction == "right" and yaw_difference > 0:
            return True
        else:
            return False

    @staticmethod
    def relative_position(
        reference_transform: carla.Transform,
        transform: carla.Transform,
        side_range: float = 1.0,
    ) -> str:
        """Given two transforms, tell the relative position of the second transform with respect to the first transform.

        Args:
            reference_transform (Transform): The reference transform.
            transform (Transform): The transform to be compared.
            side_range (float): The range to consider the second transform on the side of the reference transform.

        Returns:
            str: The relative position of the second transform with respect to the first transform.
        """

        def to_numpy(vector: carla.Vector3D):
            return np.array([vector.x, vector.y, vector.z])

        reference_forward = to_numpy(reference_transform.get_forward_vector())
        reference_loc = to_numpy(reference_transform.location)
        current_loc = to_numpy(transform.location)
        ref_to_cur = current_loc - reference_loc

        forward_projection = np.dot(ref_to_cur, reference_forward)
        reference_side = np.array([-reference_forward[1], reference_forward[0], 0])
        side_projection = np.dot(ref_to_cur, reference_side)

        if abs(forward_projection) <= side_range and abs(side_projection) <= side_range:
            return "left_side" if side_projection < 0 else "right_side"
        elif forward_projection > 0:
            return "front"
        elif forward_projection < 0:
            return "rear"
