from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union

import carla

from cmad.agent.action.action_interface import AbstractAction, ActionInterface
from cmad.simulation.data.simulator import Simulator
from cmad.simulation.maps.nav_utils import PathTracker
from cmad.simulation.PythonAPI.carla.agents.navigation import LocalPlanner, RoadOption

Drivable = (
    carla.LaneType.Driving | carla.LaneType.Shoulder | carla.LaneType.Bidirectional
)


class WaypointFollower:
    _planner_cache: Dict[int, LocalPlanner] = {}

    def __init__(
        self,
        actor: carla.Vehicle,
        target_speed: float = None,
        plan: Union[
            list[carla.Waypoint],
            list[carla.Location],
            list[Tuple[carla.Location, RoadOption]],
        ] = None,
        duration: float = math.inf,
    ):
        """This is an atomic behavior to follow waypoints while maintaining a given speed.

        If no plan is provided, the actor will follow its forward waypoints indefinitely.
        Otherwise, the behavior will end with SUCCESS upon reaching the end of the plan.
        If no target velocity is provided, the actor will try to follow the speed limit.

        Args:
            actor (carla.Vehicle): Actor to apply the behavior.
            target_speed (float, optional): Target speed in m/s. Defaults to None, meaning the speed limit of the road.
            plan (Union[List[carla.Waypoint], List[carla.Location], List[Tuple[carla.Location, RoadOption]]], optional): Plan to follow. Defaults to None.
            duration (float, optional): Duration of the behavior in ticks. Defaults to math.inf
        """
        self._actor = actor
        self._target_speed = target_speed
        self._plan = plan
        self.duration = duration

        self._local_planner = None
        self._args_lateral_dict = {"K_P": 1.0, "K_D": 0.01, "K_I": 0.0, "dt": 0.05}

        self.succeed = False
        self._apply_local_planner()

    def _apply_local_planner(self):
        if self._target_speed is None:
            self._target_speed = self._actor.get_speed_limit()

        if self._actor.id not in WaypointFollower._planner_cache:
            self._planner_cache[self._actor.id] = self._local_planner = LocalPlanner(
                self._actor,
                opt_dict={
                    "target_speed": self._target_speed * 3.6,
                    "lateral_control_dict": self._args_lateral_dict,
                },
            )
        else:
            self._local_planner = WaypointFollower._planner_cache[self._actor.id]
            self._local_planner.set_speed(self._target_speed * 3.6)

        if self._plan is not None:
            if isinstance(self._plan[0], carla.Waypoint):
                plan = [(waypoint, RoadOption.LANEFOLLOW) for waypoint in self._plan]
            elif isinstance(self._plan[0], carla.Location):
                plan = []
                for location in self._plan:
                    waypoint = self._local_planner._map.get_waypoint(
                        location, project_to_road=True, lane_type=carla.LaneType.Any
                    )
                    plan.append((waypoint, RoadOption.LANEFOLLOW))
                self._local_planner.set_global_plan(plan)
            else:
                self._local_planner.set_global_plan(self._plan)

    def set_target_speed(self, target_speed: float):
        """set a new target speed for the behavior

        Args:
            target_speed (float): speed in m/s
        """
        self._local_planner.set_speed(target_speed * 3.6)

    def done(self) -> bool:
        """Whether the behavior is done"""
        return self.succeed or self.duration <= 0

    def run_step(self, _) -> carla.VehicleControl:
        """Apply one step of the behavior.

        Args:
            _ (carla.Vehicle): Unused actor reference

        Returns:
            carla.VehicleControl: The control to apply to the actor
        """
        control = None
        if self._actor is not None and self._actor.is_alive:
            control = self._local_planner.run_step(debug=False)
            if self._local_planner.done():
                self.succeed = True

        if control is None:
            control = self._actor.get_control()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 0.0

        self.duration -= 1
        return control


class Forward(WaypointFollower):
    def __init__(
        self,
        actor: carla.Vehicle,
        target_speed: Optional[float] = None,
    ):
        """Follow the forward waypoints of the given actor. (LaneFollowing)

        Args:
            actor (carla.Vehicle): The actor to apply the behavior.
            target_speed (float, optional): Target speed in m/s, if None, use the speed limit of the road. Defaults to None.
        """
        map = Simulator.get_map()
        actor_transform = actor.get_transform()
        wpt = map.get_waypoint(actor_transform.location, lane_type=Drivable)
        plan, plan_end = PathTracker.generate_target_waypoint_list(
            wpt, 0, vehicle_yaw=actor_transform.rotation.yaw
        )
        super().__init__(actor, target_speed, plan, 100)


class MakeTurn(WaypointFollower):
    def __init__(
        self,
        actor: carla.Vehicle,
        target_speed: Optional[float] = None,
        direction: str = "left",
    ):
        """Make a turn in the given direction from the given waypoint.

        Args:
            actor (carla.Vehicle): The actor to apply the behavior.
            target_speed (float, optional): Target speed in m/s, if None, use the speed limit of the road. Defaults to None.
            direction (str, optional): The turning direction, "left" or "right". Defaults to "left".
        """
        map = Simulator.get_map()
        actor_transform = actor.get_transform()
        wpt = map.get_waypoint(actor_transform.location, lane_type=Drivable)
        plan, plan_end = PathTracker.generate_target_waypoint_list(
            wpt,
            -1 if direction == "left" else 1,
            vehicle_yaw=actor_transform.rotation.yaw,
        )
        super().__init__(actor, target_speed, plan, 80)

    @staticmethod
    def is_valid(
        waypoint: carla.Waypoint, direction: str, vehicle_yaw: Optional[float] = None
    ):
        """Check if it is possible to make a turn in the given direction from the given waypoint.

        Args:
            waypoint (carla.Waypoint): the waypoint to check
            direction (str): left or right
            vehicle_yaw (float, optional): the yaw of the vehicle. If None, use the yaw of the waypoint. Defaults to None.

        Returns:
            bool: Whether it is possible to make a turn in the given direction from the given waypoint
        """
        try:
            plan, plan_end = PathTracker.generate_target_waypoint_list(
                waypoint, -1 if direction == "left" else 1, vehicle_yaw=vehicle_yaw
            )
        except:
            return False

        # if the distance between is too large, we consider it as invalid
        if (
            not plan_end
            or plan_end.transform.location.distance(waypoint.transform.location) > 35
        ):
            return False

        return PathTracker.is_valid_turn(waypoint, plan_end, direction)


class LaneChange(WaypointFollower):
    def __init__(
        self,
        actor: carla.Vehicle,
        target_speed: Optional[float] = None,
        direction: str = "left",
        behavior: str = "normal",
    ):
        if behavior == "aggressive ":
            distance_same_lane = 1
            distance_other_lane = 4
            total_lane_change_distance = 5
            duration = 50
        elif behavior == "normal":
            distance_same_lane = 3
            distance_other_lane = 7
            total_lane_change_distance = 10
            duration = 70
        elif behavior == "cautious":
            distance_same_lane = 5
            distance_other_lane = 10
            total_lane_change_distance = 15
            duration = 90

        map = Simulator.get_map()
        actor_transform = actor.get_transform()
        wpt = map.get_waypoint(actor_transform.location, lane_type=Drivable)
        plan, target_lane_id = PathTracker.generate_target_waypoint_list_multilane(
            wpt,
            change=direction,
            check=False,
            distance_same_lane=distance_same_lane,
            distance_other_lane=distance_other_lane,
            total_lane_change_distance=total_lane_change_distance,
            vehicle_yaw=actor_transform.rotation.yaw,
        )
        super().__init__(actor, target_speed, plan, duration)

    @staticmethod
    def is_valid(
        waypoint: carla.Waypoint,
        direction: str,
        distance_same_lane: float = 2,
        distance_other_lane: float = 4,
        total_lane_change_distance: float = 6,
        vehicle_yaw: Optional[float] = None,
    ):
        """Check if it is possible to change lane in the given direction from the given waypoint.

        Args:
            waypoint (carla.Waypoint): the waypoint to check
            direction (str): left or right
            distance_same_lane (float, optional): the distance to check in the same lane. Defaults to 2.
            distance_other_lane (float), optional): the distance to check in the other lane. Defaults to 4.
            total_lane_change_distance (float, optional): the total distance to check. Defaults to 6.
            vehicle_yaw (float, optional): the yaw of the vehicle. If None, use the yaw of the waypoint. Defaults to None.

        Returns:
            bool: Whether it is possible to change lane in the given direction from the given waypoint
        """
        try:
            plan, target_lane_id = PathTracker.generate_target_waypoint_list_multilane(
                waypoint,
                change=direction,
                check=False,
                distance_same_lane=distance_same_lane,
                distance_other_lane=distance_other_lane,
                total_lane_change_distance=total_lane_change_distance,
                vehicle_yaw=vehicle_yaw,
            )
        except:
            return False

        if plan is None:
            return False

        return True


class StopAction(AbstractAction):
    def __init__(self):
        super().__init__("stop", 20)

    def run_step(self, actor: carla.Vehicle):
        if isinstance(actor, carla.Vehicle):
            control = carla.VehicleControl(
                throttle=0.0, steer=0.0, brake=1.0, hand_brake=True
            )
        else:
            control = carla.WalkerControl(direction=carla.Vector3D(0, 0, 0), speed=0.0)
        return control

    def set_target_speed(self, target_speed: float):
        pass


class PersistAction(AbstractAction):
    _atomic_action_cache: Dict[int, Tuple[str, WaypointFollower]] = {}

    def __init__(self, action, duration: int = 100):
        if not isinstance(action, (list, tuple)):
            action = (action, None)
        self.atomic_action = None
        super().__init__(action, duration)

    def run_step(self, actor: carla.Vehicle):
        if self.atomic_action is None:
            action_type, target_speed = self.action
            self.atomic_action = PersistAction.create_atomic_action(
                action_type, target_speed, actor
            )
            self.duration = self.atomic_action.duration

        self.duration -= 1
        return self.atomic_action.run_step(actor)

    def done(self):
        done = False
        if self.atomic_action:
            done = self.atomic_action.done()

        return done or self.duration <= 0

    @staticmethod
    def create_atomic_action(
        action_type: str,
        target_speed: float,
        actor: carla.Vehicle,
    ):
        """Factory method to create an atomic action based on the action type.

        Args:
            action_type (str): type of the atomic action, e.g. "stop", "left_lane_change", etc.
            target_speed (float): target speed of the atomic action
            actor (carla.Vehicle): the actor to apply the atomic action

        Returns:
            AtomicAction: the created atomic action

        Note:
            We assume action_mask is enabled, so we don't need to check if the action is valid.
        """
        prev_action = PersistAction._atomic_action_cache.get(actor.id, None)

        if prev_action and prev_action[0] == action_type and not prev_action[1].done():
            if target_speed is None:
                target_speed = actor.get_speed_limit() / 3.6
            prev_action[1].set_target_speed(target_speed)
            return prev_action[1]

        if action_type == "stop":
            new_action = StopAction()
        elif action_type == "lane_follow":
            new_action = Forward(actor, target_speed)
        elif action_type == "left_lane_change":
            new_action = LaneChange(actor, target_speed, direction="left")
        elif action_type == "right_lane_change":
            new_action = LaneChange(actor, target_speed, direction="right")
        elif action_type == "turn_left":
            new_action = MakeTurn(actor, target_speed, direction="left")
        elif action_type == "turn_right":
            new_action = MakeTurn(actor, target_speed, direction="right")
        else:
            raise ValueError(f"Unknown action type {action_type}")

        PersistAction._atomic_action_cache[actor.id] = (action_type, new_action)
        return new_action


class VehicleAtomicAction(ActionInterface):
    def __init__(self, action_config: dict):
        super().__init__(action_config)

    def convert_single_action(self, action, done_state=False) -> AbstractAction:
        if done_state:
            return self.stop_action(env_action=False)
        else:
            return PersistAction(action)

    def get_action_mask(
        self, actor: carla.Vehicle, action_space: "tuple[dict, dict]"
    ) -> dict:
        """Return the action mask for the given actor

        Args:
            actor (carla.Vehicle): the actor to get the action mask
            action_space: the action space of the actor

        Returns:
            dict: the action mask
        """
        action_validity = {"stop": 1, "lane_follow": 1}

        wpt = Simulator.get_map().get_waypoint(actor.get_location(), lane_type=Drivable)
        actor_yaw = actor.get_transform().rotation.yaw
        action_validity["turn_left"] = (
            1 if MakeTurn.is_valid(wpt, "left", vehicle_yaw=actor_yaw) else 0
        )
        action_validity["turn_right"] = (
            1 if MakeTurn.is_valid(wpt, "right", vehicle_yaw=actor_yaw) else 0
        )
        action_validity["left_lane_change"] = (
            1 if LaneChange.is_valid(wpt, "left", vehicle_yaw=actor_yaw) else 0
        )
        action_validity["right_lane_change"] = (
            1 if LaneChange.is_valid(wpt, "right", vehicle_yaw=actor_yaw) else 0
        )

        prev_action = PersistAction._atomic_action_cache.get(actor.id, None)
        if prev_action and not prev_action[1].done():
            if "turn" in prev_action[0]:
                action_validity = {k: 0 for k in action_validity}
            action_validity[prev_action[0]] = 1

        return (
            {action: action_validity[action] for action in action_space[0].values()},
            {speed: True for speed in action_space[1].values()},
        )

    def stop_action(
        self, env_action=True, use_discrete=False
    ) -> Union[int, AbstractAction]:
        """Return the stop action representation in atomic action space

        Args:
            env_action (bool): Whether using env action space
            use_discrete (bool): Whether using discrete action space

        Returns:
            list[int, int]: if env_action is True and use_discrete is False, return the stop action in the continuous action space.
            int: if env_action is True and use_discrete is True, return the index of the stop action in the discrete action space.
            PersistAction: if env_action is False, return the stop action in the action space of the action handler.
        """
        if not env_action:
            return PersistAction(("stop", 0))

        return [0, 0]
