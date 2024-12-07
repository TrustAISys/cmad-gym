from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import carla

if TYPE_CHECKING:
    from cmad.envs.multi_env import MultiCarlaEnv

from cmad.agent.action.action_interface import AbstractAction, ActionInterface
from cmad.simulation.data.local_carla_api import Waypoint
from cmad.simulation.PythonAPI.carla.agents.navigation import (LocalPlanner,
                                                               RoadOption)


class WaypointFollower:
    def __init__(
        self,
        actor: carla.Vehicle,
        target_speed: float = None,
        plan: List[Tuple[carla.Waypoint, RoadOption]] = None,
        start_from_vehicle: bool = False,
    ):
        """This is an atomic behavior to follow waypoints while maintaining a given speed.

        If no plan is provided, the actor will follow its foward waypoints indefinetely.
        Otherwise, the behavior will end with SUCCESS upon reaching the end of the plan.
        If no target velocity is provided, the actor will try to follow the speed limit.

        Args:
            actor (carla.Vehicle): Actor to apply the behavior.
            target_speed (float, optional): Target speed in m/s. Defaults to None, meaning the speed limit of the road.
            plan (List[carla.Waypoint] | List[carla.Location] | List[Tuple[carla.Location, RoadOption]]): Plan to follow. Defaults to None.
            start_from_vehicle (bool, optional): Whether to start the plan from the current vehicle location. Defaults to False.
        """
        self._actor = actor
        self._target_speed = target_speed
        self._plan = plan
        self._start_from_vehicle = start_from_vehicle

        self._local_planner = None
        self._args_lateral_dict = {"K_P": 1.0, "K_D": 0.01, "K_I": 0.0, "dt": 0.05}

        self.succeed = False
        self._apply_local_planner()

    def _apply_local_planner(self):
        if self._target_speed is None:
            self._target_speed = self._actor.get_speed_limit()

        self._local_planner = LocalPlanner(
            self._actor,
            opt_dict={
                "target_speed": self._target_speed * 3.6,
                "lateral_control_dict": self._args_lateral_dict,
            },
        )

        if self._plan is not None:
            if isinstance(self._plan[0], carla.Waypoint):
                self._plan = [(waypoint, RoadOption.LANEFOLLOW) for waypoint in self._plan]
            elif isinstance(self._plan[0], carla.Location):
                pseudo_rotation = carla.Rotation()
                self._plan = list(
                    map(
                        lambda location: (
                            Waypoint(transform=carla.Transform(location, pseudo_rotation)),
                            RoadOption.LANEFOLLOW,
                        ),
                        self._plan,
                    )
                )
            elif isinstance(self._plan[0], list):
                pseudo_rotation = carla.Rotation()
                self._plan = list(
                    map(
                        lambda point: (
                            Waypoint(transform=carla.Transform(carla.Location(*point), pseudo_rotation)),
                            RoadOption.LANEFOLLOW,
                        ),
                        self._plan,
                    )
                )
            self._local_planner.set_global_plan(self._plan_from_vehicle())

    def _plan_from_vehicle(self):
        """Slice the plan to start from the current vehicle location"""
        if not self._start_from_vehicle:
            return self._plan

        # Find the nearest waypoint to the vehicle
        vehicle_location = self._actor.get_location()
        min_distance = float("inf")
        nearest_index = 0
        for i, (waypoint, _) in enumerate(self._plan):
            distance = waypoint.transform.location.distance(vehicle_location)
            if distance < min_distance:
                min_distance = distance
                nearest_index = i
        return self._plan[nearest_index:]

    def set_target_speed(self, target_speed: float):
        """set a new target speed for the behavior

        Args:
            target_speed (float): speed in m/s
        """
        self._local_planner.set_speed(target_speed * 3.6)

    def reset_plan(self):
        self._local_planner.set_global_plan(self._plan_from_vehicle())
        self.succeed = False

    def done(self) -> bool:
        """Whether the behavior is done"""
        local_plan = self._local_planner.get_plan()
        if (
            len(local_plan) == 1
            and self._actor.get_location().distance(local_plan[0][0].transform.location)
            < 5
        ):
            self.succeed = True

        return self.succeed

    def run_step(self, actor: carla.Vehicle) -> carla.VehicleControl:
        """Apply one step of the behavior.

        Args:
            actor (carla.Vehicle): Unused actor reference

        Returns:
            carla.VehicleControl: The control to apply to the actor
        """
        self._actor = actor

        control = None
        if not self.succeed and actor and actor.is_active:
            self._local_planner.reset_vehicle(actor)
            control = self._local_planner.run_step(debug=False)
            if self._local_planner.done():
                self.succeed = True

        if control is None:
            control = carla.VehicleControl()

        return control


class FollowAction(AbstractAction):
    def __init__(
        self,
        action,
        duration: int = 20,
        waypoints: Optional[list] = None,
        loop: bool = False,
        start_from_vehicle: bool = False,
    ):
        super().__init__(action, duration)
        self.waypoints = waypoints
        self.target_speed = self.action
        self.loop = loop
        self.start_from_vehicle = start_from_vehicle
        self.follower = None

    def run_step(self, actor: carla.Vehicle):
        self.duration -= 1

        if self.follower is None:
            self.follower = WaypointFollower(
                actor,
                target_speed=self.target_speed,
                plan=self.waypoints,
                start_from_vehicle=self.start_from_vehicle,
            )
        elif self.loop and self.follower.done():
            self.reset_follower()
        else:
            self.follower.set_target_speed(self.target_speed)

        return self.follower.run_step(actor)

    def reset_follower(self):
        if self.follower:
            self.follower.reset_plan()

    def update_action(self, action: float):
        self.duration = 20
        self.action = action
        self.target_speed = action

    def done(self):
        done = False
        if not self.loop and self.follower:
            done = self.follower.done()

        return done or self.duration <= 0


class VehicleRouteAction(ActionInterface):
    def __init__(self, action_config: dict):
        super().__init__(action_config)
        self.current_episode = -1
        self.env: "MultiCarlaEnv" = action_config["env"]

        self.follow_action = FollowAction(
            0,
            waypoints=action_config["waypoints"],
            loop=action_config.get("loop", False),
            start_from_vehicle=action_config.get("start_from_vehicle", False),
        )

    def convert_single_action(
        self, action: float, done_state: bool = False
    ) -> AbstractAction:
        if self.current_episode != self.env.episode_id:
            self.follow_action.reset_follower()
            self.current_episode = self.env.episode_id

        if done_state:
            return self.stop_action(env_action=False)
        else:
            self.follow_action.update_action(action)
            return self.follow_action

    def get_action_mask(self, actor: carla.Vehicle, action_space: dict):
        return True

    def stop_action(self, env_action: bool = True, use_discrete: bool = False):
        if not env_action:
            self.follow_action.update_action(0)
            return self.follow_action

        return 0
