#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the base class for all autonomous agents
"""

from __future__ import annotations, print_function

from typing import List

import carla

from cmad.simulation.data.sensor_interface import SensorInterface
from cmad.simulation.PythonAPI.carla.agents.navigation import RoadOption


def downsample_route(route: List[carla.Waypoint, RoadOption], sample_factor: float):
    """
    Downsample the route by some factor.
    :param route: the trajectory , has to contain the waypoints and the road options
    :param sample_factor: Maximum distance between samples
    :return: returns the ids of the final route that can
    """

    ids_to_sample = []
    prev_option = None
    dist = 0

    for i, point in enumerate(route):
        curr_option = point[1]

        # Lane changing
        if curr_option in (RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT):
            ids_to_sample.append(i)
            dist = 0

        # When road option changes
        elif prev_option != curr_option and prev_option not in (
            RoadOption.CHANGELANELEFT,
            RoadOption.CHANGELANERIGHT,
        ):
            ids_to_sample.append(i)
            dist = 0

        # After a certain max distance
        elif dist > sample_factor:
            ids_to_sample.append(i)
            dist = 0

        # At the end
        elif i == len(route) - 1:
            ids_to_sample.append(i)
            dist = 0

        # Compute the distance traveled
        else:
            curr_location = point[0].location
            prev_location = route[i - 1][0].location
            dist += curr_location.distance(prev_location)

        prev_option = curr_option

    return ids_to_sample


class AutonomousAgent(object):
    """
    Autonomous agent base class. All user agents have to be derived from this class
    """

    def __init__(self, path_to_conf_file):
        #  current global plans to reach a destination
        self._global_plan = None
        self._global_plan_world_coord = None

        # this data structure will contain all sensor data
        self.sensor_interface = SensorInterface()

        # agent's initialization
        self.setup(path_to_conf_file)

    def setup(self, path_to_conf_file):
        """
        Initialize everything needed by your agent and set the track attribute to the right type:
        """
        pass

    def sensors(self):  # pylint: disable=no-self-use
        """
        Define the sensor suite required by the agent

        :return: a list containing the required sensors in the following format:

        [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Left'},

            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Right'},

            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
             'id': 'LIDAR'}
        ]

        """
        sensors = []

        return sensors

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        :return: control
        """
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0
        control.hand_brake = False

        return control

    def destroy(self):
        """
        Destroy (clean-up) the agent
        :return:
        """
        pass

    def __call__(self):
        """
        Execute the agent call, e.g. agent()
        Returns the next vehicle controls
        """
        input_data = self.sensor_interface.get_data()

        timestamp = None
        control = self.run_step(input_data, timestamp)
        control.manual_gear_shift = False

        return control

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        """
        Set the plan (route) for the agent
        """

        ds_ids = downsample_route(global_plan_world_coord, 1)
        self._global_plan_world_coord = [
            (global_plan_world_coord[x][0], global_plan_world_coord[x][1])
            for x in ds_ids
        ]
        self._global_plan = [global_plan_gps[x] for x in ds_ids]
