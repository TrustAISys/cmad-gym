from __future__ import annotations

import logging
import math
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import carla

from cmad.simulation.maps.nodeid_coord_map import MAP_TO_COORDS_MAPPING
from cmad.misc import retry

logger = logging.getLogger(__name__)


class ScenarioHelper:
    """Helper functions for parsing scenario's configuration"""

    @staticmethod
    def parse_random_range(
        random_range: "tuple[float, float] | list | float",
    ) -> tuple[float, float]:
        """Parse the random range from the given input.

        Args:
            random_range: A tuple or list with two elements defining the lower and upper bound,
            or a float defining the symmetric range around zero.

        Returns:
            lower_bound, upper_bound (tuple).
        """
        if isinstance(random_range, (tuple, list)):
            return tuple(random_range)
        else:
            return (-random_range, random_range)

    @staticmethod
    def get_location_from_index(index: int, carla_map: carla.Map) -> tuple[float, ...]:
        """Return a predefined location from the given index.

        Args:
            index (int): The index of the location
            carla_map (carla.Map): Which map to use

        Returns:
            tuple[float, ...]: The corresponding coordinates
        """
        map_name = carla_map.name.split("/")[-1]
        return tuple(MAP_TO_COORDS_MAPPING[map_name][str(index)])

    @staticmethod
    @retry(exceptions=AttributeError, tries=5, logger=logger)
    def get_location_from_dict(
        loc_conf: dict, carla_map: carla.Map
    ) -> tuple[float, ...]:
        """Parse a location configuration dictionary and return the corresponding coordinates.

        Args:
            loc_conf (dict): location configuration dictionary
            carla_map (carla.Map): Carla map

        Raises:
            ValueError: Unsupported location configuration

        Returns:
            location (tuple[float, ...]): The corresponding coordinates
        """
        h = loc_conf.get("h", None)
        yaw = loc_conf.get("yaw", math.degrees(h) if h is not None else None)
        z = loc_conf.get("z", 2.0)

        # Handle OpenDrive location specification
        if {"road_id", "lane_id", "s"}.issubset(loc_conf):
            road_id, lane_id, s = (
                loc_conf["road_id"],
                loc_conf["lane_id"],
                loc_conf["s"],
            )

            # Apply random offset to 's' if specified
            if "random_range_s" in loc_conf:
                range_value_s = loc_conf["random_range_s"]
                s = random.uniform(range_value_s[0], range_value_s[1])
            elif "random_s" in loc_conf:
                lower, upper = ScenarioHelper.parse_random_range(loc_conf["random_s"])
                s = max(0.1, s + random.uniform(lower, upper))

            waypoint = carla_map.get_waypoint_xodr(road_id, lane_id, s)
            x, y, z = (
                waypoint.transform.location.x,
                waypoint.transform.location.y,
                max(z, waypoint.transform.location.z),
            )

            dh = loc_conf.get("dh", None)
            dyaw = loc_conf.get("dyaw", math.degrees(dh) if dh is not None else None)
            if dyaw is not None:
                yaw = waypoint.transform.rotation.yaw + dyaw
        # Handle absolute position
        elif "x" in loc_conf and "y" in loc_conf:
            x, y = loc_conf["x"], loc_conf["y"]
        # If a nested 'location' is provided, recursively parse the location
        elif "location" in loc_conf:
            location = ScenarioHelper.parse_location(loc_conf["location"], carla_map)
            x, y, z = location[0], location[1], max(location[2], z)
            if len(location) > 3:
                yaw = location[3]
        else:
            raise ValueError(
                "Dictionary location configuration must contain one of: 'location' /"
                "'road_id', 'lane_id', 's' / 'x' and 'y' keys."
            )

        # Apply random offsets if specified
        variables = {"x": x, "y": y, "z": z, "yaw": yaw}
        for axis in variables.keys():
            random_key = f"random_{axis}"
            random_range_key = f"random_range_{axis}"

            if random_range_key in loc_conf:
                range_value = loc_conf[random_range_key]
                variables[axis] = random.uniform(range_value[0], range_value[1])
            elif random_key in loc_conf:
                lower, upper = ScenarioHelper.parse_random_range(loc_conf[random_key])
                variables[axis] += random.uniform(lower, upper)

        x, y, z, yaw = variables.values()
        return (x, y, z) if yaw is None else (x, y, z, yaw)

    @staticmethod
    def parse_location(
        loc_conf: "int | tuple[float, ...] | list | dict", carla_map: carla.Map
    ):
        """Parse a location configuration and return the corresponding coordinates.

        Args:
            loc_conf (int | tuple[float, ...] | list | dict): location configuration
            carla_map (carla.Map): map in which the location is defined

        Raises:
            ValueError: Unsupported location configuration

        Returns:
            location (tuple[float, ...]): The corresponding coordinates (x, y, z, yaw)
        """
        if isinstance(loc_conf, int):
            return ScenarioHelper.get_location_from_index(loc_conf, carla_map)
        elif isinstance(loc_conf, (tuple, list)):
            return tuple(loc_conf)
        elif isinstance(loc_conf, dict):
            return ScenarioHelper.get_location_from_dict(loc_conf, carla_map)
        else:
            raise ValueError(f"Unsupported location configuration: {loc_conf}")
