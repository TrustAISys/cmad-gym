#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This file containts CallBack class and SensorInterface, responsible of
handling the use of sensors for the agents
"""
from __future__ import annotations

import copy
import logging
from typing import Dict, Set, Tuple

import carla
import numpy as np

from cmad.simulation.sensors.derived_sensors import CollisionSensor, LaneInvasionSensor
from cmad.viz.carla_birdeye_view import BirdViewProducer

try:
    from queue import Empty, Queue
except ImportError:
    from Queue import Empty, Queue


class SensorReceivedNoData(Exception):
    """
    Exceptions thrown when the sensors used by the agent take too long to receive data
    """


class SensorDataProvider:
    """
    All sensor data will be buffered in this class

    The data can be retreived in following data structure:

    {
        'camera': {
            'actor_id': {
                'sensor_id': (frame: int, processed_data: np.ndarray),
                ...
            },
            ...
        },
        'collision': {
            'actor_id': CollisionSensor,
            ...
        },
        'lane_invasion': {
            'actor_id': LaneInvasionSensor,
            ...
        },
        ...
    }
    """

    _camera_config: Dict[str, dict] = {}
    _camera_data_dict: Dict[str, dict[str, Tuple[int, np.ndarray]]] = {}
    _collision_sensors: Dict[str, CollisionSensor] = {}
    _lane_invasion_sensors: Dict[str, LaneInvasionSensor] = {}
    _birdeye_sensors: Dict[Tuple[int, int], BirdViewProducer] = {}
    _filter_camera_ids: Set[str] = set(["ManualControl"])

    @staticmethod
    def update_camera_config(actor_id: str, config: dict):
        """
        Updates the camera config

        Args:
            actor_id (str): actor id
            config (Dict): camera config
        """
        SensorDataProvider._camera_config[actor_id] = config

    @staticmethod
    def update_camera_data(actor_id: str, data: Dict[str, Tuple[int, np.ndarray]]):
        """Updates the camera data

        Args:
            actor_id (str): actor id
            data (Dict): image data from sensor_interface.get_data(). E.g.

            data = {
                "sensor_id": (frame : int, processed_data : ndarray),
                ...
            }
        """
        if data is not None:
            filter_data = {
                k: v
                for k, v in data.items()
                if k not in SensorDataProvider._filter_camera_ids
            }
            SensorDataProvider._camera_data_dict[actor_id] = filter_data

    @staticmethod
    def update_collision_sensor(actor_id: str, sensor: CollisionSensor):
        """
        Updates a collision sensor
        """
        SensorDataProvider._collision_sensors[actor_id] = sensor

    @staticmethod
    def update_birdeye_sensor(spec: Tuple[int, int], sensor: BirdViewProducer):
        """Updates a birdeye sensor"""
        SensorDataProvider._birdeye_sensors[spec] = sensor

    @staticmethod
    def update_lane_invasion_sensor(actor_id: str, sensor: LaneInvasionSensor):
        """Updates a lane invasion sensor"""
        SensorDataProvider._lane_invasion_sensors[actor_id] = sensor

    @staticmethod
    def get_camera_data(actor_id: str) -> Dict[str, Tuple[int, np.ndarray]]:
        """Returns the camera data of the actor

        Returns:
            Dict: image data from sensor_interface.get_data(). E.g.

            data = {
                "sensor_id": (frame : int, processed_data : ndarray),
                ...
            }
        """
        data = SensorDataProvider._camera_data_dict.get(actor_id, None)
        if data is not None and "bev" in data:
            data = data.copy()
            spec = (
                SensorDataProvider._camera_config[actor_id]["x_res"],
                SensorDataProvider._camera_config[actor_id]["y_res"],
            )
            data["bev"] = (
                data["bev"][0],
                SensorDataProvider.get_birdeye_data(spec, data["bev"][1]),
            )

        return data

    @staticmethod
    def get_birdeye_data(spec: Tuple[int, int], actor: carla.Actor) -> np.ndarray:
        """Generate a birdeye data for the actor

        Args:
            spec (tuple): width, height
            actor (carla.Actor): actor

        Returns:
            ndarray: birdeye data
        """
        return BirdViewProducer.as_rgb(
            SensorDataProvider._birdeye_sensors[spec].produce(actor)
        )

    @staticmethod
    def get_collision_sensor(actor_id: str) -> CollisionSensor:
        """
        Returns:
            CollisionSensor: collision sensor of the actor
        """
        return SensorDataProvider._collision_sensors.get(actor_id, None)

    @staticmethod
    def get_lane_invasion_sensor(actor_id: str) -> LaneInvasionSensor:
        """
        Returns:
            LaneInvasionSensor: lane invasion sensor of the actor
        """
        return SensorDataProvider._lane_invasion_sensors.get(actor_id, None)

    @staticmethod
    def get_birdeye_sensor(spec: "tuple[int, int]") -> BirdViewProducer:
        """Get a birdeye sensor based on the spec (width, height)

        Args:
            spec (tuple): width, height

        Returns:
            BirdViewProducer: birdeye sensor of the actor
        """
        return SensorDataProvider._birdeye_sensors.get(spec, None)

    @staticmethod
    def get_all_data() -> Dict[str, dict]:
        """Returns all sensor data"""
        return {
            "camera": SensorDataProvider._camera_data_dict,
            "birdeye": SensorDataProvider._birdeye_sensors,
            "collision": SensorDataProvider._collision_sensors,
            "lane_invasion": SensorDataProvider._lane_invasion_sensors,
        }

    @staticmethod
    def cleanup(soft_reset: bool = False):
        """Cleanup the sensor data

        Args:
            soft_reset (bool, optional): If True, the sensors will not be destroyed. Defaults to False.
        """

        def destroy_sensor(sensor_wrapper):
            sensor: carla.Sensor = sensor_wrapper.sensor
            if sensor.is_alive:
                if sensor.is_listening:
                    sensor.stop()
                sensor.destroy()
            sensor_wrapper.sensor = None

        if soft_reset:
            for colli_sensor in SensorDataProvider._collision_sensors.values():
                colli_sensor.reset()
            for lane_sensor in SensorDataProvider._lane_invasion_sensors.values():
                lane_sensor.reset()
        else:
            for colli_sensor in SensorDataProvider._collision_sensors.values():
                destroy_sensor(colli_sensor)
            for lane_sensor in SensorDataProvider._lane_invasion_sensors.values():
                destroy_sensor(lane_sensor)

            SensorDataProvider._camera_config = {}
            SensorDataProvider._birdeye_sensors = {}
            SensorDataProvider._collision_sensors = {}
            SensorDataProvider._lane_invasion_sensors = {}

        SensorDataProvider._camera_data_dict = {}


class CallBack(object):
    """
    Class the sensors listen to in order to receive their data each frame
    """

    def __init__(
        self, tag: str, sensor: carla.Sensor, data_provider: "SensorInterface"
    ):
        """
        Initializes the call back
        """
        self._tag = tag
        self._sensor = sensor
        self._data_provider = data_provider

        self._data_provider.register_sensor(tag, sensor)

    def __call__(self, data):
        """
        call function
        """
        if self._tag == "bev":
            self._pseudo_bev_cb(data, self._tag)
        elif isinstance(data, carla.Image):
            self._parse_image_cb(data, self._tag)
        elif isinstance(data, carla.LidarMeasurement):
            self._parse_lidar_cb(data, self._tag)
        elif isinstance(data, carla.RadarMeasurement):
            self._parse_radar_cb(data, self._tag)
        elif isinstance(data, carla.GnssMeasurement):
            self._parse_gnss_cb(data, self._tag)
        elif isinstance(data, carla.IMUMeasurement):
            self._parse_imu_cb(data, self._tag)
        else:
            logging.error("No callback method for this sensor.")

    def _converter(self, tag: str) -> carla.ColorConverter:
        """
        Get the converter for the sensor according to tag
        """
        tag = tag.lower()

        if "depth" in tag:
            return carla.ColorConverter.LogarithmicDepth  # carla.ColorConverter.Depth
        elif "semseg" in tag:
            return carla.ColorConverter.CityScapesPalette
        else:
            return carla.ColorConverter.Raw

    # Parsing CARLA physical Sensors
    def _parse_image_cb(self, image: carla.Image, tag: str):
        """
        parses cameras
        """
        image.convert(self._converter(tag))
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = copy.deepcopy(array)
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self._data_provider.update_sensor(tag, array, image.frame)

    def _parse_lidar_cb(self, lidar_data: carla.LidarMeasurement, tag: str):
        """
        parses lidar sensors
        """
        points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype("f4"))
        points = copy.deepcopy(points)
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        self._data_provider.update_sensor(tag, points, lidar_data.frame)

    def _parse_radar_cb(self, radar_data: carla.RadarMeasurement, tag: str):
        """
        parses radar sensors
        """
        # [depth, azimuth, altitute, velocity]
        points = np.frombuffer(radar_data.raw_data, dtype=np.dtype("f4"))
        points = copy.deepcopy(points)
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        points = np.flip(points, 1)
        self._data_provider.update_sensor(tag, points, radar_data.frame)

    def _parse_gnss_cb(self, gnss_data: carla.GnssMeasurement, tag: str):
        """
        parses gnss sensors
        """
        array = np.array(
            [gnss_data.latitude, gnss_data.longitude, gnss_data.altitude],
            dtype=np.float64,
        )
        self._data_provider.update_sensor(tag, array, gnss_data.frame)

    def _parse_imu_cb(self, imu_data: carla.IMUMeasurement, tag: str):
        """
        parses IMU sensors
        """
        array = np.array(
            [
                imu_data.accelerometer.x,
                imu_data.accelerometer.y,
                imu_data.accelerometer.z,
                imu_data.gyroscope.x,
                imu_data.gyroscope.y,
                imu_data.gyroscope.z,
                imu_data.compass,
            ],
            dtype=np.float64,
        )
        self._data_provider.update_sensor(tag, array, imu_data.frame)

    def _pseudo_bev_cb(self, pseudo_data, tag: str):
        """
        pseudo callback for bev sensor to apply the same logic as other sensors
        """
        self._data_provider.update_sensor(tag, self._sensor.parent, pseudo_data.frame)


class SensorInterface(object):
    """
    Class that contains all sensor data for one agent
    """

    def __init__(self):
        """
        Initializes the class
        """
        self._sensors_objects = {}
        self._new_data_buffers = Queue()
        self._queue_timeout = 10

    def register_sensor(self, tag: str, sensor: carla.Sensor):
        """
        Registers the sensors
        """
        if tag in self._sensors_objects:
            raise ValueError(f"Duplicated sensor tag [{tag}]")

        self._sensors_objects[tag] = sensor

    def update_sensor(self, tag: str, data: np.ndarray, frame: int):
        """
        Updates the sensor
        """
        if tag not in self._sensors_objects:
            raise ValueError(f"The sensor with tag [{tag}] has not been created!")

        self._new_data_buffers.put((tag, frame, data))

    def get_data(self, timeout: int = None) -> Dict[str, Tuple[int, np.ndarray]]:
        """
        Returns the data of a sensor
        """
        try:
            data_dict = {}
            timeout = self._queue_timeout if timeout is None else timeout
            while len(data_dict.keys()) < len(self._sensors_objects.keys()):
                sensor_data = self._new_data_buffers.get(True, timeout)

                # data_dict["sensor_id"] = (frame, processed_data)
                data_dict[sensor_data[0]] = (sensor_data[1], sensor_data[2])

        except Empty:
            raise SensorReceivedNoData("A sensor took too long to send its data")

        return data_dict
