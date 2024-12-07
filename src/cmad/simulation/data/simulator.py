from __future__ import annotations

import atexit
import json
import logging
import math
import os
import random
import signal
import time
import weakref
from functools import lru_cache
from typing import TYPE_CHECKING, Optional, Tuple, Union

import carla
import redis

if TYPE_CHECKING:
    from cmad.envs.multi_env import MultiCarlaEnv

import cmad.simulation.data.local_carla_api as local_carla
from cmad.misc import (
    get_attributes,
    get_tcp_port,
    start_carla_server,
    stop_carla_server,
)
from cmad.simulation.data.carla_data_provider import CarlaDataProvider
from cmad.simulation.data.replay_parser import Replay
from cmad.simulation.data.sensor_interface import SensorDataProvider
from cmad.simulation.data.timer import GameTime

logger = logging.getLogger(__name__)


class Weather:
    """Weather presets for Simulator"""

    PRESETS: dict[int, carla.WeatherParameters] = {
        0: carla.WeatherParameters.ClearNoon,
        1: carla.WeatherParameters.CloudyNoon,
        2: carla.WeatherParameters.WetNoon,
        3: carla.WeatherParameters.WetCloudyNoon,
        4: carla.WeatherParameters.MidRainyNoon,
        5: carla.WeatherParameters.HardRainNoon,
        6: carla.WeatherParameters.SoftRainNoon,
        7: carla.WeatherParameters.ClearSunset,
        8: carla.WeatherParameters.CloudySunset,
        9: carla.WeatherParameters.WetSunset,
        10: carla.WeatherParameters.WetCloudySunset,
        11: carla.WeatherParameters.MidRainSunset,
        12: carla.WeatherParameters.HardRainSunset,
        13: carla.WeatherParameters.SoftRainSunset,
    }


class Simulator:
    """Simulator class for interacting with CARLA simulator

    This class will establish a connection with CARLA simulator and provide a set of APIs for
    interacting with the simulator. It also provides a set of APIs for interacting with the
    sensors attached to the ego vehicle.

    The connection could either via carla.Client or a BridgeServer. The former is used for
    connecting to a simulator running on the same machine. The latter is used for connecting
    to a simulator running on a remote machine.

    Note:
        There are two kinds of id used in this class:
        1. actor_id: the id which is speicified by user in the config file
        2. id: the id which is assigned by CARLA simulator
        You should judge by the name and the argument type to determine which id is used.
    """

    _client: carla.Client = None
    _redis: redis.Redis = None
    _process = None
    _clear_flag = False

    ready = False
    replay_tick = None
    data_provider = CarlaDataProvider
    sensor_provider = SensorDataProvider
    game_time = GameTime

    @staticmethod
    def setup(env: "MultiCarlaEnv"):
        Simulator.ready = False

        # handle termination (should only called once)
        Simulator._clear_flag = False
        weak_env = weakref.ref(env)

        def termination_cleanup(signum, frame):
            logger.warning("Received signal %d, terminating...", signum)
            env_ref = weak_env()
            if env_ref and not Simulator._clear_flag:
                env_ref.close()

        for sig in [signal.SIGTERM, signal.SIGABRT, signal.SIGINT]:
            signal.signal(sig, termination_cleanup)
        atexit.register(termination_cleanup, 0, None)

        Simulator.init_server(env)
        Simulator.ready = True

    @staticmethod
    def init_server(env: "MultiCarlaEnv"):
        """Create the server based on MultiCarlaEnv's config

        Args:
            env (MultiCarlaEnv): MultiCarlaEnv instance
        """

        # Create server if not already specified
        if env._server_port is None:
            env._server_port = get_tcp_port()
            Simulator._process = start_carla_server(
                server_port=env._server_port,
                render=env._render_simulator,
                x_res=env.env_config.get("render_x_res", 800),
                y_res=env.env_config.get("render_y_res", 600),
            )

        # Establish connection with redis
        if env.env_config.get("use_redis", False):
            redis_host = env.env_config.get("redis_host", "localhost")
            redis_port = env.env_config.get("redis_port", 6379)
            redis_pass = env.env_config.get("redis_pass", None)
            redis_db = env.env_config.get("redis_db", 0)
            Simulator._redis = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                password=redis_pass,
                decode_responses=True,
            )
            Simulator._redis.set("reset", "no")
            Simulator._redis.set("START_EGO", "0")

            env._sync_server = False
        else:
            Simulator._redis = None

        # Start client
        retry_cnt = 0
        while Simulator._client is None:
            try:
                Simulator._client = carla.Client(env._server_ip, env._server_port)
                # The socket establishment could takes some time
                time.sleep(1)
                Simulator._client.set_timeout(2.0)
                logger.info(
                    "Client successfully connected to server, Carla-Server version: %s",
                    Simulator._client.get_server_version(),
                )
            except RuntimeError as re:
                error_msg = str(re)
                if "timeout" not in error_msg and "time-out" not in error_msg:
                    logger.error(
                        "Could not connect to Carla server because: %s", error_msg
                    )

                if retry_cnt > 5:
                    if Simulator._redis:
                        logger.info("Sending UE_START signal to redis")
                        Simulator._redis.set("UE_START", "1")
                        while Simulator._redis.get("UE_START") == "1":
                            time.sleep(1)
                    elif env._server_ip in ["127.0.0.1", "localhost"]:
                        logger.info("Starting Carla server locally")
                        if Simulator._process is not None:
                            stop_carla_server(Simulator._process)
                            Simulator._process = None

                        env._server_port = get_tcp_port()
                        Simulator._process = start_carla_server(
                            server_port=env._server_port, render=env._render_simulator
                        )
                    # Allow more time for UE to fully start
                    retry_cnt = -5

                retry_cnt += 1
                Simulator._client = None
                logger.warning("Cannot connect to server, retrying: %d", retry_cnt)

        Simulator._client.set_timeout(20.0)
        world = Simulator._client.get_world()
        if env._reload or env._map_name not in world.get_map().name:
            world = Simulator._client.load_world(env._server_map)
            time.sleep(2)
            env._reload = False
            logger.info("Loaded new map: %s", env._map_name)

        world_settings = world.get_settings()
        world_settings.synchronous_mode = env._sync_server
        world_settings.fixed_delta_seconds = env._fixed_delta_seconds
        world_settings.no_rendering_mode = not env._render_simulator
        world.apply_settings(world_settings)

        # Set up traffic manager
        tm_port = get_tcp_port()
        traffic_manager = Simulator._client.get_trafficmanager(tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        traffic_manager.set_respawn_dormant_vehicles(True)
        traffic_manager.set_synchronous_mode(env._sync_server)

        # Prepare data provider
        Simulator.data_provider.set_client(Simulator._client)
        Simulator.data_provider.set_world(world)
        Simulator.data_provider.set_traffic_manager_port(tm_port)

        # Disable traffic light if specified
        if not env._enable_traffic_light:
            world.freeze_all_traffic_lights(True)
            for tl in Simulator.data_provider._traffic_light_map:
                tl.set_state(carla.TrafficLightState.Off)
        else:
            world.freeze_all_traffic_lights(False)
            world.reset_all_traffic_lights()

        # Set the spectator/server view
        if env.env_config.get("spectator_loc"):
            spectator = world.get_spectator()
            spectator_loc = env.env_config["spectator_loc"]

            if len(spectator_loc) > 3:
                spectator_rot = carla.Rotation(*spectator_loc[3:])
                spectator_loc = carla.Location(*spectator_loc[:3])
            else:
                d = 6.4
                angle = 160  # degrees
                a = math.radians(angle)
                spectator_rot = carla.Rotation(yaw=180 + angle, pitch=-15)
                spectator_loc = carla.Location(
                    d * math.cos(a), d * math.sin(a), 2.0
                ) + carla.Location(*spectator_loc)

            spectator.set_transform(carla.Transform(spectator_loc, spectator_rot))

    @staticmethod
    def get_world():
        """Get the world.

        Returns:
            carla.World: The world.
        """
        return Simulator.data_provider.get_world()

    @staticmethod
    def get_map():
        """Get the map.

        Returns:
            carla.Map: The map.
        """
        return Simulator.data_provider.get_map()

    @staticmethod
    def is_sync_mode():
        """Get the synchronous mode.

        Returns:
            bool: The synchronous mode.
        """
        return Simulator.data_provider.is_sync_mode()

    @staticmethod
    def get_traffic_manager(port=None):
        """Get a traffic manager.
        This function will try to find an existing TM on the given port.
        If no port is given, it will use the current port in the env.

        Returns:
            carla.TrafficManager: The traffic manager.
        """
        if port is None:
            port = Simulator.get_traffic_manager_port()

        return Simulator._client.get_trafficmanager(port)

    @staticmethod
    def get_traffic_manager_port():
        """Get the traffic manager port.

        Returns:
            int: The traffic manager port.
        """
        return Simulator.data_provider.get_traffic_manager_port()

    @staticmethod
    def get_actor_by_id(id: int, from_world: bool = False) -> carla.Actor:
        """Get an actor by id.

        Args:
            id (int): Actor id.
            from_world (bool): If True, get the actor directly from the Simulator. Otherwise, get it from the registed dictionary.

        Returns:
            carla.Actor: The actor.
        """
        return (
            Simulator.get_world().get_actor(id)
            if from_world
            else Simulator.data_provider.get_actor_by_id(id)
        )

    @staticmethod
    def get_actor_by_rolename(rolename: str, from_world: bool = True) -> carla.Actor:
        """Get an actor by rolename. This is mainly used for Ego Vehicles.

        Args:
            rolename (str): Actor rolename. None if not found.
            from_world (bool): If True, get the actor directly from the world. Otherwise, get it from the registed dictionary.

        Returns:
            carla.Actor: Actor with the rolename specified. None if not found
        """
        actors = (
            Simulator.get_world().get_actors()
            if from_world
            else Simulator.data_provider.get_actors(actor_only=True)
        )

        for actor in actors:
            if (
                "role_name" in actor.attributes
                and actor.attributes["role_name"] == rolename
            ):
                return actor

        return None

    @staticmethod
    def get_actor_control(id: int, to_dict: bool = True):
        """Get an actor's last control.

        Args:
            id (int): Actor id.
            to_dict (bool): If True, convert the control to a dictionary.

        Returns:
            carla.VehicleControl | carla.WalkerControl | dict: The actor's control.
        """
        actor = Simulator.get_actor_by_id(id)
        control = actor.get_control()
        if to_dict:
            control = get_attributes(control)

        return control

    @staticmethod
    def get_actor_location(id: int, use_local_api: bool = False) -> carla.Location:
        """Get an actor's location.

        Args:
            id (int): Actor id.
            use_local_api (bool): If True, return a Location object.

        Returns:
            carla.Location: The actor's location.
        """
        carla_loc = Simulator.data_provider.get_location_by_id(id)
        if carla_loc is not None and use_local_api:
            return local_carla.Location.from_simulator_location(carla_loc)
        return carla_loc

    @staticmethod
    def get_actor_velocity(id: int, use_local_api: bool = False) -> carla.Vector3D:
        """Get an actor's velocity.

        Args:
            id (int): Actor id.
            use_local_api (bool): If True, return a Vector3D object when using use_vector.

        Returns:
            float | carla.Vector3D: Vector3D object representing velocity or speed in m/s.
        """
        velocity = Simulator.data_provider.get_velocity_by_id(id, use_vector=True)

        if velocity is not None and use_local_api:
            velocity = local_carla.Vector3D.from_simulator_vector(velocity)
        return velocity

    @staticmethod
    def get_actor_acceleration(id: int, use_local_api: bool = False) -> carla.Vector3D:
        """Get an actor's acceleration.

        Args:
            id (int): Actor id.
            use_local_api (bool): If True, return a Vector3D object.

        Returns:
            carla.Vector3D: The actor's acceleration in m/s^2.
        """
        acceleration = Simulator.data_provider.get_acceleration_by_id(id)
        if acceleration is not None and use_local_api:
            acceleration = local_carla.Vector3D.from_simulator_vector(acceleration)
        return acceleration

    @staticmethod
    def get_actor_transform(id: int, use_local_api: bool = False) -> carla.Transform:
        """Get an actor's transform.

        Args:
            id (int): Actor id.
            use_local_api (bool): If True, return a Transform object.

        Returns:
            carla.Transform: The actor's transform.
        """
        carla_transform = Simulator.data_provider.get_transform_by_id(id)
        if carla_transform is not None and use_local_api:
            return local_carla.Transform.from_simulator_transform(carla_transform)
        return carla_transform

    @staticmethod
    def get_actor_forward(id: int, use_local_api: bool = False) -> carla.Vector3D:
        """Get an actor's forward vector.

        Args:
            id (int): Actor id.
            use_local_api (bool): If True, return a Vector3D object.

        Returns:
            carla.Vector3D: The actor's forward vector. (global reference, unit vector)
        """
        carla_transform = Simulator.data_provider.get_transform_by_id(id)
        forward_vector = carla_transform.get_forward_vector()

        if forward_vector is not None and use_local_api:
            forward_vector = local_carla.Vector3D.from_simulator_vector(forward_vector)
        return forward_vector

    @staticmethod
    @lru_cache
    def get_actor_bounding_box(
        id: int, use_local_api: bool = False
    ) -> carla.BoundingBox:
        """Get an actor's bounding box.

        Args:
            id (int): Actor id.
            use_local_api (bool): If True, return a BoundingBox object.

        Returns:
            carla.BoundingBox | BoundingBox: The actor's bounding box.
        """
        actor = Simulator.get_actor_by_id(id)
        carla_bb = getattr(actor, "bounding_box", None)

        if carla_bb is not None and use_local_api:
            carla_bb = local_carla.BoundingBox.from_simulator_bounding_box(carla_bb)
        return carla_bb

    @staticmethod
    def get_actor_waypoint(id: int, use_local_api: bool = False) -> carla.Waypoint:
        """Get an actor's waypoint, projected on the road.

        Args:
            id (int): Actor id.
            use_local_api (bool): If True, return a Waypoint object.

        Returns:
            carla.Waypoint: The actor's waypoint.
        """
        actor = Simulator.get_actor_by_id(id)
        lane_type = (
            carla.LaneType.Driving
            if isinstance(actor, carla.Vehicle)
            else carla.LaneType.Any
        )
        wpt = Simulator.data_provider.get_map().get_waypoint(
            actor.get_transform().location, project_to_road=True, lane_type=lane_type
        )

        if wpt is not None and use_local_api:
            wpt = local_carla.Waypoint.from_simulator_waypoint(wpt)
        return wpt

    @staticmethod
    def get_actor_camera_data(actor_id: str):
        """Get an actor's camera data.

        Args:
            actor_id (str): Actor id.

        Returns:
            Dict: image data from sensor_interface.get_data(). E.g.

            data = {
                "sensor_id": (frame : int, processed_data : ndarray),
                ...
            }
        """
        return Simulator.sensor_provider.get_camera_data(actor_id)

    @staticmethod
    def get_actor_collision_sensor(actor_id: str):
        """Get an actor's collision sensor.

        Args:
            actor_id (str): Actor id.

        Returns:
            CollisionSensor: The collision sensor.
        """
        coll_sensor = Simulator.sensor_provider.get_collision_sensor(actor_id)
        return coll_sensor

    @staticmethod
    def get_actor_lane_invasion_sensor(actor_id: str):
        """Get an actor's lane invasion sensor.

        Args:
            actor_id (str): Actor id.

        Returns:
            LaneInvasionSensor: The lane invasion sensor.
        """
        lane_sensor = Simulator.sensor_provider.get_lane_invasion_sensor(actor_id)
        return lane_sensor

    @staticmethod
    def set_weather(
        index: Union[int, list], extra_spec: dict = {}
    ) -> local_carla.WeatherParameters:
        """Set the weather.

        Args:
            index (int | list): The index of the weather.
            extra_spec (dict): Extra weather specs.

        Returns:
            weather specs (WeatherParamerters)
        """
        if isinstance(index, (list, tuple)):
            index = random.choice(index)

        if index == -1:
            weather = Simulator.get_world().get_weather()
        else:
            try:
                weather = Weather.PRESETS[index]
            except KeyError as e:
                logger.warning("Weather preset %s not found, using default 0", e)
                weather = Weather.PRESETS[0]

            for key in extra_spec:
                if hasattr(weather, key):
                    setattr(weather, key, extra_spec[key])

            Simulator.get_world().set_weather(weather)

        return local_carla.WeatherParameters.from_simulator_weather_parameters(weather)

    @staticmethod
    def set_actor_speed(id: int, speed: float, ret_command: bool = False):
        """Set the target speed of an actor.

        This function can be used to set a initial speed for an actor. Note that, this speed will be applied
        before the physics simulation starts. Therefore, the actor final speed may be different from the target speed.

        Args:
            id (int): Actor id.
            speed (float): The target speed in m/s.
            ret_command (bool): If True, return the command instead of applying it.
        """
        actor = Simulator.get_actor_by_id(id)
        if speed == 0:
            target_speed = carla.Vector3D(0, 0, 0)
        else:
            yaw = actor.get_transform().rotation.yaw * (math.pi / 180)
            vx = speed * math.cos(yaw)
            vy = speed * math.sin(yaw)
            target_speed = carla.Vector3D(vx, vy, 0)

        if not ret_command and hasattr(actor, "set_target_velocity"):
            actor.set_target_velocity(target_speed)
            return None
        else:
            return carla.command.ApplyTargetVelocity(id, target_speed)

    @staticmethod
    def set_actor_angular_speed(id: int, speed: float, ret_command: bool = False):
        """Set the target angular speed of an actor.

        Args:
            id (int): Actor id.
            speed (float): The target speed in m/s.
            ret_command (bool): If True, return the command instead of applying it.
        """
        actor = Simulator.get_actor_by_id(id)
        if speed == 0:
            target_speed = carla.Vector3D(0, 0, 0)
        else:
            yaw = actor.get_transform().rotation.yaw * (math.pi / 180)
            vx = speed * math.cos(yaw)
            vy = speed * math.sin(yaw)
            target_speed = carla.Vector3D(vx, vy, 0)

        if not ret_command and hasattr(actor, "set_target_angular_velocity"):
            actor.set_target_angular_velocity(target_speed)
            return None
        else:
            return carla.command.ApplyTargetAngularVelocity(id, target_speed)

    @staticmethod
    def request_new_actor(
        model: str,
        spawn_point: carla.Transform,
        attach_to: Optional[carla.Actor] = None,
        rolename: str = "scenario",
        autopilot: bool = False,
        random_location: bool = False,
        color: Optional[carla.Color] = None,
        actor_category: str = "car",
        safe_blueprint: bool = False,
        blueprint: Optional[carla.ActorBlueprint] = None,
        immortal: bool = True,
        tick: bool = True,
    ) -> carla.Actor:
        """Request a new actor.

        Args:
            model (str): The model name.
            spawn_point (carla.Transform): The spawn point.
            attach_to (carla.Actor): The actor to attach to. (Sensor only)
            rolename (str): The actor's rolename.
            autopilot (bool): Whether to enable autopilot.
            random_location (bool): Whether to spawn the actor at a random location.
            color (carla.Color): The actor's color.
            actor_category (str): The actor's category.
            safe_blueprint (bool): Whether to use the safe blueprint.
            blueprint (carla.ActorBlueprint): The blueprint to use.
            immortal (bool): Whether to make the actor immortal. (Walker only)
            tick (bool): Whether to tick the world after creation.

        Returns:
            carla.Actor: The actor.
        """
        actor = Simulator.data_provider.request_new_actor(
            model,
            spawn_point,
            rolename,
            autopilot,
            random_location,
            color,
            actor_category,
            safe_blueprint,
            blueprint,
            attach_to,
            immortal,
            tick,
        )
        return actor

    @staticmethod
    def register_actor(actor: carla.Actor, add_to_pool: bool = False):
        """Register an actor.

        Args:
            actor (carla.Actor): The actor.
            add_to_pool (bool): If True, add the actor to the actor pool. By doing this, we will handle the actor's destruction.
        """
        if add_to_pool:
            Simulator.data_provider._carla_actor_pool[actor.id] = actor
        Simulator.data_provider.register_actor(actor)

    @staticmethod
    def register_collision_sensor(actor_id: str, actor: carla.Actor):
        """Register a collision sensor.

        Args:
            actor_id (str): The actor id.
            actor (carla.Actor): The actor which the sensor is attached to.
        """
        from cmad.simulation.sensors.derived_sensors import CollisionSensor

        Simulator.sensor_provider.update_collision_sensor(
            actor_id, CollisionSensor(actor)
        )

    @staticmethod
    def register_lane_invasion_sensor(actor_id: str, actor: carla.Actor):
        """Register a lane invasion sensor.

        Args:
            actor_id (str): The actor id.
            actor (carla.Actor): The actor which the sensor is attached to.
        """
        from cmad.simulation.sensors.derived_sensors import LaneInvasionSensor

        Simulator.sensor_provider.update_lane_invasion_sensor(
            actor_id, LaneInvasionSensor(actor)
        )

    @staticmethod
    def register_birdeye_sensor(spec: Tuple[int, int]):
        """Register a birdeye sensor.

        Args:
            actor_id (str): The actor id.
            spec (Tuple[int, int]): The sensor spec (width, height).
        """
        from cmad.viz.carla_birdeye_view import (
            BirdViewCropType,
            BirdViewProducer,
            PixelDimensions,
        )

        if Simulator.sensor_provider.get_birdeye_sensor(spec) is not None:
            return

        producer = BirdViewProducer(
            target_size=PixelDimensions(width=spec[0], height=spec[1]),
            render_lanes_on_junctions=False,
            pixels_per_meter=4,
            crop_type=BirdViewCropType.FRONT_AND_REAR_AREA,
        )
        Simulator.sensor_provider.update_birdeye_sensor(spec, producer)

    @staticmethod
    def set_actor_transform(
        id: int, transform: carla.Transform, ret_command: bool = False
    ):
        """Set an actor's transform.

        Args:
            id (int): Actor id.
            transform (carla.Transform): The transform to set.
            ret_command (bool): If True, return the command instead of applying it.
        """
        if not ret_command:
            actor = Simulator.get_actor_by_id(id)
            actor.set_transform(transform)
            return None
        else:
            return carla.command.ApplyTransform(id, transform)

    @staticmethod
    def apply_actor_control(id: int, control=None, ret_command: bool = False):
        """Apply control to an actor.

        Args:
            id (int): Actor id.
            control (carla.VehicleControl | str): The control to apply.
            ret_command (bool): If True, return the command instead of applying it.
        """
        actor = Simulator.get_actor_by_id(id)
        if not actor or not actor.is_active or not hasattr(actor, "apply_control"):
            return None

        if control is None:
            control = actor.get_control()
        elif control == "reset":
            control = type(actor.get_control())()
        elif control == "stop":
            if isinstance(actor, carla.Vehicle):
                control = carla.VehicleControl(
                    throttle=0.0, steer=0.0, brake=1.0, hand_brake=True
                )
            elif isinstance(actor, carla.Walker):
                control = carla.WalkerControl(
                    direction=carla.Vector3D(0, 0, 0), speed=0.0
                )

        if ret_command:
            if isinstance(actor, carla.Vehicle):
                return carla.command.ApplyVehicleControl(id, control)
            elif isinstance(actor, carla.Walker):
                return carla.command.ApplyWalkerControl(id, control)
        else:
            actor.apply_control(control)
            return None

    @staticmethod
    def toggle_actor_physic(id: int, physic_on: bool, ret_command: bool = False):
        """Toggle an actor's physic.

        Args:
            id (int): Actor id.
            physic (bool): Whether to enable physic.
            ret_command (bool): If True, return the command instead of applying it.
        """
        if not ret_command:
            actor = Simulator.get_actor_by_id(id)
            actor.set_simulate_physics(physic_on)
            return None
        else:
            return carla.command.SetSimulatePhysics(id, physic_on)

    @staticmethod
    def toggle_actor_autopilot(id: int, autopilot: bool):
        """Toggle an actor's autopilot.

        Args:
            id (int): Actor id.
            autopilot (bool): Whether to enable autopilot.
        """
        actor = Simulator.get_actor_by_id(id)
        if hasattr(actor, "set_autopilot"):
            actor.set_autopilot(autopilot, Simulator.get_traffic_manager_port())
        else:
            logger.warning("Trying to toggle autopilot on a non-vehicle actor")

    @staticmethod
    def toggle_world_settings(
        synchronous_mode=False,
        fixed_delta_seconds=0.05,
        no_rendering_mode=False,
        substepping=True,
        max_substep_delta_time=0.01,
        max_substeps=10,
        max_culling_distance=0.0,
        deterministic_ragdolls=True,
        **kwargs,
    ):
        """Toggle world settings

        Args:
            Please refer to: https://carla.readthedocs.io/en/0.9.13/python_api/#carla.WorldSettings
        """
        Simulator.toggle_sync_mode(synchronous_mode, fixed_delta_seconds)

        world = Simulator.get_world()
        settings = world.get_settings()
        settings.no_rendering_mode = no_rendering_mode
        settings.substepping = substepping
        settings.max_substep_delta_time = max_substep_delta_time
        settings.max_substeps = max_substeps
        settings.max_culling_distance = max_culling_distance
        settings.deterministic_ragdolls = deterministic_ragdolls
        if kwargs:
            for key, value in kwargs.items():
                setattr(settings, key, value)
        world.apply_settings(settings)

    @staticmethod
    def apply_traffic(
        num_vehicles: int,
        num_pedestrians: int,
        percentagePedestriansRunning: float = 0.0,
        percentagePedestriansCrossing: float = 0.0,
        ref_position: carla.Location = None,
        spawn_range: float = float("inf"),
        safe: bool = False,
    ):
        """Generate traffic.

        Args:
            num_vehicles (int): Number of vehicles to spawn.
            num_pedestrians (int): Number of pedestrians to spawn.
            percentagePedestriansRunning (float): Percentage of pedestrians running.
            percentagePedestriansCrossing (float): Percentage of pedestrians crossing.
            ref_position (carla.Location): Pivot position to spawn traffic flow next to it.
            spawn_range (float): Distance relative to the ref_position, default to no limit.
            safe (bool): Whether to spawn vehicles in safe mode.

        Returns:
            list: List of spawned vehicles.
            tuple: Tuple of spawned pedestrians and their controllers.
        """
        # --------------
        # Spawn vehicles
        # --------------
        world = Simulator.get_world()
        traffic_manager = Simulator.get_traffic_manager()

        spawn_points = Simulator.data_provider._spawn_points
        ref_position = carla.Location(*ref_position) if ref_position else None

        # Filter spawn points based on reference position and range if provided
        if ref_position:
            filtered_spawn_points = []
            for spawn_point in spawn_points:
                distance = spawn_point.location.distance(ref_position)
                if distance <= spawn_range:
                    filtered_spawn_points.append(spawn_point)
            spawn_points = filtered_spawn_points

        number_of_spawn_points = len(spawn_points)
        if number_of_spawn_points == 0:
            logger.warning("No spawn points found within the specified range")
            return [], ([], [])

        random.shuffle(spawn_points)
        if num_vehicles <= number_of_spawn_points:
            spawn_points = random.sample(spawn_points, num_vehicles)
        else:
            logger.warning(
                "requested %d vehicles, but could only find %d spawn points",
                num_vehicles,
                number_of_spawn_points,
            )
            num_vehicles = number_of_spawn_points

        vehicles_list = []
        failed_v = 0
        for n, transform in enumerate(spawn_points):
            vehicle = Simulator.data_provider.request_new_actor(
                "vehicle",
                transform,
                rolename="autopilot",
                autopilot=True,
                safe_blueprint=safe,
                tick=False,
            )
            if vehicle is not None:
                vehicles_list.append(vehicle)
            else:
                failed_v += 1

        logger.info(
            "%d/%d vehicles correctly spawned.", num_vehicles - failed_v, num_vehicles
        )

        # -------------
        # Spawn Walkers
        # -------------
        blueprints = Simulator.data_provider._blueprint_library.filter(
            "walker.pedestrian.*"
        )
        pedestrian_controller_bp = Simulator.data_provider._blueprint_library.find(
            "controller.ai.walker"
        )

        # Take all the random locations to spawn
        spawn_points = []
        max_attempts = num_pedestrians * 20
        attempts = 0

        while len(spawn_points) < num_pedestrians and attempts < max_attempts:
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()
            attempts += 1

            if loc is not None:
                # Check if the spawn point is within range if reference position is provided
                if ref_position:
                    distance = loc.distance(ref_position)
                    if distance <= spawn_range:
                        spawn_point.location = loc
                        spawn_points.append(spawn_point)
                else:
                    spawn_point.location = loc
                    spawn_points.append(spawn_point)

        if len(spawn_points) < num_pedestrians:
            logger.warning(
                "Could only find %d valid pedestrian spawn points within range out of %d requested",
                len(spawn_points),
                num_pedestrians,
            )
            num_pedestrians = len(spawn_points)

        # Spawn the walker object
        pedestrians_list = []
        controllers_list = []
        pedestrians_speed = []
        failed_p = 0
        for spawn_point in spawn_points:
            pedestrian_bp = random.choice(blueprints)
            if pedestrian_bp.has_attribute("is_invincible"):
                pedestrian_bp.set_attribute("is_invincible", "false")
            if pedestrian_bp.has_attribute("speed"):
                if random.random() > percentagePedestriansRunning:
                    speed = pedestrian_bp.get_attribute("speed").recommended_values[1]
                else:
                    speed = pedestrian_bp.get_attribute("speed").recommended_values[2]
            else:
                speed = 0.0
            pedestrian = Simulator.data_provider.request_new_actor(
                "walker.pedestrian",
                spawn_point,
                actor_category="pedestrian",
                blueprint=pedestrian_bp,
                tick=False,
            )
            if pedestrian is not None:
                controller = Simulator.data_provider.request_new_actor(
                    "controller.ai.walker",
                    carla.Transform(),
                    attach_to=pedestrian,
                    blueprint=pedestrian_controller_bp,
                    tick=False,
                )
                if controller is not None:
                    pedestrians_list.append(pedestrian)
                    controllers_list.append(controller)
                    pedestrians_speed.append(speed)
                else:
                    Simulator.data_provider.remove_actor_by_id(pedestrian.id)
                    failed_p += 1
            else:
                failed_p += 1

        logger.info(
            "%d/%d pedestrians correctly spawned.",
            num_pedestrians - failed_p,
            num_pedestrians,
        )

        # Initialize each controller and set target to walk
        Simulator.tick()
        world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i, controller in enumerate(controllers_list):
            controller.start()
            controller.go_to_location(world.get_random_location_from_navigation())
            controller.set_max_speed(float(pedestrians_speed[int(i / 2)]))  # max speed

        traffic_manager.global_percentage_speed_difference(30.0)

        return vehicles_list, (pedestrians_list, controllers_list)

    def tick():
        """Tick the simulator.

        Returns:
            int: The current frame number.
        """
        world = Simulator.get_world()
        if Simulator.data_provider.is_sync_mode():
            frame = world.tick()
        else:
            world.wait_for_tick()

        snapshot = world.get_snapshot()
        timestamp = snapshot.timestamp
        frame = timestamp.frame

        if Simulator.replay_tick is not None:
            Simulator.replay_tick += 1

        Simulator.data_provider.on_carla_tick()
        Simulator.game_time.on_carla_tick(timestamp)
        return frame

    @staticmethod
    def teleport_actor(
        id: int, transform: Union[carla.Transform, tuple], ret_command: bool = False
    ):
        """Teleport an actor to given transform, guaranteeing that the actor will hold still after teleporting.

        Args:
            id (int): Actor id.
            transform (carla.Transform | tuple): The transform to teleport to, either a carla.Transform object or a tuple of (x, y, z).
            ret_command (bool): If True, return the command instead of applying it.
        """
        if not isinstance(transform, carla.Transform):
            transform = Simulator.generate_spawn_point(Simulator.get_map(), transform)

        batch = []
        batch.append(Simulator.toggle_actor_physic(id, False, ret_command))
        batch.append(Simulator.apply_actor_control(id, "stop", ret_command))
        batch.append(Simulator.set_actor_transform(id, transform, ret_command))
        batch.append(Simulator.apply_actor_control(id, "reset", ret_command))
        batch.append(Simulator.toggle_actor_physic(id, True, ret_command))
        return batch

    @staticmethod
    def send_batch(batch_of_command: list, tick: int = 0):
        """Send a batch of commands to the simulator.

        This will ensure that all command are executed in the same tick.

        Args:
            batch_of_command (list): A list of carla.command objects.
            tick (int): Tick how many times after sending the batch. Default to 0.

        Returns:
            list[command.Response]: A list of carla.command.Response objects.
        """
        batch = list(filter(lambda x: x is not None, batch_of_command))
        if len(batch) > 0:
            res = Simulator._client.apply_batch_sync(batch)
        else:
            res = []

        while tick:
            # We didn't use the `due_tick_cue` directly in `apply_batch_sync`
            # Because it only applies to synchronous mode
            Simulator.tick()
            tick -= 1

        return res

    @staticmethod
    def toggle_sync_mode(is_sync: bool, fixed_delta_seconds: float = 0.05):
        """Toggle the simulator sync mode.

        Args:
            is_sync (bool): Whether to enable sync mode.
            fixed_delta_seconds (float): The fixed delta seconds to use in sync mode.
        """
        world = Simulator.get_world()
        traffic_manager = Simulator.get_traffic_manager()

        world_settings = world.get_settings()
        world_settings.synchronous_mode = is_sync
        world_settings.fixed_delta_seconds = fixed_delta_seconds
        world.apply_settings(world_settings)
        traffic_manager.set_synchronous_mode(is_sync)

        Simulator.data_provider._sync_flag = is_sync

    @staticmethod
    def add_callback(func):
        """Add a callback to the simulator.

        Args:
            func (callable): A function to be called on every tick.

        Returns:
            id (int) : The id of the callback.

        Example:
            >>> simulator.add_callback(lambda snapshot: print(snapshot.timestamp))
        """
        return Simulator.get_world().on_tick(func)

    @staticmethod
    def remove_callback(callback_id: int):
        """Remove a callback from the simulator.

        Args:
            callback_id (int): The id of the callback.
        """
        Simulator.get_world().remove_on_tick(callback_id)

    @staticmethod
    def clean_ego_attachment(ego_id: int, ignores: "list[int]" = None):
        """Clean up the ego vehicle's attachment.

        Args:
            ego_id (int): The id of the ego vehicle.
            ignores (list[int]): Attachment that is meant to be kept.

        Returns:
            list[int]: A list of the id of the removed actors.
        """
        colli_sensor = Simulator.sensor_provider.get_collision_sensor("ego")
        colli_id = -1
        if colli_sensor is not None:
            colli_id = colli_sensor.sensor.id

        lane_sensor = Simulator.sensor_provider.get_lane_invasion_sensor("ego")
        lane_id = -1
        if lane_sensor is not None:
            lane_id = lane_sensor.sensor.id

        if ignores is None:
            ignores = [colli_id, lane_id]
        else:
            ignores = ignores + [colli_id, lane_id]

        return Simulator.data_provider.remove_actor_attachment(ego_id, ignores)

    @staticmethod
    def cleanup(soft_reset: bool = False, completely: bool = False):
        """Clean up the simulator.

        Args:
            soft_reset (bool): Whether to destroy actors or not.
        """
        Simulator.sensor_provider.cleanup(soft_reset)
        Simulator.data_provider.cleanup(soft_reset, completely)
        Simulator.game_time.restart()

    @staticmethod
    def clear_server_state():
        """Clear server process"""
        if Simulator._clear_flag:
            return

        logger.info("Clearing Carla server state")
        try:
            if Simulator._client:
                Simulator._client.set_timeout(2.0)
                Simulator.toggle_sync_mode(False, None)
                Simulator.cleanup(soft_reset=False, completely=True)
                Simulator._client = None
        except Exception as e:
            logger.warning("Error disconnecting client: %s", e)

        if Simulator._process:
            stop_carla_server(Simulator._process)
            Simulator._process = None

        if Simulator._redis:
            Simulator._redis.close()
            Simulator._redis = None

        Simulator._clear_flag = True
        Simulator.ready = False

    @staticmethod
    def start_recorder(filename: str, all_info: bool = False):
        """Start the recorder

        Args:
            filename (str): The path to the file.
            all_info (bool): Whether to record all info.
        """
        if Simulator._client is not None:
            Simulator._replay_filepath = filename
            Simulator._client.start_recorder(filename, all_info)

    @staticmethod
    def stop_recorder(keep_file: bool = True, add_prefix: str = None):
        """Stop the recorder

        Args:
            keep_file (bool): Whether to keep the recorded file.
            add_prefix (str): The prefix to add to the file name. Typically used to add the cause of done.
        """
        if Simulator._client is not None:
            Simulator._client.stop_recorder()
            time.sleep(0.05)

        if Simulator._replay_filepath is not None:
            if not keep_file:
                os.remove(Simulator._replay_filepath)
            elif add_prefix is not None:
                os.rename(
                    Simulator._replay_filepath,
                    os.path.join(
                        os.path.dirname(Simulator._replay_filepath),
                        add_prefix + "_" + os.path.basename(Simulator._replay_filepath),
                    ),
                )

        Simulator._replay_filepath = None

    @staticmethod
    def replay_file(filepath: str, follow_vehicle: str = None):
        """Start to replay a file in sync mode, tick till vehicle is spawned.

        Args:
            filepath (str): The path to the file.
            follow_vehicle (str): The role_name of the vehicle to follow. If None, set to free camera.

        Returns:
            replay (Replay): The replay parser.
            carla_actors (list[carla.Actor]): The list of carla actors.
            start_time (float): The start time of the replay.
        """
        # fmt: off
        if Simulator._client is not None:
            replay = Replay(Simulator._client, filepath, lazy_init=False)

            # Replay
            if not Simulator.data_provider.get_map().name.endswith(replay.map_name):
                Simulator._client.load_world(replay.map_name)

            Simulator.toggle_sync_mode(True)

            # Get the id of the vehicle to follow
            follow_vehicle_id = 0
            if follow_vehicle is not None:
                target = replay.get_actor_by_rolename(follow_vehicle)
                if target is not None:
                    follow_vehicle_id = target.id
                elif follow_vehicle == "ego":
                    hero = replay.get_actor_by_rolename("hero")
                    follow_vehicle_id = hero.id if hero is not None else 0

            # Get the valid start time (Ego moved)
            Simulator.replay_tick = 1
            hero_actor = replay.get_actor_by_rolename("hero") or replay.get_actor_by_rolename("ego")
            if hero_actor is not None:
                hero_id = hero_actor.id
                second_frame = replay.get_frame(2)
                for frame_id in range(3, len(replay.frame_info)):
                    frame = replay.get_frame(frame_id)
                    if (frame.transform[hero_id].location - second_frame.transform[hero_id].location).length() > 0.2:
                        Simulator.replay_tick = frame_id
                        break

            start_time = replay.frame_info[Simulator.replay_tick - 1].frame_time
            Simulator._client.replay_file(filepath, start_time, 0, follow_vehicle_id)

            # Reset DataProvider
            tm_port = Simulator.data_provider.get_traffic_manager_port()
            Simulator.data_provider.reset(completely=True)
            Simulator.data_provider.set_client(Simulator._client)
            Simulator.data_provider.set_world(Simulator._client.get_world())
            Simulator.data_provider.set_traffic_manager_port(tm_port)

            # Tick till vehicle is spawned
            while True:
                Simulator.tick()
                carla_actors = [
                    Simulator.get_actor_by_id(actor.id, from_world=True)
                    for actor in replay.actors
                ]
                if all(actor is not None for actor in carla_actors):
                    break

            return replay, carla_actors, start_time
        # fmt: on
        raise Exception("Client is not connected")

    @staticmethod
    @lru_cache
    def generate_spawn_point(
        map: carla.Map,
        pos: Tuple[float, float, float],
        rot: Optional[Tuple[float, float, float]] = None,
        project_to_road: bool = True,
        min_z: float = 0.5,
    ):
        """Generate a spawn point.

        Args:
            map (carla.Map): The map object.
            pos (list|tuple): The position of the spawn point in (x, y, z, yaw=0)
            rot (list|tuple): The rotation of the spawn point in (pitch, yaw, roll)
            project_to_road (bool): Whether to project the spawn point to the road.
            min_z (float): The minimum z value.

        Returns:
            spawn_point (carla.Transform): The spawn point.
        """
        location = carla.Location(pos[0], pos[1], max(pos[2], min_z))
        rotation = carla.Rotation()

        if rot is not None:
            rotation = carla.Rotation(*rot)
        elif project_to_road:
            wpt = map.get_waypoint(location, project_to_road=True)
            if wpt is not None:
                rotation = wpt.transform.rotation

        if len(pos) > 3:
            rotation.yaw = pos[3]

        return carla.Transform(location, rotation)

    @staticmethod
    def get_lane_end(
        map: carla.Map, location: Union[Tuple[float, float, float], carla.Location]
    ):
        """Return the lane end waypoint from given location.

        Args:
            map (carla.Map): The map object.
            location (carla.Location | list[float, float, float] | tuple[float, float, float]): The location.

        Returns:
            carla.Waypoint: The lane end waypoint.
        """
        if isinstance(location, (list, tuple)):
            location = carla.Location(*location)

        current_waypoint = map.get_waypoint(location, project_to_road=True)
        lane_end = current_waypoint.next_until_lane_end(1.0)[-1]

        return lane_end
