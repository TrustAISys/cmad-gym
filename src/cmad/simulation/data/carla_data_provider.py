#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides all frequently used data from CARLA via
local buffers to avoid blocking calls to CARLA
"""

from __future__ import annotations

import logging
import math
import re
from typing import Dict, List, Optional, Tuple, Union

import carla
from numpy import random
from six import iteritems

logger = logging.getLogger(__name__)


def calculate_velocity(actor: carla.Actor, use_kmh: bool = False):
    """
    Method to calculate the velocity of a actor in km/h or m/s
    """
    velocity = actor.get_velocity()
    velocity_squared = velocity.x**2
    velocity_squared += velocity.y**2
    velocity_squared += velocity.z**2
    return 3.6 * math.sqrt(velocity_squared) if use_kmh else math.sqrt(velocity_squared)


class CarlaDataProvider(object):  # pylint: disable=too-many-public-methods
    """
    This class provides access to various data of all registered actors
    It buffers the data and updates it on every CARLA tick

    Currently available data:
    - Absolute velocity
    - Location
    - Transform

    Potential additions:
    - Acceleration

    In addition it provides access to the map and the transform of all traffic lights
    """

    _actor_velocity_map: Dict[carla.Actor, carla.Vector3D] = {}
    _actor_acceleration_map: Dict[carla.Actor, carla.Vector3D] = {}
    _actor_location_map: Dict[carla.Actor, carla.Location] = {}
    _actor_transform_map: Dict[carla.Actor, carla.Transform] = {}
    _traffic_light_map: Dict[carla.TrafficLight, carla.Transform] = {}
    _carla_actor_pool: Dict[int, carla.Actor] = {}
    _global_osc_parameters: Dict[str, str] = {}
    _client: carla.Client = None
    _world: carla.World = None
    _map: carla.Map = None
    _sync_flag: bool = False
    _spawn_points: List[carla.Transform] = None
    _spawn_index: int = 0
    _blueprint_library: carla.BlueprintLibrary = None
    _ego_vehicle_route: List[carla.Waypoint] = None
    _traffic_manager_port: int = 8000
    _random_seed: int = 2000
    _rng = random.RandomState(_random_seed)

    @staticmethod
    def register_actor(actor: carla.Actor):
        """
        Add new actor to dictionaries
        If actor already exists, throw an exception
        """
        if actor in CarlaDataProvider._actor_velocity_map:
            raise KeyError(
                f"Vehicle '{actor.id}' already registered. Cannot register twice!"
            )
        else:
            CarlaDataProvider._actor_velocity_map[actor] = None

        if actor in CarlaDataProvider._actor_acceleration_map:
            raise KeyError(
                f"Vehicle '{actor.id}' already registered. Cannot register twice!"
            )
        else:
            CarlaDataProvider._actor_acceleration_map[actor] = None

        if actor in CarlaDataProvider._actor_location_map:
            raise KeyError(
                f"Vehicle '{actor.id}' already registered. Cannot register twice!"
            )
        else:
            CarlaDataProvider._actor_location_map[actor] = None

        if actor in CarlaDataProvider._actor_transform_map:
            raise KeyError(
                f"Vehicle '{actor.id}' already registered. Cannot register twice!"
            )
        else:
            CarlaDataProvider._actor_transform_map[actor] = None

    @staticmethod
    def update_osc_global_params(parameters: dict):
        """
        updates/initializes global osc parameters.
        """
        CarlaDataProvider._global_osc_parameters.update(parameters)

    @staticmethod
    def get_osc_global_param_value(ref: str):
        """
        returns updated global osc parameter value.
        """
        return CarlaDataProvider._global_osc_parameters.get(ref.replace("$", ""))

    @staticmethod
    def register_actors(actors: carla.Actor):
        """
        Add new set of actors to dictionaries
        """
        for actor in actors:
            CarlaDataProvider.register_actor(actor)

    @staticmethod
    def on_carla_tick():
        """
        Callback from CARLA
        """
        for actor in CarlaDataProvider._actor_velocity_map:
            if actor is not None and actor.is_alive:
                CarlaDataProvider._actor_velocity_map[actor] = actor.get_velocity()

        for actor in CarlaDataProvider._actor_acceleration_map:
            if actor is not None and actor.is_alive:
                CarlaDataProvider._actor_acceleration_map[actor] = (
                    actor.get_acceleration()
                )

        for actor in CarlaDataProvider._actor_location_map:
            if actor is not None and actor.is_alive:
                CarlaDataProvider._actor_location_map[actor] = actor.get_location()

        for actor in CarlaDataProvider._actor_transform_map:
            if actor is not None and actor.is_alive:
                CarlaDataProvider._actor_transform_map[actor] = actor.get_transform()

        world = CarlaDataProvider._world
        if world is None:
            logger.warning("WARNING: CarlaDataProvider couldn't find the world")

    @staticmethod
    def get_velocity(
        actor: carla.Actor, use_vector: bool = False
    ) -> Union[float, carla.Vector3D]:
        """
        returns the velocity for the given actor
        """
        return CarlaDataProvider.get_velocity_by_id(actor.id, use_vector)

    @staticmethod
    def get_velocity_by_id(
        actor_id: int, use_vector: bool = False
    ) -> Union[float, carla.Vector3D]:
        """
        returns the velocity for the given actor
        """
        for key in CarlaDataProvider._actor_velocity_map:
            if key.id == actor_id:
                vector = CarlaDataProvider._actor_velocity_map[key]
                return vector.length() if not use_vector else vector

        # We are intentionally not throwing here
        # This may cause exception loops in py_trees
        logger.warning("%s.get_velocity: %s not found!", __name__, actor_id)
        return 0.0

    @staticmethod
    def get_acceleration(actor: carla.Actor) -> carla.Vector3D:
        """
        returns the acceleration for the given actor
        """
        return CarlaDataProvider.get_acceleration_by_id(actor.id)

    @staticmethod
    def get_acceleration_by_id(actor_id: int) -> carla.Vector3D:
        """
        returns the acceleration for the given actor
        """
        for key in CarlaDataProvider._actor_acceleration_map:
            if key.id == actor_id:
                return CarlaDataProvider._actor_acceleration_map[key]

        # We are intentionally not throwing here
        # This may cause exception loops in py_trees
        logger.warning("%s.get_acceleration: %s not found!", __name__, actor_id)
        return None

    @staticmethod
    def get_location(actor: carla.Actor) -> carla.Location:
        """
        returns the location for the given actor
        """
        return CarlaDataProvider.get_location_by_id(actor.id)

    @staticmethod
    def get_location_by_id(actor_id: int) -> carla.Location:
        """
        returns the location for the given actor
        """
        for key in CarlaDataProvider._actor_location_map:
            if key.id == actor_id:
                return CarlaDataProvider._actor_location_map[key]

        # We are intentionally not throwing here
        # This may cause exception loops in py_trees
        logger.warning("%s.get_location: %s not found!", __name__, actor_id)
        return None

    @staticmethod
    def get_transform(actor: carla.Actor) -> carla.Transform:
        """
        returns the transform for the given actor
        """
        return CarlaDataProvider.get_transform_by_id(actor.id)

    @staticmethod
    def get_transform_by_id(actor_id: int) -> carla.Transform:
        """
        returns the transform for the given actor
        """
        for key in CarlaDataProvider._actor_transform_map:
            if key.id == actor_id:
                return CarlaDataProvider._actor_transform_map[key]

        # We are intentionally not throwing here
        # This may cause exception loops in py_trees
        logger.warning("%s.get_transform: %s not found!", __name__, actor_id)
        return None

    @staticmethod
    def set_client(client: carla.Client):
        """
        Set the CARLA client
        """
        CarlaDataProvider._client = client

    @staticmethod
    def get_client() -> carla.Client:
        """
        Get the CARLA client
        """
        return CarlaDataProvider._client

    @staticmethod
    def set_world(world: carla.World):
        """
        Set the world and world settings
        """
        CarlaDataProvider._world = world
        CarlaDataProvider._sync_flag = world.get_settings().synchronous_mode
        CarlaDataProvider._map = world.get_map()
        CarlaDataProvider._blueprint_library = world.get_blueprint_library()
        CarlaDataProvider.generate_spawn_points()
        CarlaDataProvider.prepare_map()

    @staticmethod
    def get_world() -> carla.World:
        """
        Return world
        """
        return CarlaDataProvider._world

    @staticmethod
    def get_map(world: carla.World = Optional[None]) -> carla.Map:
        """
        Get the current map
        """
        if CarlaDataProvider._map is None:
            if world is None:
                if CarlaDataProvider._world is None:
                    raise ValueError("class member 'world'' not initialized yet")
                else:
                    CarlaDataProvider._map = CarlaDataProvider._world.get_map()
            else:
                CarlaDataProvider._map = world.get_map()

        return CarlaDataProvider._map

    @staticmethod
    def is_sync_mode():
        """
        @return true if syncronuous mode is used
        """
        return CarlaDataProvider._sync_flag

    @staticmethod
    def find_weather_presets():
        """
        Get weather presets from CARLA
        """
        rgx = re.compile(".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)")

        def name(x):
            return " ".join(m.group(0) for m in rgx.finditer(x))

        presets = [x for x in dir(carla.WeatherParameters) if re.match("[A-Z].+", x)]
        return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

    @staticmethod
    def prepare_map():
        """
        This function set the current map and loads all traffic lights for this map to
        _traffic_light_map
        """
        if CarlaDataProvider._map is None:
            CarlaDataProvider._map = CarlaDataProvider._world.get_map()

        # Parse all traffic lights
        CarlaDataProvider._traffic_light_map.clear()
        for traffic_light in CarlaDataProvider._world.get_actors().filter(
            "*traffic_light*"
        ):
            if traffic_light not in CarlaDataProvider._traffic_light_map.keys():
                CarlaDataProvider._traffic_light_map[traffic_light] = (
                    traffic_light.get_transform()
                )
            else:
                raise KeyError(
                    f"Traffic light '{traffic_light.id}' already registered. Cannot register twice!"
                )

    @staticmethod
    def annotate_trafficlight_in_group(
        traffic_light: carla.TrafficLight,
    ) -> Dict[str, List[carla.TrafficLight]]:
        """
        Get dictionary with traffic light group info for a given traffic light
        """
        dict_annotations = {"ref": [], "opposite": [], "left": [], "right": []}

        # Get the waypoints
        ref_location = CarlaDataProvider.get_trafficlight_trigger_location(
            traffic_light
        )
        ref_waypoint = CarlaDataProvider.get_map().get_waypoint(ref_location)
        ref_yaw = ref_waypoint.transform.rotation.yaw

        group_tl = traffic_light.get_group_traffic_lights()

        for target_tl in group_tl:
            if traffic_light.id == target_tl.id:
                dict_annotations["ref"].append(target_tl)
            else:
                # Get the angle between yaws
                target_location = CarlaDataProvider.get_trafficlight_trigger_location(
                    target_tl
                )
                target_waypoint = CarlaDataProvider.get_map().get_waypoint(
                    target_location
                )
                target_yaw = target_waypoint.transform.rotation.yaw

                diff = (target_yaw - ref_yaw) % 360

                if diff > 330:
                    continue
                elif diff > 225:
                    dict_annotations["right"].append(target_tl)
                elif diff > 135.0:
                    dict_annotations["opposite"].append(target_tl)
                elif diff > 30:
                    dict_annotations["left"].append(target_tl)

        return dict_annotations

    @staticmethod
    def get_trafficlight_trigger_location(
        traffic_light: carla.TrafficLight,
    ) -> carla.Location:
        """
        Calculates the yaw of the waypoint that represents the trigger volume of the traffic light
        """

        def rotate_point(point: carla.Vector3D, angle: float):
            """
            rotate a given point by a given angle
            """
            x_ = (
                math.cos(math.radians(angle)) * point.x
                - math.sin(math.radians(angle)) * point.y
            )
            y_ = (
                math.sin(math.radians(angle)) * point.x
                - math.cos(math.radians(angle)) * point.y
            )
            return carla.Vector3D(x_, y_, point.z)

        base_transform = traffic_light.get_transform()
        base_rot = base_transform.rotation.yaw
        area_loc = base_transform.transform(traffic_light.trigger_volume.location)
        area_ext = traffic_light.trigger_volume.extent

        point = rotate_point(carla.Vector3D(0, 0, area_ext.z), base_rot)
        point_location = area_loc + carla.Location(x=point.x, y=point.y)

        return carla.Location(point_location.x, point_location.y, point_location.z)

    @staticmethod
    def update_light_states(
        ego_light: carla.TrafficLight,
        annotations: Dict[str, List[carla.TrafficLight]],
        states: Dict[str, carla.TrafficLightState],
        freeze: bool = False,
        timeout: int = 1000000000,
    ) -> dict:
        """
        Update traffic light states
        """
        reset_params = []

        for state in states:
            relevant_lights = []
            if state == "ego":
                relevant_lights = [ego_light]
            else:
                relevant_lights = annotations[state]
            for light in relevant_lights:
                prev_state = light.get_state()
                prev_green_time = light.get_green_time()
                prev_red_time = light.get_red_time()
                prev_yellow_time = light.get_yellow_time()
                reset_params.append(
                    {
                        "light": light,
                        "state": prev_state,
                        "green_time": prev_green_time,
                        "red_time": prev_red_time,
                        "yellow_time": prev_yellow_time,
                    }
                )

                light.set_state(states[state])
                if freeze:
                    light.set_green_time(timeout)
                    light.set_red_time(timeout)
                    light.set_yellow_time(timeout)

        return reset_params

    @staticmethod
    def reset_lights(reset_params: dict):
        """
        Reset traffic lights
        """
        for param in reset_params:
            light: carla.TrafficLight = param["light"]
            light.set_state(param["state"])
            light.set_green_time(param["green_time"])
            light.set_red_time(param["red_time"])
            light.set_yellow_time(param["yellow_time"])

    @staticmethod
    def get_next_traffic_light(actor: carla.Actor, use_cached_location: bool = True):
        """
        returns the next relevant traffic light for the provided actor
        """

        if not use_cached_location:
            location = actor.get_transform().location
        else:
            location = CarlaDataProvider.get_location(actor)

        waypoint = CarlaDataProvider.get_map().get_waypoint(location)
        # Create list of all waypoints until next intersection
        list_of_waypoints = []
        while waypoint and not waypoint.is_intersection:
            list_of_waypoints.append(waypoint)
            waypoint = waypoint.next(2.0)[0]

        # If the list is empty, the actor is in an intersection
        if not list_of_waypoints:
            return None

        relevant_traffic_light = None
        distance_to_relevant_traffic_light = float("inf")

        for traffic_light in CarlaDataProvider._traffic_light_map:
            if hasattr(traffic_light, "trigger_volume"):
                tl_t = CarlaDataProvider._traffic_light_map[traffic_light]
                transformed_tv = tl_t.transform(traffic_light.trigger_volume.location)
                distance = carla.Location(transformed_tv).distance(
                    list_of_waypoints[-1].transform.location
                )

                if distance < distance_to_relevant_traffic_light:
                    relevant_traffic_light = traffic_light
                    distance_to_relevant_traffic_light = distance

        return relevant_traffic_light

    @staticmethod
    def set_ego_vehicle_route(route: List[carla.Waypoint]):
        """
        Set the route of the ego vehicle

        @todo extend ego_vehicle_route concept to support multi ego_vehicle scenarios
        """
        CarlaDataProvider._ego_vehicle_route = route

    @staticmethod
    def get_ego_vehicle_route():
        """
        returns the currently set route of the ego vehicle
        Note: Can be None
        """
        return CarlaDataProvider._ego_vehicle_route

    @staticmethod
    def generate_spawn_points():
        """
        Generate spawn points for the current map
        """
        spawn_points = list(
            CarlaDataProvider.get_map(CarlaDataProvider._world).get_spawn_points()
        )
        CarlaDataProvider._rng.shuffle(spawn_points)
        CarlaDataProvider._spawn_points = spawn_points
        CarlaDataProvider._spawn_index = 0

    @staticmethod
    def create_blueprint(
        model: str,
        rolename: str = "scenario",
        color: Optional[carla.Color] = None,
        actor_category: str = "car",
        safe: bool = False,
        immortal: bool = True,
    ) -> carla.ActorBlueprint:
        """
        Function to setup the blueprint of an actor given its model and other relevant parameters
        """

        _actor_blueprint_categories = {
            "car": "vehicle.tesla.model3",
            "van": "vehicle.volkswagen.t2",
            "truck": "vehicle.carlamotors.carlacola",
            "trailer": "",
            "semitrailer": "",
            "bus": "vehicle.volkswagen.t2",
            "motorbike": "vehicle.kawasaki.ninja",
            "bicycle": "vehicle.diamondback.century",
            "train": "",
            "tram": "",
            "pedestrian": "walker.pedestrian.0001",
        }

        # Set the model
        try:
            blueprints = CarlaDataProvider._blueprint_library.filter(model)
            blueprints_ = []
            if safe and actor_category != "pedestrian" and len(blueprints) > 1:
                for bp in blueprints:
                    if (
                        bp.id.endswith("firetruck")
                        or bp.id.endswith("ambulance")
                        or int(bp.get_attribute("number_of_wheels")) < 4
                    ):
                        # Two wheeled vehicles take much longer to render + bicicles shouldn't behave like cars
                        continue
                    blueprints_.append(bp)
            else:
                blueprints_ = blueprints

            blueprint = CarlaDataProvider._rng.choice(blueprints_)
        except Exception:
            # The model is not part of the blueprint library. Let's take a default one for the given category
            bp_filter = "vehicle.*"
            new_model = _actor_blueprint_categories.get(
                actor_category, "vehicle.tesla.model3"
            )
            if new_model != "":
                bp_filter = new_model
            logger.warning(
                "WARNING: Actor model %s not available. Using instead %s",
                model,
                new_model,
            )
            blueprint = CarlaDataProvider._rng.choice(
                CarlaDataProvider._blueprint_library.filter(bp_filter)
            )

        # Set the color
        if color:
            if not blueprint.has_attribute("color"):
                logger.warning(
                    "WARNING: Cannot set Color ({%s}) for actor %d due to missing blueprint attribute",
                    color,
                    blueprint.id,
                )
            else:
                default_color_rgba = blueprint.get_attribute("color").as_color()
                default_color = "({}, {}, {})".format(
                    default_color_rgba.r, default_color_rgba.g, default_color_rgba.b
                )
                try:
                    blueprint.set_attribute("color", color)
                except ValueError:
                    # Color can't be set for this vehicle
                    logger.warning(
                        "WARNING: Color (%s) cannot be set for actor %d. Using instead: (%s)",
                        color,
                        blueprint.id,
                        default_color,
                    )
                    blueprint.set_attribute("color", default_color)
        else:
            if blueprint.has_attribute("color") and rolename != "hero":
                color = CarlaDataProvider._rng.choice(
                    blueprint.get_attribute("color").recommended_values
                )
                blueprint.set_attribute("color", color)

        # Make pedestrians mortal
        if blueprint.has_attribute("is_invincible") and not immortal:
            blueprint.set_attribute("is_invincible", "false")

        # Set the rolename
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", rolename)

        return blueprint

    @staticmethod
    def handle_actor_batch(batch: list, tick: bool = True) -> List[carla.Actor]:
        """
        Forward a CARLA command batch to spawn actors to CARLA, and gather the responses.
        Returns list of actors on success, none otherwise
        """
        sync_mode = CarlaDataProvider.is_sync_mode()
        actors = []

        if CarlaDataProvider._client:
            responses = CarlaDataProvider._client.apply_batch_sync(
                batch, sync_mode and tick
            )
        else:
            raise ValueError("class member 'client'' not initialized yet")

        # Wait (or not) for the actors to be spawned properly before we do anything
        if not tick:
            pass
        elif sync_mode:
            CarlaDataProvider._world.tick()
        else:
            CarlaDataProvider._world.wait_for_tick()

        actor_ids = [r.actor_id for r in responses if not r.error]
        for r in responses:
            if r.error:
                logger.warning("WARNING: Not all actors were spawned")
                break
        actors = list(CarlaDataProvider._world.get_actors(actor_ids))
        return actors

    @staticmethod
    def request_new_actor(
        model: str,
        spawn_point: carla.Transform,
        rolename: str = "scenario",
        autopilot: bool = False,
        random_location: bool = False,
        color: Optional[carla.Color] = None,
        actor_category: str = "car",
        safe_blueprint: bool = False,
        blueprint: Optional[carla.ActorBlueprint] = None,
        attach_to: Optional[carla.Actor] = None,
        immortal: bool = True,
        tick: bool = True,
    ) -> carla.Actor:
        """
        This method tries to create a new actor, returning it if successful (None otherwise).

        Args:
            model (str): model of the actor to be spawned, e.g. 'vehicle.tesla.model3'
            spawn_point (carla.Transform): spawn point of the actor
            rolename (str): name of the actor
            autopilot (bool): if True, the actor will be spawned with autopilot enabled
            random_location (bool): if True, the actor will be spawned at a random spawn point
            color (carla.Color): color of the actor
            actor_category (str): category of the actor, e.g. 'car', 'pedestrian', etc.
            safe_blueprint (bool): if True, the blueprint will be filtered to avoid spawning firetrucks, ambulances, etc.
            blueprint (carla.ActorBlueprint): blueprint to be used for spawning the actor. If None, a new blueprint will be created
            attach_to (carla.Actor): attach the new actor to an existing actor (e.g. WalkerController attch to a pedestrian)
            immortal (bool): if True, the actor will be spawned with invicibility enabled (walker only)
            tick (bool): if True, the world will be ticked after spawning the actor

            Other arguments are the same as in create_blueprint.
        """
        if blueprint is None:
            blueprint = CarlaDataProvider.create_blueprint(
                model, rolename, color, actor_category, safe_blueprint, immortal
            )

        if random_location:
            actor = None
            while not actor:
                spawn_point = CarlaDataProvider._rng.choice(
                    CarlaDataProvider._spawn_points
                )
                actor = CarlaDataProvider._world.try_spawn_actor(
                    blueprint, spawn_point, attach_to
                )

        else:
            # slightly lift the actor to avoid collisions with ground when spawning the actor
            # DO NOT USE spawn_point directly, as this will modify spawn_point permanently
            _spawn_point = carla.Transform(carla.Location(), spawn_point.rotation)
            _spawn_point.location.x = spawn_point.location.x
            _spawn_point.location.y = spawn_point.location.y
            _spawn_point.location.z = spawn_point.location.z + 0.2
            try:
                actor = CarlaDataProvider._world.spawn_actor(
                    blueprint, _spawn_point, attach_to
                )
            except Exception as e:
                logger.warning(
                    "WARNING: Cannot spawn actor %s (%s) at position %s",
                    model,
                    rolename,
                    spawn_point.location,
                )
                logger.warning("SpawnError: %s", e)
                return None

        # De/activate the autopilot of the actor if it belongs to vehicle
        if actor.type_id.startswith("vehicle."):
            actor.set_autopilot(autopilot, CarlaDataProvider._traffic_manager_port)
        elif autopilot:
            logger.warning("WARNING: Tried to set the autopilot of a non vehicle actor")

        # Wait for the actor to be spawned properly before we do anything
        if not tick:
            pass
        elif CarlaDataProvider.is_sync_mode():
            CarlaDataProvider._world.tick()
        else:
            CarlaDataProvider._world.wait_for_tick()

        CarlaDataProvider._carla_actor_pool[actor.id] = actor
        CarlaDataProvider.register_actor(actor)
        return actor

    @staticmethod
    def request_new_actors(
        actor_list: List[ActorConfigurationData], safe_blueprint=False, tick=True
    ) -> List[carla.Actor]:
        """
        This method tries to series of actor in batch. If this was successful,
        the new actors are returned, None otherwise.

        param:
        - actor_list: list of ActorConfigurationData
        """

        SpawnActor = carla.command.SpawnActor
        PhysicsCommand = carla.command.SetSimulatePhysics
        FutureActor = carla.command.FutureActor
        ApplyTransform = carla.command.ApplyTransform
        SetAutopilot = carla.command.SetAutopilot
        SetVehicleLightState = carla.command.SetVehicleLightState

        batch = []

        CarlaDataProvider.generate_spawn_points()

        for actor in actor_list:
            # Get the blueprint
            blueprint = CarlaDataProvider.create_blueprint(
                actor.model, actor.rolename, actor.color, actor.category, safe_blueprint
            )

            # Get the spawn point
            transform = actor.transform
            if actor.random_location:
                if CarlaDataProvider._spawn_index >= len(
                    CarlaDataProvider._spawn_points
                ):
                    logger.warning("No more spawn points to use")
                    break
                else:
                    _spawn_point = CarlaDataProvider._spawn_points[
                        CarlaDataProvider._spawn_index
                    ]
                    CarlaDataProvider._spawn_index += 1

            else:
                _spawn_point = carla.Transform()
                _spawn_point.rotation = transform.rotation
                _spawn_point.location.x = transform.location.x
                _spawn_point.location.y = transform.location.y
                if blueprint.has_tag("walker"):
                    # On imported OpenDRIVE maps, spawning of pedestrians can fail.
                    # By increasing the z-value the chances of success are increased.
                    map_name = CarlaDataProvider._map.name.split("/")[-1]
                    if not map_name.startswith("OpenDrive"):
                        _spawn_point.location.z = transform.location.z + 0.2
                    else:
                        _spawn_point.location.z = transform.location.z + 0.8
                else:
                    _spawn_point.location.z = transform.location.z + 0.2

            # Get the command
            command = SpawnActor(blueprint, _spawn_point)
            command.then(
                SetAutopilot(
                    FutureActor,
                    actor.autopilot,
                    CarlaDataProvider._traffic_manager_port,
                )
            )

            if (
                actor.args is not None
                and "physics" in actor.args
                and actor.args["physics"] == "off"
            ):
                command.then(ApplyTransform(FutureActor, _spawn_point)).then(
                    PhysicsCommand(FutureActor, False)
                )
            elif actor.category == "misc":
                command.then(PhysicsCommand(FutureActor, True))
            if (
                actor.args is not None
                and "lights" in actor.args
                and actor.args["lights"] == "on"
            ):
                command.then(
                    SetVehicleLightState(FutureActor, carla.VehicleLightState.All)
                )

            batch.append(command)

        actors = CarlaDataProvider.handle_actor_batch(batch, tick)
        for actor in actors:
            if actor is None:
                continue
            CarlaDataProvider._carla_actor_pool[actor.id] = actor
            CarlaDataProvider.register_actor(actor)

        return actors

    @staticmethod
    def request_new_batch_actors(
        model: str,
        amount: int,
        spawn_points: List[carla.Transform],
        autopilot: bool = False,
        random_location: bool = False,
        rolename: str = "scenario",
        safe_blueprint: bool = False,
        tick: bool = True,
    ) -> List[carla.Actor]:
        """
        Simplified version of "request_new_actors". This method also create several actors in batch.

        Instead of needing a list of ActorConfigurationData, an "amount" parameter is used.
        This makes actor spawning easier but reduces the amount of configurability.

        Some parameters are the same for all actors (rolename, autopilot and random location)
        while others are randomized (color)
        """

        SpawnActor = carla.command.SpawnActor  # pylint: disable=invalid-name
        SetAutopilot = carla.command.SetAutopilot  # pylint: disable=invalid-name
        FutureActor = carla.command.FutureActor  # pylint: disable=invalid-name

        CarlaDataProvider.generate_spawn_points()

        batch = []

        for i in range(amount):
            # Get vehicle by model
            blueprint = CarlaDataProvider.create_blueprint(
                model, rolename, safe=safe_blueprint
            )

            if random_location:
                if CarlaDataProvider._spawn_index >= len(
                    CarlaDataProvider._spawn_points
                ):
                    logger.warning(
                        "No more spawn points to use. Spawned %d actors out of %d",
                        i + 1,
                        amount,
                    )
                    break
                else:
                    spawn_point = CarlaDataProvider._spawn_points[
                        CarlaDataProvider._spawn_index
                    ]
                    CarlaDataProvider._spawn_index += 1
            else:
                try:
                    spawn_point = spawn_points[i]
                except IndexError:
                    logger.warning(
                        "The amount of spawn points is lower than the amount of vehicles spawned"
                    )
                    break

            if spawn_point:
                batch.append(
                    SpawnActor(blueprint, spawn_point).then(
                        SetAutopilot(
                            FutureActor,
                            autopilot,
                            CarlaDataProvider._traffic_manager_port,
                        )
                    )
                )

        actors = CarlaDataProvider.handle_actor_batch(batch, tick)
        for actor in actors:
            if actor is None:
                continue
            CarlaDataProvider._carla_actor_pool[actor.id] = actor
            CarlaDataProvider.register_actor(actor)

        return actors

    @staticmethod
    def get_actors(
        actor_only: bool = False,
    ) -> List[Union[Tuple[int, carla.Actor]], List[carla.Actor]]:
        """
        Return list of actors and their ids

        Note: iteritems from six is used to allow compatibility with Python 2 and 3
        """
        return (
            CarlaDataProvider._carla_actor_pool.values()
            if actor_only
            else iteritems(CarlaDataProvider._carla_actor_pool)
        )

    @staticmethod
    def actor_id_exists(actor_id: int) -> bool:
        """
        Check if a certain id is still at the simulation
        """
        if actor_id in CarlaDataProvider._carla_actor_pool:
            return True

        return False

    @staticmethod
    def get_hero_actor(rolename: str = "hero") -> carla.Actor:
        """
        Get the actor object of the hero actor if it exists, returns none otherwise.
        """
        for actor_id in CarlaDataProvider._carla_actor_pool:
            if (
                CarlaDataProvider._carla_actor_pool[actor_id].attributes["role_name"]
                == rolename
            ):
                return CarlaDataProvider._carla_actor_pool[actor_id]
        return None

    @staticmethod
    def get_actor_by_id(actor_id: int) -> carla.Actor:
        """
        Get an actor from the pool by using its ID. If the actor
        does not exist, None is returned.
        """
        if actor_id in CarlaDataProvider._carla_actor_pool:
            return CarlaDataProvider._carla_actor_pool[actor_id]

        logger.warning("Non-existing actor id %d", actor_id)
        return None

    @staticmethod
    def remove_actor_by_id(actor_id: int):
        """
        Remove an actor from the pool using its ID
        """
        if actor_id in CarlaDataProvider._carla_actor_pool:
            actor = CarlaDataProvider._carla_actor_pool.pop(actor_id)
            logger.info(
                "Removing actor id %d, final transform %s.",
                actor_id,
                actor.get_transform(),
            )

            actor.destroy()
            CarlaDataProvider._actor_velocity_map.pop(actor, None)
            CarlaDataProvider._actor_acceleration_map.pop(actor, None)
            CarlaDataProvider._actor_location_map.pop(actor, None)
            CarlaDataProvider._actor_transform_map.pop(actor, None)
        else:
            logger.warning("Trying to remove a non-existing actor id %d", actor_id)

    @staticmethod
    def remove_actor_attachment(actor_id: int, ignores: List[int] = None):
        """Remove all actor attached to given actor_id's actor"""
        ignores = ignores or []
        all_actors = CarlaDataProvider.get_world().get_actors()
        removed_ids = []
        for actor in all_actors:
            if (
                actor.parent
                and actor.parent.id == actor_id
                and (actor.id not in ignores)
            ):
                if hasattr(actor, "stop") and actor.is_listening:
                    actor.stop()
                actor.destroy()
                removed_ids.append(actor.id)

        return removed_ids

    @staticmethod
    def remove_actors_in_surrounding(location: carla.Location, distance: float):
        """
        Remove all actors from the pool that are closer than distance to the
        provided location
        """
        for actor_id in CarlaDataProvider._carla_actor_pool.copy():
            if (
                CarlaDataProvider._carla_actor_pool[actor_id]
                .get_location()
                .distance(location)
                < distance
            ):
                CarlaDataProvider._carla_actor_pool[actor_id].destroy()
                CarlaDataProvider._carla_actor_pool.pop(actor_id)

        # Remove all keys with None values
        CarlaDataProvider._carla_actor_pool = dict(
            {k: v for k, v in CarlaDataProvider._carla_actor_pool.items() if v}
        )

    @staticmethod
    def get_traffic_manager_port() -> int:
        """
        Get the port of the traffic manager.
        """
        return CarlaDataProvider._traffic_manager_port

    @staticmethod
    def set_traffic_manager_port(tm_port: int):
        """
        Set the port to use for the traffic manager.
        """
        CarlaDataProvider._traffic_manager_port = tm_port

    @staticmethod
    def cleanup(soft_reset: bool = False, completely: bool = False):
        """
        Cleanup and remove all entries from all dictionaries

        Args:
            soft_reset (bool): If True, the actors will not be destroyed
            completely (bool): If True, the client will be destroyed
        """

        if not soft_reset:
            DestroyActor = carla.command.DestroyActor  # pylint: disable=invalid-name
            batch = []

            for actor_id in CarlaDataProvider._carla_actor_pool.copy():
                actor = CarlaDataProvider._carla_actor_pool[actor_id]
                if actor is not None and actor.is_alive:
                    if isinstance(actor, carla.WalkerAIController):
                        actor.stop()
                    batch.append(DestroyActor(actor))

            if CarlaDataProvider._client:
                try:
                    CarlaDataProvider._client.apply_batch_sync(batch)
                except RuntimeError as e:
                    if "time-out" in str(e):
                        pass
                    else:
                        raise e

        CarlaDataProvider.reset(completely, soft_reset)

    @staticmethod
    def reset(completely: bool = False, soft_reset: bool = False):
        """
        Reset the data provider
        """
        CarlaDataProvider._rng = random.RandomState(CarlaDataProvider._random_seed)

        if soft_reset:
            for actor in CarlaDataProvider._carla_actor_pool.values():
                CarlaDataProvider._actor_velocity_map[actor] = None
                CarlaDataProvider._actor_acceleration_map[actor] = None
                CarlaDataProvider._actor_location_map[actor] = None
                CarlaDataProvider._actor_transform_map[actor] = None
        else:
            CarlaDataProvider._actor_velocity_map.clear()
            CarlaDataProvider._actor_acceleration_map.clear()
            CarlaDataProvider._actor_location_map.clear()
            CarlaDataProvider._actor_transform_map.clear()
            CarlaDataProvider._ego_vehicle_route = None
            CarlaDataProvider._carla_actor_pool = {}
            CarlaDataProvider._spawn_index = 0

        if completely:
            CarlaDataProvider._map = None
            CarlaDataProvider._world = None
            CarlaDataProvider._sync_flag = False
            CarlaDataProvider._client = None
            CarlaDataProvider._spawn_points = None
            CarlaDataProvider._traffic_light_map.clear()


class ActorConfigurationData(object):
    """
    This is a configuration base class to hold model and transform attributes
    """

    def __init__(
        CarlaDataProvider,
        model: str,
        transform: carla.Transform,
        rolename: str = "other",
        speed: float = 0,
        autopilot: bool = False,
        random: bool = False,
        color: Optional[carla.Color] = None,
        category: str = "car",
        args=None,
    ):
        CarlaDataProvider.model = model
        CarlaDataProvider.rolename = rolename
        CarlaDataProvider.transform = transform
        CarlaDataProvider.speed = speed
        CarlaDataProvider.autopilot = autopilot
        CarlaDataProvider.random_location = random
        CarlaDataProvider.color = color
        CarlaDataProvider.category = category
        CarlaDataProvider.args = args
