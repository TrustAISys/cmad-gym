"""
Author: Morphlng
Date: 2023-07-10 10:28:33
LastEditTime: 2023-12-07 15:07:23
LastEditors: Morphlng
Description: Generate a scenario from xosc file.
"""
import argparse
import json
import math
import os
import re
import xml.etree.ElementTree as ET


def load_map(map_name):
    import carla
    from cmad.envs.static_asset import SYS_ASSETS

    xodr_path = os.path.join(
        SYS_ASSETS.paths.directory,
        "CarlaUE4/Content/Carla/Maps/OpenDrive",
        map_name + ".xodr",
    )
    if not os.path.exists(xodr_path):
        raise ValueError(f"Map {map_name} does not exist.")

    with open(xodr_path, "r") as f:
        map = carla.Map(map_name, f.read())

    return map


def iterfind(elem: ET.Element, key: str):
    """Recursively iterate all children of an element node until find the given tag name. BFS"""
    queue = [elem]
    while queue:
        node = queue.pop(0)
        if node.tag == key:
            return node
        queue.extend(node)

    return None


def parse_params(param_file: str, carla_coord: bool = True):
    with open(param_file, "r", encoding="utf8") as f:
        json_data = json.load(f)

    if not isinstance(json_data, list):
        json_data = [json_data]

    params_dict: dict[str, list] = {}
    for item in json_data:
        if "param" in item:
            for param in item["param"]:
                if "children" in param:
                    for child in param["children"]:
                        if "key" in child and "value" in child:
                            params_dict[child["key"]] = child["value"]

    if carla_coord:
        for key in params_dict:
            if key in ["y", "yaw", "dyaw", "h"] or any(
                key.endswith(x) for x in ["_y", "_yaw", "_dyaw", "_h"]
            ):
                for idx, _ in enumerate(params_dict[key]):
                    params_dict[key][idx] = -params_dict[key][idx]
                params_dict[key] = sorted(params_dict[key])

    return params_dict


def parse_location(
    location_elem: ET.Element,
    default_params: dict,
    param_dict: dict,
    carla_coordinate: bool = True,
):
    location = {}

    for attr in ["x", "y", "z", "s", "h"]:
        if attr in location_elem.attrib:
            value = location_elem.attrib[attr]
            if value.startswith("$"):
                param = value[1:]
                if param in param_dict:
                    location[f"random_range_{attr}"] = param_dict[param]
                location[attr] = float(default_params[param])
            else:
                location[attr] = float(value)

    if "roadId" in location_elem.attrib:
        location["road_id"] = int(location_elem.attrib["roadId"])
        location["lane_id"] = int(location_elem.attrib.get("laneId", 0))
        orientation = location_elem.find("Orientation")
        if orientation is not None:
            if orientation.attrib.get("type") == "absolute":
                location["yaw"] = math.degrees(float(orientation.attrib["h"]))
            else:
                location["dyaw"] = math.degrees(float(orientation.attrib["h"]))

    if carla_coordinate:
        if "y" in location:
            location["y"] = -location["y"]
        if "yaw" in location:
            location["yaw"] = -location["yaw"]
        if "dyaw" in location:
            location["dyaw"] = -location["dyaw"]
        if "h" in location:
            location["h"] = -location["h"]

    return dict(sorted(location.items(), key=lambda item: len(item[0])))


def parse_xosc(file_path: str, param_file: str = None, carla_coordinate: bool = True):
    with open(file_path, "r", encoding="utf8") as xml_file:
        xml_string = xml_file.read()

    if param_file is None:
        # extract parameter range from annotations
        pattern = r'<ParameterDeclaration name="([^"]+)"[^>]+>[\s]*<!--[^[]*\[([+-]?[0-9]+(?:\.[0-9]+)?),\s*([+-]?[0-9]+(?:\.[0-9]+)?)\]'
        matches = re.findall(pattern, xml_string)
        param_dict = {name: [float(start), float(end)] for name, start, end in matches}
    else:
        param_dict = parse_params(param_file, carla_coordinate)

    # Extract parameters from 'ParameterDeclarations' section
    root = ET.fromstring(xml_string)
    param_section = root.find("ParameterDeclarations")
    default_params = {}
    for param in param_section:
        param_name = param.attrib.get("name")
        param_value = param.attrib.get("value")
        default_params[param_name] = param_value

    # Extract map
    map_section = root.find("RoadNetwork")
    map_name = map_section.find("LogicFile").attrib.get("filepath")

    # Extract actors
    entities_section = root.find("Entities")
    actors = []

    for entity in entities_section:
        actor = {}
        actor["name"] = entity.attrib.get("name")
        blueprint = entity.find("Vehicle")
        if blueprint is not None:
            actor["blueprint"] = blueprint.attrib.get("name")
        else:
            catalog = iterfind(entity, "CatalogReference")
            if catalog is not None:
                actor["blueprint"] = catalog.attrib.get("entryName")
            else:
                actor["blueprint"] = None

        actors.append(actor)

    # Extract start and end positions
    init_section = root.find("Storyboard").find("Init")

    # Extract start locations from 'TeleportAction'
    start_locations = {}
    for private in init_section.iter("Private"):
        actor_name = private.attrib.get("entityRef")
        for action in private.iter("TeleportAction"):
            for pos in action.iter("Position"):
                road_position = pos.find("RoadPosition")
                lane_position = pos.find("LanePosition")
                world_position = pos.find("WorldPosition")

                # Choose the first non-None position
                chosen_position = None
                for position in [road_position, lane_position, world_position]:
                    if position is not None:
                        chosen_position = position
                        break

                start_locations[actor_name] = parse_location(
                    chosen_position,
                    default_params,
                    param_dict,
                    carla_coordinate,
                )

    # xosc file does not specify end locations
    end_locations = {}

    return map_name, actors, start_locations, end_locations, default_params


def generate_scenario(
    map_name: str, actors: dict, start_locations: dict = {}, end_locations: dict = {}
):
    # Generate a pseudo scenario
    scenario = {
        "actors": {},
        "map": map_name,
        "max_steps": 550,
        "weather_distribution": [0],
    }
    for actor in actors:
        name = actor["name"]

        scenario["actors"][name] = {
            "start": start_locations.get(name, {}),
            "end": end_locations.get(name, {}),
        }
    return scenario


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xosc", type=str, required=True)
    parser.add_argument("--params", type=str, required=False, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--carla_coord", action="store_true")
    args = parser.parse_args()

    # Parse xosc file
    map_name, vehicles, start_locations, end_locations, parameters = parse_xosc(
        args.xosc, args.params, args.carla_coord
    )

    # Generate scenario config
    scenario = generate_scenario(map_name, vehicles, start_locations, end_locations)

    # Save config
    with open(args.output, "w") as f:
        json.dump(scenario, f, indent=2)
