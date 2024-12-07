from __future__ import annotations, print_function

import json

try:
    import pygame
    from pygame.locals import (
        K_DOWN,
        K_LEFT,
        K_RIGHT,
        K_SPACE,
        K_UP,
        K_a,
        K_d,
        K_p,
        K_q,
        K_s,
        K_w,
    )
except ImportError:
    raise RuntimeError("cannot import pygame, make sure pygame package is installed")

import carla

from cmad.simulation.agents.cmad_agent import CmadAgent
from cmad.simulation.data.simulator import Simulator
from cmad.simulation.sensors.hud import HUD
from cmad.viz.render import Render


class HumanInterface(object):

    """
    Class to control a vehicle manually for debugging purposes
    """

    def __init__(self, config):
        self._surface = None
        self._render_pos = config["render_pos"]
        self._hud = HUD(config["width"], config["height"])
        self._clock = pygame.time.Clock()

    def run_interface(self, input_data):
        """
        Run the GUI
        """
        self._clock.tick(60)

        # process sensor data
        image = input_data["ManualControl"][1]

        # display image
        self._surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))

        if self._surface is not None:
            Render.draw(self._surface, self._render_pos)

        # display hud
        self._hud.render(Render.get_screen(), self._render_pos)

        pygame.display.flip()

    def quit_interface(self):
        """
        Stops the pygame window
        """
        self._surface = None
        self._hud = None


class HumanAgent(CmadAgent):

    """
    Human agent to control the ego vehicle via keyboard
    """

    current_control = None
    agent_engaged = False
    use_autopilot = False
    prev_timestamp = 0

    def setup(self, actor_config):
        """
        Setup the agent parameters
        """
        super().setup(actor_config)
        self.sensor_list.append(
            {
                "id": "ManualControl",
                "type": "sensor.camera.rgb",
                "width": actor_config["render_config"]["width"],
                "height": actor_config["render_config"]["height"],
                "fov": 100,
                "manual_control": True,
            }
        )

        self.agent_engaged = False
        self.use_autopilot = self.actor_config.get("auto_control", False)
        self.prev_timestamp = 0

        self._hic = HumanInterface(actor_config["render_config"])
        self.callbacks.append(Simulator.add_callback(self._hic._hud.on_world_tick))

        self._controller = KeyboardControl(
            actor_config.get("manual_control_config_file", None)
        )
        self._controller._auto_control = self.use_autopilot

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.

        Args:

            input_data (dict): dictionary of sensor data. E.g.

            input_data = {
                "sensor_name": (raw_data, processed_data)
                ...
            }

            timestamp (float): timestamp of the current step
        """
        self.agent_engaged = True
        self._hic.run_interface(input_data)

        control = self._controller.parse_events(timestamp - self.prev_timestamp)
        self.use_autopilot = self._controller._auto_control
        self.prev_timestamp = timestamp
        self.current_control = control

        return control


class KeyboardControl(object):

    """
    Keyboard control for the human agent
    """

    def __init__(self, path_to_conf_file=None):
        """
        Init
        """
        self._auto_control = False
        self._control = carla.VehicleControl()
        self._steer_cache = 0.0

        # Get the mode
        if path_to_conf_file:
            with open(path_to_conf_file, "r") as f:
                lines = f.read().split("\n")
                self._mode = lines[0].split(" ")[1]
                self._endpoint = lines[1].split(" ")[1]

            # Get the needed vars
            if self._mode == "log":
                self._log_data = {"records": []}

            elif self._mode == "playback":
                self._index = 0
                self._control_list = []

                with open(self._endpoint) as fd:
                    try:
                        self._records = json.load(fd)
                        self._json_to_control()
                    except ValueError:
                        # Moving to Python 3.5+ this can be replaced with json.JSONDecodeError
                        pass
        else:
            self._mode = "normal"
            self._endpoint = None

    def _json_to_control(self):
        """
        Parses the json file into a list of carla.VehicleControl
        """

        # transform strs into VehicleControl commands
        for entry in self._records["records"]:
            control = carla.VehicleControl(
                throttle=entry["control"]["throttle"],
                steer=entry["control"]["steer"],
                brake=entry["control"]["brake"],
                hand_brake=entry["control"]["hand_brake"],
                reverse=entry["control"]["reverse"],
                manual_gear_shift=entry["control"]["manual_gear_shift"],
                gear=entry["control"]["gear"],
            )
            self._control_list.append(control)

    def parse_events(self, timestamp):
        """
        Parse the keyboard events and set the vehicle controls accordingly
        """
        # Move the vehicle
        if self._mode == "playback":
            self._parse_json_control()
        else:
            self._parse_vehicle_keys(pygame.key.get_pressed(), timestamp * 1000)

        # Record the control
        if self._mode == "log":
            self._record_control()

        return self._control

    def _parse_vehicle_keys(self, keys, milliseconds):
        """
        Calculate new vehicle controls based on input keys
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.KEYUP:
                if event.key == K_q:
                    self._control.gear = 1 if self._control.reverse else -1
                    self._control.reverse = self._control.gear < 0
                elif event.key == K_p:
                    self._auto_control = not self._auto_control

        if self._auto_control:
            return

        if keys[K_UP] or keys[K_w]:
            self._control.throttle = min(self._control.throttle + 0.1, 1.0)
        else:
            self._control.throttle = 0.0

        if keys[K_DOWN] or keys[K_s]:
            self._control.brake = min(self._control.brake + 0.2, 1.0)
        else:
            self._control.brake = 0

        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.hand_brake = keys[K_SPACE]

    def _parse_json_control(self):
        """
        Gets the control corresponding to the current frame
        """

        if self._index < len(self._control_list):
            self._control = self._control_list[self._index]
            self._index += 1
        else:
            print("JSON file has no more entries")

    def _record_control(self):
        """
        Saves the list of control into a json file
        """

        new_record = {
            "control": {
                "throttle": self._control.throttle,
                "steer": self._control.steer,
                "brake": self._control.brake,
                "hand_brake": self._control.hand_brake,
                "reverse": self._control.reverse,
                "manual_gear_shift": self._control.manual_gear_shift,
                "gear": self._control.gear,
            }
        }

        self._log_data["records"].append(new_record)

    def __del__(self):
        """
        Delete method
        """
        # Get ready to log user commands
        if self._mode == "log" and self._log_data:
            with open(self._endpoint, "w") as fd:
                json.dump(self._log_data, fd, indent=4, sort_keys=True)
