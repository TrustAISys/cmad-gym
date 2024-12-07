from __future__ import annotations

import datetime
import logging
import math

import pygame

logger = logging.getLogger(__name__)


def get_actor_display_name(actor, truncate=250):
    name = " ".join(actor.type_id.replace("_", ".").title().split(".")[1:])
    return (name[: truncate - 1] + "\u2026") if len(name) > truncate else name


class HUD(object):
    def __init__(self, width: int, height: int):
        if pygame.get_init() is False:
            pygame.init()

        self.dim = (width, height)
        fonts = [x for x in pygame.font.get_fonts() if x is not None and "mono" in x]
        default_font = pygame.font.get_default_font()
        mono = default_font if len(fonts) == 0 else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 14)
        self.server_fps = 0
        self.frame_number = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame_number = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, vehicle, collision_sensor, clock):
        if not self._show_info:
            return

        world = vehicle.get_world()
        t = vehicle.get_transform()
        v = vehicle.get_velocity()
        c = vehicle.get_control()

        heading = "N" if abs(t.rotation.yaw) < 89.5 else ""
        heading += "S" if abs(t.rotation.yaw) > 90.5 else ""
        heading += "E" if 179.5 > t.rotation.yaw > 0.5 else ""
        heading += "W" if -0.5 > t.rotation.yaw > -179.5 else ""
        colhist = collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame_number - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.get_actors().filter("vehicle.*")
        self._info_text = [
            "Server:  % 16d FPS" % self.server_fps,
            "Client:  % 16d FPS" % clock.get_fps(),
            "",
            "Vehicle: % 20s" % get_actor_display_name(vehicle, truncate=20),
            "Simulation time: % 12s"
            % datetime.timedelta(seconds=int(self.simulation_time)),
            "",
            "Speed:   % 15.0f km/h" % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            "Heading:% 16.0f\N{DEGREE SIGN} % 2s" % (t.rotation.yaw, heading),
            "Location:% 20s" % ("(% 5.1f, % 5.1f)" % (t.location.x, t.location.y)),
            "Height:  % 18.0f m" % t.location.z,
            "",
            ("Throttle:", c.throttle, 0.0, 1.0),
            ("Steer:", c.steer, -1.0, 1.0),
            ("Brake:", c.brake, 0.0, 1.0),
            ("Reverse:", c.reverse),
            ("Hand brake:", c.hand_brake),
            "",
            "Collision:",
            collision,
            "",
            "Number of vehicles: % 8d" % len(vehicles),
        ]
        if len(vehicles) > 1:
            self._info_text += ["Nearby vehicles:"]
            vehicles = [
                (self.distance(x.get_location(), t), x)
                for x in vehicles
                if x.id != vehicle.id
            ]
            for d, vehicle in sorted(vehicles):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append("% 4dm %s" % (d, vehicle_type))

    def distance(self, l, t):  # noqa: E741
        return math.sqrt(
            (l.x - t.location.x) ** 2
            + (l.y - t.location.y) ** 2
            + (l.z - t.location.z) ** 2
        )

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        logger.info("Notification disabled: %s", text)

    def error(self, text):
        logger.info("Notification error disabled: %s", text)

    def render(self, display, render_pose=(0, 0)):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, render_pose)
            v_offset = 4 + render_pose[1]
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [
                            (x + 8, v_offset + 8 + (1.0 - y) * 30)
                            for x, y in enumerate(item)
                        ]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(
                            display, (255, 255, 255), rect, 0 if item[1] else 1
                        )
                    else:
                        rect_border = pygame.Rect(
                            (bar_h_offset, v_offset + 8), (bar_width, 6)
                        )
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect(
                                (bar_h_offset + f * (bar_width - 6), v_offset + 8),
                                (6, 6),
                            )
                        else:
                            rect = pygame.Rect(
                                (bar_h_offset, v_offset + 8), (f * bar_width, 6)
                            )
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
