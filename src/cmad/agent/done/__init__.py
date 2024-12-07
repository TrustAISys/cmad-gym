from cmad.agent.done.done import Done
from cmad.simulation.data.measurement import Measurements


def setup_done_strategy():

    from cmad.simulation.data.local_carla_api import Location, Transform
    from cmad.simulation.maps import PathTracker

    def _done_never(actor_id: str, measurements: Measurements):
        """Never done."""
        return False

    def _done_timeout(actor_id: str, measurements: Measurements):
        """Done if actor reaches the max steps limit."""

        if actor_id not in measurements:
            return False

        return (
            measurements[actor_id].exp_info.step
            >= measurements[actor_id].exp_info.max_steps
        )

    def _done_reach_goal(actor_id: str, measurements: Measurements):
        """Done if actor reaches the goal."""

        if actor_id not in measurements:
            return False

        measurement = measurements[actor_id]
        if measurement.exp_info.next_command in [
            "REACH_GOAL",
            "PASS_GOAL",
        ]:
            return True

        if 0 <= measurement.exp_info.distance_to_goal_euclidean <= 1.0:
            return True

        return False

    def _done_pass_end(actor_id: str, measurements: Measurements):
        """Done if actor passes the end point."""

        if actor_id not in measurements:
            return False

        actor_m = measurements[actor_id]
        end_pos = actor_m.exp_info.end_pos
        end_transform = Transform(Location(*end_pos))

        relative_position = PathTracker.relative_position(
            actor_m.transform, end_transform, side_range=1e-6
        )
        return relative_position != "front"

    def _done_travel_distance(
        actor_id: str, measurements: Measurements, distance: float = 17
    ):
        """Done if actor travels the given distance threshold. Defaults to 17m."""

        if actor_id not in measurements:
            return False

        actor_m = measurements[actor_id]
        start_pos = actor_m.exp_info.start_pos[:3]
        curr_loc = actor_m.transform.location
        return curr_loc.distance(Location(*start_pos)) >= distance

    def _done_collision(actor_id: str, measurements: Measurements):
        """Done if actor collides with other actors."""

        if actor_id not in measurements:
            return False

        measurement = measurements[actor_id]
        return (
            measurement.collision.vehicles > 0
            or measurement.collision.pedestrians > 0
            or measurement.collision.others > 0
        )

    def _done_rollover(actor_id: str, measurements: Measurements):
        """Done if actor rolls over."""

        if actor_id not in measurements:
            return False

        measurement = measurements[actor_id]
        return abs(measurement.transform.rotation.roll) > 100

    def _done_offroad(
        actor_id: str,
        measurements: Measurements,
        threshold_times: float = 2.0,
        threshold_distance: float = None,
    ):
        """Done if actor drove off the planned route. Default threshold is 2 times the lane width."""

        if actor_id not in measurements:
            return False

        measurement = measurements[actor_id]
        threshold = threshold_distance or (
            threshold_times * measurement.planned_waypoint.lane_width
        )
        return abs(measurement.road_offset) > threshold

    def _done_all_npc(actor_id: str, measurements: Measurements):
        """Episode done if all NPC actors are done."""

        return all(
            m.exp_info.done
            for k, m in measurements.items()
            if k != "ego" and "static" not in m.type
        )

    Done.register_done("never", _done_never)
    Done.register_done("timeout", _done_timeout)
    Done.register_done("reach_goal", _done_reach_goal)
    Done.register_done("pass_end", _done_pass_end)
    Done.register_done("travel_distance", _done_travel_distance)
    Done.register_done("collision", _done_collision)
    Done.register_done("offroad", _done_offroad)
    Done.register_done("rollover", _done_rollover)
    Done.register_done("npc_done", _done_all_npc)


setup_done_strategy()

__all__ = ["Done"]
