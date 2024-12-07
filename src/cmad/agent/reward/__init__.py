from cmad.agent.reward.reward import Reward, RewardState
from cmad.agent.reward.reward_logger import RewardLogger


def setup_reward():
    import numpy as np

    def calculate_distance(reward_state: RewardState, ego_id: str, actor_id: str):
        actor_loc = reward_state.current_measurements[actor_id].transform.location
        ego_transform = reward_state.current_measurements[ego_id].transform
        ego_bbox = reward_state.current_measurements[ego_id].bounding_box
        ego_world_bb = ego_bbox.transform_to_world_frame(ego_transform, bottom_only=True)
        return actor_loc.distance_to_polygon(ego_world_bb)

    def compute_collision_potential_reward(
        reward_state: RewardState,
        ego_id: str,
        actor_id: str,
        min_distance_with_ego_threshold: float,
        total_ego_collision_potential_reward: float,
    ):
        ego_collision_reward = 0
        actor_cache = reward_state.cache[actor_id]
        if ego_id in reward_state.current_measurements:
            distance_with_ego = calculate_distance(reward_state, ego_id, actor_id)

            if "distance_with_ego_cache" not in actor_cache:
                actor_cache["distance_with_ego_cache"] = min_distance_with_ego_threshold
            ego_collision_reward += (
                total_ego_collision_potential_reward
                * 0.5
                * (
                    (
                        (min_distance_with_ego_threshold - distance_with_ego)
                        / min_distance_with_ego_threshold
                    )
                    ** 2
                    - (
                        (
                            min_distance_with_ego_threshold
                            - actor_cache["distance_with_ego_cache"]
                        )
                        / min_distance_with_ego_threshold
                    )
                    ** 2
                )
                if distance_with_ego < actor_cache["distance_with_ego_cache"]
                else 0
            )
            actor_cache["distance_with_ego_cache"] = min(
                distance_with_ego,
                actor_cache["distance_with_ego_cache"],
            )
        return ego_collision_reward

    @Reward.reward_signature
    def compute_reward_ego(actor_id: str, reward_state: RewardState):
        """Reward function based on Ego vehicle's measurement

        Note: This reward will be directly plused to the reward of other NPC's reward
        """
        reward = 0.0
        actor_cache = reward_state.cache[actor_id]
        is_active = actor_cache["active_state"]
        curr_m = reward_state.current_measurements[actor_id]
        prev_m = reward_state.prev_measurements[actor_id]

        if is_active:
            # Ego timeout reward
            if curr_m.exp_info.step >= curr_m.exp_info.max_steps and (
                not curr_m.exp_info.done
            ):
                reward += 100.0

            # Ego offroad reward
            if curr_m.road_offset > curr_m.waypoint.lane_width * 2:
                reward += 1.0

            # Ego arrived penalty
            if curr_m.exp_info.next_command in ["REACH_GOAL", "PASS_GOAL"]:
                reward -= 10.0

            # Ego Collision reward
            if curr_m.collision.diff(prev_m.collision) > 0:
                reward += 100.0

        actor_cache["active_state"] = not curr_m.exp_info.done
        return reward

    @Reward.reward_signature
    def compute_reward_corl2017(actor_id: str, reward_state: RewardState):
        reward = 0.0
        actor_cache = reward_state.cache[actor_id]
        is_active = actor_cache["active_state"]
        curr_m = reward_state.current_measurements[actor_id]
        prev_m = reward_state.prev_measurements[actor_id]

        if is_active:
            cur_dist = curr_m.exp_info.distance_to_goal
            prev_dist = prev_m.exp_info.distance_to_goal
            # Distance travelled toward the goal in m
            reward += np.clip(prev_dist - cur_dist, -10.0, 10.0)
            # Change in speed (km/h)
            reward += 0.05 * (curr_m.speed - prev_m.speed)
            # New collision damage
            # TODO: This can't be right
            reward -= 0.00002 * curr_m.collision.diff(prev_m.collision)

            # New sidewalk intersection
            reward -= 2 * (curr_m.lane_invasion.offroad - prev_m.lane_invasion.offroad)

            # New opposite lane intersection
            reward -= 2 * (
                curr_m.lane_invasion.otherlane - prev_m.lane_invasion.otherlane
            )

        actor_cache["active_state"] = not curr_m.exp_info.done
        return reward

    @Reward.reward_signature
    def compute_reward_lane_keep(actor_id: str, reward_state: RewardState):
        reward = 0.0
        actor_cache = reward_state.cache[actor_id]
        is_active = actor_cache["active_state"]
        curr_m = reward_state.current_measurements[actor_id]
        prev_m = reward_state.prev_measurements[actor_id]

        if is_active:
            # Speed reward, up 30.0 (km/h)
            reward += np.clip(curr_m.speed, 0.0, 30.0) / 10
            # New collision damage
            if curr_m.collision.diff(prev_m.collision) > 0:
                reward -= 100.0
            # Sidewalk intersection
            reward -= curr_m.lane_invasion.offroad
            # Opposite lane intersection
            reward -= curr_m.lane_invasion.otherlane

        actor_cache["active_state"] = not curr_m.exp_info.done
        return reward

    @Reward.reward_signature
    def compute_reward_npc(actor_id: str, reward_state: RewardState):
        step_reward = 0.0
        actor_cache = reward_state.cache[actor_id]
        is_active = actor_cache["active_state"]
        curr_m = reward_state.current_measurements[actor_id]
        prev_m = reward_state.prev_measurements[actor_id]

        if is_active:
            # SECTION 1: Continuous reward
            # 1. Lane keeping reward
            waypoint_reward = 0
            distance_to_goal = curr_m.exp_info.distance_to_goal

            if "distance_to_goal_cache" not in actor_cache:
                actor_cache["distance_to_goal_cache"] = distance_to_goal
                actor_cache["min_distance_to_goal_cache"] = distance_to_goal

            if distance_to_goal < actor_cache["min_distance_to_goal_cache"]:
                waypoint_reward += (
                    200
                    * (actor_cache["min_distance_to_goal_cache"] - distance_to_goal)
                    / actor_cache["distance_to_goal_cache"]
                )
                actor_cache["min_distance_to_goal_cache"] = distance_to_goal
            step_reward += waypoint_reward

            # 2. Keeping speed reward
            # Speed diff to target_speed (m/s)
            current_speed, target_speed = (
                curr_m.speed,
                curr_m.exp_info.target_speed,
            )
            # The v_min stand for the least speed you need to finish the episode
            v_min = curr_m.exp_info.distance_to_goal / (
                max(curr_m.exp_info.max_steps - curr_m.exp_info.step, 1)
                * curr_m.exp_info.step_time
            )

            if v_min < target_speed:
                keep_speed_reward = np.clip(
                    1 - abs(current_speed - target_speed) / (0.5 * target_speed), -1, 1
                )
            else:
                keep_speed_reward = np.clip(current_speed - v_min, -1, 0)
            keep_speed_reward *= 3.0
            step_reward += keep_speed_reward

            # 3. Orientation reward
            steer_reward = 0
            abs_diff = abs(curr_m.orientation_diff)
            if abs_diff > 3:
                steer_reward -= 1
            else:
                steer_reward += 1 - abs_diff / 3

            change_in_orientation = abs(
                curr_m.orientation_diff - prev_m.orientation_diff
            )
            if change_in_orientation > 2:
                steer_reward -= 1
            step_reward += steer_reward

            # SECTION 2: Discrete reward
            # 4. Ego Collision reward/penalty
            collision_reward = 0
            ego_id = curr_m.exp_info.actor_in_scene.get("ego", -1)
            if ego_id != -1:
                distance_with_ego = calculate_distance(reward_state, "ego", actor_id)
                min_distance_with_ego_threshold = 20

                if "distance_with_ego_cache" not in actor_cache:
                    actor_cache[
                        "distance_with_ego_cache"
                    ] = min_distance_with_ego_threshold

                collision_reward += (
                    200
                    * 0.5
                    * (
                        (
                            (min_distance_with_ego_threshold - distance_with_ego)
                            / min_distance_with_ego_threshold
                        )
                        ** 2
                        - (
                            (
                                min_distance_with_ego_threshold
                                - actor_cache["distance_with_ego_cache"]
                            )
                            / min_distance_with_ego_threshold
                        )
                        ** 2
                    )
                    if distance_with_ego < actor_cache["distance_with_ego_cache"]
                    else 0
                )
                actor_cache["distance_with_ego_cache"] = min(
                    distance_with_ego,
                    actor_cache["distance_with_ego_cache"],
                )

            if curr_m.collision.vehicles - prev_m.collision.vehicles > 0:
                if ego_id not in curr_m.collision.id_set:
                    collision_reward -= 100.0

            if curr_m.collision.diff(prev_m.collision, check_vehicle=False) > 0:
                collision_reward -= 100.0
            step_reward += collision_reward

            # 5. Sidewalk intersection penalty
            offroad_reward = -10 * (
                curr_m.lane_invasion.offroad - prev_m.lane_invasion.offroad
            )
            step_reward += offroad_reward

            # 6. Opposite lane intersection penalty
            opp_lane_reward = -10 * (
                curr_m.lane_invasion.otherlane - prev_m.lane_invasion.otherlane
            )
            step_reward += opp_lane_reward

        log_data = Reward.capture_reward_detail(locals())
        log_data["episode_id"] = curr_m.exp_info.episode_id
        RewardLogger.log(actor_id, log_data)

        actor_cache["active_state"] = not curr_m.exp_info.done
        return step_reward

    @Reward.reward_signature
    def compute_reward_vehicle_atomic(
        actor_id: str, reward_state: RewardState, **kwargs
    ):
        constant = {
            "total_waypoint_reward": 150,
            "max_speed_coefficient": 0.4,
            "max_speed_penalty_range": 0.2,
            "constant_npc_collision_penalty": 50,
            "total_ego_collision_potential_reward": 250,
            "min_distance_with_ego_threshold": 20,
        }
        constant.update(kwargs)

        actor_cache = reward_state.cache[actor_id]
        is_active = actor_cache["active_state"]
        curr_m = reward_state.current_measurements[actor_id]
        prev_m = reward_state.prev_measurements[actor_id]

        reward = 0.0
        if is_active:
            # waypoint potential reward
            waypoint_reward = 0
            distance_to_goal = curr_m.exp_info.distance_to_goal
            if "distance_to_goal_cache" not in actor_cache:
                actor_cache["distance_to_goal_cache"] = distance_to_goal
                actor_cache["min_distance_to_goal_cache"] = distance_to_goal

            if distance_to_goal < actor_cache["min_distance_to_goal_cache"]:
                waypoint_reward += (
                    constant["total_waypoint_reward"]
                    * (actor_cache["min_distance_to_goal_cache"] - distance_to_goal)
                    / max(0.1, actor_cache["distance_to_goal_cache"])
                )
                actor_cache["min_distance_to_goal_cache"] = distance_to_goal
            reward += waypoint_reward

            # max speed penalty
            max_speed_reward = 0
            # calculate max_speed_penalty_coefficient
            if "max_speed_penalty_coefficient_cache" not in actor_cache:
                max_speed = actor_cache["distance_to_goal_cache"] / (
                    constant["max_speed_coefficient"] * curr_m.exp_info.max_steps
                )
                # Assuming a penalty coefficient 'a'
                # penalty for exceeding the speed beyond the 'penalty_range' can negate the reward from waypoints.
                # 'a' is calculated based on the equation
                # penalty_range*max_speed*a = (total_waypoint_reward / distance_to_goal)*max_speed.

                max_speed_penalty_coefficient = (
                    constant["max_speed_penalty_range"]
                    * constant["total_waypoint_reward"]
                    / max(0.1, actor_cache["distance_to_goal_cache"])
                )
                actor_cache["max_speed"] = max_speed
                actor_cache[
                    "max_speed_penalty_coefficient_cache"
                ] = max_speed_penalty_coefficient
            else:
                max_speed_reward -= actor_cache[
                    "max_speed_penalty_coefficient_cache"
                ] * max(0, curr_m.speed - actor_cache["max_speed"])
            reward += max_speed_reward

            # ego collision reward
            ego_collision_reward = 0
            npc_collision_reward = 0

            ego_collision_reward = compute_collision_potential_reward(
                reward_state,
                "ego",
                actor_id,
                constant["min_distance_with_ego_threshold"],
                constant["total_ego_collision_potential_reward"],
            )
            reward += ego_collision_reward

            # npc collision reward
            ego_id = curr_m.exp_info.actor_in_scene.get("ego", -1)
            if curr_m.collision.vehicles - prev_m.collision.vehicles > 0:
                if ego_id not in curr_m.collision.id_set:
                    npc_collision_reward -= constant["constant_npc_collision_penalty"]
            if curr_m.collision.diff(prev_m.collision, check_vehicle=False) > 0:
                npc_collision_reward -= constant["constant_npc_collision_penalty"]
            reward += npc_collision_reward

        log_data = Reward.capture_reward_detail(locals())
        log_data["episode_id"] = curr_m.exp_info.episode_id
        RewardLogger.log(actor_id, log_data)

        actor_cache["active_state"] = not curr_m.exp_info.done
        return reward

    @Reward.reward_signature
    def compute_reward_walker(actor_id: str, reward_state: RewardState, **kwargs):
        constant = {
            "constant_npc_collision_penalty": 50,
            "constant_ego_collision_penalty": 25,
            "total_ego_collision_potential_reward": 200,
            "min_distance_with_ego_threshold": 20,
        }
        constant.update(kwargs)

        actor_cache = reward_state.cache[actor_id]
        is_active = actor_cache["active_state"]
        curr_m = reward_state.current_measurements[actor_id]
        prev_m = reward_state.prev_measurements[actor_id]

        reward = 0.0
        if is_active:
            # ego collision reward
            npc_collision_reward = 0

            ego_collision_reward = compute_collision_potential_reward(
                reward_state,
                "ego",
                actor_id,
                constant["min_distance_with_ego_threshold"],
                constant["total_ego_collision_potential_reward"],
            )
            reward += ego_collision_reward

            # npc collision reward
            ego_id = curr_m.exp_info.actor_in_scene.get("ego", -1)
            if curr_m.collision.vehicles - prev_m.collision.vehicles > 0:
                if ego_id in curr_m.collision.id_set:
                    npc_collision_reward -= constant["constant_ego_collision_penalty"]
                else:
                    npc_collision_reward -= constant["constant_npc_collision_penalty"]

            if curr_m.collision.diff(prev_m.collision, check_vehicle=False) > 0:
                npc_collision_reward -= constant["constant_npc_collision_penalty"]
            reward += npc_collision_reward

        log_data = Reward.capture_reward_detail(locals())
        log_data["episode_id"] = curr_m.exp_info.episode_id
        RewardLogger.log(actor_id, log_data)

        actor_cache["active_state"] = not curr_m.exp_info.done
        return reward

    @Reward.reward_signature
    def compute_reward_custom(actor_id: str, reward_state: RewardState):
        """Demonstrtation of how to use the reward function

        Args:
            reward_state (Reward): Reward object

        Returns:
            float: reward of the single step
        """
        reward = 0.0
        actor_cache = reward_state.cache[actor_id]
        is_active = actor_cache["active_state"]
        curr_m = reward_state.current_measurements[actor_id]
        prev_m = reward_state.prev_measurements[actor_id]

        if is_active:
            # Potential function - reward for distance to goal,200 is the max reward
            waypoint_reward = 0
            distance_to_goal = curr_m.exp_info.distance_to_goal

            if "distance_to_goal_cache" not in actor_cache:
                actor_cache["distance_to_goal_cache"] = distance_to_goal
                actor_cache["min_distance_to_goal_cache"] = distance_to_goal

            if distance_to_goal < actor_cache["min_distance_to_goal_cache"]:
                waypoint_reward += (
                    200
                    * (actor_cache["min_distance_to_goal_cache"] - distance_to_goal)
                    / actor_cache["distance_to_goal_cache"]
                )
                actor_cache["min_distance_to_goal_cache"] = distance_to_goal
            reward += waypoint_reward

            # max speed - penalty for speed > 6m/s
            max_speed_reward = 0
            current_speed = curr_m.speed
            max_speed_reward -= 3.5 * max(0, current_speed - 6)
            reward += max_speed_reward

            # Orientation  - penalty for orientation diff > 3 and change in orientation > 2
            steer_reward = 0
            abs_diff = abs(curr_m.orientation_diff)
            if abs_diff > 3:
                steer_reward -= 1
            change_in_orientation = abs(
                curr_m.orientation_diff - prev_m.orientation_diff
            )
            if change_in_orientation > 2:
                steer_reward -= 1
            reward += steer_reward

            # offset - pentalty for offset > (lane_width - car_width) / 2
            offset_reward = 0
            offset_reward -= 0.5 * max(
                0,
                curr_m.road_offset
                - (curr_m.waypoint.lane_width / 2 - curr_m.bounding_box.extent.y),
            )
            reward += offset_reward

            # Collision - penalty for collision with vehicles and pedestrians
            collision_reward = 0
            if (
                curr_m.collision.pedestrians
                + curr_m.collision.others
                - prev_m.collision.pedestrians
                - prev_m.collision.others
            ) > 0:
                collision_reward -= 100.0
            reward += collision_reward

        log_data = Reward.capture_reward_detail(locals())
        log_data["episode_id"] = curr_m.exp_info.episode_id
        RewardLogger.log(actor_id, log_data)

        actor_cache["active_state"] = not curr_m.exp_info.done
        return reward

    Reward.register_reward("corl2017", compute_reward_corl2017)
    Reward.register_reward("lane_keep", compute_reward_lane_keep)
    Reward.register_reward("ego", compute_reward_ego)
    Reward.register_reward("npc", compute_reward_npc)
    Reward.register_reward("npc_vehicle", compute_reward_vehicle_atomic)
    Reward.register_reward("npc_walker", compute_reward_walker)
    Reward.register_reward("custom", compute_reward_custom)


setup_reward()


__all__ = ["Reward", "RewardLogger"]
