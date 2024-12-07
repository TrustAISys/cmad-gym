"""
Author: Morphlng
Date: 2023-07-10 10:28:33
LastEditTime: 2023-11-23 11:03:16
LastEditors: Morphlng
Description: Demonstration of how to collect data for offline training (with MARLlib 1.0.3 and Ray 1.8.0)
"""

import os

from ray.rllib.agents import DefaultCallbacks
from ray.rllib.evaluation.sample_batch_builder import MultiAgentSampleBatchBuilder
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.offline.json_writer import JsonWriter
from ray.rllib.policy import Policy


class PseudoPolicy(Policy):
    def compute_actions(
        self,
        obs_batch,
        state_batches,
        prev_action_batch,
        prev_reward_batch,
        info_batch,
        episodes,
        explore,
        timestep,
        **kwargs
    ):
        pass


if __name__ == "__main__":
    using_marllib = True
    try:
        from marllib.envs.base_env import RLlibCmad

        env = RLlibCmad({"map_name": "Town01", "use_only_semantic": True})
        observation_space = env.observation_space
        action_space = env.action_space
    except ImportError as e:
        print(
            "Unable to use marllib, using cmad-gym instead. "
            "You have to postprocess the data yourself."
        )
        using_marllib = False
        from cmad.envs.multi_env import MultiCarlaEnv
        from cmad.misc.config import (
            gen_actor_config,
            gen_env_config,
            gen_scenario_config,
        )

        # Create a pseudo config
        actors = ["ego", "car1", "car2", "car3", "car4"]
        configs = {
            "scenarios": [gen_scenario_config("Town01", actors, 550)],
            "env": gen_env_config("127.0.0.1", 2000, "/Game/Carla/Maps/Town01"),
            "actors": {
                actor_id: gen_actor_config(
                    actor_id,
                    spawn=False,
                    camera_type="rgb",
                    render=False,
                    reward="ego" if actor_id == "ego" else "npc",
                )
                for actor_id in actors
            },
        }
        configs["actors"]["car4"]["type"] = "static_vehicle"
        env = MultiCarlaEnv(configs)

        # Sample is per agent, thus we only need the space for a single agent
        observation_space = next(iter(env.observation_space.values()))
        action_space = next(iter(env.action_space.values()))

    # RLlib uses preprocessors to implement transforms such as one-hot encoding
    # and flattening of tuple and dict observations. For CartPole a no-op
    # preprocessor is used, but this may be relevant for more complex envs.
    prep = get_preprocessor(observation_space)(observation_space)
    print("The preprocessor is", prep)

    # Prepare to collect samples
    batch_builder = MultiAgentSampleBatchBuilder(
        {"shared_policy": PseudoPolicy(observation_space, action_space, {})},
        False,
        DefaultCallbacks(),
    )
    writer = JsonWriter(os.path.join(os.path.dirname(__file__), "./offline_demo"))

    replay_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "./replays"))
    replay_files = [
        os.path.join(replay_root, f)
        for f in os.listdir(replay_root)
        if f.endswith(".dat")
    ]

    for replay in replay_files:
        raw_env = env.env if using_marllib else env
        obs, steps = raw_env.replay(replay, follow_vehicle="ego")
        prev_action = {
            k: 0 if raw_env._discrete_actions else [0, 0] for k in obs.keys()
        }
        prev_reward = {k: 0 for k in obs.keys()}
        done = {k: False for k in obs.keys()}
        done["__all__"] = False

        if using_marllib:
            obs, _, _, _ = env._process_return(obs)

        # RLlib requires only one "done entry" each episode each agent
        # We need to keep track of the first done entry
        first_done = done.copy()

        t = 0
        for ep in range(steps + 1):
            action_dict = raw_env.action_space.sample()
            new_obs, reward, done, info = raw_env.step(action_dict, replay=True)
            if using_marllib:
                new_obs, reward, done, info = env._process_return(
                    new_obs, reward, done, info
                )

            for idx, actor_id in enumerate(obs.keys()):
                if first_done[actor_id]:
                    continue

                batch_builder.add_values(
                    agent_id=actor_id,
                    policy_id="shared_policy",
                    t=t,
                    eps_id=raw_env.episode_id,
                    agent_index=idx,
                    obs=prep.transform(obs[actor_id]),
                    actions=action_dict[actor_id],
                    action_prob=1.0,  # put the true action probability here
                    action_logp=0.0,
                    rewards=reward[actor_id],
                    prev_actions=prev_action[actor_id],
                    prev_rewards=prev_reward[actor_id],
                    dones=done[actor_id],
                    new_obs=prep.transform(new_obs[actor_id]),
                )

                first_done[actor_id] = done[actor_id]

            obs = new_obs
            prev_action = action_dict
            prev_reward = reward
            t += 1

            if done["__all__"]:
                break

        # Save the data
        writer.write(batch_builder.build_and_reset())
        # reset the environment
        raw_env._clean_world()

    env.close()
