import gym
import gym.spaces

import cmad


def test_all_envs():
    for env_id in cmad._AVAILABLE_ENVS.keys():
        env = gym.make(env_id)
        assert isinstance(env.action_space, gym.spaces.Dict), (
            "Multi Actor/Agent environment should"
            "have Dict action space, one"
            "key-value pair per actor/agent"
        )
        assert isinstance(env.observation_space, gym.spaces.Dict), (
            "Multi Actor/Agent env should have "
            "Dict Obs space, one key-value pair"
            "per actor/agent"
        )
