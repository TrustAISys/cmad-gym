"""
Author: Morphlng
Date: 2023-11-22 09:47:59
LastEditTime: 2023-11-22 22:55:37
LastEditors: Morphlng
Description: Demonstration of how to collect data during running
"""

from cmad.envs.example import Town01Sim
from cmad.misc.experiment import DataCollectWrapper

if __name__ == "__main__":
    env = DataCollectWrapper(Town01Sim(), save_format="npz")

    for _ in range(3):
        env.reset()

        done = {"__all__": False}

        while not done["__all__"]:
            obs, reward, done, info = env.step(env.action_space.sample())

    env.close()
