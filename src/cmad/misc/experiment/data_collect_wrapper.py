from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional

import gym
import numpy as np
import orjson as json
import pandas as pd

if TYPE_CHECKING:
    from cmad.envs.multi_env import MultiCarlaEnv

from cmad.envs.static_asset import SYS_ASSETS


class DataCollectWrapper(gym.Wrapper):
    """This wrapper will collect data from environment and save it to file episodically.

    Args:
        env (MultiCarlaEnv): Environment to wrap.
        save_folder (str, optional): Root folder to save data. Defaults to "~/carla_out"
        save_format (str, optional): Format of the saved data. Defaults to "npz".

    Note:
        - "npz" format is the recommended format since all data is in numpy format.
        - "csv" format is fast, but could be hard to load.
        - "json" format will be very slow if no third-party library is installed. We strongly suggest to install orjson or ujson when using this format.
    """

    def __init__(
        self,
        env: "MultiCarlaEnv",
        save_folder: Optional[str] = None,
        save_format: str = "npz",
    ):
        super(DataCollectWrapper, self).__init__(env)

        self.env = env
        self.actor_configs = env.actor_configs

        self._datalog_file_dict: dict[str, str] = {}
        self._datalog_buffer: defaultdict[str, dict[str, list]] = defaultdict(
            lambda: {"obs": [], "action": [], "reward": [], "done": []}
        )

        self._save_format = save_format
        self._exp_name = type(self.env).__name__
        self._timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        if save_folder is not None:
            self._save_folder = os.path.join(
                save_folder, self._exp_name, self._timestamp
            )
            os.makedirs(self._save_folder, exist_ok=True)
        else:
            self._save_folder = os.path.join(
                SYS_ASSETS.paths.output, self._exp_name, self._timestamp
            )
            os.makedirs(self._save_folder, exist_ok=True)

    def reset(self):
        obs_dict = self.env.reset()
        self._log_data(obs_dict)
        return obs_dict

    def replay(self, file: str, follow_vehicle: str = None):
        obs_dict, steps = self.env.replay(file, follow_vehicle)
        self._log_data(obs_dict)
        return obs_dict, steps

    def step(self, action_dict):
        obs_dict, reward_dict, done_dict, info_dict = self.env.step(action_dict)
        self._log_data(obs_dict, action_dict, reward_dict, done_dict)

        if done_dict["__all__"]:
            self._serialize()

        return obs_dict, reward_dict, done_dict, info_dict

    def _log_data(
        self,
        obs_dict: dict,
        action_dict: dict = {},
        reward_dict: dict = {},
        done_dict: dict = {},
    ):
        """Log data to buffer and file.

        Args:
            obs_dict (dict): Observation dict.
            action_dict (dict, optional): Action dict. Defaults to {}.
            reward_dict (dict, optional): Reward dict. Defaults to {}.
            done_dict (dict, optional): Done dict. Defaults to {}.
        """
        for actor_id in self.actor_configs.keys():
            if actor_id in self.env.background_actor_ids:
                continue

            if actor_id not in self._datalog_file_dict:
                self._datalog_file_dict[actor_id] = os.path.join(
                    self._save_folder,
                    f"episode{self.env.episode_id}_{actor_id}.{self._save_format}",
                )

            self._datalog_buffer[actor_id]["obs"].append(obs_dict.get(actor_id, None))
            self._datalog_buffer[actor_id]["action"].append(
                action_dict.get(actor_id, None)
            )
            self._datalog_buffer[actor_id]["reward"].append(
                reward_dict.get(actor_id, 0)
            )
            self._datalog_buffer[actor_id]["done"].append(
                done_dict.get(actor_id, False)
            )

    def _serialize(self):
        """Serialize the data to user-defined format."""

        # convert cached data to step-wise data
        stepwise_data: dict[str, list[dict]] = {}
        for actor_id in self._datalog_buffer:
            data_dict = self._datalog_buffer[actor_id]
            stepwise_data[actor_id] = [
                dict(zip(data_dict.keys(), step_data))
                for step_data in zip(*data_dict.values())
            ]

        if self._save_format == "json":
            self._json_serialize(stepwise_data)
        elif self._save_format == "csv":
            self._csv_serialize(stepwise_data)
        else:
            self._npz_serialize(stepwise_data)

        self._datalog_file_dict.clear()
        self._datalog_buffer.clear()

    def _json_serialize(self, data: dict[str, list[dict]]):
        """Serialize the data to json format."""
        for actor_id, log_file in self._datalog_file_dict.items():
            with open(log_file, "wb") as f:
                f.write(json.dumps(data[actor_id], option=json.OPT_SERIALIZE_NUMPY))

    def _npz_serialize(self, data: Dict[str, List[dict]]):
        """Serialize the data to npz format.

        Note:
            Using np.savez can speed up the serialization a lot. Yet, the file size is significantly larger.
        """
        for actor_id, log_file in self._datalog_file_dict.items():
            np.savez_compressed(log_file, data=data[actor_id])

    def _csv_serialize(self, data: dict[str, list[dict]]):
        """Serialize the data to csv format."""
        for actor_id, log_file in self._datalog_file_dict.items():
            df = pd.DataFrame(data[actor_id])
            df.to_csv(log_file, index=False)

    def _convert_numpy_to_python(self, data: dict):
        """Convert numpy data to pure python data."""
        converted_data = {}
        for key, item in data.items():
            if isinstance(item, list):
                converted_data[key] = list(map(self._convert_numpy_to_python, item))
            elif isinstance(item, dict):
                converted_data[key] = self._convert_numpy_to_python(item)
            elif isinstance(item, np.ndarray):
                converted_data[key] = item.tolist()
            elif isinstance(item, np.generic):
                converted_data[key] = item.item()
            else:
                converted_data[key] = item
        return converted_data
