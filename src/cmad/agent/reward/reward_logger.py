from __future__ import annotations

import os
import time
from collections import defaultdict

try:
    import pandas as pd

    LOG_AVAILABLE = True
except ImportError as e:
    import logging

    logger = logging.getLogger(__name__)
    logger.warning("Pandas not installed, can't log reward data")
    LOG_AVAILABLE = False


class RewardLogger:
    """A static class to log the reward data to csv file.

    The only method you need to call is RewardLogger.log(actor_id, log_data).

    - log_buffer (defaultdict): A dict to store the log data.

        Note: The log_buffer will not be reset after each episode, but after each log_steps.
    """

    _init = False
    LOG_ON = LOG_AVAILABLE

    @staticmethod
    def init(enable_logging: bool = False, log_steps: int = 10000):
        RewardLogger.LOG_ON = enable_logging
        RewardLogger.LOG_STPES = log_steps
        RewardLogger.TIME_STAMP = time.strftime("%Y%m%d-%H_%M_%S", time.localtime())
        RewardLogger.log_buffer = defaultdict(lambda: defaultdict(list))
        RewardLogger._init = True

    @staticmethod
    def log(actor_id: str, log_data: dict):
        if not RewardLogger._init:
            RewardLogger.init(enable_logging=LOG_AVAILABLE)

        if not RewardLogger.LOG_ON:
            return

        buffer = RewardLogger.log_buffer[actor_id]
        for key, value in log_data.items():
            buffer[key].append(value)

        if (
            len(buffer.get("episode_id", next(iter(buffer.values()), [])))
            >= RewardLogger.LOG_STPES
        ):
            RewardLogger.save_to_file(actor_id, buffer)
            RewardLogger.log_buffer[actor_id].clear()

    @staticmethod
    def save_to_file(actor_id: str, reward_buffer: "dict[str, list]"):
        log_dataframe = pd.DataFrame(reward_buffer)
        log_path = os.path.expanduser(
            f"~/reward_log/{RewardLogger.TIME_STAMP}_{actor_id}.csv"
        )
        if not os.path.exists(log_path):
            reward_dir = os.path.dirname(log_path)
            if not os.path.exists(reward_dir):
                os.makedirs(reward_dir)

            pd.DataFrame(columns=log_dataframe.columns).to_csv(
                log_path, header=True, index=False
            )

        log_dataframe.to_csv(log_path, mode="a", header=False, index=False)
