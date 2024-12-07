from __future__ import annotations

import functools
import logging
import os
import random
import shutil
import signal
import socket
import subprocess
import time
import weakref
from typing import TYPE_CHECKING, Tuple, Union

import GPUtil
import numpy as np

if TYPE_CHECKING:
    from cmad.envs.multi_env import MultiCarlaEnv

logger = logging.getLogger(__file__)


def change_logging_level(level: Union[str, int] = logging.INFO):
    """Change the logging level of the root logger.

    Args:
        level (str | int): logging level.
    """
    if isinstance(level, str):
        level = level.upper()

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    for handler in root_logger.handlers:
        handler.setLevel(level)


def weak_lru(maxsize: int = 128, typed: bool = False):
    """LRU Cache decorator that keeps a weak reference to 'self'"""

    def wrapper(func):
        @functools.lru_cache(maxsize, typed)
        def _func(_self, *args, **kwargs):
            return func(_self(), *args, **kwargs)

        @functools.wraps(func)
        def inner(self, *args, **kwargs):
            return _func(weakref.ref(self), *args, **kwargs)

        return inner

    return wrapper


def retry(
    exceptions: Union[Union[Exception, Tuple[Exception, ...]]],
    tries: int = -1,
    delay: float = 0,
    logger: logging.Logger = None,
):
    """Return a retry decorator

    Args:
        exceptions (Exception | tuple[Exception, ...]): exceptions to be caught.
        tries (int, optional): number of tries. Defaults to -1 (infinite).
        delay (float, optional): delay between tries. Defaults to 0.
        logger (logging.Logger, optional): logger to log the exceptions. Defaults to None.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _tries, _delay = tries, delay
            attempt = 0
            while _tries == -1 or attempt < _tries:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    if logger:
                        logger.warning(
                            "Attempt %d for %s failed with %s, %d tries left.",
                            attempt,
                            func.__name__,
                            e,
                            max(_tries - attempt, 0),
                        )

                    if _delay > 0:
                        time.sleep(_delay)

                    if _tries != -1 and attempt >= _tries:
                        raise  # Reraising the last exception if all tries are exhausted

        return wrapper

    return decorator


def get_attributes(
    obj: object, include_callable: bool = False, include_private: bool = False
):
    """Get all attributes of an object.

    Args:
        obj (object): object to be checked.
        include_callable (bool): whether to include callable attributes.
        include_private (bool): whether to include private attributes.

    Returns:
        dict: attributes of the object.
    """
    attrs = {}
    for attr in dir(obj):
        value = getattr(obj, attr)
        if not include_private and attr.startswith("_"):
            continue
        if not include_callable and callable(value):
            continue
        attrs[attr] = value
    return attrs


def one_hot_dict(ids_list: Union[list, set], first_id: str = None):
    """One-hot encoding for a list of ids.

    Args:
        ids_list (list | set): list of ids.
        first_id (str): rearrange the ids_list to make the first_id as the first element.

    Returns:
        dict: one-hot encoding dict.
    """
    ids_list = list(ids_list)

    if first_id is not None:
        try:
            ids_list.remove(first_id)
            ids_list.insert(0, first_id)
        except ValueError:
            pass

    one_hot_array = np.eye(len(ids_list))
    one_hot_dict = {}
    for i, id in enumerate(ids_list):
        one_hot_dict[id] = one_hot_array[i].tolist()
    return one_hot_dict


def flatten_dict(dicionary: dict):
    """Flatten a nested dict into a 1D array.

    Args:
        dicionary (dict): nested dict.

    Returns:
        np.ndarray: 1D array.
    """
    values = []
    for key in dicionary.keys():
        data = dicionary[key]
        if isinstance(data, dict):
            values.extend(flatten_dict(data))
        elif isinstance(data, (list, np.ndarray)):
            values.extend(np.array(data).flatten())
        else:
            values.append(data)

    return np.array(values, dtype=np.float32).flatten()


def get_tcp_port(port: int = 0) -> int:
    """
    Get a free tcp port number
    :param port: (default 0) port number. When `0` it will be assigned a free port dynamically
    :return: a port number requested if free otherwise an unhandled exception would be thrown
    """
    s = socket.socket()
    s.bind(("", port))
    server_port = s.getsockname()[1]
    s.close()
    return server_port


def start_carla_server(
    server_port: int = 2000, render: bool = False, x_res: int = 800, y_res: int = 600
) -> subprocess.Popen:
    """Start a carla server process.

    Args:
        server_port (int, optional): The rpc server port. Defaults to 2000.
        render (bool, optional): Whether to run Carla in headless mode or render mode. Defaults to False.
        x_res (int, optional): The window's x resolution. Defaults to 800.
        y_res (int, optional): The window's y resolution. Defaults to 600.

    Raises:
        FileNotFoundError: If the carla server executable is not found.

    Returns:
        subprocess.Popen: The carla server process.
    """
    from cmad import LOG_DIR
    from cmad.envs.static_asset import SYS_ASSETS

    if not os.path.exists(SYS_ASSETS.paths.executable):
        raise FileNotFoundError(
            "Make sure CARLA_SERVER environment"
            " variable is set & is pointing to the"
            " CARLA server startup script (Carla"
            "UE4.sh). Refer to the README file/docs."
        )

    process = None
    multigpu_success = False
    gpus = GPUtil.getGPUs()

    log_file = os.path.join(LOG_DIR, "server_" + str(server_port) + ".log")
    logging.info(
        "1. Server: localhost\n"
        f"2. Port: {server_port}\n"
        f"3. Binary: {SYS_ASSETS.paths.executable}"
    )

    if not render and (gpus is not None and len(gpus)) > 0:
        try:
            min_index = 0
            for i, gpu in enumerate(gpus):
                if gpu.load < gpus[min_index].load:
                    min_index = i
            # Check if vglrun is setup to launch sim on multipl GPUs
            if shutil.which("vglrun") is not None:
                process = subprocess.Popen(
                    (
                        "DISPLAY=:8 vglrun -d :7.{} {} -benchmark -fps=20"
                        " -carla-server -world-port={}"
                        " -carla-streaming-port=0".format(
                            min_index,
                            SYS_ASSETS.paths.executable,
                            server_port,
                        )
                    ),
                    shell=True,
                    # for Linux
                    preexec_fn=None if SYS_ASSETS.is_windows_platform else os.setsid,
                    # for Windows (not necessary)
                    creationflags=(
                        subprocess.CREATE_NEW_PROCESS_GROUP
                        if SYS_ASSETS.is_windows_platform
                        else 0
                    ),
                    stdout=open(log_file, "w"),
                )

            # Else, run in headless mode
            else:
                # Since carla 0.9.12+ use -RenderOffScreen to start headlessly
                # https://carla.readthedocs.io/en/latest/adv_rendering_options/
                process = subprocess.Popen(
                    (
                        '"{}" -RenderOffScreen -benchmark -fps=20 -carla-server'
                        " -world-port={} -carla-streaming-port=0".format(
                            SYS_ASSETS.paths.executable,
                            server_port,
                        )
                    ),
                    shell=True,
                    # for Linux
                    preexec_fn=None if SYS_ASSETS.is_windows_platform else os.setsid,
                    # for Windows (not necessary)
                    creationflags=(
                        subprocess.CREATE_NEW_PROCESS_GROUP
                        if SYS_ASSETS.is_windows_platform
                        else 0
                    ),
                    stdout=open(log_file, "w"),
                )
        except Exception as e:
            logger.error(e)

        if os.path.isfile(log_file):
            multigpu_success = True
        else:
            multigpu_success = False

        if multigpu_success:
            logger.info("Running sim servers in headless/multi-GPU mode")

    # Rendering mode and also a fallback if headless/multi-GPU doesn't work
    if multigpu_success is False:
        try:
            logger.info("Using single gpu to initialize carla server")

            process = subprocess.Popen(
                [
                    SYS_ASSETS.paths.executable,
                    "-windowed",
                    "-ResX=",
                    str(x_res),
                    "-ResY=",
                    str(y_res),
                    "-benchmark",
                    "-fps=20",
                    "-carla-server",
                    f"-carla-rpc-port={server_port}",
                    "-carla-streaming-port=0",
                ],
                # for Linux
                preexec_fn=None if SYS_ASSETS.is_windows_platform else os.setsid,
                # for Windows (not necessary)
                creationflags=(
                    subprocess.CREATE_NEW_PROCESS_GROUP
                    if SYS_ASSETS.is_windows_platform
                    else 0
                ),
                stdout=open(log_file, "w"),
            )
            logger.info("Running simulation in single-GPU mode")
        except Exception as e:
            logger.exception("FATAL ERROR while launching server:")

    # Wait for the server to be ready
    time.sleep(3)
    return process


def stop_carla_server(process: subprocess.Popen):
    from cmad.envs.static_asset import SYS_ASSETS

    logger.info("Killing live carla process: %s", process)
    if SYS_ASSETS.is_windows_platform:
        subprocess.call(["taskkill", "/F", "/T", "/PID", str(process.pid)])
    else:
        pgid = os.getpgid(process.pid)
        os.killpg(pgid, signal.SIGKILL)


def test_run(env: "MultiCarlaEnv", iteration: int = 2):
    """This is a function to test the environment.

    Args:
        env (MultiCarlaEnv): An instance of MultiCarlaEnv.
        iteration (int): Number of iterations to run.
    """
    for ep in range(iteration):
        obs = env.reset()

        total_reward_dict = {actor_id: 0 for actor_id in obs.keys()}
        action_dict = {actor_id: None for actor_id in obs.keys()}

        i = 0
        done = {"__all__": False}
        while not done["__all__"]:
            i += 1

            for actor_id in obs.keys():
                agent_action_space = env.action_space[actor_id]
                if "action_mask" in obs[actor_id]:
                    action_mask = obs[actor_id]["action_mask"]
                    if hasattr(agent_action_space, "nvec"):
                        # multi-discrete
                        action = []
                        nvec = agent_action_space.nvec
                        for j in range(len(nvec)):
                            lidx = 0 if j == 0 else nvec[j - 1]
                            ridx = nvec[j] + lidx
                            action.append(
                                random.choice(action_mask[lidx:ridx].nonzero()[0])
                            )
                        action_dict[actor_id] = action
                    else:
                        action_dict[actor_id] = random.choice(action_mask.nonzero()[0])
                else:
                    action_dict[actor_id] = agent_action_space.sample()

            obs, reward, done, info = env.step(action_dict)
            for actor_id in total_reward_dict.keys():
                total_reward_dict[actor_id] += reward[actor_id]

            print(
                ": {}\n\t".join(["Step#", "rew", "ep_rew", "done: {}"]).format(
                    i, reward, total_reward_dict, done
                )
            )

    env.close()
