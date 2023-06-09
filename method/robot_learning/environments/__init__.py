"""
Provide helper functions to load environments.
"""

# OpenAI gym interface
import gym
import dmc2gym

from ..utils.logger import logger
from ..utils.gym_env import DictWrapper, FrameStackWrapper, GymWrapper, AbsorbingWrapper
from ..utils.subproc_vec_env import SubprocVecEnv


def make_env(name, config=None):
    """
    Creates a new environment instance with @name and @config.
    """
    # get default config if not provided
    if config is None:
        from ..config import argparser

        config, unparsed = argparser()

    return get_gym_env(name, config)


def get_gym_env(env_id, config):
    if env_id.startswith("dm."):
        # environment name of dm_control: dm.DOMAIN_NAME.TASK_NAME
        _, domain_name, task_name = env_id.split(".")
        env = dmc2gym.make(
            domain_name=domain_name,
            task_name=task_name,
            seed=config.seed,
            visualize_reward=False,
            from_pixels=(config.encoder_type == "cnn"),
            height=config.screen_height,
            width=config.screen_width,
            frame_skip=config.action_repeat,
            channels_first=True,
        )
    else:
        env_kwargs = config.__dict__.copy()
        try:
            env = gym.make(env_id, **env_kwargs)
        except Exception as e:
            logger.warning("Failed to launch an environment with config.")
            logger.warning(e)
            logger.warning("Launch an environment without config.")
            env = gym.make(env_id)
        env.seed(config.seed)
        env = GymWrapper(
            env=env,
            from_pixels=(config.encoder_type == "cnn"),
            height=config.screen_height,
            width=config.screen_width,
            channels_first=True,
            frame_skip=config.action_repeat,
            return_state=(config.encoder_type == "cnn" and config.asym_ac),
        )
    env = DictWrapper(
        env, return_state=(config.encoder_type == "cnn" and config.asym_ac)
    )
    # if config.encoder_type == "cnn":
    #     env = FrameStackWrapper(
    #         env,
    #         frame_stack=3,
    #         return_state=(config.encoder_type == "cnn" and config.asym_ac),
    #     )
    if config.absorbing_state:
        env = AbsorbingWrapper(env)

    return env


def make_vec_env(env_id, num_env, config=None, env_kwargs=None):
    """
    Creates a wrapped SubprocVecEnv using OpenAI gym interface.
    Unity app will use the port number from @config.port to (@config.port + @num_env - 1).

    Code modified based on
    https://github.com/openai/baselines/blob/master/baselines/common/cmd_util.py

    Args:
        env_id: environment id registered in in `env/__init__.py`.
        num_env: number of environments to launch.
        config: general configuration for the environment.
    """
    env_kwargs = env_kwargs or {}

    if config is not None:
        for key, value in config.__dict__.items():
            env_kwargs[key] = value

    def make_thunk(rank):
        new_env_kwargs = env_kwargs.copy()
        if "port" in new_env_kwargs:
            new_env_kwargs["port"] = env_kwargs["port"] + rank
        new_env_kwargs["seed"] = env_kwargs["seed"] + rank
        return lambda: get_gym_env(env_id, new_env_kwargs)

    return SubprocVecEnv([make_thunk(i) for i in range(num_env)])
