from __future__ import annotations

from typing import Tuple

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import FrameStack, RecordEpisodeStatistics, TransformObservation
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing


ATARI_ENVS = {"BreakoutNoFrameskip-v4", "PongNoFrameskip-v4"}


def make_env(env_id: str, seed: int, frame_stack: int = 4) -> gym.Env:
    env = gym.make(env_id)
    env.action_space.seed(seed)

    if env_id in ATARI_ENVS:
        env = AtariPreprocessing(env, grayscale_obs=True, frame_skip=4, screen_size=84, scale_obs=False)
        env = TransformObservation(env, lambda obs: np.array(obs, dtype=np.float32) / 255.0)
        env = FrameStack(env, frame_stack)

    env = RecordEpisodeStatistics(env)
    env.reset(seed=seed)
    return env


def extract_observation_shape(env: gym.Env) -> Tuple[int, ...]:
    obs_space = env.observation_space
    if isinstance(obs_space, gym.spaces.Box):
        shape = obs_space.shape
        if len(shape) == 3:
            # Convert channel-last to channel-first
            return (shape[2], shape[0], shape[1])
        return shape
    if isinstance(obs_space, gym.spaces.Discrete):
        return (1,)
    raise ValueError(f"Unsupported observation space: {obs_space}")
