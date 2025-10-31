from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple

import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

import numpy as np
from gymnasium.wrappers import (
    RecordEpisodeStatistics,
    RecordVideo,
    TransformObservation,
)

from gymnasium.wrappers import FrameStackObservation as FrameStack
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing


ATARI_PREFIX = "ALE/"

ATARI_ENVS = {
    "Breakout-v5",
    "Pong-v5",
}


def make_env(
    env_id: str,
    seed: int,
    frame_stack: int = 4,
    *,
    render_mode: Optional[str] = None,
    video_dir: Optional[Path | str] = None,
    video_trigger: Optional[Callable[[int], bool]] = None,
    video_prefix: str = "rl-video",
) -> gym.Env:
    
    if env_id in ATARI_ENVS:
        env = gym.make(f"{ATARI_PREFIX}{env_id}", render_mode=render_mode, frameskip=1)
        env.action_space.seed(seed)
        env = AtariPreprocessing(env, grayscale_obs=True, frame_skip=4, screen_size=84, scale_obs=False)
        env = TransformObservation(env, lambda obs: np.array(obs, dtype=np.float32) / 255.0)
        env = FrameStack(env, frame_stack)
    else:
        env = gym.make(env_id, render_mode=render_mode)
        env.action_space.seed(seed)

    if video_dir is not None:
        if render_mode not in {"rgb_array", "rgb_array_list"}:
            raise ValueError("Recording video requires render_mode='rgb_array' or 'rgb_array_list'")
        video_path = Path(video_dir)
        video_path.mkdir(parents=True, exist_ok=True)
        trigger = video_trigger if video_trigger is not None else (lambda episode_id: True)
        env = RecordVideo(
            env,
            video_folder=str(video_path),
            episode_trigger=trigger,
            name_prefix=video_prefix,
            disable_logger=True,
        )

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
