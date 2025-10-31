from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class ReplayBuffer:
    capacity: int
    obs_shape: Tuple[int, ...]
    action_shape: Tuple[int, ...]
    device: str

    def __post_init__(self) -> None:
        self._pos = 0
        self._full = False
        self.observations = np.zeros((self.capacity, *self.obs_shape), dtype=np.float32)
        self.next_observations = np.zeros((self.capacity, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.capacity, *self.action_shape), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.terminateds = np.zeros((self.capacity,), dtype=np.float32)
        self.truncateds = np.zeros((self.capacity,), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        self.observations[self._pos] = obs
        self.actions[self._pos] = action
        self.rewards[self._pos] = reward
        self.next_observations[self._pos] = next_obs
        self.terminateds[self._pos] = float(terminated)
        self.truncateds[self._pos] = float(truncated)

        self._pos = (self._pos + 1) % self.capacity
        if self._pos == 0:
            self._full = True

    def __len__(self) -> int:
        return self.capacity if self._full else self._pos

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert len(self) >= batch_size, "Not enough samples in the replay buffer"
        indices = np.random.randint(0, len(self), size=batch_size)
        return (
            self.observations[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_observations[indices],
            self.terminateds[indices],
            self.truncateds[indices],
        )
