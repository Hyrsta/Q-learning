from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def _cnn_block(in_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU(),
    )


def _mlp_block(input_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
    )


class QNetwork(nn.Module):
    def __init__(self, observation_shape: Tuple[int, ...], num_actions: int) -> None:
        super().__init__()
        self.is_image = len(observation_shape) == 3
        self.channel_last = False

        if self.is_image:
            c, h, w = observation_shape
            if c > observation_shape[-1]:
                # Handle environments that expose channel-last observations
                c, h, w = observation_shape[2], observation_shape[0], observation_shape[1]
                self.channel_last = True
            self.features = _cnn_block(c)
            with torch.no_grad():
                n_flatten = self.features(torch.zeros(1, c, h, w)).view(1, -1).size(1)
            self.head = nn.Sequential(nn.Linear(n_flatten, 512), nn.ReLU(), nn.Linear(512, num_actions))
        else:
            input_dim = int(torch.tensor(observation_shape).prod().item())
            self.features = _mlp_block(input_dim)
            self.head = nn.Linear(256, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_image and self.channel_last:
            x = x.permute(0, 3, 1, 2)
        if not self.is_image:
            x = x.view(x.size(0), -1)
        features = self.features(x)
        if self.is_image:
            features = features.view(features.size(0), -1)
        return self.head(features)


class DuelingQNetwork(nn.Module):
    def __init__(self, observation_shape: Tuple[int, ...], num_actions: int) -> None:
        super().__init__()
        self.is_image = len(observation_shape) == 3
        self.channel_last = False

        if self.is_image:
            c, h, w = observation_shape
            if c > observation_shape[-1]:
                c, h, w = observation_shape[2], observation_shape[0], observation_shape[1]
                self.channel_last = True
            self.features = _cnn_block(c)
            with torch.no_grad():
                n_flatten = self.features(torch.zeros(1, c, h, w)).view(1, -1).size(1)
            feature_dim = n_flatten
        else:
            input_dim = int(torch.tensor(observation_shape).prod().item())
            self.features = _mlp_block(input_dim)
            feature_dim = 256

        self.value_stream = nn.Sequential(nn.Linear(feature_dim, 512), nn.ReLU(), nn.Linear(512, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(feature_dim, 512), nn.ReLU(), nn.Linear(512, num_actions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_image and self.channel_last:
            x = x.permute(0, 3, 1, 2)
        if not self.is_image:
            x = x.view(x.size(0), -1)
        features = self.features(x)
        if self.is_image:
            features = features.view(features.size(0), -1)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_values = value + advantages - advantages.mean(dim=1, keepdim=True)
        return q_values


def build_network(observation_shape: Tuple[int, ...], num_actions: int, dueling: bool) -> nn.Module:
    if dueling:
        return DuelingQNetwork(observation_shape, num_actions)
    return QNetwork(observation_shape, num_actions)
