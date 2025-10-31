from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(state: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)


def polyak_update(model: torch.nn.Module, target_model: torch.nn.Module, tau: float) -> None:
    """Perform Polyak averaging of parameters.

    Args:
        model: The source model providing the new parameters.
        target_model: The target model to be updated.
        tau: Interpolation factor. ``tau=1.0`` copies the parameters
            directly, while lower values perform a soft update.
    """

    if not 0.0 < tau <= 1.0:
        raise ValueError(f"tau must be in (0, 1], got {tau}")

    with torch.no_grad():
        for param, target_param in zip(model.parameters(), target_model.parameters()):
            target_param.data.lerp_(param.data, tau)
