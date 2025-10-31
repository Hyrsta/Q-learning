from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from gymnasium.wrappers import RecordVideo

from .envs import extract_observation_shape, make_env
from .networks import build_network
from .utils import load_checkpoint


ALGORITHMS: Sequence[str] = ("dqn", "double_dqn", "dueling_dqn")
ALGORITHM_LABELS: Dict[str, str] = {
    "dqn": "DQN",
    "double_dqn": "Double-DQN",
    "dueling_dqn": "Dueling-DQN",
}

ENVIRONMENT_IDS: Dict[str, str] = {
    "CartPole": "CartPole-v1",
    "LunarLander": "LunarLander-v3",
    "Breakout": "Breakout-v5",
    "Pong": "Pong-v5",
}


@dataclass
class VideoTask:
    display_name: str
    env_id: str
    algo: str
    checkpoint_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate qualitative videos for trained DQN variants",
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("videos"))
    parser.add_argument(
        "--env",
        dest="envs",
        action="append",
        choices=sorted(ENVIRONMENT_IDS.keys()),
        help="Restrict video generation to the specified environment(s)",
    )
    parser.add_argument(
        "--algo",
        dest="algorithms",
        action="append",
        choices=sorted(ALGORITHMS),
        help="Restrict video generation to the specified algorithm(s)",
    )
    parser.add_argument("--episodes", type=int, default=1, help="Episodes to record per video")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for rollouts")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device used for policy inference",
    )
    return parser.parse_args()


def discover_tasks(args: argparse.Namespace) -> List[VideoTask]:
    tasks: List[VideoTask] = []
    selected_envs: Iterable[str] = args.envs if args.envs else ENVIRONMENT_IDS.keys()
    selected_algorithms: Iterable[str] = (
        args.algorithms if args.algorithms else ALGORITHMS
    )

    for env_display in selected_envs:
        env_id = ENVIRONMENT_IDS[env_display]
        env_root = args.runs_dir / env_id

        for algo in selected_algorithms:
            checkpoint_path = env_root / algo / "model.pt"
            if not checkpoint_path.exists():
                print(
                    f"[skip] Missing checkpoint for {ALGORITHM_LABELS.get(algo, algo)} on {env_display}"
                    f" at {checkpoint_path}",
                )
                continue
            tasks.append(
                VideoTask(
                    display_name=env_display,
                    env_id=env_id,
                    algo=algo,
                    checkpoint_path=checkpoint_path,
                )
            )
    return tasks


def build_policy(
    checkpoint: Dict[str, object],
    env_id: str,
    algo: str,
    device: torch.device,
) -> torch.nn.Module:
    env = make_env(env_id, seed=0)
    obs_shape = extract_observation_shape(env)
    num_actions = env.action_space.n
    env.close()

    dueling = algo == "dueling_dqn"
    policy = build_network(obs_shape, num_actions, dueling=dueling).to(device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()
    return policy


def record_video(task: VideoTask, args: argparse.Namespace, device: torch.device) -> None:
    output_dir = args.output_dir / task.display_name
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(task.checkpoint_path, map_location=device)
    metadata = checkpoint.get("metadata", {})
    env_id = metadata.get("env_id", task.env_id)
    algo = metadata.get("algo", task.algo)

    policy = build_policy(checkpoint, env_id, algo, device)

    base_env = make_env(env_id, seed=args.seed, render_mode="rgb_array")
    env = RecordVideo(
        base_env,
        video_folder=str(output_dir),
        episode_trigger=lambda episode_id: episode_id < args.episodes,
        name_prefix=f"{task.display_name.lower()}_{algo}",
        disable_logger=True,
    )

    total_videos = 0
    for episode in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + episode)
        done = False
        while not done:
            obs_array = np.array(obs, copy=False)
            if obs_array.ndim == 3:
                obs_array = np.transpose(obs_array, (2, 0, 1))
            obs_tensor = torch.tensor(obs_array, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q_values = policy(obs_tensor)
            action = int(torch.argmax(q_values, dim=1).item())
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        total_videos += 1

    env.close()

    label = ALGORITHM_LABELS.get(algo, algo)
    print(
        f"[done] Saved {total_videos} video(s) for {label} on {task.display_name} to {output_dir}",
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tasks = discover_tasks(args)
    if not tasks:
        print("No checkpoints found. Nothing to render.")
        return

    for task in tasks:
        record_video(task, args, device)


if __name__ == "__main__":
    main()
