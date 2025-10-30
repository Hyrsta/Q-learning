from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .envs import make_env, extract_observation_shape
from .networks import build_network
from .utils import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN agent")
    parser.add_argument("--env-id", type=str, required=True)
    parser.add_argument("--algo", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def evaluate() -> None:
    args = parse_args()
    device = torch.device(args.device)

    checkpoint = load_checkpoint(Path(args.checkpoint), map_location=device)
    metadata = checkpoint["metadata"]
    if args.algo != metadata["algo"]:
        raise ValueError(f"Checkpoint was trained with {metadata['algo']} but --algo was {args.algo}")

    env = make_env(args.env_id, seed=args.seed)
    obs_shape = extract_observation_shape(env)
    num_actions = env.action_space.n

    dueling = metadata["algo"] == "dueling_dqn"
    policy = build_network(obs_shape, num_actions, dueling=dueling).to(device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()

    returns = []
    for episode in range(args.episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            obs_array = np.array(obs, copy=False)
            if len(obs_array.shape) == 3:
                obs_array = np.transpose(obs_array, (2, 0, 1))
            obs_tensor = torch.tensor(obs_array, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q_values = policy(obs_tensor)
            action = int(torch.argmax(q_values, dim=1).item())
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        returns.append(total_reward)
        print(f"Episode {episode + 1}: return={total_reward:.2f}")
    env.close()
    print(f"Mean return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")


if __name__ == "__main__":
    evaluate()
