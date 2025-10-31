from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn.functional as F

from .buffers import ReplayBuffer
from .envs import extract_observation_shape, make_env
from .networks import build_network
from .schedules import linear_schedule
from .utils import polyak_update, save_checkpoint, set_seed


ALGORITHMS = {"dqn", "double_dqn", "dueling_dqn"}


def _to_serializable(value: Any) -> Any:
    if isinstance(value, (np.ndarray, np.generic)):
        array_value = np.asarray(value)
        if array_value.shape == ():
            return array_value.item()
        if array_value.size == 1:
            return array_value.reshape(()).item()
        return array_value.tolist()
    return value


def get_default_hyperparameters(env_id: str) -> Dict[str, Any]:
    if env_id == "CartPole-v1":
        return dict(
            total_timesteps=100_000,
            learning_rate=5e-4,
            buffer_size=50_000,
            batch_size=64,
            gamma=0.99,
            train_frequency=1,
            gradient_steps=1,
            target_update_interval=1_000,
            target_update_tau= 0.01,
            learning_starts=1_000,
            exploration_fraction=0.2,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
        )
    if env_id == "LunarLander-v2":
        return dict(
            total_timesteps=500_000,
            learning_rate=5e-4,
            buffer_size=200_000,
            batch_size=128,
            gamma=0.99,
            train_frequency=4,
            gradient_steps=1,
            target_update_interval=4_000,
            target_update_tau=0.005,
            learning_starts=10_000,
            exploration_fraction=0.4,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.02,
        )
    if env_id in {"BreakoutNoFrameskip-v4", "PongNoFrameskip-v4"}:
        return dict(
            total_timesteps=10_000_000,
            learning_rate=1e-4,
            buffer_size=1_000_000,
            batch_size=32,
            gamma=0.99,
            train_frequency=4,
            gradient_steps=1,
            target_update_interval=10_000,
            target_update_tau=0.001,
            learning_starts=50_000,
            exploration_fraction=0.1,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.01,
        )
    raise ValueError(f"Unsupported environment id: {env_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN variants")
    parser.add_argument("--env-id", type=str, required=True, help="Gymnasium environment id")
    parser.add_argument(
        "--algo",
        type=str,
        choices=sorted(ALGORITHMS),
        default="dqn",
        help="Which algorithm variant to train",
    )
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--buffer-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--train-frequency", type=int, default=None)
    parser.add_argument("--gradient-steps", type=int, default=None)
    parser.add_argument("--target-update-interval", type=int, default=None)
    parser.add_argument("--target-update-tau", type=float, default=None)
    parser.add_argument("--learning-starts", type=int, default=None)
    parser.add_argument("--exploration-fraction", type=float, default=None)
    parser.add_argument("--exploration-initial-eps", type=float, default=None)
    parser.add_argument("--exploration-final-eps", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", type=str, default="runs")
    parser.add_argument("--eval-frequency", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument(
        "--loss",
        type=str,
        choices={"mse", "huber"},
        default="huber",
        help="Loss function to optimize.",
    )
    return parser.parse_args()


def evaluate(agent: torch.nn.Module, env_id: str, seed: int, episodes: int, device: torch.device) -> Dict[str, float]:
    env = make_env(env_id, seed=seed + 10_000)
    agent.eval()
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            obs_array = np.array(obs, copy=False)
            if len(obs_array.shape) == 3:
                obs_array = np.transpose(obs_array, (2, 0, 1))
            obs_tensor = torch.tensor(obs_array, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q_values = agent(obs_tensor)
            action = int(torch.argmax(q_values, dim=1).item())
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        returns.append(total_reward)
    env.close()
    agent.train()
    return {"mean_return": float(np.mean(returns)), "std_return": float(np.std(returns))}


def train() -> None:
    args = parse_args()
    set_seed(args.seed)

    env = make_env(args.env_id, seed=args.seed)
    obs_shape = extract_observation_shape(env)
    num_actions = env.action_space.n

    defaults = get_default_hyperparameters(args.env_id)
    hyperparams = {key: getattr(args, key) if getattr(args, key) is not None else defaults[key] for key in defaults}

    if args.eval_frequency is None:
        args.eval_frequency = max(1, hyperparams["total_timesteps"] // 100)

    device = torch.device(args.device)
    dueling = args.algo == "dueling_dqn"
    use_double = args.algo in {"double_dqn", "dueling_dqn"}

    policy = build_network(obs_shape, num_actions, dueling=dueling).to(device)
    target_policy = build_network(obs_shape, num_actions, dueling=dueling).to(device)
    target_policy.load_state_dict(policy.state_dict())

    optimizer = torch.optim.Adam(policy.parameters(), lr=hyperparams["learning_rate"])

    if args.loss == "huber":
        loss_fn = F.smooth_l1_loss
    else:
        loss_fn = F.mse_loss

    buffer = ReplayBuffer(
        capacity=hyperparams["buffer_size"],
        obs_shape=obs_shape,
        action_shape=(1,),
        device=device.type,
    )

    epsilon_fn = linear_schedule(
        hyperparams["exploration_initial_eps"],
        hyperparams["exploration_final_eps"],
        int(hyperparams["exploration_fraction"] * hyperparams["total_timesteps"]),
    )

    obs, _ = env.reset()
    global_step = 0
    episode = 0
    episode_return = 0.0
    episode_length = 0
    results = []
    evaluations = []

    best_eval_return = float("-inf")
    best_policy_state = None

    while global_step < hyperparams["total_timesteps"]:
        epsilon = epsilon_fn(global_step)
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            obs_array = np.array(obs, copy=False)
            if len(obs_array.shape) == 3:
                obs_array = np.transpose(obs_array, (2, 0, 1))
            obs_tensor = torch.tensor(obs_array, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q_values = policy(obs_tensor)
            action = int(torch.argmax(q_values, dim=1).item())

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_return += reward
        episode_length += 1

        obs_array = np.array(obs, copy=False)
        next_obs_array = np.array(next_obs, copy=False)
        if len(obs_array.shape) == 3:
            obs_array = np.transpose(obs_array, (2, 0, 1))
            next_obs_array = np.transpose(next_obs_array, (2, 0, 1))
        buffer.add(
            obs_array,
            np.array([action]),
            reward,
            next_obs_array,
            terminated,
            truncated,
        )

        obs = next_obs
        global_step += 1

        if done:
            if "episode" in info:
                episode_info = info["episode"]

                episode_return_logged = float(_to_serializable(episode_info.get("r", episode_return)))
                episode_length_logged = int(_to_serializable(episode_info.get("l", episode_length)))

                sanitized_episode_info = {
                    key: _to_serializable(value) for key, value in episode_info.items()
                }
                sanitized_episode_info.setdefault("r", episode_return_logged)
                sanitized_episode_info.setdefault("l", episode_length_logged)
                results.append(sanitized_episode_info)
                print(
                    f"global_step={global_step} episode={episode} return={float(episode_return_logged):.2f} length={int(episode_length_logged)} epsilon={epsilon:.3f}",
                    flush=True,
                )
            obs, _ = env.reset()
            episode += 1
            episode_return = 0.0
            episode_length = 0

        if global_step > hyperparams["learning_starts"] and global_step % hyperparams["train_frequency"] == 0:
            for _ in range(hyperparams["gradient_steps"]):
                batch = buffer.sample(hyperparams["batch_size"])
                (
                    observations,
                    actions,
                    rewards,
                    next_observations,
                    terminateds,
                    _,
                ) = batch

                observations_tensor = torch.tensor(observations, dtype=torch.float32, device=device)
                actions_tensor = torch.tensor(actions, dtype=torch.long, device=device)
                rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
                next_observations_tensor = torch.tensor(next_observations, dtype=torch.float32, device=device)
                terminateds_tensor = torch.tensor(terminateds, dtype=torch.float32, device=device)

                q_values = policy(observations_tensor)
                q_value = q_values.gather(1, actions_tensor)

                with torch.no_grad():
                    if use_double:
                        next_actions = policy(next_observations_tensor).argmax(dim=1, keepdim=True)
                        next_q_values = target_policy(next_observations_tensor).gather(1, next_actions)
                    else:
                        next_q_values = target_policy(next_observations_tensor).max(dim=1, keepdim=True)[0]
                    non_terminal = 1 - terminateds_tensor.unsqueeze(1)
                    target = rewards_tensor.unsqueeze(1) + hyperparams["gamma"] * non_terminal * next_q_values

                loss = loss_fn(q_value, target)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
                optimizer.step()

        if global_step % hyperparams["target_update_interval"] == 0:
            polyak_update(policy, target_policy, hyperparams["target_update_tau"])

        if args.eval_frequency and global_step % args.eval_frequency == 0:
            metrics = evaluate(policy, args.env_id, args.seed, args.eval_episodes, device)
            print(
                f"[evaluation] step={global_step} mean_return={metrics['mean_return']:.2f} std_return={metrics['std_return']:.2f}",
                flush=True,
            )
            evaluations.append({"step": global_step, **metrics})

            if metrics["mean_return"] > best_eval_return:
                best_eval_return = metrics["mean_return"]
                best_policy_state = {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()}

    env.close()

    if best_policy_state is not None:
        policy.load_state_dict(best_policy_state)

    run_dir = Path(args.save_dir) / args.env_id / args.algo
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "model.pt"
    metadata = {
        "algo": args.algo,
        "env_id": args.env_id,
        "seed": args.seed,
        "hyperparameters": hyperparams,
        "loss": args.loss,
        "episodes": episode,
        "steps": global_step,
        "best_evaluation_return": None if best_eval_return == float("-inf") else best_eval_return,
    }
    save_checkpoint({"model_state_dict": policy.state_dict(), "metadata": metadata}, checkpoint_path)
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    with open(run_dir / "training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    if evaluations:
        with open(run_dir / "evaluation_metrics.json", "w", encoding="utf-8") as f:
            json.dump(evaluations, f, indent=2)
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    train()
