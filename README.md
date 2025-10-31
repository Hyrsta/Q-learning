# Reproduce DQN variants

This project provides ready-to-run PyTorch implementations of three Deep Q-Network (DQN) agents together with scripts for training and evaluating them on the requested environments.

## Implemented algorithms

1. **Vanilla DQN** – single Q-network with a target network and max-backup targets.
2. **Double DQN** – identical architecture that uses the Double Q-learning target to reduce overestimation bias.
3. **Dueling DQN** – Double DQN with a dueling Q-network head (value/advantage streams).

All agents share the same replay buffer, epsilon-greedy exploration schedule, and network architectures. Convolutional networks are used for the Atari environments and multilayer perceptrons for the classic-control tasks.

## Supported environments

1. `CartPole-v1`
2. `LunarLander-v3`
3. `BreakoutNoFrameskip-v4`
4. `PongNoFrameskip-v4`

The Atari environments are automatically wrapped with standard preprocessing (grayscale, down-sampling to 84×84, frame stacking) to match the original DQN papers.

## Setup

1. **Create and activate a Python environment** (Python 3.9+ recommended).

    ```bash
    conda create -n qlearning python=3.11
    ```

2. **Install Torch**:

    ```bash
    pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
    ```

3. **Install other dependencies**:

    ```bash
    pip install -r requirements.txt
    ```
    The requirements rely entirely on open-source libraries: [PyTorch](https://pytorch.org) for deep learning and [Gymnasium](https://gymnasium.farama.org) (with `ale-py`) for the environments.

4. **Accept the Atari ROM license** on first launch. The `gymnasium[atari,accept-rom-license]` extra handles downloading the free and legal ROMs via `AutoROM`.

## Training

The training script lives at `src/dqn_experiments/train.py`. It exposes CLI flags for selecting the environment, algorithm, and common hyper-parameters. Default hyper-parameters are provided for each task (based on widely used community baselines) and can be overridden via the CLI.

### Examples

Train a vanilla DQN agent on CartPole:

```bash
python -m src.dqn_experiments.train --env-id CartPole-v1 --algo dqn # double_dqn / dueling_dqn
```

Train a Double DQN agent on LunarLander with a custom random seed:

```bash
python -m src.dqn_experiments.train --env-id LunarLander-v3 --algo double_dqn --seed 123
```

Train the dueling variant on Breakout (this uses the default 10M environment steps, so expect a long run):

```bash
python -m src.dqn_experiments.train --env-id BreakoutNoFrameskip-v4 --algo dueling_dqn --device cuda
```

Outputs are written to `runs/<env-id>/<algo>/` and include the PyTorch checkpoint (`model.pt`), metadata about the run, raw episode statistics, and (when periodic evaluation is enabled) aggregated evaluation results for each checkpointed step.

### Useful CLI options

* `--total-timesteps` – override the number of environment steps.
* `--learning-rate`, `--batch-size`, `--buffer-size`, etc. – customise hyper-parameters.
* `--eval-frequency` / `--eval-episodes` – enable periodic evaluation runs during training.
* `--save-dir` – change the base directory for experiment artefacts.

Run `python -m src.dqn_experiments.train --help` to see the complete list.

## Evaluation

Use `src/dqn_experiments/evaluate.py` to load a saved checkpoint and run evaluation episodes:

```bash
python -m src.dqn_experiments.evaluate \
  --env-id BreakoutNoFrameskip-v4 \
  --algo dueling_dqn \
  --checkpoint runs/BreakoutNoFrameskip-v4/dueling_dqn/model.pt \
  --episodes 20
```

The evaluator verifies that the supplied `--algo` matches the saved metadata and reports per-episode returns plus the aggregate mean ± standard deviation.

## Extending the experiments

* Adjust the default hyper-parameters in `get_default_hyperparameters` inside `src/dqn_experiments/train.py` to fine-tune results.
* Swap in alternative exploration schedules via `src/dqn_experiments/schedules.py`.
* Extend `ATARI_ENVS` in `src/dqn_experiments/envs.py` to add more Atari tasks that share the same preprocessing pipeline.

Happy experimenting!

## Visualising training progress

The plotting utility at `src/dqn_experiments/plotting.py` summarises multiple runs into the three report-ready figures described in the brief:

```bash
python -m src.dqn_experiments.plotting --runs-dir runs --output-dir figures
```

The script looks for experiment folders that contain `training_metrics.json`, `evaluation_metrics.json`, and the saved metadata. It then emits three PNG files:

1. `learning_curves.png` – evaluation return versus environment steps for each environment (mean ± 95% CI over seeds).
2. `sample_efficiency.png` – grouped bar chart of the average number of environment steps needed to reach the default target score per algorithm.
3. `stability.png` – box plots of the final evaluation returns for every algorithm/environment combination.

Use `--target-score ENV=VALUE` to override the default thresholds that drive the sample-efficiency figure, and `--show` to display the figures interactively after saving.

## Qualitative result videos

To generate qualitative rollouts—one per algorithm for each environment—use the video renderer at `src/dqn_experiments/videos.py`:

```bash
python -m src.dqn_experiments.videos --runs-dir runs --output-dir videos
```

The script searches for checkpoints in `runs/<env-id>/<algo>/model.pt` and records one evaluation episode per pairing to MP4 files inside the requested `--output-dir`. Use `--env` and `--algo` flags to restrict the subset of environments or algorithms (e.g. `--env CartPole --algo dueling_dqn`).
