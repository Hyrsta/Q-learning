# save as run_all.py, then:  python run_all.py
import subprocess, sys, time, os, pathlib

cmds = [
    ("cartpole_dqn",         ["python","-m","src.dqn_experiments.train","--env-id","CartPole-v1","--algo","dqn"]),
    ("cartpole_double_dqn",  ["python","-m","src.dqn_experiments.train","--env-id","CartPole-v1","--algo","double_dqn"]),
    ("cartpole_dueling_dqn", ["python","-m","src.dqn_experiments.train","--env-id","CartPole-v1","--algo","dueling_dqn"]),
    # ("LunarLander_dqn",         ["python","-m","src.dqn_experiments.train","--env-id","LunarLander-v2","--algo","dqn"]),
    # ("LunarLander_double_dqn",  ["python","-m","src.dqn_experiments.train","--env-id","LunarLander-v2","--algo","double_dqn"]),
    # ("LunarLander_dueling_dqn", ["python","-m","src.dqn_experiments.train","--env-id","LunarLander-v2","--algo","dueling_dqn"]),
]

logdir = pathlib.Path("logs") / time.strftime("%Y%m%d_%H%M%S")
logdir.mkdir(parents=True, exist_ok=True)

for name, cmd in cmds:
    print(f">>> Running: {' '.join(cmd)}")
    with open(logdir / f"{name}.log", "w", buffering=1) as f:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
        rc = proc.wait()
        if rc != 0:
            print(f"[ERROR] Command failed: {' '.join(cmd)} (exit {rc})")
            sys.exit(rc)

print(f"Done. Logs in: {logdir}")
