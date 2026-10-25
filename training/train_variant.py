"""Train one variant of Unitree-G1-Flat (training/variants.py) with Unitree's training script.

Run from inside the unitree_rl_mjlab checkout (it imports its scripts/train.py and its tasks):

    python /path/to/training/train_variant.py --variant ladder_0 --seed 1 --iterations 1000 --gpu 0

Same as `scripts/train.py Unitree-G1-Flat ...` except that the environment config is changed by the variant before
training. Logs go to logs/rsl_rl/g1_velocity/<date>_<variant>_seed<n>/ (params/env.yaml records the config used).
"""

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # variants.py
sys.path.insert(0, str(Path.cwd() / "scripts"))             # Unitree's train.py

TASK = "Unitree-G1-Flat"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=4096)
    args = parser.parse_args()

    import mjlab.tasks  # noqa: F401  (registers mjlab's tasks, as train.py does)
    import src.tasks    # noqa: F401  (registers Unitree's tasks)
    from train import TrainConfig, launch_training
    from variants import VARIANTS

    cfg = TrainConfig.from_task(TASK)
    VARIANTS[args.variant](cfg.env)
    cfg.env.scene.num_envs = args.num_envs
    cfg.agent.max_iterations = args.iterations
    cfg.agent.seed = args.seed
    cfg.agent.logger = "tensorboard"            # wandb would ask for a login
    cfg.agent.run_name = f"{args.variant}_seed{args.seed}"
    cfg = dataclasses.replace(cfg, gpu_ids=[args.gpu])
    print(f"[variant] {args.variant}: rewards = {sorted(cfg.env.rewards)}", flush=True)
    launch_training(task_id=TASK, args=cfg)


if __name__ == "__main__":
    main()
