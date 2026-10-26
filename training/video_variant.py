"""Video of a trained policy in its own training simulator (mjlab / MuJoCo Warp), with a fixed command.

The evaluation scripts (report.py, render_policy.py) run policies in plain MuJoCo on the CPU. This shows the same
policy in the environment it was trained in, so a policy that only works there (or only in plain MuJoCo) is visible.
One robot, Unitree-G1-Flat's play settings, stand for STAND seconds, then walk forward at VX.

Run from inside the unitree_rl_mjlab checkout:

    python /path/to/training/video_variant.py --checkpoint logs/.../model_999.pt --out walk.mp4 --vx 0.5

Writes the MP4 and a CSV next to it (t, x, y, z, reset): `reset` = 1 when the episode ended (a fall) and the robot
was put back at its start.
"""

import argparse
import csv
import os
import sys
from dataclasses import asdict
from pathlib import Path

TASK = "Unitree-G1-Flat"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--vx", type=float, default=0.5)
    parser.add_argument("--stand", type=float, default=4.0)
    parser.add_argument("--seconds", type=float, default=20.0)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")      # headless offscreen rendering (as Unitree's train.py sets)
    sys.path.insert(0, str(Path.cwd()))
    import imageio.v2 as imageio
    import mjlab.tasks  # noqa: F401
    import src.tasks    # noqa: F401
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env_cfg = load_env_cfg(TASK, play=True)
    agent_cfg = load_rl_cfg(TASK)
    env_cfg.scene.num_envs = 1
    twist = env_cfg.commands["twist"]
    twist.ranges.lin_vel_x = (0.0, 0.0)           # start standing; the forward command is set at t = STAND
    twist.ranges.lin_vel_y = (0.0, 0.0)
    twist.ranges.ang_vel_z = (0.0, 0.0)
    twist.heading_command = False
    twist.rel_standing_envs = 0.0
    twist.resampling_time_range = (1e6, 1e6)      # never resample: the command stays what we set

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = load_runner_cls(TASK)(wrapped, asdict(agent_cfg), device=device)
    runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)
    command = env.command_manager.get_term("twist")
    robot = env.scene["robot"]

    dt = env.step_dt
    obs = wrapped.get_observations()
    frames, rows = [], []
    for step in range(round(args.seconds / dt)):
        t = step * dt
        if t >= args.stand:
            command.vel_command_b[:, 0] = args.vx
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, dones, _ = wrapped.step(actions)
        frames.append(env.render())
        p = robot.data.root_link_pos_w[0].tolist()
        rows.append([f"{t + dt:.2f}", *(f"{v:.4f}" for v in p), int(dones[0].item())])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, fps=round(1 / dt))
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "x", "y", "z", "reset"])
        writer.writerows(rows)
    resets = sum(r[-1] for r in rows)
    print(f"[video] {out.name}: {len(frames)} frames, {resets} reset(s) (falls) in {args.seconds:.0f} s")
    env.close()


if __name__ == "__main__":
    main()
