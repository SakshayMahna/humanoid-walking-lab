"""Q6: is Gazebo's larger left drift of Unitree's policy (+1.0 °/s vs +0.72 °/s in MuJoCo) a real Gazebo effect?

Two things could make MuJoCo's own number move, so measure them first, in MuJoCo only:
  1. physics step: 5 ms (training), 2.5 ms, 2 ms, 1 ms (Gazebo's step; policy still every 20 ms)
  2. run-to-run spread: tiny changes to the start (random joint velocities, std 1e-3 rad/s) change nothing
     physically, but a walking robot is chaotic enough that the heading can still differ

Unitree's own MuJoCo model with its position actuators (training setup), command: stand 4 s, then vx = 0.5,
24 s in total; turn rate and wander measured from 8 s, as in gazebo/summarize_runs.py.

    .venv/bin/python open_questions/q6_drift/mujoco_drift_spread.py
"""

import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "exercises"))
from policy_utils import yaw  # noqa: E402
from unitree_utils import build_unitree_model, load_unitree_policy  # noqa: E402

SEEDS = 4
SECONDS, STAND, VX = 24.0, 4.0, 0.5
POLICY_DT = 0.02


def run(model, data, policy, seed):
    mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
    if seed > 0:
        data.qvel[6:] = np.random.default_rng(seed).normal(0.0, 1e-3, model.nv - 6)
    mujoco.mj_forward(model, data)
    decimation = round(POLICY_DT / model.opt.timestep)
    last_action = np.zeros(len(policy["default_pos"]))
    t, yaws = [], []
    for step in range(round(SECONDS / model.opt.timestep)):
        if step % decimation == 0:
            cmd = np.array([VX if data.time >= STAND else 0.0, 0.0, 0.0])
            obs = policy["obs_fn"](model, data, policy["default_pos"], last_action, cmd)
            action = policy["session"].run(None, {policy["input_name"]: obs[None, :]})[0][0]
            data.ctrl[:] = (policy["default_pos"] + policy["action_scale"] * action)[policy["joint_index_of_actuator"]]
            last_action = action
        mujoco.mj_step(model, data)
        t.append(data.time)
        yaws.append(yaw(data.qpos[3:7]))
    t, yaws = np.array(t), np.degrees(np.unwrap(yaws))
    w = t >= STAND + 4.0
    fit = np.polyfit(t[w], yaws[w], 1)
    return fit[0], np.std(yaws[w] - np.polyval(fit, t[w]))


def main():
    model, data, joints, actuators = build_unitree_model()
    policy = load_unitree_policy(joints, actuators)
    for dt in (0.005, 0.0025, 0.002, 0.001):
        model.opt.timestep = dt
        results = np.array([run(model, data, policy, seed) for seed in range(SEEDS)])
        turns = results[:, 0]
        print(f"dt {dt * 1000:.1f} ms: turn (deg/s) unperturbed {turns[0]:+.2f}, "
              f"perturbed mean {turns[1:].mean():+.2f} ± {turns[1:].std():.2f} "
              f"(range {turns[1:].min():+.2f} … {turns[1:].max():+.2f}); wander {results[:, 1].mean():.2f}")


if __name__ == "__main__":
    main()
