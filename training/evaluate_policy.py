"""Evaluate trained G1 velocity policies in plain MuJoCo, next to Unitree's own policy.

Every policy runs in the same setup as Exercise 7 (Unitree's MuJoCo model, home pose, implicitfast, 5 ms physics,
policy every 20 ms, 98-number observation with the gait clock), so only the policy changes. Standard tests:
  stand       zero command, 20 s: how far it drifts
  vx 0.2..1.0 forward speeds, including the small ones (dead zone check)
  vy 0.3      sideways
  wz 0.5      turning in place
Each command runs for 12 s; speeds are measured over the last 6 s (pelvis frame), heading change over the run.

    .venv/bin/python training/evaluate_policy.py training/kaggle/output/v4/logs/rsl_rl/g1_velocity/*/policy.onnx
"""

import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "exercises"))
from policy_utils import load_policy, yaw  # noqa: E402
from unitree_utils import HOME, UNITREE_POLICY, build_unitree_model, make_unitree_observation  # noqa: E402

DECIMATION = 4          # 4 x 5 ms = 20 ms policy period, as in training
SECONDS, MEASURE = 12.0, 6.0
COMMANDS = {"stand": (0.0, 0.0, 0.0), "vx 0.2": (0.2, 0.0, 0.0), "vx 0.3": (0.3, 0.0, 0.0),
            "vx 0.5": (0.5, 0.0, 0.0), "vx 1.0": (1.0, 0.0, 0.0), "vy 0.3": (0.0, 0.3, 0.0),
            "wz 0.5": (0.0, 0.0, 0.5)}


def unstable(data):
    """True once MuJoCo has flagged non-finite or huge accelerations or velocities (it then resets the state)."""
    return any(data.warning[w].number > 0 for w in (mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADQVEL,
                                                    mujoco.mjtWarning.mjWARN_BADQPOS))


def run(model, data, policy, command, seconds):
    """Returns (mean vx, vy in the pelvis frame over the last MEASURE s, mean yaw rate, distance moved, fell).
    If the simulation becomes unstable, returns NaNs and fell = True."""
    mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
    mujoco.mj_forward(model, data)
    command = np.array(command)
    last_action = np.zeros(len(policy["default_pos"]))
    start = data.qpos[:2].copy()
    v_body, yaws, fell = [], [], False
    for step in range(round(seconds / model.opt.timestep)):
        if step % DECIMATION == 0:
            obs = policy["obs_fn"](model, data, policy["default_pos"], last_action, command)
            action = policy["session"].run(None, {policy["input_name"]: obs[None, :]})[0][0]
            data.ctrl[:] = (policy["default_pos"] + policy["action_scale"] * action)[policy["joint_index_of_actuator"]]
            last_action = action
        mujoco.mj_step(model, data)
        if unstable(data):                 # MuJoCo blew up (and reset the robot itself): report, don't measure
            return np.nan, np.nan, np.nan, np.nan, True
        fell |= data.qpos[2] < 0.4
        if data.time >= seconds - MEASURE:
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, data.qpos[3:7])
            v_body.append(R.reshape(3, 3).T @ data.qvel[:3])
            yaws.append(yaw(data.qpos[3:7]))
    v = np.mean(v_body, axis=0)
    wz = np.mean(np.diff(np.unwrap(yaws))) / model.opt.timestep
    return v[0], v[1], wz, np.linalg.norm(data.qpos[:2] - start), fell


def main(paths):
    model, data, joints, actuators = build_unitree_model()
    policies = {"Unitree (reference)": UNITREE_POLICY}
    policies.update({Path(p).parent.name: Path(p) for p in paths})
    print(f"{'policy':40s} " + "  ".join(f"{name:>16s}" for name in COMMANDS))
    for label, path in policies.items():
        policy = load_policy(path, joints, actuators, pose=HOME, obs_fn=make_unitree_observation)
        cells = []
        for name, command in COMMANDS.items():
            vx, vy, wz, moved, fell = run(model, data, policy, command, SECONDS)
            if fell:
                cells.append("fell")
            elif name == "stand":
                cells.append(f"moved {moved:.2f} m")
            elif name.startswith("wz"):
                cells.append(f"wz {wz:+.2f}")
            elif name.startswith("vy"):
                cells.append(f"vy {vy:+.2f}")
            else:
                cells.append(f"vx {vx:+.2f} vy {vy:+.2f}")
        print(f"{label[:40]:40s} " + "  ".join(f"{c:>16s}" for c in cells))
    print("speeds m/s (pelvis frame, last 6 s of 12), wz rad/s; stand: distance moved in 12 s")


if __name__ == "__main__":
    main(sys.argv[1:])
