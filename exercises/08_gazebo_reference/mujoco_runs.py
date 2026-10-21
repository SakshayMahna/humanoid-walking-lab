"""MuJoCo reference runs in the same CSV format as gazebo/scenario_logger.py (Gazebo), for gazebo/summarize_runs.py.

Stand 4 s, then vx = 0.5 for 20 s, both policies. Velocities are read at the imu_in_pelvis site, the same point
as Gazebo's odometry (xyz_offset).

    .venv/bin/python exercises/08_gazebo_reference/mujoco_runs.py      (from the repo root)
"""

import csv
import sys
from pathlib import Path

import mujoco

script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir.parent))  # make exercises/*_utils.py importable
from g1_utils import build_g1_model
from observation_utils import read_sensor
from policy_utils import DIEGO_POLICY, load_policy, make_policy_controller, yaw
from unitree_utils import build_unitree_model, load_unitree_policy

REPO = script_dir.parent.parent
OUT = REPO / "recordings" / "mujoco"


def run(name, model, data, policy, seconds=24.0, stand=4.0, vx=0.5, after_step=None, out=OUT):
    """after_step(model, data), if given, runs after every physics step (e.g. to keep a sensor history).
    Writes out/name.csv."""
    mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
    mujoco.mj_forward(model, data)
    control = make_policy_controller(policy, lambda t: [vx if t >= stand else 0.0, 0.0, 0.0])
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{name}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "x", "y", "z", "yaw", "vx_body", "vy_body", "wz", "cmd_vx", "cmd_vy", "cmd_wz"])
        for step in range(round(seconds / model.opt.timestep)):
            control(model, data)
            mujoco.mj_step(model, data)
            if after_step is not None:
                after_step(model, data)
            if step % 10 == 0:   # 20 Hz, like scenario_logger.py
                v = read_sensor(model, data, "imu_lin_vel")
                w = read_sensor(model, data, "imu_ang_vel")
                cmd = vx if data.time >= stand else 0.0
                writer.writerow([f"{data.time:.3f}", *(f"{q:.4f}" for q in data.qpos[0:3]),
                                 f"{yaw(data.qpos[3:7]):.4f}", f"{v[0]:.4f}", f"{v[1]:.4f}", f"{w[2]:.4f}",
                                 f"{cmd:.3f}", "0.000", "0.000"])
    print(f"wrote {out / name}.csv")


if __name__ == "__main__":
    model, data, joints, actuators = build_g1_model(REPO / "robot" / "g1.xml")
    run("mujoco_walk_1", model, data, load_policy(DIEGO_POLICY, joints, actuators))
    model, data, joints, actuators = build_unitree_model()
    run("mujoco_unitree_1", model, data, load_unitree_policy(joints, actuators))
