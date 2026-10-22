"""MuJoCo reference for the velocity log: walk.onnx in the same four setups as the Gazebo runs (1 ms steps, the
Q1b runner), logging the base_lin_vel input every 3 ms in the same format as Gazebo's <name>.odom.csv.

In MuJoCo the input is the velocimeter at imu_in_pelvis (exact, pelvis frame); the pose columns are that site's
position and the pelvis orientation, so velocity_analysis.py can treat both simulators the same way.
Output: recordings/q2/mj_walk_<setup>_<scenario>.odom.csv (and the usual 20 Hz CSV, from run())

    .venv/bin/python open_questions/q2_walk_onnx/mujoco_velocity_log.py
"""

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "exercises"))
sys.path.insert(0, str(HERE.parent / "q1b_controller"))
from g1_utils import build_g1_model  # noqa: E402
from mujoco_controller_test import DT, SCENARIOS, run  # noqa: E402
from observation_utils import read_sensor  # noqa: E402
from policy_utils import DIEGO_POLICY, load_policy  # noqa: E402

OUT = REPO / "recordings" / "q2"
EVERY = 3                  # log every 3 physics steps = 3 ms, like Gazebo's odometry at 1 ms steps

# setup name (as in gazebo_repeats.sh): armature, gains, limits, D term, Luis's ankle extras
SETUPS = {
    "original": ("off", "luis", "luis", "err", True),
    "original_armature": ("on", "luis", "luis", "err", True),
    "corrected": ("off", "policy", "policy", "jnt", False),
    "corrected_armature": ("on", "policy", "policy", "jnt", False),
}


class OdomLog:
    """on_step hook: the walk.onnx velocity input and the imu site pose, every EVERY steps."""

    def __init__(self, path):
        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["t", "x", "y", "z", "qw", "qx", "qy", "qz", "vx", "vy", "vz", "wx", "wy", "wz"])
        self.step = 0

    def __call__(self, model, data):
        if self.step % EVERY == 0:
            p = data.site("imu_in_pelvis").xpos
            v, w = read_sensor(model, data, "imu_lin_vel"), read_sensor(model, data, "imu_ang_vel")
            self.writer.writerow([f"{data.time:.4f}", *(f"{a:.6f}" for a in (*p, *data.qpos[3:7])),
                                  *(f"{a:.5f}" for a in (*v, *w))])
        self.step += 1

    def close(self):
        self.file.close()


def main():
    model, data, joints, actuators = build_g1_model(REPO / "robot" / "g1.xml")
    model.opt.timestep = DT
    model.actuator_gainprm[:, 0] = 0.0
    model.actuator_biasprm[:, 0:3] = 0.0
    armature = model.dof_armature.copy()
    policy = load_policy(DIEGO_POLICY, joints, actuators)
    for setup, (arm, gains, limits, dterm, extras) in SETUPS.items():
        model.dof_armature[:] = armature if arm == "on" else 0.0
        for scenario in SCENARIOS:
            log = OdomLog(OUT / f"mj_walk_{setup}_{scenario}.odom.csv")
            run(model, data, policy, joints, gains, limits, dterm, arm, scenario, on_step=log, extras=extras,
                out=OUT, prefix=f"mj_walk_{setup}_")
            log.close()
            print("logged", setup, scenario)


if __name__ == "__main__":
    main()
