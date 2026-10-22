"""Q2: why does walk.onnx get worse with the corrected controller (mj_gains_vel_d) while Unitree's policy improves?

First test: the Q1b controller factorial, now with walk.onnx. Same runner (q1b_controller/mujoco_controller_test.py):
MuJoCo at Gazebo's 1 ms step, policy every 20 steps, MuJoCo's actuators switched off and the joint torques applied
by the runner, so only the controller changes. Factors (all 24 combinations):

  armature  on / off            reflected motor inertia (training) / none (Gazebo's SDF)
  gains     policy / luis       kp, kd from the ONNX metadata (checked below) / Luis's Gazebo values
  limits    policy / luis       torque limits from the motor model / Luis's cmd_max
  dterm     vel / jnt / err     explicit -kd*q_dot / joint damping (implicit) / Gazebo's PID on the error (kick)

Gazebo's corrected controller (mj_gains_vel_d) = armature off, policy gains and limits, jnt.
Gazebo's original controller = armature off, luis gains and limits, err (+ Luis's ankle extras: one extra run).

walk.onnx runs on its own MuJoCo model (robot/g1.xml, knees-bent pose, Euler: the setup validated in
exercises 05-06); the policy file is the one the Gazebo config loads (same MD5).

    .venv/bin/python open_questions/q2_walk_onnx/mujoco_factorial.py
Output: recordings/q2/walk_<combination>_{stand,walk}.csv
"""

import hashlib
import itertools
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "exercises"))
sys.path.insert(0, str(HERE.parent / "q1b_controller"))
from g1_utils import build_g1_model  # noqa: E402
from mujoco_controller_test import DT, SCENARIOS, policy_params, run  # noqa: E402
from policy_utils import DIEGO_POLICY, load_policy  # noqa: E402

GAZEBO_POLICY = REPO / "3p/2025-tfg-diego-lopez/g1_locomotion/policies/walk.onnx"   # ros2/.../config/walk.yaml
OUT = REPO / "recordings" / "q2"


def check_policy(policy, joints):
    """Same file as Gazebo, and the runner's 'policy' gains are the ones in the ONNX metadata."""
    md5 = [hashlib.md5(p.read_bytes()).hexdigest() for p in (DIEGO_POLICY, GAZEBO_POLICY)]
    assert md5[0] == md5[1], f"MuJoCo and Gazebo walk.onnx differ: {md5}"
    meta = policy["meta"]
    assert meta["joint_names"].split(",") == joints
    kp, kd, _ = policy_params(joints)
    for key, ours in (("joint_stiffness", kp), ("joint_damping", kd)):
        theirs = np.array([float(v) for v in meta[key].split(",")])
        assert np.allclose(ours, theirs, atol=1e-3), f"{key} differs from the ONNX metadata"
    default = np.array([float(v) for v in meta["default_joint_pos"].split(",")])
    assert np.allclose(policy["default_pos"], default, atol=1e-3)
    print(f"walk.onnx {md5[0]}: same file as Gazebo; kp, kd, default pose match the metadata")


def main():
    model, data, joints, actuators = build_g1_model(REPO / "robot" / "g1.xml")
    model.opt.timestep = DT
    model.actuator_gainprm[:, 0] = 0.0      # switch off MuJoCo's position actuators:
    model.actuator_biasprm[:, 0:3] = 0.0    # the runner applies the joint torques itself
    policy = load_policy(DIEGO_POLICY, joints, actuators)
    check_policy(policy, joints)
    armature = model.dof_armature.copy()
    combos = list(itertools.product(["on", "off"], ["policy", "luis"], ["policy", "luis"], ["vel", "jnt", "err"]))
    combos.append(("off", "luis", "luis", "err", True))   # the rebuilt original, with Luis's ankle extras
    for arm, gains, limits, dterm, *extras in combos:
        model.dof_armature[:] = armature if arm == "on" else 0.0
        for scenario in SCENARIOS:
            print("ran", run(model, data, policy, joints, gains, limits, dterm, arm, scenario,
                             extras=bool(extras), out=OUT, prefix="walk_"))


if __name__ == "__main__":
    main()
