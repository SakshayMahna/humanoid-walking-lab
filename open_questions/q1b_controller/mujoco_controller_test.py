"""Q1b: which part of Gazebo's original joint controller breaks Unitree's policy? Tested in MuJoCo.

Everything is MuJoCo (Unitree's model, implicitfast), at Gazebo's 1 ms physics step with the policy every 20
steps (50 Hz). The model's position actuators are switched off; each step, this script computes the joint
torques itself and applies them (data.qfrc_applied), so the ONLY thing that changes between runs is the
controller. Four factors, all 24 combinations:

  gains   policy  kp, kd from the motor model (= ONNX metadata, what training used)
          luis    Luis's p_gain / d_gain from Diego's Gazebo model (3p/.../g1_29dof/model.sdf)
  limits  policy  torque limits from the motor model (ankle roll 50 N·m)
          luis    Luis's cmd_max (ankle roll 10 N·m)
  dterm   vel     tau = kp (target - q) - kd q_dot, applied as an explicit torque
          jnt     tau = kp (target - q), and kd as joint damping, which MuJoCo integrates implicitly
                  (like SDF <damping> in Gazebo/DART: the corrected controller)
          err     e = q - target; tau = -kp e - kd (e - e_prev) / dt         (Gazebo's JointPositionController /
                                                                              gz-math PID::Update)
  armature on     reflected motor inertia on each joint (MuJoCo model, training)
           off    no armature (Gazebo's SDF has none)

Scenarios: "stand" (zero command for 30 s) and "walk" (zero for 4 s, then vx = 0.5 for 20 s).
CSV per run in recordings/q1b/ (same format as the Gazebo logs), plus a knee trace for the kick plot.

    .venv/bin/python open_questions/q1b_controller/mujoco_controller_test.py
"""

import csv
import itertools
import re
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "exercises"))
from g1_utils import ACTUATOR_GROUPS, group_params  # noqa: E402
from observation_utils import read_sensor  # noqa: E402
from policy_utils import yaw  # noqa: E402
from unitree_utils import build_unitree_model, load_unitree_policy  # noqa: E402

GAZEBO_SDF = REPO / "3p/2025-tfg-diego-lopez/g1_sim/models/g1_29dof/model.sdf"
OUT = REPO / "recordings" / "q1b"
DT = 0.001                 # Gazebo's physics step
DECIMATION = 20            # 20 x 1 ms = 20 ms policy period (50 Hz), as in training
TRACE_JOINT = "left_knee_joint"
SCENARIOS = {"stand": (30.0, 30.0, 0.0), "walk": (24.0, 4.0, 0.5)}   # seconds, stand seconds, vx


def policy_params(joints):
    """kp, kd, torque limit per joint (joint order) from the motor model."""
    kp, kd, lim = np.zeros(len(joints)), np.zeros(len(joints)), np.zeros(len(joints))
    for patterns, motor, n in ACTUATOR_GROUPS:
        p, d, effort, _ = group_params(motor, n)
        for i, name in enumerate(joints):
            if any(re.fullmatch(pat, name) for pat in patterns):
                kp[i], kd[i], lim[i] = p, d, effort
    return kp, kd, lim


def luis_params(joints):
    """p_gain, d_gain, cmd_max per joint (joint order) from Diego's Gazebo model."""
    sdf = GAZEBO_SDF.read_text()
    values = {}
    for m in re.finditer(r"<joint_name>(\w+)</joint_name>.*?<p_gain>([\d.]+)</p_gain>.*?<d_gain>([\d.]+)</d_gain>"
                         r".*?<cmd_max>([\d.]+)</cmd_max>", sdf, flags=re.S):
        values[m.group(1)] = tuple(float(v) for v in m.group(2, 3, 4))
    kp, kd, lim = (np.array([values[j][k] for j in joints]) for k in range(3))
    return kp, kd, lim


def run(model, data, policy, joints, gains, limits, dterm, armature, scenario, trace=False, on_step=None,
        extras=False, out=OUT, prefix=""):
    """One run. model.dof_armature must already be set for `armature`. on_step(model, data), if given, runs
    after every physics step (e.g. to render a video). extras=True adds what Luis's Gazebo model has on the
    ankle-roll joints: joint damping 1.5 (replaced by kd when dterm is "jnt", as the Gazebo variants do) and
    friction 0.1 (MuJoCo frictionloss). CSVs go to out/, names start with prefix (Q2 reuses this for walk.onnx)."""
    seconds, stand, vx = SCENARIOS[scenario]
    pol, luis = policy_params(joints), luis_params(joints)
    kp, kd = (pol if gains == "policy" else luis)[:2]
    lim = (pol if limits == "policy" else luis)[2]

    model.dof_damping[6:] = kd if dterm == "jnt" else 0.0
    model.dof_frictionloss[6:] = 0.0
    if extras:
        for i, joint in enumerate(joints):
            if joint.endswith("ankle_roll_joint"):
                if dterm != "jnt":
                    model.dof_damping[6 + i] = 1.5
                model.dof_frictionloss[6 + i] = 0.1
    mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
    mujoco.mj_forward(model, data)
    n = len(joints)
    target = policy["default_pos"].copy()
    last_action = np.zeros(n)
    e_prev = np.zeros(n)               # gz-math PID starts with pErrLast = 0
    k = joints.index(TRACE_JOINT)

    name = f"{prefix}arm{armature}_{gains}gains_{limits}limits_{dterm}{'_x' if extras else ''}_{scenario}"
    out.mkdir(parents=True, exist_ok=True)
    rows, trace_rows = [], []
    for step in range(round(seconds / DT)):
        if step % DECIMATION == 0:
            cmd = np.array([vx if data.time >= stand else 0.0, 0.0, 0.0])
            obs = policy["obs_fn"](model, data, policy["default_pos"], last_action, cmd)
            action = policy["session"].run(None, {policy["input_name"]: obs[None, :]})[0][0]
            target = policy["default_pos"] + policy["action_scale"] * action
            last_action = action

        q, qd = data.qpos[7:], data.qvel[6:]
        if dterm == "vel":
            tau = kp * (target - q) - kd * qd
        elif dterm == "jnt":
            tau = kp * (target - q)          # damping comes from model.dof_damping (implicit)
        else:
            e = q - target
            tau = -kp * e - kd * (e - e_prev) / DT
            e_prev = e.copy()
        tau = np.clip(tau, -lim, lim)
        data.qfrc_applied[6:] = tau
        mujoco.mj_step(model, data)
        if on_step is not None:
            on_step(model, data)

        if step % 50 == 0:            # 20 Hz log
            v = read_sensor(model, data, "imu_lin_vel")
            w = read_sensor(model, data, "imu_ang_vel")
            rows.append([f"{data.time:.3f}", *(f"{x:.4f}" for x in data.qpos[0:3]), f"{yaw(data.qpos[3:7]):.4f}",
                         f"{v[0]:.4f}", f"{v[1]:.4f}", f"{w[2]:.4f}", f"{cmd[0]:.3f}", "0.000", "0.000"])
        if trace and 10.0 <= data.time < 10.5:   # 0.5 s of knee torque, every step
            trace_rows.append([f"{data.time:.4f}", f"{target[k]:.5f}", f"{q[k]:.5f}", f"{tau[k]:.3f}"])

    with open(out / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "x", "y", "z", "yaw", "vx_body", "vy_body", "wz", "cmd_vx", "cmd_vy", "cmd_wz"])
        w.writerows(rows)
    if trace:
        with open(out / f"trace_{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "target", "q", "tau"])
            w.writerows(trace_rows)
    return name


def main():
    model, data, joints, actuators = build_unitree_model()
    model.opt.timestep = DT
    model.actuator_gainprm[:, 0] = 0.0      # switch off MuJoCo's position actuators:
    model.actuator_biasprm[:, 0:3] = 0.0    # this script applies the joint torques itself
    policy = load_unitree_policy(joints, actuators)
    armature = model.dof_armature.copy()
    for arm, gains, limits, dterm in itertools.product(["on", "off"], ["policy", "luis"], ["policy", "luis"],
                                                         ["vel", "jnt", "err"]):
        model.dof_armature[:] = armature if arm == "on" else 0.0
        for scenario in SCENARIOS:
            trace = scenario == "walk" and gains == "policy" and limits == "policy"
            print("ran", run(model, data, policy, joints, gains, limits, dterm, arm, scenario, trace))


if __name__ == "__main__":
    main()
