"""Q2: do the few link differences between Gazebo's model and MuJoCo's explain why walk.onnx needs armature in
Gazebo? compare_inertias.py found that only the torso (with the head and logo fixed to it) and the two wrist-yaw
links (with the rubber hands) differ: same mass, different centre of mass and inertia. Here MuJoCo's model gets
Gazebo's values for those three bodies (fixed parts combined with the parallel-axis theorem), and the corrected
controller without armature is run again (the setup that fails in Gazebo but works in MuJoCo).

    .venv/bin/python open_questions/q2_walk_onnx/mujoco_gazebo_inertias.py
Output: recordings/q2/gzinertia_*.csv (summarize.py)
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "exercises"))
sys.path.insert(0, str(HERE.parent / "q1b_controller"))
from g1_utils import build_g1_model  # noqa: E402
from mujoco_controller_test import DT, SCENARIOS, run  # noqa: E402
from policy_utils import DIEGO_POLICY, load_policy  # noqa: E402

SDF = REPO / "3p/2025-tfg-diego-lopez/g1_sim/models/g1_29dof/model.sdf"
OUT = REPO / "recordings" / "q2"
MERGED = {"torso_link": ["head_link", "logo_link"],
          "left_wrist_yaw_link": ["left_rubber_hand"], "right_wrist_yaw_link": ["right_rubber_hand"]}


def sdf_inertial(root, name):
    """(mass, centre of mass, inertia tensor about the centre of mass) of an SDF link, in the link frame.
    The fixed parts here have pose 0 and an unrotated inertial frame (checked), so all share the parent's frame."""
    link = next(l for l in root.iter("link") if l.get("name") == name)
    inertial = link.find("inertial")
    pose = [float(v) for v in inertial.find("pose").text.split()]
    assert not any(pose[3:]), f"{name}: rotated inertial frame"
    i = {k: float(inertial.find(f"inertia/{k}").text) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")}
    tensor = np.array([[i["ixx"], i["ixy"], i["ixz"]], [i["ixy"], i["iyy"], i["iyz"]], [i["ixz"], i["iyz"], i["izz"]]])
    return float(inertial.find("mass").text), np.array(pose[:3]), tensor


def combined(root, names):
    """Mass, centre of mass and inertia (about that centre) of rigidly joined links: parallel-axis theorem."""
    parts = [sdf_inertial(root, n) for n in names]
    mass = sum(m for m, _, _ in parts)
    com = sum(m * c for m, c, _ in parts) / mass
    inertia = sum(I + m * ((c - com) @ (c - com) * np.eye(3) - np.outer(c - com, c - com)) for m, c, I in parts)
    return mass, com, inertia


def apply_gazebo_inertias(model):
    root = ET.parse(SDF).getroot()
    for body, fixed in MERGED.items():
        mass, com, inertia = combined(root, [body] + fixed)
        moments, axes = np.linalg.eigh(inertia)
        if np.linalg.det(axes) < 0:
            axes[:, 0] *= -1                           # proper rotation
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, axes.flatten())
        b = model.body(body)
        b.mass[0], b.ipos[:], b.inertia[:], b.iquat[:] = mass, com, moments, quat
        print(f"{body}: mass {mass:.4f}, com {com.round(4)}, principal {moments.round(6)}")


def main():
    model, data, joints, actuators = build_g1_model(REPO / "robot" / "g1.xml")
    model.opt.timestep = DT
    model.actuator_gainprm[:, 0] = 0.0
    model.actuator_biasprm[:, 0:3] = 0.0
    policy = load_policy(DIEGO_POLICY, joints, actuators)
    apply_gazebo_inertias(model)
    armature = model.dof_armature.copy()
    for arm, dterm in (("off", "jnt"), ("on", "jnt"), ("off", "err")):
        model.dof_armature[:] = armature if arm == "on" else 0.0
        for scenario in SCENARIOS:
            print("ran", run(model, data, policy, joints, "policy", "policy", dterm, arm, scenario, out=OUT,
                             prefix="gzinertia_walk_"))


if __name__ == "__main__":
    main()
