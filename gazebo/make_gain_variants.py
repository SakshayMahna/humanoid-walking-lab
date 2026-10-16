"""Write variants of Diego's g1_29dof model.sdf that differ only in the joint controllers.

  mj_gains        kp, kd and torque limits from our motor model (= the policies' ONNX metadata), instead of Luis's
  mj_gains_vel_d  same kp; D term moved from Gazebo's PID (kd * d(error)/dt, kicks when the target jumps)
                  to joint damping (kd * q_dot, like MuJoCo's position actuator)

Output: recordings/gz_models/<variant>/model.sdf (meshes resolve through model://, so any folder works).
Run from the repo root:  .venv/bin/python gazebo/make_gain_variants.py
"""

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "exercises"))
from g1_utils import ACTUATOR_GROUPS, group_params  # noqa: E402

SOURCE = REPO / "3p/2025-tfg-diego-lopez/g1_sim/models/g1_29dof/model.sdf"
OUT = REPO / "recordings/gz_models"


def mujoco_params(joint):
    """(kp, kd, effort) for a joint, from the same motor groups as the MuJoCo model."""
    for patterns, motor, n in ACTUATOR_GROUPS:
        if any(re.fullmatch(p, joint) for p in patterns):
            kp, kd, effort, _ = group_params(motor, n)
            return kp, kd, effort
    raise KeyError(joint)


def set_controller(match, velocity_d):
    plugin = match.group(0)
    joint = re.search(r"<joint_name>(.*?)</joint_name>", plugin).group(1)
    kp, kd, effort = mujoco_params(joint)
    plugin = re.sub(r"<p_gain>.*?</p_gain>", f"<p_gain>{kp:.4f}</p_gain>", plugin)
    plugin = re.sub(r"<d_gain>.*?</d_gain>", f"<d_gain>{0.0 if velocity_d else kd:.4f}</d_gain>", plugin)
    plugin = re.sub(r"<cmd_min>.*?</cmd_min>", f"<cmd_min>{-effort:g}</cmd_min>", plugin)
    plugin = re.sub(r"<cmd_max>.*?</cmd_max>", f"<cmd_max>{effort:g}</cmd_max>", plugin)
    return plugin


def add_joint_damping(sdf):
    """Insert <damping>kd</damping> into each controlled joint's <dynamics> block."""
    def per_joint(match):
        block = match.group(0)
        name = match.group(1)
        try:
            _, kd, _ = mujoco_params(name)
        except KeyError:
            return block                       # fixed / free joints
        return block.replace("<dynamics>", f"<dynamics>\n          <damping>{kd:.4f}</damping>", 1)
    return re.sub(r"<joint name=['\"]([^'\"]+)['\"].*?</joint>", per_joint, sdf, flags=re.S)


def main():
    sdf = SOURCE.read_text()
    controller = r"<plugin filename=\"gz-sim-joint-position-controller-system\".*?</plugin>"
    for name, velocity_d in [("mj_gains", False), ("mj_gains_vel_d", True)]:
        out = re.sub(controller, lambda m: set_controller(m, velocity_d), sdf, flags=re.S)
        if velocity_d:
            out = add_joint_damping(out)
        path = OUT / name / "model.sdf"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(out)
        print(f"wrote {path.relative_to(REPO)}: {out.count('<damping>')} joint dampings")


if __name__ == "__main__":
    main()
