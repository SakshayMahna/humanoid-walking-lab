"""Write variants of Diego's g1_29dof model.sdf that differ in the joint controllers (or, for one, in inertia).

  mj_gains        kp, kd and torque limits from our motor model (= the policies' ONNX metadata), instead of Luis's
  mj_gains_vel_d  same kp; D term moved from Gazebo's PID (kd * d(error)/dt, kicks when the target jumps)
                  to joint damping (kd * q_dot, like MuJoCo's position actuator; DART applies it implicitly)
  luis_gains_vel_d  Luis's kp, kd and limits, with the D term moved to joint damping like mj_gains_vel_d
  original_armature  Luis's controllers unchanged, plus each motor's armature (reflected rotor inertia, as in
                  MuJoCo) added to its child link's inertia: SDF/Gazebo has no joint armature, so this approximates it

Joint damping replaces any <damping> already there (Luis's model has 1.5 on the ankle-roll joints).

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


def motor_params(joint):
    """(kp, kd, effort, armature) for a joint, from the same motor groups as the MuJoCo model."""
    for patterns, motor, n in ACTUATOR_GROUPS:
        if any(re.fullmatch(p, joint) for p in patterns):
            return group_params(motor, n)
    raise KeyError(joint)


def mujoco_params(joint):
    """(kp, kd, effort) for a joint, from the same motor groups as the MuJoCo model."""
    return motor_params(joint)[:3]


def luis_params(sdf):
    """joint -> (p_gain, d_gain, cmd_max) as set in Diego's (Luis's) original controllers."""
    found = re.findall(r"<joint_name>(\w+)</joint_name>.*?<p_gain>([\d.]+)</p_gain>.*?<d_gain>([\d.]+)</d_gain>"
                       r".*?<cmd_max>([\d.]+)</cmd_max>", sdf, flags=re.S)
    return {joint: (float(p), float(d), float(lim)) for joint, p, d, lim in found}


def set_controller(match, params, velocity_d):
    plugin = match.group(0)
    joint = re.search(r"<joint_name>(.*?)</joint_name>", plugin).group(1)
    kp, kd, effort = params(joint)
    plugin = re.sub(r"<p_gain>.*?</p_gain>", f"<p_gain>{kp:.4f}</p_gain>", plugin)
    plugin = re.sub(r"<d_gain>.*?</d_gain>", f"<d_gain>{0.0 if velocity_d else kd:.4f}</d_gain>", plugin)
    plugin = re.sub(r"<cmd_min>.*?</cmd_min>", f"<cmd_min>{-effort:g}</cmd_min>", plugin)
    plugin = re.sub(r"<cmd_max>.*?</cmd_max>", f"<cmd_max>{effort:g}</cmd_max>", plugin)
    return plugin


def add_joint_damping(sdf, params):
    """Insert <damping>kd</damping> into each controlled joint's <dynamics> block."""
    def per_joint(match):
        block = match.group(0)
        name = match.group(1)
        try:
            _, kd, _ = params(name)
        except KeyError:
            return block                       # fixed / free joints
        if "<damping>" in block:                # replace Luis's value (ankle roll) instead of adding a second one
            return re.sub(r"<damping>[^<]*</damping>", f"<damping>{kd:.4f}</damping>", block, count=1)
        return block.replace("<dynamics>", f"<dynamics>\n          <damping>{kd:.4f}</damping>", 1)
    return re.sub(r"<joint name=['\"]([^'\"]+)['\"].*?</joint>", per_joint, sdf, flags=re.S)


AXIS_INDEX = {"1 0 0": 0, "0 1 0": 1, "0 0 1": 2}
INERTIA_DIAG = ["ixx", "iyy", "izz"]


def add_armature_inertia(sdf):
    """Approximate joint armature: add each motor's armature I_a to its child link's inertia about the joint axis.

    A link inertia must be physically valid (each principal moment <= the sum of the other two), so the axis term
    I_a needs an equal term on one perpendicular axis. That axis is chosen perpendicular to the parent joint's axis
    too, so the joint just above sees as little extra inertia as possible. Approximations: link and inertial frames
    are treated as aligned with the joint frame (true here up to small rotations), and the added inertia also counts
    for the joints higher up the chain (armature would not), which is small next to the leg/arm inertias.
    """
    joints = re.findall(r"<joint name='(\w+)' type='revolute'>.*?<child>(\w+)</child>.*?<xyz>([^<]+)</xyz>", sdf, flags=re.S)
    axis_of_link = {child: AXIS_INDEX[xyz.strip()] for _, child, xyz in joints}
    parent_of = {child: parent for parent, child in re.findall(r"<parent>(\w+)</parent>\s*<child>(\w+)</child>", sdf)}
    for joint, child, xyz in joints:
        armature = motor_params(joint)[3]
        k = AXIS_INDEX[xyz.strip()]
        parent_axis = axis_of_link.get(parent_of[child], k)
        perpendicular = next(i for i in range(3) if i not in (k, parent_axis))
        link = re.search(rf"<link name='{child}'>.*?</inertia>", sdf, flags=re.S)
        block = link.group(0)
        for i in (k, perpendicular):
            tag = INERTIA_DIAG[i]
            value = float(re.search(rf"<{tag}>([^<]+)</{tag}>", block).group(1))
            block = re.sub(rf"<{tag}>[^<]+</{tag}>", f"<{tag}>{value + armature:.8f}</{tag}>", block, count=1)
        sdf = sdf[:link.start()] + block + sdf[link.end():]
    return sdf


def main():
    sdf = SOURCE.read_text()
    luis = luis_params(sdf)
    sources = {"mj": mujoco_params, "luis": lambda joint: luis[joint]}
    controller = r"<plugin filename=\"gz-sim-joint-position-controller-system\".*?</plugin>"
    # name: (gains and limits from, D term on joint velocity?)
    variants = {"mj_gains": ("mj", False), "mj_gains_vel_d": ("mj", True), "luis_gains_vel_d": ("luis", True),
                "original_armature": ("luis", False)}
    for name, (source, velocity_d) in variants.items():
        params = sources[source]
        out = re.sub(controller, lambda m: set_controller(m, params, velocity_d), sdf, flags=re.S)
        if velocity_d:
            out = add_joint_damping(out, params)
        if name == "original_armature":
            out = add_armature_inertia(out)
        path = OUT / name / "model.sdf"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(out)
        print(f"wrote {path.relative_to(REPO)}: {out.count('<damping>')} joint dampings")


if __name__ == "__main__":
    main()
