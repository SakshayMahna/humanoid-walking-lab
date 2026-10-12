"""G1 motor models and actuator setup.

Values come from 3p/2025-tfg-diego-lopez/pruebas/G1/g1_constants.py (Unitree motor specs)
and match Unitree's deploy.yaml. Formula for reflected inertia: mjlab/utils/actuator.py.
"""

import re

import mujoco
import numpy as np

NATURAL_FREQ_HZ = 10.0          # target natural frequency (Hz); converted to rad/s in motor_gains
DAMPING_RATIO = 2.0             # overdamped: no overshoot


def reflected_inertia(rotor_inertias, gears):
    """Armature (kg·m²) of a two-stage planetary gearbox, seen from the joint.

    rotor_inertias: (r1, r2, r3) inertia of each rotating stage, motor rotor first
    gears:          (1, g2, g3) gear ratio of each stage (first entry is always 1)
    """
    r1, r2, r3 = rotor_inertias[:]
    g1, g2, g3 = gears[:]
    return r1 * (g2 * g3)**2 + r2 * g3**2 + r3


def motor_gains(armature, natural_freq_hz=NATURAL_FREQ_HZ, damping_ratio=DAMPING_RATIO):
    """PD gains (kp, kd) that give the chosen natural frequency (Hz) and damping ratio."""
    wn = 2 * np.pi * natural_freq_hz      # Hz -> rad/s (the formulas need rad/s)
    zeta = damping_ratio
    kp = armature * wn**2
    kd = 2 * zeta * armature * wn
    return kp, kd


# Motor types: name -> (rotor_inertias, gears, effort_limit in N·m)
MOTORS = {
    "5020": ([0.139e-4, 0.017e-4, 0.169e-4], [1, 1 + 46/18, 1 + 56/16], 25.0),
    "7520_14": ([0.489e-4, 0.098e-4, 0.533e-4], [1, 4.5, 1 + 48/22], 88.0),
    "7520_22": ([0.489e-4, 0.109e-4, 0.738e-4], [1, 4.5, 5.0], 139.0),
    "4010": ([0.068e-4, 0.0, 0.0], [1, 5, 5], 5.0),
}


# Actuator groups, in the order the reference adds them:
# list of (joint name patterns, motor name, number of motors driving the joint)
ACTUATOR_GROUPS = [
    ((".*_elbow_joint", ".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint",
      ".*_wrist_roll_joint"), "5020", 1),
    ((".*_hip_pitch_joint", ".*_hip_yaw_joint", "waist_yaw_joint"), "7520_14", 1),
    ((".*_hip_roll_joint", ".*_knee_joint"), "7520_22", 1),
    ((".*_wrist_pitch_joint", ".*_wrist_yaw_joint"), "4010", 1),
    # 4-bar linkages driven by two 5020 motors, assumed 1:1 -> everything doubles
    (("waist_pitch_joint", "waist_roll_joint"), "5020", 2),
    ((".*_ankle_pitch_joint", ".*_ankle_roll_joint"), "5020", 2),
]


def group_params(motor_name, n_motors=1):
    """(kp, kd, effort_limit, armature) for a joint driven by n_motors of this type."""
    rotor_inertias, gears, effort_limit = MOTORS[motor_name]
    armature = n_motors * reflected_inertia(rotor_inertias, gears)   # n motors in parallel: inertias add
    kp, kd = motor_gains(armature)                                   # gains scale with armature -> also x n
    return kp, kd, n_motors * effort_limit, armature


def hinge_joint_names(spec: mujoco.MjSpec) -> list[str]:
    """Joint names in model order, skipping the free joint (floating base)."""
    return [j.name for j in spec.joints if j.type != mujoco.mjtJoint.mjJNT_FREE]


def add_position_actuators(spec: mujoco.MjSpec) -> list[str]:
    """Add one position actuator per hinge joint, in group order. Returns actuator names in order."""
    joints = hinge_joint_names(spec)
    added = []
    for patterns, motor_name, n_motors in ACTUATOR_GROUPS:
        kp, kd, effort, armature = group_params(motor_name, n_motors)
        for name in joints:  # within a group, joints keep model order
            if not any(re.fullmatch(p, name) for p in patterns):
                continue
            act = spec.add_actuator(name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
            act.set_to_position(kp=kp, kv=kd)       # gainprm = [kp], biasprm = [0, -kp, -kd]
            act.forcelimited = True
            act.forcerange = [-effort, effort]      # motor peak torque
            spec.joint(name).armature = armature    # armature belongs to the joint, not the actuator
            added.append(name)

    # Every hinge joint must be driven by exactly one actuator
    assert sorted(added) == sorted(joints), (
        f"unmatched: {set(joints) - set(added)}, duplicates: {[n for n in added if added.count(n) > 1]}")
    return added


# Knees-bent starting pose (KNEES_BENT_KEYFRAME in g1_constants.py). Joints not listed stay at 0.
BASE_HEIGHT = 0.76  # pelvis height (m)
KNEES_BENT = {
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
}


def default_joint_pos(joint_names, pose=KNEES_BENT) -> np.ndarray:
    """One angle per joint (rad), in the order of joint_names."""
    joint_pos = np.zeros(len(joint_names))
    for pattern, value in pose.items():
        for i, name in enumerate(joint_names):
            if re.fullmatch(pattern, name):
                joint_pos[i] = value
    return joint_pos


def add_init_keyframe(spec: mujoco.MjSpec, actuator_names: list[str]) -> None:
    """Add an 'init_state' keyframe: knees-bent pose with matching actuator targets."""
    joints = hinge_joint_names(spec)
    joint_pos = default_joint_pos(joints)

    # qpos = base position (3) + base quaternion wxyz (4, upright) + joint angles in JOINT order
    qpos = np.concatenate([[0.0, 0.0, BASE_HEIGHT], [1.0, 0.0, 0.0, 0.0], joint_pos])

    # ctrl = position targets in ACTUATOR order (differs from joint order!)
    target_by_name = dict(zip(joints, joint_pos))
    ctrl = [target_by_name[name] for name in actuator_names]

    spec.add_key(name="init_state", qpos=qpos.tolist(), ctrl=ctrl)

# Friction parameters
FOOT_PATTERN = r"^(left|right)_foot[1-7]_collision$"

def apply_collision_settings(spec: mujoco.MjSpec) -> None:
    """Diego's FULL_COLLISION: feet grip (condim 3, friction 0.6, priority 1), other collision geoms are
    frictionless (condim 1), every other geom is disabled. Call BEFORE adding the floor, or it gets disabled."""
    for geom in spec.geoms:
        if re.fullmatch(FOOT_PATTERN, geom.name):
            geom.condim = 3            # normal + 2D sliding friction
            geom.priority = 1          # beats the floor (0): contacts use the foot's friction
            geom.friction[0] = 0.6     # sliding friction; torsional/rolling unchanged
            geom.contype = geom.conaffinity = 1
        elif re.fullmatch(".*_collision", geom.name):
            geom.condim = 1            # frictionless: self-collisions just push apart
            geom.priority = 0
            geom.contype = geom.conaffinity = 1
        else:
            geom.contype = geom.conaffinity = 0   # visual meshes and unnamed geoms: no collisions

def build_g1_model(robot_xml) -> tuple[mujoco.MjModel, mujoco.MjData, list[str], list[str]]:
    """Diego's simulatable G1 (walk_fixed.ipynb). Returns (model, data, joint_names, actuator_names).

    data starts at the 'init_state' keyframe (knees bent) with derived quantities computed.
    """
    spec = mujoco.MjSpec.from_file(str(robot_xml))

    apply_collision_settings(spec)
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, 0.1])

    joint_names = hinge_joint_names(spec)
    actuator_names = add_position_actuators(spec)
    add_init_keyframe(spec, actuator_names)

    spec.option.timestep = 0.005
    spec.option.iterations = 10
    spec.option.ls_iterations = 20
    spec.option.ccd_iterations = 500

    model = spec.compile()
    data = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)   # sets qpos, qvel, ctrl only
    mujoco.mj_forward(model, data)   # compute xpos, xmat, sensordata... for that state

    return model, data, joint_names, actuator_names