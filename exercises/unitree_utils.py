"""Unitree's G1 velocity policy (unitree_rl_mjlab): model settings, gait phase, 98-number observation."""

from pathlib import Path

import mujoco
import numpy as np

from g1_utils import build_g1_model
from observation_utils import projected_gravity, read_sensor
from policy_utils import load_policy

REPO_ROOT = Path(__file__).resolve().parent.parent
UNITREE_REPO = REPO_ROOT / "3p" / "unitree_rl_mjlab"
UNITREE_XML = UNITREE_REPO / "src" / "assets" / "robots" / "unitree_g1" / "xmls" / "g1.xml"
UNITREE_POLICY = UNITREE_REPO / "deploy" / "robots" / "g1" / "config" / "policy" / "velocity" / "v0" / "exported" / "policy.onnx"

# HOME_KEYFRAME in Unitree's g1_constants.py. Joints not listed stay at 0.
HOME_HEIGHT = 0.8   # pelvis height (m)
HOME = {
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    ".*_elbow_joint": 0.87,
    "left_shoulder_roll_joint": 0.18,
    "left_shoulder_pitch_joint": 0.35,
    "right_shoulder_roll_joint": -0.18,
    "right_shoulder_pitch_joint": 0.35,
}

GAIT_PERIOD = 0.6   # s per stride (Unitree's phase observation, period=0.6)


def gait_phase(t, command, period=GAIT_PERIOD) -> np.ndarray:
    """Unitree's gait clock: [sin 2πφ, cos 2πφ] with φ = (t mod period) / period; [0, 0] when standing.

    t is the time since reset (training: episode steps x 0.02 s). sin/cos instead of φ itself so the
    clock has no jump when φ wraps from 1 back to 0. ‖command‖ includes the turn rate wz.
    """
    phi = (t % period) / period
    phase = np.array([np.sin(2 * np.pi * phi), np.cos(2 * np.pi * phi)])
    if np.linalg.norm(command) < 0.1:   # stand-still command: clock off
        phase = np.zeros(2)
    return phase


# Layout from Unitree's policy.onnx metadata (observation_names): name -> size
UNITREE_OBS_LAYOUT = [("base_ang_vel", 3), ("projected_gravity", 3), ("command", 3), ("phase", 2),
                      ("joint_pos", 29), ("joint_vel", 29), ("actions", 29)]
UNITREE_OBS_DIM = sum(size for _, size in UNITREE_OBS_LAYOUT)   # 98


def make_unitree_observation(model: mujoco.MjModel, data: mujoco.MjData, default_pos, last_action, command) -> np.ndarray:
    """The 98 numbers Unitree's policy expects, in its order. Joint terms are in joint order (qpos[7:], qvel[6:]).

    vs Diego's 99: no base_lin_vel, command moved to 3rd, gait phase added after it.
    """
    obs = np.concatenate([
        read_sensor(model, data, "imu_ang_vel"),   # angular velocity, pelvis frame
        projected_gravity(model, data),            # "down" in the pelvis frame
        command,                                   # [vx, vy, wz]
        gait_phase(data.time, command),            # [sin, cos] stride clock; [0, 0] when standing
        data.qpos[7:] - default_pos,               # joint angles relative to the default (home) pose
        data.qvel[6:],                             # joint velocities
        last_action,                               # previous raw policy output
    ]).astype(np.float32)                          # ONNX expects float32
    assert obs.shape == (UNITREE_OBS_DIM,), f"observation has {obs.shape[0]} numbers, expected {UNITREE_OBS_DIM}"
    return obs


UNITREE_MODEL_SETTINGS = dict(
    pose=HOME, base_height=HOME_HEIGHT,
    integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST,   # mjlab's training default
    ccd_iterations=50,                                    # flat task; no effect here (only capsules/spheres collide)
)


def build_unitree_model(decorate=None):
    """Unitree's XML with the home pose, implicitfast and ccd 50. Returns (model, data, joint_names, actuator_names)."""
    return build_g1_model(UNITREE_XML, **UNITREE_MODEL_SETTINGS, decorate=decorate)


def load_unitree_policy(joint_names, actuator_names, obs_fn=make_unitree_observation) -> dict:
    """Unitree's policy with its home pose and 98-number observation."""
    return load_policy(UNITREE_POLICY, joint_names, actuator_names, pose=HOME, obs_fn=obs_fn)
