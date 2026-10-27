"""Load a trained G1 velocity policy for plain MuJoCo, with its observation built from its own ONNX metadata.

Unitree's policy (and our baseline) observe 98 numbers; variants can drop the gait clock (96) or add the base
linear velocity (101). The ONNX file lists its observation terms in order (`observation_names`), so the observation
is assembled from those names, with the same definitions as exercises/unitree_utils.py.
"""

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "exercises"))
from observation_utils import projected_gravity, read_sensor  # noqa: E402
from policy_utils import load_policy  # noqa: E402
from unitree_utils import HOME, gait_phase  # noqa: E402

TERMS = {
    "base_lin_vel": lambda m, d, default, last, cmd: read_sensor(m, d, "imu_lin_vel"),   # pelvis frame
    "base_ang_vel": lambda m, d, default, last, cmd: read_sensor(m, d, "imu_ang_vel"),
    "projected_gravity": lambda m, d, default, last, cmd: projected_gravity(m, d),
    "command": lambda m, d, default, last, cmd: cmd,
    "phase": lambda m, d, default, last, cmd: gait_phase(d.time, cmd),
    "joint_pos": lambda m, d, default, last, cmd: d.qpos[7:] - default,
    "joint_vel": lambda m, d, default, last, cmd: d.qvel[6:],
    "actions": lambda m, d, default, last, cmd: last,
}


def observation_names(path):
    meta = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).get_modelmeta().custom_metadata_map
    return meta["observation_names"].split(",")


def make_observation_fn(names):
    unknown = [n for n in names if n not in TERMS]
    assert not unknown, f"observation terms this loader doesn't know: {unknown}"

    def obs_fn(model, data, default_pos, last_action, command):
        return np.concatenate([TERMS[n](model, data, default_pos, last_action, command) for n in names]).astype(np.float32)
    return obs_fn


def load_trained(path, joint_names, actuator_names):
    """load_policy with Unitree's home pose and the observation the file declares."""
    return load_policy(path, joint_names, actuator_names, pose=HOME, obs_fn=make_observation_fn(observation_names(path)))
