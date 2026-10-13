"""Running an ONNX walking policy on the MuJoCo G1 (Exercise 5)."""

from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

from g1_utils import default_joint_pos
from observation_utils import make_observation

REPO_ROOT = Path(__file__).resolve().parent.parent
DIEGO_POLICY = REPO_ROOT / "3p" / "2025-tfg-diego-lopez" / "pruebas" / "G1" / "2026-01-26_10-42-02.onnx"
DECIMATION = 4   # policy runs every 4 physics steps: 4 x 5 ms = 20 ms (50 Hz)


def load_policy(policy_path, joint_names, actuator_names) -> dict:
    """Load an ONNX policy plus everything needed to turn its actions into ctrl."""
    session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
    meta = session.get_modelmeta().custom_metadata_map   # name -> comma-separated string
    return {
        "session": session,
        "input_name": session.get_inputs()[0].name,
        "meta": meta,
        "default_pos": default_joint_pos(joint_names),                                 # joint order
        "action_scale": np.array([float(v) for v in meta["action_scale"].split(",")]),  # joint order
        # For each actuator, the index of its joint in joint order (not model.joint(name).id:
        # that counts the free joint as 0, so it would be off by one)
        "joint_index_of_actuator": np.array([joint_names.index(name) for name in actuator_names]),
    }


def action_to_ctrl(policy, action):
    """Raw policy output (joint order) -> position targets for data.ctrl (actuator order)."""
    target = policy["default_pos"] + policy["action_scale"] * action   # joint order
    return target[policy["joint_index_of_actuator"]]                   # actuator order


def make_policy_controller(policy, command, fresh_obs=False):
    """Controller to call before EVERY mj_step; it runs the policy on every DECIMATION-th call.

    command: [vx, vy, wz], or a function of simulation time t -> [vx, vy, wz] (for command changes)
    fresh_obs=False: build the observation straight after mj_step (stale IMU/gravity, as in Diego's notebook)
    fresh_obs=True:  mj_forward first (current IMU/gravity, as mjlab 1.2.0 does in training)
    """
    state = {"step": 0, "last_action": np.zeros(29)}

    def control(model, data):
        if state["step"] % DECIMATION == 0:
            if fresh_obs:
                mujoco.mj_forward(model, data)
            cmd = np.asarray(command(data.time) if callable(command) else command, dtype=float)
            obs = make_observation(model, data, policy["default_pos"], state["last_action"], cmd)
            action = policy["session"].run(None, {policy["input_name"]: obs[None, :]})[0][0]   # (1,99) -> (1,29)
            data.ctrl[:] = action_to_ctrl(policy, action)   # no clipping (clipping made Diego's robot fall)
            state["last_action"] = action                   # raw output goes into the next observation
        state["step"] += 1                                  # ctrl holds its value for the other 3 physics steps

    return control


def walk(model, data, policy, command, seconds=6.0, fresh_obs=False):
    """Run the policy from the keyframe. Returns (displacement xyz in m, pelvis heights per step)."""
    mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
    mujoco.mj_forward(model, data)
    start = data.qpos[0:3].copy()
    control = make_policy_controller(policy, command, fresh_obs)
    heights = []
    for _ in range(round(seconds / model.opt.timestep)):   # count PHYSICS steps; control decides when to act
        control(model, data)
        mujoco.mj_step(model, data)
        heights.append(data.qpos[2])
    return data.qpos[0:3] - start, np.array(heights)
