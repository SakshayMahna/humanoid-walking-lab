import sys
from pathlib import Path

import mujoco
import numpy as np

script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent.parent
sys.path.insert(0, str(script_dir.parent))  # make exercises/*_utils.py importable
from g1_utils import build_g1_model
from observation_utils import OBS_DIM, OBS_LAYOUT
from policy_utils import DIEGO_POLICY, action_to_ctrl, load_policy, make_policy_controller, walk
from render_utils import make_camera, record_gif

model, data, joint_names, actuator_names = build_g1_model(repo_root / "robot" / "g1.xml")
policy = load_policy(DIEGO_POLICY, joint_names, actuator_names)

# Part A: Load the policy and check its metadata against our model
print("\n=== Part A: Policy vs model ===")
session = policy["session"]
policy_input, policy_output = session.get_inputs()[0], session.get_outputs()[0]   # lists of one each
print(f"input  {policy_input.name} {policy_input.shape}, output {policy_output.name} {policy_output.shape}")
assert policy_input.shape == [1, OBS_DIM] and policy_output.shape == [1, 29]

meta = policy["meta"]   # name -> comma-separated string


def meta_names(key):
    return meta[key].split(",")


def meta_floats(key):
    return np.array([float(v) for v in meta[key].split(",")])


def per_joint(values_by_actuator):
    """Reorder an actuator-ordered array into joint order (the policy's order)."""
    return np.array([values_by_actuator[model.actuator(name).id] for name in joint_names])


default_pos = policy["default_pos"]
kp = per_joint(model.actuator_gainprm[:, 0])          # gainprm[0] = KP
kd = per_joint(-model.actuator_biasprm[:, 2])         # biasprm[2] = -KD
effort = per_joint(model.actuator_forcerange[:, 1])   # upper force limit
action_scale = policy["action_scale"]                # used in Part B (loaded from metadata)

# Metadata numbers are rounded to 3 decimals, hence atol=1e-3
checks = [
    ("joint_names",       joint_names == meta_names("joint_names")),
    ("observation_names", [name for name, _ in OBS_LAYOUT] == meta_names("observation_names")),
    ("default_joint_pos", np.allclose(default_pos, meta_floats("default_joint_pos"), atol=1e-3)),
    ("joint_stiffness",   np.allclose(kp, meta_floats("joint_stiffness"), atol=1e-3)),
    ("joint_damping",     np.allclose(kd, meta_floats("joint_damping"), atol=1e-3)),
    ("action_scale",      np.allclose(0.25 * effort / kp, action_scale, atol=1e-3)),   # 0.25 * effort / KP
]
for name, ok in checks:
    print(f"  {name:<18} {'OK' if ok else 'MISMATCH'}")
assert all(ok for _, ok in checks), "model does not match what the policy was trained with"

print(f"  first joints: KP {np.round(kp[:3], 2)}, KD {np.round(kd[:3], 2)}, scale {action_scale[:3]}")
print(f"  run_path: {meta['run_path']}, command: {meta['command_names']}")

# Part B: Actions -> ctrl
print("\n=== Part B: Actions -> ctrl ===")

# action_to_ctrl lives in policy_utils.py: target = default + scale * action, reordered to actuator order
# Zero action = hold the default pose = exactly the keyframe's ctrl
zero_ctrl = action_to_ctrl(policy, np.zeros(29))
assert np.allclose(zero_ctrl, model.key("init_state").ctrl), "zero action should give the keyframe ctrl"

# +1 on one joint must change only that joint's actuator, by exactly its scale
test_joint = "left_knee_joint"
action = np.zeros(29)
action[joint_names.index(test_joint)] = 1.0
changed = np.nonzero(~np.isclose(action_to_ctrl(policy, action), zero_ctrl))[0]
print(f"+1 on {test_joint}: changed ctrl index {changed} = actuator {[actuator_names[i] for i in changed]}, "
      f"by {(action_to_ctrl(policy, action) - zero_ctrl)[changed]}")
assert len(changed) == 1 and actuator_names[changed[0]] == test_joint, "reordering is wrong"
assert np.isclose((action_to_ctrl(policy, action) - zero_ctrl)[changed[0]], action_scale[joint_names.index(test_joint)])

# Part C: Control loop (50 Hz policy, 200 Hz physics)
# make_policy_controller and walk live in policy_utils.py

print("\n=== Part C: Walking ===")
command = np.array([0.6, 0.0, 0.0])   # forward 0.6 m/s, as in Diego's walk_fixed.ipynb
displacement, heights = walk(model, data, policy, command)
print(f"command {command} for 6 s (stale obs)")
print(f"  moved x {displacement[0]:+.2f} m, y {displacement[1]:+.2f} m   (perfect tracking: {command[0] * 6:.1f} m)")
print(f"  pelvis height min {heights.min():.3f} m, final {heights[-1]:.3f} m")

# Video: same controller, driven by record_gif before every physics step
cam = make_camera(lookat=[1.5, 0, 0.6], distance=4.0, azimuth=110, elevation=-12)   # side-on view of the walk
mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
mujoco.mj_forward(model, data)
n = record_gif(model, data, script_dir / "walk_forward.gif", duration=6.0, camera=cam,
               control=make_policy_controller(policy, command), capture_fps=20,
               label=f"walk.onnx  command vx={command[0]} m/s")
print(f"  saved {n} frames to walk_forward.gif")

# Part D: Did it walk?
print("\n=== Part D: Did it walk? ===")
# Walking = moved forward a good fraction of the command, and never fell (pelvis stayed above ~0.6 m)
walked = displacement[0] > 0.5 * command[0] * 6 and heights.min() > 0.6
print(f"  walked: {walked}")
assert walked, "robot did not walk forward without falling"

# Stale vs fresh observations: does the observation timing change the gait?
for fresh in (False, True):
    d, h = walk(model, data, policy, command, fresh_obs=fresh)
    print(f"  fresh_obs={str(fresh):<5}: x {d[0]:+.2f} m, y {d[1]:+.2f} m, min height {h.min():.3f} m")
