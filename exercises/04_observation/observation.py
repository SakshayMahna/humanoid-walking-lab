import sys
from pathlib import Path

import mujoco
import numpy as np

script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent.parent
sys.path.insert(0, str(script_dir.parent))  # make exercises/*_utils.py importable
from g1_utils import build_g1_model, default_joint_pos

model, data, joint_names, actuator_names = build_g1_model(repo_root / "robot" / "g1.xml")

# Part A: IMU sensors and their frames
print("\n=== Part A: IMU sensors ===")


def sensor(name):
    """Slice of sensordata for one sensor (a view: it changes when the simulation does)."""
    s = model.sensor(name)
    return data.sensordata[s.adr[0] : s.adr[0] + s.dim[0]]


# A known tilted, moving state (all zeros would prove nothing)
test_quat = np.array([np.cos(0.15), 0.1, 0.13, 0.2])
data.qpos[3:7] = test_quat / np.linalg.norm(test_quat)   # MuJoCo expects unit quaternions
data.qvel[0:3] = [0.5, -0.2, 0.1]   # free joint linear velocity: WORLD frame (m/s)
data.qvel[3:6] = [0.3, 0.4, -0.6]   # free joint angular velocity: PELVIS frame (rad/s)
mujoco.mj_forward(model, data)      # state changed: recompute sensors

# Gyro: IMU site isn't rotated relative to the pelvis, so it reads qvel[3:6] directly
gyro_error = np.abs(sensor("imu_ang_vel") - data.qvel[3:6]).max()
print(f"gyro vs qvel[3:6]:                  {gyro_error:.1e}")
assert gyro_error < 1e-9, "gyro should equal the pelvis-frame angular velocity"

# Velocimeter: world -> pelvis frame (R.T), plus the IMU's offset from the pelvis origin (w x r)
R = data.xmat[model.body("pelvis").id].reshape(3, 3)   # pelvis frame -> world
w = data.qvel[3:6]
r = model.site("imu_in_pelvis").pos                     # IMU position in the pelvis frame
v_pelvis = R.T @ data.qvel[0:3]                         # pelvis-origin velocity, pelvis frame
v_imu = v_pelvis + np.cross(w, r)                       # IMU-point velocity, pelvis frame

velo_error = np.abs(sensor("imu_lin_vel") - v_imu).max()
print(f"velocimeter vs R.T v + w x r:       {velo_error:.1e}")
print(f"velocimeter vs R.T v (no offset):   {np.abs(sensor('imu_lin_vel') - v_pelvis).max():.3f} m/s")
assert velo_error < 1e-9, "velocimeter should equal the IMU-point velocity in the pelvis frame"

# Part B: Projected gravity
print("\n=== Part B: Projected gravity ===")


def projected_gravity(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Unit gravity direction [0, 0, -1] expressed in the pelvis frame (needs xmat to be current)."""
    R = data.xmat[model.body("pelvis").id].reshape(3, 3)   # pelvis frame -> world
    return R.T @ np.array([0.0, 0.0, -1.0])                 # world -> pelvis frame


def set_base_rotation(axis, angle_deg):
    """Reset to the keyframe, then rotate the base by angle about axis (x/y/z unit vector)."""
    mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
    half = np.radians(angle_deg) / 2
    data.qpos[3:7] = np.concatenate([[np.cos(half)], np.sin(half) * np.array(axis)])   # [w, x, y, z]
    mujoco.mj_forward(model, data)   # recompute xmat for the new orientation


s, c = np.sin(np.radians(30)), np.cos(np.radians(30))
cases = [
    # name,         axis,      angle, expected (pelvis frame)
    ("upright",     [0, 1, 0],   0, [0, 0, -1]),
    ("pitch +30°",  [0, 1, 0],  30, [s, 0, -c]),    # nose down: "down" leans forward (+x)
    ("roll +30°",   [1, 0, 0],  30, [0, -s, -c]),   # left side up: "down" leans to the right (-y)
]
for name, axis, angle, expected in cases:
    set_base_rotation(axis, angle)
    g = projected_gravity(model, data)
    print(f"{name:<11} -> {np.round(g, 3)}   expected {np.round(expected, 3)}")
    assert np.allclose(g, expected), f"{name}: got {g}, expected {expected}"
    assert np.isclose(np.linalg.norm(g), 1.0), "a rotation must keep the vector's length"

# Part C: Assemble the 99-number observation
print("\n=== Part C: Observation ===")

# Layout from walk.onnx metadata (observation_names): name -> size
OBS_LAYOUT = [("base_lin_vel", 3), ("base_ang_vel", 3), ("projected_gravity", 3),
              ("joint_pos", 29), ("joint_vel", 29), ("actions", 29), ("command", 3)]
OBS_DIM = sum(size for _, size in OBS_LAYOUT)   # 99


def read_sensor(model, data, name):
    """Copy of one sensor's reading (copy: a view would change as the simulation runs)."""
    s = model.sensor(name)
    return data.sensordata[s.adr[0] : s.adr[0] + s.dim[0]].copy()


def make_observation(model: mujoco.MjModel, data: mujoco.MjData, default_pos, last_action, command) -> np.ndarray:
    """The 99 numbers walk.onnx expects, in its order. Joint terms are in joint order (qpos[7:], qvel[6:])."""
    obs = np.concatenate([
        read_sensor(model, data, "imu_lin_vel"),   # IMU-point velocity, pelvis frame
        read_sensor(model, data, "imu_ang_vel"),   # angular velocity, pelvis frame
        projected_gravity(model, data),            # "down" in the pelvis frame
        data.qpos[7:] - default_pos,               # joint angles relative to the default pose
        data.qvel[6:],                             # joint velocities
        last_action,                               # previous raw policy output
        command,                                   # [vx, vy, wz]
    ]).astype(np.float32)                          # ONNX expects float32
    assert obs.shape == (OBS_DIM,), f"observation has {obs.shape[0]} numbers, expected {OBS_DIM}"
    return obs


def print_observation(obs):
    """Each section with its index range."""
    start = 0
    for name, size in OBS_LAYOUT:
        part = obs[start : start + size]
        shown = np.round(part, 3) if size <= 3 else f"max |value| = {np.abs(part).max():.3f}"
        print(f"  [{start:>2}:{start + size:>2}] {name:<17} {shown}")
        start += size


default_pos = default_joint_pos(joint_names)   # knees-bent pose, joint order
command = np.array([0.5, 0.0, 0.0])            # walk forward at 0.5 m/s

mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
mujoco.mj_forward(model, data)
obs = make_observation(model, data, default_pos, np.zeros(29), command)
print_observation(obs)

# At the keyframe the robot is still and exactly in its default pose
assert obs.shape == (99,) and obs.dtype == np.float32
assert np.allclose(obs[0:6], 0, atol=1e-6), "velocities should be zero at rest"   # slice end is exclusive: 0..5
assert np.allclose(obs[6:9], [0, 0, -1], atol=1e-6), "gravity should point straight down when upright"
assert np.allclose(obs[9:38], 0, atol=1e-6), "joint positions should equal the default pose"
assert np.allclose(obs[-3:], command), "command should be the last 3 numbers"


# Part D: Forward vs step lag
# mj_step = [compute sensors, xmat, ... from the CURRENT state] then [integrate qpos/qvel by dt].
# So right after mj_step, sensordata/xmat describe the state at the START of the step, while
# qpos/qvel are already at the END. One mj_forward recomputes them from the end-of-step state.
print("\n=== Part D: Forward vs step lag ===")


def section_diffs(a, b):
    """Max |a - b| per observation section."""
    diffs, start = {}, 0
    for name, size in OBS_LAYOUT:
        diffs[name] = np.abs(a[start : start + size] - b[start : start + size]).max()
        start += size
    return diffs


def stale_and_fresh(model, data):
    """One mj_step, then the observation before and after an extra mj_forward."""
    mujoco.mj_step(model, data)
    obs_stale = make_observation(model, data, default_pos, np.zeros(29), command)   # Diego's notebook

    qpos_before, qvel_before = data.qpos.copy(), data.qvel.copy()   # .copy(): a plain = would alias
    mujoco.mj_forward(model, data)
    assert np.array_equal(data.qpos, qpos_before) and np.array_equal(data.qvel, qvel_before), \
        "mj_forward must not change the state"

    obs_fresh = make_observation(model, data, default_pos, np.zeros(29), command)   # mjlab 1.2.0
    return obs_stale, obs_fresh


# 1. Let the robot move: without the policy it sinks and tips, so it has real velocities after 0.5 s
mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
for _ in range(round(0.5 / model.opt.timestep)):
    mujoco.mj_step(model, data)

# 2-4. One more step, then compare the two observations section by section
obs_stale, obs_fresh = stale_and_fresh(model, data)
print("one step at t = 0.5 s (max |fresh - stale| per section):")
for name, diff in section_diffs(obs_stale, obs_fresh).items():
    print(f"  {name:<17} {diff:.5f}")

# Only the terms computed by forward (sensors, xmat) lag; qpos/qvel terms are already current
d = section_diffs(obs_stale, obs_fresh)
assert d["joint_pos"] == 0 and d["joint_vel"] == 0, "joint terms come from qpos/qvel, which never lag"
assert d["base_ang_vel"] > 0, "the robot is moving, so the IMU reading should lag"

# 5. Track the worst lag per section while the robot falls (2 s from the keyframe)
mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
worst = {name: 0.0 for name, _ in OBS_LAYOUT}
for _ in range(round(2.0 / model.opt.timestep)):
    for name, diff in section_diffs(*stale_and_fresh(model, data)).items():
        worst[name] = max(worst[name], diff)
print("worst lag over a 2 s fall:")
for name, diff in worst.items():
    print(f"  {name:<17} {diff:.5f}")

