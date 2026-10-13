import mujoco
import numpy as np

# Layout from walk.onnx metadata (observation_names): name -> size
OBS_LAYOUT = [("base_lin_vel", 3), ("base_ang_vel", 3), ("projected_gravity", 3),
              ("joint_pos", 29), ("joint_vel", 29), ("actions", 29), ("command", 3)]
OBS_DIM = sum(size for _, size in OBS_LAYOUT)   # 99


def read_sensor(model, data, name):
    """Copy of one sensor's reading (copy: a view would change as the simulation runs)."""
    s = model.sensor(name)
    return data.sensordata[s.adr[0] : s.adr[0] + s.dim[0]].copy()

def projected_gravity(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Unit gravity direction [0, 0, -1] expressed in the pelvis frame (needs xmat to be current)."""
    R = data.xmat[model.body("pelvis").id].reshape(3, 3)   # pelvis frame -> world
    return R.T @ np.array([0.0, 0.0, -1.0])                 # world -> pelvis frame

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