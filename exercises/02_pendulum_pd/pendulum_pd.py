import sys
from pathlib import Path

import mujoco
import numpy as np

script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir.parent))  # make exercises/render_utils.py importable
from render_utils import make_camera, record_gif, render_image

from pendulum import pendulum_xml

# Create and load a pendulum
L = 0.5  # rod length (m)
xml = pendulum_xml(L)

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

print("\n=== Model sizes ===")
print(f"nq           = {model.nq}")  # 1: one hinge angle, no free joint (attached to world)
print(f"nv           = {model.nv}")  # 1: one angular velocity
print(f"nu           = {model.nu}")  # 0: no actuator yet

print("\n=== Joints ===")
header = f"{'id':>3} | {'name':<28} | {'type':<6} | {'qposadr':>7} | {'dofadr':>6} | range"
print(header)
print("-" * len(header))
for i in range(model.njnt):
    joint = model.joint(i)
    print(f"{i:>3} | {joint.name:<28} | {mujoco.mjtJoint(joint.type[0]).name.removeprefix('mjJNT_'):<6} | {joint.qposadr[0]:>7} | {joint.dofadr[0]:>6} | {joint.range}")
# range [0, 0] = no limits set (unlimited), not locked

# qpos = 0 means hanging straight down (geoms were defined along -z)
mujoco.mj_forward(model, data)
print(f"Joint position (rad) = {data.qpos[0]}")
print(f"ang vel (rad/s)      = {data.qvel[0]}")
render_image(model, data, script_dir / "pendulum_initial.png", show=True)


# Period Check
print("\n=== Period check ===")
duration = 5.0                  # simulated seconds (~3.5 periods)
g = abs(model.opt.gravity[2])   # magnitude: the formula needs g > 0
q0 = 0.1                        # initial angle (rad), small so sin(q) ≈ q holds

mujoco.mj_resetData(model, data)
data.qpos[0] = q0

zero_crossings = []             # times of + to - crossings: one per full swing
while data.time < duration:
    qprev = data.qpos[0]        # scalar index -> copy, not a view
    mujoco.mj_step(model, data)
    q = data.qpos[0]
    if qprev > 0 and q < 0:
        zero_crossings.append(data.time)

assert len(zero_crossings) >= 2, f"need >= 2 crossings for a period, got {len(zero_crossings)}"
measured_period = np.diff(zero_crossings).mean()
expected_period = 2 * np.pi * np.sqrt(L / g)   # small-angle point-mass pendulum
rel_error = abs(measured_period - expected_period) / expected_period

print(f"crossings (s)   = {np.round(zero_crossings, 3)}")
print(f"measured period = {measured_period:.4f} s")
print(f"expected period = {expected_period:.4f} s")
print(f"relative error  = {rel_error * 100:.2f} %")

# 1%: well above the expected gap (sphere inertia, finite amplitude, crossing times
# quantized to dt), well below a real modeling mistake (wrong L or g).
assert rel_error < 0.01, f"period error {rel_error:.2%} exceeds 1%"

# The expected error is around 0.2% - 0.4%, because of:
# 1. Sphere inertia (Not using rotational inertia for pendulum time period)
# 2. Assuming sin(theta) ~ theta in the formula
# 3. Crossing recorded error somewhere between 0 to dt
# 4. Semi-implicit Euler (negligble)

# Simulation video
print("\n=== Pendulum video ===")
cam = make_camera(lookat=[0, 0, -0.25], distance=1.5, azimuth=90, elevation=0)  # side view of the x-z swing plane

mujoco.mj_resetData(model, data)
data.qpos[0] = 0.5              # larger angle so the swing is clearly visible
n_frames = record_gif(
    model, data, script_dir / "pendulum_swing.gif",
    duration=3.0, camera=cam,   # ~2 periods, real time
    capture_fps=30,
)
print(f"saved {n_frames} frames to {script_dir / 'pendulum_swing.gif'}")

# PD Controller
print("\n=== PD controller ===")

# Motor: torque based controller
xml = pendulum_xml(L, '<motor joint="hinge"/>')  # motor: ctrl is applied directly as torque (N·m)

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

assert model.nu == 1, f"expected 1 actuator, got {model.nu}"

KP = 20.0       # stiffness (N·m/rad)
KD = 2.0        # damping (N·m·s/rad)
TARGET = 0.5    # desired angle (rad)

# PD controller produces torque from two terms
# KP (proportional) -> acts like a spring
# KD (derivative) -> acts like a damper
def pd_control(model, data):
    q, q_dot = data.qpos[0], data.qvel[0]
    data.ctrl[0] = KP * (TARGET - q) - KD * q_dot

duration = 3.0  # long enough to settle (damping ratio ≈ 0.4)

mujoco.mj_resetData(model, data)  # start at rest, hanging down

q_motor = []
while data.time < duration:
    pd_control(model, data)       # compute torque from the current state
    mujoco.mj_step(model, data)
    q_motor.append(data.qpos[0])
q_motor = np.array(q_motor)

q_final = q_motor[-1]
error = TARGET - q_final
m = model.body("pole").mass[0]            # named access returns a 1-element array
pd_torque = KP * error                    # PD torque at rest (q_dot = 0)
gravity_torque = m * g * L * np.sin(q_final)

print(f"final angle    = {q_final:.4f} rad (target {TARGET})")
print(f"error          = {error:.4f} rad")
print(f"PD torque      = {pd_torque:.4f} N·m")
print(f"gravity torque = {gravity_torque:.4f} N·m")

# Settled: no motion left, otherwise the torque balance below doesn't apply.
assert abs(data.qvel[0]) < 1e-3, f"not settled, q_dot = {data.qvel[0]:.4f}"
# At rest the PD only pushes when there is an error, so it settles below the target
# exactly where its torque cancels gravity: KP * error = m * g * L * sin(q).
assert abs(pd_torque - gravity_torque) / gravity_torque < 0.01, "torque balance failed"

# Simulation video
print("\n=== PD controller video ===")
mujoco.mj_resetData(model, data)  # start at rest so the video shows the rise to the target
n_frames = record_gif(
    model, data, script_dir / "pendulum_pd_motor.gif",
    duration=3.0, camera=cam,
    capture_fps=30, control=pd_control,  # PD runs before every mj_step
)
print(f"saved {n_frames} frames to {script_dir / 'pendulum_pd_motor.gif'}")


# Built-in Position Actuator (Part C)
# TODO


# Torque Limit and Armature (Part D)
# TODO
