import sys
from pathlib import Path

import mujoco
import numpy as np

script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir.parent))  # make exercises/render_utils.py importable
from render_utils import make_camera, record_gif, render_image

from pendulum import MOTOR, make_pd, metrics, pendulum_xml

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
print("\n=== Built-in position actuator ===")
xml = pendulum_xml(L, f'<position joint="hinge" kp="{KP}" kv="{KD}"/>')

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

assert model.nu == 1, f"expected 1 actuator, got {model.nu}"

# gainprm/biasprm are fixed-size rows of 10 (mjNGAIN = mjNBIAS = 10) shared by every
# actuator type; a position actuator only uses the first 3, the rest stay 0.
gain = model.actuator_gainprm[0, :3]   # row 0 = our only actuator
bias = model.actuator_biasprm[0, :3]
print(f"gainprm[:3] = {gain}")
print(f"biasprm[:3] = {bias}")

# MuJoCo actuator force:  force = gain[0]*ctrl + bias[0] + bias[1]*q + bias[2]*q_dot
# position actuator sets: gain = [KP, 0, 0], bias = [0, -KP, -KV]
#                      => force = KP*ctrl - KP*q - KV*q_dot = KP*(ctrl - q) - KV*q_dot
# i.e. the PD law from Part B, with ctrl as the target angle.
assert np.allclose(gain, [KP, 0, 0]), f"unexpected gainprm {gain}"
assert np.allclose(bias, [0, -KP, -KD]), f"unexpected biasprm {bias}"

# Run simulation
mujoco.mj_resetData(model, data)  # start at rest, hanging down

data.ctrl[0] = TARGET            # ctrl is now the target angle (rad), not a torque; it persists across steps
q_position = []
while data.time < duration:
    mujoco.mj_step(model, data)
    q_position.append(data.qpos[0])
q_position = np.array(q_position)

max_diff = np.max(np.abs(q_position - q_motor))   # element-wise difference between the two runs
print(f"max |q_position - q_motor| = {max_diff:.2e} rad")

# Same law, same state at the start of each step, same dt -> identical up to float rounding.
assert max_diff < 1e-9, f"built-in actuator differs from manual PD by {max_diff:.2e} rad"

# Bonus: Integrator vs. Stability
print("\n=== Bonus: integrator vs. stability (KD beyond the Euler limit) ===")
KD_BIG = 150  # above KD_max = 2*I/dt ≈ 100, where Euler diverges


def run_case(actuator, integrator, control=None, duration=3.0):
    """Run from rest; return (diverged, final angle, max |q|)."""
    model = mujoco.MjModel.from_xml_string(pendulum_xml(L, actuator, integrator=integrator))
    data = mujoco.MjData(model)
    data.ctrl[0] = TARGET  # only used by the position actuator; overwritten by `control` for the motor
    max_q = 0.0
    # Step count, not data.time: a blow-up makes MuJoCo reset data (time -> 0) and a time loop never ends.
    for _ in range(round(duration / model.opt.timestep)):
        if control is not None:
            control(model, data)
        mujoco.mj_step(model, data)
        max_q = max(max_q, abs(data.qpos[0]))
        if data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0 or abs(data.qpos[0]) > np.pi:
            return True, float("nan"), max_q
    return False, data.qpos[0], max_q


cases = {
    "manual PD (motor)": (MOTOR, make_pd(KP, KD_BIG, TARGET)),
    "built-in position": (f'<position joint="hinge" kp="{KP}" kv="{KD_BIG}"/>', None),
}
results = {}
print(f"{'controller':<20} | {'integrator':<12} | {'result':<10} | final (rad)")
print("-" * 60)
for name, (actuator, control) in cases.items():
    for integrator in ("Euler", "implicitfast"):
        diverged, final, max_q = run_case(actuator, integrator, control)
        results[(name, integrator)] = diverged
        print(f"{name:<20} | {integrator:<12} | {'DIVERGED' if diverged else 'stable':<10} | {final:.4f}")

# Euler: both controllers apply -KD*q_dot using the velocity at the START of the step.
# With KD*dt/I > 2, each step overcorrects by more than the error -> both diverge.
assert results[("manual PD (motor)", "Euler")] and results[("built-in position", "Euler")]
# implicitfast: MuJoCo knows the position actuator's force depends on velocity (-KV*q_dot),
# so it solves for the END-of-step velocity instead -> no overcorrection, stable for any KV.
assert not results[("built-in position", "implicitfast")]
# The manual PD is just a number written to ctrl; MuJoCo can't see that it depends on q_dot,
# so the integrator can't treat it implicitly and it still diverges.
assert results[("manual PD (motor)", "implicitfast")]


# Torque Limit and Armature (Part D)
print("\n=== D1: Torque limit ===")
FORCE_LIMIT = 1.0  # max actuator torque (N·m)
xml = pendulum_xml(
    L,
    f'<position joint="hinge" kp="{KP}" kv="{KD}" '
    f'forcelimited="true" forcerange="{-FORCE_LIMIT} {FORCE_LIMIT}"/>',  # clamp force to ±FORCE_LIMIT
)

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

assert model.nu == 1, f"expected 1 actuator, got {model.nu}"

# Before running: torque needed to hold the pendulum where the unlimited PD settled (Part B)
q_rest = q_motor[-1]
hold_torque = m * g * L * np.sin(q_rest)
print(f"torque to hold {q_rest:.3f} rad = {hold_torque:.3f} N·m vs limit {FORCE_LIMIT} N·m")

# Run 6 s from rest with the target held in ctrl
mujoco.mj_resetData(model, data)
data.ctrl[0] = TARGET
q_limited, force_limited = [], []
for _ in range(round(6.0 / model.opt.timestep)):
    mujoco.mj_step(model, data)
    q_limited.append(data.qpos[0])
    force_limited.append(data.actuator_force[0])   # force actually applied, after clamping
q_limited = np.array(q_limited)
force_limited = np.array(force_limited)

# Angle where gravity's torque equals the most the actuator can give: m*g*L*sin(q) = FORCE_LIMIT
q_balance = np.arcsin(FORCE_LIMIT / (m * g * L))
last = q_limited[-round(1.5 / model.opt.timestep):]  # last 1.5 s (~1 swing)

print(f"max |actuator force| = {np.abs(force_limited).max():.4f} N·m")
print(f"max angle            = {q_limited.max():.4f} rad (target {TARGET})")
print(f"balance angle        = {q_balance:.4f} rad")
print(f"last 1.5 s: mean     = {last.mean():.4f} rad, range {last.min():.3f} .. {last.max():.3f}")

assert np.abs(force_limited).max() <= FORCE_LIMIT + 1e-9, "force exceeded the limit"
assert q_limited.max() < TARGET, "torque-limited pendulum reached the target"
assert abs(last.mean() - q_balance) < 0.05, "not centred on the balance angle"

# Actuator saturation: the PD asks for more than 1 N·m throughout, so the applied torque is a
# constant +1 N·m. The -KD*q_dot term is clamped away with it -> no damping, so the pendulum
# swings forever around q_balance (gravity torque = force limit).

# Video: start at rest, watch it rise and keep swinging short of the target
mujoco.mj_resetData(model, data)
data.ctrl[0] = TARGET
n_frames = record_gif(
    model, data, script_dir / "pendulum_torque_limit.gif",
    duration=6.0, camera=cam, capture_fps=30,
    label=f"position KP={KP:g} KD={KD:g}  limit ±{FORCE_LIMIT:g} N·m",
)
print(f"saved {n_frames} frames to {script_dir / 'pendulum_torque_limit.gif'}")

print("\n=== D2: Armature ===")
ARMATURE = 0.1  # kg·m², reflected rotor inertia added to the joint
POSITION = f'<position joint="hinge" kp="{KP}" kv="{KD}"/>'


def run_position(armature, duration=3.0):
    """Position actuator from rest, ctrl = TARGET. Returns (t, q, force, inertia, model, data)."""
    model = mujoco.MjModel.from_xml_string(pendulum_xml(L, POSITION, armature=armature))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    inertia = data.M[0]                 # joint-space inertia MuJoCo uses (includes armature)
    data.ctrl[0] = TARGET
    t, q, force = [], [], []
    for _ in range(round(duration / model.opt.timestep)):
        mujoco.mj_step(model, data)
        t.append(data.time)
        q.append(data.qpos[0])
        force.append(data.actuator_force[0])
    return np.array(t), np.array(q), np.array(force), inertia, model, data


t0, q0_arm, f0, I0, _, _ = run_position(0.0)
t1, q1_arm, f1, I1, model, data = run_position(ARMATURE)
m0 = metrics(t0, q0_arm, f0, diverged=False)
m1 = metrics(t1, q1_arm, f1, diverged=False)

# Linear theory: armature adds straight to the inertia; stiffness is unchanged
k_eff = KP + m * g * L * np.cos(q_rest)
for name, I_, met in (("no armature", I0, m0), (f"armature {ARMATURE}", I1, m1)):
    wn = np.sqrt(k_eff / I_)
    zeta = KD / (2 * np.sqrt(k_eff * I_))
    print(f"{name:<14} | I = {I_:.3f} | ω_n = {wn:5.2f} rad/s | ζ = {zeta:.2f} | "
          f"final = {met['final']:.4f} | overshoot = {met['overshoot']:.4f} | settle = {met['settling_time']:.2f} s")

# MuJoCo adds armature directly to the joint's inertia
assert abs(I1 - (I0 + ARMATURE)) < 1e-9, f"inertia {I1} != {I0} + {ARMATURE}"
# Same gains, more inertia: slower (lower ω_n), less damped (lower ζ) -> more overshoot, longer settling
assert m1["overshoot"] > m0["overshoot"] and m1["settling_time"] > m0["settling_time"]
# Static balance KP*error = m*g*L*sin(q) has no inertia in it -> same final angle
assert abs(m1["final"] - m0["final"]) < 1e-3

# Video with armature, from rest
mujoco.mj_resetData(model, data)
data.ctrl[0] = TARGET
n_frames = record_gif(
    model, data, script_dir / "pendulum_armature.gif",
    duration=3.0, camera=cam, capture_fps=30,
    label=f"position KP={KP:g} KD={KD:g}  armature={ARMATURE:g}",
)
print(f"saved {n_frames} frames to {script_dir / 'pendulum_armature.gif'}")

