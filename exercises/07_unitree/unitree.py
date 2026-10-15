import sys
from pathlib import Path

import mujoco
import numpy as np

script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent.parent
sys.path.insert(0, str(script_dir.parent))  # make exercises/*_utils.py importable
from g1_utils import build_g1_model, default_joint_pos
from policy_utils import DIEGO_POLICY, load_policy, make_policy_controller, steady_state, trajectory, walk
from render_utils import make_camera, record_frames, record_gif, render_image, save_gif, side_by_side
from unitree_utils import (HOME, HOME_HEIGHT, UNITREE_OBS_DIM, UNITREE_OBS_LAYOUT, UNITREE_POLICY,
                           UNITREE_XML, gait_phase, make_unitree_observation)

# Part A: Unitree's model (home pose, implicitfast, ccd 50)
print("\n=== Part A: Unitree model ===")
# HOME pose (HOME_KEYFRAME in Unitree's g1_constants.py) lives in unitree_utils.py

model, data, joint_names, actuator_names = build_g1_model(
    UNITREE_XML, pose=HOME, base_height=HOME_HEIGHT,
    integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST,   # mjlab's training default
    ccd_iterations=50,                                    # flat task; no effect here (only capsules/spheres collide)
)

checks = [
    ("29 actuators",            model.nu == 29),
    ("integrator implicitfast", model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST),
    ("ccd_iterations 50",       model.opt.ccd_iterations == 50),
    ("pelvis starts at 0.8 m",  np.isclose(data.qpos[2], HOME_HEIGHT)),
    ("knees at 0.3 rad",        np.isclose(data.qpos[7 + joint_names.index("left_knee_joint")], 0.3)),
    # Unitree's XML defaults collision geoms to priority=1 condim=6; apply_collision_settings overrides them
    ("foot geom condim 3",      model.geom("left_foot3_collision").condim[0] == 3),
    ("other collision condim 1", model.geom("pelvis_collision").condim[0] == 1
                                 and model.geom("pelvis_collision").priority[0] == 0),
]
for name, ok in checks:
    print(f"  {'OK  ' if ok else 'FAIL'} {name}")
assert all(ok for _, ok in checks), "Unitree model setup is wrong"

cam = make_camera(lookat=[0, 0, 0.5], distance=2.6, azimuth=150, elevation=-12)
render_image(model, data, script_dir / "init_pose.png", camera=cam)
print("  saved init_pose.png")

# Part B: Gait phase
print("\n=== Part B: Gait phase ===")
# gait_phase lives in unitree_utils.py: [sin 2πφ, cos 2πφ], φ = (t mod 0.6) / 0.6; [0, 0] if |command| < 0.1


walk_cmd = [0.5, 0.0, 0.0]
cases = [
    # name,                          t,     command,          expected
    ("start of stride",              0.0,   walk_cmd,         [0, 1]),
    ("quarter stride (0.15 s)",      0.15,  walk_cmd,         [1, 0]),
    ("half stride (0.3 s)",          0.3,   walk_cmd,         [0, -1]),
    ("full stride wraps (0.6 s)",    0.6,   walk_cmd,         [0, 1]),
    ("zero command: clock off",      0.15,  [0, 0, 0],        [0, 0]),
    ("tiny command: clock off",      0.15,  [0.05, 0, 0.05],  [0, 0]),     # ‖·‖ = 0.07 < 0.1
    ("turn in place: clock ON",      0.15,  [0, 0, 0.8],      [1, 0]),     # ‖·‖ = 0.8 -> keeps stepping
]
for name, t, cmd, expected in cases:
    phase = gait_phase(t, cmd)
    ok = np.allclose(phase, expected, atol=1e-9)
    print(f"  {'OK  ' if ok else 'FAIL'} {name:<26} -> {np.round(phase, 3)}")
    assert ok, f"{name}: got {phase}, expected {expected}"

# Part C: 98-number observation
print("\n=== Part C: Observation ===")
# UNITREE_OBS_LAYOUT and make_unitree_observation live in unitree_utils.py
# (ang_vel, gravity, command, phase, joint_pos, joint_vel, actions = 98 numbers)

default_pos = default_joint_pos(joint_names, HOME)   # HOME, not Diego's KNEES_BENT
command = np.array([0.5, 0.0, 0.0])
mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
mujoco.mj_forward(model, data)
obs = make_unitree_observation(model, data, default_pos, np.zeros(29), command)

start = 0
for name, size in UNITREE_OBS_LAYOUT:
    part = obs[start : start + size]
    shown = np.round(part, 3) if size <= 3 else f"max |value| = {np.abs(part).max():.3f}"
    print(f"  [{start:>2}:{start + size:>2}] {name:<17} {shown}")
    start += size

assert obs.shape == (98,) and obs.dtype == np.float32
assert np.allclose(obs[0:3], 0, atol=1e-6), "angular velocity should be zero at rest"
assert np.allclose(obs[3:6], [0, 0, -1], atol=1e-6), "gravity should point straight down"
assert np.allclose(obs[6:9], command), "command should be at 6:9"
assert np.allclose(obs[9:11], [0, 1]), "phase at t = 0 with a walking command should be [0, 1]"
assert np.allclose(obs[11:40], 0, atol=1e-6), "joint positions should equal the home pose"

# Part D: Policy check, walk, dead-zone comparison
print("\n=== Part D1: Unitree policy vs model ===")
unitree = load_policy(UNITREE_POLICY, joint_names, actuator_names, pose=HOME, obs_fn=make_unitree_observation)
session, meta = unitree["session"], unitree["meta"]
assert session.get_inputs()[0].shape == [1, UNITREE_OBS_DIM] and session.get_outputs()[0].shape == [1, 29]


def meta_floats(key):
    return np.array([float(v) for v in meta[key].split(",")])


def per_joint(values_by_actuator):
    """Reorder an actuator-ordered array into joint order (the policy's order)."""
    return np.array([values_by_actuator[model.actuator(name).id] for name in joint_names])


kp = per_joint(model.actuator_gainprm[:, 0])
kd = per_joint(-model.actuator_biasprm[:, 2])
effort = per_joint(model.actuator_forcerange[:, 1])
checks = [   # metadata is rounded to 3 decimals
    ("joint_names",       joint_names == meta["joint_names"].split(",")),
    ("observation_names", [name for name, _ in UNITREE_OBS_LAYOUT] == meta["observation_names"].split(",")),
    ("default_joint_pos", np.allclose(unitree["default_pos"], meta_floats("default_joint_pos"), atol=1e-3)),
    ("joint_stiffness",   np.allclose(kp, meta_floats("joint_stiffness"), atol=1e-3)),
    ("joint_damping",     np.allclose(kd, meta_floats("joint_damping"), atol=1e-3)),
    ("action_scale",      np.allclose(0.25 * effort / kp, unitree["action_scale"], atol=1e-3)),
]
for name, ok in checks:
    print(f"  {'OK  ' if ok else 'MISMATCH'} {name}")
assert all(ok for _, ok in checks), "model does not match what Unitree's policy was trained with"

print("\n=== Part D2: Walk ===")
displacement, heights = walk(model, data, unitree, np.array([0.5, 0.0, 0.0]), seconds=6.0)
print(f"  command [0.5, 0, 0] for 6 s: moved x {displacement[0]:+.2f} m, y {displacement[1]:+.2f} m, "
      f"min height {heights.min():.3f} m")
assert displacement[0] > 1.5 and heights.min() > 0.6, "Unitree policy did not walk forward"

print("\n=== Part D3: Dead zone, Diego vs Unitree (steady state, last 3 s of 8 s) ===")
d_model, d_data, d_joints, d_actuators = build_g1_model(repo_root / "robot" / "g1.xml")   # Diego's defaults
diego = load_policy(DIEGO_POLICY, d_joints, d_actuators)

SWEEP = [[0.2, 0, 0], [0.3, 0, 0], [0.4, 0, 0], [0.5, 0, 0], [0.8, 0, 0],
         [0, 0.3, 0], [0, 0.5, 0],
         [0, 0, 0.3], [0, 0, 0.5], [0, 0, 0.8],
         [0.5, 0, 0.5], [0, 0, 0]]
print(f"{'command':<16} | {'Diego  speed  wz':<20} | {'Unitree speed  wz':<20}")
print("-" * 62)
sweep = {}
for cmd in SWEEP:
    row = []
    for m, d, p in ((d_model, d_data, diego), (model, data, unitree)):
        v, wz, hmin = steady_state(m, d, p, np.array(cmd, dtype=float))
        assert hmin > 0.6, f"fell during {cmd}"
        row.append((np.linalg.norm(v), wz))
    sweep[tuple(cmd)] = row
    (dv, dw), (uv, uw) = row
    print(f"{str(cmd):<16} | {dv:5.2f} m/s {dw:+5.2f} rad/s | {uv:5.2f} m/s {uw:+5.2f} rad/s")

# Part E: Experiment - is the gait clock what makes Unitree's policy step at small commands?
print("\n=== Part E: Gait clock experiment (steady state, last 3 s of 8 s) ===")


def make_obs_with_phase(phase_override):
    """Unitree observation with the phase entries (9:11) replaced by a fixed value."""
    def obs_fn(model, data, default_pos, last_action, command):
        obs = make_unitree_observation(model, data, default_pos, last_action, command)
        obs[9:11] = phase_override
        return obs
    return obs_fn


VARIANTS = {
    "normal clock":         make_unitree_observation,
    "clock off [0, 0]":     make_obs_with_phase([0.0, 0.0]),   # the "stand still" signal from training
    "clock frozen [0, 1]":  make_obs_with_phase([0.0, 1.0]),   # clock on, but not ticking
}
EXPERIMENT = [[0.0, 0.0, 0.8], [0.3, 0.0, 0.0], [0.0, 0.3, 0.0], [0.6, 0.0, 0.0]]
print(f"{'command':<16} | " + " | ".join(f"{name:<20}" for name in VARIANTS))
print("-" * 86)
experiment = {}
for cmd in EXPERIMENT:
    cells = []
    for name, obs_fn in VARIANTS.items():
        variant = load_policy(UNITREE_POLICY, joint_names, actuator_names, pose=HOME, obs_fn=obs_fn)
        v, wz, hmin = steady_state(model, data, variant, np.array(cmd))
        experiment[(tuple(cmd), name)] = (np.linalg.norm(v), wz, hmin)
        fell = " FELL" if hmin < 0.6 else ""
        cells.append(f"{np.linalg.norm(v):4.2f} m/s {wz:+5.2f} rad/s{fell}")
    print(f"{str(cmd):<16} | " + " | ".join(f"{c:<20}" for c in cells))


# Part F: The Day 6 GIF set for Unitree's policy
print("\n=== Part F: GIFs ===")
gif_dir = script_dir / "gifs"
gif_dir.mkdir(exist_ok=True)

# Same commands as exercises/06_validation (Diego) so the GIFs can be compared one to one
COMMANDS = {
    "forward":        [0.6, 0.0, 0.0],
    "backward":       [-0.5, 0.0, 0.0],
    "sideways left":  [0.0, 0.5, 0.0],
    "walk and turn":  [0.5, 0.0, 0.5],
    "slow forward":   [0.3, 0.0, 0.0],
    "slow sideways":  [0.0, 0.3, 0.0],
    "turn in place":  [0.0, 0.0, 0.8],
    "stand":          [0.0, 0.0, 0.0],
}
SCHEDULE = [([0.5, 0.0, 0.8], 2.0), ([0.0, 0.0, 0.0], 2.0), ([-0.5, 0.0, 0.0], 2.0), ([0.0, 0.0, 0.8], 2.0)]


def command_at(t):
    """Scheduled command at time t (Diego's control.ipynb pattern)."""
    end = 0.0
    for cmd, duration in SCHEDULE:
        end += duration
        if t < end:
            return cmd
    return SCHEDULE[-1][0]


def save_command_gif(name, command, seconds, label):
    """Run once to find the path, then record the same run with the camera centred on it."""
    xy, _, _ = trajectory(model, data, unitree, command, seconds)
    center = (xy.min(axis=0) + xy.max(axis=0)) / 2
    cam = make_camera(lookat=[center[0], center[1], 0.6], distance=3.5 + np.ptp(xy, axis=0).max(),
                      azimuth=120, elevation=-15)
    mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
    mujoco.mj_forward(model, data)
    record_gif(model, data, gif_dir / f"{name}.gif", duration=seconds, camera=cam,
               control=make_policy_controller(unitree, command), capture_fps=20, width=480, height=360,
               label=label)


for name, cmd in COMMANDS.items():
    save_command_gif(name.replace(" ", "_"), np.array(cmd), 5.0, f"Unitree {name}: vx={cmd[0]} vy={cmd[1]} wz={cmd[2]}")
save_command_gif("schedule", command_at, sum(d for _, d in SCHEDULE),
         lambda d: f"Unitree t={d.time:4.1f} s  command {command_at(d.time)}")
print(f"saved {len(COMMANDS) + 1} GIFs to {gif_dir}")


# Part G: Experiment GIFs - one per command, panels: normal clock | clock off | clock frozen
print("\n=== Part G: Experiment GIFs ===")
exp_dir = gif_dir / "clock_experiment"
exp_dir.mkdir(exist_ok=True)
EXP_SECONDS = 5.0
for cmd in EXPERIMENT:
    variants = {name: load_policy(UNITREE_POLICY, joint_names, actuator_names, pose=HOME, obs_fn=obs_fn)
                for name, obs_fn in VARIANTS.items()}
    panels = []
    for name, variant in variants.items():
        # Each panel framed on its own path (same viewing angle), so a robot that wanders off
        # doesn't shrink the others
        path = trajectory(model, data, variant, np.array(cmd), EXP_SECONDS)[0]
        center = (path.min(axis=0) + path.max(axis=0)) / 2
        cam = make_camera(lookat=[center[0], center[1], 0.6], distance=3.0 + np.ptp(path, axis=0).max(),
                          azimuth=120, elevation=-15)
        mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
        mujoco.mj_forward(model, data)
        panels.append(record_frames(model, data, EXP_SECONDS, camera=cam,
                                    control=make_policy_controller(variant, np.array(cmd)),
                                    capture_fps=20, width=400, height=300, label=f"{name}  cmd {cmd}"))
    file = exp_dir / f"cmd_{cmd[0]}_{cmd[1]}_{cmd[2]}.gif"
    save_gif(side_by_side(panels), file, fps=20)
    print(f"  saved {file.name}")
