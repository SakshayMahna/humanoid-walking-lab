import sys
from pathlib import Path

import mujoco
import numpy as np

script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent.parent
sys.path.insert(0, str(script_dir.parent))  # make exercises/*_utils.py importable
from g1_utils import build_g1_model
from policy_utils import DIEGO_POLICY, load_policy, make_policy_controller, trajectory
from render_utils import make_camera, record_gif

model, data, joint_names, actuator_names = build_g1_model(repo_root / "robot" / "g1.xml")
policy = load_policy(DIEGO_POLICY, joint_names, actuator_names)
# walk(model, data, policy, command, seconds) -> (displacement xyz, pelvis heights)

# Part 1: Command table
print("\n=== Part 1: Command table ===")


def run(command, seconds):
    """xy, yaw (unwrapped), pelvis height per physics step; see policy_utils.trajectory."""
    return trajectory(model, data, policy, command, seconds)


SECONDS = 5.0
# Observed: walk.onnx has a dead zone. Speeds <= 0.3 m/s (forward or sideways) make it stand still,
# and it does not turn on the spot (wz alone, up to 0.8) but turns while walking. Steady-state sweep:
# vx 0.3 -> 0.0, 0.4 -> 0.30, 0.5 -> 0.43, 0.8 -> 0.76 m/s; vy 0.5 -> 0.43; [0.5, 0, 0.5] -> wz 0.47.
# Cause not verified: mjlab 1.2.0's command code has no small-command threshold, so it's likely learned.
COMMANDS = {                                 # [vx (m/s), vy (m/s), wz (rad/s)], body frame
    "forward":         [0.6, 0.0, 0.0],
    "backward":        [-0.5, 0.0, 0.0],
    "sideways left":   [0.0, 0.5, 0.0],
    "walk and turn":   [0.5, 0.0, 0.5],     # turning works while walking
    "slow forward":    [0.3, 0.0, 0.0],     # dead zone: stands
    "slow sideways":   [0.0, 0.3, 0.0],     # dead zone: stands
    "turn in place":   [0.0, 0.0, 0.8],     # barely turns on the spot
    "stand":           [0.0, 0.0, 0.0],
}
gif_dir = script_dir / "gifs"
gif_dir.mkdir(exist_ok=True)

results = {}
print(f"{'command':<14} | {'[vx, vy, wz]':<16} | {'dx':>6} {'dy':>6} {'dyaw':>6} | "
      f"{'perfect dx dy dyaw':<20} | {'min height':>10}")
print("-" * 92)
for name, cmd in COMMANDS.items():
    xy, yaws, heights = run(np.array(cmd), SECONDS)
    dx, dy = xy[-1] - xy[0]
    dyaw = yaws[-1] - yaws[0]
    results[name] = dict(dx=dx, dy=dy, dyaw=dyaw, min_height=heights.min())
    # x/y are world frame; the robot starts facing +x, so for non-turning commands world ~ body
    perfect = f"{cmd[0] * SECONDS:+.1f} {cmd[1] * SECONDS:+.1f} {cmd[2] * SECONDS:+.1f}"
    print(f"{name:<14} | {str(cmd):<16} | {dx:+6.2f} {dy:+6.2f} {dyaw:+6.2f} | {perfect:<20} | "
          f"{heights.min():>10.3f}")

    # GIF of the same run, camera centred on the path it actually took
    mid = (xy[0] + xy[-1]) / 2
    cam = make_camera(lookat=[mid[0], mid[1], 0.6], distance=3.5 + np.linalg.norm(xy[-1] - xy[0]),
                      azimuth=120, elevation=-15)
    mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
    mujoco.mj_forward(model, data)
    record_gif(model, data, gif_dir / f"{name.replace(' ', '_')}.gif", duration=SECONDS, camera=cam,
               control=make_policy_controller(policy, np.array(cmd)), capture_fps=20, width=480, height=360,
               label=f"{name}: vx={cmd[0]} vy={cmd[1]} wz={cmd[2]}")


# Part 2: Command schedule (Diego's control.ipynb pattern)
print("\n=== Part 2: Command schedule ===")
# control.ipynb switches the twist every 2 s (20 ticks of its 10 Hz control thread)
SCHEDULE = [                       # (command [vx, vy, wz], duration in s)
    ([0.5, 0.0, 0.8], 2.0),        # walk forward while turning left
    ([0.0, 0.0, 0.0], 2.0),        # stop
    ([-0.5, 0.0, 0.0], 2.0),       # walk backward
    ([0.0, 0.0, 0.8], 2.0),        # turn in place (inside the dead zone: expect little turning)
]
SCHEDULE_SECONDS = sum(duration for _, duration in SCHEDULE)


def command_at(t):
    """Scheduled command at simulation time t (holds the last one after the schedule ends)."""
    end = 0.0
    for cmd, duration in SCHEDULE:
        end += duration
        if t < end:
            return cmd
    return SCHEDULE[-1][0]


xy, yaws, heights = run(command_at, SCHEDULE_SECONDS)
steps_per_s = round(1 / model.opt.timestep)
start = 0
print(f"{'segment':<16} | {'dx':>6} {'dy':>6} {'dyaw':>6} | min height")
for cmd, duration in SCHEDULE:
    end = start + round(duration * steps_per_s) - 1
    dx, dy = xy[end] - xy[start]
    print(f"{str(cmd):<16} | {dx:+6.2f} {dy:+6.2f} {yaws[end] - yaws[start]:+6.2f} | {heights[start:end + 1].min():.3f}")
    start = end + 1
schedule_min_height = heights.min()

# GIF with the active command in the label; camera centred on the area the robot covered
center = (xy.min(axis=0) + xy.max(axis=0)) / 2
cam = make_camera(lookat=[center[0], center[1], 0.6], distance=4.5 + np.ptp(xy, axis=0).max(),
                  azimuth=120, elevation=-20)
mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
mujoco.mj_forward(model, data)
record_gif(model, data, gif_dir / "schedule.gif", duration=SCHEDULE_SECONDS, camera=cam,
           control=make_policy_controller(policy, command_at), capture_fps=20, width=480, height=360,
           label=lambda d: f"t={d.time:4.1f} s  command {command_at(d.time)}")
print("saved schedule.gif")

# Part 3: Checks
# These pin down the measured behavior: a change that breaks walking or shifts the dead zone fails here.
print("\n=== Part 3: Checks ===")
r = results
checks = [
    ("no falls (pelvis > 0.6 m in every run)", all(v["min_height"] > 0.6 for v in r.values())
                                               and schedule_min_height > 0.6),
    ("forward moves +x",                       r["forward"]["dx"] > 1.0),
    ("backward moves -x",                      r["backward"]["dx"] < -1.0),
    ("sideways moves +y",                      r["sideways left"]["dy"] > 1.0),
    ("walk and turn turns left",               r["walk and turn"]["dyaw"] > 1.0),
    ("stand stays put",                        abs(r["stand"]["dx"]) < 0.1 and abs(r["stand"]["dy"]) < 0.1
                                               and abs(r["stand"]["dyaw"]) < 0.1),
    # Known limits of walk.onnx (dead zone from standstill)
    ("slow forward stays in dead zone",        abs(r["slow forward"]["dx"]) < 0.3),
    ("slow sideways stays in dead zone",       abs(r["slow sideways"]["dy"]) < 0.3),
    ("turn in place barely turns",             r["turn in place"]["dyaw"] < 1.0),
]
for name, ok in checks:
    print(f"  {'OK  ' if ok else 'FAIL'} {name}")
assert all(ok for _, ok in checks), "validation failed"
print(f"\nGIFs saved to {gif_dir}")
