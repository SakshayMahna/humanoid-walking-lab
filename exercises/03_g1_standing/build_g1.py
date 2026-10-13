import re
import sys
from pathlib import Path

import mujoco

script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent.parent
sys.path.insert(0, str(script_dir.parent))  # make exercises/render_utils.py importable
from render_utils import make_camera, record_gif, render_image

from g1_utils import (FOOT_PATTERN, add_floor, add_init_keyframe, add_position_actuators,
                      apply_collision_settings, apply_solver_settings, hinge_joint_names)

robot_xml = repo_root / "robot" / "g1.xml"

# Load the robot as an editable spec
spec = mujoco.MjSpec.from_file(str(robot_xml))   # MjSpec: editable model; compile() turns it into an MjModel
n_before = spec.compile().ngeom                  # compiling doesn't change the spec, so we can compile twice

# Collision settings (must come before the floor, which would otherwise be disabled)
apply_collision_settings(spec)

# Floor: plane size = [half_x, half_y, grid spacing]. Default contype = conaffinity = 1, so it collides
# with the robot's collision capsules (1/1) but not its visual meshes (0/0).
add_floor(spec)

# Compile
print("\n=== Model ===")
model = spec.compile()
data = mujoco.MjData(model)
print(f"geoms: {n_before} -> {model.ngeom}")
assert model.ngeom == n_before + 1, f"expected one new geom, got {model.ngeom - n_before}"

# Collision check: count each category and spot-check one geom of each
names = [model.geom(i).name for i in range(model.ngeom)]
feet = [n for n in names if re.fullmatch(FOOT_PATTERN, n)]
other = [n for n in names if re.fullmatch(".*_collision", n) and n not in feet]
print(f"collision geoms: {len(feet)} feet (condim 3) + {len(other)} other (condim 1)")
for name in ("left_foot3_collision", "pelvis_collision", "floor"):
    g = model.geom(name)
    print(f"  {name:<22} condim={g.condim[0]} priority={g.priority[0]} friction={g.friction} "
          f"contype/conaffinity={g.contype[0]}/{g.conaffinity[0]}")
assert len(feet) == 14, f"expected 14 foot geoms, got {len(feet)}"
assert model.geom("floor").contype[0] == 1, "floor collisions disabled: add the floor after the collision settings"

# Drop test: no actuators yet, so the robot should collapse onto the floor
print("\n=== Drop test ===")
cam = make_camera(lookat=[0, 0, 0.4], distance=3.0, azimuth=135, elevation=-15)  # robot + floor in view

mujoco.mj_forward(model, data)   # forward: compute derived quantities (poses, contacts) for rendering
render_image(model, data, script_dir / "drop_before.png", camera=cam)

for _ in range(round(2.0 / model.opt.timestep)):
    mujoco.mj_step(model, data)  # step: forward + integrate, i.e. actually advance time by dt

render_image(model, data, script_dir / "drop_after.png", camera=cam)
print(f"pelvis height after 2 s = {data.qpos[2]:.3f} m")
print(f"active contacts         = {data.ncon}")

assert data.qpos[2] > 0, "pelvis below the floor: robot fell through"
assert data.ncon > 0, "no contacts: the floor is not touching the robot"

# Video of the collapse (reset and run the same 2 s again)
mujoco.mj_resetData(model, data)
n_frames = record_gif(model, data, script_dir / "drop.gif", duration=2.0, camera=cam, capture_fps=30)
print(f"saved {n_frames} frames to {script_dir / 'drop.gif'}")

# Actuators and starting pose
print("\n=== Actuators ===")
actuator_names = add_position_actuators(spec)
add_init_keyframe(spec, actuator_names)

# Solver settings from Diego's walk_fixed.ipynb: dt 0.005, iterations 10/20, ccd 500, Euler
apply_solver_settings(spec)

model = spec.compile()   # recompile: the drop-test model has no actuators or keyframe
data = mujoco.MjData(model)
assert model.nu == 29, f"expected 29 actuators, got {model.nu}"

joint_names = hinge_joint_names(spec)
print(f"{'#':>2} | {'joint order':<28} | {'actuator order':<28}")
print("-" * 64)
for i, (j, a) in enumerate(zip(joint_names, actuator_names)):
    print(f"{i:>2} | {j:<28} | {a:<28}")

# Starting pose from the keyframe
print("\n=== Starting pose ===")
mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)   # sets qpos, qvel=0 and ctrl
mujoco.mj_forward(model, data)
render_image(model, data, script_dir / "init_pose.png", camera=cam)
print(f"pelvis height {data.qpos[2]:.3f} m, ctrl[0] ({actuator_names[0]}) = {data.ctrl[0]:.3f} rad")

# Holding the pose without the policy: the robot sinks and falls. That's expected: the policy gains
# are soft, and the trained policy is what balances the robot. Standing is tested on Day 5.
n_frames = record_gif(model, data, script_dir / "hold_pose.gif", duration=3.0, camera=cam, capture_fps=20)
print(f"saved {n_frames} frames to {script_dir / 'hold_pose.gif'}")