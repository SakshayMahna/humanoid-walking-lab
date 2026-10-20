# Exercises: MuJoCo, from a pendulum to a walking G1

A learning path through MuJoCo, ending with trained walking policies running on the Unitree G1 without any
training framework. Each folder is one exercise; later ones reuse the modules in this folder.

Run any script from anywhere (paths are resolved from the script's location):

```bash
.venv/bin/python exercises/05_walking/walk.py
```

Images and GIFs are written next to the script and are not tracked (`.gitignore`); rerun to regenerate them.
Exercises 05-08 need the policies in `3p/` (see the main [README](../README.md#third-party-repositories)).

## The path

| # | Folder | Topic | What it checks / shows |
|---|---|---|---|
| 01 | [`01_inspect_model`](01_inspect_model/) | `MjModel` vs `MjData`, `nq`/`nv`, joint addresses, the free joint, quaternion order | Size asserts; free fall matches z₀ − ½gt² (and the integrator's ½·g·dt·t error) |
| 02 | [`02_pendulum_pd`](02_pendulum_pd/) | One-hinge pendulum: own PD torque vs MuJoCo's position actuator, torque limits, armature, `implicitfast` | Both controllers match; gain sweeps vs linear theory (natural frequency, damping ratio, overshoot, stability limit) |
| 03 | [`03_g1_standing`](03_g1_standing/) | Make the G1 simulatable: floor, contacts, 29 position actuators from motor models, `init_state` keyframe | Gains match the policy metadata; with these soft gains the robot cannot stand alone (the policy balances it) |
| 04 | [`04_observation`](04_observation/) | The 99 numbers Diego's policy sees: IMU velocities, projected gravity, joint state, last action, command | Frames (world vs pelvis), projected gravity for a known tilt |
| 05 | [`05_walking`](05_walking/) | ONNX policy in a 200 Hz physics / 50 Hz policy loop: metadata, action → joint targets | 0.6 m/s command walks ≈ 3 m in 6 s |
| 06 | [`06_validation`](06_validation/) | `walk.onnx` across many commands | Tracking table; finds the dead zone (no motion for ≤ 0.3 m/s, no turning in place) |
| 07 | [`07_unitree`](07_unitree/) | Unitree's policy: its model settings, 98-number observation with a gait clock | Walks without a dead zone; zeroing the gait clock reproduces Diego's dead zone |
| 08 | [`08_gazebo_reference`](08_gazebo_reference/) | MuJoCo reference runs in the same CSV format as the Gazebo logs; odometry-velocity dry run | Baseline for [`gazebo/`](../gazebo/) metrics; shows Gazebo-style velocity averaging does not change `walk.onnx` |

Write-ups: Week 1 (exercises 01-02) and Week 2 (03-07) on the [blog](https://sakshaymahna.github.io/humanoid-walking-lab/).
Exercise 08 results are in [notes/gazebo_experiments.md](../notes/gazebo_experiments.md).

## Shared modules

| Module | Provides |
|---|---|
| [`g1_utils.py`](g1_utils.py) | Motor models (reflected inertia → kp, kd), actuator groups, collision and solver settings, `build_g1_model()` |
| [`observation_utils.py`](observation_utils.py) | `read_sensor`, `projected_gravity`, `make_observation()` (Diego's 99-number layout) |
| [`policy_utils.py`](policy_utils.py) | `load_policy()` (ONNX + metadata), `action_to_ctrl`, `make_policy_controller()`, trajectory and steady-state helpers |
| [`unitree_utils.py`](unitree_utils.py) | Unitree's home pose, gait phase, 98-number observation, `build_unitree_model()`, `load_unitree_policy()` |
| [`render_utils.py`](render_utils.py) | Cameras, still images, frame recording, GIFs, side-by-side layouts |

Exercise scripts add this folder to `sys.path` so the modules import directly (`from g1_utils import ...`).

## Models and policies used

| | Diego (`walk.onnx`) | Unitree (`policy.onnx`) |
|---|---|---|
| Robot model | [`robot/g1.xml`](../robot/g1.xml) | `3p/unitree_rl_mjlab/src/assets/robots/unitree_g1/xmls/g1.xml` |
| Policy file | `3p/2025-tfg-diego-lopez/pruebas/G1/2026-01-26_10-42-02.onnx` | `3p/unitree_rl_mjlab/deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx` |
| Start pose | knees bent, pelvis at 0.76 m | home pose, pelvis at 0.8 m |
| Integrator | Euler | implicitfast |
| Observation | 99 (with base linear velocity) | 98 (no linear velocity, + gait phase) |
