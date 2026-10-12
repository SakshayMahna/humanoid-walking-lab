# Diego's policy vs Unitree's policy: differences

What to change when moving from **replicating Diego** to **building on Unitree**.
Both are G1 29-DoF velocity-command policies (forward, sideways, turning), trained with mjlab
on the **flat** task (neither actor takes `height_scan`, which only the rough task's actor has).

Every row below was checked against a file (verified 2026-10-12). Open questions are listed at the end.

Sources:
- Diego: `3p/2025-tfg-diego-lopez/` — `pruebas/G1/walk_fixed.ipynb`, `pruebas/G1/control.ipynb`,
  `pruebas/G1/g1_constants.py`, `g1_locomotion/{include,src}/g1_locomotion/Inference.{hpp,cpp}`,
  `g1_sim/models/g1_29dof/model.sdf`, `g1_sim/worlds/empty_world.sdf`, blog `docs/_posts/`
- Unitree: `3p/unitree_rl_mjlab/` — `deploy/robots/g1/config/policy/velocity/v0/params/deploy.yaml`,
  `src/assets/robots/unitree_g1/`, `src/tasks/velocity/`, `deploy/include/isaaclab/envs/mdp/observations/observations.h`,
  `simulate/`; pins `mjlab==1.2.0` (`setup.py`)
- mjlab 1.2.0 source: `3p/mjlab_pkg/src/mjlab/`; mjlab git history (github.com/mujocolab/mjlab)
- ONNX metadata and input/output shapes read directly from each `.onnx` file

## Policy files

| File | Input → output | `run_path` | Used by |
|---|---|---|---|
| Diego `pruebas/G1/2026-01-26_10-42-02.onnx` = `g1_locomotion/policies/walk.onnx` (same MD5) | 99 → 29 | `2026-01-26_10-42-02` | `walk_fixed.ipynb` |
| Diego `g1_locomotion/policies/walk_free_elbows.onnx` | 97 → 27 (elbows not actuated) | `local` | **the ROS node** (`Inference.hpp:27`) |
| Unitree `deploy/.../velocity/v0/exported/policy.onnx` | 98 → 29 | `2026-03-18_18-40-20` | Unitree deploy |

The table below compares `walk.onnx` with Unitree's policy. Diego's Gazebo deployment currently runs the free-elbows variant.

## Diego's mjlab version

- His `g1_constants.py` is byte-identical to mjlab main from `0a0864d4` (2025-11-12) up to `7700a1ff` (2025-12-27, exclusive).
  His `g1.xml` blob was current on main from 2025-10-30 to 2026-01-21. `walk_fixed.ipynb` uses `joint_names_expr`,
  which was renamed in `7700a1ff`. **So the notebook ran on mjlab main from between 2025-11-12 and 2025-12-27** (older than 1.2.0).
- The **training** version (run 2026-01-26) is not recorded anywhere. Two candidates:
  - **A**: the same checkout (2025-11-12 … 2025-12-27)
  - **B**: mjlab main as of the run date (`a3f28e50`, 2026-01-25)
- In both A and B the actor observations, observation noise, rewards, command ranges and integrator are identical. They differ only in randomization (see below).

## Same in both

| | Value | Checked in |
|---|---|---|
| Joints | 29 hinges, same names, same order | ONNX `joint_names` (identical) |
| Motors and armature | 5020 / 7520_14 / 7520_22 / 4010, two-stage planetary reflected inertia | actuator blocks identical in all three `g1_constants.py` |
| Gains | KP = armature·ωₙ², KD = 2ζ·armature·ωₙ, ωₙ = 2π·10 rad/s, ζ = 2 (hip pitch 40.179 / 2.558) | ONNX `joint_stiffness` / `joint_damping` (identical) |
| Effort limits | 25 / 88 / 139 / 5 N·m; waist pitch/roll and ankles 2 × 5020 = 50 N·m | `g1_constants.py` |
| Action | target = default_pose + scale · action, scale = 0.25 · effort / KP (0.548 / 0.351 / 0.439 / 0.075) | ONNX `action_scale` (identical) |
| Timing | dt = 0.005 (200 Hz physics), decimation 4 (50 Hz policy) | both `velocity_env_cfg.py` |
| Solver | iterations = 10, ls_iterations = 20 | both `velocity_env_cfg.py` |
| Integrator (training) | **implicitfast** (mjlab `SimulationCfg` default, not overridden) | mjlab `sim/sim.py` in A, B and 1.2.0 |
| Collision config | `FULL_COLLISION` (see Friction section) | `g1_constants.py`; `CollisionCfg` code unchanged between A and 1.2.0 |
| Actor observation noise (shared terms) | ang_vel ±0.2, projected_gravity ±0.05, joint_pos ±0.01, joint_vel ±1.5 | both `velocity_env_cfg.py` |
| Command | 3 numbers: lin_vel_x, lin_vel_y, ang_vel_z (`twist`) | ONNX `command_names` |

## Different

| | Diego | Unitree |
|---|---|---|
| Observation size | **99** | **98** |
| Observation order (ONNX `observation_names`) | base_lin_vel (3), base_ang_vel (3), projected_gravity (3), joint_pos (29), joint_vel (29), actions (29), command (3) | base_ang_vel (3), projected_gravity (3), command (3), **phase (2)**, joint_pos (29), joint_vel (29), actions (29) |
| Linear velocity | actor input (trained with ±0.5 noise); in Gazebo from the `OdometryPublisher` plugin (ground truth) on `/g1/pelvis/odometry` | **critic only** |
| Gait phase | none | `[sin 2πφ, cos 2πφ]`, period 0.6 s; `[0, 0]` when ‖command‖ < 0.1. Training: φ from episode time (restarts each episode). Deploy: φ += 0.02 / 0.6 per step from node start |
| Gait reward | none (mjlab `air_time` weight 0) | `foot_gait` +0.5: left offset 0, right 0.5; a leg is in stance when its phase < **0.56** |
| Default pose (ONNX) | **knees bent**: hip pitch −0.312, knee 0.669, ankle pitch −0.363, elbow 0.6, shoulder pitch 0.2, shoulder roll ±0.2 | **home**: hip pitch −0.1, knee 0.3, ankle pitch −0.2, elbow 0.87, shoulder pitch 0.35, shoulder roll ±0.18 |
| Base height at reset | 0.76 m (`KNEES_BENT_KEYFRAME`) | 0.8 m (`HOME_KEYFRAME`, `init_state=HOME_KEYFRAME`) |
| Robot XML | old mjlab `g1.xml` (`robot/g1.xml` is the same file). Differs from mjlab 1.2.0 only cosmetically (shadow flag, hand rgba) | Unitree `xmls/g1.xml`: collision class `priority=1 condim=6`, pelvis z 0.79 (vs 0.793), foot sites z −0.035 (vs −0.037) |
| Rewards | track lin/ang vel 2.0 each, upright +1, pose +1, body_ang_vel −0.05, angular_momentum −0.02, dof_pos_limits −1, action_rate −0.1, foot_clearance −2, foot_swing_height −0.25, foot_slip −0.1, soft_landing −1e-5, self_collisions −1 | track lin/ang vel 1.0 each, body_orientation −1, pose +1, body_ang_vel −0.05, angular_momentum −0.025, is_terminated −200, joint_acc −2.5e-7, joint_pos_limits −10, action_rate −0.05, foot_gait +0.5, foot_clearance −1, foot_slip −0.25, soft_landing −1e-3, stand_still −1, self_collisions −1 |
| Training commands | x ±1, y ±1, yaw ±0.5; 10 % standing, 30 % heading envs; curriculum → x −1.5…2, yaw ±0.7 (120k steps) → x −2…3 (240k) | x −0.5…1, y ±0.5, yaw ±1; 5 % standing; curriculum → x −1…2, y ±1 (120k steps) |
| Deploy commands | notebooks use up to 0.6 m/s forward, 0.8 rad/s yaw (`walk_fixed`, `control.ipynb`) | `deploy.yaml`: x −0.5…1.0, y ±0.5, yaw ±1.0 |
| MuJoCo replay | `walk_fixed.ipynb`: integrator never set → **Euler**; `ccd_iterations = 500` (training flat task: 50) | `simulate/` uses a different model `scene_g1.xml`: no `<option>` → **Euler**, dt 0.002, all joints armature 0.01 / damping 0.05 / frictionloss 0.2, torque motors with PD from deploy gains |
| Gazebo PD gains (Diego `model.sdf`) | **do not match training**: hip pitch/yaw, waist yaw 44.11/2.81 (train 40.18/2.56); hip roll, knee 148.56/9.46 (99.10/6.31); ankles, waist roll/pitch 34.44/2.19 (28.50/1.81); shoulders, elbows, wrist roll 17.22/1.10 (14.25/0.91); wrist pitch/yaw 5.00/0.50 (16.78/1.07). Ankle roll clamped to **±10 N·m** (train ±50) | — (not deployed to Gazebo) |
| Policy rate in Gazebo | ROS wall timer 20 ms. Blog (Summer post): tested 10–70 Hz, ~60 Hz best in practice, stable at 20 Hz | 50 Hz (`step_dt: 0.02`) |
| Spec format | notebook code + mjlab internals | one file: `deploy.yaml` |
| Training config available | no (run not recorded; candidates A/B above) | yes (`src/tasks/velocity/`, mjlab 1.2.0) |

## Friction and contacts

Both apply the same `FULL_COLLISION` (mjlab `CollisionCfg`) to every geom named `.*_collision`.
Non-matching geoms get contype = conaffinity = 0. Patterns are checked in order, first match wins.
Unmatched fields fall back to the defaults condim 3, contype/conaffinity 1, priority 0, friction unchanged.

| Geoms | condim | friction (sliding) | priority | contype / conaffinity |
|---|---|---|---|---|
| Feet `^(left\|right)_foot[1-7]_collision$` (14) | 3 | **0.6** | 1 | 1 / 1 |
| Other `.*_collision` | 1 (frictionless) | unchanged | 0 | 1 / 1 |
| Everything else (visual meshes, unnamed) | — | — | — | 0 / 0 |
| Floor | MuJoCo default 3 | default 1.0 | 0 | 1 / 1 |

- Foot priority 1 > floor priority 0, so foot–floor contacts use the **foot's** condim 3 and friction 0.6
  (without priority MuJoCo would take the max friction, 1.0).
- XML defaults differ (Unitree's collision class sets `priority=1 condim=6`; Diego's leaves MuJoCo defaults),
  but `FULL_COLLISION` overwrites condim, priority and contype/conaffinity on every collision geom,
  so the **final contact settings are identical**.
- The floor must be added **after** applying `FULL_COLLISION`, otherwise it gets disabled.

### Domain randomization (training only)

| | Diego A (notebook's mjlab) | Diego B (mjlab main 2026-01-25) | Unitree (mjlab 1.2.0 task) |
|---|---|---|---|
| Foot friction | 0.3–1.2, sampled **independently per foot geom** | 0.3–1.2, shared by all foot geoms | 0.3–1.6, shared |
| Pushes | every 1–3 s, **x/y ±0.5 m/s only** | every 1–3 s: x/y ±0.5, z ±0.4 m/s, roll/pitch ±0.52, yaw ±0.78 rad/s | every **5–6 s**, same velocity ranges as B |
| Encoder bias | **none** | ±0.015 rad | ±0.015 rad |
| Torso COM offset | **none** | ±2.5 cm x/y, ±3 cm z | **±5 cm** x/y/z |
| Reset | x/y ±0.5 m, yaw ±π | x/y ±0.5 m, z +0.01…0.05 m, yaw ±π | x/y ±0.5 m, yaw ±π |

Sources: `velocity_env_cfg.py` at mjlab commits `7700a1ff^` (A; unchanged since 2025-11-15) and `a3f28e50` (B);
shared-friction fix `685c9204` (2026-01-13); encoder bias and COM added in `0db87649` (2025-12-31).

Either way, Unitree trains with a wider friction range and less frequent pushes.
Whether Diego's policy saw encoder bias or COM offsets depends on A vs B.

At deploy time all three use the nominal foot friction 0.6.

## Open questions (not resolvable from the files)

1. Which mjlab version trained `walk.onnx` (A or B). Only Diego can answer this.
2. Which mjlab version trained `walk_free_elbows.onnx`, and whether its elbow handling matches the ROS node:
   its ONNX `default_joint_pos` has elbows = 0.0, but `Inference.hpp` subtracts 0.6 for the elbows.
3. Gazebo physics and floor friction: `empty_world.sdf` sets neither, so Gazebo's defaults apply (not in the repo).
4. The frame of the Gazebo `OdometryPublisher` twist (body vs world) is set by gz-sim source, which isn't in the repo.
   The policy expects body-frame velocity.

## To do when switching to Unitree

1. Robot XML → Unitree's `g1.xml` (or confirm differences don't matter).
2. Default pose → home pose; base height 0.8.
3. Observation builder → 98-number layout with gait phase, no linear velocity.
4. MuJoCo replay → implicitfast, ccd_iterations 50 (match training, not Unitree's `simulate/`).
5. Commands → clip to deploy ranges; ‖command‖ < 0.1 means stand.
6. Gazebo → set PD gains and effort limits to the training values (Diego's `model.sdf` does not).
