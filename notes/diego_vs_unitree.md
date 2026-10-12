# Diego's policy vs Unitree's policy: differences

What to change when moving from **replicating Diego** to **building on Unitree**.
Both are G1 29-DoF velocity-command policies (forward, sideways, turning), trained with mjlab.

Sources:
- Diego: `3p/2025-tfg-diego-lopez/pruebas/G1/walk_fixed.ipynb`, `g1_constants.py`,
  ROS node `g1_locomotion/src/g1_locomotion/Inference.cpp`
- Unitree: `3p/unitree_rl_mjlab/` — `deploy/robots/g1/config/policy/velocity/v0/params/deploy.yaml`,
  `src/assets/robots/unitree_g1/g1_constants.py`, `src/tasks/velocity/velocity_env_cfg.py`,
  `src/tasks/velocity/config/g1/env_cfgs.py`, `deploy/include/isaaclab/envs/mdp/observations/observations.h`
- mjlab 1.2.0 source: `3p/mjlab_pkg/src/mjlab/` (defaults for both)

## Same in both

| | Value |
|---|---|
| Joints | 29 hinges, same names and model order |
| Motors and armature | 5020 / 7520_14 / 7520_22 / 4010, two-stage planetary reflected inertia |
| Gains | KP = armature·ωₙ², KD = 2ζ·armature·ωₙ, ωₙ = 2π·10 rad/s, ζ = 2 (e.g. hip pitch 40.2 / 2.56) |
| Effort limits | 25 / 88 / 139 / 5 N·m; waist pitch/roll and ankles 2 × 5020 (50 N·m) |
| Action | target = default_pose + scale · action, scale = 0.25 · effort / KP (0.55 / 0.35 / 0.44 / 0.07) |
| Timing | dt = 0.005 (200 Hz physics), decimation 4 (50 Hz policy) |
| Solver | iterations = 10, ls_iterations = 20 |
| Collision config | `FULL_COLLISION` (see Friction section) |
| Command | 3 numbers: lin_vel_x, lin_vel_y, ang_vel_z |

## Different

| | Diego | Unitree |
|---|---|---|
| Policy file | `pruebas/G1/2026-01-26_10-42-02.onnx` (= `g1_locomotion/policies/walk.onnx`) | `deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx` |
| Robot XML | mjlab asset zoo `g1.xml` (our `robot/g1.xml` is Diego's local copy) | Unitree's `xmls/g1.xml`: collision class `priority=1 condim=6`, pelvis at z = 0.79 (vs 0.793) |
| Default pose | **knees bent**: hip pitch −0.312, knee 0.669, ankle pitch −0.363, elbow 0.6, shoulder pitch 0.2, shoulder roll ±0.2 | **home**: hip pitch −0.1, knee 0.3, ankle pitch −0.2, elbow 0.87, shoulder pitch 0.35, shoulder roll ±0.18 |
| Base height at reset | 0.76 m (knees-bent keyframe) | 0.8 m (home keyframe, `init_state=HOME_KEYFRAME`) |
| Observation size | **99** | **98** |
| Observation order | lin_vel (3), ang_vel (3), projected_gravity (3), joint_pos_rel (29), joint_vel (29), last_action (29), command (3) | ang_vel (3), projected_gravity (3), command (3), **gait_phase (2)**, joint_pos_rel (29), joint_vel_rel (29), last_action (29) |
| Linear velocity input | yes — in Gazebo from ground-truth odometry `/g1/pelvis/odometry` (not available on a real robot) | no (only the critic sees it during training) |
| Gait phase | none | `[sin 2πφ, cos 2πφ]`, φ += 0.02 / 0.6 per policy step (0.6 s stride); `[0, 0]` when ‖command‖ < 0.1 (stand still) |
| Command range | demos use ±0.6 m/s, 0.8 rad/s | deploy: x −0.5…1.0 m/s, y ±0.5 m/s, yaw ±1.0 rad/s |
| Integrator | **Euler** in `walk_fixed.ipynb` (never set) — mismatch with training | mjlab default **implicitfast** (training and play) |
| ccd_iterations | 500 (`walk_fixed.ipynb`) | 500 on rough terrain, **50 on flat** (the deployed velocity policy is the flat task) |
| Policy rate in Gazebo | ROS wall timer 20 ms; blog says 60 Hz worked better | 50 Hz (`step_dt: 0.02`) |
| Spec format | notebook code + mjlab internals | one file: `deploy.yaml` |
| Training config available | no (private wandb run) | yes (`src/tasks/velocity/`) |

## Friction and contacts

Both apply the same `FULL_COLLISION` (mjlab `CollisionCfg`) to every geom named `.*_collision`;
non-matching geoms get contype = conaffinity = 0. Patterns are checked in order, first match wins.

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

### Friction randomization (training only)

| | Diego | Unitree |
|---|---|---|
| Foot friction | unknown (training config not available) | sampled once per env at startup from **0.3–1.6**, shared by all foot geoms |
| Other randomization | unknown | pushes every 5–6 s (±0.5 m/s x/y, ±0.4 z), joint encoder bias ±0.015 rad, torso COM offset, random reset pose/yaw |

At deploy time both use the nominal 0.6. A policy trained over 0.3–1.6 should tolerate a
different floor friction (e.g. Gazebo's) better.

## To do when switching to Unitree

1. Robot XML → Unitree's `g1.xml` (or confirm differences don't matter).
2. Default pose → home pose; base height 0.8.
3. Observation builder → 98-number layout with gait phase, no linear velocity.
4. Integrator → implicitfast; ccd_iterations → 50.
5. Commands → clip to deploy ranges; ‖command‖ < 0.1 means stand.
