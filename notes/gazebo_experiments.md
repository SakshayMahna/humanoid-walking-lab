# Gazebo experiments: running the MuJoCo-trained policies in Gazebo

Goal: run Diego's `walk.onnx` and Unitree's `policy.onnx` in Diego's Gazebo setup (`g1_sim`), compare them with
MuJoCo, and find what has to match for them to behave the same. The underlying research question (why the
simulators differ) lives in [sim2sim_mujoco_vs_gazebo.md](sim2sim_mujoco_vs_gazebo.md).

All numbers below come from runs on 2026-10-16 (Docker on an Apple Silicon Mac, Gazebo RTF ≈ 0.9).

---

## 1. Setup

| Piece | What it is |
|---|---|
| `docker/Dockerfile`, `docker/compose.yaml` | `tiryoh/ros2-desktop-vnc:jazzy` (Ubuntu 24.04, ROS 2 Jazzy, ros_gz 1.0.24 = Gazebo Harmonic / gz-sim 8) + ONNX Runtime 1.17.1 aarch64. Mounts Diego's repo and `ros2/` into `~/ros2_ws/src`, and the whole lab repo at `~/humanoid-walking-lab` |
| `ros2/g1_policy_runner` | Our node: runs any mjlab-exported velocity policy; reads joint order, default pose, action scale and observation layout from the ONNX metadata |
| `gazebo/run_scenario.sh` | Headless run (no GUI) of one scenario → `recordings/NAME.csv` |
| `gazebo/scenario_logger.py` | Publishes the command **on simulation time** and logs pelvis odometry (+ the command seen on `/cmd_vel`) at 20 Hz |
| `gazebo/record.sh` (+ `record_inside.sh`, `gui_record.config`) | Same scenario with the Gazebo GUI, screen-recorded → `recordings/NAME.mp4/.gif` |
| `gazebo/make_gain_variants.py` | Writes modified copies of Diego's robot SDF with different joint controllers |
| `gazebo/gain_experiment.sh` | Runs all controller variants × both policies × repeats |
| `gazebo/summarize_runs.py` | Metrics table from the CSVs |
| `exercises/08_gazebo_reference/mujoco_runs.py`, `exercises/08_gazebo_reference/odometry_dry_run.py` | MuJoCo reference runs in the same CSV format |

```bash
docker compose -f docker/compose.yaml up -d --build
# headless, logged:   VX VY WZ SECONDS STAND NAME "launch args"
docker compose -f docker/compose.yaml exec -u ubuntu ros bash ~/humanoid-walking-lab/gazebo/run_scenario.sh \
    0.5 0 0 24 4 test "policy:=unitree clock:=wall"
# video:
gazebo/record.sh 0.5 0 0 24 4 test "policy:=walk clock:=sim"
# controller variants (generate the SDFs on the Mac first: the container has no MuJoCo/numpy)
.venv/bin/python gazebo/make_gain_variants.py
docker compose -f docker/compose.yaml exec -u ubuntu ros bash ~/humanoid-walking-lab/gazebo/gain_experiment.sh 2 wall
.venv/bin/python gazebo/summarize_runs.py recordings/*.csv
```

Launch arguments of `ros2 launch g1_policy_runner policy.launch.py`: `policy:=walk|unitree`, `clock:=wall|sim`,
`policy_period:=0.02`, `gui:=true|false`, `robot_model:=g1_free_roam|<path to model.sdf>`, `spawn_z:=0.80`.

**Scenario used everywhere below:** zero command for 4 s (standing), then `vx = 0.5 m/s` for 20 s.

**Metrics** (`summarize_runs.py`): `stand_rot` / `stand_move` = rotation (deg) / displacement (m) during the
zero-command phase; `vx`, `vy` = mean pelvis-frame velocity from 4 s after the command starts; `turn` = slope of
the heading (deg/s); `wander` = std of the heading around that slope (deg).

---

## 2. What has to match MuJoCo, and how the code does it

### 2a. In the policy node (done, `ros2/g1_policy_runner/src/policy_runner.cpp`)

| Training (mjlab / MuJoCo) | Our node | Code |
|---|---|---|
| Joint order = metadata `joint_names` (≠ Gazebo's `/joint_states` order) | Reorder by name, `find()` not `operator[]` | `joint_state_callback` (index `k` = Gazebo, `i` = policy) |
| Joint positions relative to `default_joint_pos` | `position[k] - default_pos_[i]` | `joint_state_callback` |
| Gravity in the pelvis frame: Rᵀ[0,0,−1] | tf2 quaternion (x,y,z,w order!) → `R.transpose() * (0,0,-1)` | `imu_callback` |
| Angular velocity, pelvis frame | IMU `angular_velocity` | `imu_callback` |
| Linear velocity of the `imu_in_pelvis` point, pelvis frame (walk.onnx only) | Odometry twist (see §4) | `odometry_callback` |
| Observation order = metadata `observation_names` | Loop over the names, throw on unknown | `build_observation` |
| Unitree gait phase [sin, cos], off when ‖cmd‖ < 0.1 | t = `steps_ * policy_period_` | `gait_phase` |
| Target = default + scale · action, **no clipping** | Published per joint on `/g1/cmd_pos/<joint>` | `policy_step` |
| Previous **raw** action fed back | `last_action_ = action` | `policy_step` |
| One action per joint | Startup check (rejects `walk_free_elbows.onnx`: 27 actions) | `load_metadata` |
| Policy every 20 ms of **simulated** time | `clock:=sim` → timer on `/clock`; `clock:=wall` = Diego's real-time timer | constructor (timer) |

The timer line:

```cpp
timer_ = create_wall_timer(period, std::bind(&PolicyRunner::policy_step, this));   // clock:=wall
timer_ = create_timer(period, std::bind(&PolicyRunner::policy_step, this));        // clock:=sim
```

- `period` is `policy_period_` (0.02 s) converted to `std::chrono::nanoseconds`.
- `policy_step` is a member function, so it needs an object to run on. `std::bind(&PolicyRunner::policy_step, this)`
  makes a callable with no arguments that calls `this->policy_step()`. (Same as the lambda
  `[this]() { policy_step(); }`.) The timer calls it every period.
- `create_wall_timer` counts real time. `create_timer` counts the node's clock; the launch file sets
  `use_sim_time:=true` for `clock:=sim`, so that clock is Gazebo's `/clock`.

### 2b. In the Gazebo robot (not done in Diego's setup; tested with model variants, §3)

| MuJoCo (training) | Diego's Gazebo model (Luis's) | Variant tested |
|---|---|---|
| kp, kd from the motor model (= ONNX `joint_stiffness/damping`) | Different values (knee ×1.5, see sim2sim notes) | `mj`: metadata values |
| Torque limits: ankle roll 50 N·m | ankle roll 10 N·m | `mj`: 50 N·m |
| Damping on joint velocity: τ = kp·(target − q) − kd·q̇ | gz PID: kd · Δerror/dt → kick every time the target changes (every 20 ms) | `mj_vel_d`: `d_gain` 0 + joint `<damping>` kd |
| Armature (reflected rotor inertia) | none (no SDF parameter); ankle roll has Luis's joint damping 1.5 + friction 0.1 | approximated in link inertia: `original_armature` ([Q1b](../open_questions/q1b_controller/)) |
| Foot friction 0.6, MuJoCo soft contacts | ground mu 1.0, DART + ODE collision | not tested yet |
| Physics step 5 ms, policy synchronous | 1 ms, policy asynchronous (ROS topics) | 0.2 ms step tested with the original controller ([Q1b](../open_questions/q1b_controller/)); synchronous execution not tested |

**How "D on velocity" is implemented** (`gazebo/make_gain_variants.py`):

1. In each `JointPositionController` plugin: `p_gain` = metadata kp, **`d_gain` = 0**, `cmd_min/max` = motor limit.
   The controller is now a pure P controller: τ = kp · (target − q). No derivative of the error → no kick.
2. In each joint's `<axis><dynamics>`: add **`<damping>kd</damping>`**. That is SDF's viscous joint damping,
   τ = −kd · q̇, applied by the physics engine on the joint's own velocity.
3. Sum: τ = kp·(target − q) − kd·q̇, the same law as MuJoCo's position actuator (`gainprm = kp`,
   `biasprm = [0, −kp, −kd]`).

Remaining difference: in MuJoCo the whole actuator force (P and D) is clipped to the torque limit; here only the
P part is clipped (`cmd_max`), the joint damping is not.

---

## 3. Results: controller variants (wall clock, 2 runs each)

`gazebo/gain_experiment.sh 2 wall`, command timed on simulation time (§5).

| Policy | Setup | stand_rot (°) | stand_move (m) | vx (cmd 0.5) | vy | turn (°/s) | wander (°) |
|---|---|---|---|---|---|---|---|
| Unitree | **MuJoCo** | 0.7 | 0.01 | 0.50 | −0.01 | +0.72 | 0.39 |
| Unitree | Luis's gains | 10.8 ± 1.1 | 0.40 ± 0.16 | 0.57 | 0.03 | +2.6 ± 2.6 | 6.7 ± 0.8 |
| Unitree | `mj` gains | 8.5 ± 3.2 | 0.73 ± 0.33 | 0.53 | 0.01 | +0.9 ± 1.2 | 8.1 ± 1.0 |
| Unitree | `mj_vel_d` | 16.8 ± 0.2 | 1.02 ± 0.00 | **0.50** | −0.02 | **+1.01 ± 0.04** | **0.49** |
| walk | **MuJoCo** | 0.5 | 0.00 | 0.43 | −0.01 | +0.14 | 0.18 |
| walk | Luis's gains | 11.9 ± 4.3 | 0.22 ± 0.10 | 0.60 | −0.06 | −2.5 ± 2.1 | 3.5 ± 1.6 |
| walk | `mj` gains | 5.5 ± 0.7 | 0.24 ± 0.07 | 0.57 | −0.03 | +1.4 ± 0.1 | 1.4 ± 0.1 |
| walk | `mj_vel_d` | 4.9 ± 0.8 | 1.21 ± 0.04 | 0.73 | −0.24 | −2.2 ± 0.0 | 1.6 ± 0.1 |

Findings:

- **Unitree walking matches MuJoCo only with the D term on velocity.** Gains alone don't fix the heading wander
  (6.7° → 8.1°); moving D to the velocity does (0.49° vs MuJoCo 0.39°), and speed and turn rate also match.
  The run-to-run spread also collapses (± 0.04°/s).
- **Standing (zero command):** the standing numbers above came from a long-lived container; later tests
  (Q1) showed Unitree stands still with `mj_vel_d` (0.00-0.04 m in 30 s, with or without GUI) and drifts metres
  only with the original controller. The ≈ 0.6 m/s walk-offs seen here were not reproduced. See
  [open_questions/q1_standing](../open_questions/q1_standing/) and [q1b_controller](../open_questions/q1b_controller/).
- **walk.onnx gets worse with `mj_vel_d`**: it crouches (pelvis z 0.57-0.61 m vs ≈ 0.68), walks too fast (0.73)
  and slides sideways (−0.24). **Open (Q2):** why only walk.onnx. It is the policy that uses
  base_lin_vel, but §4 shows the odometry signal itself is not the problem.

### Earlier single runs (before the controller variants; Luis's gains)

| Run | Final xy (m) | Final yaw | Turn, last 15 s |
|---|---|---|---|
| MuJoCo walk.onnx | (8.41, 0.30) | +4.0° | +0.14°/s |
| MuJoCo Unitree | (9.73, 1.44) | +15.7° | +0.72°/s |
| Gazebo walk, wall clock | (9.59, −7.10) | −50.5° | −0.33°/s |
| Gazebo walk, sim clock | (6.48, −9.72) | −70.5° | −2.73°/s |
| Gazebo Unitree, wall clock | (10.29, −4.83) | −22.2° | −0.46°/s |
| Gazebo Unitree, sim clock | (9.20, −5.34) | −48.3° | −2.50°/s |

Single runs with large wander: wall vs sim clock is **not** conclusive. Needs repeats (to do).

### Diego's node as-is (`walk_free_elbows.onnx`)

His launch file runs `walk_free_elbows.onnx` (27 actions, elbows not controlled). Nothing commands the elbows,
so Gazebo holds them at 0 rad (`JointPositionController` default `jointPosCmd{0.0}`). It turns while commanded
straight in **both** simulators: MuJoCo +0.77°/s (elbows 0 rad), −1.46°/s (elbows 0.6 rad); Gazebo ≈ +1.5°/s.
`walk.onnx` in MuJoCo: +0.09°/s. So that drift is mostly the policy + elbow pose.

---

## 4. The odometry velocity input (walk.onnx's `base_lin_vel`)

**What it is.** walk.onnx's first 3 observations are the linear velocity of the pelvis IMU point, in the pelvis
frame. In MuJoCo training this is a `velocimeter` sensor at `imu_in_pelvis`: the exact velocity at this instant.
Gazebo has no velocimeter in Diego's model, so his node (and ours) reads `/g1/pelvis/odometry`, published by
Gazebo's `OdometryPublisher` plugin.

**How Gazebo computes it** (gz-sim8 `OdometryPublisher.cc`): every 1 ms physics step, the finite difference of
the pose at the `xyz_offset` point (= `imu_in_pelvis`, checked), rotated into the current pelvis frame, pushed
into a rolling mean of 10 samples. So the published value is ≈ (p(t) − p(t − 10 ms)) / 10 ms: a 10 ms average,
about 5 ms late, published at 400 Hz. Unitree's policy has no linear-velocity input, so this only affects walk.onnx.

**The suspected problem:** a smoothed, delayed velocity could make the policy chase its own estimate.

**Dry run** (`exercises/08_gazebo_reference/odometry_dry_run.py`): MuJoCo only, walk.onnx, everything identical except the velocity
input:

| base_lin_vel = | vx (cmd 0.5) | vy | turn (°/s) | wander (°) |
|---|---|---|---|---|
| velocimeter (training) | 0.43 | −0.01 | +0.14 | 0.18 |
| Gazebo-style 10 ms average | 0.43 | −0.01 | +0.16 | 0.18 |
| 40 ms average (4× stronger) | 0.43 | −0.01 | +0.05 | 0.15 |
| zero | **1.77** | −0.17 | −3.83 | 3.37 |

**Result: the averaging is not the problem.** The policy depends heavily on this input (zero → it runs at
1.77 m/s), but a 10 ms or even 40 ms average changes nothing. Not yet tested: noise, and the asynchronous read
(the policy takes whatever message arrived last, up to 2.5 ms old, plus ROS latency).

---

## 5. Measurement pitfall: the "standing" timing error (fixed)

The first version of `run_scenario.sh` started the logger, slept 4 s of **wall** time, then started
`ros2 topic pub /cmd_vel`. But the logger's t = 0 is its first odometry message, which arrives only after Python
and rclpy have started (≈ 1-2 s). So the command landed at t ≈ 2-4 s in the log, varying per run, and the
"standing" window (first 4 s) included walking. All standing numbers from that version were discarded.

Fix: `scenario_logger.py` publishes the command itself, switching at t = STAND on the odometry timestamps
(simulation time), and records both the command it sent and the last command seen on `/cmd_vel`.

---

## 6. Videos

In `recordings/` (gitignored; regenerate with `gazebo/record.sh`). Gazebo's GUI draws only ≈ 9-10
distinct frames/s in Docker on the Mac (CPU rendering), whatever the capture rate; files are 30 fps capture,
25 fps GIF.

| File | What |
|---|---|
| `video_{luis,mj,mj_vel_d}_{walk,unitree}` | The three controller setups × both policies (wall clock) |
| `walk_wall_forward`, `walk_sim_forward`, `unitree_wall_forward`, `unitree_sim_forward` | First policy-runner runs, Luis's gains |
| `diego_free_elbows_forward` | Diego's node as-is |

---

## 7. Open questions

Worked through one at a time, in this order: Q1 and Q2 decide whether the Gazebo setup is fair enough to
compare the two policies; Q3-Q4 are policy-execution experiments; Q5-Q7 are sim2sim research.

| # | Question | What we know | Status |
|---|---|---|---|
| Q1 | Why is standing (zero command) unstable in Gazebo? | Unitree: caused by the original Gazebo joint controller (sustained drift, 2.3-4.0 m and ≈ −90° to −130° in 30 s). With the corrected controller it stands still with and without the GUI (RTF 0.90 / 0.99). Earlier headless walk-offs not reproduced. [open_questions/q1_standing](../open_questions/q1_standing/) | **done** (Unitree); walk.onnx → Q2 |
| Q1b | Why does the corrected controller work and the original not? | No armature + damping applied explicitly (in the PID plugin) → the ankle-roll damper is numerically unstable (condition 9-11 vs < 4); the kick is a smaller extra; gains/limits barely matter. Confirmed in Gazebo three ways with Unitree: joint damping (implicit), armature added to the links, or a 0.2 ms step each remove the drift. [open_questions/q1b_controller](../open_questions/q1b_controller/) | **done** |
| Q2 | Why does walk.onnx get worse with `mj_vel_d` while Unitree gets better? | Crouches (z 0.57-0.61), 0.73 m/s for 0.5, slides sideways. Odometry averaging ruled out (§4) | open |
| Q3 | Wall vs sim clock | Single runs only, not conclusive | open |
| Q4 | Policy rate (Diego: best ≈ 60 Hz of 10-70 Hz) | Not repeated; his sweep code is not in his repo | open |
| Q5 | Where Luis's PD gains come from | Same ωₙ, ζ as MuJoCo, inertias ×1.10 / ×1.50 / ×1.21; no derivation found ([sim2sim notes](sim2sim_mujoco_vs_gazebo.md)). Q1b: his 10 N·m ankle-roll limit is what keeps the original setup from falling, so the tuning may compensate for the missing armature (our interpretation) | open (ask Luis?) |
| Q6 | Why Gazebo roughly doubles Unitree's left drift even with `mj_vel_d` (+1.0 to +1.8°/s vs +0.7°/s) | Remaining differences: armature, friction, contacts, 1 ms vs 5 ms step | open |
| Q7 | Parked sim2sim experiments (D-term graphs, contact, armature, solver, in-loop policy) | [sim2sim notes](sim2sim_mujoco_vs_gazebo.md) | parked |

After Q1-Q2: comparison metrics for Diego vs Unitree across more scenarios (slow, sideways, turning), with a
heading controller option so drift doesn't dominate.

---

## 8. Upstream (Gazebo) and tooling

### Existing Gazebo issues (searched 2026-10-21, all gazebosim repositories)

- **Joint armature / rotor inertia:** no issue, PR or code in gazebosim before ours (sdformat#1734) (the only "armature" hits are skeletal
  animation). SDFormat 14, gz-physics 7's DART plugin, gz-sim and DART's joint headers have no such parameter
  (also checked in the container).
- [sdformat#127](https://github.com/gazebosim/sdformat/issues/127) (open since 2016): "too many simulation problems
  that are caused by explicit damping"; "the failure mode of explicit damping is numerical instability". Same
  mechanism as Q1b, for Gazebo Classic's joint damping.
- [sdformat#473](https://github.com/gazebosim/sdformat/issues/473) (closed): Drake uses its own
  `drake:rotor_inertia` SDFormat extension for what the native schema can't express.
- `JointPositionController` D term: no issue about the derivative kick / explicit damping. gz-sim `main` still calls
  `posPid.Update(error, dt)` (D on the error); gz-math's `PID` already has `Update(error, errorRate, dt)`, so a
  "derivative on measurement" option would be a small change (raised: gz-sim#4092).

### Our issues (posted 2026-10-21)

| Issue | Asks for |
|---|---|
| [sdformat#1734](https://github.com/gazebosim/sdformat/issues/1734): Support joint armature (reflected rotor inertia) | an `<armature>` under `<axis>/<dynamics>`; the gz-physics MuJoCo plugin could map it in one line |
| [gz-sim#4092](https://github.com/gazebosim/gz-sim/issues/4092): JointPositionController: option to use joint velocity for the D term | "derivative on measurement" via gz-math's `PID::Update(error, errorRate, dt)`, and docs saying `d_gain` acts on the change of the error |

Generic framing (moving MuJoCo-trained locomotion policies to Gazebo, Unitree G1). Not ros2_control: Diego's setup
doesn't use it (topics → ros_gz_bridge → Gazebo's own plugin). Gazebo's forum (for a possible write-up linking
both) is the Open Robotics Discourse, Gazebo category.

### Real robot vs the simulators (for context)

- The real G1's motors apply PD themselves: Unitree's deploy code sends per motor the target `q`, `kp`, `kd`, with
  `dq = 0` and `tau = 0` (`3p/unitree_rl_mjlab/deploy/include/FSM/State_RLBase.h:20-23`), i.e.
  τ = kp·(q_target − q) − kd·q̇: D on the measured velocity, no kick. The real joints have physical armature.
- So MuJoCo as trained (armature + D on velocity, implicit) matches the real control law and inertia best; the
  corrected Gazebo controller has the right law but no armature; Diego's original Gazebo controller differs in both.
- Which simulator is "closer to reality" overall can't be said without real-robot data: it depends mostly on the
  model parameters (armature, friction, contacts). Gazebo's advantage for this work is the deployment-like software
  setup (ROS 2 topics, separate processes and rates), not demonstrated physics fidelity.

### Idea: a MuJoCo → Gazebo policy-transfer package (later)

What this work has shown a transfer needs, which a reusable package could do automatically:

1. **Robot model:** convert the training model (MJCF / mjlab config) to SDF, keeping what training used: armature
   (approximated in link inertias until Gazebo supports it), joint damping, friction, torque limits.
2. **Joint control matching training:** P in the controller plugin, D as implicit joint damping (no derivative kick,
   no explicit damping), gains and limits from the policy's ONNX metadata.
3. **Policy execution:** a generic runner like `g1_policy_runner` (observations built from the ONNX metadata, sim or
   wall clock, configurable rate).
4. **Validation:** the same scenarios in MuJoCo and Gazebo (standing, walking, turning), logged in one format and
   compared (`summarize_runs.py`, `standing_summary.py`, side-by-side videos), with a stability check per joint
   ((ωₙΔt)² + 2·kd·Δt/I for explicit damping).
