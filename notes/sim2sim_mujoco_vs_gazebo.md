# Sim-to-sim: MuJoCo (training) vs Gazebo (Diego's ROS 2 deployment)

Parked for later. Everything here was checked against files or source code (links below); open items are marked.
The policies were trained in mjlab (MuJoCo); Diego's deployment runs them in Gazebo Harmonic via `g1_sim`.

## Where the Gazebo setup came from

- Diego's `g1_sim` is based on Luis Domínguez Besteiro's [gazebo-harmonic-ros](https://github.com/luisdbesteiro/gazebo-harmonic-ros)
  (Docker, Gazebo Harmonic + ROS 2 Humble, created 2026-05-08, 32 commits, all by Luis) and Unitree's
  [g1_description](https://github.com/unitreerobotics/unitree_ros/tree/master/robots/g1_description).
  Source: Diego's post "Weeks 29-30 – Simulating the robot in Gazebo" (`docs/_posts/2026-05-28-weeks29-30.md`).
- Diego ported it to Jazzy and a normal colcon workspace. Local clone of Luis's repo: `3p/gazebo-harmonic-ros`.

## Diego's core idea: split execution into rates

The research question is how to run a policy trained in MuJoCo (mjlab) inside a ROS 2 / Gazebo pipeline, and his
approach is to decouple simulation, inference and commands into separate loops at their own rates.
From his posts in `3p/2025-tfg-diego-lopez/docs/_posts/`:

- **Weeks 20-22** (`2026-04-15-weeks20-22.md`): instead of "inference every simulation iteration before updating
  the simulator", three threads: **simulation** (MuJoCo, as fast as possible), **network** (inference at
  **50 Hz**), **control** (velocity commands at **10 Hz**). "Should improve performance and make the execution
  flow more realistic by separating simulation, control, and inference tasks."
- **Weeks 23-28** (`2026-05-13-weeks23-28.md`): MuJoCo ran much faster than real time (16 s simulated in about
  2.7 s), so the threads were moved to **simulation time**, with a "micro-sleep" loop that waits until sim time
  reaches the next scheduled tick. (Oscar fixed the observation/action bugs in the same period.)
- **Weeks 29-30**: in Gazebo, the same split comes from ROS 2. Gazebo steps physics (1 ms), sensors publish at
  400 Hz, and the C++ node runs inference on a 20 ms timer. That timer is **wall clock**, unlike his MuJoCo
  threads (sim time) and Luis's Python node (sim time).
- **Summer** (`2026-09-03-Summer.md`): a ROS 2 test node sends a Twist for 10 s and measures the error between
  the ideal final position and the reached one. Gazebo error > MuJoCo error. Inference rate swept **10-70 Hz**:
  "best in practice around 60 Hz", "stable even at 20 Hz". He also saw the robot "tends to turn slightly to
  the right" and suspected the ROS 2 code. **Unknown:** which policy file and elbow setup those tests used.

What this means for our comparisons: the rate of each loop, which clock it follows, and whether loops run in sync
or not are all variables. Training had one: synchronous, 200 Hz physics, 50 Hz policy, both in sim time.
`g1_policy_runner` exposes `clock:=wall|sim` and `policy_period` so we can repeat his rate sweep with both clocks.

## Differences found so far

| Area | MuJoCo (mjlab training / our exercises) | Gazebo (`g1_sim`) |
|---|---|---|
| Physics engine | MuJoCo | DART (`gz-physics-dartsim-plugin`), ODE collision detector |
| Timestep | 5 ms, policy every 4 steps | 1 ms (`max_step_size`), policy on its own timer |
| PD gains | from armature: ωₙ = 10 Hz, ζ = 2 (= ONNX metadata) | Luis's values, see below |
| D term | −kd·q̇ (joint velocity) | kd·Δe/dt on the position error (derivative kick) |
| Torque limits | ankle roll 50 N·m | ankle roll 10 N·m (`cmd_min/max`); others match |
| Armature (reflected rotor inertia) | yes, per motor type | none in the SDF |
| Joint damping/friction | none extra | ankle roll: damping 1.5, friction 0.1 (Luis); all other joints none |
| Foot friction | 0.6 on foot geoms (priority 1), condim 3 | ground plane mu = mu2 = 1.0 |
| Collision geometry | foot spheres/capsules from mjlab XML | Unitree URDF-derived collisions (not yet compared) |
| Linear velocity input | velocimeter at `imu_in_pelvis` site | OdometryPublisher: finite difference of the pose at offset (0.04525, 0, −0.08339) = same point, body frame, moving average |
| IMU | MuJoCo gyro + framequat | gz IMU sensor, 400 Hz |
| Policy clock | exactly 20 ms of sim time | Diego: `create_wall_timer(20ms)` (real time); Luis's Python node used sim time (commit `e0a9110`) |
| Policy/sim coupling | synchronous (same loop) | asynchronous: callbacks store the latest values, timer reads them |
| Default world | — | launch default is `empty_world.sdf` (no `<physics>` block: Gazebo defaults) |
| Spawn | keyframe at 0.76 m (Diego) / 0.8 m (Unitree) | `spawn_z` 0.80 |
| Elbows (free-elbows policy) | — | no command, no `initial_position` → held at 0 rad (`jointPosCmd{0.0}`) |

### PD gains in Gazebo (mystery, partly explained)

| Joints | MuJoCo kp / kd | Gazebo kp / kd | Ratio |
|---|---|---|---|
| hip pitch/yaw, waist yaw (7520-14) | 40.18 / 2.56 | 44.11 / 2.81 | ×1.10 |
| hip roll, knee (7520-22) | 99.10 / 6.31 | 148.56 / 9.46 | ×1.50 |
| shoulders, elbows, wrist roll (5020) | 14.25 / 0.91 | 17.22 / 1.10 | ×1.21 |
| waist roll/pitch, ankles (2×5020) | 28.50 / 1.81 | 34.44 / 2.19 | ×1.21 |
| wrist pitch/yaw (4010) | 16.78 / 1.07 | 5.00 / 0.50 | hand-tuned |

- Set by Luis in commit `e00af69` (2026-05-22) "Adjusted PD controllers in base model to better align with MuJoCo
  training" (before: round values like 70/5). Wrists retuned in `411c053` "reducing induced oscillations".
- kd/kp is identical in both sets (0.0637 = 2ζ/ωₙ), so ωₙ = 10 Hz and ζ = 2 were kept and only the inertia
  changed (×1.10 / ×1.50 / ×1.21). **Unknown:** which inertia he used. No derivation in his repo; web search found nothing.

### D term: why Gazebo kicks

gz-math `PID::Update(error, dt)` ([source, gz-math7](https://github.com/gazebosim/gz-math/blob/gz-math7/src/PID.cc)) uses
`dErr = (error - errorLast) / dt` with `error = q - target`
([JointPositionController.cc, gz-sim8](https://github.com/gazebosim/gz-sim/blob/gz-sim8/src/systems/joint_position_controller/JointPositionController.cc)).

- Target constant (19 of every 20 physics steps): Δe/dt = q̇ → same as MuJoCo's −kd·q̇.
- Target jumps (once per policy step): Δe = −Δtarget in one 1 ms step → torque +kd·Δtarget/0.001,
  pointing toward the new target (a "derivative kick").
  Knee example: 9.46 × 0.05 rad / 0.001 s ≈ 473 N·m → clipped to 139 N·m for 1 ms.
- Measured and explained in [open_questions/q1b_controller](../open_questions/q1b_controller/): the kick alone is
  minor; the real problem is that the plugin applies damping explicitly on joints with no armature (unstable at
  ankle roll and wrist roll). SDF joint damping is implicit in DART and fixes it.

## Experiments to run later

1. **D-term kick (graphs).** One joint (e.g. knee), same target sequence (step every 20 ms, or a recorded policy
   target trace) in both simulators. Plot: target, q, torque vs time. Expect: MuJoCo smooth, Gazebo spikes to the
   torque limit at each target change. Variants: Gazebo with `d_gain = 0` + joint `<damping>` (≈ −kd·q̇).
   Could start in MuJoCo alone by implementing the gz-math PID in Python as a custom controller (same physics,
   only the D term differs), then confirm in Gazebo.
2. **Wall clock vs sim time.** Same policy and command, timer on wall clock vs `/clock`; also under CPU load
   (RTF < 1). Measure real policy rate in sim time, velocity tracking, yaw drift, falls. (Part of the policy
   package work: `clock:=wall|sim`.)
3. **Contact and friction.** Foot friction 0.6 vs 1.0, collision geometry (spheres vs URDF meshes/boxes),
   contact softness (MuJoCo solref/solimp vs DART). Slip and foot placement during walking.
4. **Joint dynamics and actuator control.** Armature present vs absent, gains (metadata vs Luis), torque limits
   (ankle roll 10 vs 50 N·m), the D term above. Step response of a single joint in each simulator.
5. **Integration and solver.** MuJoCo Euler / implicitfast at 5 ms vs DART at 1 ms. Same standing/walking
   scenario at different steps; energy drift; penetration.
6. **Policy execution timing.** Asynchronous sensing (stale observations), observation latency, timer jitter,
   running inside the physics loop (Luis's Gazebo system plugin, commit `0fa18e0`, `policy_gz_system.cpp`).

## Measurements

Experiment runs, scripts and results are in [gazebo_experiments.md](gazebo_experiments.md).
