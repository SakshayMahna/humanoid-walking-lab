# Q1b: Why does the corrected controller work and the original one not? (done)

Follow-up to [Q1](../q1_standing/): Unitree's policy drifts and turns in Gazebo with Diego's original joint
controller, and is MuJoCo-like with the corrected one. The two differ in three things at once. This folder finds
out which one matters, and why.

**Short answer:** the cause is a *combination* of two things, neither of which breaks the robot alone:
**joints without armature** (Gazebo's G1 has none, so the ankle-roll joint is 19× lighter than in training) and
**damping applied explicitly** (inside the PID plugin). Together they make the ankle-roll damper numerically
unstable. The derivative kick is a smaller, third effect; the gains and torque limits barely matter. The fix
applies the damping implicitly (SDF joint damping, integrated by the physics engine), which is stable at any
inertia, like MuJoCo in training.

## The two controllers

Gazebo drives each joint with a `JointPositionController` plugin. Every 1 ms physics step it computes a torque
from the joint's target (sent by the policy every 20 ms) and its measured angle q.

| | Original (Diego's model, set by Luis) | Corrected (`mj_gains_vel_d`) |
|---|---|---|
| (a) gains kp, kd | Luis's values (e.g. knee 148.6 / 9.46) | the policy's training values (knee 99.1 / 6.31) |
| (b) torque limits | ankle roll 10 N·m | ankle roll 50 N·m |
| (c) D term | inside the plugin's PID: on the change of the error | SDF joint `<damping>`, applied by the physics engine |

**How the original D term works** (gz-math `PID::Update`, read from the source). Error e = q − target;
torque = −kp·e − kd·(e − e_previous)/dt. Expanding the second term:

- **Target constant** (19 of every 20 steps): the error changes only because q moves, so Δe/dt = q̇ and the term
  is −kd·q̇: ordinary damping, the same as MuJoCo's position actuator.
- **Target jumps by Δ** (every 20 ms): e jumps by −Δ in one 1 ms step, so the term is +kd·Δ/0.001: a spike
  *toward* the new target, clipped at the motor limit. This is a "derivative kick".
- In both cases the plugin applies the damping as an **explicit** torque.

**Already in Luis's model:** the two ankle-roll joints also have SDF joint damping 1.5 and friction 0.1 (since
his first commit; Unitree's URDF has no damping anywhere). Every other joint has none. The corrected variants
replace that 1.5 with kd and keep the friction.

**The corrected D term:** the plugin is a pure P controller (`d_gain` 0), and kd becomes the joint's `<damping>`.
DART (Gazebo's physics engine) handles joint damping **implicitly**: its header says *"inverse of projected
articulated inertia for implicit joint damping and spring forces"*
(`dart/dynamics/GenericJoint.hpp` in the container). Same law, τ = kp·(target − q) − kd·q̇, but stable.

## Experiment 1: every combination, in MuJoCo

[`mujoco_controller_test.py`](mujoco_controller_test.py): Unitree's model and policy in MuJoCo at Gazebo's 1 ms
step (policy every 20 steps). MuJoCo's position actuators are switched off; the script computes and applies the
joint torques itself, so **only the controller changes**. Factors: gains (policy / Luis), limits (policy / Luis),
D term (explicit on velocity / implicit joint damping / on error, Gazebo-style), armature (on / off: MuJoCo's
joints carry the motors' reflected inertia; Gazebo's SDF has none). 24 combinations × standing (30 s) and walking
(stand 4 s, then vx = 0.5).

| Armature | D term | Gains, limits | Standing 30 s: moved, turned | Walking: wander | Fell |
|---|---|---|---|---|---|
| on | velocity, explicit | any | 0.01 m, +2.4° | 0.42-0.59° | no |
| on | joint damping, implicit | any | 0.01 m, +2.4° | 0.41-0.56° | no |
| on | on error (Gazebo) | any | 0.08-0.15 m, up to 14° | 0.88-1.46° | no |
| **off** | velocity, explicit | policy limits | spins (−500° to −1500°) | 44-99° | **yes** |
| **off** | velocity, explicit | Luis limits | 2.3-2.4 m, +238° to +361° | 1.6-1.8° | no |
| **off** | joint damping, implicit | any | **0.01 m, +2.5°** | **0.37-0.64°** | no |
| **off** | on error (Gazebo) | policy limits | 1.1-1.5 m, −113° to −734° | 68-73° | **yes** |
| **off** | on error (Gazebo) | Luis limits | **3.6-3.8 m**, +5° to +43° | **3.9-6.9°** | no |

The last row is Diego's original setup rebuilt inside MuJoCo, and it reproduces Gazebo's behaviour (Gazebo: 2.3-4.0 m
standing drift, 6.7° wander). The second-to-last row with joint damping reproduces the corrected setup.

### Why armature matters: the explicit-damping stability limit

Model one joint as the Week 1 pendulum: inertia I, a spring kp pulling toward the target, a damper kd resisting
speed, simulated in steps of Δt = 1 ms.

**Damper only, explicit** (force from the speed at the *start* of the step, applied for the whole step):

    v_next = v − Δt·(kd/I)·v = v·(1 − a),   with a = kd·Δt/I

- a < 1: the speed shrinks each step.
- 1 < a < 2: the speed flips sign each step but still shrinks (wobbly, stable).
- a > 2: the speed flips sign *and grows*: it explodes.

**With the spring** (ωₙ² = kp/I), semi-implicit Euler updates angle and speed with the matrix
[[1 − ωₙ²Δt², Δt(1 − a)], [−ωₙ²Δt, 1 − a]]. Its determinant is 1 − a and its trace is 2 − ωₙ²Δt² − a; it is stable
when |det| < 1 and |trace| < 1 + det, which gives the condition from the Week 1 post:

    (ωₙ·Δt)² + 2·kd·Δt/I < 4

For a pure damper this is a < 2 again.

**Left ankle roll, kd = 1.81, Δt = 1 ms:** with armature I = 0.0076, so a = 0.24 (the speed decays smoothly).
Without armature I = 0.0004, so **a = 4.5**: v_next = v·(1 − 4.5) = −3.5·v. Every millisecond the damper doesn't
just stop the joint, it throws it backwards 3.5× faster; only the torque limit stops it, so the ankle chatters.

**The same damper, implicit** (force from the speed at the *end* of the step):

    v_next = v − Δt·(kd/I)·v_next   →   v_next = v / (1 + a)

always shrinks and never flips, whatever the inertia. MuJoCo's `implicitfast` (Unitree's training setting) treats
actuator damping this way; DART does it for SDF joint damping.

Condition per joint (joint inertia I from MuJoCo's mass matrix, Δt = 1 ms):

| Joint | I with armature | I without | Condition, policy gains: with → without |
|---|---|---|---|
| ankle roll | 0.0076 | 0.0004 (armature = 95%) | 0.5 → **9.4** |
| wrist roll | 0.0040 | 0.0004 (91%) | → **5.0** |
| ankle pitch | 0.0100 | 0.0028 (72%) | stable |
| knee | 0.139 | 0.114 (18%) | stable |
| hip pitch | 0.913 | 0.903 (1%) | stable |

With Luis's gains: ankle roll 11.4, wrist roll 6.1. Every other joint stays below 4. With armature, every joint is
below 0.6.

## Experiment 2: the missing combination, in Gazebo

[`gazebo_confirm.sh`](gazebo_confirm.sh) runs a new variant, `luis_gains_vel_d` (Luis's kp, kd and limits; D as
joint damping), 2 repeats of each scenario:

| Unitree in Gazebo | Standing 30 s | Walking: wander | Walking: turn |
|---|---|---|---|
| Original (Luis, D in the PID) | 2.3-4.0 m, −91° to −128° | 6.7° | +2.6 ± 2.6°/s |
| **Luis's gains and limits, D as joint damping** | **0.03 m, 0.2°** | **0.36°** | **+0.49°/s** |
| Policy gains, D as joint damping (corrected) | 0.00-0.04 m, < 0.5° | 0.49° | +1.0°/s |
| MuJoCo reference | 0.01 m, 2.5° | 0.39° | +0.72°/s |

## The derivative kick, seen

[`plot_kick.py`](plot_kick.py) → `recordings/q1b/kick.png`: left knee, 0.5 s of walking, armature on. With D on the
error, every policy target step gives a 1 ms spike to the motor limit (±139 N·m vs ±24 N·m), toward the new target.
With armature, the knee's motion barely changes: the heavy joint absorbs it.

## What was happening, and the fix

**How much each factor matters** (Experiment 1, MuJoCo, Unitree's policy, policy gains and limits):

| Armature | Damping | Standing 30 s | Walking: wander |
|---|---|---|---|
| yes | explicit, on velocity | 0.01 m | 0.42° |
| yes | implicit (joint damping) | 0.01 m | 0.41° |
| yes | explicit + kick (Gazebo PID) | 0.10 m | 0.88° |
| **no** | implicit (joint damping) | **0.01 m** | **0.37°** |
| **no** | explicit, on velocity (no kick) | **falls** | falls |
| **no** | explicit + kick (Gazebo PID) | **falls** | falls |

(With Luis's 10 N·m ankle-roll limit, the last two rows don't fall but drift 2.4 m and 3.8 m.)

- **No armature alone** (4th row): fine, if the damping is implicit.
- **Explicit damping alone** (1st row): fine, if the joints have armature.
- **Both together** (5th, 6th rows): broken. This is the cause, and it is isolated: rows 4 and 5 differ only in
  how the damping is applied.
- **Derivative kick:** row 1 vs 3 (with armature), or 2.4 m → 3.8 m (without, Luis's limits). It makes things
  worse, but it is the smaller part.
- **Gains:** policy vs Luis changes the wander by ≈ 0.1-0.2° in the stable setups. The policy tolerates a 1.5×
  change in gains.
- **Torque limits:** only matter once a joint is unstable: they decide between "falls" and "chatters and drifts".
  Luis's 10 N·m ankle-roll limit is what keeps the original setup standing, so his tuning looks like a workaround
  for this instability (our interpretation; his commits don't say).
- **Wrist roll** is also past the limit (5.0-6.1) but barely affects balance. The ankle roll carries the sideways
  balance, so its chattering shows up as the drifting and turning.

**What was happening in Gazebo (original controller):** the plugin applies kd explicitly on joints without
armature → the ankle-roll damper overshoots every millisecond and chatters at the 10 N·m limit, with a kick every
20 ms on top → the policy sees the chattering in its joint and IMU observations and keeps stepping: sustained
drift and turning.

**What the fix does (corrected controller):** `d_gain` 0 in the plugin (pure P controller: no kick) and kd as SDF
joint `<damping>`. The torque law is the same as MuJoCo's position actuator, kp·(target − q) − kd·q̇, and it is now
integrated implicitly, as in training. It does **not** add armature: Gazebo's ankle is still 19× lighter than in
training, but implicit damping is stable anyway, and the policy tolerates the remaining difference.

### MuJoCo check with Luis's ankle damping and friction

Experiment 1 first left out Luis's ankle-roll damping (1.5) and friction (0.1). Rerun without armature, with them
(`run(..., extras=True)`):

| No armature, Luis's ankle damping + friction | Standing 30 s | Walking: wander |
|---|---|---|
| Luis's gains + limits, explicit PID + kick (= Gazebo original) | 1.67 m, −86.5° | 9.2° |
| Luis's gains + limits, explicit damping on velocity | 4.23 m, +219.7° | 0.9° |
| Luis's gains + limits, implicit joint damping | 0.01 m, +2.6° | 0.66° |
| Policy gains + limits, explicit PID + kick | 4.32 m, +59.9° | 4.6° |
| Policy gains + limits, implicit joint damping | 0.01 m, +2.5° | 0.40° |

Still the same picture, and the "original" row is now even closer to Gazebo (2.3-4.0 m, ≈ −90° to −130°, 6.7°).
The extra ankle damping stops the falls; the explicit setups still drift.

## Experiments 3 and 4: removing the instability in Gazebo *without* changing the controller

[`gazebo_armature_step.sh`](gazebo_armature_step.sh). Diego's original controller unchanged (Luis's gains and
limits, explicit PID with the kick), Unitree's policy, `clock:=sim` (the 0.2 ms runs are ≈ 0.6× real time),
2 repeats each:

| | Standing 30 s | Walking: wander | Walking: turn |
|---|---|---|---|
| Baseline (no armature, 1 ms) | 3.6-6.2 m, −151° to −163° | 5.1° | −0.4 to −1.4°/s |
| **A: + armature (approximated, below)** | **0.10 m**, −18° (settles) | **0.84-0.96°** | +1.1 to +1.4°/s |
| **B: 0.2 ms physics step** | **0.01-0.02 m**, +8° to +9° (settles) | **0.55°** | +0.05°/s |
| MuJoCo prediction for A (Experiment 1: armature, explicit PID + kick) | 0.10 m | 0.88° | |

Both remove the drift with the original controller left exactly as it is:

- **A** gives back the joints' inertia, so the explicit damper is stable again (a = 0.24 at the ankle). It lands
  exactly on MuJoCo's prediction; what remains is the kick's small cost (≈ 2× the wander of the corrected setup).
- **B** keeps the light joints but shrinks Δt: a = kd·Δt/I = 1.81 × 0.0002 / 0.0004 = 0.9, so 2a ≈ 1.8 < 4.
  That it works confirms the instability is numerical, not something about the robot or the policy.

The baseline with the sim clock is as bad as with the wall clock (Q1), so the clock was not the problem.

### How the armature is approximated in Gazebo

Gazebo Harmonic has no joint armature parameter in this stack: "armature", "rotor inertia" or "reflected" appear
in none of SDFormat 14, gz-physics 7's DART plugin, gz-sim or DART's joint headers (checked in the container).
So `make_gain_variants.py` (variant `original_armature`) puts it in the links instead:

1. For each of the 29 joints, take the motor's armature I_a from the same motor model as MuJoCo (e.g. ankle roll
   0.0072 kg·m², knee 0.0251).
2. Add I_a to the **child link's** inertia about the joint axis. All axes are along x, y or z, and every link frame
   coincides with its joint frame, so this adds I_a to one diagonal element (e.g. `ixx` of
   `left_ankle_roll_link` for the roll axis x).
3. A link inertia must be physically valid: each principal moment at most the sum of the other two. Adding I_a to
   one axis of a tiny link breaks that (ankle roll link: 0.0002 + 0.0072 > 0.0016 + 0.0017). So the same I_a is
   added on one perpendicular axis too, which is the inertia of a thin rod. That axis is chosen perpendicular to
   the *parent* joint's axis as well (ankle roll: x for the joint, y for the ankle pitch above, so z), so the joint
   just above barely sees the extra inertia.
4. Example, left ankle roll link: (ixx, iyy, izz) = (0.00022, 0.00162, 0.00167) → (0.00744, 0.00162, 0.00889).

How it differs from real armature: armature only adds to its own joint. Inertia added to a link also counts for
every joint above it in the chain (ankle → knee → hip). That is small next to the leg's own inertia (≈ 0.007 vs
0.9 at the hip), but not zero; the perpendicular term also adds some inertia about a third axis.

## Videos

In `recordings/q1b/` (not tracked; regenerate with the commands below):

| File | Shows |
|---|---|
| `grid_stand.mp4`, `grid_walk.mp4` | MuJoCo, 2×2: (1) like training, (2) original controller without armature: drifts, (3) the same with a 50 N·m ankle limit: falls, (4) corrected controller without armature: stable |
| `gazebo_stand_compare.mp4` | Gazebo, standing: original controller vs Luis's gains with joint damping vs corrected. Same gains in the first two; only the damping changes |
| `armature_step_stand.mp4`, `armature_step_walk.mp4` | Gazebo, original controller unchanged: no armature vs + armature in the links (A) vs 0.2 ms step (B). Wall-time videos: B runs at ≈ 0.4-0.6× real time with the GUI, so it plays slower |
| `kick.png` | Knee torque with and without the derivative kick |

## Files

| File | |
|---|---|
| [`mujoco_controller_test.py`](mujoco_controller_test.py) | Experiment 1: 24 controller combinations × 2 scenarios in MuJoCo → `recordings/q1b/` |
| [`plot_kick.py`](plot_kick.py) | Knee torque plot from the traces of experiment 1 |
| [`gazebo_confirm.sh`](gazebo_confirm.sh) | Experiment 2: a controller variant in Gazebo, standing and walking |
| [`render_videos.py`](render_videos.py) | MuJoCo videos of four representative setups, as 2×2 grids |
| [`gazebo_armature_step.sh`](gazebo_armature_step.sh) | Experiments 3-4: original controller + armature, and at a 0.2 ms step |
| [`record_armature_step.sh`](record_armature_step.sh) | Recorded versions of experiments 3-4 and their side-by-side videos |

```bash
.venv/bin/python open_questions/q1b_controller/mujoco_controller_test.py
.venv/bin/python open_questions/q1b_controller/plot_kick.py
open_questions/q1b_controller/gazebo_confirm.sh 2 luis_gains_vel_d
.venv/bin/python open_questions/q1b_controller/render_videos.py
open_questions/q1b_controller/gazebo_armature_step.sh 2
open_questions/q1b_controller/record_armature_step.sh
```

Raised upstream: joint armature in SDFormat ([sdformat#1734](https://github.com/gazebosim/sdformat/issues/1734)) and a
velocity-based D term for `JointPositionController` ([gz-sim#4092](https://github.com/gazebosim/gz-sim/issues/4092)).
