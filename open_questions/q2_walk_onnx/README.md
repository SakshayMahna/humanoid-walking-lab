# Q2: why does walk.onnx get worse with the corrected controller?

**Status: open, narrowed down.**

- Test 1 (MuJoCo): without armature, implicit damping is enough for walk.onnx, as for Unitree's policy.
- Test 2-3 (Gazebo, 3 repeats): without armature it is **not** enough. walk.onnx fails (walks off while standing,
  0.72 m/s and sliding while walking); Unitree's policy walks off while standing in 2 of 3 runs on the sim clock.
  **Corrected controller + armature fixes both policies in every run.**
- The velocity input is ruled out: walk.onnx sees its true velocity (same frame, scale 1.00, no extra noise).
- Remaining: why Gazebo needs the armature when MuJoCo does not.

## The question

In Gazebo, the corrected controller (`mj_gains_vel_d`: the policy's kp, kd and limits, D term as joint damping)
fixes Unitree's policy but makes `walk.onnx` worse. Gazebo logs (`recordings/gains/`, wall clock, 2 runs each),
command vx = 0.5:

| Gazebo controller | Policy | vx (m/s) | vy (m/s) | wander (deg) | pelvis z (m) |
|---|---|---|---|---|---|
| original (Luis) | walk.onnx | 0.59-0.61 | −0.04 to −0.08 | 1.9-5.2 | 0.68-0.69 |
| `mj_gains` | walk.onnx | 0.56-0.57 | −0.03 to −0.04 | 1.3-1.6 | 0.66-0.67 |
| **`mj_gains_vel_d`** | **walk.onnx** | **0.72-0.73** | **−0.24** | 1.6-1.7 | **0.60** |
| `mj_gains_vel_d` | Unitree | 0.50 | −0.02 | 0.49 | 0.68 |

So with `mj_vel_d`, walk.onnx crouches (7 cm lower), walks too fast and slides sideways.

Pelvis heights can only be compared **within** one simulator: Unitree's policy reads 0.78 m in MuJoCo and
0.68-0.70 m in every Gazebo variant, so Gazebo's odometry height sits about 0.1 m lower.

## Test 1: the Q1b controller factorial with walk.onnx (MuJoCo)

**Idea.** In Q1b, Unitree's policy did not care about the missing armature as long as the damping was implicit.
The corrected Gazebo controller also has no armature. Maybe walk.onnx is more sensitive to it.

**Setup.** Exactly the Q1b runner ([`../q1b_controller/mujoco_controller_test.py`](../q1b_controller/mujoco_controller_test.py)):
MuJoCo at Gazebo's 1 ms step, policy every 20 ms, MuJoCo's actuators off and the joint torques computed by the
script, so only the controller changes. walk.onnx runs on its own MuJoCo model (`robot/g1.xml`, knees-bent pose,
the setup validated in exercises 05-06).

The script first checks that it runs the right policy: the file has the same MD5 as the one Gazebo loads
(`7bf59b5a…`, from `ros2/g1_policy_runner/config/walk.yaml`), and the "policy" kp, kd and default pose equal the
values stored in the ONNX metadata.

Factors (24 combinations, plus the rebuilt original with Luis's ankle damping and friction, `_x`):

| Factor | Values |
|---|---|
| armature | on (training) / off (Gazebo) |
| gains | policy (ONNX metadata) / luis (Gazebo original) |
| limits | policy / luis |
| D term | `vel` explicit −kd·q̇ / `jnt` joint damping, implicit (= corrected controller) / `err` Gazebo PID on the error, with the kick (= original controller) |

Scenarios: stand (zero command, 30 s) and walk (zero for 4 s, then vx = 0.5).

### Results (policy gains and limits; the luis rows behave the same way, full table below)

| Armature | D term | Stand: moved in 30 s | Walk vx | Walk vy | Wander | Pelvis z |
|---|---|---|---|---|---|---|
| on | `vel` | 0.01 m | 0.45 | −0.01 | 0.22° | 0.76 |
| on | `jnt` | 0.01 m | 0.45 | −0.01 | 0.23° | 0.76 |
| on | `err` | 0.09 m | 0.50 | −0.03 | 0.72° | 0.75 |
| **off** | **`jnt`** (corrected Gazebo controller) | **0.01 m** | **0.45** | **−0.01** | **0.28°** | **0.76** |
| off | `vel` | falls | falls | | | |
| off | `err` | falls | falls | | | |
| off, luis gains + limits | `err` + ankle extras (rebuilt original) | 1.86 m, −136° | 0.59 | 0.01 | 8.3° | 0.77 |

**What it shows.** walk.onnx reacts to the controller **exactly like Unitree's policy did** in Q1b:

- No armature with explicit damping (`vel`, `err`) is unstable: it falls with the policy limits and drifts
  metres with Luis's lower limits.
- No armature with **implicit** damping (`jnt`, the corrected controller) is as good as the training setup:
  stands still, walks at 0.45 m/s (its MuJoCo speed), no sideways slide, no crouch.

So in MuJoCo the corrected controller does **not** make walk.onnx crouch, over-speed or slide. On its own, this
MuJoCo test cannot explain Q2 — Test 2 checks the same setups in Gazebo itself.

## Test 2: the same four setups in Gazebo (recorded, sim clock)

**Idea.** Test 1 says "armature does not matter with the corrected controller" — but that is MuJoCo. Check it in
Gazebo: walk.onnx with the original and corrected controllers, each with and without armature (armature
approximated in the link inertias, as in Q1b; the new variant `mj_gains_vel_d_armature` has both fixes).
One recorded run per setup and scenario, `clock:=sim`, RTF 0.90-0.92.

| Gazebo setup | Stand: moved, turned in 30 s | Walk vx | Walk vy | Wander | Pelvis z (walk) |
|---|---|---|---|---|---|
| 1. Original controller | 1.25 m, −18° | 0.59 | −0.08 | 3.8° | 0.68 |
| 2. Original + armature | 0.62 m, −30° | **0.00** (steps in place) | 0.01 | 0.6° | 0.65 |
| 3. Corrected controller | **6.76 m**, +49° (walks off) | **0.72** | **−0.24** | 1.2° | **0.60** |
| **4. Corrected + armature** | **0.00 m, 0°** | **0.55** | **−0.02** | **0.4°** | 0.65 |

MuJoCo, same four setups (Test 1): 1 drifts 1.86 m / −136° and wanders 8°; 2, 3 and 4 all stand still and walk
at 0.45-0.47 m/s.

**What it shows.**

- Setup 3 reproduces the old Gazebo result on the sim clock (0.72 m/s, −0.24 sideways, crouched), and it is worse
  standing: walk.onnx walks off 6.8 m with a zero command. So Q2 is real, not a timing artifact.
- **Setup 4 fixes it**: walk.onnx stands perfectly still and walks straight at 0.55 m/s, like Unitree's policy
  with the corrected controller. So in Gazebo, walk.onnx needs **both** fixes; Unitree's policy needed only the
  corrected controller.
- Setup 2 (original + armature) stepped in place in this single recorded run; it did not reproduce in Test 3
  (3 of 3 walk at 0.55 m/s).

**So the controller and armature are not ruled out after all** — Test 1 ruled them out only *in MuJoCo*. The
open question becomes: why does Gazebo without armature break walk.onnx (and not Unitree's policy) when MuJoCo
without armature does not? MuJoCo's no-armature model is not a faithful copy of Gazebo's: something else differs
between the two robots or engines, and walk.onnx is more sensitive to it.

## Test 3: headless repeats with the velocity input logged (Gazebo, sim clock, 3 runs each)

[`gazebo_repeats.sh`](gazebo_repeats.sh): the four setups for walk.onnx, plus Unitree's policy with the corrected
controller with and without armature. Every odometry message (≈ 330 Hz: 400 Hz on 1 ms steps) is logged, i.e.
exactly the `base_lin_vel` walk.onnx receives (the node copies `twist.linear` into the observation unchanged).
RTF 0.96-1.00.

| Policy, Gazebo setup | Stand: moved, \|turned\| in 30 s | Walk vx | vy | turn (°/s) | wander | z (walk) |
|---|---|---|---|---|---|---|
| walk.onnx, original | 0.42 m, 33° | 0.60 | −0.05 | −1.6 | 4.6° | 0.69 |
| walk.onnx, original + armature | 0.34 m, 35° | 0.55 | −0.01 | −2.5 | 0.8° | 0.68 |
| walk.onnx, corrected | **3.0-4.0 m** (3 of 3 walk off) | **0.72** | **−0.23** | −2.4 | 1.4° | **0.60** |
| **walk.onnx, corrected + armature** | **0.03 m, 0.6°** | **0.55** | −0.02 | 0.3 | **0.4°** | 0.65 |
| Unitree, corrected | **9.6 / 0.02 / 10.0 m** (2 of 3 walk off) | 0.51 | −0.02 | 0.9 | 0.5° | 0.68 |
| **Unitree, corrected + armature** | **0.02 m, 0.2°** | 0.49 | −0.01 | 0.7 | 0.7° | 0.69 |

MuJoCo references (training-like setup): walk.onnx 0.45 m/s, Unitree 0.50 m/s, both stand still.

**Findings.**

- **Both fixes together work for both policies, in every run.** Corrected + armature: both stand still
  (≤ 0.03 m), walk straight, wander < 1°.
- **The corrected controller alone is marginal for both policies in Gazebo**, not only for walk.onnx. walk.onnx
  fails standing and walking; Unitree's policy fails standing in 2 of 3 sim-clock runs (it starts moving 3-5 s
  after spawn and walks ≈ 10 m). So the original Q2 premise ("Unitree gets better") is only half true.
- **The clock matters for that marginal case** (a Q3 finding): Unitree, corrected, standing, **wall** clock:
  0.03-0.04 m in 3 of 3 runs (`wall_unitree_corrected_stand_*`), and 6 of 6 in Q1. Sim clock: 2 of 3 walk off.
  With armature, the sim clock is fine too.
- **The single "steps in place" run (Test 2, original + armature) did not reproduce**: 3 of 3 walk at 0.55 m/s.
- One difference survives the fixes: walk.onnx walks at 0.55 m/s in Gazebo vs 0.45 in MuJoCo (Unitree: 0.49 vs
  0.50).

### The velocity input is not the problem

[`velocity_analysis.py`](velocity_analysis.py) compares what walk.onnx saw with the velocity of the same point
computed from the logged pose ((p(t) − p(t − 10 ms)) / 10 ms, rotated into the pelvis frame), and measures the fast
"shake" (std around a 0.1 s moving average). MuJoCo reference: [`mujoco_velocity_log.py`](mujoco_velocity_log.py)
(velocimeter, same four setups).

| Walking (cmd 0.5) | seen vx | true vx | seen vy | slope seen/true | shake x, y, z (m/s) |
|---|---|---|---|---|---|
| Gazebo, walk.onnx, corrected | 0.722 | 0.722 | −0.234 | 1.00 | 0.009, 0.012, 0.022 |
| Gazebo, walk.onnx, corrected + armature | 0.553 | 0.553 | −0.020 | 1.00 | 0.008, 0.006, 0.017 |
| MuJoCo, walk.onnx, corrected | 0.453 | 0.453 | −0.012 | 1.00 | 0.008, 0.007, 0.022 |

- **Seen = true** in every run: right frame, right scale, no offset. When walk.onnx goes 0.72 m/s, it *knows* it
  is going 0.72 m/s; it just cannot slow down. So this is not a measurement error.
- **No extra noise**: Gazebo's shake is the same size as MuJoCo's velocimeter.
- Side note: seen vz = −0.073 while walking with the corrected controller (vs −0.018 with armature). In the pelvis
  frame that means the pelvis is pitched forward by about atan(0.073 / 0.72) ≈ 6°: walk.onnx walks crouched and
  leaning forward.

## What is left

Why does Gazebo without armature break the policies when MuJoCo without armature does not? Differences that remain
between the two no-armature setups:

| Difference | Notes |
|---|---|
| Physics engine: DART vs MuJoCo (contacts, joint integration) | MuJoCo's no-armature test is not a faithful copy of Gazebo |
| Robot model: SDF link masses and inertias, foot collision shapes vs MuJoCo's XML | Without armature the small links (ankles, wrists) have tiny inertias, so their values matter more |
| Policy timing: sim-clock timer (fails more often) vs wall clock | Q3 |

Candidate next tests: compare the link inertias and masses of the SDF with the MuJoCo XML (cheap, no simulation);
Q3 (the clock) with repeats.

## Videos (`recordings/q2/`, not tracked)

| File | What |
|---|---|
| `mujoco_grid_stand.mp4`, `mujoco_grid_walk.mp4` | Test 1, the four setups in MuJoCo |
| `gazebo_grid_stand.mp4`, `gazebo_grid_walk.mp4` | Test 2, the same four setups in Gazebo |
| `policy_compare_stand.mp4`, `policy_compare_walk.mp4` | Corrected controller without / with armature, Unitree vs walk.onnx, Gazebo, sim clock ([`record_policy_compare.sh`](record_policy_compare.sh)). In this recording Unitree without armature happened to stand still (it walks off in 2 of 3 headless sim-clock runs; now 2 of 4) |
| `gz_<setup>_<scenario>.mp4`, `gz_unitree_<setup>_<scenario>.mp4` | Single Gazebo recordings |

## Files

| File | What it does |
|---|---|
| [`mujoco_factorial.py`](mujoco_factorial.py) | Test 1: checks the policy file and gains, then the 24-combination factorial + rebuilt original → `recordings/q2/` |
| [`render_videos.py`](render_videos.py) | Test 1 videos: four setups in MuJoCo → `mujoco_grid_{stand,walk}.mp4` |
| [`gazebo_record.sh`](gazebo_record.sh) | Test 2: the same four setups in Gazebo, recorded, sim clock → `gazebo_grid_{stand,walk}.mp4` |
| [`record_policy_compare.sh`](record_policy_compare.sh) | Video: corrected controller without / with armature, both policies → `policy_compare_{stand,walk}.mp4` |
| [`gazebo_repeats.sh`](gazebo_repeats.sh) | Test 3: headless repeats, both policies, velocity input logged → `rep_*.{csv,odom.csv}` |
| [`mujoco_velocity_log.py`](mujoco_velocity_log.py) | MuJoCo reference for the velocity log (velocimeter, four setups) → `mj_walk_*.odom.csv` |
| [`velocity_analysis.py`](velocity_analysis.py) | Seen vs true velocity, scale and shake, for Gazebo or MuJoCo `.odom.csv` |
| [`summarize.py`](summarize.py) | Table of standing and walking metrics (with pelvis height) for MuJoCo or Gazebo CSVs |

```bash
.venv/bin/python open_questions/q2_walk_onnx/mujoco_factorial.py          # ~1 min
.venv/bin/python open_questions/q2_walk_onnx/render_videos.py
open_questions/q2_walk_onnx/gazebo_record.sh                              # ~15 min
open_questions/q2_walk_onnx/gazebo_repeats.sh 3                           # ~35 min
.venv/bin/python open_questions/q2_walk_onnx/mujoco_velocity_log.py
.venv/bin/python open_questions/q2_walk_onnx/summarize.py recordings/q2/rep_*.csv
.venv/bin/python open_questions/q2_walk_onnx/velocity_analysis.py recordings/q2/rep_*.odom.csv recordings/q2/mj_walk_*.odom.csv
.venv/bin/python open_questions/q2_walk_onnx/summarize.py recordings/q2/gz_*.csv
.venv/bin/python open_questions/q2_walk_onnx/summarize.py recordings/q2/*.csv
.venv/bin/python open_questions/q2_walk_onnx/summarize.py recordings/gains/*.csv recordings/mujoco/mujoco_*.csv
```

<details><summary>Full factorial table</summary>

```
run                                            move     rot  z_st |    vx     vy   turn wander  z_wk
walk_armoff_luisgains_luislimits_err           3.12    18.4  0.77 |  0.58   0.03  -4.67   5.10  0.77
walk_armoff_luisgains_luislimits_err_x         1.86  -135.8  0.77 |  0.59   0.01  -1.17   8.30  0.77
walk_armoff_luisgains_luislimits_jnt           0.01    -3.6  0.77 |  0.45  -0.01  -1.09   0.80  0.77
walk_armoff_luisgains_luislimits_vel           2.82  -188.8  0.78 |  0.51  -0.01 -23.82   4.23  0.78
walk_armoff_luisgains_policylimits_err         fell                |   fell
walk_armoff_luisgains_policylimits_jnt         0.01    -3.6  0.77 |  0.45  -0.01  -1.09   0.80  0.77
walk_armoff_luisgains_policylimits_vel         fell                |   fell
walk_armoff_policygains_luislimits_err         2.60    16.6  0.75 |  0.52   0.04  -0.61   6.09  0.75
walk_armoff_policygains_luislimits_jnt         0.01    -1.4  0.76 |  0.45  -0.01   0.50   0.28  0.76
walk_armoff_policygains_luislimits_vel         4.47  -107.9  0.76 |  0.40  -0.01 -13.49   2.51  0.76
walk_armoff_policygains_policylimits_err       fell                |   fell
walk_armoff_policygains_policylimits_jnt       0.01    -1.4  0.76 |  0.45  -0.01   0.50   0.28  0.76
walk_armoff_policygains_policylimits_vel       fell                |   fell
walk_armon_luisgains_luislimits_err            0.05    -4.5  0.76 |  0.47   0.00  -1.86   1.13  0.77
walk_armon_luisgains_luislimits_jnt            0.01    -3.7  0.77 |  0.45  -0.01  -1.82   0.72  0.77
walk_armon_luisgains_luislimits_vel            0.01    -3.6  0.77 |  0.44  -0.01  -1.85   0.71  0.77
walk_armon_luisgains_policylimits_err          0.05    -4.6  0.76 |  0.48   0.00  -1.42   0.98  0.77
walk_armon_luisgains_policylimits_jnt          0.01    -3.7  0.77 |  0.45  -0.01  -1.82   0.72  0.77
walk_armon_luisgains_policylimits_vel          0.01    -3.6  0.77 |  0.44  -0.01  -1.85   0.71  0.77
walk_armon_policygains_luislimits_err          0.23     0.9  0.75 |  0.50  -0.02  -0.25   0.59  0.75
walk_armon_policygains_luislimits_jnt          0.01    -1.4  0.76 |  0.45  -0.01  -0.09   0.23  0.76
walk_armon_policygains_luislimits_vel          0.01    -1.4  0.76 |  0.45  -0.01  -0.02   0.22  0.76
walk_armon_policygains_policylimits_err        0.09    10.0  0.76 |  0.50  -0.03   0.23   0.72  0.75
walk_armon_policygains_policylimits_jnt        0.01    -1.4  0.76 |  0.45  -0.01  -0.09   0.23  0.76
walk_armon_policygains_policylimits_vel        0.01    -1.4  0.76 |  0.45  -0.01  -0.02   0.22  0.76
stand: 30 s zero command, move m, rot deg, z_st mean pelvis height after 5 s (m)
walk: cmd vx 0.5 from 4 s, measured from 8 s: vx, vy m/s (pelvis frame), turn deg/s, wander deg, z_wk mean pelvis height (m)
```

</details>
