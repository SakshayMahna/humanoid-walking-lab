# Q1: Why can't the robot stand still in Gazebo? (done)

With a zero command, the G1 stands still in MuJoCo (< 1°, ≈ 1 cm) but rotates and drifts in Gazebo.
Part of the open questions in [notes/gazebo_experiments.md](../../notes/gazebo_experiments.md#7-open-questions).

## Reasoning

Two kinds of cause would look the same in a short run:

- **Start-up cause:** something only at the beginning disturbs the robot, and the policy steps to recover.
  In Gazebo the start differs from MuJoCo: the policy node runs ≈ 2 s on all-zero observations before the robot
  exists (its outputs are fed back as `last_action`), the robot spawns with **all joints at 0 rad** (straight legs;
  the SDF has no initial positions) and is dropped from 0.80 m. In MuJoCo it starts at rest in the policy's
  default pose. The start is also the least repeatable part (wall-clock timer vs the spawn moment), which would
  explain run-to-run differences.
- **Sustained cause:** something acts all the time (controller, contacts, a biased observation).

**Test 1** tells them apart: stand for 30 s with a zero command. A start-up cause fades (≈ 0 movement late in
the run); a sustained cause keeps moving.

## Test 1: 30 s standing, Unitree's policy (2026-10-21)

Two Gazebo controller setups × 3 recorded repeats, plus MuJoCo. "Original controller" = Diego's robot model as-is (Luis's gains, D on
the error). "Corrected" = policy gains + D on joint velocity (`gazebo/make_gain_variants.py`).

| Run | Moved after 30 s | Turned | Movement 0-10 s | 10-20 s | 20-30 s | Verdict |
|---|---|---|---|---|---|---|
| MuJoCo | 0.01 m | +2.5° | 0.01 | 0.00 | 0.00 | settled |
| Original controller, 1 | 3.96 m | −117.7° | 1.60 | 0.97 | 2.55 | still moving |
| Original controller, 2 | 2.30 m | −91.4° | 0.54 | 0.91 | 1.25 | still moving |
| Original controller, 3 | 3.06 m | −128.0° | 1.43 | 1.27 | 1.58 | still moving |
| Corrected, 1 | 0.00 m | −0.2° | 0.00 | 0.00 | 0.00 | settled |
| Corrected, 2 | 0.00 m | +0.0° | 0.00 | 0.00 | 0.00 | settled |
| Corrected, 3 | 0.00 m | −0.0° | 0.00 | 0.00 | 0.00 | settled |

Videos: `recordings/standing/compare_1.mp4` … `compare_3.mp4`, all three stacked in `compare_all.mp4`.

**Results:**

1. **Original controller: sustained.** The robot keeps walking and turning for the whole 30 s (1-2.5 m in *every*
   10 s window, ≈ −100° to −130° in total). Not a start-up transient. Since the corrected controller removes it
   completely, the cause is the joint controller: Luis's gains and/or the D-term kick on every new target.
2. **Corrected controller: stands perfectly still** in all three runs, even more still than MuJoCo.

**Caveat:** in recorded runs, logging and video start at ≈ 5.3 s of simulated time, about 3 s after the robot
spawns (the script waits for the GUI). The first seconds after spawn are not in these logs.

## Test 2: without the Gazebo window (2026-10-21)

Why: two earlier **headless** runs with the corrected controller (`recordings/gains/mj_vel_d_unitree_wall_*`,
gain experiment) walked off at ≈ 0.6 m/s about 2 s after spawn while `/cmd_vel` was 0, but the recorded runs of
test 1 never did. The window loads the CPU, which lowers the real-time factor, so the wall-clock policy acts at a
different rate in simulated time. Test: same zero-command run, corrected controller, Unitree, headless
(`gui:=false`), logged from the moment joint states appear, 3 repeats.

| Run | Window | Real-time factor | Moved after 30 s | Turned | Movement 20-30 s | Verdict |
|---|---|---|---|---|---|---|
| Headless 1 | no | 0.995 | 0.01 m | −0.5° | 0.00 | settled |
| Headless 2 | no | 0.997 | 0.03 m | −0.1° | 0.00 | settled |
| Headless 3 | no | 0.994 | 0.04 m | +0.4° | 0.00 | settled |
| Recorded 1-3 (test 1) | yes | 0.900-0.907 | 0.00 m | ≤ 0.2° | 0.00 | settled |

**Result:** the window is not the cause. Headless runs settle within 1-4 cm (the small movement is in the first
seconds after spawn, which these logs include) and stay still. The window changes the real-time factor by ≈ 10%
(0.99 → 0.90), so the policy's rate in simulated time changes too, and it makes no difference.

The two earlier walk-offs were **not reproduced** (0 of 6 runs today). They ran in a long-lived container that had
been through many runs; it was recreated today. Their cause is unknown; if it shows up again, check for leftover
processes publishing on the robot's topics.

## Conclusion (Unitree's policy)

- The standing drift is caused by the **original Gazebo joint controller** (Luis's gains and/or the D-term kick).
  It is sustained, not a start-up transient.
- With the **corrected controller** (policy gains + D on joint velocity), the robot stands still, with or without
  the window: Q1 is answered for Unitree's policy.
- Not covered: Diego's `walk.onnx`; it goes into Q2 (it also misbehaves while walking).

## Files

| File | |
|---|---|
| [`standing_test.sh`](standing_test.sh) | Recorded zero-command runs: setups × policies × repeats (`POLICIES=unitree` for Unitree only) |
| [`standing_reference.py`](standing_reference.py) | MuJoCo reference run and video (same CSV format) |
| [`standing_summary.py`](standing_summary.py) | Distance/heading over time, movement per 10 s window, settled or still moving |
| [`headless_test.sh`](headless_test.sh) | Test 2: the same runs headless, with the real-time factor recorded |
| [`make_videos.sh`](make_videos.sh) | Side-by-side videos (uses [`gazebo/side_by_side.py`](../../gazebo/side_by_side.py)) |

```bash
POLICIES=unitree open_questions/q1_standing/standing_test.sh 3 30
.venv/bin/python open_questions/q1_standing/standing_reference.py
.venv/bin/python open_questions/q1_standing/standing_summary.py recordings/standing/*.csv
open_questions/q1_standing/make_videos.sh 3
open_questions/q1_standing/headless_test.sh 3 30
```
