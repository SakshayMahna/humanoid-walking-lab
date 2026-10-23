# Q6: does Gazebo double Unitree's left drift?

**Status: done.** No. The comparison mixed two physics step sizes. At the same 1 ms step, MuJoCo drifts as much
as Gazebo.

## The question

Walking straight (command vx = 0.5), Unitree's policy turns slowly left in both simulators. The numbers we had:
MuJoCo +0.72 °/s, Gazebo with the corrected controller +1.0 °/s (and up to +1.8 °/s in some older runs). So it
looked like Gazebo adds about as much drift again.

## Test: MuJoCo's own turn rate, by physics step and with tiny start-up changes

[`mujoco_drift_spread.py`](mujoco_drift_spread.py): Unitree's MuJoCo model with its own position actuators (the
training setup), policy every 20 ms, physics step 5 / 2.5 / 2 / 1 ms. At each step size, one plain run and three
runs with tiny random joint velocities at the start (std 0.001 rad/s), to see whether the heading is sensitive to
small differences.

| MuJoCo physics step | Turn (°/s) | Spread over start-up changes | Wander (°) |
|---|---|---|---|
| 5 ms (training) | +0.72 | ± 0.00 | 0.38 |
| 2.5 ms | +0.99 | ± 0.00 | 0.38 |
| 2 ms | +1.04 | ± 0.00 | 0.39 |
| **1 ms (Gazebo's step)** | **+1.03** | ± 0.00 | 0.40 |

Gazebo (1 ms step), per run:

| Gazebo setup | Turn (°/s), each run |
|---|---|
| corrected controller, wall clock (`recordings/gains/`) | +1.05, +0.97 |
| corrected controller, sim clock (Q2 Test 3) | +0.83, +0.87, +0.88 |
| corrected controller + armature, sim clock (Q2 Test 3) | +0.69, +0.71, +0.69 |

## What it shows

- **The drift is not chaotic**: tiny start-up changes move it by less than 0.01 °/s. So differences of 0.1-0.3 °/s
  between setups are real effects, not noise.
- **The physics step changes MuJoCo's own drift**: +0.72 °/s at the 5 ms training step, about +1.0 °/s at 2.5 ms
  and below. A smaller step is a slightly different simulation, and the policy reacts to it (which part of the
  physics causes the change was not investigated).
- **At the same 1 ms step, Gazebo matches MuJoCo**: +0.97 to +1.05 °/s with the corrected controller vs +1.03 in
  MuJoCo. The "doubling" was the 5 ms vs 1 ms difference, not Gazebo.
- With armature, Gazebo drifts a little less (+0.70 °/s). All values sit between MuJoCo's 5 ms and 1 ms numbers.

For scale: 0.3 °/s is about 5° over the 16 s measuring window. The policy's command is a turn rate, which it tracks
only approximately; a heading controller on top (turning the heading error into a turn-rate command, as in
training's heading environments) would remove the drift in a real deployment.

    .venv/bin/python open_questions/q6_drift/mujoco_drift_spread.py      (~15 s)
