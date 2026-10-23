# Q3: does it matter whether the policy runs on the wall clock or the simulation clock?

**Status: done.** With armature, no: both clocks give the same result in every run. Without armature, walking is
unaffected too; only **standing** without armature depends on the clock, and it is fragile on both (fails 3 of 3 on
the sim clock, 2 of 3 to 1 of 3 on the wall clock). The timing itself is fine on both clocks once running; both
have an irregular start-up.

## The question

The policy node runs a 50 Hz timer (one policy step every 20 ms). Two choices (`clock:=` in `policy.launch.py`):

- **wall**: every 20 ms of real time (Diego's default). If Gazebo runs slower than real time, the robot gets
  steps more often in simulated time.
- **sim**: every 20 ms of simulated time, from Gazebo's `/clock` (what training assumes).

Earlier hints: Unitree's policy with the corrected controller and no armature stood still on the wall clock (6 of
6 in Q1) but walked off on the sim clock (2 of 4 in Q2).

## Test

[`clock_repeats.sh`](clock_repeats.sh): both policies × corrected controller with and without armature × both
clocks × standing (30 s, zero command) and walking (stand 4 s, then vx = 0.5), 3 repeats each: 48 headless runs,
RTF 0.96-1.00. The logger records the wall and sim time of **every policy step** (each command the node publishes;
`STEP_LOG=1`, no change to the node). [`analyze.py`](analyze.py) summarizes outcome and timing.

## Results

**Outcome** (standing: distance moved in 30 s; walking: vx / vy in m/s, command 0.5 / 0):

| Policy, setup | Clock | Standing, 3 runs | Walking, 3 runs |
|---|---|---|---|
| Unitree, corrected + armature | wall | 0.00, 0.02, 0.02 m | 0.50 / −0.01 (all 3) |
| Unitree, corrected + armature | sim | 0.02, 0.02, 0.02 m | 0.49-0.50 / −0.01 |
| walk.onnx, corrected + armature | wall | 0.03, 0.04, 0.04 m | 0.55 / −0.02 (all 3) |
| walk.onnx, corrected + armature | sim | 0.06, 0.02, 0.01 m | 0.55 / −0.02 (all 3) |
| Unitree, corrected (no armature) | wall | **0.75, 1.36**, 0.03 m | 0.51 / −0.02 (all 3) |
| Unitree, corrected (no armature) | sim | **9.40, 9.97, 9.45 m** | 0.51 / −0.02 (all 3) |
| walk.onnx, corrected (no armature) | wall | **2.80**, 0.13, 0.16 m | 0.73 / −0.24 (all 3) |
| walk.onnx, corrected (no armature) | sim | **8.34, 2.91, 5.11 m** | 0.71-0.72 / −0.24 |

**Timing** (in simulated time, which is what the robot experiences):

| | Wall clock | Sim clock |
|---|---|---|
| Average period after start-up | 19.91 ms (19.57-19.99 per run) | 20.00 ms exactly |
| Steps more than 1 ms off 20 ms, after start-up | 3-18 % (jitter of 1-2 ms, longest gap 22 ms) | under 1 % |
| First 1.5 s after the first step | irregular: bursts of steps 0-1 ms apart, and pauses of up to 110 ms (once 230 ms) | the same kind: bursts 0-1 ms apart, pauses up to 110 ms (twice ≈ 400 ms) |

## What it shows

- **With armature, the clock doesn't matter**: 24 of 24 runs stand still and walk the same on both clocks.
- **Walking doesn't depend on the clock** even without armature: the numbers are identical on both clocks, good
  (Unitree) or bad (walk.onnx) alike.
- **Standing without armature fails on both clocks**, more often and further on the sim clock. The earlier
  "Unitree stands still on the wall clock" (9 of 9 runs, Q1 and Q2) did not hold up: here it moved 0.75 m and
  1.36 m in 2 of 3 runs. So the wall clock is not a fix; it changes the odds of a failure that the missing armature
  causes.
- **The step timing does not explain the difference**: once running, the sim clock is the more regular of the two
  (exactly 20 ms), yet it fails more. Both clocks have an irregular start-up while Gazebo spawns the robot: on the
  sim clock the timer catches up on missed periods (bursts of steps 1 ms apart), on the wall clock Gazebo stalls
  while real time runs on. One difference we did not measure: the wall-clock node starts stepping 2 s before
  Gazebo exists (it is launched first), so it reaches the spawn with a different internal state (last action) than
  the sim-clock node, which only starts when `/clock` does. A start-up difference fits a failure that happens in
  the first seconds while standing.

**Practical answer**: use the sim clock (it matches training: exactly 20 ms of simulated time per step) together
with the corrected controller and armature. Then the clock is irrelevant.

## Files

| File | What it does |
|---|---|
| [`clock_repeats.sh`](clock_repeats.sh) | The 48 runs, with policy-step timing → `recordings/q3/` |
| [`analyze.py`](analyze.py) | Outcome and timing per setup and clock |

```bash
open_questions/q3_clock/clock_repeats.sh 3                                # ~90 min
.venv/bin/python open_questions/q3_clock/analyze.py recordings/q3/*_[0-9].csv
```
