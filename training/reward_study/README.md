# What does each reward term teach the robot?

A study of the 16 reward terms in Unitree's G1 walking task (`Unitree-G1-Flat`, mjlab 1.2.0, `rsl_rl` PPO): what each
one computes, and what the robot learns, or fails to learn, with and without it.

- Generated charts and the full evaluation table: [results.md](results.md) (rebuilt with
  [`training/study.py`](../study.py) after every round).
- Variants: [`training/variants.py`](../variants.py). Plan: [PLAN.md](../../PLAN.md), Track 1.
- Runs, checkpoints and videos: private Kaggle dataset `sakshaymahna/g1-walking-runs` and the outputs of the training
  notebook's versions (listed per run in [results.md](results.md)).

## Method

- **Add one in:** start from velocity tracking only, add terms in groups (ladder steps 0-8). Step 8 is the full
  reward set, i.e. the baseline. Shows what each group *builds*.
- **Leave one out:** the full reward set without one term. Shows what each term *protects against*.
- Everything else identical: same task, robot, observations, randomization, PPO settings; 4096 robots, **1000
  iterations**, seed 1 (a second seed where an effect is small or surprising).
- Each policy is checked three ways: its **training curves** (every term, every iteration), **videos in its training
  simulator** (MuJoCo Warp), and the **evaluation report** in plain MuJoCo ([`report.py`](../report.py): tracking, gait,
  effort, posture, robustness).
- A prediction is written down before each result.

## The 16 terms

$$v$$, $$\omega$$: base linear and angular velocity (pelvis frame), $$c$$: velocity command, $$g$$: gravity in the torso
frame, $$q$$: joint angles, $$a$$: policy action. "When moving" = only while the command is larger than 0.1 (0.01
for `foot_slip`). Weights as in Unitree's G1 config.

| Step | Term | Weight | Computes | Purpose |
|---|---|---|---|---|
| 0 | `track_linear_velocity` | +1.0 | $$\exp(-(\lVert c_{xy}-v_{xy}\rVert^2 + 2v_z^2)/0.25)$$ | follow the forward/sideways command, don't bounce |
| 0 | `track_angular_velocity` | +1.0 | $$\exp(-((c_z-\omega_z)^2 + 0.05\lVert\omega_{xy}\rVert^2)/0.5)$$ | follow the turn command, don't rock |
| 0 | `is_terminated` | −200 | 1 when the episode ends by a fall (torso tilted > 70°) | don't fall |
| 1 | `body_orientation_l2` | −1.0 | $$\lVert g_{xy}\rVert^2$$ of the torso | keep the torso upright |
| 2 | `pose` | +1.0 | $$\exp(-\text{mean}((q-q_0)^2/\sigma^2))$$, $$\sigma$$ per joint and per speed (0.05 rad standing) | stay near the default posture |
| 3 | `action_rate_l2` | −0.05 | $$\lVert a_t - a_{t-1}\rVert^2$$ | no jumpy actions |
| 3 | `joint_acc_l2` | −2.5e-7 | $$\lVert\ddot q\rVert^2$$ | no violent joint accelerations |
| 4 | `foot_gait` | +0.5 | share of feet whose contact matches the gait clock (stance when phase < 0.56, right foot half a period after the left), when moving | step on the clock's rhythm |
| 5 | `foot_clearance` | −1.0 | $$\sum_\text{feet} \lvert z_\text{foot} - 0.10\rvert \cdot \lVert v_{\text{foot},xy}\rVert$$, when moving | lift moving feet to 10 cm |
| 5 | `foot_slip` | −0.25 | $$\sum_\text{feet in contact} \lVert v_{\text{foot},xy}\rVert^2$$, when moving | don't skid |
| 5 | `soft_landing` | −0.001 | contact force at each touchdown, when moving | land softly |
| 6 | `stand_still` | −1.0 | $$\lVert q - q_0\rVert^2$$ when the command is ~0 | really stand still |
| 7 | `body_ang_vel` | −0.05 | $$\lVert\omega_{xy}\rVert^2$$ of the torso | no torso wobble |
| 7 | `angular_momentum` | −0.025 | $$\lVert L\rVert^2$$, whole-body angular momentum | no swinging |
| 8 | `joint_pos_limits` | −10 | how far joints go past their soft limits | stay inside the joint range |
| 8 | `self_collisions` | −1.0 | steps with a self-contact force over 10 N | legs and arms don't collide |

## Results

### Step 0: velocity tracking only (`ladder_0`)

**Prediction:** it learns to move in some way, probably ugly (lunging, shuffling), possibly unable to stand still.

**Training:** it learns to stay up fast. Episodes reach the 20 s limit by about iteration 70 (the full baseline needs
about 250: with 16 terms there is much more to satisfy at once). Forward tracking settles at 0.37 (the baseline
reaches 0.77 after 3000 iterations), and turning stays at about 0.2, then jumps to 0.45 between iterations 850 and 1000:
it was still learning when training stopped.

**Plain MuJoCo:** falls in every test, within about 2.5 s even when told to stand still, and in some tests drives
the simulation unstable (infinite accelerations).

### Step 1: + upright (`ladder_1`)

**Prediction:** the torso stays upright; otherwise like step 0.

**Training:** the tilt penalty is large at first (the robot leans while learning to stay up) and drops to almost zero
by iteration 200: it does learn to keep the torso upright. Forward tracking is unchanged (0.37), but **turning
collapses**: from 0.18 at iteration 150 it falls steadily to 0.09 by iteration 1000. Total reward 8.6 vs 17.1 for step 0.

**Plain MuJoCo:** falls in every test, like step 0.

### What steps 0-1 show so far

1. **Tracking alone is not enough for a usable policy.** Both policies keep the robot up in their training simulator
   but collapse within seconds in plain MuJoCo, where the full reward set works (and transfers to Gazebo). The likely
   reason: without smoothness and posture terms, the policy finds jerky, high-frequency control that balances only in
   the exact simulator it was trained in. The videos in the training simulator (round 3) will show how they move. Step 2 (pose) fails the same way.
2. **Terms compete.** Adding the upright penalty made the policy give up turning: turning probably needs some torso
   tilt or rocking, and the cheapest way to avoid the penalty was to turn less.
3. **Fewer terms, faster learning to survive.** Steps 0-1 stay up three times sooner than the full set; the full set
   pays for its extra goals early (every penalty gets worse during iterations 0-250 while the robot learns to stay up,
   then improves).

**To check:** a second seed for step 1 (is the turning collapse real?), and the 1000-iteration budget (step 0 was still
changing at 1000).

### The reference: full reward set at 1000 iterations (`baseline`, round 2)

**Training:** episodes reach the 20 s limit by about iteration 250-300 (later than steps 0-2: it has 16 goals at
once). Mean reward 32.3 at iteration 1000 (39.2 after 3000).

**Plain MuJoCo:** walks properly. Stands still (0.00 m in 12 s), tracks 0.2 / 0.5 / 1.0 m/s as 0.24 / 0.49 / 0.96,
sideways 0.20, turning 0.44 rad/s; survives all four pushes; walks on friction 0.3 and 1.2. Compared with the same
setup after 3000 iterations: the same behaviour, slightly *better* at slow speed (0.24 vs 0.18 at 0.2 m/s), a little
lower feet (3.5 vs 4.8 cm clearance). Like every trained run so far (but not Unitree's released policy), it falls
standing still without armature.

**So 1000 iterations is enough for the full reward set**, and this run is the reference every ladder step is compared
with.

### Step 2: + pose (`ladder_2`)

**Prediction:** the robot keeps a posture near the default (knees slightly bent, arms down), and maybe transfers
better to plain MuJoCo because a familiar posture is a stable one.

**Training:** the pose reward rises to 0.09 by iteration 110, while the robot learns to stay up, then **falls back to
zero** (0.0004 at iteration 500, 0.0000 at 1000). The policy abandons the default posture for one that tracks better.
Tracking is only slightly better than step 1 (forward 0.40, turning 0.12). Total reward 9.8.

**Plain MuJoCo:** falls in every test, like steps 0-1.

**Why the pose reward stops working:** it is $$\exp(-\text{mean}((q-q_0)^2/\sigma^2))$$ with tight tolerances (0.05 rad
when standing, 0.1-0.5 rad when walking). Close to the default pose it pays well and pulls the joints back; far from it,
the reward and its gradient are both practically zero, so there is nothing left to pull with. Once the policy drifted
away (while learning to stay up and to track), the pose term could no longer bring it back. In the full reward set the
same term rises steadily to 0.71: there, other terms keep the posture close enough for it to keep working, in particular
`stand_still`, a squared penalty that keeps growing the further the joints are from the default, and the smoothness and
gait terms.

**Lesson:** an exponential ("Gaussian") reward only guides a policy that is already near its target. It needs either a
wider tolerance, or another term that brings the policy into its range.

### Steps 3-8, leave-one-out, observations

Pending (rounds 3 onwards).

## Log

| Round | Notebook version | Runs | Status |
|---|---|---|---|
| 1 | 5 | `ladder_0`, `ladder_1` | done |
| 2 | 6 | `baseline` (1000 it), `ladder_2` | done (videos failed: fixed for round 3) |
| 3 | 7 | `ladder_3`, `ladder_4`, + training-simulator videos of `ladder_0`-`ladder_2` and `baseline` | running |
