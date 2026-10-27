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

## Summary: what each term does (so far)

| Term | What the robot learns from it | Evidence |
|---|---|---|
| tracking (linear, angular) | to follow commands, but they also pay ~half the reward for not moving at all; alone, the robot **sits down** | ladder 0-2, videos |
| `is_terminated` (fall = tilt > 70°) | not to tip over; height is not checked, so sitting counts as "not fallen" (and wins whenever walking is hard to find: ladder 0-2, 2 of 3 `dr_armature` runs) | ladder 0-2, rounds 12-13 |
| `body_orientation_l2` | to sit up instead of lying (ladder); little left to do in the full set | ladder 1, leave-one-out |
| `pose` | to keep the **upper body still**: without it the arms swing 10-25× more and power doubles; useless while sitting (exponential reward, out of range) | ladder 2, leave-one-out |
| `action_rate_l2` | **to stand up** (sitting needs big action changes) and to move smoothly; without it actions are 2× jumpier and **turning breaks** | ladder 2+, leave-one-out |
| `joint_acc_l2` | little on top of `action_rate_l2` | leave-one-out |
| `foot_gait` | **to step**: alternating steps on the clock; without it a fast shuffle and the **dead zone** (no motion at slow commands) | ladder 4, leave-one-out |
| gait clock input | nothing by itself: removing it on top of `foot_gait` changes nothing; it is what `foot_gait` is written in | `obs_no_clock` |
| base velocity input | nothing measurable in plain MuJoCo (no fragility, no dead zone) | `obs_lin_vel` |
| `foot_clearance` | about **1 cm** more foot lift | ladder 5 and leave-one-out agree |
| `foot_slip`, `soft_landing` | small: less slip (within seed noise) | ladder 5, leave-one-out |
| `stand_still` | nothing visible in these tests (`pose` already holds the robot still) | ladder 6, leave-one-out |
| `body_ang_vel`, `angular_momentum` | a calmer, cheaper walk (small) | ladder 7 |
| `joint_pos_limits`, `self_collisions` | not tested separately (rarely active) | – |

**Build order matters:** steps 0-4 decide *whether* the robot walks (sit → stand → shuffle → step); the remaining
terms polish *how*. **Open thread for Gazebo (Q2):** every policy trained without `foot_gait` stands still without
armature (4 of 4); every policy trained with it falls (12 of 12) unless the armature is randomized in training
(`dr_armature` stands, but only 1 of 3 runs learned to walk at all; the others sat down). With a height fall check
added (`dr_armature_height`), 2 of 2 seeds stand without armature, though slow walkers after 1000 iterations. So it
is the training conditions, not the gait reward alone.

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

![Steps 0-4 and the full set in their training simulator](images/training_sim_steps0-4.jpg)

*Each policy in its own training simulator (MuJoCo Warp), standing for 4 s, then told to walk at 0.5 m/s.*

| Policy (1000 it) | Pelvis height in training sim | Speed at 0.5 m/s command (training sim) | Plain MuJoCo |
|---|---|---|---|
| step 0: tracking only | 0.11 m (lying on its back) | 0.00 m/s | "falls" (pelvis below 0.4 m) |
| step 1: + upright | 0.14 m (sitting, propped on its arms) | 0.07 m/s | "falls" |
| step 2: + pose | 0.14 m (sitting upright) | 0.03 m/s | "falls" |
| step 3: + smoothness | 0.79 m | 0.46 m/s | walks (shuffling) |
| step 4: + foot gait | 0.78 m | 0.53 m/s | walks |
| full set (baseline) | 0.78 m | 0.48 m/s | walks |

### Steps 0-2: the robot learns to sit down (`ladder_0`, `ladder_1`, `ladder_2`)

**Predictions:** step 0 moves in some ugly way; step 1 keeps the torso upright; step 2 keeps a posture near the default.

**What they learned instead: to sit or lie on the floor and do (almost) nothing.** Step 0 lies on its back with its
legs in the air, step 1 sits propped on its arms, step 2 sits upright. None of them ever walks (0.00-0.07 m/s when
told to walk at 0.5 m/s). During training their episodes almost always reach the 20 s limit, as if they never fall.

**Why this is the best strategy the rewards allow:**

1. **The fall check ignores height.** An episode ends (with the −200 `is_terminated` penalty) only when the pelvis
   tilts more than 70°. Sitting or lying with the pelvis fairly upright is never a "fall".
2. **The tracking rewards pay for doing nothing.** They are $$\exp(-\text{error}^2/\sigma^2)$$: a robot that never moves
   still gets $$\exp(-c^2/\sigma^2)$$. Averaged over the training commands (forward −0.5 to 1.0 m/s, sideways ±0.5,
   turning ±1 rad/s, 5 % standing), a robot that never moves collects **43 % of the maximum linear-tracking reward
   and 62 % of the angular one**, at every step.
3. So sitting down is safe (no fall penalty, ever) and pays almost half of what walking would. Learning to walk
   needs a long phase of falling (−200 each time) before it pays more. With only the tracking rewards, PPO settles on
   sitting: it learns to "survive" by iteration ~70, three times sooner than the full reward set needs to stand.

**What the extra terms changed:** the upright term (step 1) made the robot sit up instead of lying on its back, and
it reduced turning (the turning reward fell from 0.18 to 0.09: turning while sitting means rocking, which the upright
term penalizes). The pose term (step 2) paid briefly while the robot was still upright (0.09 at iteration 110) and
then nothing once it sat down: see "Why the pose reward stops working" below.

**Correction:** after round 1, from the plain-MuJoCo results alone, I guessed these policies balanced with jerky control
that only works in their own simulator. The videos show they never balanced at all: they sat down. Plain MuJoCo
reported "falls" because my evaluation counts a pelvis below 0.4 m as a fall.

**Why the pose reward stops working (step 2):** it is $$\exp(-\text{mean}((q-q_0)^2/\sigma^2))$$ with tight
tolerances (0.05 rad when standing, 0.1-0.5 rad when walking). Close to the default pose it pays and pulls the joints
back; far from it (sitting), the reward and its gradient are practically zero. In the full set the same term rises
steadily to 0.71, because the robot stays near the default pose there.

**Lesson:** a reward is only as good as its loopholes. "Don't fall" defined as "don't tilt" allows sitting, and an
exponential tracking reward pays for standing still. Exponential ("Gaussian") rewards also only guide a policy that is
already close to their target.

### Step 3: + smoothness (`ladder_3`: `action_rate_l2`, `joint_acc_l2`)

**Prediction:** less jitter; the sitting strategy probably stays.

**What it learned: to stand up and walk**, the first policy that does. In plain MuJoCo it stands still (0.01 m),
walks at 0.49 m/s (command 0.5) and 0.90 (command 1.0), steps sideways (0.24) and turns (0.44 rad/s), survives all
four pushes, and walks on friction 0.3 and 1.2. Training rewards: tracking 0.77 / 0.67, pose 0.80 (higher than the
full set: it stays closest to the default pose).

**But it shuffles.** 6.1 steps per second with 0.09 s in the air per step, feet lifted only 1.4 cm, sliding 16 cm/s
while on the ground, both feet down 39 % of the time (baseline: 3.3 steps/s, 0.25 s, 3.5 cm, 5 cm/s, 17 %). It also
has **a dead zone**: at 0.2 m/s it barely moves (0.07 m/s), like `walk.onnx` in Week 2, which was also trained without
a gait reward. And it is the only trained policy so far that stands still **without armature** (0.03 m).

**Why smoothness turned sitting into standing** is not clear from one run; the most likely reading is that sitting and
lying policies rely on large, fast joint motions (getting down, rocking, propping), which these terms make expensive,
so standing became the cheaper way to keep the tracking reward. To check: a second seed, and the leave-one-out runs.

### Step 4: + foot gait (`ladder_4`)

**Prediction:** proper alternating steps on the gait clock's rhythm.

**What it learned:** exactly that. 3.38 steps per second (the gait clock's 0.6 s period: one step per foot per
period), 0.25 s in the air, 2.5 cm clearance, 5.5 cm/s slip, 16 % double support. **The dead zone disappears**
(0.24 m/s at a 0.2 command): the gait reward makes the policy step even when the command is small. This is the Week 2
gait-clock result seen from the reward side. Its walking is already close to the full set; it loses the ability to
stand without armature.

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

### Steps 5 and 6: + feet, + stand still (`ladder_5`, `ladder_6`)

**Predictions:** step 5 (`foot_clearance`, `foot_slip`, `soft_landing`) lifts the feet higher and stops skidding;
step 6 (`stand_still`) makes standing quieter.

**What changed** (plain MuJoCo, compared with step 4):

| | Step 4 | Step 5: + feet | Step 6: + stand still | Full set |
|---|---|---|---|---|
| foot clearance | 2.5 cm | **3.4 cm** | 2.9 cm | 3.5 cm |
| foot slip in stance | 5.5 cm/s | **4.9 cm/s** | 4.5 cm/s | 5.0 cm/s |
| turning (command 0.5 rad/s) | 0.45 | **0.52** | 0.52 | 0.44 |
| slow walking (command 0.2 m/s) | 0.24 | 0.17 | 0.19 | 0.24 |
| standing still, 12 s | 0.00 m | 0.00 m | 0.02 m | 0.00 m |

Step 5 does what its terms ask, a little: feet lift ~1 cm higher and slip a little less (the clearance target is
10 cm, but the penalty is weighted by foot speed, so it only pushes swinging feet up). Step 6 shows no effect in these
tests: standing was already still without `stand_still`.

**These effects are small**, about the size of the differences between two seeds of the same setup (the two 3000-
iteration baselines differ by 0.1 cm in clearance, 0.3 cm/s in slip, 0.02-0.04 m/s in tracking). One seed per step
cannot separate them from chance. **So the order matters: steps 0-4 build the gait (stand up, then step), steps 5-8
fine-tune it.** For the fine-tuning terms, the leave-one-out runs (full set minus one term) are the better test.

### Step 7: + torso wobble (`ladder_7`: `body_ang_vel`, `angular_momentum`)

**Prediction:** less rocking of the torso.

**What changed** (plain MuJoCo, vs step 6): torso tilt 2.3° → 1.4°, action rate 0.48 → 0.41, mechanical power
80 → 69 W; tracking and gait unchanged. A calmer, cheaper walk; small effects like steps 5-6.

Step 8 adds `joint_pos_limits` and `self_collisions`; with them the reward set is complete, so step 8 is the
reference run (`baseline`, 1000 iterations).

### Leave one out: without `foot_gait` (`loo_foot_gait`)

**Prediction:** the gait falls apart into the step-3 shuffle, and the dead zone comes back, even though the foot terms
(clearance, slip, soft landing) are still there.

**What happened:** exactly that.

| Plain MuJoCo | Step 3 (no gait term) | **Full set − `foot_gait`** | Full set |
|---|---|---|---|
| steps per second | 6.1 | **4.9** | 3.3 |
| time in the air per step | 0.09 s | **0.11 s** | 0.25 s |
| foot clearance | 1.4 cm | **1.1 cm** | 3.5 cm |
| foot slip in stance | 16 cm/s | **12 cm/s** | 5 cm/s |
| both feet down | 39 % | **39 %** | 17 % |
| forward 0.2 / 0.5 / 1.0 m/s | 0.07 / 0.49 / 0.90 | **0.00 / 0.38 / 0.64** | 0.24 / 0.49 / 0.96 |
| sideways 0.3 m/s | 0.24 | **0.00** | 0.20 |
| standing without armature | stands (0.03 m) | **stands (0.11 m)** | falls |

In its training simulator it also walks slower (0.39 m/s at a 0.5 command). The foot terms alone don't create
steps: they only refine a gait that `foot_gait` creates. **`foot_gait` is what makes the robot step**, and without it
the dead zone returns in full (no motion at all at 0.2 m/s forward or 0.3 m/s sideways). This is the strongest link so
far to `walk.onnx`'s dead zone (Week 2): it was trained without a gait clock or a gait reward.

**A pattern to check:** the only trained policies that stand still without armature (as in Gazebo) are the two
without `foot_gait` (step 3 and this one). Two cases are not enough to call it a cause.

### Leave one out: without `pose` (`loo_pose`)

**Prediction:** the posture drifts (it did in step 2, where the pose term was "dead" while sitting), but with the
other terms present the robot still walks.

**What happened:** it walks and tracks normally, but in an unusual and expensive way. **The arms swing hard**:

| Mean distance from the default pose while walking (rad) | legs | waist | shoulders | elbows | wrists |
|---|---|---|---|---|---|
| full set | 0.07 | 0.02 | 0.03 | 0.01 | 0.01 |
| full set − `pose` | 0.12 | 0.05 | **0.29** | **0.25** | **0.20** |

With the arms swinging, mechanical power doubles (144 vs 72 W), and actions are jumpier (action rate 0.77 vs
0.43). The feet also lift twice as high (7.1 vs 3.5 cm, about the same as Unitree's released policy). Tracking,
pushes and friction are fine.

Nothing else in the reward set says anything about the arms. Walking doesn't need them, so without `pose` they are
free, and the policy uses them (likely for balance and for momentum when turning). **The pose term is mainly what keeps
the upper body still**, and with it the energy use low. (In the add-one-in ladder, step 2, the same term did nothing:
the robot sat down, out of the pose reward's narrow range. Its job only exists once the robot walks.)

### Leave one out: without `stand_still` (`loo_stand_still`)

**Prediction:** more fidgeting or drift when standing.

**What happened:** no measurable difference from the full set. It stands still (0.01 m in 12 s), walks, steps and
recovers from pushes like the full set; actions are even a little smoother (0.34 vs 0.43). Like step 6 of the ladder,
**`stand_still` has no visible job in these tests**: `pose` (with its tight 0.05 rad standing tolerance) already
holds the robot still at zero command. It might matter for disturbances while standing, which these tests don't
include.

### Leave one out: the two smoothness terms (`loo_action_rate_l2`, `loo_joint_acc_l2`)

Ladder step 3 added both at once, and the robot stood up. Which one matters?

| | Full set | − `action_rate_l2` | − `joint_acc_l2` |
|---|---|---|---|
| action rate (plain MuJoCo) | 0.43 | **0.95** | 0.41 |
| joint-acceleration penalty in training | −0.07 | **−0.34** (5×) | (not in the reward) |
| turning, command 0.5 rad/s | 0.44 | **0.15** | 0.50 |
| turning reward in training | 0.57 | **0.23** | 0.58 |
| forward 0.5 m/s | 0.49 | 0.42 | 0.46 |
| mechanical power | 72 W | 76 W | 60 W |
| total training reward | 32.3 | **22.4** | 33.3 |

**Without `action_rate_l2`** the policy is jittery: its actions change twice as much from step to step, and during
training its joints accelerate so violently that the joint-acceleration penalty grows five-fold. **Turning breaks**
(0.15 rad/s for a 0.5 command, in training and in plain MuJoCo), while walking straight still works and it survives
pushes. Without `joint_acc_l2`, nothing gets worse; the walk is even a little cheaper.

**So `action_rate_l2` does the smoothing work**, and `joint_acc_l2` adds little on top of it (its weight, 2.5e-7, is
tiny next to the accelerations involved, and smooth actions already mean smooth joints). It is also the more likely of
the two to be what made the robot stand up at ladder step 3, but that is a reading, not a result: step 3 added both
terms, and a ladder step with only one of them would be needed to show it.

### Leave one out: the foot terms (`loo_foot_clearance`, `loo_foot_slip`)

| Plain MuJoCo | Full set | − `foot_clearance` | − `foot_slip` |
|---|---|---|---|
| foot clearance | 3.5 cm | **2.6 cm** | 2.4 cm |
| foot slip in stance | 5.0 cm/s | 3.5 cm/s | **5.8 cm/s** |
| both feet down | 17 % | 13 % | 24 % |
| mechanical power | 72 W | 68 W | 55 W |
| tracking, pushes, friction | fine | fine | fine |

- **`foot_clearance` buys about 1 cm of foot lift, measured both ways:** removing it from the full set lowers the
  feet by 0.9 cm, and adding it in ladder step 5 raised them by 0.9 cm. Two independent comparisons agree, so this
  one is a real effect, not seed noise.
- **`foot_slip`:** removing it makes the feet slip a little more (+0.8 cm/s) and keeps both feet down longer (24 %
  vs 17 %): the expected direction, but within the spread between seeds.
- Neither changes tracking or robustness. Without either, the walk is even a little cheaper: lifting the feet and
  avoiding slip both cost energy.

### Which smoothness term makes the robot stand up? (`ladder_2_plus_action_rate_l2`)

Ladder step 2 (tracking, upright, pose: the robot sits) plus **only** `action_rate_l2`:

| | Step 2 | Step 2 + `action_rate_l2` | Step 3 (both smoothness terms) |
|---|---|---|---|
| pelvis in its training simulator | 0.14 m (sits) | **0.78 m (stands)** | 0.79 m |
| steps per second / both feet down | – | 6.8 / 50 % | 6.1 / 39 % |
| foot slip in stance | – | 21 cm/s | 16 cm/s |
| standing without armature | – | stands (0.22 m) | stands (0.03 m) |

**`action_rate_l2` alone turns sitting into standing.** It is the term that makes "sit down and collect the tracking
reward for free" a bad deal: getting down, rocking and propping all need large changes in the action from one step to
the next. It produces an even more frantic shuffle than step 3, which also has `joint_acc_l2`.

### Leave one out: without `body_orientation_l2` (`loo_body_orientation_l2`)

Walks normally: torso tilt 1.5° (full set 1.2°), forward tracking a little lower at 0.5 m/s (0.41 vs 0.49), turning,
pushes and friction fine. **In the full set the upright term has little left to do**: `pose` and `foot_gait` already
keep the torso upright. Its job showed up in the ladder (step 1: sit up instead of lying on the back).

### Observations and randomization

Two changes to what the policy *sees*, with the full reward set otherwise (round 10):

- `obs_no_clock`: no gait clock input (96 inputs instead of 98). `foot_gait` is defined by the clock, so it goes too.
  This is `loo_foot_gait` minus the clock input.
- `obs_lin_vel`: the base linear velocity as an extra input (101 inputs), as in `walk.onnx`, with the same noise the
  critic gets (±0.5 m/s).

| Plain MuJoCo | Full set | − `foot_gait` | **no clock** | **+ base velocity** |
|---|---|---|---|---|
| forward 0.2 / 0.5 / 1.0 m/s | 0.24 / 0.49 / 0.96 | 0.00 / 0.38 / 0.64 | **0.08 / 0.42 / 0.81** | **0.20 / 0.46 / 0.99** |
| sideways 0.3 m/s | 0.20 | 0.00 | **0.05** | **0.24** |
| turning 0.5 rad/s | 0.44 | 0.40 | **0.49** | **0.47** |
| steps per second / both feet down | 3.3 / 17 % | 4.9 / 39 % | **4.9 / 28 %** | **3.4 / 16 %** |
| foot clearance | 3.5 cm | 1.1 cm | **1.2 cm** | **2.7 cm** |
| mechanical power | 72 W | 56 W | **62 W** | **62 W** |
| pushes, friction 0.3 / 1.2 | fine | fine | **fine** | **fine** |
| standing without armature | falls | stands (0.11 m) | **stands (0.21 m)** | **falls** |

- **No clock:** the same shuffle and (nearly) the same dead zone as `loo_foot_gait`. Taking the clock *input* away
  adds nothing to taking `foot_gait` away: an input the rewards don't use teaches nothing. **The stepping comes from
  the gait reward, and the clock is only what that reward is written in.** The no-clock policy reaches higher speeds
  than `loo_foot_gait` (0.81 vs 0.64 m/s at 1.0), but with one seed each that is within the spread we have seen.
- **Base velocity input:** walks like the full set, with slightly better tracking at 1.0 m/s and sideways, and
  survives pushes and both frictions. **In plain MuJoCo the extra input doesn't make the policy fragile.** It doesn't
  explain `walk.onnx`'s dead zone either: this policy has none. That fits the `foot_gait` result: `walk.onnx` was trained
  without a gait clock or gait reward, and its dead zone looks like the no-clock one. (Whether the velocity input
  explains `walk.onnx` getting worse with Gazebo's velocity damping is a separate question, not tested here.)
- **The no-armature pattern holds:** without `foot_gait` the policy stands without armature (now 4 of 4); with it,
  it falls (now 11 of 11). Round 11 (`dr_armature`) tests the training-side fix.

**Armature randomization (`dr_armature`, round 11).** Full reward set; at startup each of the 4096 training robots
gets every joint's armature scaled by a random factor between 0 (none, as in Gazebo) and 1 (Unitree's value).

| Plain MuJoCo | Full set, seed 1 | Full set, seed 2 | **armature randomized** |
|---|---|---|---|
| forward 0.2 / 0.5 / 1.0 m/s | 0.24 / 0.49 / 0.96 | 0.20 / 0.46 / 1.02 | **0.11 / 0.37 / 0.76** |
| sideways 0.3 m/s / turning 0.5 rad/s | 0.20 / 0.44 | 0.18 / 0.46 | **0.18 / 0.18** |
| standing (with armature) | 0.00 m | 0.00 m | **0.06 m** |
| pushes survived | 4 of 4 | 4 of 4 | **3 of 4** |
| mechanical power | 72 W | 74 W | **51 W** |
| **standing without armature** | falls | falls | **stands (0.07 m)** |
| walking without armature, 0.5 m/s | 0.49 | 0.54 | **0.42** |

- **It is the first policy with `foot_gait` that stands without armature.** So the falls were not caused by the gait
  reward itself: a gait-trained policy can learn to stand on joints without armature if it meets them in training.
- **It is behind on everything else**, turning most of all (0.18 rad/s, like the policy without `action_rate_l2`). The
  training curves show why: with randomized joints the task is harder and learning runs about 300 iterations behind
  the baseline (episodes reach full length near iteration 700 instead of 400), and its reward was still climbing
  steeply at iteration 999. One seed at 1000 iterations is not a verdict on what it can reach. **Round 12** trains it
  for 2000 iterations with two seeds.

**Round 12: the 2000-iteration runs sat down.** Both seeds (1 and 2) go down within 5 s in their training simulator
(pelvis at 0.11-0.15 m for the rest of the video) and in every plain-MuJoCo test. The training logs show why:

| Seed 1, iteration | 100 | 300 | 400 | 500 | 700 | 1000 |
|---|---|---|---|---|---|---|
| episode length, round 11 run | 13 | 103 | 230 | 351 | 943 | 989 |
| episode length, round 12 run | 15 | 116 | **663** | 884 | 917 | 924 |
| `pose` reward, round 11 run | 0.00 | 0.05 | 0.11 | 0.16 | **0.47** | **0.57** |
| `pose` reward, round 12 run | 0.00 | 0.05 | 0.09 | 0.12 | 0.12 | 0.09 |

The two seed-1 runs have the same config (only `max_iterations` differs, which changes nothing before the end) but
GPU arithmetic is not bit-for-bit repeatable. Around iteration 300-400 they split. The round 11 run slowly learned
to walk; the round 12 run found that **sitting** stops the episodes ending (episode length jumps to 660 by iteration
400) and never left: `pose` stays near zero (its reward is out of range when sitting), and linear tracking stays at
0.37 (the "free" tracking reward for not moving). It is the ladder 0-2 exploit again, now **with the full reward set**:
randomizing the armature makes walking harder to find, and the fall check (tilt only) still allows sitting.

So `dr_armature` worked in **1 of 3 runs**. **Round 13** (`dr_armature_height`) closes the loophole: a robot whose
pelvis drops below 0.5 m has fallen (it stands at 0.78 m, sits at 0.11-0.15 m).

**Round 13: with the height check, neither seed sits.** In both, `pose` climbs normally (0.68 and 0.64 at iteration
1000) and the episodes reach full length (997 and 994 steps).

| Plain MuJoCo | Full set (seed 1) | **armature randomized + height check, seed 1** | **seed 2** |
|---|---|---|---|
| forward 0.2 / 0.5 / 1.0 m/s | 0.24 / 0.49 / 0.96 | **0.07 / 0.10 / 0.13** | **0.18 / 0.33 / 0.37** |
| sideways 0.3 m/s / turning 0.5 rad/s | 0.20 / 0.44 | 0.11 / 0.29 | 0.19 / 0.21 |
| pushes survived | 4 of 4 | 4 of 4 | 4 of 4 |
| mechanical power | 72 W | 19 W | 42 W |
| **standing without armature** | falls | **stands (0.00 m)** | **stands (0.04 m)** |

- **Both seeds stand without armature** and survive every push: with sitting ruled out, armature randomization
  reliably produces policies that stand on joints without armature (2 of 2).
- **Both are under-trained.** Learning runs well behind the baseline (episodes reach full length only after iteration 700,
  baseline near 400), and the reward was still rising steeply at iteration 1000 (seed 1: −8.7 at 700, +14.0 at 1000).
  Seed 1 barely walks yet (0.10 m/s at a 0.5 command, 46 % double support); seed 2 is further along.
- **Round 14** trains the same variant for 2000 iterations. Without the sitting loophole, longer training should only
  help, which is what round 12 could not test.
- **Baseline seed 2** (1000 iterations) agrees with seed 1 to within 0.06 m/s on tracking and 0.1 cm on foot clearance, and
  also falls standing without armature. That gives a rough size for seed noise in this study.

## Log

| Round | Notebook version | Runs | Status |
|---|---|---|---|
| 1 | 5 | `ladder_0`, `ladder_1` | done |
| 2 | 6 | `baseline` (1000 it), `ladder_2` | done (videos failed: fixed for round 3) |
| 3 | 7 | `ladder_3`, `ladder_4`, + training-simulator videos of `ladder_0`-`ladder_2` and `baseline` | done (6 videos took 80 min: now every 2nd frame, both GPUs) |
| 4 | 8 | `ladder_5`, `ladder_6` | done (`ladder_6` video failed: EGL device fix) |
| 5 | 9 | `ladder_7`, `loo_foot_gait`, + video of `ladder_6` | done |
| 6 | 10 | `loo_stand_still`, `loo_pose` | done |
| 7 | 11 | `loo_action_rate_l2`, `loo_joint_acc_l2` | done |
| 8 | 12 | `loo_foot_slip`, `loo_foot_clearance` | done |
| 9 | 13 | `loo_body_orientation_l2`, `ladder_2_plus_action_rate_l2` | done |
| 10 | 14 | `obs_no_clock`, `obs_lin_vel` | done (videos failed: the filming environment now applies the run's variant) |
| 11 | 15 | `dr_armature`, `baseline` seed 2, + videos of `obs_no_clock`, `obs_lin_vel` | done |
| 12 | 16 | `dr_armature` seeds 1 and 2, **2000 iterations** | done (both sat down) |
| 13 | 17 | `dr_armature_height` seeds 1 and 2 | done |
| 14 | 18 | `dr_armature_height` seeds 1 and 2, **2000 iterations** | running |
