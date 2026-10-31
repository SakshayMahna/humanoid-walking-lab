# Plan

Two tracks: learning-based walking (training a policy from scratch) and classical control for humanoids.

---

## Track 1: Learning-based walking, from scratch (week of 2026-10-26)

**Goal.** Find out by experiment what every part of a walking-policy training setup does to the final policy:
each reward term, the key observations, and the learning algorithm. Write each part myself. End with complete
training and inference written from scratch, as Kaggle notebooks and blog posts.

**Reference.** Unitree's `Unitree-G1-Flat` task (mjlab 1.2.0) trained with `rsl_rl`'s PPO. Baseline done:
3000 iterations, 2 seeds, walks like Unitree's released policy in MuJoCo and Gazebo ([training/](training/)).

**Rules.**
- One change per run; write the prediction before the run.
- Same budget, robots and PPO settings for every run; two seeds where an effect is small (heading drift
  differs between seeds).
- Code is written while runs train: a push trains two variants at once (one per T4), about 2 h each.

**Run budget.** Kaggle gives about 30 GPU hours a week → about 15 pushes → **about 30 runs** this week at
1000-1500 iterations. The run list below is ordered by priority; if runs pile up, rent a GPU for the extra.

### Day 1: experiment setup

- [x] Choose the training budget: evaluate the baseline's checkpoints (every 100 iterations) and find where the
      walking stops changing (expected 1000-1500 iterations). **1000 iterations** (round 2, reward study README).
- [x] `training/variants.py`: each variant = a function that changes one thing in the baseline config.
- [x] Notebook: a `VARIANT` setting; clones this repo at a fixed commit, registers the variant, trains it.
- [x] Evaluation report per policy ([`training/report.py`](training/report.py), videos with `video_variant.py`): command tracking (exists), gait (clearance, air time, slip), smoothness
      and effort (action rate, joint acceleration, torque, power), posture, robustness (pushes, friction, no
      armature), per-term rewards from the training log, side-by-side video against the baseline.
- [x] First push: reward ladder steps 0 and 1 (notebook v5).

### Days 2-3: the reward terms, written myself

For each term: read Unitree's version, write my own (mjlab reward signature `func(env, **params)` → one value
per robot), test it gives the same values as Unitree's on the same states, predict, run.

**Add one in** (each step keeps the previous ones):

| Step | Added | Question |
|---|---|---|
| 0 | `track_linear_velocity`, `track_angular_velocity`, `is_terminated` | Does pure command-following learn to move, and how? |
| 1 | `body_orientation_l2` | Upright instead of lunging? |
| 2 | `pose` | Near the default posture? |
| 3 | `action_rate_l2`, `joint_acc_l2` | What jitter looks like without smoothness terms |
| 4 | `foot_gait` | Alternating steps on the clock's rhythm |
| 5 | `foot_clearance`, `foot_slip`, `soft_landing` | Lift the feet, don't skid, land softly |
| 6 | `stand_still` | Really standing still at zero command |
| 7 | `body_ang_vel`, `angular_momentum` | Less torso wobble |
| 8 | `joint_pos_limits`, `self_collisions` | Joints in range, legs don't collide |

**Leave one out** from the full baseline, in priority order: `foot_gait`, `stand_still`, `pose`,
`action_rate_l2`, `joint_acc_l2`, `foot_slip`, `foot_clearance`, `body_orientation_l2` (the rest if budget
allows).

**Observations and randomization:** gait clock removed; base linear velocity added to the actor (the `walk.onnx`
design); armature / PD-gain randomization on (does it fix Gazebo without armature?).

- [ ] My implementations of all 16 terms, each tested against Unitree's. **Not done:** the study ran Unitree's own
      terms; moved to Day 4 (my own task config).
- [x] Ladder runs 0-8 (9: steps 0-7 + baseline as step 8), leave-one-out runs (8), observation/randomization runs
      (3). Extra rounds 11-16: armature randomization with a height fall check, strict fall check, Humanoid-Gym's
      hard tracking terms, Gazebo repeats.
- [x] Results table and videos per run ([training/reward_study/](training/reward_study/)); Week 4 post
      *What Each Reward Teaches a Walking Robot* (2026-10-29).

### Day 4: the task from scratch

- [ ] My own observation, command, termination and randomization terms, plus my rewards, in my own task config
      (no Unitree task code)
- [ ] Term-by-term check against `Unitree-G1-Flat`; one training run with `rsl_rl` matching the baseline

### Days 5-6: PPO from scratch

- [ ] Read `rsl_rl` 5.0.1's PPO and list the details: actor-critic 512-256-128 ELU, Gaussian policy with learned
      std, GAE (γ 0.99, λ 0.95), clipped surrogate 0.2, clipped value loss, entropy 0.01, 5 epochs × 4
      mini-batches, observation normalization, adaptive learning rate from the KL divergence, gradient clipping,
      timeout bootstrapping
- [ ] Write it piece by piece, tested on a small CPU problem first (GAE against a hand-computed example, then the
      update, then the loop)
- [ ] Plug into mjlab's environment; train `Unitree-G1-Flat` and my own task; compare with the `rsl_rl` baseline
- [ ] Ablate PPO details if budget allows: no observation normalization, fixed learning rate, no entropy bonus

### Day 7: fully from scratch, and the content

- [ ] Stretch: replace mjlab with my own environment on MuJoCo Warp (parallel robots, resets, commands,
      observations, rewards, terminations, randomization), trained with my PPO. Inference is already mine
      (MuJoCo control loop, ROS 2 node for Gazebo)
- [ ] Notebooks: *What each reward term does*, *A walking task from scratch*, *PPO from scratch*, *Humanoid
      walking from scratch: training and inference*
- [ ] Blog post and evaluation of the final policy in MuJoCo and Gazebo

**Fallback:** if the MuJoCo Warp environment doesn't work in time, my own task (Day 4) + my own PPO (Days 5-6)
already make the from-scratch story; the environment moves to the next week.

---

## Track 2: Classical control for humanoids

Course syllabus with modules, labs and references: [classical/README.md](classical/README.md). Weekly schedule
to be planned.

- [x] Module 1 Part B: LQR balance of the cart-pole ([classical/01_dynamics_lqr_mpc/lqr.py](classical/01_dynamics_lqr_mpc/lqr.py))
- [x] Module 1 Part C: iLQR swing-up ([classical/01_dynamics_lqr_mpc/ilqr.py](classical/01_dynamics_lqr_mpc/ilqr.py))
- [x] Module 1 Part D: iLQR as MPC ([classical/01_dynamics_lqr_mpc/mpc.py](classical/01_dynamics_lqr_mpc/mpc.py))
- [ ] Module 1 Part E: G1 CoM and foot Jacobians

---

## Done

- **Weeks 1-2 (2026-10-09 to 2026-10-15):** MuJoCo from scratch, the G1 walking with `walk.onnx` and Unitree's
  policy in plain MuJoCo ([exercises/](exercises/)).
- **Week 3 (2026-10-16 to 2026-10-24):** ROS 2 + Gazebo deployment, the joint-controller and armature
  investigation, upstream issues ([open_questions/](open_questions/), [notes/gazebo_experiments.md](notes/gazebo_experiments.md));
  training on Kaggle, baseline run ([training/](training/)).
