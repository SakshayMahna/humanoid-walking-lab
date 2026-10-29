---
title: "Week 4 — What Each Reward Teaches a Walking Robot"
excerpt: "Training the Unitree G1 to walk with reward terms added and removed one at a time, to see what each one really does."
categories: blog
header:
  overlay_image: /assets/images/week-4/banner.png
  overlay_filter: 0.4
  teaser: /assets/images/week-4/banner.png
---

Last week ([Week 3]({{ site.baseurl }}{% post_url 2026-10-24-from-mujoco-to-gazebo %})) ended with a question I couldn't answer by looking at simulators: what does a walking policy actually learn from its training? Unitree's walking task gives the robot **16 rewards**, and I had no idea which ones matter.

So this week I trained the robot myself, more than 30 times, each time with a different set of rewards. With only the speed command as a goal, the robot lies down on its back. One more penalty and it stands up and shuffles. One more and it walks.

At the end, I took the trained policies to Gazebo, last week's simulator, to see how they cope there.

All the code is in the [`training/`](https://github.com/SakshayMahna/humanoid-walking-lab/tree/main/training) folder, and every number is in the [study notes](https://github.com/SakshayMahna/humanoid-walking-lab/blob/main/training/reward_study/README.md).

## How the robot learns to walk

The robot is controlled by a small neural network, the **policy**. Every 20 ms it gets 98 numbers about the robot:

- how fast the body is turning;
- which way gravity points (whether it's tilting);
- the speed command;
- a gait clock;
- the 29 joint angles and joint speeds;
- the last action.

From these it outputs 29 numbers, a target angle for each motor.

At first the network is random and the robot falls over. Training changes it, step by step, so that the robot collects more **reward**. The reward is a score given every 20 ms, made of many terms added together:

$$
r_t = \sum_i w_i \, r_i(t)
$$

Each term $$r_i$$ measures one thing (am I going at the commanded speed? are my feet slipping?), and its weight $$w_i$$ says how much it counts. A positive weight makes it a goal; a negative weight makes it a penalty. The training algorithm (PPO) only ever sees this one number. Whatever earns more of it, the robot will learn to do, whether or not that's what we meant.

An episode lasts at most 20 seconds. It ends early when the robot **falls**, and a fall costs 200 points. In Unitree's task a fall means one thing only: the body tilts more than 70°. That detail matters a lot later.

Training runs 4096 robots at the same time on a GPU, in MuJoCo Warp. One run is 1000 iterations of 24 steps per robot, about 100 million steps in total. That takes about 1.4 hours on one of Kaggle's free T4 GPUs.

## The method

I used Unitree's own G1 walking task (`Unitree-G1-Flat`) and changed **only the rewards**. Everything else stayed the same: the robot, the observations, the random pushes and frictions during training, the training settings. I tried two ways of changing the rewards:

- **Add one in.** Start with only "follow the speed command", then add the other terms in groups, one group per run, keeping the previous ones. This shows what each group *builds*.
- **Leave one out.** Take all 16 terms and remove one. This shows what each term *protects against* once everything else is in place.

Each variant is a small function that edits the task's reward table:

```python
LADDER = [
    ["track_linear_velocity", "track_angular_velocity", "is_terminated"],   # 0: follow the command
    ["body_orientation_l2"],                                               # 1: upright
    ["pose"],                                                              # 2: default posture
    ["action_rate_l2", "joint_acc_l2"],                                    # 3: smoothness
    ["foot_gait"],                                                         # 4: step on the clock
    ...                                                                    # 5-8: feet, standing, wobble, limits
]

def leave_out(term):
    def variant(cfg):
        keep_rewards(cfg, [t for t in ALL_TERMS if t != term])
    return variant
```

(simplified; the full file is [`variants.py`](https://github.com/SakshayMahna/humanoid-walking-lab/blob/main/training/variants.py))

Every trained policy then got two checks:

1. **Training curves:** how each reward term changed during training.
2. **Tests in plain MuJoCo** (my [Week 2]({{ site.baseurl }}{% post_url 2026-10-15-making-the-unitree-g1-walk %}) setup):
   - walking at several speeds, sideways and turning;
   - steps per second, foot lift and foot slip;
   - power used;
   - four pushes, two floor frictions;
   - the robot without armature;
   - a video: stand for 4 seconds, then walk forward at 0.5 m/s, always filmed with the same camera.

Most variants were trained once. To know how much a result changes by chance, I trained the full set twice with different random seeds. The two runs differed by up to 0.06 m/s in walking speed and 0.1 cm in foot lift. Anything smaller than that, I count as "no clear effect".

## Adding rewards one group at a time

Each video shows one step next to the step before it, in plain MuJoCo: stand for 4 seconds, then walk forward at 0.5 m/s.

### Step 0: follow the command

The two tracking rewards compare the robot's speed $$v$$ with the command $$c$$:

$$
r_\text{lin} = \exp\!\left(-\frac{\lVert c_{xy} - v_{xy}\rVert^2 + 2v_z^2}{0.25}\right), \qquad
r_\text{ang} = \exp\!\left(-\frac{(c_z - \omega_z)^2 + 0.05\,\lVert\omega_{xy}\rVert^2}{0.5}\right)
$$

In plain English: you get the full point when your forward, sideways and turning speeds match the command, and less the further off you are. The extra pieces punish bouncing up and down and rocking. Together with the fall penalty, this is the smallest reward that describes walking.

**What the robot learned: to lie on its back with its legs in the air.** It never walks.

<video autoplay loop muted playsinline width="50%" aria-label="Step 0: the robot lies on its back with its legs in the air">
  <source src="{{ '/assets/videos/week-4/step_0.mp4' | relative_url }}" type="video/mp4">
</video>

Two things make lying down the best option:

- **Doing nothing still pays.** If the robot doesn't move, its speed error is just the command itself. For a command of 0.5 m/s, that still earns $$\exp(-0.5^2/0.25) = 0.37$$ of the point, every 20 ms.
- **Lying down is not a fall.** The fall check only looks at tilt, and a robot can lie with its pelvis not tilted much.

So lying down is safe and pays something. Walking pays more, but only after a long phase of falling, at −200 each time.

<div class="notice--info" markdown="1">
**How much does standing still pay?** Training picks random commands: forward between −0.5 and 1.0 m/s, sideways ±0.5 m/s, turning ±1 rad/s, and zero for 5% of robots. Averaged over those commands, a robot that never moves earns **43%** of the maximum linear-tracking reward and **62%** of the turning reward. A narrow exponential reward like this is good at fine-tuning a robot that's already close, and bad at stopping a robot from giving up.
</div>

### Step 1: + stay upright

$$
r_\text{upright} = -1.0 \cdot \lVert g_{xy} \rVert^2
$$

$$g$$ is the direction of gravity seen from the torso. When the torso is upright, gravity points straight down and its sideways part $$g_{xy}$$ is zero. Any lean makes it grow, and the penalty with it.

**What the robot learned: to sit up, propped on its arms.** The torso is upright now, so the penalty is gone, but the robot is still on the floor.

<video autoplay loop muted playsinline width="100%" aria-label="Step 0 lies on its back; step 1 sits up, propped on its arms">
  <source src="{{ '/assets/videos/week-4/step_1.mp4' | relative_url }}" type="video/mp4">
</video>

### Step 2: + keep the default posture

$$
r_\text{pose} = \exp\!\left(-\,\text{mean}_j \frac{(q_j - q_{0,j})^2}{\sigma_j^2}\right)
$$

$$q_j$$ is joint $$j$$'s angle and $$q_{0,j}$$ its angle in the default standing pose. $$\sigma_j$$ is how much freedom the joint gets: 0.05 rad (3°) when standing, more for the legs when walking.

**What the robot learned: to sit upright.** The posture reward doesn't help here, and the reason is its shape. Near the default pose it pays well and pulls the joints back. A sitting robot is far from the default pose, where this reward is practically zero *and flat*. Moving a little closer still earns nothing, so the reward gives the training no direction to go in.

<video autoplay loop muted playsinline width="100%" aria-label="Step 1 sits propped on its arms; step 2 sits upright">
  <source src="{{ '/assets/videos/week-4/step_2.mp4' | relative_url }}" type="video/mp4">
</video>

### Step 3: + move smoothly

$$
r_\text{rate} = -0.05\,\lVert a_t - a_{t-1}\rVert^2, \qquad r_\text{acc} = -2.5\times10^{-7}\,\lVert \ddot q \rVert^2
$$

The first penalises changing the motor targets a lot from one step to the next. The second penalises fast joint accelerations.

**What the robot learned: to stand up.** It's the first policy that walks. Getting down to the floor, rocking and propping all need big, sudden changes in the motor targets, and now those cost reward. Sitting no longer pays.

<video autoplay loop muted playsinline width="100%" aria-label="Step 2 sits upright; step 3 stands up and shuffles forward">
  <source src="{{ '/assets/videos/week-4/step_3.mp4' | relative_url }}" type="video/mp4">
</video>

Two follow-up runs showed which term does it:

- step 2 plus **only** `action_rate_l2` also stands up;
- removing `joint_acc_l2` from the full set changes nothing.

But this robot **shuffles**: 6.1 tiny steps per second, feet lifted only 1.4 cm, both feet on the ground 39% of the time. It also has a **dead zone**: told to walk at 0.2 m/s, it barely moves (0.07 m/s). `walk.onnx` from Week 2 has the same dead zone, and it was also trained without a gait reward.

### Step 4: + step on the gait clock

The gait clock is a timer that runs through a cycle every 0.6 seconds, with the right foot half a cycle behind the left. For each foot, the clock says "this foot should be on the ground" for the first 56% of its cycle and "in the air" for the rest:

$$
\phi_\text{foot} = \left(\frac{t}{0.6} + \text{offset}_\text{foot}\right) \bmod 1, \qquad
r_\text{gait} = 0.5 \cdot \frac{\#\{\text{feet where contact matches } \phi < 0.56\}}{2}
$$

The reward is the share of feet doing what the clock says. It only counts when the robot is told to move.

**What the robot learned: real, alternating steps.** 3.4 steps per second (one step per foot in each 0.6 s clock cycle), 0.25 s in the air per step, and **the dead zone is gone**: at 0.2 m/s it now walks at 0.24 m/s. The gait reward makes the robot step even when the command is small.

<video autoplay loop muted playsinline width="100%" aria-label="Step 3 shuffles with tiny steps; step 4 takes real alternating steps">
  <source src="{{ '/assets/videos/week-4/step_4.mp4' | relative_url }}" type="video/mp4">
</video>

### Steps 5 to 8: polishing the walk

The remaining nine terms each target one detail. All are penalties, and most only count while the robot is told to move:

| Term | Penalty | In plain English |
|---|---|---|
| `foot_clearance` | $$\sum_\text{feet} \lvert z_\text{foot} - 0.10 \rvert \cdot \lVert v_\text{foot} \rVert$$ | moving feet should be 10 cm up |
| `foot_slip` | $$\sum_\text{feet on ground} \lVert v_\text{foot} \rVert^2$$ | a foot on the ground shouldn't slide |
| `soft_landing` | contact force at each touchdown | land gently |
| `stand_still` | $$\lVert q - q_0 \rVert^2$$ when the command is zero | really stand still |
| `body_ang_vel` | $$\lVert \omega_{xy} \rVert^2$$ of the torso | don't rock the torso |
| `angular_momentum` | whole-body angular momentum squared | don't swing around |
| `joint_pos_limits` | how far joints go past their limits | stay inside the joint range |
| `self_collisions` | steps with legs or arms hitting each other | don't kick yourself |

With these added, the walk changed only a little:

- **Feet lift about 1 cm higher** (2.5 → 3.4 cm). The clearance target is 10 cm, but the penalty is multiplied by the foot's speed, so it mainly lifts feet that are swinging.
- **The torso terms make the walk calmer and cheaper:** less torso tilt, 80 → 69 W.
- **`stand_still` showed no effect.** The robot already stood still because of `pose`.
- **`foot_slip` and `soft_landing` changed less than the seed noise.**

<video autoplay loop muted playsinline width="100%" aria-label="Step 4 and the full set of 16 rewards walk almost the same way">
  <source src="{{ '/assets/videos/week-4/steps_5_8.mp4' | relative_url }}" type="video/mp4">
</video>

Steps 0 to 4 decide **whether** the robot walks. Steps 5 to 8 only change **how** it walks.

## Taking one reward away

The other direction: all 16 terms, minus one. Each video shows the full set next to the set without one term.

**Without `foot_gait`, the shuffle comes back**, even though all the foot terms are still there:

- 4.9 steps per second and 1.1 cm of foot lift;
- the dead zone in full: no movement at all at 0.2 m/s forward or 0.3 m/s sideways;
- 0.64 m/s when told to walk at 1.0.

<video autoplay loop muted playsinline width="100%" aria-label="The full set takes real steps; without foot_gait the robot shuffles">
  <source src="{{ '/assets/videos/week-4/without_foot_gait.mp4' | relative_url }}" type="video/mp4">
</video>

The foot terms only refine a gait; they don't create one.

**Without `pose`, the arms swing.** Nothing else in the reward says anything about the arms, so the policy uses them freely:

- shoulders and elbows move 10 to 25 times more than with the full set;
- the robot uses twice the power (144 W vs 72 W);
- its feet lift twice as high.

<video autoplay loop muted playsinline width="100%" aria-label="The full set keeps its arms still; without pose the arms swing">
  <source src="{{ '/assets/videos/week-4/without_pose.mp4' | relative_url }}" type="video/mp4">
</video>

`pose` is mainly what keeps the upper body still.

**Without `action_rate_l2`, the robot jitters:**

- its actions change twice as much from step to step;
- turning nearly breaks (0.15 rad/s when told to turn at 0.5);
- walking straight still works.

<video autoplay loop muted playsinline width="100%" aria-label="The full set and the policy trained without action_rate_l2, both walking straight">
  <source src="{{ '/assets/videos/week-4/without_action_rate.mp4' | relative_url }}" type="video/mp4">
</video>

| Without | What changed |
|---|---|
| `foot_gait` | shuffle and dead zone |
| `pose` | swinging arms, twice the power |
| `action_rate_l2` | jittery actions, turning breaks |
| `foot_clearance` | feet 0.9 cm lower (the same 1 cm as adding it) |
| `body_orientation_l2` | slightly more torso tilt (1.5° vs 1.2°) |
| `joint_acc_l2`, `foot_slip`, `stand_still` | no clear effect |

I also tried two changes to what the policy *sees*:

- **Removing the gait clock input**, together with `foot_gait`: the same result as removing only `foot_gait`. The input by itself teaches nothing; the reward that uses it does.
- **Adding the body's speed as an input** (as `walk.onnx` has): nothing measurable in plain MuJoCo, and no dead zone.

## Closing the two loopholes

Two weak spots came up in both directions:

- **"Don't fall" only checks tilt**, so lying and sitting count as standing (steps 0 to 2).
- **The tracking reward pays for standing still** (43% of the maximum), so a robot that can't find a gait for slow commands just stops: the dead zone (step 3, and without `foot_gait`).

Other humanoid training setups close both. Unitree's older Isaac Gym code ends an episode when the pelvis touches the ground and penalises the pelvis for being away from its standing height. Humanoid-Gym adds tracking rewards that punish moving too slowly. I tried each.

### A stricter fall check

The episode now also ends when the **pelvis or torso touches the ground**, and the pelvis height $$h$$ is pulled towards standing height:

$$
r_\text{height} = -10\,(h - 0.78)^2
$$

Sitting at 0.15 m now costs about 4 per step, twice what tracking can pay. I added this to step 0 (tracking and the fall penalty only) and to the full set.

<video autoplay loop muted playsinline width="100%" aria-label="Step 0 lies on its back; with the strict fall check it stands up but jitters forward with bent knees">
  <source src="{{ '/assets/videos/week-4/strict_fall.mp4' | relative_url }}" type="video/mp4">
</video>

**Step 0 now stands up, but it doesn't walk.** It jitters forward, hunched, with bent knees:

| Plain MuJoCo | All 16 rewards | Step 0 + strict fall check |
|---|---|---|
| forward at 0.2 / 0.5 / 1.0 m/s | 0.24 / 0.49 / 0.96 | **0.00** / 0.43 / 0.71 |
| sideways 0.3 m/s / turning 0.5 rad/s | 0.20 / 0.44 | **0.00 / 0.01** |
| time in the air per step | 0.25 s | **0.02 s** |
| both feet on the ground | 17% | **74%** |
| action rate (change of motor targets) | 0.43 | **193** |
| power | 72 W | **592 W** |

Its motor targets jump about 450 times more than in a normal walk. The feet barely leave the ground and slide 45 cm/s. The stricter check decides **whether** the robot stays up. It takes `action_rate_l2` to turn that into steps.

On the full set, the stricter check changes almost nothing in MuJoCo, because the full set never sat down anyway.

### Harder speed tracking

These are Humanoid-Gym's two extra tracking rewards, with its weights. With $$e_\text{lin} = \lVert c_{xy} - v_{xy} \rVert$$ and $$e_\text{ang} = \lvert c_z - \omega_z \rvert$$:

$$
r_\text{hard} = 0.5\left[\tfrac{1}{2}\left(e^{-10\,e_\text{lin}} + e^{-10\,e_\text{ang}}\right) - 0.2\,(e_\text{lin} + e_\text{ang})\right]
$$

$$
r_\text{low speed} = 0.2 \cdot
\begin{cases}
-2 & \text{moving the wrong way} \\
-1 & \lvert v_x\rvert < 0.5\,\lvert c_x\rvert \\
+1.2 & 0.5\,\lvert c_x\rvert \le \lvert v_x\rvert \le 1.2\,\lvert c_x\rvert
\end{cases}
$$

The first is a sharper peak plus a straight-line part that keeps pulling even far from the target. The second says it directly: being too slow costs reward. Standing still when told to walk no longer earns anything.

I added them to the policy **without `foot_gait`**, the one with the dead zone, and to the full set:

<video autoplay loop muted playsinline width="100%" aria-label="Without foot_gait the robot shuffles; with hard tracking it still shuffles, but keeps up with the command">
  <source src="{{ '/assets/videos/week-4/hard_tracking.mp4' | relative_url }}" type="video/mp4">
</video>

| Plain MuJoCo | Without `foot_gait` | Without `foot_gait` + hard tracking |
|---|---|---|
| forward at 0.2 / 0.5 / 1.0 m/s | **0.00** / 0.38 / 0.64 | **0.19** / 0.45 / 0.93 |
| sideways 0.3 m/s | **0.00** | **0.20** |
| steps per second | 4.9 | 5.1 |
| foot lift | 1.1 cm | 1.8 cm |
| both feet on the ground | 39% | 31% |

**The dead zone is gone, but the shuffle stays.** So the dead zone came from the tracking reward paying for standing still, not from the missing gait reward. Real steps still need `foot_gait`. Added to the full set, the hard tracking changed little, because the full set already tracked well.

## What each reward does

| Reward | What the robot learns from it | Matters? |
|---|---|---|
| speed tracking | to follow the command, but alone it also pays for lying down | the goal |
| fall penalty | not to tip over; sitting doesn't count as falling | yes, with a loophole |
| `body_orientation_l2` | to sit up instead of lying down | a little |
| `pose` | to keep the upper body still | yes |
| `action_rate_l2` | to stand up, and to move smoothly | yes, a lot |
| `foot_gait` | to take real steps | yes, a lot |
| `foot_clearance` | about 1 cm more foot lift | a little |
| torso wobble terms | a calmer, cheaper walk | a little |
| `joint_acc_l2`, `foot_slip`, `soft_landing`, `stand_still` | nothing clear in these tests | not here |
| *strict fall check* (added) | to stay on its feet, but without the other rewards it jitters instead of walking | closes the sitting loophole |
| *hard speed tracking* (added) | to follow slow commands too: no dead zone, but no real steps | closes the dead-zone loophole |

## Training for Gazebo's joints

So far everything ran in MuJoCo. Before going to Gazebo, one more experiment aimed at it. In [Week 3]({{ site.baseurl }}{% post_url 2026-10-24-from-mujoco-to-gazebo %}), adding the motors' **armature** (the extra inertia of each motor's rotor) to Gazebo's robot model was part of what made the policies behave there. Plain MuJoCo shows the same weak spot: remove the armature, and every policy I trained with `foot_gait` falls over while standing still (at the training time step of 5 ms).

So I tried training for it: give every one of the 4096 training robots a random amount of armature, between none and the full value:

$$
\text{armature}_j \leftarrow s \cdot \text{armature}_j, \qquad s \sim \mathcal{U}(0, 1) \text{ per robot}
$$

The task got harder, and **the robots sat down again**, even with all 16 rewards. Two of three runs found sitting before they found walking, and stayed there. It's the same loophole as step 0: the fall check only looks at tilt.

Here I closed it with a simple height check: a robot whose pelvis drops below 0.5 m has fallen too. It stands at 0.78 m and sits at about 0.15 m.

<video autoplay loop muted playsinline width="100%" aria-label="With randomized armature the robot sits down; with a height fall check added it walks">
  <source src="{{ '/assets/videos/week-4/armature.mp4' | relative_url }}" type="video/mp4">
</video>

With the height check, both random seeds learned to walk. They needed twice the training (2000 iterations). In plain MuJoCo they now stand still without armature:

| Plain MuJoCo | All 16 rewards | Randomized armature + height check |
|---|---|---|
| walking at 0.5 / 1.0 m/s | 0.49 / 0.96 | 0.42 / 0.87 |
| pushes, frictions | fine | fine |
| **standing without armature** | **falls** | **stands** |

## And in Gazebo?

The real test is Gazebo, with the corrected controller from Week 3: Unitree's gains, with the D part done by the joint's damping, exactly as in training. I ran six policies there: Unitree's, the full set (two seeds), the full set without `foot_gait`, and the two random-armature seeds, each without armature (the hard case) and with it. All runs use the same settings as in Week 3: Gazebo's simulation clock, headless, standing 30 seconds (three runs, seven for the first three policies) and walking at 0.5 m/s. The videos are separate recorded runs.

<video autoplay loop muted playsinline width="100%" aria-label="Gazebo without armature, standing 30 s: five policies stay up, random armature seed 1 falls">
  <source src="{{ '/assets/videos/week-4/gazebo_stand.mp4' | relative_url }}" type="video/mp4">
</video>

<video autoplay loop muted playsinline width="100%" aria-label="Gazebo without armature, walking at 0.5 m/s: five policies walk, the one without foot_gait falls">
  <source src="{{ '/assets/videos/week-4/gazebo_walk.mp4' | relative_url }}" type="video/mp4">
</video>

| Gazebo, corrected controller | No armature: standing 30 s | No armature: walking at 0.5 m/s | Armature: standing 30 s | Armature: walking at 0.5 m/s |
|---|---|---|---|---|
| Unitree's policy | 0.01-0.03 m, **walks off 9.7 m in 1 of 7 runs** | 0.51 | 0.02 m | 0.49 |
| All 16 rewards, seed 1 | 0.14-0.19 m (7 runs) | 0.63 | 0.00-0.02 m | 0.52 |
| All 16 rewards, seed 2 | 0.01-0.03 m (7 runs) | 0.57 | 0.01-0.02 m | 0.48 |
| Without `foot_gait` | 0.03-0.04 m | **falls after 11 s** | 0.02 m | 0.42 |
| Random armature, seed 1 | **1.1-1.5 m** | 0.50 | 0.04-0.05 m | 0.45 |
| Random armature, seed 2 | 0.08-0.35 m | 0.55 | **0.45-0.49 m** | 0.47 |

**The normal policies work in Gazebo, even without armature.** Seed 2 stands as still as Unitree's policy, and seed 1 creeps about 15 cm in 30 seconds. Neither walked off in seven runs. Unitree's policy did once, the same thing it did in 5 of 7 runs in Week 3. Without armature, our policies walk a bit too fast (0.57 and 0.63 m/s instead of 0.5) and turn more on the way (31° and 73° in 20 seconds, against 16° for Unitree's policy; one run each).

So falling without armature in MuJoCo didn't carry over to Gazebo. In plain MuJoCo, the time step decides it: training runs the physics every 5 ms, Gazebo every 1 ms. At 5 ms without armature, the two normal policies fall; at 1 ms, all six policies stand. The fall I trained against only happens at the training time step.

**Training with random armature didn't help in Gazebo.** Without armature, seed 1 drifts more than a metre in every run, the most of all six. In the recorded runs (the video) it even fell, in two of three. Seed 2 drifts about half a metre *with* armature. I fixed a problem Gazebo doesn't have, and the policies got a little worse at everything else.

**The policy without `foot_gait` stands but can't walk without armature.** It shuffles off, starts spinning and falls after about 11 seconds, in every run I tried. With armature it walks fine. I haven't untangled why.

**The two loophole fixes don't help in Gazebo either.** The step-0 policy with the strict fall check falls in every run without armature, and with armature it falls while walking. The full set with the strict check walks off while standing in all three runs without armature (1.4 to 5.8 m), where the normal full set stays within 20 cm. Hard tracking makes standing less steady: the full set with it drifts 0.2 to 1.2 m in 30 seconds without armature. The policy without `foot_gait` + hard tracking walked at 0.46 m/s without falling, where the plain one fell (one walking run each, so not conclusive). With armature, all of them except the step-0 policy stand and walk.

## What I learned

- **A reward is only as good as its loopholes.** "Don't fall", written as "don't tilt", allows sitting. The robot found that loophole three times, whenever walking was hard to learn. Closing it made the robot stand, but not walk.
- **An exponential reward pays for doing nothing.** It's good at fine-tuning a robot that's nearly right, and gives no direction to one that's far off. A reward that punishes being too slow removed the dead zone.
- **A few rewards do most of the work.** Tracking, the fall penalty, `action_rate_l2` and `foot_gait` decide whether the robot walks. The rest polish the walk, and several made no difference I could measure.
- **Check that a test really stands in for the target.** Plain MuJoCo without armature looked like a model of Gazebo, but its failures came from the 5 ms time step, which Gazebo doesn't have. Training against it made one policy worse in Gazebo.
- **Run things more than once.** A single Gazebo run can show a robot walking away that stands still in the next six.

## Next

- **Classical control.** Compare these learned policies with classical controllers for the G1, such as LQR and MPC.
