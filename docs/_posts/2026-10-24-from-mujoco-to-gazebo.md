---
title: "Week 3 — From MuJoCo to Gazebo"
excerpt: "Running the same walking policies in a second simulator, and finding out why they behave differently."
categories: blog
redirect_from:
  - /blog/2026/10/24/week-3.html
header:
  overlay_image: /assets/images/week-3/banner.png
  overlay_filter: 0.4
  teaser: /assets/images/week-3/banner.png
---

Last week the G1 walked in my own MuJoCo setup, with two trained policies: `walk.onnx` and Unitree's policy. This week I moved both into a second simulator, **Gazebo**, where Diego's project runs the robot through ROS 2.

A policy trained in one simulator should work in another. Same robot, same physics, roughly. It didn't. Standing still, the robot slowly walked away. Walking, it wandered from side to side. Fixing that took me deep into how Gazebo controls a joint, and it ended with two feature requests to Gazebo itself.

All the code is in the [`ros2/`](https://github.com/SakshayMahna/humanoid-walking-lab/tree/main/ros2), [`gazebo/`](https://github.com/SakshayMahna/humanoid-walking-lab/tree/main/gazebo) and [`open_questions/`](https://github.com/SakshayMahna/humanoid-walking-lab/tree/main/open_questions) folders.

## The setup

Gazebo runs on Linux with ROS 2, so everything runs in a Docker container on my Mac: Ubuntu 24.04, ROS 2 Jazzy and Gazebo Harmonic. The robot model, the empty world and the bridge between Gazebo and ROS 2 come from Diego's project, which builds on [Luis Domínguez's Gazebo setup](https://github.com/luisdbesteiro/gazebo-harmonic-ros).

I wrote my own ROS 2 node to run the policies, in C++ with ONNX Runtime. Every 20 ms it reads the robot's state, builds the observation, runs the network and sends 29 joint targets to Gazebo:

![ROS 2 node: sensors in, joint targets out]({{ '/assets/images/week-3/node_diagram.svg' | relative_url }})

One node runs **both** policies. Each ONNX file already carries everything the node needs in its metadata: the joint names and order, the default pose, the action scale and the list of observation terms. So the node builds the observation by name:

```cpp
for (const auto& term : observation_names_) {          // from the ONNX metadata
  if (term == "base_lin_vel")           append(lin_vel_);     // walk.onnx only
  else if (term == "base_ang_vel")      append(ang_vel_);
  else if (term == "projected_gravity") append(gravity_);
  else if (term == "command")           append(command_);
  else if (term == "phase")             append(gait_phase()); // Unitree only
  ...
}
```

(simplified; the full node is [`policy_runner.cpp`](https://github.com/SakshayMahna/humanoid-walking-lab/blob/main/ros2/g1_policy_runner/src/policy_runner.cpp))

## The first runs

Here are both policies in Gazebo, exactly as set up. First a zero command for 30 seconds, then walking at 0.5 m/s (videos at 2x speed):

<video autoplay loop muted playsinline width="100%" aria-label="Both policies in Gazebo with a zero command, original setup">
  <source src="{{ '/assets/videos/week-3/raw_stand.mp4' | relative_url }}" type="video/mp4">
</video>

<video autoplay loop muted playsinline width="100%" aria-label="Both policies in Gazebo walking at 0.5 m/s, original setup">
  <source src="{{ '/assets/videos/week-3/raw_walk.mp4' | relative_url }}" type="video/mp4">
</video>

I repeated every run several times. All runs in this post use the same settings: Gazebo's simulation clock, headless, the same commands.

| | Standing 30 s: distance moved | Walking at 0.5 m/s: speed, heading wander |
|---|---|---|
| MuJoCo, Unitree | 0.01 m | 0.50 m/s, 0.4° |
| MuJoCo, walk.onnx | 0.01 m | 0.43 m/s, 0.2° |
| **Gazebo, Unitree** | **2.6-7.1 m** (6 runs) | 0.56 m/s, **5.7°** |
| **Gazebo, walk.onnx** | 0.3-1.3 m (4 runs) | 0.60 m/s, **4.4°** |

Both policies walk, but neither behaves like in MuJoCo. Told to stand still, Unitree's policy shuffles away by metres. Walking, both wander left and right ("wander" is how far the heading swings around its average direction).

## What's different: the joint controller

The policy doesn't move the joints itself. It sends a target angle per joint, and a **joint controller** turns that into a torque. In MuJoCo (and in training) that's a PD controller, for every joint:

$$
\tau = k_p\,(q_{\text{target}} - q) - k_d\,\dot q
$$

In Gazebo it's a plugin called `JointPositionController`, set up in the robot's model file. I compared its settings with the training values and found three differences.

**1. The gains.** Different motors, different $$k_p$$ and $$k_d$$:

| Joints | Training $$k_p$$ / $$k_d$$ | Gazebo $$k_p$$ / $$k_d$$ |
|---|---|---|
| hip pitch, hip yaw, waist yaw | 40.2 / 2.56 | 44.1 / 2.81 |
| hip roll, knee | 99.1 / 6.31 | 148.6 / 9.46 |
| ankles, waist roll and pitch | 28.5 / 1.81 | 34.4 / 2.19 |
| shoulders, elbows, wrist roll | 14.3 / 0.91 | 17.2 / 1.10 |
| wrist pitch and yaw | 16.8 / 1.07 | 5.0 / 0.50 |

The ratio $$k_d / k_p$$ is the same in both, so whoever tuned them kept the same natural frequency and damping as training ([Week 2]({{ site.baseurl }}{% post_url 2026-10-15-making-the-unitree-g1-walk %})) and used larger inertias. I couldn't find where those came from, so I asked on [Luis's repository](https://github.com/luisdbesteiro/gazebo-harmonic-ros/issues/1).

**2. The torque limits.** Mostly the same, except the ankle roll: 10 N·m in Gazebo, 50 in training.

**3. The D term.** This one took me longest. Gazebo's plugin uses a general-purpose PID controller. It computes the D term from the change in the **error**, not from the joint's speed:

$$
e = q - q_{\text{target}}, \qquad \tau = -k_p\, e - k_d\,\frac{e - e_{\text{prev}}}{\Delta t}
$$

When the target stays the same, the error only changes because the joint moves, so this is just $$-k_d\,\dot q$$, the same as training. But the target only changes every 20 ms, when the policy sends a new one. At that moment the error jumps in a single 1 ms physics step, and the D term produces a huge spike:

$$
k_d\,\frac{\Delta q_{\text{target}}}{\Delta t} = 9.46 \times \frac{0.05\ \text{rad}}{0.001\ \text{s}} \approx 473\ \text{N·m}
$$

for a knee target that moves by 0.05 rad. The motor limit clips it to 139 N·m, but it's still a hard kick toward the new target, 50 times a second. This is called a **derivative kick**.

![The derivative kick: knee torque with D on the error vs D on velocity]({{ '/assets/images/week-3/derivative_kick.svg' | relative_url }})

### Trying controller variants

I wrote a script that makes copies of the robot's model file with different controllers ([`make_gain_variants.py`](https://github.com/SakshayMahna/humanoid-walking-lab/blob/main/gazebo/make_gain_variants.py)):

- **Training gains**: $$k_p$$, $$k_d$$ and limits from training, D term still in the plugin.
- **Training gains, D on velocity**: the plugin only does the P part ($$k_d = 0$$), and $$k_d$$ becomes the joint's **damping** in the model file. Gazebo's physics engine then applies $$-k_d\,\dot q$$ itself, exactly the training formula, with no kick. I'll call this the **corrected controller**.

![Controller variants: standing drift, walking speed and heading wander for both policies]({{ '/assets/images/week-3/variants_chart.svg' | relative_url }})

| | Unitree: standing, walking | walk.onnx: standing, walking |
|---|---|---|
| MuJoCo (reference) | 0.01 m; 0.50 m/s, 0.4° | 0.01 m; 0.43 m/s, 0.2° |
| Original controller | 2.6-7.1 m; 0.56 m/s, 5.7° | 0.3-1.3 m; 0.60 m/s, 4.4° |
| Training gains | 6.2-9.5 m; 0.55 m/s, 5.3° | 0.4-1.0 m; 0.57 m/s, 2.3° |
| **Corrected controller** | **walks off in 5 of 7 runs; 0.51 m/s, 0.5°** | **walks off in 7 of 7; 0.72 m/s, 1.4°, slides sideways** |

The gains alone changed little. Moving the D term onto the joint's speed fixed Unitree's **walking**: the right speed and almost as straight as in MuJoCo. But it didn't fix standing: in most runs the robot still walked away, up to 10 m in 30 seconds. And **`walk.onnx` got worse**: it crouched, walked too fast and slid sideways.

<video autoplay loop muted playsinline width="100%" aria-label="Corrected controller in Gazebo: Unitree's policy walks straight, walk.onnx too fast">
  <source src="{{ '/assets/videos/week-3/corrected_gazebo.mp4' | relative_url }}" type="video/mp4">
</video>

So two questions: what exactly was wrong with the original controller, and why isn't the corrected one enough?

## Why the original controller fails

The corrected controller changes three things at once: the gains, the limits and the D term. To find out which one matters, I went back to MuJoCo, where I can change one thing at a time.

I switched off MuJoCo's own motors and computed the joint torques myself, every 1 ms, exactly the way Gazebo's plugin does ([`mujoco_controller_test.py`](https://github.com/SakshayMahna/humanoid-walking-lab/blob/main/open_questions/q1b_controller/mujoco_controller_test.py)). Then I tried every combination of gains (training or Gazebo), limits (training or Gazebo), D term (on velocity, as joint damping, or on the error) and one more thing: **armature**.

### The missing armature

In [Week 2]({{ site.baseurl }}{% post_url 2026-10-15-making-the-unitree-g1-walk %}) I worked out where armature comes from: the motor's spinning parts, seen through the gearbox, make each joint feel heavier. MuJoCo adds it to every joint. Gazebo's model format has **no armature setting at all**, so in Gazebo the joints are as light as the bare links.

For most joints that barely matters. For the light ones it's huge: the ankle roll's inertia is 0.0076 kg·m² with armature and 0.0004 without. **95 % of what that joint feels comes from the motor.**

![Every combination in MuJoCo: which controller breaks the robot]({{ '/assets/images/week-3/mujoco_grid.gif' | relative_url }})

| Armature | How the D term is applied | Standing 30 s | Walking |
|---|---|---|---|
| yes | on velocity, explicit | 0.01 m | wander 0.42° |
| yes | as joint damping, implicit | 0.01 m | wander 0.41° |
| yes | on the error (Gazebo) | 0.10 m | wander 0.88° |
| **no** | **as joint damping, implicit** | **0.01 m** | **wander 0.37°** |
| no | on velocity, explicit | **falls** | falls |
| no | on the error (Gazebo) | **falls** | falls |

The pattern is clear. With armature, everything works, even Gazebo's kicking D term. Without armature, it only works when the damping is applied **implicitly**. With Gazebo's own gains and its 10 N·m ankle limit, the robot doesn't fall but drifts 1.7 m and turns 87° (bottom right), which is what Gazebo showed. The gains and limits barely matter otherwise.

### Explicit vs implicit damping

"Explicit" and "implicit" are about *when* the simulator measures the speed it damps.

**Explicit:** the damping force uses the speed at the start of the step and pushes for the whole step. For a joint with inertia $$I$$ and damping $$k_d$$, one step of $$\Delta t$$ gives

$$
v_{\text{next}} = v\,(1 - a), \qquad a = \frac{k_d\,\Delta t}{I}
$$

If $$a > 2$$, the speed flips sign and **grows** every step: the damper throws the joint back harder than it came.

**Implicit:** the damping force uses the speed at the end of the step:

$$
v_{\text{next}} = v - a\,v_{\text{next}} \quad\Rightarrow\quad v_{\text{next}} = \frac{v}{1 + a}
$$

This always shrinks and never flips, whatever the inertia.

For the ankle roll at Gazebo's 1 ms step: with armature, $$a = 0.24$$, and the speed decays smoothly. Without armature, $$a = 4.5$$: every millisecond, $$v_{\text{next}} = -3.5\,v$$. Only the torque limit stops it from exploding.

The animation shows one ankle-roll joint, with the real gains and torque limit, stepping toward a new target 0.1 rad away. With armature, or with implicit damping, it moves there smoothly. With explicit damping and no armature, its speed flips sign every millisecond, and because the torque limit cuts the swings unevenly, the joint doesn't even stay near its target: it runs away. On the robot, the ankles push it around in the same way, and it shuffles off.

![Explicit vs implicit damping on a light joint (slow motion)]({{ '/assets/images/week-3/explicit_vs_implicit.gif' | relative_url }})

<div class="notice--info" markdown="1">
**The full stability condition.** With the spring included, this is the condition from the [Week 1]({{ site.baseurl }}{% post_url 2026-10-10-what-a-pendulum-taught-me-about-humanoid-control %}) pendulum:

$$
(\omega_n \Delta t)^2 + \frac{2\,k_d\,\Delta t}{I} < 4, \qquad \omega_n^2 = \frac{k_p}{I}
$$

Semi-implicit Euler updates angle and speed with the matrix $$\begin{bmatrix} 1 - \omega_n^2\Delta t^2 & \Delta t(1-a) \\ -\omega_n^2 \Delta t & 1 - a \end{bmatrix}$$, whose determinant is $$1 - a$$ and trace $$2 - \omega_n^2\Delta t^2 - a$$. It's stable when $$\lvert \det \rvert < 1$$ and $$\lvert \text{trace} \rvert < 1 + \det$$, which gives the condition above.

| Joint | Inertia with armature | Without | Condition with → without (limit 4) |
|---|---|---|---|
| ankle roll | 0.0076 | 0.0004 | 0.5 → **9.4** |
| wrist roll | 0.0040 | 0.0004 | → **5.0** |
| ankle pitch | 0.0100 | 0.0028 | stable |
| knee | 0.139 | 0.114 | stable |

Only the ankle roll and the wrist roll cross the limit, and only without armature. With Gazebo's gains the ankle roll is at 11.4.
</div>

Moving the damping into the joint removes this instability because Gazebo's physics engine (DART) applies joint damping **implicitly**, the same way MuJoCo's training setting does. Gazebo's PID plugin applies it explicitly, after the physics, as an ordinary force.

### Checking it in Gazebo

If this is right, the original controller should also work in Gazebo when the instability is removed. Three tests, all keeping the original controller's gains and limits:

| Unitree's policy in Gazebo | Standing 30 s | Walking: wander |
|---|---|---|
| Original controller | 2.6-7.1 m (6 runs) | 5.7° |
| + D term moved to joint damping (implicit) | 0.03 m (2 runs) | 0.3° |
| + armature added to the links | 0.10-0.14 m (3 runs) | 0.9° |
| + 5 times smaller physics step (0.2 ms) | 0.01-0.02 m (3 runs) | 0.55° |

All three remove the drift. Since Gazebo has no armature setting, I approximated it by adding each motor's armature to its link's inertia around the joint axis. That's not exactly the same (the extra inertia also counts for the joints above), but close enough to confirm the cause.

<video autoplay loop muted playsinline width="100%" aria-label="Gazebo, original controller: unchanged, with armature, and with a 0.2 ms step">
  <source src="{{ '/assets/videos/week-3/gazebo_checks.mp4' | relative_url }}" type="video/mp4">
</video>

Two things in Gazebo would make this easier, so I opened issues for them: **support for joint armature** ([sdformat#1734](https://github.com/gazebosim/sdformat/issues/1734)) and an **option for the joint controller to use the joint's speed for its D term** ([gz-sim#4092](https://github.com/gazebosim/gz-sim/issues/4092)). The second one matters beyond simulation: Unitree's real motor drivers use the measured speed too.

## Why the corrected controller isn't enough

That explains the original controller. But in MuJoCo, the corrected controller without armature works for **both** policies: they stand still and walk normally. In Gazebo it doesn't: Unitree's policy walks off while standing in most runs, and `walk.onnx` always does, and walks too fast.

The missing piece is armature again. With the corrected controller **and** armature added to the links, both policies behave like in MuJoCo, in every run:

| Gazebo, corrected controller | Standing 30 s | Walking at 0.5 m/s: speed, sideways, wander |
|---|---|---|
| Unitree, no armature | walks off in 5 of 7 runs (up to 10 m) | 0.51 m/s, −0.02, 0.5° |
| **Unitree, + armature** | **0.00-0.02 m (7 runs)** | **0.49 m/s, −0.01, 0.7°** |
| walk.onnx, no armature | walks off in 7 of 7 runs (2.9-8.3 m) | 0.72 m/s, −0.24, 1.4° |
| **walk.onnx, + armature** | **0.00-0.06 m (7 runs)** | **0.55 m/s, −0.02, 0.5°** |

<video autoplay loop muted playsinline width="100%" aria-label="Corrected controller with and without armature, both policies walking">
  <source src="{{ '/assets/videos/week-3/policy_compare.mp4' | relative_url }}" type="video/mp4">
</video>

So Gazebo needs armature even with implicit damping, while MuJoCo doesn't. Unitree's policy copes with its absence better than `walk.onnx`, but neither copes fully. One more puzzle: with Gazebo's original gains and implicit damping, Unitree's policy stood still in both runs, while with the training gains it doesn't. I haven't untangled that one yet.

Why does Gazebo need the armature when MuJoCo doesn't? I checked the obvious suspects, and none of them is it:

- **The velocity input.** `walk.onnx` is the only policy that reads the robot's speed, which Gazebo computes differently. I logged every value it received: it matches the true speed exactly, with no extra noise.
- **The robot itself.** I compared every link's mass and inertia between Gazebo's model and MuJoCo's, and the feet: Gazebo's touch the ground at four small spheres, MuJoCo's with seven capsules. Giving MuJoCo Gazebo's inertias, or Gazebo's feet, changes nothing.

What's left is the physics engine itself (how DART and MuJoCo handle very light joints in contact), and **how each policy was trained**. Unitree's training is quite different: the policy doesn't get the speed as an input, it has a gait clock ([Week 2]({{ site.baseurl }}{% post_url 2026-10-15-making-the-unitree-g1-walk %})), and it's penalised for moving while it should stand still, for fast joint movements and for falling. Neither was trained with randomized joint dynamics, so neither has learned to cope with joints that feel different.

Answering that needs training, which is next week.

## What I learned

- **A simulator isn't just physics.** The robot model, the joint controller and how damping is integrated all matter. Two simulators can share a robot model and still not run the same robot.
- **Explicit damping has a speed limit.** On a light joint it becomes unstable, and armature is what keeps the G1's ankles heavy enough. Implicit damping has no such limit, but in Gazebo the policies still needed the armature.
- **Change one thing at a time.** The corrected controller changed three things, and only rebuilding Gazebo's controller inside MuJoCo showed which one mattered.

## Next

- **Training.** I'll train walking policies myself, on Kaggle's free GPUs, and change the rewards and observations one at a time to see what makes a policy robust to differences like these.

I'm using Claude as a tutor for this project. It suggests experiments, explains concepts when I get stuck and reviews my code, while I write most of it myself.
