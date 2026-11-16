---
title: "Week 6 — Six Weeks of Humanoid Walking"
excerpt: "What I built and learned in six weeks with the Unitree G1: MuJoCo, a walking policy, Gazebo, training, classical control, and this week, the contact forces under the feet."
categories: blog
header:
  overlay_image: /assets/images/week-6/banner.png
  overlay_filter: 0.4
  teaser: /assets/images/week-6/banner.png
---

Six weeks ago I wanted to understand how a learned walking policy works. A neural network reads some sensor values, outputs some numbers, and a robot with almost 30 joints walks. I didn't know what happened in between.

This post looks back at what I built and what I learned, week by week. It ends with one more piece from this week, the forces under the robot's feet, and with what comes next.

All the code is in the [project repository](https://github.com/SakshayMahna/humanoid-walking-lab).

## Week 1: one joint at a time

I started with MuJoCo itself and a single pendulum. Two things stayed with me. First, simulation errors can be predicted: the integrator's error in free fall matched the formula to the millimetre. Second, a PD controller holding a joint against gravity always stops short of its target. So a policy that outputs joint targets learns to aim past the angle it wants, and changing the gains changes what every action means.

![Stiffness sweep]({{ '/assets/images/week-1/pd_stiffness.gif' | relative_url }}){: loading="lazy"}

[Read Week 1]({{ site.baseurl }}{% post_url 2026-10-10-what-a-pendulum-taught-me-about-humanoid-control %})

## Week 2: the G1 walks

Next I built the whole robot in plain MuJoCo: the floor, 29 motors, the starting pose, and every number the policy reads. Then I ran a trained policy, `walk.onnx`, and the G1 walked.

![walk.onnx walking in my MuJoCo setup]({{ '/assets/images/week-2/walk_forward.gif' | relative_url }}){: loading="lazy"}

But it had a dead zone. Below about 0.3 m/s it just stood, and it wouldn't turn on the spot. Unitree's own policy has a gait clock, a ticking input that tells it when to step. With the clock turned off, Unitree's policy turned on the spot at 0.02 rad/s instead of 0.81. It behaved just like `walk.onnx`, which has no clock.

[Read Week 2]({{ site.baseurl }}{% post_url 2026-10-15-making-the-unitree-g1-walk %})

## Week 3: a second simulator

I moved both policies into Gazebo, through my own ROS 2 node. Same robot, roughly the same physics, but not the same behaviour. Told to stand still, Unitree's policy walked away by 2.6 to 7.1 m.

The cause was in how Gazebo controls a joint. Its damping was explicit, and the light ankle joints had no armature (the rotor inertia of the motors). With the controller corrected and the armature added, both policies stood still in every run, within 2 cm for Unitree's policy. It ended with two feature requests to Gazebo.

![Explicit vs implicit damping on a light joint (slow motion)]({{ '/assets/images/week-3/explicit_vs_implicit.gif' | relative_url }}){: loading="lazy"}

[Read Week 3]({{ site.baseurl }}{% post_url 2026-10-24-from-mujoco-to-gazebo %})

## Week 4: what each reward teaches

I trained the robot myself, more than 30 times, adding and removing Unitree's 16 rewards one at a time. With only the speed command as a goal, the robot learned to lie on its back with its legs in the air. Doing nothing still earned part of the reward, and lying down didn't count as a fall.

![Step 0: the robot lies on its back with its legs in the air]({{ '/assets/images/week-4/step_0.gif' | relative_url }}){: loading="lazy"}

A few rewards did most of the work: speed tracking, the fall penalty, smooth actions (`action_rate_l2`) and stepping on the clock (`foot_gait`). The rest polished the walk. And whenever walking was hard to learn, the robot found the same loophole: sitting down doesn't count as a fall. It found it three times.

[Read Week 4]({{ site.baseurl }}{% post_url 2026-10-29-what-each-reward-teaches-a-walking-robot %})

## Week 5: classical control

Then I left the network aside and went through classical control: LQR, iLQR and MPC on a cart-pole, then Jacobians and LQR on the whole G1. It ended with the robot on one foot, moving between yoga poses, with linear gains and no neural network.

![The G1 on one foot, moving from flamingo to tree to warrior III and back, controlled by LQR (1.5x speed)]({{ '/assets/images/week-5/yoga_flow.gif' | relative_url }}){: loading="lazy"}

The best part was a link to learning: the Riccati equation behind LQR is a Bellman equation with an exact answer. Its value function $$V(s) = s^\top P s$$ is what a critic network tries to approximate.

[Read Week 5]({{ site.baseurl }}{% post_url 2026-11-08-classical-control-for-humanoid-robots-lqr-ilqr-mpc %})

## This week: what the feet feel

Every week so far, the robot stood on the floor, and I never looked at the forces that hold it up. Whatever controls the robot, learned or classical, those forces have to follow three rules:

1. **The floor only pushes.** It never pulls a foot down.
2. **Friction has a limit.** The sideways force at a contact can be at most $$\mu$$ times the force pressing it down, or the foot slips.
3. **The push acts inside the feet.** All the contact forces together act at one point, the **centre of pressure** (CoP), and it always lies inside the outline of the feet on the floor, the **support polygon**.

### Reading the forces

Standing on both feet, the G1 touches the floor at 28 points: both ends of the 7 capsules under each foot. MuJoCo gives each contact's force in its own frame (one normal, two sideways directions), and one matrix turns it into world coordinates ([`contact_forces.py`](https://github.com/SakshayMahna/humanoid-walking-lab/blob/main/classical/02_contact_zmp_lipm/contact_forces.py), simplified):

```python
mujoco.mj_contactForce(model, data, i, f6)        # (normal, sideways 1, sideways 2)
f_world = contact.frame.reshape(3, 3).T @ f6[:3]  # the same force, in world coordinates
```

The forces add up to 327.08 N, the robot's weight (33.34 kg × 9.81). And they match MuJoCo's own solver to rounding error.

The CoP comes from the total force $$f$$ and the total moment $$n$$ about the origin. It's the point on the floor about which the forces don't try to tip anything:

$$
p_x = -\frac{n_y}{f_z}
$$

$$
p_y = \frac{n_x}{f_z}
$$

Standing still, the CoP is exactly under the centre of mass. If it weren't, the floor's push would make the robot start to rotate. When I shifted the robot's weight onto one foot and lifted the other, the CoP walked across into the standing foot. The load went from 164 N on each foot to 327 N on one.

One detail surprised me. MuJoCo's floor is a stiff spring, about 6 kN for every millimetre a foot sinks in. With 310 of the 327 N on the left foot, the left foot sinks 0.66 mm and the right 0.60 mm. A difference of 0.06 mm decides where the weight goes.

### Slide or tip?

To see rules 2 and 3 at work, I pulled the robot sideways at the hips, harder every second, on two floors. A balance controller (LQR, as in Week 5) held its pose. Once a foot left the floor, I switched the controller off and held the joints still, so it falls like a statue.

![The same sideways pull on a grippy floor (the robot tips over) and on ice (it slides)]({{ '/assets/images/week-6/slide_or_tip.gif' | relative_url }}){: loading="lazy"}

On ice ($$\mu = 0.1$$), the robot slides at 32 N. Rule 2 predicts $$\mu W$$ = 33 N.

On the grippy floor ($$\mu = 0.6$$), friction would hold up to 196 N, so rule 3 runs out first. The CoP reaches the outer edge of the right foot, the left foot lifts off at 56 N, and the robot tips over.

Which one happens first depends on one number: friction against the robot's proportions. The robot slides first if $$\mu$$ is below $$d/h$$, and tips first otherwise. Here $$d$$ is the distance from the centre of mass to the outer edge of the foot, and $$h$$ is the height of the pull.

<div class="notice--info" markdown="1">
**Where $$d/h$$ comes from.** A pull $$F$$ at height $$h$$ tries to tip the robot over the outer edge of its foot. The weight $$W$$, at a distance $$d$$ from that edge, holds it down. It tips when

$$
F h > W d
$$

and it slides when

$$
F > \mu W
$$

So it slides first when $$\mu W < W d / h$$, that is, when $$\mu < d/h$$. For the G1, $$d$$ = 0.155 m and $$h$$ = 0.68 m, so $$d/h$$ = 0.23.

As a rigid body, the robot would tip at $$W d / h$$ = 74 N. It tipped at 56 N, because it isn't rigid. The pull had moved its centre of mass 36 mm towards the foot. With the centre of mass where it really was, the same formula gives 54 N.
</div>

## What six weeks taught me

- **Every detail of the setup matters.** A policy only knows the world through its observations, in the right frame and order. Two simulators with the same robot model can still run different robots.
- **Change one thing at a time, and run it more than once.** Most of what I found came from one-change experiments. A single run once showed a robot walking away that stood still in the next six.
- **A reward is a description, and the robot reads it literally.** Lying down and sitting both earned reward, so the robot learned them.
- **Learning and classical control share their foundations.** LQR's value function $$s^\top P s$$ is the exact version of what a critic network learns.
- **Contact is where every controller meets the world.** Friction and the support polygon limit a learned policy just as much as an LQR.

## What's next

Walking gave me a foundation: physics simulation, robot models, training, control, and now contact. Next I'm moving from how humanoids move to how they perceive the world around them. The classical control course stays in the repository, paused just before the zero-moment point and the linear inverted pendulum, for when I come back to walking.
