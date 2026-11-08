---
title: "Week 5 — Classical Control for Humanoid Robots: LQR, iLQR and MPC"
excerpt: "LQR, iLQR and MPC on a cart-pole, then Jacobians, null-space projection and LQR on the Unitree G1, balancing on one foot in yoga poses."
categories: blog
header:
  overlay_image: /assets/images/week-5/banner.png
  overlay_filter: 0.4
  teaser: /assets/images/week-5/banner.png
---

Since Week 2, the G1 has walked with a neural network. It works, but I can't look inside and see why. This week I left the network aside and went through classical control: controllers you can derive on paper, and analyse with linear algebra once they run.

I started on the smallest system that shows each idea, a cart-pole, and went through LQR, iLQR and MPC. Then I moved to the real robot: 35 velocities, a floating base, and Jacobians that took me a while to understand. It ended here, with the G1 on one foot, moving between three yoga poses. There is no neural network in this clip, only linear gains that change along the way (the clip plays at 1.5x speed):

![The G1 on one foot, moving from flamingo to tree to warrior III and back, controlled by LQR (1.5x speed)]({{ '/assets/images/week-5/yoga_flow.gif' | relative_url }}){: loading="lazy"}

All the code is in [`classical/01_dynamics_lqr_mpc/`](https://github.com/SakshayMahna/humanoid-walking-lab/tree/main/classical/01_dynamics_lqr_mpc), with a [course outline](https://github.com/SakshayMahna/humanoid-walking-lab/blob/main/classical/README.md) and references.

## LQR: the one problem with an exact answer

A cart-pole is a pole on a hinge, on a cart that slides on a rail. The only input is a force $$u$$ on the cart. The state is

$$
s = [x,\ \theta,\ \dot x,\ \dot\theta]
$$

with $$\theta = 0$$ upright. Near upright, the motion is almost linear. With a time step of 10 ms it becomes:

$$
s_{t+1} = A\,s_t + B\,u_t
$$

One eigenvalue of $$A$$ is 1.0518 per step. A tilt grows by 5% every 10 ms and doubles every 137 ms. That is the pole falling.

**LQR** (linear-quadratic regulator) finds the best feedback for this model. "Best" means the lowest total cost over all future steps:

$$
J = \sum_t \left( s_t^\top Q\, s_t + u_t^\top R\, u_t \right)
$$

$$Q$$ says how much a wrong state costs, and $$R$$ how much force costs. The answer is a single matrix of gains:

$$
u = -K s
$$

The idea behind it is **dynamic programming**. Let $$V(s)$$ be the total cost from state $$s$$ onwards, if you act as well as possible. Then the best action now is the one that minimises this step's cost plus the cost from wherever you land:

$$
V(s) = \min_u \left[ s^\top Q s + u^\top R u + V(As + Bu) \right]
$$

This is the **Bellman equation**, the same one behind reinforcement learning. [Last week's]({{ site.baseurl }}{% post_url 2026-10-29-what-each-reward-teaches-a-walking-robot %}) PPO trains a critic network to approximate $$V$$ from experience. Here, because the model is linear and the cost quadratic, $$V$$ has an exact form:

$$
V(s) = s^\top P s
$$

and working out the minimum gives $$K$$ and $$P$$ directly.

<div class="notice--info" markdown="1">
**From the Bellman equation to the Riccati equation.** Put $$V(s) = s^\top P s$$ into the right-hand side:

$$
s^\top Q s + u^\top R u + (As + Bu)^\top P (As + Bu)
$$

Set the gradient with respect to $$u$$ to zero:

$$
R u + B^\top P (As + Bu) = 0
$$

Solving for $$u$$ gives $$u = -Ks$$ with

$$
K = (R + B^\top P B)^{-1} B^\top P A
$$

Putting $$u$$ back in gives the next value function:

$$
P \leftarrow Q + K^\top R K + (A - BK)^\top P (A - BK)
$$

This is the Riccati equation. Each pass adds one more step of look-ahead. Repeat until $$P$$ stops changing.
</div>

Each pass of that loop answers the question "what if there were one more step left?". The gains settle as the look-ahead grows, and the constant $$K$$ at the end is LQR. The pole's gains appear first, within half a second of look-ahead. The cart's only matter once the controller can see about 2 seconds ahead:

![The four gains, as a fraction of their final value, settling as the look-ahead grows]({{ '/assets/images/week-5/riccati_gains.svg' | relative_url }}){: loading="lazy"}

Eight lines of NumPy do all of it:

```python
P = Q.copy()
for _ in range(max_iter):
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    A_cl = A - B @ K
    P_next = Q + K.T @ R @ K + A_cl.T @ P @ A_cl
    if converged(P_next, P):
        break
    P = P_next
```

(simplified; the full file is [`lqr.py`](https://github.com/SakshayMahna/humanoid-walking-lab/blob/main/classical/01_dynamics_lqr_mpc/lqr.py))

The result matches SciPy's Riccati solver to 6 parts in a billion, and it holds the pole:

![LQR bringing the cart-pole back to upright from a 0.2 rad tilt]({{ '/assets/images/week-5/lqr_balance.gif' | relative_url }}){: loading="lazy"}

### What the analysis says

LQR also makes predictions you can check.

**$$V$$ predicts the cost of the run.** From a 0.2 rad tilt, $$s_0^\top P s_0 = 76.2$$. Adding up the actual cost of the MuJoCo run gives 78.9. They differ by 4%, because the real pole is not quite linear at 0.2 rad.

**$$Q$$ and $$R$$ are a trade-off you can read directly.** Making force more expensive gives gentler forces, and the cart then needs more room to catch the pole:

| Force cost $$R$$ | Peak force | Cart travel |
|---|---|---|
| 0.01 | 14.0 N | 0.24 m |
| 0.1 | 8.3 N | 0.35 m |
| 1 | 6.4 N | 0.53 m |
| 10 | 5.7 N | 0.85 m |

**LQR doesn't know about limits.** It assumes any force is available. A real motor has a limit, and the command just gets clipped. So I tried 957 starting states, with the same gain and three force limits, and marked which ones recovered:

![Maps of the starting angles and speeds LQR recovers from: 719 of 957 with no limit, 225 with 5 N, 91 with 2 N]({{ '/assets/images/week-5/lqr_force_limits_map.svg' | relative_url }}){: loading="lazy"}

The diagonal band makes sense: a pole tilted one way but already moving back the other way is the easy case. From 0.5 rad, the same gain succeeds or fails depending only on the limit:

![The same LQR gain from 0.5 rad: it recovers with no limit, and the pole falls with a 5 N or 2 N limit]({{ '/assets/images/week-5/lqr_force_limits.gif' | relative_url }}){: loading="lazy"}

## iLQR: LQR along a trajectory

Swinging the pole up from hanging can't be done with one linear model. The physics changes along the way. Here's what one step of force does to the pole's speed at different angles:

| Pole angle | Pole speed change per newton, per step |
|---|---|
| hanging ($$\theta = 3.14$$) | +0.0200 |
| $$\theta = 2.0$$ | +0.0068 |
| $$\theta = 1.0$$ | −0.0088 |
| upright ($$\theta = 0$$) | −0.0200 |

Even the sign flips. So **iLQR** (iterative LQR) linearizes at every point along a trajectory, instead of once:

1. Guess a sequence of forces and simulate it. This gives a trajectory.
2. **Backward pass:** linearize along the trajectory and solve a time-varying LQR problem, from the last step back to the first.
3. **Forward pass:** simulate the improved forces. Keep them if the cost went down.
4. Repeat.

The backward pass is the same Bellman step as before, with one difference: the trajectory isn't optimal yet, so there is a direction to improve in. Each step gets two outputs:

$$
\delta u_t = k_t + K_t\,\delta s_t
$$

$$k_t$$ is the improvement to the plan's force. $$K_t$$ is the feedback: how to react if the state is off the plan.

<div class="notice--info" markdown="1">
**The backward pass.** Around the trajectory, the cost of taking $$\delta u$$ now and then acting optimally is approximated by a quadratic, with these terms ($$A_t, B_t$$ are the linearization at step $$t$$, $$V'$$ is the value at step $$t+1$$):

$$
Q_u = l_u + B_t^\top V_x'
$$

$$
Q_{uu} = l_{uu} + B_t^\top V_{xx}' B_t
$$

$$
Q_{ux} = l_{ux} + B_t^\top V_{xx}' A_t
$$

Minimising it gives

$$
k_t = -Q_{uu}^{-1} Q_u, \qquad K_t = -Q_{uu}^{-1} Q_{ux}
$$

The $$Q_{uu}$$ and $$Q_{ux}$$ terms are the same algebra as the Riccati step. The force limit is handled here too: $$k_t$$ is clamped so the force stays in range, and where it is clamped, $$K_t = 0$$. This follows [Tassa et al., 2012](https://roboti.us/lab/papers/TassaIROS12.pdf).
</div>

There's a direct way to check this. On a problem that really is linear and quadratic, one iteration should land exactly on the best answer. It does: the predicted cost drop and the actual one are both 0.045982, and a second iteration finds only 1.6 × 10⁻⁹ N left to change.

On the real swing-up, with a 5 N limit, it starts from zero force and takes 67 iterations. The cost drops from 2467 to 1.94. The plan swings the pole back and forth to build energy, and rides the force limit for 96 of its 300 steps:

![The iLQR plan: the pole swings back and forth and arrives upright at 3 s, while the force sits at the plus or minus 5 N limit for much of the time]({{ '/assets/images/week-5/ilqr_swingup_plan.svg' | relative_url }}){: loading="lazy"}

Those gains $$K_t$$ turned out to matter a lot. Here the plan runs on a pole 20% heavier than the planner assumed. Playing back the forces alone misses, and the pole never gets up. Adding the gains corrects the same plan on the way, and the pole arrives upright:

![The same plan on a pole 20% heavier: the forces alone miss, the plan with its feedback gains arrives upright]({{ '/assets/images/week-5/ilqr_swingup.gif' | relative_url }}){: loading="lazy"}

## MPC: keep re-planning

iLQR plans once, before moving, and that first plan needs many iterations. If the world differs too much from the plan, the feedback gains can't save it. **MPC** (model predictive control) plans all the time: every 50 ms it measures the state, solves iLQR over the next 3 seconds from there, and switches to the new plan.

Two things make this possible in real time.

**The solve takes time.** While the computer solves, the robot keeps moving under the old plan. So a new plan is only switched on one interval later, and a solve has to finish in that 50 ms.

**A good starting guess.** A cold start needs 67 iterations, and there's no time for that. The new solve starts from the previous plan, and needs only about 2.

The second one took me a while to understand, because there are two ways to reuse the old plan, and they behave very differently:

- **Replay the old forces.** Take the old plan's force sequence and simulate it from the state you're in now.
- **Replay the old policy.** Simulate the old plan's forces *with its feedback gains*: $$u = U_t + K_t (s - S_t)$$.

These look equivalent, but with a wrong model the state is always a little off the old plan. An upright pole is unstable, with errors growing like $$e^{5t}$$. Over a 3 second replay that's a factor of about 3 million, so replaying the forces alone gives a guess where the pole has fallen over, and the solve starts from nonsense. With the gains, the simulated guess gets corrected back towards the old plan and stays good:

```python
def _warm_start(self, plan_start, s):
    start, _, U_prev, _ = self.active
    U0 = np.zeros(self.H)
    for j in range(self.H - (plan_start - start)):     # the part the old plan still covers
        U0[j] = self._policy(self.active, plan_start + j, s)   # u = U + K (s - S)
        s = step(s, U0[j])                              # ... through our own model
    return U0
```

(simplified; the full file is [`mpc.py`](https://github.com/SakshayMahna/humanoid-walking-lab/blob/main/classical/01_dynamics_lqr_mpc/mpc.py))

Here all three run on a pole 30% heavier than the planner thinks, a mismatch that separates them. Only MPC with the old policy gets the pole up and keeps it there:

![Pole 30% heavier: the fixed plan fails, MPC replaying old forces fails, MPC replaying the old policy swings up and holds]({{ '/assets/images/week-5/mpc_heavy_pole.gif' | relative_url }}){: loading="lazy"}

| Swing-up scenario | Fixed plan + gains | MPC, old forces | MPC, old policy |
|---|---|---|---|
| exact model | holds | holds | holds |
| pole 20% heavier | holds | **falls** | holds |
| pole 30% heavier | **falls** | **falls** | holds |
| pole 50% heavier | **falls** | **falls** | holds |
| push of −2 rad/s | **falls** | holds | holds |
| push of −4 rad/s | **falls** | holds | holds |

What does a decision cost? Timing depends on the machine, so these are from one run on my laptop, against the 50 ms budget:

| Horizon | Iterations | Mean solve | Slowest solve | Swing-up |
|---|---|---|---|---|
| 1 s | 1 | 11 ms | 15 ms | fails |
| 1 s | 3 | 26 ms | 49 ms | fails |
| 2 s | 1 | 26 ms | 55 ms | holds |
| 2 s | 3 | 42 ms | 101 ms | holds |
| 3 s | 1 | 32 ms | 44 ms | holds |
| 3 s | 3 | 46 ms | 76 ms | holds |

With a 1 second horizon, the swing-up fails. The 3 s, 3 iteration setup used above misses the budget on its slowest solves. The simulation waits for each solve, so that doesn't show in the results, but on a real robot it would.

MPC also fixes LQR's problem with force limits, because the limit is part of what it plans with. With a 2 N limit, LQR recovers from 9 of these 81 starts, and MPC from 23:

![Starts recovered with a 2 N limit: 9 of 81 for LQR, 23 of 81 for MPC]({{ '/assets/images/week-5/lqr_vs_mpc_region.svg' | relative_url }}){: loading="lazy"}

### Where it still falls short

MPC only finds the best answer near its starting guess, and only for the cost you wrote. Starting at 0.1 rad, with a guess of zero force, iLQR plans to let the pole fall all the way round and swing back up (cost 0.061). Given a starting guess that catches the pole, it plans to catch it (cost 0.081). Both are local optima, and under this cost the detour is the cheaper one, because the running cost only charges force and cart position. That's a planner doing exactly what I asked.

A short horizon could be rescued by a terminal cost that says how good the final state is, in other words a good estimate of $$V$$. So I wrote one by hand, using the pole's energy far from upright and LQR's $$s^\top P s$$ near it. It didn't help at 1 s, and at 1.5 s it made things worse: the plain cost holds and mine fails. Writing a value function by hand is hard, and that's what an RL critic learns from data. I now understand better why people combine the two.

## The G1: 36 positions, 35 velocities

Moving from two coordinates to the whole robot is where I got stuck for a while. Not on the maths of Jacobians so much as on how the pieces fit: a floating base, what MuJoCo gives you, and in which frame.

The first confusing thing: the G1 has **36** position numbers and **35** velocity numbers. The pelvis is free in space, so its orientation is stored as a quaternion, 4 numbers. But it can only turn in 3 directions, so its velocity is an angular velocity, 3 numbers. So this is wrong:

$$
q_{t+1} \ne q_t + \dot q\,\Delta t
$$

Each part has its own rule:

| Part | Position | Velocity | Update |
|---|---|---|---|
| pelvis position | `qpos[0:3]` | `qvel[0:3]`, world frame | add $$v\,\Delta t$$ |
| pelvis orientation | `qpos[3:7]`, quaternion | `qvel[3:6]`, body frame | $$\text{quat} \otimes \exp(\omega\,\Delta t)$$ |
| 29 joints | `qpos[7:]` | `qvel[6:]` | add $$\dot q\,\Delta t$$ |

MuJoCo has `mj_integratePos` to go from positions and velocities to new positions, and `mj_differentiatePos` for the reverse. Any "difference between two poses" needs the second one. That's also why the state for LQR below has 70 numbers (35 + 35) and not 71.

### What a Jacobian is

A Jacobian maps all the robot's velocities to the velocity of one point:

$$
v_\text{point} = J(q)\,\dot q
$$

Each column says what one joint does to that point. For a hinge with axis $$a$$ at position $$p_j$$, it's the lever-arm formula:

$$
J_i = a \times (p - p_j)
$$

Seeing the structure helped me most. A joint can only move a point if it lies between that point and the pelvis, so most of the matrix is exactly zero:

![Grid of which velocities move what: each foot depends on the base and its own leg only, while the centre of mass depends on everything]({{ '/assets/images/week-5/jacobian_structure.svg' | relative_url }}){: loading="lazy"}

The left foot depends on 12 of the 35 velocities: the 6 of the base and the 6 of its leg. The centre of mass (CoM) is different. It's a weighted average over all bodies, and so is its Jacobian:

$$
J_\text{com} = \frac{1}{M} \sum_i m_i J_i
$$

so all 35 velocities move it, some only barely. The waist and hip pitch joints move it most, about 6 cm per radian.

The same matrix also works in reverse. A force $$F$$ on a point needs these joint torques:

$$
\tau = J^\top F
$$

which is just force times lever arm, written for all joints at once. If each foot carries half the weight (163.5 N) with the knees bent, the knee sits 6.5 cm from the foot horizontally and needs 10.6 N·m. That's easy to check by hand, and it matches.

### What MuJoCo gives you

This is the map I wish I'd had at the start:

| You want | Call | Point used | Rows |
|---|---|---|---|
| Jacobian of a site (a foot) | `mj_jacSite` | `data.site_xpos` | 3 + 3 |
| ...of a body's frame | `mj_jacBody` | `data.xpos` | 3 + 3 |
| ...of a body's own CoM | `mj_jacBodyCom` | `data.xipos` | 3 + 3 |
| ...of any point on a body | `mj_jac` | your point | 3 + 3 |
| ...of the CoM of a whole subtree | `mj_jacSubtreeCom` | `data.subtree_com` | 3 |

A few things that tripped me up:

- **Jacobians aren't stored.** They're built on demand from what `mj_forward` already computed, so run `mj_forward` after changing `qpos`.
- **The linear and rotation rows come back separately**, in the world frame.
- **The first four calls are one function.** Each is `mj_jac` at a different point. It starts at the body's joint, walks up the tree towards the root, and fills one column per joint it meets. Every other column stays exactly zero, which is the structure in the picture above. `mj_jacSubtreeCom` adds up `mj_jac` over the bodies, weighted by mass.
- **For the whole robot's CoM,** use the pelvis as the subtree: `data.subtree_com[pelvis_id]` and `mj_jacSubtreeCom(..., pelvis_id)`.

```python
jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
mujoco.mj_jacSite(model, data, jacp, jacr, model.site("left_foot").id)   # foot: speed rows, turning rows

J_com = np.zeros((3, model.nv))
mujoco.mj_jacSubtreeCom(model, data, J_com, model.body("pelvis").id)     # centre of mass
```

I didn't want to take these on trust, so I checked them against finite differences: move the robot a tiny step along random velocities and measure how the foot and CoM moved. They agree to about 10⁻¹⁰ m/s. Those checks use only positions, so they don't depend on MuJoCo's Jacobian code.

## Null-space projection

With Jacobians, here's the first thing you can do: move the CoM around with both feet planted. The unknown is $$\dot q$$, all 35 velocities, and there are several demands on it:

- both feet stay still (12 rows);
- the pelvis doesn't turn (3 rows);
- the waist and arms stay still (17 rows);
- the CoM follows a target (3 rows).

The first three are hard requirements, stacked as $$C \dot q = b$$. They're 32 rows against 35 unknowns, so 3 directions are left free, and the CoM task can use those.

**Null-space projection** is the way to use them without disturbing the requirements. First, take the smallest velocities that meet the requirements:

$$
v_0 = C^+ b
$$

Next, find the directions you can add without breaking anything. Those are the **null space** of $$C$$, and this matrix projects any velocity onto them:

$$
N = I - C^+ C
$$

Since $$C N = 0$$, anything of the form $$N w$$ leaves $$C \dot q$$ unchanged. Finally, choose $$w$$ so the task is met as well as it can be:

$$
\dot q = v_0 + N \left( J_\text{com} N \right)^+ \left( \dot c_\text{target} - J_\text{com} v_0 \right)
$$

It looks heavier than it is. With only two joints and one requirement, you can draw it. Every velocity meeting the requirement lies on the blue line. $$v_0$$ is the point on it closest to no motion. The null space is the direction along the line, and you slide along it until the task is met too:

![Two-joint example: v0 is the closest point to zero on the requirement line; moving along that line reaches the point that also meets the task]({{ '/assets/images/week-5/null_space.svg' | relative_url }}){: loading="lazy"}

In code it's just as short:

```python
C_pinv = np.linalg.pinv(C)
v0 = C_pinv @ b                                  # meets the hard requirements
N = np.eye(model.nv) - C_pinv @ C                # directions that keep meeting them
task = target_dot + k * (target - com)
v = v0 + N @ np.linalg.pinv(J_com @ N) @ (task - J_com @ v0)
mujoco.mj_integratePos(model, q, v, dt)
```

(simplified; the full file is [`jacobians.py`](https://github.com/SakshayMahna/humanoid-walking-lab/blob/main/classical/01_dynamics_lqr_mpc/jacobians.py))

The white dot on the floor marks where the CoM is (a red ball marks the CoM itself, mostly hidden inside the pelvis). It goes down 8 cm, over the left foot, over the right foot and back up. The CoM stays within 0.16 mm of its target, and the feet move by at most 0.04 mm:

![The G1 squatting and shifting its weight between the feet, with the centre of mass marked]({{ '/assets/images/week-5/null_space_squat.gif' | relative_url }}){: loading="lazy"}

This is only kinematics. No physics runs here: the robot is posed, frame by frame, so nothing can fall. It shows where the joints should go. Balancing needs dynamics.

## LQR on one leg

The cart-pole had 4 states and 1 input. The G1 has 70 states (35 positions and 35 velocities) and 29 motors. The recipe is exactly the same, and I followed [DeepMind's MuJoCo LQR tutorial](https://github.com/google-deepmind/mujoco/blob/main/python/LQR.ipynb), which does this for a generic humanoid.

**1. A pose to balance around.** A small IK solve, built on the same Jacobians: standing foot flat, CoM over it, and pose-specific targets for the other foot, the torso and the arms. There are three poses, all on the left foot: flamingo, tree and warrior III.

**2. The torques that hold it still.** Set the accelerations to zero, and `mj_inverse` gives the forces needed. The six for the pelvis have no motor, so in a true equilibrium the floor must provide them. MuJoCo's contact is soft: the floor pushes harder the deeper a foot sinks. So I search for the height where the floor carries exactly the weight. The foot really has to be flat: tilt the whole flamingo by 0.34° and the pelvis would need 25 N·m from nowhere, because that small tilt puts the weight on one end of the foot.

**3. A linear model around it.** MuJoCo works this out by finite differences, contact included:

```python
mujoco.mj_forward(model, data)     # at the pose, with the holding torques
A, B = np.zeros((2 * nv, 2 * nv)), np.zeros((2 * nv, nu))
mujoco.mjd_transitionFD(model, data, 1e-6, True, A, B, None, None)
```

**4. A cost that says what "balanced" means.** A robot falls when its CoM leaves its foot, and Jacobians can express exactly that:

$$
J_\text{bal} = J_\text{com} - J_\text{foot}
$$

The cost charges this heavily, adds small costs for joints leaving the pose, and charges each torque relative to its motor's limit:

$$
Q = 1000\, J_\text{bal}^\top J_\text{bal} + \text{(small joint costs)}
$$

$$
R = \text{diag}\left( 1 / \tau_\text{limit}^2 \right)
$$

**5. Run it.**

$$
u = u_0 - K (s - s_0)
$$

$$K$$ is now a table of 29 × 70 numbers. The same `dlqr` loop from the cart-pole solves it, though slowly (about 100,000 iterations), and matches SciPy.

The details matter here. Without feedback, the holding torques alone fall over within 4 seconds. With a plain $$R = I$$ on raw N·m, torque looks far too expensive, the gains come out 7 times smaller, and the robot falls as well. And without the balance term, the robot survives only 6-8 N sideways instead of 40.

Each pose is held by its own gain. Here each one gets a 30 N sideways push for 0.1 s on the torso:

![Flamingo, tree and warrior III on one foot, each recovering from a 30 N sideways push]({{ '/assets/images/week-5/yoga_push.gif' | relative_url }}){: loading="lazy"}

| Largest 0.1 s push on the torso | Forward | Back | Left | Right |
|---|---|---|---|---|
| flamingo | 66 N | 108 N | 40 N | 41 N |
| tree | 67 N | 104 N | 43 N | 32 N |
| warrior III | 138 N | 55 N | 42 N | 36 N |

The directions are the way the push goes. Flamingo and tree take backward pushes best. Warrior III, leaning forward with its leg out behind, is the opposite: it takes 138 N forwards but only 55 N backwards.

To move between poses, LQR needs more than one linear model, since each is only true near its own pose. So the flow uses a chain: 61 poses from flamingo to tree to warrior III and back, each with its own equilibrium and its own gain. The reference slides along the chain, and the gains are blended between neighbours. This is called **gain scheduling**. In the clip at the top, the robot spends 17 seconds on one foot, and that foot moves by 0.4 mm.

Here is the same flow with and without scheduling. On the left the gains follow the chain. On the right, flamingo's gain is used throughout, and the robot falls about 11 seconds in, near warrior III (1.5x speed):

![The same yoga flow side by side: with blended gains the robot stays up; with flamingo's gain throughout it falls near warrior III]({{ '/assets/images/week-5/gain_schedule.gif' | relative_url }}){: loading="lazy"}

## What the controller is doing

A 70-state robot can still be analysed with the same tools as the cart-pole. This helped me understand what the numbers mean.

### The modes

Each eigenvalue of $$A$$ describes one **mode**: a pattern of joint motion that grows or shrinks by a fixed factor every step. On a 2 ms step, an eigenvalue of 1.0377 looks harmless, until you turn it into a rate:

$$
\text{rate} = \frac{\ln \lvert\lambda\rvert}{\Delta t} = 18.5 \text{ per second}
$$

That's a motion that doubles every 37 ms. The cart-pole's fall doubled every 137 ms, so the balancing G1 falls almost 4 times faster.

Without feedback, 12 of the flamingo's 70 modes grow. The fastest is the whole robot folding at the hips and waist. With LQR, none do:

![The flamingo's 70 modes in the complex plane: 12 grow without feedback, none with LQR]({{ '/assets/images/week-5/eigenvalues.svg' | relative_url }}){: loading="lazy"}

Each mode also comes with a shape, which says which joints move together. Here are the four fastest runaways without feedback:

| Growth rate | Doubles every | Joints that move most |
|---|---|---|
| 18.5 per second | 37 ms | standing hip pitch, waist pitch, free hip pitch, pelvis pitch |
| 16.7 per second | 42 ms | waist pitch, free hip pitch, pelvis pitch, standing hip pitch |
| 11.2 per second | 62 ms | left shoulder roll, waist roll, standing hip roll, left elbow |
| 9.4 per second | 74 ms | right shoulder roll, right wrist roll, right elbow, left shoulder roll |

The two fastest fold the robot at the waist and hips. The next two are mostly roll motion in the arms and waist. With LQR on, these are the slowest modes that remain:

| Time constant | Oscillates at | Damping ratio | Joints that move most |
|---|---|---|---|
| 0.28 s | 0.6 Hz | 0.72 | free hip pitch, free knee, right shoulder roll |
| 0.19 s | 0.8 Hz | 0.73 | free hip roll, left shoulder roll, free hip yaw |
| 0.17 s | 1.0 Hz | 0.70 | left shoulder pitch, right shoulder pitch, left shoulder roll |

The slowest modes are now in the free leg and the arms, which the cost charges least. A damping ratio of 0.7 means they settle with very little overshoot.

Before the controller could do this, the modes had to be within reach of the motors. A test of that (the [PBH test](https://en.wikipedia.org/wiki/Hautus_lemma)) confirms all 12 are controllable.

You can also watch a mode, by moving the robot back and forth along its shape. Here, exaggerated, are the fastest runaway, the slowest mode once LQR is on (time constant 0.28 s), and one that neither grows nor shrinks:

![Three mode shapes animated on the flamingo: the fastest runaway, the slowest decaying mode with LQR, and a neutral mode]({{ '/assets/images/week-5/modes.gif' | relative_url }}){: loading="lazy"}

Three modes stay at 1 even with LQR: where the robot stands on the floor (x and y) and which way it faces. The cost doesn't care about them, and the motors barely reach them. So the controller leaves them alone, and rightly so.

### The gains are a balance strategy

$$K$$ is too big to read directly, but you can ask it a question. Lean the CoM 1 cm in some direction, and see which torques come back:

| CoM leaning 1 cm | Largest torques |
|---|---|
| forward | standing ankle pitch +15.7 N·m, standing hip pitch −12.4, other hip pitch +8.8 |
| sideways | standing hip roll +21.6 N·m, waist yaw +12.9, standing ankle roll −11.6 |

So the controller pushes with the ankle, works the hip, and uses the free leg and waist to help. Nobody wrote that down. Humanoid textbooks call these the **ankle and hip strategies**, and here they come straight out of the Riccati equation.

### What the linear model doesn't know

The cart-pole's $$V(s)$$ predicted the cost of a run within 4%. So I expected $$V$$ to predict which pushes are dangerous on the G1 too. It doesn't:

| Direction | $$V$$ after a 10 N push | Largest push survived |
|---|---|---|
| forward | 0.50 | 66 N |
| back | 2.17 | 108 N |
| left | 0.94 | 40 N |
| right | 0.97 | 41 N |

A backward push looks the most expensive and turns out to be the safest. The correlation between predicted and measured is −0.50, the wrong sign. $$V$$ is the cost of recovering in the *linear* model, which has no edges. The robot actually falls at things that model doesn't contain, such as the foot rolling onto its edge, or a torque limit. A critic trained on real outcomes, falls included, would at least see those.

How far is the linear model right? Lean the CoM sideways, let go, and compare MuJoCo with the prediction:

![CoM recovery from 1, 2 and 4 cm leans: MuJoCo follows the linear prediction closely at 1 cm and less so at larger leans]({{ '/assets/images/week-5/linear_vs_mujoco.svg' | relative_url }}){: loading="lazy"}

At 1 cm, the largest gap is 12% of the lean. At 2 cm it's already 30%. The foot is only about 3 cm from its centre to its edge sideways, so a 4 cm lean starts with the CoM past that edge. The controller still recovers from it, well beyond the range where its model is accurate.

### Why the flow needs more than one gain

Does each pose really need its own gain? I tried each pose's gain on every other pose's linear model:

![Matrix of which pose's gain stabilises which pose's model: mostly stable, with warrior III's gain and the poses around it standing out]({{ '/assets/images/week-5/cross_stability.svg' | relative_url }}){: loading="lazy"}

75% of the pairs are stable. Gains are more tolerant than I expected, but warrior III is different enough that flamingo's gain can't hold it. Then I ran the actual flow with different schedules:

| Gains used during the flow | Falls? | Largest joint error |
|---|---|---|
| blended between all 61 poses | no | 0.116 rad |
| nearest of the 61 poses | no | 0.116 rad |
| only the 3 yoga poses' gains | no | 0.109 rad |
| flamingo's gain throughout | **yes** | 1.122 rad |

One gain isn't enough, but three are. The 61 poses are still needed for the reference and the holding torques, but not for the gains.

## What I learned

- **LQR and RL share a foundation.** The Riccati equation is a Bellman equation with an exact answer. Seeing $$V(s) = s^\top P s$$ gave me a concrete picture of what a critic network is trying to approximate.
- **Feedback does more than the plan.** Plans fail on small model errors without their gains, and MPC's warm start only works if it reuses them.
- **MPC's quality depends on its guess and its cost.** It found a strange but valid detour, and my hand-written value function made it worse.
- **The same recipe scales.** From 4 states to 70, the steps were the same. The hard part moved from the algebra to the details: a flat foot, a sensible cost, torque scaling.
- **Linear analysis explains a lot, but not everything.** The modes and gains made sense physically. But the value function couldn't predict which push would knock the robot over, because falling happens where the linear model stops being true.

## Next

- **Contact.** So far "balanced" meant "CoM over the foot". Next is what balance really requires of the contact forces: the centre of pressure and the zero-moment point, then the linear inverted pendulum model used to plan walking.

The main references for this week were Tedrake's [*Underactuated Robotics*](https://underactuated.csail.mit.edu/), [Tassa et al., 2012](https://roboti.us/lab/papers/TassaIROS12.pdf), and [DeepMind's LQR tutorial](https://github.com/google-deepmind/mujoco/blob/main/python/LQR.ipynb).
