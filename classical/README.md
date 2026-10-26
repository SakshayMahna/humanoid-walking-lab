# Classical control for humanoid walking

A short course in model-based humanoid control: enough to make the Unitree G1 balance, recover from pushes
and walk without a neural network, and to understand what learned controllers are doing.

**Why, given that the state of the art is learned.** Loco-manipulation at the frontier is a vision-language-
action model on top of an RL whole-body controller trained on human motion (Figure's Helix 02, the Boston
Dynamics / TRI Large Behavior Model on Atlas). Classical control is not how those systems are built, but it is
inside them: reward and observation design, retargeting human motion to the robot, the actuator and PD layer,
state estimation, teleoperation for data collection, and explaining why a policy fails. This course covers
those parts and leaves out the rest.

The starting point was Johannson, *Control for Humanoid Locomotion: From Pendulum Models to MPC with
Centroidal Dynamics* (TU Darmstadt seminar).

## How the course works

- **Derive, implement, check.** Each module's `notes.md` holds the derivation; the code is plain NumPy plus
  MuJoCo; every lab ends with a "done when" check against an analytic answer or a reference library.
- **Not rewritten:** the simulator (MuJoCo) and QP/NLP solvers (OSQP through `qpsolvers`, CasADi + IPOPT).
- **Toy model first, G1 second.** Each idea runs on the smallest system that shows it.
- **TODO mechanism.** Each lab script has `TODO` functions (raising `NotImplementedError`) and given checks.
  Running it executes the parts in order and stops at the first unfinished TODO, so every finished part is
  re-checked on each run. Outputs go to the module's `results/` (not tracked). Extra packages:
  `.venv/bin/pip install scipy matplotlib`.
- **Every module ends with a "for learning" link:** a reward term, an observation, or a tool from that
  module, used in Track 1.

## Modules

| # | Module | Ends with | Link to learning |
|---|---|---|---|
| 1 | Dynamics, LQR and MPC | iLQR-MPC swing-up of a cart-pole | value functions, MPC vs. policies, sampling MPC |
| 2 | Contact, ZMP and the LIPM | ZMP preview control for a footstep plan | what "balanced" means; gait rewards |
| 3 | Capture point and push recovery | stepping to recover from pushes | push-robustness metrics, foot-placement rewards |
| 4 | Whole-body IK and retargeting | G1 retargeted from human motion data | motion data for tracking policies |
| 5 | The G1 walks | LIPM/DCM planner + IK walking in MuJoCo | a classical baseline for the Track 1 benchmarks |
| 6 | State estimation | walking on estimated base velocity | the base-velocity observation for policies |

Optional, if one catches your interest: torque-based whole-body control (inverse-dynamics QP), single-rigid-
body MPC, whole-body DDP with Crocoddyl, limit cycles and hybrid zero dynamics. Reading for each is at the end.

## Core reading

- R. Tedrake, *Underactuated Robotics* (MIT 6.832 course notes): <https://underactuated.csail.mit.edu/>.
  Cited as "UR ch. N".
- S. Kajita, H. Hirukawa, K. Harada, K. Yokoi, *Introduction to Humanoid Robotics*, Springer STAR 101, 2014.
  [doi:10.1007/978-3-642-54536-8](https://doi.org/10.1007/978-3-642-54536-8). Cited as "Kajita book".

---

## Module 1: Dynamics, LQR and MPC

**Why.** All model-based control rests on $M(q)\ddot q + h(q,\dot q) = S^\top\tau + J_c^\top\lambda$. LQR is the
one case where the optimal policy and value function are exact, which makes it the cleanest bridge to RL. MPC
is the model-based alternative to a trained policy.

**Theory.** Equations of motion, floating base (6 unactuated coordinates), Jacobians; linearization and
stability; LQR from dynamic programming (the Riccati equation is a Bellman equation); trajectory optimization
by shooting and collocation; iLQR; MPC as re-planning from the current state.

**Lab.**
1. Cart-pole in MuJoCo: derive the dynamics, check them against MuJoCo, LQR balance.
   *Done when:* your Riccati solution matches `scipy.linalg.solve_discrete_are`.
2. Swing-up with your own iLQR, then iLQR as MPC with a wrong pole mass and pushes.
3. G1: CoM, CoM Jacobian, foot Jacobians from MuJoCo, checked by finite differences.

**For learning.** Compare LQR's value function with a critic learned by PPO on the same cart-pole. Try
[MuJoCo MPC](https://github.com/google-deepmind/mujoco_mpc), sampling-based MPC that sits between
optimization and RL.

**Reading.** UR ch. 3, 7, 8, 10 (including "Iterative LQR and Differential Dynamic Programming");
Y. Tassa, T. Erez, E. Todorov, "Synthesis and stabilization of complex behaviors through online trajectory
optimization", IROS 2012 ([PDF](https://www.cs.washington.edu/homes/todorov/courses/amath533/Online.pdf)).

## Module 2: Contact, ZMP and the LIPM

**Why.** Flat-footed balance comes down to one condition: the contact forces must exist. Feet only push, they
don't slip, and the CoP stays inside the foot. The LIPM turns that into a linear model you can plan with.

**Theory.** Contact wrenches and friction cones; CoP and ZMP, and why they are equal on flat ground; support
polygon. LIPM derivation and closed-form solution; the cart-table model; ZMP preview control.

**Lab.**
1. G1 standing and swaying: CoP from contact forces vs. ZMP from CoM motion.
   *Done when:* the two agree, and a push that tips the robot drives the ZMP to the edge of the foot first.
2. 3D LIPM: CoM trajectory for a footstep plan with preview control (Kajita 2003).
   *Done when:* the ZMP tracks the stepped reference, with the CoM moving before each step as in the paper.
3. Optional: the same plan as a ZMP-MPC QP with OSQP (Wieber 2006).

**For learning.** Log the ZMP and CoP of the Track 1 policies. Do learned gaits keep the ZMP inside the
foot, or do they rely on dynamic falling?

**Reading.**
- Kajita book ch. 3-4; UR ch. 5 ("Center of pressure (CoP) and Zero-moment point (ZMP)", "ZMP-based planning").
- S. Kajita et al., "Biped walking pattern generation by using preview control of zero-moment point", ICRA
  2003. [doi:10.1109/ROBOT.2003.1241826](https://doi.org/10.1109/ROBOT.2003.1241826),
  [PDF](https://people.csail.mit.edu/katiebyl/kb/DW2008/papers_of_tangential_interest/kajita03.pdf)
- P.-B. Wieber, "Trajectory free linear model predictive control for stable walking in the presence of strong
  perturbations", Humanoids 2006. [PDF](https://www.cs.cmu.edu/~cga/ew/wieber-humanoids.pdf)

## Module 3: Capture point and push recovery

**Why.** It gives an exact answer to "where must I step to stop?". It's the best mental model for push
robustness, and the basis for foot-placement rewards and robustness metrics.

**Theory.** DCM / capture point (the unstable part of the LIPM); capture region and N-step capturability;
DCM tracking; balance strategies: ankle (CoP), hip (angular momentum, LIPM + flywheel) and stepping.

**Lab.**
1. LIPM pushes: compute the capture region; step into it.
   *Done when:* pushes inside the one-step capture region stop in one step, larger ones need more, as
   predicted.
2. LIPM + flywheel: how much the hip strategy enlarges the capture region.

**For learning.** Measure capture-point error on Track 1 policies under pushes. Do they step where capture
point theory says they should?

**Reading.**
- J. Pratt, J. Carff, S. Drakunov, A. Goswami, "Capture Point: A step toward humanoid push recovery",
  Humanoids 2006, pp. 200-207.
- T. Koolen et al., "Capturability-based analysis and control of legged locomotion, Part 1", *IJRR*
  31(9):1094-1113, 2012. [doi:10.1177/0278364912452673](https://doi.org/10.1177/0278364912452673)
- J. Englsberger, C. Ott, A. Albu-Schäffer, "Three-dimensional bipedal walking control based on divergent
  component of motion", *IEEE T-RO* 31(2):355-368, 2015. [doi:10.1109/TRO.2015.2405592](https://doi.org/10.1109/TRO.2015.2405592)

## Module 4: Whole-body IK and retargeting

**Why.** The motion-tracking controllers under today's foundation models are trained on human motion
retargeted to the robot, and retargeting is whole-body inverse kinematics. The same IK solver drives the
walker in Module 5 and teleoperation.

**Theory.** Differential IK; damped least squares; tasks and null spaces; IK as a QP with joint limits and
weighted tasks; retargeting: scaling human keypoints, foot contact constraints, avoiding foot skating.

**Lab.**
1. QP IK on the G1: move the CoM and hands with feet fixed.
   *Done when:* results match [pink](https://github.com/pink-kinematics/pink) for the same tasks and weights.
2. Retarget a human motion clip (e.g. from AMASS) to the G1 and replay it kinematically.
   *Done when:* feet don't slide during contact and the CoM stays over the support polygon in static poses.

**For learning.** The retargeted clip is training data for a motion-tracking policy, the controller type
used under VLA models.

**Reading.** K. Lynch, F. Park, *Modern Robotics*, Cambridge University Press, 2017, ch. 5-6;
S. Caron, A. Kheddar, O. Tempier, "Stair climbing stabilization of the HRP-4 humanoid robot using whole-body
admittance control", ICRA 2019 ([arXiv:1809.07073](https://arxiv.org/abs/1809.07073)) for how IK and
admittance control run a real position-controlled humanoid.

## Module 5: The G1 walks

**Why.** The most satisfying milestone: the G1 walking with no network, built only from Modules 2-4. It also
gives a classical baseline for the Track 1 benchmarks.

**Theory.** Gait state machine (double and single support); footstep planning from a velocity command;
swing-foot trajectories; the planner → IK → position-control pipeline (how ZMP-based humanoids such as HRP
robots walk on position control).

**Lab.**
1. Footsteps from a velocity command → LIPM/DCM CoM plan → swing feet → IK → the G1's existing position
   actuators.
2. Velocity commands with the same interface as the RL policies; push recovery with step adjustment.
   *Done when:* the walker runs through the Track 1 evaluation report next to the learned policies.

**For learning.** Side-by-side: classical vs. learned gait on tracking, effort, smoothness and pushes.

**Reading.** Kajita book ch. 4; Englsberger 2015; [lipm_walking_controller](https://github.com/jrl-umi3218/lipm_walking_controller)
(Caron et al. 2019) as a complete reference design.

## Module 6: State estimation

**Why.** Policies and controllers in simulation read the true base velocity; a real robot has to estimate it
from an IMU and leg kinematics. This decides whether a policy that uses base velocity can be deployed.

**Theory.** Kalman filter and EKF; IMU process model; stance foot as a zero-velocity measurement; contact
detection; what cannot be observed (absolute position, yaw).

**Lab.** Simulated IMU and encoder noise; your own EKF; Module 5 walker on estimated state.
*Done when:* it still walks; report estimation error vs. ground truth.

**For learning.** Run a Track 1 policy that uses base linear velocity (the `walk.onnx` design) on your
estimate instead of ground truth.

**Reading.**
- M. Bloesch et al., "State estimation for legged robots: Consistent fusion of leg kinematics and IMU", RSS
  2012. [doi:10.15607/RSS.2012.VIII.003](https://doi.org/10.15607/RSS.2012.VIII.003)
- N. Rotella et al., "State estimation for a humanoid robot", IROS 2014. [arXiv:1402.5450](https://arxiv.org/abs/1402.5450)

---

## Optional deep dives

- **Torque-based whole-body control** (inverse-dynamics QP with contact forces; needs a torque-actuated G1):
  S. Feng et al., "Optimization-based full body control for the DARPA Robotics Challenge", *J. Field
  Robotics* 32(2):293-312, 2015; A. Herzog et al., "Momentum control with hierarchical inverse dynamics on a
  torque-controlled humanoid", *Autonomous Robots* 40(3):473-491, 2016 ([arXiv:1410.7284](https://arxiv.org/abs/1410.7284));
  [TSID](https://github.com/stack-of-tasks/tsid).
- **Centroidal dynamics and single-rigid-body MPC** (survey §IV-A): D. Orin, A. Goswami, S.-H. Lee,
  "Centroidal dynamics of a humanoid robot", *Autonomous Robots* 35(2-3):161-176, 2013; J. Di Carlo et al.,
  "Dynamic locomotion in the MIT Cheetah 3 through convex model-predictive control", IROS 2018
  ([PDF](https://dspace.mit.edu/handle/1721.1/138000)).
- **Whole-body MPC with DDP** (survey §V-A): C. Mastalli et al., "Crocoddyl: An efficient and versatile
  framework for multi-contact optimal control", ICRA 2020 ([arXiv:1909.04947](https://arxiv.org/abs/1909.04947)).
- **Limit cycles and hybrid zero dynamics:** UR ch. 4 and 16; E. Westervelt, J. Grizzle, D. Koditschek,
  "Hybrid zero dynamics of planar biped walkers", *IEEE TAC* 48(1):42-56, 2003
  ([PDF](https://grizzle.robotics.umich.edu/files/hybrid_zdyn_TAC.pdf)).
- **Kino-dynamic planning** (survey §IV-B): H. Dai, A. Valenzuela, R. Tedrake, "Whole-body motion planning
  with centroidal dynamics and full kinematics", Humanoids 2014
  ([PDF](https://groups.csail.mit.edu/robotics-center/public_papers/Dai14.pdf)).

## Layout

```
classical/
  README.md         this syllabus
  common/           shared code (contact/CoP utilities, IK, plotting)
  01_dynamics_lqr_mpc/
  02_zmp_lipm/
  ...               one folder per module: notes.md (derivations), code, results
```
