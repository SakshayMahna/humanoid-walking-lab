# Plan

**Goal (by 2026-10-15):** make the Unitree G1 walk in plain MuJoCo using the
trained ONNX policy from the reference repo, without mjlab. Every piece
(model, actuators, observation, control loop) is built and understood here.

Reference: `3p/2025-tfg-diego-lopez/pruebas/G1/walk_fixed.ipynb`.

## Days 1–2: MuJoCo basics

- [x] **Day 1 — Ex 1: Inspect the G1 model.**
  MjModel vs MjData, `nq`/`nv`, `jnt_qposadr`/`jnt_dofadr`, free joint,
  quaternion convention, sensor layout. Open it in the passive viewer
  (`mjpython` on macOS).
  *Check:* asserts on sizes; free fall matches `z0 - ½gt²`.

- [x] **Day 2 — Ex 2: Pendulum + PD control.**
  Write a one-hinge MJCF. Control it with your own PD (torque motor),
  then with MuJoCo's built-in position actuator (`gainprm`/`biasprm`).
  Armature and force limits.
  *Check:* both controllers give matching trajectories; explain the
  steady-state error under gravity.

## Days 3–7: Reproduce G1 walking

- [x] **Day 3 — Ex 3: Make the G1 simulatable.**
  Add floor, 29 position actuators (gains from the G1 constants), and an
  `init_state` keyframe. Verify joint order vs actuator order.
  *Check:* 29 actuators with gains matching deploy.yaml, keyframe loads the pose.
  (Finding: with the soft policy gains the robot can't stand on its own; the policy balances it.)

- [ ] **Day 4 — Ex 4: Observations and frames.**
  IMU sensors (site frame), free-joint `qvel` frames, projected gravity.
  *Check:* known tilt gives the analytic projected gravity.

- [ ] **Day 5 — Ex 5: Control loop + policy.**
  200 Hz physics, 50 Hz policy (decimation 4), action → joint targets,
  99-dim observation, ONNX inference.
  *Check:* zero-action policy holds pose; real policy walks.

- [ ] **Day 6 — Validate.**
  Forward / backward / turning commands; zero command stands still.
  *Check:* 0.6 m/s command → ~3 m in 6 s without falling.

- [ ] **Day 7 — Buffer, blog post, cleanup.**

## Later phases

- [ ] **Replicate the ROS 2 + Gazebo deployment** from Diego's repo
  (`g1_sim`, `g1_locomotion`): same policy file, 50 Hz node, `/cmd_vel` interface.
  Needs Ubuntu 24.04 + ROS 2 Jazzy + Gazebo Harmonic (Docker or Linux machine).
- [ ] **Build on it with Unitree's policy** (`3p/unitree_rl_mjlab`, `deploy.yaml`):
  98-number observation with gait phase, no linear velocity. Compare both policies
  in MuJoCo and Gazebo.

## Known risks

- Local `g1_constants.py` may not match the mjlab config the policy was
  trained with (actuator gains, default pose). Resolve on Day 3, possibly
  by reading mjlab's G1 config.
- Day 5 needs `onnxruntime`.
