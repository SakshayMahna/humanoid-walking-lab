# g1_policy_runner

A ROS 2 (Jazzy, C++) node that runs an mjlab-exported velocity policy on the Unitree G1 in Gazebo, using the
topics of Diego's `g1_sim`. The same node runs Diego's `walk.onnx` and Unitree's `policy.onnx`.

Everything policy-specific is read from the ONNX file's metadata, so nothing is hard-coded per policy:

| Metadata key | Used for |
|---|---|
| `joint_names` | Policy joint order (Gazebo's `/joint_states` order differs; joints are matched by name) |
| `default_joint_pos` | Joint positions are observed relative to it; targets are built around it |
| `action_scale` | Target = default + scale × action |
| `observation_names` | Which terms make up the observation, and in which order |

Input and output sizes come from the network itself (99 → 29 for `walk.onnx`, 98 → 29 for Unitree's).
The node refuses policies with fewer actions than joints (e.g. Diego's 27-action `walk_free_elbows.onnx`).

## Interface

| Topic | Type | Direction | Used for |
|---|---|---|---|
| `/cmd_vel` | `geometry_msgs/Twist` | in | Command: `linear.x`, `linear.y`, `angular.z` |
| `/g1/joint_states` | `sensor_msgs/JointState` | in | Joint positions and velocities |
| `/g1/imu/pelvis` | `sensor_msgs/Imu` | in | Angular velocity; orientation → projected gravity |
| `/g1/pelvis/odometry` | `nav_msgs/Odometry` | in | Linear velocity (only `walk.onnx` uses it) |
| `/g1/cmd_pos/<joint>` | `std_msgs/Float64` | out | One position target per joint, every policy step |

Supported observation terms: `base_lin_vel`, `base_ang_vel`, `projected_gravity`, `command`, `phase`
(Unitree's gait clock), `joint_pos`, `joint_vel`, `actions` (previous raw network output).

| Parameter | Default | Meaning |
|---|---|---|
| `policy_file` | (required) | Path to the `.onnx` file |
| `clock` | `wall` | `wall`: timer on real time (Diego's node). `sim`: timer on the node clock, i.e. Gazebo's `/clock` with `use_sim_time` |
| `policy_period` | `0.02` | Seconds between policy steps (training: 0.02, 50 Hz) |
| `gait_period` | `0.6` | Stride period of Unitree's gait clock (s) |

## Running

Inside the container (see [`gazebo/README.md`](../../gazebo/README.md)):

```bash
cd ~/ros2_ws && colcon build --packages-select g1_policy_runner && source install/setup.bash
ros2 launch g1_policy_runner policy.launch.py policy:=unitree clock:=sim
ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.5}}"
```

The launch file starts the node, then Gazebo with Diego's `g1_sim/sim_core.py` 2 s later (the same order as his
launch file), so both policies get exactly the same robot, world and bridge.

| Launch argument | Default | |
|---|---|---|
| `policy` | `walk` | Loads [`config/<policy>.yaml`](config/) (`walk` or `unitree`) |
| `clock` | `wall` | `wall` or `sim` |
| `policy_period` | `0.02` | |
| `gui` | `true` | Gazebo GUI |
| `world` | `empty` | `g1_sim` world alias |
| `robot_model` | `g1_free_roam` | `g1_sim` robot alias, or a path to a `model.sdf` (e.g. a controller variant) |
| `spawn_z` | `0.80` | Spawn height (m) |

The policy paths in [`config/`](config/) point into `~/humanoid-walking-lab/3p/`, i.e. the container layout.

## Implementation notes

- ONNX Runtime with one intra-op thread: the network is tiny (about 0.2 M multiply-adds, ≈ 17 µs per call),
  and more threads cost CPU that Gazebo needs.
- Actions are not clipped, and the raw action (not the target) is fed back as `actions`, as in training.
- The gait clock uses t = steps × `policy_period`: identical to training at 0.02 s.
- Callbacks only store the latest values; the timer builds the observation from them (single-threaded executor,
  so no locking is needed).

Code: [`src/policy_runner.cpp`](src/policy_runner.cpp), [`include/g1_policy_runner/policy_runner.hpp`](include/g1_policy_runner/policy_runner.hpp).
