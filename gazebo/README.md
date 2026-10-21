# Gazebo experiments

Scripts to run the walking policies on the G1 in Gazebo Harmonic (ROS 2 Jazzy), log what happens, record videos
and compare against MuJoCo. Everything runs in the Docker container defined in [`docker/`](../docker/).
Results and explanations: [notes/gazebo_experiments.md](../notes/gazebo_experiments.md).

## The container

[`docker/Dockerfile`](../docker/Dockerfile) builds on `tiryoh/ros2-desktop-vnc:jazzy` (Ubuntu 24.04, ROS 2 Jazzy,
`ros_gz` with Gazebo Harmonic, a VNC desktop) and adds ONNX Runtime 1.17.1 (C++), picking the build that matches
the machine (aarch64 on Apple Silicon, x64 otherwise).

[`docker/compose.yaml`](../docker/compose.yaml) mounts:

| Host | Container | Why |
|---|---|---|
| `3p/2025-tfg-diego-lopez` | `~/ros2_ws/src/2025-tfg-diego-lopez` | Diego's packages: `g1_sim` (robot, worlds, bridge) and `g1_locomotion` (his node) |
| `ros2/` | `~/ros2_ws/src/lab` | Our package, `g1_policy_runner` |
| repo root | `~/humanoid-walking-lab` | Scripts, policies in `3p/`, and `recordings/` for outputs |

```bash
docker compose -f docker/compose.yaml up -d --build      # start (build and install stay inside the container)
docker compose -f docker/compose.yaml exec -u ubuntu ros bash   # shell
docker compose -f docker/compose.yaml down               # stop
```

The VNC desktop is at http://localhost:6080 (password `ubuntu`), but none of the scripts need it.
The scripts build the ROS 2 packages on first use (`colcon build`, about 15 s).

## Scripts

| Script | Where it runs | What it does |
|---|---|---|
| [`run_scenario.sh`](run_scenario.sh) | container | Headless run (no GUI) of one scenario → `recordings/NAME.csv` |
| [`scenario_logger.py`](scenario_logger.py) | container | Sends the command on **simulation time** (zero for STAND s, then the command) and logs pelvis odometry at 20 Hz |
| [`record.sh`](record.sh) | Mac | Starts the container and runs [`record_inside.sh`](record_inside.sh): same scenario with the Gazebo GUI, screen-recorded → `NAME.mp4`, `NAME.gif`, `NAME.csv` |
| [`gui_record.config`](gui_record.config) | container | Gazebo GUI layout for recording: only the 3D view and the sim-time / RTF overlay |
| [`make_gain_variants.py`](make_gain_variants.py) | Mac | Writes copies of Diego's robot SDF with other joint controllers (`mj_gains`, `mj_gains_vel_d` = corrected, `luis_gains_vel_d`) or with armature added to the link inertias (`original_armature`) → `recordings/gz_models/` |
| [`gain_experiment.sh`](gain_experiment.sh) | container | All controller variants × both policies × repeats → `recordings/gains/` |
| [`summarize_runs.py`](summarize_runs.py) | Mac | Metrics table from any set of CSVs (Gazebo or MuJoCo) |
| [`side_by_side.py`](side_by_side.py) | Mac | Labelled side-by-side or grid videos from recordings, with each run's result from its CSV |

Without launch arguments, `run_scenario.sh` and `record.sh` run Diego's own node (`g1_locomotion`, which loads
`walk_free_elbows.onnx`). With them, they run [`g1_policy_runner`](../ros2/g1_policy_runner/).

## Examples

```bash
# Headless: stand 4 s, then vx = 0.5 m/s, 24 s in total, Unitree's policy, wall-clock timer
docker compose -f docker/compose.yaml exec -u ubuntu ros bash ~/humanoid-walking-lab/gazebo/run_scenario.sh \
    0.5 0 0 24 4 unitree_fwd "policy:=unitree clock:=wall"

# Same scenario with video (run on the Mac)
gazebo/record.sh 0.5 0 0 24 4 walk_fwd "policy:=walk clock:=sim"

# Controller experiment: Luis's gains / policy gains / policy gains with damping on joint velocity
.venv/bin/python gazebo/make_gain_variants.py
docker compose -f docker/compose.yaml exec -u ubuntu ros bash ~/humanoid-walking-lab/gazebo/gain_experiment.sh 2 wall

# Metrics (MuJoCo references: exercises/08_gazebo_reference/mujoco_runs.py)
.venv/bin/python gazebo/summarize_runs.py recordings/gains/*.csv recordings/mujoco/mujoco_*.csv
```

Arguments of both run scripts: `VX VY WZ SECONDS STAND_SECONDS NAME ["launch arguments"]`. Launch arguments are
those of `g1_policy_runner`'s `policy.launch.py` (`policy`, `clock`, `policy_period`, `robot_model`, ...).
`record.sh` also reads `FPS` (capture, default 30) and `GIF_FPS` (default 25).

## Metrics

`summarize_runs.py` groups runs by name (dropping the trailing `_<repeat>`) and reports mean ± std of:

| Metric | Meaning |
|---|---|
| `stand_rot`, `stand_move` | Rotation (deg) and displacement (m) while the command is zero |
| `vx`, `vy` | Mean pelvis-frame velocity from 4 s after the command starts (m/s) |
| `turn` | Slope of the heading over that window (deg/s) |
| `wander` | Std of the heading around that slope (deg) |
| `falls` | Runs where the pelvis dropped below 0.4 m |

## Good to know

- On an Apple Silicon Mac, Gazebo runs at a real-time factor of about 0.9. The GUI renders on the CPU and draws
  only about 10 distinct frames per second, so videos look choppy whatever the capture rate.
- With `clock:=wall` (Diego's default) the policy runs every 20 ms of real time, so at RTF 0.9 it acts about 10%
  more often in simulated time than in training. `clock:=sim` follows Gazebo's `/clock`.
- The scripts stop any previous Gazebo, bridge or policy processes first: a leftover bridge forwards every
  message twice.
