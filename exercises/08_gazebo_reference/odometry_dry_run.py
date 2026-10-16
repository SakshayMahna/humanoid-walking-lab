"""Dry run in MuJoCo: does walk.onnx care HOW its base_lin_vel input is measured?

In MuJoCo (training) base_lin_vel is a velocimeter: the exact velocity of the imu_in_pelvis site, in the
pelvis frame, at this instant. In Gazebo, Diego's node reads it from the OdometryPublisher plugin
(gz-sim8 OdometryPublisher.cc), which every 1 ms physics step takes the finite difference of the pose at
the same point and publishes the rolling mean of the last 10 = (p(t) - p(t - 10 ms)) / 10 ms, rotated into
the current pelvis frame. That is a 10 ms average, lagging about 5 ms.

Everything else stays MuJoCo, so any change comes from the velocity input alone.
Variants: velocimeter (training), Gazebo-style 10 ms average, 40 ms average (sensitivity), zero (no input).

    .venv/bin/python exercises/08_gazebo_reference/odometry_dry_run.py      (from the repo root)
    .venv/bin/python gazebo/summarize_runs.py recordings/mujoco/odom_*.csv
"""

from collections import deque

from mujoco_runs import REPO, run   # also puts exercises/ on sys.path
from g1_utils import build_g1_model
from observation_utils import make_observation
from policy_utils import DIEGO_POLICY, load_policy


def averaged_velocity_obs(window_s, dt):
    """obs_fn whose base_lin_vel is (p(t) - p(t - window)) / window of the imu site, in the pelvis frame.

    Returns (obs_fn, after_step); after_step records the site position after every physics step.
    """
    n = max(1, round(window_s / dt))
    history = deque(maxlen=n + 1)

    def after_step(model, data):
        history.append(data.site("imu_in_pelvis").xpos.copy())

    def obs_fn(model, data, default_pos, last_action, command):
        obs = make_observation(model, data, default_pos, last_action, command)
        if len(history) == n + 1:
            world_v = (history[-1] - history[0]) / (n * dt)
            R = data.site("imu_in_pelvis").xmat.reshape(3, 3)   # site frame = pelvis frame (no rotation)
            obs[0:3] = R.T @ world_v
        return obs

    return obs_fn, after_step


def zero_velocity_obs(model, data, default_pos, last_action, command):
    obs = make_observation(model, data, default_pos, last_action, command)
    obs[0:3] = 0.0
    return obs


if __name__ == "__main__":
    model, data, joints, actuators = build_g1_model(REPO / "robot" / "g1.xml")
    dt = model.opt.timestep
    # MuJoCo is deterministic: one run per variant is enough ("_1" keeps summarize_runs.py's naming)
    run("odom_velocimeter_1", model, data, load_policy(DIEGO_POLICY, joints, actuators))

    obs_fn, after_step = averaged_velocity_obs(0.010, dt)
    run("odom_gazebo10ms_1", model, data,
        load_policy(DIEGO_POLICY, joints, actuators, obs_fn=obs_fn), after_step=after_step)

    obs_fn, after_step = averaged_velocity_obs(0.040, dt)
    run("odom_avg40ms_1", model, data,
        load_policy(DIEGO_POLICY, joints, actuators, obs_fn=obs_fn), after_step=after_step)

    run("odom_zero_1", model, data, load_policy(DIEGO_POLICY, joints, actuators, obs_fn=zero_velocity_obs))
