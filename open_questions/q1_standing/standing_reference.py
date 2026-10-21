"""MuJoCo reference for Q1 test 1 (standing): Unitree's policy, zero command for 30 s.

Writes recordings/standing/stand_mujoco_unitree_1.csv (same format as the Gazebo logs) and
recordings/standing/stand_mujoco_unitree_1.mp4, filmed from behind-left like the Gazebo recordings.

    .venv/bin/python open_questions/q1_standing/standing_reference.py      (from the repo root; needs ffmpeg)
"""

import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "exercises" / "08_gazebo_reference"))
from mujoco_runs import REPO, run   # also puts exercises/ on sys.path
from render_utils import make_camera
from unitree_utils import build_unitree_model, load_unitree_policy

SECONDS = 30.0
WIDTH, HEIGHT, FPS = 640, 336, 25
OUT = REPO / "recordings" / "standing"


def record_video(model, data, policy, path):
    """Replay the same run (deterministic) and pipe rendered frames into ffmpeg."""
    from policy_utils import make_policy_controller
    mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
    mujoco.mj_forward(model, data)
    control = make_policy_controller(policy, [0.0, 0.0, 0.0])
    cam = make_camera(lookat=[0, 0, 0.6], distance=3.7, azimuth=45, elevation=-18)
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        stdin=subprocess.PIPE)
    steps_per_frame = round(1 / (FPS * model.opt.timestep))   # 200 Hz physics / 25 fps = 8
    with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
        for step in range(round(SECONDS / model.opt.timestep)):
            control(model, data)
            mujoco.mj_step(model, data)
            if step % steps_per_frame == 0:
                cam.lookat[:2] = data.qpos[0:2]          # follow the robot, like Gazebo's /gui/follow
                renderer.update_scene(data, camera=cam)
                ffmpeg.stdin.write(np.ascontiguousarray(renderer.render()).tobytes())
    ffmpeg.stdin.close()
    ffmpeg.wait()
    print(f"wrote {path}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    model, data, joints, actuators = build_unitree_model()
    policy = load_unitree_policy(joints, actuators)
    run("stand_mujoco_unitree_1", model, data, policy, seconds=SECONDS, stand=SECONDS, vx=0.0, out=OUT)
    record_video(model, data, load_unitree_policy(joints, actuators), OUT / "stand_mujoco_unitree_1.mp4")
