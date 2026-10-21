"""Videos of representative runs from mujoco_controller_test.py, as labelled 2x2 grids (standing and walking).

Same simulation as the experiment (it calls its run() and films it), so the result labels match its CSVs.
Output: recordings/q1b/video_<config>_<scenario>.mp4 and recordings/q1b/grid_{stand,walk}.mp4

    .venv/bin/python open_questions/q1b_controller/render_videos.py      (from the repo root; needs ffmpeg)
"""

import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mujoco_controller_test import DT, OUT, REPO, SCENARIOS, run  # noqa: E402
from render_utils import make_camera  # noqa: E402  (exercises/ is on sys.path via mujoco_controller_test)
from unitree_utils import build_unitree_model, load_unitree_policy  # noqa: E402

WIDTH, HEIGHT, FPS = 640, 336, 25

# (label, armature, gains, limits, D term)
CONFIGS = [
    ("1. Like training: armature, D on velocity", "on", "policy", "policy", "vel"),
    ("2. Original controller, no armature", "off", "luis", "luis", "err"),
    ("3. Original, with a 50 N.m ankle limit", "off", "luis", "policy", "err"),
    ("4. Corrected controller, no armature", "off", "policy", "policy", "jnt"),
]


class VideoWriter:
    """on_step hook: renders a frame every 1/FPS s of simulation into an ffmpeg pipe, camera following."""

    def __init__(self, model, path):
        self.renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
        self.cam = make_camera(lookat=[0, 0, 0.6], distance=3.7, azimuth=45, elevation=-18)
        self.every = round(1 / (FPS * DT))
        self.step = 0
        self.ffmpeg = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}",
             "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)

    def __call__(self, model, data):
        if self.step % self.every == 0:
            self.cam.lookat[:2] = data.qpos[0:2]
            self.renderer.update_scene(data, camera=self.cam)
            self.ffmpeg.stdin.write(np.ascontiguousarray(self.renderer.render()).tobytes())
        self.step += 1

    def close(self):
        self.ffmpeg.stdin.close()
        self.ffmpeg.wait()
        self.renderer.close()


def scenery(spec):
    """Visual only: a large grid floor, so the robot stays on visible ground while walking.
    (A MuJoCo plane collides as an infinite plane whatever its size, so physics is unchanged.)"""
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.16, 0.17, 0.24], rgb2=[0.11, 0.12, 0.18], mark=mujoco.mjtMark.mjMARK_EDGE,
                     markrgb=[0.35, 0.37, 0.5], width=512, height=512)
    spec.add_material(name="grid", textures=["", "grid"], texrepeat=[1, 1], texuniform=True)
    floor = spec.geom("floor")
    floor.size = [50, 50, 0.1]
    floor.material = "grid"


def main():
    model, data, joints, actuators = build_unitree_model(decorate=scenery)
    model.opt.timestep = DT
    model.actuator_gainprm[:, 0] = 0.0
    model.actuator_biasprm[:, 0:3] = 0.0
    armature = model.dof_armature.copy()
    policy = load_unitree_policy(joints, actuators)

    for scenario in SCENARIOS:
        tiles = []
        for label, arm, gains, limits, dterm in CONFIGS:
            model.dof_armature[:] = armature if arm == "on" else 0.0
            video = OUT / f"video_arm{arm}_{gains}gains_{limits}limits_{dterm}_{scenario}.mp4"
            writer = VideoWriter(model, video)
            name = run(model, data, policy, joints, gains, limits, dterm, arm, scenario, on_step=writer)
            writer.close()
            tiles.append(f"{label}|{video}|{OUT / name}.csv")
            print("rendered", video.name)

        seconds = SCENARIOS[scenario][0]
        what = "zero command for 30 s" if scenario == "stand" else "stand 4 s, then walk at 0.5 m/s"
        subprocess.run([sys.executable, str(REPO / "gazebo" / "side_by_side.py"), str(OUT / f"grid_{scenario}.mp4"),
                        "--title", f"Unitree policy in MuJoCo, 1 ms steps: joint controllers compared ({what})",
                        "--seconds", str(seconds), "--cols", "2", *tiles], check=True)


if __name__ == "__main__":
    main()
