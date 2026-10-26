"""Video of trained G1 policies in MuJoCo next to Unitree's policy, following the same sequence of commands.

Same simulation as evaluate_policy.py (Unitree's model, 5 ms physics, policy every 20 ms). Each tile shows the
current command; the camera follows the robot. Tiles are joined with gazebo/side_by_side.py.

    .venv/bin/python training/render_policy.py training/kaggle/output/v4/logs/rsl_rl/g1_velocity/*/policy.onnx
Output: recordings/training/mujoco_<label>.mp4 and recordings/training/mujoco_compare.mp4
"""

import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "exercises"))
sys.path.insert(0, str(REPO / "open_questions" / "q1b_controller"))
from policy_utils import load_policy  # noqa: E402
from render_utils import make_camera  # noqa: E402
from render_videos import scenery  # noqa: E402  (large grid floor, visual only)
from unitree_utils import HOME, UNITREE_POLICY, build_unitree_model, make_unitree_observation  # noqa: E402

OUT = REPO / "recordings" / "training"
WIDTH, HEIGHT, FPS = 640, 360, 25   # 25 fps = one frame every 8 physics steps (exact real time)
DECIMATION = 4
# (seconds, [vx, vy, wz], label)
SEQUENCE = [(3, (0.0, 0.0, 0.0), "stand"), (5, (0.5, 0.0, 0.0), "forward 0.5 m/s"),
            (4, (1.0, 0.0, 0.0), "forward 1.0 m/s"), (4, (0.0, 0.3, 0.0), "sideways 0.3 m/s"),
            (5, (0.0, 0.0, 0.5), "turn 0.5 rad/s"), (3, (0.2, 0.0, 0.0), "slow 0.2 m/s"),
            (3, (0.0, 0.0, 0.0), "stand")]


def command_at(t):
    for seconds, command, label in SEQUENCE:
        if t < seconds:
            return np.array(command), label
        t -= seconds
    return np.zeros(3), "stand"


def render(model, data, policy, path):
    mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    cam = make_camera(lookat=[0, 0, 0.7], distance=3.2, azimuth=135, elevation=-15)
    font = ImageFont.load_default(size=22)
    ffmpeg = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                               "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                               "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
    last_action = np.zeros(len(policy["default_pos"]))
    every = round(1 / (FPS * model.opt.timestep))
    total = sum(s for s, _, _ in SEQUENCE)
    for step in range(round(total / model.opt.timestep)):
        command, label = command_at(data.time)
        if step % DECIMATION == 0:
            obs = policy["obs_fn"](model, data, policy["default_pos"], last_action, command)
            action = policy["session"].run(None, {policy["input_name"]: obs[None, :]})[0][0]
            data.ctrl[:] = (policy["default_pos"] + policy["action_scale"] * action)[policy["joint_index_of_actuator"]]
            last_action = action
        mujoco.mj_step(model, data)
        if step % every == 0:
            cam.lookat[:2] = data.qpos[:2]
            renderer.update_scene(data, camera=cam)
            frame = Image.fromarray(renderer.render())
            ImageDraw.Draw(frame).text((12, HEIGHT - 34), f"command: {label}", fill="yellow", font=font)
            ffmpeg.stdin.write(np.asarray(frame).tobytes())
    ffmpeg.stdin.close()
    ffmpeg.wait()
    renderer.close()
    return total


def main(paths):
    OUT.mkdir(parents=True, exist_ok=True)
    model, data, joints, actuators = build_unitree_model(decorate=scenery)
    policies = [("Unitree's policy", "unitree", UNITREE_POLICY)]
    for p in paths:
        run = Path(p).parent.name                                   # <date>_<time>_<variant>_seed<n>
        name = run.split("_", 2)[2] if run.count("_") >= 2 else run
        policies.append((name, run, Path(p)))
    tiles = []
    for title, label, path in policies:
        policy = load_policy(path, joints, actuators, pose=HOME, obs_fn=make_unitree_observation)
        video = OUT / f"mujoco_{label}.mp4"
        seconds = render(model, data, policy, video)
        tiles.append(f"{title}|{video}")
        print("rendered", video.name)
    subprocess.run([sys.executable, str(REPO / "gazebo" / "side_by_side.py"), str(OUT / "mujoco_compare.mp4"),
                    "--title", "MuJoCo (plain, CPU): Unitree's released policy vs our trained runs, same commands",
                    "--seconds", str(seconds), *tiles], check=True)


if __name__ == "__main__":
    main(sys.argv[1:])
