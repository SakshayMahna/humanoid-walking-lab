"""Banner for the Week 4 blog post: the reward ladder as one picture.

Each robot is a real trained policy (lying, propped, sitting, shuffling, walking), simulated in plain MuJoCo with the
Week 1-3 look and composited side by side with segmentation masks.

Run from the repo root:  .venv/bin/python docs/_scripts/week4_banner.py [preview_dir]
Output: docs/assets/images/week-4/banner.png (and a mock preview with the post title in preview_dir)
"""

import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "docs" / "_scripts"))
sys.path.insert(0, str(REPO / "training"))
from week3_assets import BLUE, mujoco_look  # noqa: E402
from policy_io import load_trained  # noqa: E402
from unitree_utils import build_unitree_model  # noqa: E402

OUT = REPO / "docs" / "assets" / "images" / "week-4"
RUNS = REPO / "training" / "kaggle" / "output"
W, H = 1280, 400
# (policy run, time of the snapshot in s[, forward command]): steps 0-3 and the full set, told to walk after 1 s
# (0.5 m/s; the full set at 1.0 m/s for a longer stride)
LADDER = [
    ("v5/logs/rsl_rl/g1_velocity/2026-10-25_15-26-07_ladder_0_seed1", 9.0),
    ("v5/logs/rsl_rl/g1_velocity/2026-10-25_15-26-08_ladder_1_seed1", 9.0),
    ("v6/logs/rsl_rl/g1_velocity/2026-10-26_03-05-28_ladder_2_seed1", 9.0),
    ("v7/logs/rsl_rl/g1_velocity/2026-10-26_04-41-30_ladder_3_seed1", 7.0),
    ("v6/logs/rsl_rl/g1_velocity/2026-10-26_03-05-28_baseline_seed1", 7.0, 1.0),
]


def snapshot(run, t_snap, screen_x, vx=0.5, stride=False, distance=3.8, elevation=-4):
    """Image, robot mask and empty-floor background of one policy at t_snap, side-on; the robot is moved
    sideways in the image to screen_x (0..1)."""
    model, data, joints, acts = build_unitree_model(decorate=lambda spec: mujoco_look(spec, W, H))
    policy = load_trained(RUNS / run / "policy.onnx", joints, acts)
    mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
    mujoco.mj_forward(model, data)
    last = np.zeros(len(policy["default_pos"]))
    best = (-1.0, None)
    for step in range(round(t_snap / model.opt.timestep)):
        if step % 4 == 0:
            command = np.array([vx if data.time >= 1.0 else 0.0, 0.0, 0.0])
            obs = policy["obs_fn"](model, data, policy["default_pos"], last, command)
            last = policy["session"].run(None, {policy["input_name"]: obs[None, :]})[0][0]
            data.ctrl[:] = (policy["default_pos"] + policy["action_scale"] * last)[policy["joint_index_of_actuator"]]
        mujoco.mj_step(model, data)
        if stride and data.time > t_snap - 0.6:     # keep the pose with the widest stride in the last 0.6 s
            feet = [data.xpos[model.body(f"{side}_ankle_roll_link").id][:2] for side in ("left", "right")]
            gap = np.linalg.norm(feet[0] - feet[1])
            if gap > best[0]:
                best = (gap, data.qpos.copy())
    if stride:
        data.qpos[:] = best[1]
        mujoco.mj_forward(model, data)
    w, x, y, z = data.qpos[3:7]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))   # robot heading
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = distance, np.degrees(yaw) + 90, elevation   # side-on, robot centred
    cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.62]       # level, side-on, as in the Week 3 banner
    floor = model.geom("floor").id
    with mujoco.Renderer(model, height=H, width=W) as r:
        r.update_scene(data, camera=cam)
        img = np.asarray(r.render()).astype(float)
        r.enable_segmentation_rendering()
        seg = r.render()
        r.disable_segmentation_rendering()
        qpos = data.qpos.copy()
        data.qpos[0] += 100.0                               # empty floor, robot moved out of frame
        mujoco.mj_forward(model, data)
        cam.azimuth, cam.lookat[:] = 90.0, [0.5, 0.0, 0.62]  # one level background for all robots
        r.update_scene(data, camera=cam)
        background = np.asarray(r.render()).astype(float)
        data.qpos[:] = qpos
    mask = (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (seg[..., 0] != floor)
    # slide the robot sideways so that it appears at screen_x
    dx = round((screen_x - 0.5) * W)
    img, mask = np.roll(img, dx, axis=1), np.roll(mask, dx, axis=1)
    if dx > 0:
        mask[:, :dx] = False
    elif dx < 0:
        mask[:, dx:] = False
    return img, mask[..., None], background


def compose(shots, background, alphas):
    img = background.copy()
    for (snap, mask, _), a in zip(shots, alphas):
        img = np.where(mask, img * (1 - a) + snap * a, img)
    return img


def preview(img, path):
    """Mock of the theme's overlay header (dark filter + title)."""
    prev = Image.fromarray((img * 0.6).clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(prev)
    d.text((60, 140), "Week 4 - What Each Reward Teaches a Walking Robot", fill=(250, 250, 255),
           font=ImageFont.load_default(size=44))
    d.text((60, 205), "Training the Unitree G1 to walk with reward terms added and removed\n"
                      "one at a time, to see what each one really does.", fill=(225, 225, 235),
           font=ImageFont.load_default(size=19))
    prev.save(path)


def main(preview_dir=None):
    """The ladder as a line-up of ghosts, left to right (lying -> walking), on the right so the title stays readable."""
    OUT.mkdir(parents=True, exist_ok=True)
    xs = [0.52, 0.62, 0.72, 0.82, 0.92]
    shots = [snapshot(*entry[:2], x, *entry[2:], stride=k >= 3) for k, (entry, x) in enumerate(zip(LADDER, xs))]
    img = compose(shots, shots[-1][2], [0.5, 0.6, 0.7, 0.85, 1.0])   # ghosts fading in, the walker solid
    Image.fromarray(img.clip(0, 255).astype(np.uint8)).save(OUT / "banner.png")
    print("saved banner.png")
    if preview_dir:
        preview(img, Path(preview_dir) / "banner_week4_preview.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
