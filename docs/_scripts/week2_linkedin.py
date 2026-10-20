"""LinkedIn GIF (1080x1350): two G1 walking policies side by side on four commands.

Run from the repo root:  .venv/bin/python docs/_scripts/week2_linkedin.py
Output: docs/assets/videos/week-2/policy_compare.gif
Policies are shown as "Policy 1" (walk.onnx) and "Policy 2" (Unitree's).
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from week2_assets import BG, BLUE, DIM, FG, GREEN, REPO, font  # noqa: E402

sys.path.insert(0, str(REPO / "exercises"))
import mujoco  # noqa: E402
from g1_utils import build_g1_model  # noqa: E402
from policy_utils import DIEGO_POLICY, load_policy, make_policy_controller  # noqa: E402
from render_utils import make_camera  # noqa: E402
from unitree_utils import build_unitree_model, load_unitree_policy  # noqa: E402

OUT = REPO / "docs" / "assets" / "videos" / "week-2"
OUT.mkdir(parents=True, exist_ok=True)

W, H, FPS, SECONDS = 1080, 1350, 10, 5.0
HEADER_H, ROW_TITLE_H, GAP = 130, 34, 6
COMMANDS = [("Walk forward", "0.6 m/s", [0.6, 0, 0]),
            ("Walk slowly", "0.3 m/s", [0.3, 0, 0]),
            ("Turn in place", "0.8 rad/s", [0, 0, 0.8]),
            ("Step sideways", "0.3 m/s", [0, 0.3, 0])]
POLICIES = [("Policy 1", BLUE), ("Policy 2", GREEN)]


def scenery(spec):
    """Visual only, same look as the blog GIFs. Physics unchanged."""
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.30, 0.40, 0.85], rgb2=[0.03, 0.03, 0.09], width=256, height=256)
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.20, 0.24, 0.42], rgb2=[0.12, 0.14, 0.27], width=256, height=256)
    mat = spec.add_material(name="grid", texrepeat=[2, 2], texuniform=True, reflectance=0.15)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    spec.geom("floor").material = "grid"
    spec.worldbody.add_light(pos=[2, -2, 4], dir=[-0.4, 0.4, -1], diffuse=[0.8, 0.8, 0.85], castshadow=True)
    spec.visual.headlight.ambient = [0.3, 0.3, 0.35]
    spec.visual.headlight.diffuse = [0.35, 0.35, 0.35]


def yaw(data):
    w, x, y, z = data.qpos[3:7]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def run(model, data, policy, cmd, turning, color, w, h):
    """One run with the camera following the pelvis and a live moved/turned readout."""
    mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
    mujoco.mj_forward(model, data)
    control = make_policy_controller(policy, np.array(cmd, dtype=float))
    cam = make_camera(lookat=[0, 0, 0.6], distance=2.4, azimuth=125, elevation=-12)
    start, turned, last_yaw = data.qpos[:2].copy(), 0.0, yaw(data)
    frames = []
    with mujoco.Renderer(model, height=h, width=w) as r:
        for i in range(round(SECONDS / model.opt.timestep)):
            control(model, data)
            mujoco.mj_step(model, data)
            turned += (yaw(data) - last_yaw + np.pi) % (2 * np.pi) - np.pi    # unwrap
            last_yaw = yaw(data)
            if len(frames) < data.time * FPS:
                cam.lookat[:2] = data.qpos[:2]
                r.update_scene(data, camera=cam)
                img = Image.fromarray(r.render())
                text = f"turned {np.degrees(abs(turned)):.0f}°" if turning \
                    else f"moved {np.linalg.norm(data.qpos[:2] - start):.2f} m"
                dr = ImageDraw.Draw(img)
                dr.rectangle((8, h - 44, 8 + dr.textlength(text, font=font(26)) + 16, h - 8), fill=BG)
                dr.text((16, h - 40), text, fill=color, font=font(26))
                frames.append(img)
    return frames


def banner(frame, path):
    """Same style as the Week 1 thumbnail: dimmed final frame with a bold title band."""
    img = Image.blend(frame, Image.new("RGB", (W, H), (10, 11, 20)), 0.35)
    d = ImageDraw.Draw(img)
    y0, y1 = H // 2 - 170, H // 2 + 170
    d.rectangle((0, y0, W, y1), fill=(10, 11, 20))
    d.rectangle((0, y0, W, y0 + 6), fill=POLICIES[0][1])
    d.rectangle((0, y1 - 6, W, y1), fill=POLICIES[1][1])
    y = y0 + 40
    for text, size, color in [("Two walking policies", 84, FG), ("Same robot. Same commands.", 46, (255, 206, 86)),
                              ("Unitree G1 in MuJoCo", 36, (170, 175, 200))]:
        tw = d.textlength(text, font=font(size))
        d.text(((W - tw) / 2, y), text, fill=color, font=font(size))
        y += size + 30
    img.save(path)
    print(f"saved {path.relative_to(REPO)}")


if __name__ == "__main__":
    d_model, d_data, d_joints, d_acts = build_g1_model(REPO / "robot" / "g1.xml", decorate=scenery)
    u_model, u_data, u_joints, u_acts = build_unitree_model(decorate=scenery)
    setups = [(d_model, d_data, load_policy(DIEGO_POLICY, d_joints, d_acts)),
              (u_model, u_data, load_unitree_policy(u_joints, u_acts))]

    rows = len(COMMANDS)
    pw = (W - 3 * GAP) // 2
    row_h = (H - HEADER_H - (rows + 1) * GAP) // rows
    ph = row_h - ROW_TITLE_H
    grid = [[run(m, d, p, cmd, cmd[2] != 0, POLICIES[c][1], pw, ph) for c, (m, d, p) in enumerate(setups)]
            for _, _, cmd in COMMANDS]

    canvases = []
    for i in range(min(len(r) for row in grid for r in row)):
        canvas = Image.new("RGB", (W, H), (10, 11, 20))
        dr = ImageDraw.Draw(canvas)
        dr.text((24, 18), "Same robot, same commands, two policies", fill=FG, font=font(40))
        for c, (name, color) in enumerate(POLICIES):
            x = GAP + c * (pw + GAP)
            dr.rectangle((x, 80, x + pw, 122), fill=BG)
            dr.rectangle((x, 118, x + pw, 122), fill=color)
            tw = dr.textlength(name, font=font(30))
            dr.text((x + (pw - tw) / 2, 84), name, fill=color, font=font(30))
        for r, (title, value, _) in enumerate(COMMANDS):
            y = HEADER_H + GAP + r * (row_h + GAP)
            dr.text((GAP + 10, y + 4), title, fill=FG, font=font(26))
            dr.text((GAP + 18 + dr.textlength(title, font=font(26)), y + 6), value, fill=DIM, font=font(24))
            for c in range(2):
                canvas.paste(grid[r][c][i], (GAP + c * (pw + GAP), y + ROW_TITLE_H))
        canvases.append(canvas)

    # One shared palette from a few frames, with the accent colours pinned exactly so they survive
    sample = Image.new("RGB", (W, H * 3))
    for k, idx in enumerate((0, len(canvases) // 2, len(canvases) - 1)):
        sample.paste(canvases[idx], (0, k * H))
    n = 78
    colors = sample.quantize(colors=n, method=Image.Quantize.MEDIANCUT).getpalette()[:n * 3]
    colors += [v for _, color in POLICIES for v in color]
    palette = Image.new("P", (1, 1))
    palette.putpalette(colors + [0] * (768 - len(colors)))
    frames = [c.quantize(palette=palette, dither=Image.Dither.NONE) for c in canvases]

    banner(canvases[-1], OUT / "banner.png")

    path = OUT / "policy_compare.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=1000 // FPS, loop=0, optimize=True)
    print(f"saved {path.relative_to(REPO)}: {len(frames)} frames, {path.stat().st_size / 1e6:.2f} MB")
