"""Header banner for the Week 1 post: the G1 next to a pendulum, with a glowing PD step response.

Run from the repo root:  .venv/bin/python docs/_scripts/week1_banner.py [preview_dir]
Writes docs/assets/images/week-1/banner.png (no text: the theme draws the title on top).
With preview_dir, also writes banner_preview.png there with a mock title overlay.
"""
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "assets" / "images" / "week-1"
PREVIEW_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else None
W, H = 1280, 400
KNEES_BENT = {"hip_pitch": -0.312, "knee": 0.669, "ankle_pitch": -0.363, "elbow": 0.6,
              "left_shoulder_roll": 0.2, "right_shoulder_roll": -0.2, "shoulder_pitch": 0.2}
YAW_TO_CAMERA = [np.cos(-np.pi / 4), 0, 0, np.sin(-np.pi / 4)]   # robot faces -y (towards the camera)


def build_scene(with_pendulum):
    spec = mujoco.MjSpec.from_file(str(REPO / "robot" / "g1.xml"))
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.30, 0.40, 0.85], rgb2=[0.03, 0.03, 0.09], width=512, height=512)
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.20, 0.24, 0.42], rgb2=[0.10, 0.12, 0.24], width=512, height=512)
    mat = spec.add_material(name="grid", texrepeat=[1, 1], texuniform=True, reflectance=0.2)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    w = spec.worldbody
    w.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[40, 40, 0.1], material="grid", contype=0, conaffinity=0)
    w.add_light(pos=[1.5, -3.0, 4.0], dir=[-0.3, 0.6, -1], diffuse=[0.9, 0.9, 0.95], castshadow=True)
    w.add_light(pos=[-3.0, -1.0, 2.0], dir=[0.6, 0.2, -0.5], diffuse=[0.30, 0.35, 0.70], castshadow=False)

    if with_pendulum:
        # A gantry at x = +1.6 with a 1 m pendulum swinging in the x-z plane (seen side-on by the camera)
        post = dict(type=mujoco.mjGEOM_CAPSULE if hasattr(mujoco, "mjGEOM_CAPSULE") else mujoco.mjtGeom.mjGEOM_CAPSULE,
                    size=[0.025, 0, 0], rgba=[0.55, 0.58, 0.72, 1], contype=0, conaffinity=0)
        px, top = 1.6, 1.55
        w.add_geom(fromto=[px - 0.7, 0, 0, px - 0.7, 0, top], **post)
        w.add_geom(fromto=[px + 0.7, 0, 0, px + 0.7, 0, top], **post)
        w.add_geom(fromto=[px - 0.7, 0, top, px + 0.7, 0, top], **post)
        body = w.add_body(name="pendulum", pos=[px, 0, top])
        body.add_joint(name="pendulum_hinge", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0])
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0, 0, 0, 0, 0, -1.0], size=[0.018, 0, 0],
                      rgba=[0.9, 0.9, 0.95, 1], contype=0, conaffinity=0)
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, pos=[0, 0, -1.0], size=[0.11, 0, 0],
                      rgba=[0.29, 0.80, 0.66, 1], contype=0, conaffinity=0)

    model = spec.compile()
    model.vis.global_.offwidth, model.vis.global_.offheight = W, H
    model.vis.headlight.ambient[:] = [0.28, 0.28, 0.33]
    model.vis.headlight.diffuse[:] = [0.30, 0.30, 0.30]
    model.vis.quality.shadowsize = 8192
    model.vis.map.zfar = 60.0          # far clip (x extent) so the floor reaches the horizon
    data = mujoco.MjData(model)

    # G1 pose: knees bent, standing, facing the camera
    for j in range(model.njnt):
        name = model.joint(j).name
        for key, val in KNEES_BENT.items():
            if key in name:
                data.qpos[model.jnt_qposadr[j]] = val
    data.qpos[0:3] = [0.0, 0.0, 0.76]
    data.qpos[3:7] = YAW_TO_CAMERA
    if with_pendulum:
        data.qpos[model.joint("pendulum_hinge").qposadr[0]] = 0.5
    mujoco.mj_forward(model, data)
    return model, data


def render(model, data, lookat, distance, azimuth, elevation):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    with mujoco.Renderer(model, height=H, width=W) as r:
        r.update_scene(data, camera=cam)
        return Image.fromarray(r.render())


def glow_line(img, pts, color, core):
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(layer).line(pts, fill=color + (170,), width=12)
    layer = layer.filter(ImageFilter.GaussianBlur(7))
    ImageDraw.Draw(layer).line(pts, fill=core + (255,), width=3)
    return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")


def preview(img, name):
    """Mock of the theme's overlay header: dark filter + title + excerpt, left aligned."""
    arr = np.asarray(img).astype(float) * 0.62          # overlay_filter ~0.4
    p = Image.fromarray(arr.astype(np.uint8))
    d = ImageDraw.Draw(p)
    d.text((70, 140), "Week 1 - What a Pendulum Taught Me\nAbout Humanoid Control", fill=(250, 250, 255), font_size=40)
    d.text((70, 250), "Loading the Unitree G1 in MuJoCo, and what a single pendulum taught me\n"
                      "about PD control, stability and actuators.", fill=(225, 225, 235), font_size=18)
    p.save(PREVIEW_DIR / f"{name}_preview.png")


def step_response(n=600, kp=20, kd=2, I=0.251, mgl=4.905, dt=0.005):
    q, v, out = 0.0, 0.0, []
    for _ in range(n):
        tau = kp * (0.5 - q) - kd * v - mgl * np.sin(q)
        v += tau / I * dt
        q += v * dt
        out.append(q)
    return np.array(out)


def draw_graph(img, qs, x0, x1, y_top, y_bot, q_max=0.65, target=0.5):
    """Glowing q(t) curve with axes, a dashed target line and labels."""
    def py(q): return y_bot - (y_bot - y_top) * q / q_max
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    # semi-transparent dark panel behind the graph so it reads as an overlay
    d.rounded_rectangle([x0 - 40, y_top - 38, x1 + 34, y_bot + 30], radius=12,
                        fill=(8, 10, 24, 150), outline=(120, 130, 180, 90), width=1)
    axis = (200, 210, 240, 200)
    d.line([(x0, y_bot), (x1 + 12, y_bot)], fill=axis, width=2)            # time axis
    d.polygon([(x1 + 12, y_bot - 5), (x1 + 22, y_bot), (x1 + 12, y_bot + 5)], fill=axis)
    d.line([(x0, y_bot), (x0, y_top - 12)], fill=axis, width=2)            # angle axis
    d.polygon([(x0 - 5, y_top - 12), (x0, y_top - 22), (x0 + 5, y_top - 12)], fill=axis)
    yt = py(target)
    for x in range(x0 + 4, x1, 14):                                         # dashed target line
        d.line([(x, yt), (x + 7, yt)], fill=(255, 206, 86, 220), width=2)
    d.text((x1 - 44, yt - 22), "target", fill=(255, 206, 86, 235), font_size=15)
    d.text((x0 - 26, y_top - 26), "q", fill=axis, font_size=17)
    d.text((x1 + 8, y_bot + 6), "t", fill=axis, font_size=17)

    pts = [(x0 + (x1 - x0) * i / (len(qs) - 1), py(q)) for i, q in enumerate(qs)]
    glow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(glow).line(pts, fill=(75, 192, 160, 90), width=6)      # soft, subtle glow
    glow = glow.filter(ImageFilter.GaussianBlur(3))
    ImageDraw.Draw(glow).line(pts, fill=(140, 235, 205, 255), width=3)
    out = Image.alpha_composite(img.convert("RGBA"), layer)
    return Image.alpha_composite(out, glow).convert("RGB")


model, data = build_scene(with_pendulum=True)
img = render(model, data, lookat=[-1.45, 0.0, 0.80], distance=3.7, azimuth=84, elevation=-3)
img = draw_graph(img, step_response(), x0=880, x1=W - 60, y_top=45, y_bot=150)
img.save(OUT / "banner.png")
print(f"saved {OUT / 'banner.png'}")
if PREVIEW_DIR:
    preview(img, "banner")
