"""Generate the images and GIFs for the Week 1 blog post.

Run from the repo root:  .venv/bin/python docs/_scripts/week1_assets.py
Outputs go to docs/assets/images/week-1/. Physics matches exercises/02_pendulum_pd;
only colours, lighting and layout are added for the blog.
"""

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "assets" / "images" / "week-1"
OUT.mkdir(parents=True, exist_ok=True)

L, TARGET = 0.5, 0.5
BG = (18, 20, 33)            # panel background (dark navy)
FG = (235, 235, 245)
PALETTE = [(255, 99, 132), (54, 162, 235), (255, 206, 86), (75, 192, 160), (153, 102, 255), (255, 159, 64)]


def font(size):
    return ImageFont.load_default(size=size)


# ---------------------------------------------------------------------------
# Pendulum (same physics as exercises/02_pendulum_pd/pendulum.py, plus visuals)
# ---------------------------------------------------------------------------

def pendulum_xml(color, actuator="", integrator="Euler", armature=0.0):
    r, g, b = (c / 255 for c in color)
    return f"""
    <mujoco>
        <option timestep="0.005" integrator="{integrator}"/>
        <visual><headlight ambient="0.35 0.35 0.35" diffuse="0.6 0.6 0.6"/></visual>
        <asset>
            <texture name="sky" type="skybox" builtin="gradient" rgb1="0.10 0.11 0.20" rgb2="0.05 0.05 0.09" width="64" height="64"/>
            <texture name="grid" type="2d" builtin="checker" rgb1="0.16 0.18 0.30" rgb2="0.10 0.11 0.20" width="128" height="128"/>
            <material name="grid" texture="grid" texrepeat="8 8"/>
        </asset>
        <worldbody>
            <light pos="0.6 -1.5 1.5" dir="-0.3 1 -1" diffuse="0.8 0.8 0.8"/>
            <geom name="floor" type="plane" pos="0 0 -0.8" size="3 3 0.1" material="grid" contype="0" conaffinity="0"/>
            <geom name="pivot" type="sphere" size="0.025" rgba="0.9 0.9 0.95 1" contype="0" conaffinity="0"/>
            <body name="pole">
                <joint name="hinge" type="hinge" axis="0 1 0" armature="{armature}"/>
                <geom name="rod" type="capsule" fromto="0 0 0 0 0 {-L}" size="0.012" density="0" rgba="0.85 0.85 0.9 1"/>
                <geom name="bob" type="sphere" pos="0 0 {-L}" size="0.05" mass="1" rgba="{r} {g} {b} 1"/>
            </body>
        </worldbody>
        <actuator>{actuator}</actuator>
    </mujoco>
    """


def make_pd(kp, kd, target=TARGET):
    def control(model, data):
        data.ctrl[0] = kp * (target - data.qpos[0]) - kd * data.qvel[0]
    return control


BAD = (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL, mujoco.mjtWarning.mjWARN_BADQACC)


def render_panel_frames(color, label, actuator, control=None, ctrl=None, integrator="Euler",
                        armature=0.0, duration=6.0, fps=15, w=320, h=190, plot_h=110, y_min=0.0, y_max=0.9, show_target=True):
    """Simulate one pendulum and return a list of panel images (render + live q(t) plot)."""
    model = mujoco.MjModel.from_xml_string(pendulum_xml(color, actuator, integrator, armature))
    data = mujoco.MjData(model)
    if ctrl is not None:
        data.ctrl[0] = ctrl
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, -0.28]
    cam.distance, cam.azimuth, cam.elevation = 1.35, 90, -8

    dt = model.opt.timestep
    n_steps = round(duration / dt)
    ts, qs, frames = [], [], []
    diverged_at = None
    top = None
    with mujoco.Renderer(model, height=h, width=w) as renderer:
        for i in range(n_steps):
            if diverged_at is None:
                if control is not None:
                    control(model, data)
                mujoco.mj_step(model, data)
                if any(data.warning[b].number for b in BAD) or abs(data.qpos[0]) > np.pi:
                    diverged_at = (i + 1) * dt
                else:
                    ts.append((i + 1) * dt)
                    qs.append(data.qpos[0])
            if len(frames) < (i + 1) * dt * fps:
                if diverged_at is None or top is None:
                    renderer.update_scene(data, camera=cam)
                    top = Image.fromarray(renderer.render())
                # after divergence `top` keeps the last good render (frozen, without overlays)
                frames.append(compose_panel(top, ts, qs, color, label, duration, plot_h, y_min, y_max, diverged_at, show_target))
    return frames


def compose_panel(top, ts, qs, color, label, duration, plot_h, y_min, y_max, diverged_at, show_target=True):
    w, h = top.size
    panel = Image.new("RGB", (w, h + plot_h), BG)
    panel.paste(top, (0, 0))
    d = ImageDraw.Draw(panel)
    d.text((8, 6), label, fill=FG, font=font(14))

    # plot area
    x0, x1, y0, y1 = 34, w - 8, h + 10, h + plot_h - 18
    def px(t): return x0 + (x1 - x0) * t / duration
    def py(q): return y1 - (y1 - y0) * (q - y_min) / (y_max - y_min)
    d.rectangle((x0, y0, x1, y1), outline=(70, 75, 100))
    yt = py(TARGET) if show_target else None
    if show_target:
        for x in range(x0, x1, 8):                      # dashed target line
            d.line((x, yt, x + 4, yt), fill=(200, 200, 210))
        d.text((4, yt - 7), "0.5", fill=(200, 200, 210), font=font(11))
    if y_min < 0:
        d.line((x0, py(0), x1, py(0)), fill=(70, 75, 100))   # zero axis when negatives are shown
    if not show_target or abs(py(0) - yt) > 12:         # skip "0" when it would overlap "0.5"
        d.text((4, py(0) - 7), "0", fill=(150, 150, 170), font=font(11))
    if not show_target:
        d.text((4, py(y_max) - 2), f"+{y_max:g}", fill=(150, 150, 170), font=font(11))
        d.text((4, py(y_min) - 12), f"{y_min:g}", fill=(150, 150, 170), font=font(11))
    d.text((x1 - 70, y1 + 3), f"time 0-{duration:g} s", fill=(150, 150, 170), font=font(11))
    if len(qs) > 1:
        pts = [(px(t), py(min(max(q, y_min - 0.05), y_max))) for t, q in zip(ts, qs)]
        d.line(pts, fill=color, width=2)
    if diverged_at is not None:
        d.text((w // 2 - 55, h // 2 - 10), "DIVERGED", fill=(255, 80, 80), font=font(22))
    return panel


def save_grid(panel_lists, path, cols, title, fps=15):  # fps < capture fps -> slow motion
    """Tile per-panel frame lists into a grid GIF with a title bar."""
    n = max(len(p) for p in panel_lists)
    pw, ph = panel_lists[0][0].size
    rows = (len(panel_lists) + cols - 1) // cols
    gap, title_h = 6, 34
    W, H = cols * pw + (cols + 1) * gap, rows * ph + (rows + 1) * gap + title_h
    frames = []
    for i in range(n):
        canvas = Image.new("RGB", (W, H), (10, 11, 20))
        ImageDraw.Draw(canvas).text((gap + 4, 8), title, fill=FG, font=font(18))
        for k, plist in enumerate(panel_lists):
            r, c = divmod(k, cols)
            canvas.paste(plist[min(i, len(plist) - 1)], (gap + c * (pw + gap), title_h + gap + r * (ph + gap)))
        frames.append(canvas.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=1000 // fps, loop=0, optimize=True)
    print(f"saved {path.name}: {len(frames)} frames, {path.stat().st_size / 1e6:.2f} MB")


def zeta(kp, kd, armature=0.0):
    I = L**2 + 0.4 * 0.05**2 + armature
    q_eq = kp * TARGET / (kp + 9.81 * L)
    k = kp + 9.81 * L * np.cos(q_eq)
    return kd / (2 * np.sqrt(k * I))


MOTOR = '<motor joint="hinge"/>'


def pd_grid(name, gains, title, duration=6.0):
    panels = [render_panel_frames(PALETTE[i], f"KP={kp:g}  KD={kd:g}  ζ={zeta(kp, kd):.2f}".replace("ζ", "zeta"),
                                  MOTOR, control=make_pd(kp, kd), duration=duration)
              for i, (kp, kd) in enumerate(gains)]
    save_grid(panels, OUT / name, cols=2, title=title)


# ---------------------------------------------------------------------------
# G1 humanoid: hero image and free-fall GIF with a colourful scene
# ---------------------------------------------------------------------------

def g1_model(with_floor):
    spec = mujoco.MjSpec.from_file(str(REPO / "robot" / "g1.xml"))
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.35, 0.45, 0.85], rgb2=[0.05, 0.05, 0.12], width=256, height=256)
    if with_floor:
        spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                         rgb1=[0.20, 0.24, 0.42], rgb2=[0.12, 0.14, 0.26], width=256, height=256)
        mat = spec.add_material(name="grid", texrepeat=[6, 6])
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[4, 4, 0.1], material="grid",
                                contype=0, conaffinity=0)
    spec.worldbody.add_light(pos=[1.5, -1.5, 3.0], dir=[-0.4, 0.4, -1], diffuse=[0.9, 0.9, 0.9])
    model = spec.compile()
    model.vis.headlight.ambient[:] = [0.35, 0.35, 0.35]
    return model


def g1_assets():
    # Hero image: G1 in its default (all-zero joint) pose above a floor
    model = g1_model(with_floor=True)
    data = mujoco.MjData(model)
    data.qpos[2] += 0.05  # zero pose has feet slightly below z=0; lift for the picture
    mujoco.mj_forward(model, data)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, 0.6]
    cam.distance, cam.azimuth, cam.elevation = 2.4, 140, -12
    with mujoco.Renderer(model, height=480, width=640) as r:  # default offscreen buffer is 640x480
        r.update_scene(data, camera=cam)
        Image.fromarray(r.render()).save(OUT / "g1.png")
    print("saved g1.png")

    # Free fall (no floor), 4x slow motion with a time label
    model = g1_model(with_floor=False)
    data = mujoco.MjData(model)
    cam.lookat[:] = [0, 0, 0.2]
    cam.distance, cam.azimuth, cam.elevation = 4.0, 135, 0
    frames, capture_fps, playback_fps = [], 120, 30
    with mujoco.Renderer(model, height=360, width=480) as r:
        while data.time < 0.5:
            mujoco.mj_step(model, data)
            if len(frames) < data.time * capture_fps:
                r.update_scene(data, camera=cam)
                img = Image.fromarray(r.render())
                d = ImageDraw.Draw(img)
                d.text((10, 8), f"free fall   t = {data.time:.2f} s   (4x slow motion)", fill=FG, font=font(16))
                d.text((10, 30), f"pelvis z = {data.qpos[2]:+.3f} m", fill=FG, font=font(16))
                frames.append(img.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
    frames[0].save(OUT / "free_fall.gif", save_all=True, append_images=frames[1:],
                   duration=1000 // playback_fps, loop=0, optimize=True)
    print(f"saved free_fall.gif: {len(frames)} frames, {(OUT / 'free_fall.gif').stat().st_size / 1e6:.2f} MB")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    g1_assets()

    # Single pendulum swing (no actuator), starting at 0.5 rad
    def start_at(model, data, _done=[False]):
        if not _done[0]:
            data.qpos[0] = 0.5
            _done[0] = True
    swing = render_panel_frames(PALETTE[1], "free swing, no actuator", "", control=start_at,
                                duration=4.0, w=480, h=300, plot_h=120, y_min=-0.6, y_max=0.6, show_target=False)
    save_grid([swing], OUT / "pendulum_swing.gif", cols=1, title="Pendulum: period check")

    pd_grid("pd_stiffness.gif", [(5, 2), (20, 2), (80, 2), (320, 2)],
            "Stiffness sweep: KD = 2, KP grows -> smaller error, more overshoot")
    pd_grid("pd_damping.gif", [(20, 0.5), (20, 2), (20, 5), (20, 20)],
            "Damping sweep: KP = 20, KD grows -> less overshoot, then slower")

    # Gravity compensation: add the gravity torque m*g*L*sin(q) to the PD output
    def pd_gravity(kp, kd, m=1.0, g=9.81):
        def control(model, data):
            q = data.qpos[0]
            data.ctrl[0] = kp * (TARGET - q) - kd * data.qvel[0] + m * g * L * np.sin(q)
        return control
    panels = [
        render_panel_frames(PALETTE[1], "PD only", MOTOR, control=make_pd(20, 2), duration=3.0),
        render_panel_frames(PALETTE[3], "PD + gravity compensation", MOTOR, control=pd_gravity(20, 2), duration=3.0),
    ]
    save_grid(panels, OUT / "gravity_comp.gif", cols=2, title="KP = 20, KD = 2: adding the gravity torque removes the error")

    # Integrator vs stability: KD = 102 is just beyond the Euler limit (~100.4), so the
    # blow-up builds over ~1.4 s and is visible (KD = 150 explodes in < 0.1 s).
    kd = 102
    wide = dict(y_min=-3.2, y_max=3.2)  # show the full swing until |q| > pi
    pos = f'<position joint="hinge" kp="20" kv="{kd}"/>'
    panels = [
        render_panel_frames(PALETTE[0], "manual PD, Euler", MOTOR, control=make_pd(20, kd), duration=3.0, **wide),
        render_panel_frames(PALETTE[2], "built-in, Euler", pos, ctrl=TARGET, duration=3.0, **wide),
        render_panel_frames(PALETTE[3], "built-in, implicitfast", pos, ctrl=TARGET, integrator="implicitfast",
                            duration=3.0, **wide),
    ]
    save_grid(panels, OUT / "integrator.gif", cols=3, fps=8,
              title=f"KD = {kd}, just above the Euler limit of ~100 (half speed)")

    # Torque limit vs none
    pos = '<position joint="hinge" kp="20" kv="2"/>'
    lim = '<position joint="hinge" kp="20" kv="2" forcelimited="true" forcerange="-1 1"/>'
    panels = [
        render_panel_frames(PALETTE[1], "no torque limit", pos, ctrl=TARGET),
        render_panel_frames(PALETTE[0], "torque limit +/-1 N·m", lim, ctrl=TARGET),
    ]
    save_grid(panels, OUT / "torque_limit.gif", cols=2, title="Torque limit: saturation removes the damping")

    # Armature vs none
    panels = [
        render_panel_frames(PALETTE[3], f"armature 0    zeta={zeta(20, 2):.2f}", pos, ctrl=TARGET, duration=3.0),
        render_panel_frames(PALETTE[4], f"armature 0.1  zeta={zeta(20, 2, 0.1):.2f}", pos, ctrl=TARGET,
                            armature=0.1, duration=3.0),
    ]
    save_grid(panels, OUT / "armature.gif", cols=2, title="Armature: more inertia -> slower, more overshoot")
