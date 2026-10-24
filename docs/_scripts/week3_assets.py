"""Images and GIFs for the Week 3 blog post (MuJoCo -> Gazebo).

Run from the repo root:  .venv/bin/python docs/_scripts/week3_assets.py [banner] [preview=DIR]
Outputs go to docs/assets/images/week-3/.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "assets" / "images" / "week-3"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "exercises"))

BG = (18, 20, 33)
FG = (235, 235, 245)
DIM = (150, 155, 180)
BLUE, RED, YELLOW, GREEN, PURPLE = (54, 162, 235), (255, 99, 132), (255, 206, 86), (75, 192, 160), (153, 102, 255)


def font(size):
    return ImageFont.load_default(size=size)


# ---------------------------------------------------------------------------
# Scenery (visual only; physics unchanged)
# ---------------------------------------------------------------------------

def mujoco_look(spec, width, height):
    """Week 1-2 look: navy gradient sky, blue checker floor."""
    import mujoco
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.30, 0.40, 0.85], rgb2=[0.03, 0.03, 0.09], width=512, height=512)
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.20, 0.24, 0.42], rgb2=[0.11, 0.13, 0.25], width=512, height=512)
    _floor_and_light(spec, width, height, reflectance=0.2)


def gazebo_look(spec, width, height):
    """Styled after Gazebo's default world: light grey sky, near-black ground with a thin white grid."""
    import mujoco
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
                     rgb1=[0.85, 0.85, 0.86], rgb2=[0.85, 0.85, 0.86], width=64, height=64)
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.04, 0.04, 0.045], rgb2=[0.04, 0.04, 0.045], mark=mujoco.mjtMark.mjMARK_EDGE,
                     markrgb=[0.85, 0.85, 0.88], width=160, height=160)
    _floor_and_light(spec, width, height, reflectance=0.0)


def _floor_and_light(spec, width, height, reflectance):
    import mujoco
    mat = spec.add_material(name="grid", texrepeat=[2, 2], texuniform=True, reflectance=reflectance)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    spec.geom("floor").material = "grid"
    spec.geom("floor").size = [40, 40, 0.1]
    spec.worldbody.add_light(pos=[1.5, -3, 4], dir=[-0.3, 0.6, -1], diffuse=[0.85, 0.85, 0.9], castshadow=True)
    spec.visual.headlight.ambient = [0.28, 0.28, 0.33]
    spec.visual.headlight.diffuse = [0.3, 0.3, 0.3]
    spec.visual.global_.offwidth, spec.visual.global_.offheight = width, height
    spec.visual.map.zfar = 60.0
    spec.visual.quality.shadowsize = 8192


# ---------------------------------------------------------------------------
# Banner: one walk (stroboscopic ghosts) that passes from MuJoCo's world into a Gazebo-style world
# ---------------------------------------------------------------------------

def strobe(look, W, H, snap_times, command):
    """Ghost-trail image of Unitree's policy walking, rendered with the given scenery."""
    import mujoco
    from policy_utils import make_policy_controller
    from unitree_utils import build_unitree_model, load_unitree_policy

    model, data, joints, acts = build_unitree_model(decorate=lambda spec: look(spec, W, H))
    control = make_policy_controller(load_unitree_policy(joints, acts), command)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.15, 0.0, 0.62]
    cam.distance, cam.azimuth, cam.elevation = 4.3, 96, -4      # nearly side-on, robot walks left -> right
    floor_id = model.geom("floor").id
    snaps, masks = {}, {}
    with mujoco.Renderer(model, height=H, width=W) as r:
        mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
        mujoco.mj_forward(model, data)
        for i in range(round(max(snap_times) / model.opt.timestep) + 1):
            t = round(i * model.opt.timestep, 4)
            if any(abs(t - st) < 1e-6 for st in snap_times):
                r.update_scene(data, camera=cam)
                snaps[t] = np.asarray(r.render()).astype(float)
                r.enable_segmentation_rendering()
                seg = r.render()
                r.disable_segmentation_rendering()
                masks[t] = ((seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (seg[..., 0] != floor_id))[..., None]
            control(model, data)
            mujoco.mj_step(model, data)
    times = sorted(snaps)
    img = snaps[times[-1]].copy()
    for k, t in enumerate(times[:-1]):
        alpha = 0.18 + 0.4 * k / max(len(times) - 2, 1)
        ghost = masks[t] & ~masks[times[-1]]
        img = np.where(ghost, img * (1 - alpha) + snaps[t] * alpha, img)
    return img


def banner(preview_dir=None):
    W, H = 1280, 400
    snap_times = [round(0.9 + 0.45 * k, 2) for k in range(8)]    # 8 poses ~0.35 m apart; the last is the robot
    command = np.array([0.8, 0.0, 0.0])
    left = strobe(mujoco_look, W, H, snap_times, command)
    right = strobe(gazebo_look, W, H, snap_times, command)

    # Diagonal seam: x position of the boundary shifts with the row, softened
    x = np.arange(W)[None, :]
    y = np.arange(H)[:, None]
    seam = 0.71 * W + (y - H / 2) * 0.35
    for name, softness, glow in (("soft", 70.0, False), ("glow", 14.0, True)):
        a = 1 / (1 + np.exp(-(x - seam) / softness))               # 0 = MuJoCo side, 1 = Gazebo side
        img = left * (1 - a[..., None]) + right * a[..., None]
        if glow:   # thin coloured line along the seam
            line = np.exp(-((x - seam) / 3.0) ** 2)[..., None]
            img = img * (1 - 0.8 * line) + np.array(BLUE) * 0.8 * line
        out = Image.fromarray(img.clip(0, 255).astype(np.uint8))
        out.save(OUT / f"banner_{name}.png")
        print(f"saved banner_{name}.png")
        if preview_dir:   # mock of the theme's overlay header (dark filter + title)
            prev = Image.fromarray((np.asarray(out).astype(float) * 0.6).astype(np.uint8))
            d = ImageDraw.Draw(prev)
            d.text((60, 140), "Week 3 - From MuJoCo to Gazebo", fill=(250, 250, 255), font=font(44))
            d.text((60, 205), "Running the same walking policies in a second simulator,\n"
                              "and finding out why they behave differently.", fill=(225, 225, 235), font=font(19))
            prev.save(Path(preview_dir) / f"banner_week3_{name}_preview.png")


# ---------------------------------------------------------------------------
# Chart helpers (SVG, dark theme). Palette: validated categorical slots on the blog background (#121421).
# ---------------------------------------------------------------------------

C_BLUE, C_ORANGE, C_AQUA, C_YELLOW = "#3987e5", "#d95926", "#199e70", "#c98500"
SVG_BG, SVG_TEXT, SVG_MUTED, SVG_GRID = "#121421", "#ebebf5", "#a3a6c0", "#2c3048"


def svg_doc(width, height, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
            f'font-family="Helvetica, Arial, sans-serif">\n<rect width="100%" height="100%" fill="{SVG_BG}" rx="8"/>\n'
            + body + "\n</svg>\n")


def svg_text(x, y, text, size=14, color=SVG_TEXT, anchor="start", weight="normal"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{text}</text>')


# ---------------------------------------------------------------------------
# The ROS 2 node, as a diagram
# ---------------------------------------------------------------------------

def node_diagram():
    W, H = 920, 340
    b = []

    def box(x, y, w, h, title, lines, color):
        b.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="#1b1e30" stroke="{color}" stroke-width="2"/>')
        b.append(svg_text(x + w / 2, y + 28, title, 17, SVG_TEXT, "middle", "bold"))
        for i, line in enumerate(lines):
            b.append(svg_text(x + w / 2, y + 54 + 21 * i, line, 13, SVG_MUTED, "middle"))

    def arrow(x1, y1, x2, y2, label_lines, color):
        sign = 1 if x2 > x1 else -1                     # head points along the arrow, either direction
        b.append(f'<line x1="{x1}" y1="{y1}" x2="{x2 - 8 * sign}" y2="{y2}" stroke="{color}" stroke-width="2"/>')
        b.append(f'<polygon points="{x2},{y2} {x2 - 10 * sign},{y2 - 5} {x2 - 10 * sign},{y2 + 5}" fill="{color}"/>')
        for i, line in enumerate(label_lines):
            b.append(svg_text((x1 + x2) / 2, y1 - 12 - 17 * (len(label_lines) - 1 - i), line, 12, SVG_MUTED, "middle"))

    box(30, 110, 220, 160, "Gazebo", ["physics, 1 ms steps", "29 joint controllers", "IMU, odometry"], C_AQUA)
    box(380, 90, 200, 200, "policy_runner", ["ROS 2 node (C++)", "every 20 ms:", "observation", "-> ONNX network", "-> 29 targets"], C_BLUE)
    box(700, 110, 190, 160, "ONNX file", ["network weights", "metadata: joints,", "default pose, scale,", "observation terms"], C_YELLOW)
    arrow(250, 160, 380, 160, ["joint states, IMU,", "odometry"], C_AQUA)
    arrow(380, 230, 250, 230, [], C_BLUE)
    b.append(svg_text(315, 254, "29 joint targets", 12, SVG_MUTED, "middle"))
    arrow(700, 190, 580, 190, ["loaded at start"], C_YELLOW)
    # velocity command from above (whoever drives the robot), arrow pointing down into the node
    b.append(f'<line x1="480" y1="34" x2="480" y2="82" stroke="{SVG_TEXT}" stroke-width="2"/>')
    b.append(f'<polygon points="480,90 475,80 485,80" fill="{SVG_TEXT}"/>')
    b.append(svg_text(492, 52, "velocity command (/cmd_vel)", 12, SVG_MUTED))
    b.append(svg_text(W / 2, 325, "The same node runs both policies: everything policy-specific comes from the ONNX metadata.",
                      13, SVG_MUTED, "middle"))
    (OUT / "node_diagram.svg").write_text(svg_doc(W, H, "\n".join(b)))
    print("saved node_diagram.svg")


# ---------------------------------------------------------------------------
# The derivative kick (MuJoCo replay of Gazebo's PID, left knee, armature on, from Q1b's traces)
# ---------------------------------------------------------------------------

def kick_plot():
    def load(dterm):
        d = np.genfromtxt(REPO / f"recordings/q1b/trace_armon_policygains_policylimits_{dterm}_walk.csv",
                          delimiter=",", names=True)
        return d["t"] - d["t"][0], d["target"], d["q"], d["tau"]

    t, target, q_err, tau_err = load("err")
    _, _, q_vel, tau_vel = load("vel")
    window = t <= 0.2                              # 200 ms: ten policy steps
    t, target, q_err, tau_err, q_vel, tau_vel = (a[window] for a in (t, target, q_err, tau_err, q_vel, tau_vel))
    W, H = 920, 560
    L, R = 90, 30
    panels = [(70, 230, "Left knee angle (rad)"), (330, 510, "Knee torque (N·m)")]
    b = [svg_text(L, 34, "The derivative kick: one 1 ms spike at every new target", 19, SVG_TEXT, weight="bold"),
         svg_text(L, 56, "MuJoCo, Unitree's policy walking, training gains: Gazebo-style D term vs training's, 200 ms (10 policy steps)",
                  13, SVG_MUTED)]

    def xs(v):
        return L + (v / 0.2) * (W - L - R)

    def series(x, y, lo, hi, top, bottom, color, width=2, step=False):
        ys = lambda v: bottom - (v - lo) / (hi - lo) * (bottom - top)
        pts = []
        for i in range(len(x)):
            if step and i:
                pts.append(f"{xs(x[i]):.1f},{ys(y[i - 1]):.1f}")
            pts.append(f"{xs(x[i]):.1f},{ys(y[i]):.1f}")
        return f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round"/>'

    def axes(top, bottom, lo, hi, ticks, label):
        out = [svg_text(L, top - 10, label, 13, SVG_MUTED)]
        for v in ticks:
            y = bottom - (v - lo) / (hi - lo) * (bottom - top)
            out.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W - R}" y2="{y:.1f}" stroke="{SVG_GRID}" stroke-width="1"/>')
            out.append(svg_text(L - 8, y + 4, f"{v:g}", 12, SVG_MUTED, "end"))
        return out

    # panel 1: target (steps) and angle
    top, bottom = 90, 230
    lo, hi = float(min(target.min(), q_err.min())) - 0.02, float(max(target.max(), q_err.max())) + 0.02
    ticks = np.round(np.linspace(lo, hi, 4), 2)
    b += axes(top, bottom, lo, hi, ticks, "Left knee angle (rad)")
    b.append(series(t, target, lo, hi, top, bottom, SVG_MUTED, 2, step=True))
    b.append(series(t, q_vel, lo, hi, top, bottom, C_BLUE, 2.5))
    b.append(svg_text(W - R, top + 4, "target from the policy (new value every 20 ms)", 12, SVG_MUTED, "end"))
    b.append(svg_text(W - R, top + 22, "joint angle", 12, SVG_TEXT, "end"))
    b.append(f'<line x1="{W - R - 104}" y1="{top + 18}" x2="{W - R - 80}" y2="{top + 18}" stroke="{C_BLUE}" stroke-width="2.5"/>')

    # panel 2: torque, D on error vs D on velocity, with the motor limit
    top, bottom = 300, 510
    lo, hi = -150, 150
    b += axes(top, bottom, lo, hi, [-139, -50, 0, 50, 139], "Knee torque (N·m)")
    for v in (-139, 139):
        y = bottom - (v - lo) / (hi - lo) * (bottom - top)
        b.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W - R}" y2="{y:.1f}" stroke="{SVG_MUTED}" stroke-width="1" stroke-dasharray="5,4"/>')
    b.append(series(t, tau_err, lo, hi, top, bottom, C_ORANGE, 1.8))
    b.append(series(t, tau_vel, lo, hi, top, bottom, C_BLUE, 2.5))
    for k in range(11):
        b.append(svg_text(xs(0.02 * k), bottom + 20, f"{20 * k}", 12, SVG_MUTED, "middle"))
    b.append(svg_text((L + W - R) / 2, bottom + 40, "time (ms)", 13, SVG_MUTED, "middle"))
    # legend (two series): marks + text in text colour
    lx, ly = L + 10, H + 4
    b.append(f'<line x1="{lx}" y1="{ly - 4}" x2="{lx + 24}" y2="{ly - 4}" stroke="{C_ORANGE}" stroke-width="3"/>')
    b.append(svg_text(lx + 32, ly, "D on the error (Gazebo's PID): spikes to the 139 N·m motor limit", 13))
    b.append(f'<line x1="{lx + 480}" y1="{ly - 4}" x2="{lx + 504}" y2="{ly - 4}" stroke="{C_BLUE}" stroke-width="3"/>')
    b.append(svg_text(lx + 512, ly, "D on velocity (training)", 13))
    (OUT / "derivative_kick.svg").write_text(svg_doc(W, H + 24, "\n".join(b)))
    print(f"saved derivative_kick.svg (max |tau|: error {abs(tau_err).max():.0f}, velocity {abs(tau_vel).max():.0f} N·m)")


# ---------------------------------------------------------------------------
# Explicit vs implicit damping on one ankle-roll joint (slow motion)
# ---------------------------------------------------------------------------

def simulate_joint(inertia, implicit, steps=60, dt=0.001, kp=28.5, kd=1.81, limit=50.0, target=0.1):
    """One joint stepping to a new target with the training PD gains, Gazebo's 1 ms step.
    Explicit: the damping torque uses the speed at the start of the step (and is clipped with the P torque, as in
    Gazebo's PID plugin). Implicit: the damping uses the speed at the end of the step (DART / MuJoCo implicitfast);
    only the P torque is clipped. Returns arrays of angle and speed, one per ms (index 0 = start)."""
    q, v = 0.0, 0.0
    qs, vs = [q], [v]
    for _ in range(steps):
        if implicit:
            tau_p = np.clip(kp * (target - q), -limit, limit)
            v = (v + dt / inertia * tau_p) / (1 + kd * dt / inertia)
        else:
            tau = np.clip(kp * (target - q) - kd * v, -limit, limit)
            v = v + dt / inertia * tau
        q = q + dt * v
        qs.append(q)
        vs.append(v)
    return np.array(qs), np.array(vs)


def damping_gif():
    cases = [("With armature (I = 0.0076), explicit", 0.0076, False, C_AQUA,
              "a = 0.24: moves smoothly toward the target"),
             ("No armature (I = 0.0004), explicit", 0.0004, False, C_ORANGE,
              "a = 4.5: speed flips sign every ms; only the\ntorque limit bounds it, and the joint runs away"),
             ("No armature (I = 0.0004), implicit", 0.0004, True, C_BLUE,
              "a = 4.5: still smooth, same as with armature")]
    steps = 60
    runs = [simulate_joint(I, imp, steps) for _, I, imp, _, _ in cases]
    PW, TOP, H = 400, 54, 560
    W = PW * 3
    qmax = 0.7
    hexrgb = lambda h: tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))
    grid = (60, 64, 90)

    def plot(d, x0, y0, y1, values, k, lo, hi, color, label, ticks):
        px0, px1 = x0 + 52, x0 + PW - 24
        d.text((px0, y0 - 20), label, fill=DIM, font=font(13))
        for t in ticks:
            y = y1 - (t - lo) / (hi - lo) * (y1 - y0)
            d.line([(px0, y), (px1, y)], fill=grid, width=1)
            d.text((px0 - 8 - 7 * len(f"{t:g}"), y - 7), f"{t:g}", fill=DIM, font=font(11))
        pts = [(px0 + j / steps * (px1 - px0), y1 - (min(max(values[j], lo), hi) - lo) / (hi - lo) * (y1 - y0))
               for j in range(k + 1)]
        if len(pts) > 1:
            d.line(pts, fill=color, width=2)
        d.ellipse([pts[-1][0] - 4, pts[-1][1] - 4, pts[-1][0] + 4, pts[-1][1] + 4], fill=color)
        return px0, px1

    frames = []
    for k in list(range(steps + 1)) + [steps] * 18:          # 1 ms per frame, then hold
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        d.text((14, 12), "Explicit vs implicit damping: one ankle-roll joint, new target 0.1 rad, 1 ms steps (slow motion)",
               fill=FG, font=font(19))
        d.text((W - 120, 36), f"t = {k} ms", fill=DIM, font=font(15))
        for i, ((title, _, _, color, note), (qs, vs)) in enumerate(zip(cases, runs)):
            x0, c = i * PW, hexrgb(color)
            if i:
                d.line([(x0, TOP + 4), (x0, H - 10)], fill=grid, width=1)
            d.text((x0 + 16, TOP + 8), title, fill=FG, font=font(15))
            # the joint at its true angle, with the target as a thin line
            cx, cy, r = x0 + PW // 2, TOP + 95, 80
            for ang, col, wdt in ((0.1, DIM, 1), (qs[k], c, 10)):
                d.line([(cx - r * np.cos(ang), cy + r * np.sin(ang)), (cx + r * np.cos(ang), cy - r * np.sin(ang))],
                       fill=col, width=wdt)
            d.ellipse([cx - 8, cy - 8, cx + 8, cy + 8], fill=FG)
            d.text((cx + r + 6, cy - r * np.sin(0.1) - 9), "target", fill=DIM, font=font(12))
            # angle: shared scale (comparable across panels)
            px0, px1 = plot(d, x0, TOP + 170, TOP + 280, qs, k, 0, qmax, c, "angle (rad), same scale in all three",
                            [0, 0.1, 0.35, 0.7])
            # speed: each panel's own scale
            vlim = float(np.ceil(abs(vs).max() * 1.15 * 10) / 10)
            plot(d, x0, TOP + 320, TOP + 430, vs, k, -vlim, vlim, c, f"speed (rad/s), scale ±{vlim:g}",
                 [-vlim, 0, vlim])
            d.text((px0, TOP + 436), "0", fill=DIM, font=font(11))
            d.text((px1 - 36, TOP + 436), f"{steps} ms", fill=DIM, font=font(11))
            if k >= 15:
                d.multiline_text((x0 + 16, TOP + 458), note, fill=FG, font=font(14), spacing=4)
        frames.append(img.quantize(colors=64, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
    path = OUT / "explicit_vs_implicit.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=90, loop=0, optimize=True)
    print(f"saved explicit_vs_implicit.gif: {len(frames)} frames, {path.stat().st_size / 1e6:.2f} MB")


# ---------------------------------------------------------------------------
# Every combination in MuJoCo (Q1b runner), standing: four representative controllers
# ---------------------------------------------------------------------------

def week2_scenery(spec):
    """Week 2 look (gradient sky, blue checker floor), with a large floor so drifting robots stay on it."""
    import mujoco
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.30, 0.40, 0.85], rgb2=[0.03, 0.03, 0.09], width=256, height=256)
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.20, 0.24, 0.42], rgb2=[0.12, 0.14, 0.27], width=256, height=256)
    mat = spec.add_material(name="grid", texrepeat=[2, 2], texuniform=True, reflectance=0.15)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    spec.geom("floor").material = "grid"
    spec.geom("floor").size = [40, 40, 0.1]
    spec.worldbody.add_light(pos=[2, -2, 4], dir=[-0.4, 0.4, -1], diffuse=[0.8, 0.8, 0.85], castshadow=True)
    spec.visual.headlight.ambient = [0.3, 0.3, 0.35]
    spec.visual.headlight.diffuse = [0.35, 0.35, 0.35]


def mujoco_grid_gif(seconds=30.0, speed=3, fps=8, tile=(340, 250)):
    import mujoco
    sys.path.insert(0, str(REPO / "open_questions" / "q1b_controller"))
    from mujoco_controller_test import DT, run
    from render_utils import make_camera
    from unitree_utils import build_unitree_model, load_unitree_policy

    configs = [("Armature, damping on velocity (like training)", "on", "policy", "policy", "vel", False),
               ("No armature, implicit damping (corrected)", "off", "policy", "policy", "jnt", False),
               ("No armature, explicit damping on velocity", "off", "policy", "policy", "vel", False),
               ("No armature, Gazebo's original controller", "off", "luis", "luis", "err", True)]
    model, data, joints, acts = build_unitree_model(decorate=week2_scenery)
    model.opt.timestep = DT
    model.actuator_gainprm[:, 0] = 0.0
    model.actuator_biasprm[:, 0:3] = 0.0
    policy = load_unitree_policy(joints, acts)
    armature = model.dof_armature.copy()
    tw, th = tile
    every = round(speed / (fps * DT))                     # physics steps per frame
    tiles = []
    for label, arm, gains, limits, dterm, extras in configs:
        model.dof_armature[:] = armature if arm == "on" else 0.0
        renderer = mujoco.Renderer(model, height=th, width=tw)
        cam = make_camera(lookat=[0, 0, 0.55], distance=4.2, azimuth=135, elevation=-18)   # fixed: drift shows
        frames, state = [], {"step": 0}

        def capture(model, data):
            if state["step"] % every == 0:
                renderer.update_scene(data, camera=cam)
                frames.append(Image.fromarray(renderer.render()))
            state["step"] += 1

        name = run(model, data, policy, joints, gains, limits, dterm, arm, "stand", on_step=capture, extras=extras,
                   out=REPO / "recordings" / "week3", prefix="grid_")
        renderer.close()
        d = np.genfromtxt(REPO / "recordings" / "week3" / f"{name}.csv", delimiter=",", names=True)
        fell = d["z"].min() < 0.4
        moved = np.hypot(d["x"][-1] - d["x"][0], d["y"][-1] - d["y"][0])
        turned = np.degrees(np.unwrap(d["yaw"]))[-1] - np.degrees(np.unwrap(d["yaw"]))[0]
        result = (f"fell after {d['t'][np.argmax(d['z'] < 0.4)]:.1f} s" if fell
                  else f"30 s: moved {moved:.2f} m, turned {turned:+.0f} deg")
        tiles.append((label, result, frames))
        print(f"{label}: {result}")
    n = min(len(f) for _, _, f in tiles)
    out = []
    bar = 36
    for k in range(n):
        canvas = Image.new("RGB", (2 * tw, 2 * (th + bar) + 44), BG)
        d = ImageDraw.Draw(canvas)
        d.text((14, 11), f"Unitree's policy in MuJoCo, zero command for 30 s ({speed}x speed, fixed camera)",
               fill=FG, font=font(15))
        for i, (label, result, frames) in enumerate(tiles):
            x, y = (i % 2) * tw, 44 + (i // 2) * (th + bar)
            canvas.paste(frames[min(k, len(frames) - 1)], (x, y + bar))
            d.text((x + 10, y + 3), label, fill=FG, font=font(13))
            d.text((x + 10, y + 20), result, fill=(255, 206, 86), font=font(13))
        out.append(canvas.quantize(colors=64, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
    path = OUT / "mujoco_grid.gif"
    out[0].save(path, save_all=True, append_images=out[1:], duration=1000 // fps, loop=0, optimize=True)
    print(f"saved mujoco_grid.gif: {len(out)} frames, {path.stat().st_size / 1e6:.2f} MB")


# ---------------------------------------------------------------------------
# Real Gazebo recordings as labelled MP4 videos (each tile's result is read from its own run log)
# ---------------------------------------------------------------------------

REC = REPO / "recordings"
VIDEOS = REPO / "docs" / "assets" / "videos" / "week-3"
VIDEOS.mkdir(parents=True, exist_ok=True)


def gazebo_video(name, title, tiles, seconds, cols=0, speed=2, width=1200):
    """tiles: (label, recording path without extension). Composes with gazebo/side_by_side.py, then a looping MP4."""
    import subprocess
    import tempfile
    tmp = Path(tempfile.mkdtemp()) / f"{name}.mp4"
    args = [f"{label}|{REC / rec}.mp4|{REC / rec}.csv" for label, rec in tiles]
    subprocess.run([sys.executable, str(REPO / "gazebo" / "side_by_side.py"), str(tmp), "--title", title,
                    "--seconds", str(seconds), *(["--cols", str(cols)] if cols else []), *args],
                   check=True, capture_output=True)
    video = VIDEOS / f"{name}.mp4"          # MP4, not GIF: screen recordings compress ~10x better
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(tmp), "-vf",
                    f"setpts=PTS/{speed},scale={width}:-2:flags=lanczos", "-an", "-c:v", "libx264", "-crf", "26",
                    "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)], check=True)
    print(f"saved {video.name}: {video.stat().st_size / 1e6:.2f} MB")


def gazebo_videos():
    gazebo_video("raw_stand", "Gazebo, original setup: zero command for 30 s (2x speed)",
               [("Unitree's policy", "q1b/rec_baseline_stand"), ("walk.onnx", "q2/gz_original_stand")], 30)
    gazebo_video("raw_walk", "Gazebo, original setup: stand 4 s, then walk at 0.5 m/s (2x speed)",
               [("Unitree's policy", "q1b/rec_baseline_walk"), ("walk.onnx", "q2/gz_original_walk")], 24)
    gazebo_video("corrected_gazebo", "Gazebo, corrected controller: stand 4 s, then walk at 0.5 m/s (2x speed)",
               [("Unitree's policy", "q2/gz_unitree_corrected_walk"), ("walk.onnx", "q2/gz_corrected_walk")], 24)
    gazebo_video("gazebo_checks", "Gazebo, Unitree's policy, original controller, zero command (2x speed)",
               [("Original", "q1b/rec_baseline_stand"), ("+ armature in the links", "q1b/rec_armature_stand"),
                ("0.2 ms physics step", "q1b/rec_step_0p2ms_stand")], 30, width=1400)
    gazebo_video("policy_compare", "Gazebo, corrected controller: walking at 0.5 m/s without and with armature (2x speed)",
               [("Unitree, no armature", "q2/gz_unitree_corrected_walk"),
                ("Unitree, + armature", "q2/gz_unitree_corrected_armature_walk"),
                ("walk.onnx, no armature", "q2/gz_corrected_walk"),
                ("walk.onnx, + armature", "q2/gz_corrected_armature_walk")], 24, cols=2)


# ---------------------------------------------------------------------------
# Controller variants in Gazebo (sim clock): standing drift, walking speed, heading wander
# ---------------------------------------------------------------------------

# (median, min, max) standing drift in 30 s (m); walking speed (m/s); heading wander (deg). Sim clock, from
# recordings/week3 (original, training gains), q1b/q2/q3 repeats (original, corrected), MuJoCo references.
VARIANTS = ["MuJoCo (reference)", "Gazebo: original", "Gazebo: training gains", "Gazebo: corrected"]
VARIANT_COLORS = [C_BLUE, C_ORANGE, C_AQUA, C_YELLOW]
VARIANT_DATA = {
    "Unitree": {"stand": [(0.01, 0.01, 0.01), (5.09, 2.56, 7.06), (7.20, 6.24, 9.46), (9.45, 0.00, 10.01)],
                "speed": [0.50, 0.56, 0.55, 0.51], "wander": [0.39, 5.67, 5.29, 0.52]},
    "walk.onnx": {"stand": [(0.01, 0.01, 0.01), (0.50, 0.28, 1.25), (0.55, 0.35, 0.98), (4.02, 2.91, 8.34)],
                  "speed": [0.43, 0.60, 0.57, 0.72], "wander": [0.18, 4.36, 2.29, 1.36]},
}


def variants_chart():
    W, H = 1000, 470
    b = [svg_text(30, 36, "Controller variants in Gazebo (sim clock, 3-7 runs each)", 19, SVG_TEXT, weight="bold")]
    # legend: always present for >= 2 series; marks carry colour, text stays in text colour
    lx = 30
    for name, color in zip(VARIANTS, VARIANT_COLORS):
        b.append(f'<rect x="{lx}" y="54" width="14" height="14" rx="3" fill="{color}"/>')
        b.append(svg_text(lx + 20, 66, name, 13))
        lx += 20 + 7.6 * len(name) + 28
    panels = [("stand", "Standing 30 s: distance moved (m)", 11, [0, 2.5, 5, 7.5, 10], "{:.2f}"),
              ("speed", "Walking: speed (m/s), command 0.5", 0.8, [0, 0.2, 0.4, 0.6, 0.8], "{:.2f}"),
              ("wander", "Walking: heading wander (deg)", 7, [0, 2, 4, 6], "{:.1f}")]
    pw, top, bottom = (W - 60) / 3, 120, 400
    bar, gap = 17, 2
    for i, (key, title, ymax, ticks, fmt) in enumerate(panels):
        x0 = 30 + i * pw
        px0, px1 = x0 + 40, x0 + pw - 16
        b.append(svg_text(x0, top - 18, title, 13, SVG_MUTED))
        for t in ticks:
            y = bottom - t / ymax * (bottom - top)
            b.append(f'<line x1="{px0}" y1="{y:.1f}" x2="{px1}" y2="{y:.1f}" stroke="{SVG_GRID}" stroke-width="1"/>')
            b.append(svg_text(px0 - 6, y + 4, f"{t:g}", 11, SVG_MUTED, "end"))
        if key == "speed":
            y = bottom - 0.5 / ymax * (bottom - top)
            b.append(f'<line x1="{px0}" y1="{y:.1f}" x2="{px1}" y2="{y:.1f}" stroke="{SVG_TEXT}" stroke-width="1" stroke-dasharray="5,4"/>')
        group_w = (px1 - px0) / 2
        for g, (policy, data) in enumerate(VARIANT_DATA.items()):
            gx = px0 + g * group_w + (group_w - 4 * bar - 3 * gap) / 2
            for k, color in enumerate(VARIANT_COLORS):
                value = data[key][k][0] if key == "stand" else data[key][k]
                x = gx + k * (bar + gap)
                h = max(value / ymax * (bottom - top), 1.5)
                y = bottom - h
                # rounded data end, square at the baseline
                b.append(f'<path d="M{x:.1f},{bottom} V{y + 4:.1f} Q{x:.1f},{y:.1f} {x + 4:.1f},{y:.1f} H{x + bar - 4:.1f} '
                         f'Q{x + bar:.1f},{y:.1f} {x + bar:.1f},{y + 4:.1f} V{bottom} Z" fill="{color}"/>')
                if key == "stand":    # run-to-run range
                    lo, hi = data[key][k][1], data[key][k][2]
                    ylo, yhi = bottom - lo / ymax * (bottom - top), bottom - hi / ymax * (bottom - top)
                    cx = x + bar / 2
                    b.append(f'<line x1="{cx:.1f}" y1="{ylo:.1f}" x2="{cx:.1f}" y2="{yhi:.1f}" stroke="{SVG_TEXT}" stroke-width="1.5"/>')
                    b.append(f'<line x1="{cx - 4:.1f}" y1="{yhi:.1f}" x2="{cx + 4:.1f}" y2="{yhi:.1f}" stroke="{SVG_TEXT}" stroke-width="1.5"/>')
                label_y = (min(y, bottom - data[key][k][2] / ymax * (bottom - top)) if key == "stand" else y) - 5
                b.append(svg_text(x + bar / 2, label_y, fmt.format(value), 9.5, SVG_MUTED, "middle"))
            b.append(svg_text(px0 + g * group_w + group_w / 2, bottom + 20, policy, 13, SVG_TEXT, "middle"))
        b.append(f'<line x1="{px0}" y1="{bottom}" x2="{px1}" y2="{bottom}" stroke="{SVG_MUTED}" stroke-width="1"/>')
    b.append(svg_text(30, H - 22, "Standing: median of the runs, with the line showing the smallest to largest. "
                                  "Walking: mean of the runs (stand 4 s, then 0.5 m/s, measured from 8 s).", 12, SVG_MUTED))
    (OUT / "variants_chart.svg").write_text(svg_doc(W, H, "\n".join(b)))
    print("saved variants_chart.svg")


if __name__ == "__main__":
    targets = sys.argv[1:] or ["banner"]
    preview = next((t.split("=", 1)[1] for t in targets if t.startswith("preview=")), None)
    if "banner" in targets:
        banner(preview)
    if "diagram" in targets:
        node_diagram()
    if "kick" in targets:
        kick_plot()
    if "damping" in targets:
        damping_gif()
    if "grid" in targets:
        mujoco_grid_gif()
    if "gazebo" in targets:
        gazebo_videos()
    if "variants" in targets:
        variants_chart()
