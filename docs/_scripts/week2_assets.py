"""Images and GIFs for the Week 2 blog post.

Run from the repo root:  .venv/bin/python docs/_scripts/week2_assets.py
Outputs go to docs/assets/images/week-2/.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "assets" / "images" / "week-2"
OUT.mkdir(parents=True, exist_ok=True)

BG = (18, 20, 33)
FG = (235, 235, 245)
DIM = (150, 155, 180)
BLUE, RED, YELLOW, GREEN, PURPLE = (54, 162, 235), (255, 99, 132), (255, 206, 86), (75, 192, 160), (153, 102, 255)


def font(size):
    return ImageFont.load_default(size=size)


# ---------------------------------------------------------------------------
# Planetary gearbox (7520_14 motor: two stages, 4.5 x 3.18 = 14.3)
# ---------------------------------------------------------------------------

def gear_polygon(cx, cy, radius, teeth, angle, internal=False, depth=None):
    """Points of a gear outline with simple trapezoidal teeth, rotated by angle (rad)."""
    depth = depth or radius * 0.12
    pts = []
    for k in range(teeth):
        base = angle + 2 * np.pi * k / teeth
        step = 2 * np.pi / teeth
        # tooth profile: root, flank up, tip, flank down
        for frac, r in ((0.0, 0), (0.15, 1), (0.45, 1), (0.6, 0)):
            rr = radius + (depth if not internal else -depth) * r - (depth / 2 if not internal else -depth / 2)
            a = base + frac * step
            pts.append((cx + rr * np.cos(a), cy + rr * np.sin(a)))
    return pts


def draw_gear(d, cx, cy, radius, teeth, angle, color, internal=False, spokes=True):
    pts = gear_polygon(cx, cy, radius, teeth, angle, internal)
    if internal:
        d.line(pts + [pts[0]], fill=color, width=3)
    else:
        d.polygon(pts, fill=tuple(int(c * 0.35) for c in color), outline=color)
        if spokes:   # spokes make the rotation visible
            for k in range(3):
                a = angle + 2 * np.pi * k / 3
                d.line([(cx, cy), (cx + 0.7 * radius * np.cos(a), cy + 0.7 * radius * np.sin(a))], fill=color, width=3)
        d.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=color)


def draw_stage(d, cx, cy, ring_r, sun_r, sun_angle, ratio, colors):
    """One planetary stage with a fixed ring: sun in, carrier out at sun_speed / ratio."""
    sun_c, planet_c, ring_c, carrier_c = colors
    planet_r = (ring_r - sun_r) / 2
    orbit = sun_r + planet_r
    carrier_angle = sun_angle / ratio
    # tooth counts proportional to radius (same tooth size within a stage)
    pitch = 2 * np.pi * sun_r / 12
    n_sun, n_planet, n_ring = 12, max(6, round(2 * np.pi * planet_r / pitch)), round(2 * np.pi * ring_r / pitch)
    # planet spin relative to the carrier for rolling without slip: -(sun_r / planet_r) * (sun - carrier)
    planet_spin = -(sun_r / planet_r) * (sun_angle - carrier_angle)
    draw_gear(d, cx, cy, ring_r + 6, n_ring, 0.0, ring_c, internal=True)          # fixed ring
    for k in range(3):
        a = carrier_angle + 2 * np.pi * k / 3
        px, py = cx + orbit * np.cos(a), cy + orbit * np.sin(a)
        d.line([(cx, cy), (px, py)], fill=carrier_c, width=6)                      # carrier arm
        draw_gear(d, px, py, planet_r - 3, n_planet, planet_spin + a, planet_c, spokes=False)
        d.ellipse([px - 5, py - 5, px + 5, py + 5], fill=carrier_c)
    draw_gear(d, cx, cy, sun_r - 3, n_sun, sun_angle, sun_c)                       # sun on top


def planetary_gif():
    W, H = 1000, 620
    g2, g3 = 4.5, 1 + 48 / 22            # stage ratios of the 7520_14 motor
    r1, r2, r3 = 0.489e-4, 0.098e-4, 0.533e-4
    frames = []
    n = 150
    for i in range(n):
        rotor = i * 0.09                  # rad per frame for the motor rotor
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        d.text((24, 16), "Two-stage planetary gearbox (7520_14 motor, hip pitch)", fill=FG, font=font(24))
        # stage 1: sun = motor rotor; carrier turns 1/g2 as fast and drives stage 2's sun
        draw_stage(d, 260, 290, 170, 48, rotor, g2, (RED, YELLOW, DIM, BLUE))
        # stage 2: sun turns at rotor/g2; carrier = joint output at rotor/(g2*g3)
        draw_stage(d, 740, 290, 150, 69, rotor / g2, g3, (BLUE, YELLOW, DIM, GREEN))
        d.line([(440, 290), (580, 290)], fill=BLUE, width=6)                     # shaft between stages
        d.polygon([(580, 280), (596, 290), (580, 300)], fill=BLUE)
        d.text((150, 480), f"stage 1  x{g2}", fill=FG, font=font(20))
        d.text((640, 480), f"stage 2  x{g3:.2f}", fill=FG, font=font(20))
        # speeds relative to the joint, and the armature term each part contributes
        rows = [
            (RED,   f"motor rotor:  {g2 * g3:5.2f} x joint speed  ->  r1 (g2 g3)^2 = {r1 * (g2 * g3) ** 2:.5f}"),
            (BLUE,  f"middle stage: {g3:5.2f} x joint speed  ->  r2 g3^2       = {r2 * g3 ** 2:.6f}"),
            (GREEN, f"output:        1.00 x joint speed  ->  r3             = {r3:.6f}"),
        ]
        for k, (c, text) in enumerate(rows):
            d.rectangle([24, 522 + 30 * k, 40, 538 + 30 * k], fill=c)
            d.text((52, 518 + 30 * k), text, fill=FG, font=font(18))
        d.text((700, 548), "armature = sum", fill=DIM, font=font(18))
        d.text((700, 576), f"= {r1 * (g2 * g3) ** 2 + r2 * g3 ** 2 + r3:.4f} kg m^2", fill=FG, font=font(20))
        frames.append(img.quantize(colors=64, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
    path = OUT / "planetary_gearbox.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=40, loop=0, optimize=True)
    print(f"saved {path.name}: {len(frames)} frames, {path.stat().st_size / 1e6:.2f} MB")


# ---------------------------------------------------------------------------
# Gait phase clock (Unitree's policy): [sin 2πφ, cos 2πφ], φ = (t mod 0.6) / 0.6
# ---------------------------------------------------------------------------

def phase_gif():
    W, H = 1300, 460
    period, seconds = 0.6, 1.8
    capture_fps, playback_fps = 40, 8      # 40 frames per simulated second, shown at 8/s = 5x slow motion
    stand_from = 1.2                       # 2 strides walking, then 0.6 s with a zero command -> clock off
    cx, cy, R = 200, 230, 130              # clock circle
    px0, px1, py_mid, py_amp = 420, 1260, 230, 120   # trace plot area: ~280 px per 0.6 s stride
    frames = []
    n = round(seconds * capture_fps)
    ts = np.arange(n + 1) / capture_fps

    def phase(t):
        if t >= stand_from:
            return np.zeros(2)
        phi = (t % period) / period
        return np.array([np.sin(2 * np.pi * phi), np.cos(2 * np.pi * phi)])

    def tx(t):
        return px0 + (px1 - px0) * t / seconds

    for i in range(n + 1):
        t = ts[i]
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        standing = t >= stand_from
        d.text((24, 16), "Unitree's gait clock: one turn per 0.6 s stride", fill=FG, font=font(24))
        d.text((W - 260, 20), f"t = {t:.2f} s   (5x slow motion)", fill=DIM, font=font(16))
        # clock circle with axes
        d.ellipse([cx - R, cy - R, cx + R, cy + R], outline=DIM, width=2)
        d.line([(cx - R - 15, cy), (cx + R + 15, cy)], fill=(70, 75, 100))
        d.line([(cx, cy - R - 15), (cx, cy + R + 15)], fill=(70, 75, 100))
        d.text((cx + R + 18, cy - 9), "sin", fill=RED, font=font(16))
        d.text((cx - 12, cy - R - 38), "cos", fill=BLUE, font=font(16))
        s, c = phase(t)
        dot = (cx + R * s, cy - R * c)      # sin on x, cos on y: starts at the top, turns clockwise
        d.line([(cx, cy), dot], fill=YELLOW, width=3)
        d.ellipse([dot[0] - 9, dot[1] - 9, dot[0] + 9, dot[1] + 9], fill=YELLOW)
        status = "STAND: clock off -> [0, 0]" if standing else f"phase = [{s:+.2f}, {c:+.2f}]"
        d.text((cx - R, cy + R + 30), status, fill=(GREEN if standing else FG), font=font(18))
        # traces
        d.rectangle([px0, py_mid - py_amp - 10, px1, py_mid + py_amp + 10], outline=(70, 75, 100))
        d.line([(px0, py_mid), (px1, py_mid)], fill=(70, 75, 100))
        for k in range(1, 3):           # stride boundaries
            x = tx(k * period)
            d.line([(x, py_mid - py_amp - 10), (x, py_mid + py_amp + 10)], fill=(45, 50, 75))
        d.text((tx(period / 2) - 30, py_mid + py_amp + 16), "stride 1", fill=DIM, font=font(14))
        d.text((tx(1.5 * period) - 30, py_mid + py_amp + 16), "stride 2", fill=DIM, font=font(14))
        d.text((tx(2.5 * period) - 45, py_mid + py_amp + 16), "standing", fill=GREEN, font=font(14))
        d.text((tx(stand_from) + 8, py_mid - py_amp - 6), "command = 0", fill=GREEN, font=font(14))
        d.text((px0, py_mid - py_amp - 34), "sin 2πφ", fill=RED, font=font(16))     # legend
        d.text((px0 + 80, py_mid - py_amp - 34), "cos 2πφ", fill=BLUE, font=font(16))
        for comp, color in ((0, RED), (1, BLUE)):
            pts = [(tx(tt), py_mid - py_amp * phase(tt)[comp]) for tt in ts[: i + 1]]
            if len(pts) > 1:
                d.line(pts, fill=color, width=3)
        d.text((px0, H - 40), "|command| < 0.1 -> clock off: the policy's signal to stand still",
               fill=DIM, font=font(16))
        frames.append(img.quantize(colors=64, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
    path = OUT / "gait_phase.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:] + [frames[-1]] * 8, duration=1000 // playback_fps,
                   loop=0, optimize=True)
    print(f"saved {path.name}: {len(frames)} frames, {path.stat().st_size / 1e6:.2f} MB")


# ---------------------------------------------------------------------------
# Robot GIFs: same physics as exercises/, plus visual-only scenery
# ---------------------------------------------------------------------------

def robot_gifs():
    import sys
    sys.path.insert(0, str(REPO / "exercises"))
    import mujoco
    from g1_utils import build_g1_model
    from policy_utils import DIEGO_POLICY, load_policy, make_policy_controller, trajectory
    from render_utils import make_camera, record_frames, side_by_side
    from unitree_utils import build_unitree_model, load_unitree_policy, make_unitree_observation

    def scenery(spec):
        """Visual only: gradient sky, checkered floor (0.5 m squares), extra light. Physics unchanged."""
        spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                         builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                         rgb1=[0.30, 0.40, 0.85], rgb2=[0.03, 0.03, 0.09], width=256, height=256)
        spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                         rgb1=[0.20, 0.24, 0.42], rgb2=[0.12, 0.14, 0.27], width=256, height=256)
        mat = spec.add_material(name="grid", texrepeat=[2, 2], texuniform=True, reflectance=0.15)
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
        spec.geom("floor").material = "grid"
        spec.worldbody.add_light(pos=[2, -2, 4], dir=[-0.4, 0.4, -1], diffuse=[0.8, 0.8, 0.85], castshadow=True)
        spec.visual.headlight.ambient = [0.3, 0.3, 0.35]
        spec.visual.headlight.diffuse = [0.35, 0.35, 0.35]

    d_model, d_data, d_joints, d_acts = build_g1_model(REPO / "robot" / "g1.xml", decorate=scenery)
    u_model, u_data, u_joints, u_acts = build_unitree_model(decorate=scenery)
    diego = load_policy(DIEGO_POLICY, d_joints, d_acts)
    unitree = load_unitree_policy(u_joints, u_acts)

    def panel(model, data, policy, cmd, seconds, label, w, h, fps=12):
        """One run: camera framed on its own path, frames returned."""
        path = trajectory(model, data, policy, np.array(cmd, dtype=float), seconds)[0]
        center = (path.min(axis=0) + path.max(axis=0)) / 2
        cam = make_camera(lookat=[center[0], center[1], 0.6], distance=2.8 + np.ptp(path, axis=0).max(),
                          azimuth=125, elevation=-14)
        mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
        mujoco.mj_forward(model, data)
        return record_frames(model, data, seconds, camera=cam,
                             control=make_policy_controller(policy, np.array(cmd, dtype=float)),
                             capture_fps=fps, width=w, height=h, label=label)

    def save(frames, name, title, fps=12):
        """Add a title bar, reduce colors, save."""
        out = []
        for f in frames:
            canvas = Image.new("RGB", (f.width, f.height + 40), BG)
            canvas.paste(f, (0, 40))
            ImageDraw.Draw(canvas).text((14, 9), title, fill=FG, font=font(20))
            out.append(canvas.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
        path = OUT / name
        out[0].save(path, save_all=True, append_images=out[1:], duration=1000 // fps, loop=0, optimize=True)
        print(f"saved {name}: {len(out)} frames, {path.stat().st_size / 1e6:.2f} MB")

    # 0. Setup: motors alone vs motors + policy (zero command, standing)
    def hold_panel(policy, label, seconds=4.0):
        cam = make_camera(lookat=[0.1, 0, 0.45], distance=2.6, azimuth=125, elevation=-12)
        mujoco.mj_resetDataKeyframe(d_model, d_data, d_model.key("init_state").id)   # ctrl = starting pose
        mujoco.mj_forward(d_model, d_data)
        control = make_policy_controller(policy, np.zeros(3)) if policy else None
        return record_frames(d_model, d_data, seconds, camera=cam, control=control,
                             capture_fps=12, width=480, height=360, label=label)

    save(side_by_side([hold_panel(None, "motors only: hold the starting pose"),
                       hold_panel(diego, "motors + policy: command = stand still")]),
         "stand_vs_policy.gif", "Without the policy the G1 can't stand. With it, it balances.")

    # 1. It walks
    save(panel(d_model, d_data, diego, [0.6, 0, 0], 6.0, "command: 0.6 m/s forward", 640, 400),
         "walk_forward.gif", "walk.onnx in my MuJoCo setup: 3.0 m in 6 s")

    # 2. Dead zone: walk.onnx | Unitree
    for name, cmd, title in (("deadzone_turn.gif", [0, 0, 0.8], "Turn in place (0.8 rad/s)"),
                             ("deadzone_slow_forward.gif", [0.3, 0, 0], "Slow forward (0.3 m/s)"),
                             ("deadzone_slow_sideways.gif", [0, 0.3, 0], "Slow sideways (0.3 m/s)")):
        frames = side_by_side([panel(d_model, d_data, diego, cmd, 4.0, "walk.onnx", 480, 340),
                               panel(u_model, u_data, unitree, cmd, 4.0, "Unitree", 480, 340)])
        save(frames, name, f"{title}: walk.onnx (left) vs Unitree (right)")

    # 3. Gait clock experiment on Unitree's policy: normal | off | frozen
    def with_phase(value):
        def obs_fn(model, data, default_pos, last_action, command):
            obs = make_unitree_observation(model, data, default_pos, last_action, command)
            obs[9:11] = value
            return obs
        return obs_fn

    variants = [("normal clock", make_unitree_observation), ("clock off [0, 0]", with_phase([0.0, 0.0])),
                ("clock frozen [0, 1]", with_phase([0.0, 1.0]))]
    for name, cmd, title in (("clock_turn.gif", [0, 0, 0.8], "Turn in place"),
                             ("clock_slow_forward.gif", [0.3, 0, 0], "Slow forward")):
        frames = side_by_side([panel(u_model, u_data, load_unitree_policy(u_joints, u_acts, obs_fn), cmd, 4.0,
                                     label, 400, 300) for label, obs_fn in variants])
        save(frames, name, f"Unitree's policy, {title.lower()}: what happens without its gait clock")


# ---------------------------------------------------------------------------
# Banner: Unitree's G1 walking (stroboscopic ghosts) with a small gait-clock inset
# ---------------------------------------------------------------------------

def banner(preview_dir=None):
    import sys
    sys.path.insert(0, str(REPO / "exercises"))
    import mujoco
    from policy_utils import make_policy_controller
    from unitree_utils import build_unitree_model, load_unitree_policy

    W, H = 1280, 400

    def scenery(spec):
        spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                         builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                         rgb1=[0.30, 0.40, 0.85], rgb2=[0.03, 0.03, 0.09], width=512, height=512)
        spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                         rgb1=[0.20, 0.24, 0.42], rgb2=[0.11, 0.13, 0.25], width=512, height=512)
        mat = spec.add_material(name="grid", texrepeat=[2, 2], texuniform=True, reflectance=0.2)
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
        spec.geom("floor").material = "grid"
        spec.geom("floor").size = [40, 40, 0.1]
        spec.worldbody.add_light(pos=[1.5, -3, 4], dir=[-0.3, 0.6, -1], diffuse=[0.85, 0.85, 0.9], castshadow=True)
        spec.visual.headlight.ambient = [0.28, 0.28, 0.33]
        spec.visual.headlight.diffuse = [0.3, 0.3, 0.3]
        spec.visual.global_.offwidth, spec.visual.global_.offheight = W, H
        spec.visual.map.zfar = 60.0
        spec.visual.quality.shadowsize = 8192

    model, data, joints, acts = build_unitree_model(decorate=scenery)
    policy = load_unitree_policy(joints, acts)
    command = np.array([0.8, 0.0, 0.0])
    control = make_policy_controller(policy, command)

    cam = mujoco.MjvCamera()
    cam.lookat[:] = [-0.15, 0.0, 0.62]
    cam.distance, cam.azimuth, cam.elevation = 3.6, 96, -4      # nearly side-on, robot walks left -> right

    snap_times = [round(0.8 + 0.4 * k, 2) for k in range(7)]     # 7 poses, ~0.32 m apart; last = the robot
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
                # exact robot silhouette: segmentation gives (object id, object type) per pixel
                r.enable_segmentation_rendering()
                seg = r.render()
                r.disable_segmentation_rendering()
                is_geom = seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
                masks[t] = (is_geom & (seg[..., 0] != floor_id))[..., None]
            control(model, data)
            mujoco.mj_step(model, data)
        # background without the robot: move it far away
        data.qpos[0] += 100.0
        mujoco.mj_forward(model, data)
        r.update_scene(data, camera=cam)
        background = np.asarray(r.render()).astype(float)

    times = sorted(snaps)
    img = snaps[times[-1]].copy()            # final frame: robot with its shadow and reflection
    n_ghosts = len(times) - 1
    for k, t in enumerate(times[:-1]):       # ghosts, oldest first, faint -> stronger
        alpha = 0.15 + 0.35 * k / max(n_ghosts - 1, 1)
        ghost_mask = masks[t] & ~masks[times[-1]]          # never paint over the real robot
        img = np.where(ghost_mask, img * (1 - alpha) + snaps[t] * alpha, img)
    out = Image.fromarray(img.clip(0, 255).astype(np.uint8))

    out.save(OUT / "banner.png")
    print("saved banner.png")
    if preview_dir:   # mock of the theme's overlay header (dark filter + title)
        prev = Image.fromarray((np.asarray(out).astype(float) * 0.62).astype(np.uint8))
        pd = ImageDraw.Draw(prev)
        pd.text((60, 140), "Week 2 - Making the G1 Walk", fill=(250, 250, 255), font=font(44))
        pd.text((60, 205), "Building the Unitree G1 in MuJoCo, running a trained walking policy,\n"
                           "and finding out why it could not turn on the spot.", fill=(225, 225, 235), font=font(19))
        prev.save(Path(preview_dir) / "banner_week2_preview.png")


if __name__ == "__main__":
    import sys
    targets = sys.argv[1:] or ["planetary", "phase", "robot"]
    if "planetary" in targets:
        planetary_gif()
    if "phase" in targets:
        phase_gif()
    if "robot" in targets:
        robot_gifs()
    if "banner" in targets:
        banner(preview_dir=next((t.split("=", 1)[1] for t in targets if t.startswith("preview=")), None))
