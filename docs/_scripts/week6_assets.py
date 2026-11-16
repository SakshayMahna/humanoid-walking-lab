"""Images and GIFs for the Week 6 post (six weeks in review) and its LinkedIn GIF: the same sideways pull on a grippy
floor (the G1 tips over) and on ice (it slides).

The physics is Part A5 of classical/02_contact_zmp_lipm/contact_forces.py, imported unchanged: same chain of
balanced poses, same LQR gains, same pull (20 N/s on the pelvis from t = 0.5 s). Only the scenery (the blog's navy
sky and checker floor, from week5_assets) and the drawn arrows are added; neither takes part in the physics.

Run from the repo root:  .venv/bin/python docs/_scripts/week6_assets.py [clip] [linkedin] [banner] [preview=DIR]
Outputs: docs/assets/images/week-6/ (blog GIF, banner), docs/assets/videos/week-6/slide_or_tip.gif (LinkedIn).
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import week5_assets as A  # noqa: E402  (patches the G1's floor with the blog's scenery)
from week5_assets import F, REPO, make_camera, mujoco  # noqa: E402

sys.path.insert(0, str(REPO / "classical" / "02_contact_zmp_lipm"))
import contact_forces as C  # noqa: E402

OUT = REPO / "docs" / "assets" / "images" / "week-6"
OUT_LI = REPO / "docs" / "assets" / "videos" / "week-6"
FPS_IN = 30
RATE = 20.0                                    # N/s, as in A5
FRICTIONS = ((0.6, "Grippy floor", "tips over"), (0.1, "Ice", "slides"))
YELLOW, AQUA, BLUE = [1.0, 0.72, 0.15, 1], [0.15, 0.85, 0.6, 1], [0.35, 0.6, 1.0, 1]


def pull(t: float) -> np.ndarray:
    return np.array([0.0, -RATE * max(0.0, t - 0.5), 0.0])     # to the robot's right, at the pelvis (A5)


def font(size):
    return ImageFont.load_default(size=size)


def overlay(scn, model, data, st, start_outline):
    """The pull (yellow arrow at the pelvis), the floor's total push (aqua arrow standing on the CoP) and where
    the feet started (blue outlines on the floor)."""
    z = 0.003
    for poly in start_outline:
        for a, b in zip(poly, np.roll(poly, -1, axis=0)):
            C._line(scn, [*a, z], [*b, z], 0.004, BLUE)
    f = pull(data.time)
    if np.linalg.norm(f) > 1:
        p = data.xipos[model.body("pelvis").id]
        C._line(scn, p, p + 0.006 * f, 0.02, YELLOW, mujoco.mjtGeom.mjGEOM_ARROW)    # from the pelvis, like a rope
    if st.cop is not None and st.total[2] > 1:
        C._line(scn, [*st.cop, z], np.array([*st.cop, z]) + 0.0016 * st.total, 0.022, AQUA, mujoco.mjtGeom.mjGEOM_ARROW)


def camera(data):
    """Front view, following the robot sideways at 60 % of its motion: a sliding robot drifts across the frame."""
    return make_camera(lookat=[0.0, 0.6 * data.qpos[1], 0.5], distance=2.9, azimuth=180, elevation=-8)


def run(model, data, chain, mu, renderers, duration=6.0, after_fall=1.0):
    """A5's run (same steps as contact_forces.simulate) until a foot leaves the floor, rendering every 1/30 s.

    After that, the balance controller is switched off and the joints are held stiff where they are, so the robot
    falls like a statue. (The LQR was designed for standing on both feet; with one foot in the air it is far
    outside that, its torques saturate and it flings the robot around, which says nothing about friction or the
    support polygon.) Returns ({renderer key: frames}, events: {name: pull in N}, per-frame (pull, first event),
    per-frame (qpos, torso tilt in degrees)).
    """
    feet = [g for g in range(model.ngeom) if re.fullmatch(C.g1_utils.FOOT_PATTERN, model.geom(g).name)]
    model.geom_friction[feet, 0] = mu
    mujoco.mj_resetData(model, data)
    data.qpos[:] = chain["q"][0]
    mujoco.mj_forward(model, data)
    ctrl, pelvis, torso = C.ChainController(model, chain), model.body("pelvis").id, model.body("torso_link").id
    qa = model.jnt_qposadr[model.actuator_trnid[:, 0]]           # each motor's joint (motor order != joint order)
    da = model.jnt_dofadr[model.actuator_trnid[:, 0]]
    start = [C.foot_outline(model, data, s) for s in C.SIDES]
    feet0 = {s: data.site(f"{s}_foot").xpos[:2].copy() for s in C.SIDES}
    frames = {k: [] for k in renderers}
    info, poses, events, off_since, fell_at, hold = [], [], {}, None, None, None
    every = round(1 / (FPS_IN * model.opt.timestep))
    for i in range(round(duration / model.opt.timestep)):
        if hold is None:
            ctrl(data, 0.0, 0.0)
        else:                                                     # statue: stiff joints around the held pose
            u0, q_hold = hold
            data.ctrl[:] = u0 + 400.0 * (q_hold - data.qpos[qa]) - 10.0 * data.qvel[da]
        data.xfrc_applied[:] = 0.0
        data.xfrc_applied[pelvis, :3] = pull(data.time)
        mujoco.mj_step(model, data)
        st = C.contact_state(model, data)
        force = -pull(data.time)[1]
        off_since = (off_since or data.time) if st.load["left"] < 1.0 else None
        if off_since is not None and data.time - off_since >= 0.05 and "lift" not in events:
            events["lift"] = -pull(off_since)[1]
            hold = (data.ctrl.copy(), data.qpos[qa].copy())
        if np.linalg.norm(data.site("right_foot").xpos[:2] - feet0["right"]) > 0.005 and "slide" not in events:
            events["slide"] = force
        if fell_at is None and C.fallen(model, data, chain["q"][0][2]):
            fell_at = data.time
        if i % every == 0:
            for key, r in renderers.items():
                r.update_scene(data, camera=camera(data))
                overlay(r.scene, model, data, st, start)
                frames[key].append(Image.fromarray(r.render()))
            info.append((force, min(events, key=events.get) if events else None))
            poses.append((data.qpos.copy(), np.degrees(np.arccos(np.clip(data.xmat[torso][8], -1, 1)))))
        if fell_at is not None and data.time > fell_at + after_fall:
            break
    model.geom_friction[feet, 0] = 0.6
    return frames, events, info, poses


def setup():
    model, data, joints = F.build_torque_g1()
    chain = C.build_chain(model, data, joints)
    return model, data, chain


def frames_gif(frames, path: Path, fps: int = 12, colors: int = 96, width: int | None = None):
    """PIL frames (30 fps) -> looping GIF via a lossless MP4 and a shared palette."""
    with tempfile.TemporaryDirectory() as tmp:
        video = Path(tmp) / "clip.mp4"
        w, h = frames[0].size
        ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                               f"{w}x{h}", "-r", str(FPS_IN), "-i", "-", "-c:v", "libx264rgb", "-crf", "0", str(video)],
                              stdin=subprocess.PIPE)
        for f in frames:
            ff.stdin.write(f.convert("RGB").tobytes())
        ff.stdin.close()
        ff.wait()
        scale = f"scale={width}:-2:flags=lanczos," if width else ""
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(video), "-vf",
                        f"fps={fps},{scale}split[a][b];[a]palettegen=max_colors={colors}:stats_mode=diff[p];"
                        "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle", "-loop", "0", str(path)],
                       check=True)
    print(f"saved {path.relative_to(REPO)}: {path.stat().st_size / 1e6:.2f} MB", flush=True)


def label_tile(img, title, force, outcome, scale=1.0):
    d = ImageDraw.Draw(img)
    s = lambda v: round(v * scale)
    d.text((s(14), s(10)), title, fill=(255, 255, 255), font=font(s(24)))
    d.text((s(14), s(42)), f"pull {force:3.0f} N", fill=(255, 205, 90), font=font(s(18)))
    if outcome:
        d.text((s(14), s(68)), outcome, fill=(120, 235, 185), font=font(s(22)))
    return img


def pad(frames, n):
    return frames + [frames[-1]] * (n - len(frames))


def make_clips(do_blog=True, do_linkedin=True):
    model, data, chain = setup()
    tile, li_w, li_h = 440, 1080, 600
    model.vis.global_.offwidth, model.vis.global_.offheight = max(tile, li_w), max(tile, li_h)
    renderers = {}
    if do_blog:
        renderers["blog"] = mujoco.Renderer(model, height=tile, width=tile)
    if do_linkedin:
        renderers["li"] = mujoco.Renderer(model, height=li_h, width=li_w)
    runs = {}
    for mu, name, outcome in FRICTIONS:
        frames, events, info, _ = run(model, data, chain, mu, renderers)
        print(f"mu = {mu}: " + ", ".join(f"{k} at {v:.0f} N" for k, v in sorted(events.items(), key=lambda kv: kv[1])))
        runs[mu] = (frames, events, info, name, outcome)
    for r in renderers.values():
        r.close()
    # the physics must be A5's: on the grippy floor the left foot lifts first, on ice the robot slides first
    assert min(runs[0.6][1], key=runs[0.6][1].get) == "lift" and min(runs[0.1][1], key=runs[0.1][1].get) == "slide"

    n = max(len(r[2]) for r in runs.values())
    tiles = {}
    for key in renderers:
        tiles[key] = []
        for mu, (frames, events, info, name, outcome) in runs.items():
            out = []
            for k in range(n):
                j = min(k, len(info) - 1)
                force, first = info[j]
                happened = first == ("lift" if mu == 0.6 else "slide")
                img = frames[key][j].copy()
                label_tile(img, f"{name} (friction {mu})", force, outcome if happened else "",
                           scale=1.0 if key == "blog" else 1.9)
                out.append(img)
            tiles[key].append(out)

    if do_blog:
        OUT.mkdir(parents=True, exist_ok=True)
        grid = []
        for a, b in zip(*tiles["blog"]):
            canvas = Image.new("RGB", (2 * tile + 6, tile), (18, 20, 33))
            canvas.paste(a, (0, 0))
            canvas.paste(b, (tile + 6, 0))
            grid.append(canvas)
        frames_gif(grid, OUT / "slide_or_tip.gif", fps=12, colors=96)
    if do_linkedin:
        OUT_LI.mkdir(parents=True, exist_ok=True)
        portrait = []
        for a, b in zip(*tiles["li"]):
            canvas = Image.new("RGB", (li_w, 1350), (18, 20, 33))
            d = ImageDraw.Draw(canvas)
            d.text((44, 26), "Same pull, two floors", fill=(255, 255, 255), font=font(64))
            d.text((48, 104), "Unitree G1  ·  MuJoCo  ·  pulled sideways, harder every second",
                   fill=(215, 220, 240), font=font(30))
            canvas.paste(a, (0, 150))
            canvas.paste(b, (0, 150 + li_h))
            portrait.append(canvas)
        frames_gif(portrait, OUT_LI / "slide_or_tip.gif", fps=12, colors=64)


def banner(preview_dir=None):
    """The robot tipping over on the grippy floor, as ghosts: level camera from the front, ghosts on the right."""
    W, H = 1280, 400
    model, data, chain = setup()
    model.vis.global_.offwidth, model.vis.global_.offheight = W, H
    _, _, _, poses = run(model, data, chain, 0.6, {})            # the tipping run, as in the clip
    tilts = np.array([t for _, t in poses])
    states = [poses[int(np.argmax(tilts >= angle))][0] for angle in (0, 12, 28, 45, 62)]   # spread by how far it leans
    floor = model.geom("floor").id
    cam = make_camera(lookat=[0.0, 0.0, 0.5], distance=3.3, azimuth=0, elevation=-4)   # from behind: tips to the right
    xs, alphas = [0.56, 0.66, 0.75, 0.84, 0.925], [0.5, 0.6, 0.72, 0.85, 1.0]      # spread out, like Week 5's
    with mujoco.Renderer(model, height=H, width=W) as r:
        data.qpos[:] = chain["q"][0]
        data.qpos[0] += 100.0                                   # empty scene for the background
        mujoco.mj_forward(model, data)
        r.update_scene(data, camera=cam)
        img = np.asarray(r.render()).astype(float)
        for q, x, alpha in zip(states, xs, alphas):
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            r.update_scene(data, camera=cam)
            shot = np.asarray(r.render()).astype(float)
            r.enable_segmentation_rendering()
            seg = r.render()
            r.disable_segmentation_rendering()
            mask = (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (seg[..., 0] != floor)
            cx = int(np.mean(np.nonzero(mask)[1]))               # centre each robot, then place it at x
            dx = round(x * W) - cx
            shot, mask = np.roll(shot, dx, axis=1), np.roll(mask, dx, axis=1)
            if dx > 0:
                mask[:, :dx] = False
            img = np.where(mask[..., None], img * (1 - alpha) + shot * alpha, img)
    OUT.mkdir(parents=True, exist_ok=True)
    out = Image.fromarray(img.clip(0, 255).astype(np.uint8))
    out.save(OUT / "banner.png")
    print("saved banner.png")
    if preview_dir:
        prev = Image.fromarray((img * 0.6).clip(0, 255).astype(np.uint8))
        d = ImageDraw.Draw(prev)
        d.text((60, 140), "Week 6 - Six Weeks of Humanoid Walking", fill=(250, 250, 255), font=font(44))
        d.text((60, 205), "What I built, what I learned, and the contact forces underneath it all.",
               fill=(225, 225, 235), font=font(19))
        prev.save(Path(preview_dir) / "banner_week6_preview.png")


if __name__ == "__main__":
    args = sys.argv[1:] or ["clip", "linkedin", "banner"]
    preview = next((a.split("=", 1)[1] for a in args if a.startswith("preview=")), None)
    if "clip" in args or "linkedin" in args:
        make_clips(do_blog="clip" in args, do_linkedin="linkedin" in args)
    if "banner" in args:
        banner(preview)
