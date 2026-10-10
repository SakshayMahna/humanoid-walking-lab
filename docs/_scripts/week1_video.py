"""Generate a short LinkedIn video (MP4, 1080x1350) for the Week 1 post.

Run from the repo root:  .venv/bin/python docs/_scripts/week1_video.py
Needs ffmpeg on PATH. Output: docs/assets/videos/week-1/pd_sweeps.mp4
Physics matches exercises/02_pendulum_pd (via week1_assets.py).
"""

import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from week1_assets import BAD, BG, FG, L, MOTOR, REPO, TARGET, font, make_pd, pendulum_xml, zeta  # noqa: E402

OUT = REPO / "docs" / "assets" / "videos" / "week-1"
OUT.mkdir(parents=True, exist_ok=True)

W, H, FPS = 1080, 1350, 30
HEADER_H, GAP = 130, 8
COLORS = [(255, 99, 132), (255, 159, 64), (255, 206, 86), (160, 220, 90), (75, 192, 160),
          (54, 200, 235), (54, 120, 235), (153, 102, 255), (230, 110, 230)]


def simulate(label, color, control, duration, pw, ph):
    """Simulate one pendulum; return a list of panel images at FPS (real time)."""
    model = mujoco.MjModel.from_xml_string(pendulum_xml(color, MOTOR))
    data = mujoco.MjData(model)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, -0.28]
    cam.distance, cam.azimuth, cam.elevation = 1.25, 90, -8
    render_h = int(ph * 0.62)
    dt = model.opt.timestep
    ts, qs, frames = [], [], []
    with mujoco.Renderer(model, height=render_h, width=pw) as r:
        for i in range(round(duration / dt)):
            control(model, data)
            mujoco.mj_step(model, data)
            ts.append(data.time)
            qs.append(data.qpos[0])
            assert not any(data.warning[b].number for b in BAD)
            if len(frames) < data.time * FPS:
                r.update_scene(data, camera=cam)
                frames.append(panel(Image.fromarray(r.render()), ts, qs, color, label, duration, pw, ph))
    return frames


def panel(top, ts, qs, color, label, duration, pw, ph):
    img = Image.new("RGB", (pw, ph), BG)
    img.paste(top, (0, 0))
    d = ImageDraw.Draw(img)
    d.text((10, 8), label, fill=FG, font=font(22))
    d.text((10, 36), f"angle {qs[-1]:.3f}", fill=color, font=font(20))

    y_min, y_max = 0.0, 0.9
    x0, x1, y0, y1 = 44, pw - 12, top.height + 12, ph - 26
    px = lambda t: x0 + (x1 - x0) * t / duration
    py = lambda q: y1 - (y1 - y0) * (q - y_min) / (y_max - y_min)
    d.rectangle((x0, y0, x1, y1), outline=(70, 75, 100))
    yt = py(TARGET)
    for x in range(x0, x1, 10):
        d.line((x, yt, x + 5, yt), fill=(200, 200, 210), width=2)
    d.text((6, yt - 9), "0.5", fill=(200, 200, 210), font=font(16))
    d.text((6, py(0) - 16), "0", fill=(150, 150, 170), font=font(16))
    d.text((x1 - 110, y1 + 4), f"time 0-{duration:g} s", fill=(150, 150, 170), font=font(16))
    if len(qs) > 1:
        d.line([(px(t), py(min(max(q, y_min), y_max))) for t, q in zip(ts, qs)], fill=color, width=3)
    return img


def scene(title, subtitle, runs, cols, duration=5.0, hold=1.5):
    """runs: list of (label, control). Returns full-size frames."""
    rows = (len(runs) + cols - 1) // cols
    pw = (W - (cols + 1) * GAP) // cols
    ph = (H - HEADER_H - (rows + 1) * GAP) // rows
    panels = [simulate(lbl, COLORS[i * len(COLORS) // len(runs)], ctl, duration, pw, ph)
              for i, (lbl, ctl) in enumerate(runs)]
    n = max(len(p) for p in panels)
    frames = []
    for i in range(n + int(hold * FPS)):
        canvas = Image.new("RGB", (W, H), (10, 11, 20))
        d = ImageDraw.Draw(canvas)
        d.text((24, 22), title, fill=FG, font=font(44))
        d.text((24, 78), subtitle, fill=(170, 175, 200), font=font(26))
        for k, plist in enumerate(panels):
            r, c = divmod(k, cols)
            canvas.paste(plist[min(i, len(plist) - 1)], (GAP + c * (pw + GAP), HEADER_H + GAP + r * (ph + GAP)))
        frames.append(canvas)
    return frames


def title_card(lines, seconds=2.0):
    img = Image.new("RGB", (W, H), (10, 11, 20))
    d = ImageDraw.Draw(img)
    y = H // 2 - 40 * len(lines)
    for text, size, color in lines:
        tw = d.textlength(text, font=font(size))
        d.text(((W - tw) / 2, y), text, fill=color, font=font(size))
        y += size + 30
    return [img] * int(seconds * FPS)


def thumbnail(frame, path):
    """Dim a settled frame and put a bold title band across the middle."""
    img = Image.blend(frame, Image.new("RGB", (W, H), (10, 11, 20)), 0.35)
    d = ImageDraw.Draw(img)
    band_y0, band_y1 = H // 2 - 170, H // 2 + 170
    d.rectangle((0, band_y0, W, band_y1), fill=(10, 11, 20))
    d.rectangle((0, band_y0, W, band_y0 + 6), fill=COLORS[0])
    d.rectangle((0, band_y1 - 6, W, band_y1), fill=COLORS[6])
    lines = [("24 pendulums", 96, FG), ("Why don't they reach the target?", 46, COLORS[2]),
             ("PD control in MuJoCo", 36, (170, 175, 200))]
    y = band_y0 + 40
    for text, size, color in lines:
        tw = d.textlength(text, font=font(size))
        d.text(((W - tw) / 2, y), text, fill=color, font=font(size))
        y += size + 28
    img.save(path)
    print(f"saved {path.relative_to(REPO)}")


def pd_gravity(kp, kd, m=1.0, g=9.81):
    def control(model, data):
        q = data.qpos[0]
        data.ctrl[0] = kp * (TARGET - q) - kd * data.qvel[0] + m * g * L * np.sin(q)
    return control


if __name__ == "__main__":
    frames = title_card([
        ("One pendulum, 24 experiments", 56, FG),
        ("PD control in MuJoCo", 34, (170, 175, 200)),
        ("target angle = 0.5 rad", 30, (200, 200, 210)),
    ])

    kps = [5, 10, 20, 40, 80, 160, 320, 640, 1280]
    frames += scene("1. Stiffer spring (KP up)", "KD = 2.  Closer to target, but more overshoot",
                    [(f"KP={kp}  zeta={zeta(kp, 2):.2f}", make_pd(kp, 2)) for kp in kps], cols=3)
    thumbnail(frames[-1], OUT / "thumbnail.png")

    kds = [0.25, 0.5, 1, 2, 3.5, 5, 8, 12, 20]
    frames += scene("2. More damping (KD up)", "KP = 20.  Less overshoot, then too slow",
                    [(f"KD={kd:g}  zeta={zeta(20, kd):.2f}", make_pd(20, kd)) for kd in kds], cols=3)

    runs = []
    for kp in (5, 20, 80):
        runs += [(f"KP={kp}  PD only", make_pd(kp, 2 * np.sqrt(kp / 20))),
                 (f"KP={kp}  + gravity", pd_gravity(kp, 2 * np.sqrt(kp / 20)))]
    frames += scene("3. Gravity compensation", "Add the gravity torque: the error disappears", runs, cols=2)

    path = OUT / "pd_sweeps.mp4"
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                           "-crf", "20", "-preset", "slow", "-movflags", "+faststart", str(path)],
                          stdin=subprocess.PIPE)
    for f in frames:
        ff.stdin.write(f.tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"saved {path.relative_to(REPO)}: {len(frames) / FPS:.1f} s, {path.stat().st_size / 1e6:.2f} MB")
