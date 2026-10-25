"""LinkedIn video (MP4, 1080x1350) and banner: Unitree's policy told to stand still in Gazebo, before and after
adding armature. Uses the full-resolution screen recordings (1280x672) and their odometry logs.

Run from the repo root:  .venv/bin/python docs/_scripts/week3_linkedin.py
Output: docs/assets/videos/week-3/linkedin_stand.mp4, docs/assets/videos/week-3/linkedin_banner.png
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from week2_assets import BG, DIM, FG, GREEN, RED, REPO, YELLOW, font  # noqa: E402

OUT = REPO / "docs" / "assets" / "videos" / "week-3"
REC = REPO / "recordings" / "q1b"
W, H, FPS, SPEED, HOLD = 1080, 1350, 30, 2, 2.5
HEADER_H, GAP, SRC_W, SRC_H = 170, 10, 1280, 672
PANEL_H = (H - HEADER_H - 3 * GAP) // 2
RUNS = [("BEFORE", "original setup", RED, "rec_baseline_stand"),
        ("AFTER", "motor inertia (armature) added", GREEN, "rec_armature_stand")]


def read_frames(path):
    """All frames of a recording, cropped to the panel's aspect ratio and scaled to the panel size."""
    crop_w = round(SRC_H * W / PANEL_H)
    vf = f"crop={crop_w}:{SRC_H}:{(SRC_W - crop_w) // 2}:0,scale={W}:{PANEL_H}:flags=lanczos"
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-vf", vf, "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], check=True, capture_output=True).stdout
    n = len(raw) // (W * PANEL_H * 3)
    return [Image.frombuffer("RGB", (W, PANEL_H), raw[i * W * PANEL_H * 3:(i + 1) * W * PANEL_H * 3]) for i in range(n)]


def drift(csv):
    """Distance from the start position over (sim) time."""
    d = np.genfromtxt(csv, delimiter=",", names=True)
    return d["t"] - d["t"][0], np.hypot(d["x"] - d["x"][0], d["y"] - d["y"][0])


def panel(frame, tag, desc, color, moved, sim_t):
    img = frame.copy()
    d = ImageDraw.Draw(img)
    tw = d.textlength(tag, font=font(40))
    d.rectangle((0, 0, tw + 40, 64), fill=color)
    d.text((20, 10), tag, fill=(10, 11, 20), font=font(40))
    d.rectangle((tw + 40, 0, tw + 60 + d.textlength(desc, font=font(28)), 64), fill=BG)
    d.text((tw + 52, 18), desc, fill=FG, font=font(28))
    text = f"moved {moved:.2f} m"
    bw = d.textlength(text, font=font(56))
    d.rectangle((0, PANEL_H - 92, bw + 44, PANEL_H), fill=BG)
    d.rectangle((0, PANEL_H - 92, 8, PANEL_H), fill=color)
    d.text((26, PANEL_H - 82), text, fill=color, font=font(56))
    clock = f"{sim_t:4.1f} s"
    cw = d.textlength(clock, font=font(30))
    d.rectangle((W - cw - 36, PANEL_H - 56, W, PANEL_H), fill=BG)
    d.text((W - cw - 18, PANEL_H - 48), clock, fill=DIM, font=font(30))
    return img


def header(canvas):
    d = ImageDraw.Draw(canvas)
    d.text((32, 24), "Told to stand still for 30 s", fill=FG, font=font(60))
    d.text((32, 104), "Same walking policy in Gazebo  ·  2x speed", fill=DIM, font=font(32))


if __name__ == "__main__":
    runs = []
    for tag, desc, color, name in RUNS:
        frames = read_frames(REC / f"{name}.mp4")
        t, moved = drift(REC / f"{name}.csv")
        runs.append((tag, desc, color, frames, t, moved))

    # The recording and the log both span the run, but each run had its own real-time factor, so both panels
    # are indexed by sim time (frame = linear map of sim time onto the recording) to stay in sync.
    t_end = min(r[4][-1] for r in runs)
    n_out = int(t_end / SPEED * FPS) + 1
    canvases = []
    for i in range(n_out + int(HOLD * FPS)):
        sim_t = min(i * SPEED / FPS, t_end)
        canvas = Image.new("RGB", (W, H), (10, 11, 20))
        header(canvas)
        for k, (tag, desc, color, frames, t, moved) in enumerate(runs):
            j = min(round(sim_t / t[-1] * (len(frames) - 1)), len(frames) - 1)
            canvas.paste(panel(frames[j], tag, desc, color, np.interp(sim_t, t, moved), sim_t),
                         (0, HEADER_H + GAP + k * (PANEL_H + GAP)))
        canvases.append(canvas)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "linkedin_stand.mp4"
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                           "-crf", "20", "-preset", "slow", "-movflags", "+faststart", str(path)],
                          stdin=subprocess.PIPE)
    for c in canvases:
        ff.stdin.write(c.tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"saved {path.relative_to(REPO)}: {len(canvases) / FPS:.1f} s, {path.stat().st_size / 1e6:.2f} MB")

    # Banner: final frame, dimmed, with a title band (same style as Weeks 1 and 2)
    final_moved = [np.interp(t_end, r[4], r[5]) for r in runs]
    img = Image.blend(canvases[-1], Image.new("RGB", (W, H), (10, 11, 20)), 0.35)
    d = ImageDraw.Draw(img)
    y0, y1 = H // 2 - 170, H // 2 + 170
    d.rectangle((0, y0, W, y1), fill=(10, 11, 20))
    d.rectangle((0, y0, W, y0 + 6), fill=RED)
    d.rectangle((0, y1 - 6, W, y1), fill=GREEN)
    y = y0 + 40
    for text, size, color in [(f"It walked {final_moved[0]:.0f} m away", 84, YELLOW),
                              ("A walking policy in Gazebo, told to stand still", 40, FG),
                              ("Before and after the fix", 36, (170, 175, 200))]:
        tw = d.textlength(text, font=font(size))
        d.text(((W - tw) / 2, y), text, fill=color, font=font(size))
        y += size + 30
    img.save(OUT / "linkedin_banner.png")
    print(f"saved docs/assets/videos/week-3/linkedin_banner.png (final drift {final_moved[0]:.2f} / {final_moved[1]:.2f} m)")
