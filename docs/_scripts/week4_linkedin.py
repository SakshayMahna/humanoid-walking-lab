"""LinkedIn GIF (1080x1350) for the Week 4 post: the reward ladder, one tile per step, all told to walk at 0.5 m/s.

Uses the tiles rendered by week4_videos.py (recordings/week4/<policy>.mp4: stand 4 s, then forward 0.5 m/s).
Run from the repo root:  .venv/bin/python docs/_scripts/week4_linkedin.py
Output: docs/assets/videos/week-4/reward_ladder.gif
"""

import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from week2_assets import BG, DIM, FG, REPO, font  # noqa: E402

TILES = REPO / "recordings" / "week4"
OUT = REPO / "docs" / "assets" / "videos" / "week-4"
W, H, FPS, START, SECONDS = 1080, 1350, 8, 3.0, 6.0      # 1 s standing, then 5 s told to walk
HEADER_H, GAP, CAPTION_H, COLS = 150, 8, 74, 2
SRC_W, SRC_H, SRC_Y0, SRC_Y1 = 640, 360, 70, 318           # crop the empty sky and the burned-in command text
# (step, reward added, what the robot learned, policy tile, colour: red -> green as it learns to walk)
LADDER = [("STEP 0", "follow the command", "lies on its back", "ladder_0", (255, 99, 132)),
          ("STEP 1", "+ stay upright", "sits up on its arms", "ladder_1", (255, 159, 64)),
          ("STEP 2", "+ keep the posture", "sits upright", "ladder_2", (255, 206, 86)),
          ("STEP 3", "+ move smoothly", "stands up, shuffles", "ladder_3", (160, 220, 90)),
          ("STEP 4", "+ gait clock", "takes real steps", "ladder_4", (75, 192, 160)),
          ("ALL 16", "+ 9 polishing terms", "walks smoothly", "baseline", (54, 162, 235))]

ROWS = (len(LADDER) + COLS - 1) // COLS
TW = (W - (COLS + 1) * GAP) // COLS
TH = (H - HEADER_H - (ROWS + 1) * GAP) // ROWS
VH = TH - CAPTION_H


def read_frames(path):
    crop_h = SRC_Y1 - SRC_Y0
    crop_w = min(round(crop_h * TW / VH), SRC_W)
    vf = (f"fps={FPS},crop={crop_w}:{crop_h}:{(SRC_W - crop_w) // 2}:{SRC_Y0},scale={TW}:{VH}:flags=lanczos,"
          "eq=gamma=1.35:brightness=0.03")
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", str(START), "-t", str(SECONDS), "-i", str(path),
                          "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], check=True, capture_output=True).stdout
    size = TW * VH * 3
    return [Image.frombuffer("RGB", (TW, VH), raw[i * size:(i + 1) * size]) for i in range(len(raw) // size)]


def tile(frame, step, added, learned, color, walking):
    img = Image.new("RGB", (TW, TH), BG)
    img.paste(frame, (0, CAPTION_H))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, TW, 5), fill=color)
    sw = d.textlength(step, font=font(26))
    d.text((14, 12), step, fill=color, font=font(26))
    d.text((24 + sw, 13), added, fill=FG, font=font(25))
    d.text((14, 44), learned, fill=color, font=font(24))
    if not walking:
        d.text((TW - 92, CAPTION_H + 10), "standing", fill=DIM, font=font(18))
    return img


if __name__ == "__main__":
    clips = [read_frames(TILES / f"{name}.mp4") for _, _, _, name, _ in LADDER]
    n = min(len(c) for c in clips)
    frames = []
    for i in range(n):
        walking = START + i / FPS >= 4.0
        canvas = Image.new("RGB", (W, H), (10, 11, 20))
        d = ImageDraw.Draw(canvas)
        d.text((24, 20), "What teaches a robot to walk?", fill=FG, font=font(54))
        sub = "Rewards added one by one  ·  " + ("told to walk at 0.5 m/s" if walking else "standing, about to walk")
        d.text((24, 92), sub, fill=DIM, font=font(30))
        for k, (step, added, learned, _, color) in enumerate(LADDER):
            r, c = divmod(k, COLS)
            canvas.paste(tile(clips[k][i], step, added, learned, color, walking),
                         (GAP + c * (TW + GAP), HEADER_H + GAP + r * (TH + GAP)))
        frames.append(canvas)

    # Encode: frames -> lossless MP4 -> GIF with one palette for the whole clip (keeps the step colours exact)
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp()) / "ladder.mp4"
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264rgb", "-crf", "0", str(tmp)], stdin=subprocess.PIPE)
    for f in frames:
        ff.stdin.write(f.tobytes())
    ff.stdin.close()
    ff.wait()
    gif = OUT / "reward_ladder.gif"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(tmp), "-vf",
                    "split[a][b];[a]palettegen=max_colors=40:stats_mode=full[p];[b][p]paletteuse=dither=none:diff_mode=rectangle",
                    "-loop", "0", str(gif)], check=True)
    print(f"saved {gif.relative_to(REPO)}: {n} frames, {gif.stat().st_size / 1e6:.2f} MB")
