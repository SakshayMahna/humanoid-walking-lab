"""LinkedIn GIF (1080x1350) for the Week 5 post: the G1 on one foot, flamingo -> tree -> warrior III -> flamingo.

The flow is week5_assets.py's clip_flow: same model, chain, gains and run_flow. Only the rendering differs: a true
portrait render (not an upscale), a camera framed for it, and a large live pose label.

Run from the repo root:  .venv/bin/python docs/_scripts/week5_linkedin.py [--preview]
Output: docs/assets/videos/week-5/yoga_flow.gif
"""

import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import week5_assets as A  # noqa: E402
from week5_assets import F, REPO, mujoco  # noqa: E402

OUT = REPO / "docs" / "assets" / "videos" / "week-5"
W, H, FPS_IN, FPS, SPEED, COLORS = 1080, 1350, 30, 12, 1.5, 56
SHOWN = {"flamingo": "Flamingo", "tree": "Tree", "warrior III": "Warrior III"}


def font(size):
    return ImageFont.load_default(size=size)


if __name__ == "__main__":
    model, data, joints, ctx = A._flamingo_setup()
    chain = F.build_flow_chain(model, data, joints, ctx)
    chain["sequence"] = ["flamingo", "tree", "warrior III", "flamingo"]
    model.vis.global_.offwidth, model.vis.global_.offheight = W, H

    work = Path(tempfile.mkdtemp())
    video = work / "flow.mp4"
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS_IN), "-i", "-", "-c:v", "libx264rgb", "-crf", "0", str(video)],
                          stdin=subprocess.PIPE)

    def overlay(img, text):
        """Replaces run_flow's small label; frames go straight to ffmpeg rather than filling memory."""
        pose = text.split("   ")[-1].strip()
        d = ImageDraw.Draw(img)
        d.text((44, 40), SHOWN[pose], fill=(255, 255, 255), font=font(92))
        d.text((48, 150), "Unitree G1 on one foot  ·  LQR, no neural network", fill=(215, 220, 240), font=font(34))
        d.text((48, H - 70), "1.5x speed", fill=(215, 220, 240), font=font(30))
        ff.stdin.write(img.tobytes())
        return None

    # run_flow's physics is untouched; these only change what it draws
    Renderer, plain_camera = mujoco.Renderer, F.make_camera
    F.label = overlay
    F.make_camera = lambda **kw: plain_camera(lookat=[0, 0, 0.62], distance=3.0, azimuth=140, elevation=-8)
    mujoco.Renderer = lambda m, height, width: Renderer(m, height=H, width=W)
    fell, worst, moved, _ = F.run_flow(model, data, chain, record=True)
    mujoco.Renderer = Renderer
    ff.stdin.close()
    ff.wait()
    print(f"flow: fell {fell}, largest joint error {worst:.3f} rad, foot moved {moved:.2f} mm")
    assert not fell

    OUT.mkdir(parents=True, exist_ok=True)
    gif = OUT / "yoga_flow.gif"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(video), "-vf",
                    f"setpts=PTS/{SPEED},fps={FPS},split[a][b];[a]palettegen=max_colors={COLORS}:stats_mode=diff[p];"
                    "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle", "-loop", "0", str(gif)], check=True)
    print(f"saved {gif.relative_to(REPO)}: {gif.stat().st_size / 1e6:.2f} MB")
