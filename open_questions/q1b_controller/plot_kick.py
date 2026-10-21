"""Plot the left-knee torque over 0.5 s of walking: D on velocity vs Gazebo's D on error (derivative kick).

Reads the traces written by mujoco_controller_test.py (armature on, policy gains and limits).
Output: recordings/q1b/kick.png

    .venv/bin/python open_questions/q1b_controller/plot_kick.py
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
Q1B = REPO / "recordings" / "q1b"
W, H, PAD = 1400, 760, 80
BG, FG, GRID = (16, 16, 32), (230, 230, 240), (60, 60, 90)
COLORS = {"vel": (90, 200, 255), "err": (255, 120, 90)}
LABELS = {"vel": "D on joint velocity (MuJoCo, corrected)", "err": "D on error (Gazebo original): kick"}


def load(dterm):
    d = np.genfromtxt(Q1B / f"trace_armon_policygains_policylimits_{dterm}_walk.csv", delimiter=",", names=True)
    return d["t"] - d["t"][0], d["target"], d["q"], d["tau"]


def panel(draw, font, box, series, y_label, ylim):
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=GRID)
    for k in range(1, 5):
        y = y0 + (y1 - y0) * k / 5
        draw.line([(x0, y), (x1, y)], fill=GRID)
    draw.text((x0, y0 - 28), y_label, fill=FG, font=font)
    lo, hi = ylim
    draw.text((x0 - 70, y0 - 8), f"{hi:.2f}" if hi < 10 else f"{hi:.0f}", fill=FG, font=font)
    draw.text((x0 - 70, y1 - 8), f"{lo:.2f}" if hi < 10 else f"{lo:.0f}", fill=FG, font=font)
    for t, v, color, width in series:
        px = x0 + (x1 - x0) * t / t[-1]
        py = y1 - (y1 - y0) * (np.clip(v, lo, hi) - lo) / (hi - lo)
        draw.line(list(zip(px, py)), fill=color, width=width)


def main():
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=20)
    title = ImageFont.load_default(size=26)
    draw.text((PAD, 20), "Left knee, 0.5 s of walking (MuJoCo, 1 ms steps, policy every 20 ms)", fill=FG, font=title)

    t, target, q_vel, tau_vel = load("vel")
    _, _, q_err, tau_err = load("err")
    angles = np.concatenate([target, q_vel, q_err])
    panel(draw, font, (PAD + 20, 110, W - PAD, 330),
          [(t, target, (200, 200, 200), 2), (t, q_vel, COLORS["vel"], 3), (t, q_err, COLORS["err"], 3)],
          "angle (rad): target from the policy (grey, steps every 20 ms) and the joint", (angles.min(), angles.max()))
    lim = max(abs(tau_vel).max(), abs(tau_err).max())
    panel(draw, font, (PAD + 20, 400, W - PAD, H - 90),
          [(t, tau_err, COLORS["err"], 2), (t, tau_vel, COLORS["vel"], 3)], "torque (N·m)", (-lim, lim))
    draw.text((PAD + 20, H - 70), "time: 0 to 0.5 s", fill=FG, font=font)
    for k, dterm in enumerate(["vel", "err"]):
        x = PAD + 380 + k * 480
        draw.line([(x, H - 58), (x + 40, H - 58)], fill=COLORS[dterm], width=4)
        draw.text((x + 50, H - 70), LABELS[dterm], fill=FG, font=font)
    img.save(Q1B / "kick.png")
    print(f"wrote {Q1B / 'kick.png'}  (peak torque: D-vel {abs(tau_vel).max():.0f} N·m, D-err {abs(tau_err).max():.0f} N·m)")


if __name__ == "__main__":
    main()
