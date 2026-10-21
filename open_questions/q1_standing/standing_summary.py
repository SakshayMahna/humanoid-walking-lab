"""Q1 test 1: does the robot settle while standing (zero command), or keep moving?

For each CSV: distance from the start position and heading change at several times, and the net movement in
each 10 s window. A start-up transient shows movement early and ~0 late; a sustained cause keeps moving.

    .venv/bin/python open_questions/q1_standing/standing_summary.py recordings/standing/*.csv
"""

import sys
from pathlib import Path

import numpy as np

TIMES = [2, 5, 10, 20, 30]          # s
WINDOWS = [(0, 10), (10, 20), (20, 30)]
SETTLED = 0.05                      # m of net movement in the last window counts as "settled"


def load(path):
    d = np.genfromtxt(path, delimiter=",", names=True)
    t = d["t"] - d["t"][0]
    dist = np.hypot(d["x"] - d["x"][0], d["y"] - d["y"][0])
    yaw = np.degrees(np.unwrap(d["yaw"]))
    return t, d["x"], d["y"], dist, yaw - yaw[0]


def at(t, values, s):
    return values[min(np.searchsorted(t, s), len(t) - 1)]


def window_move(t, x, y, a, b):
    i, j = np.searchsorted(t, a), min(np.searchsorted(t, b), len(t) - 1)
    return np.hypot(x[j] - x[i], y[j] - y[i])


def main(paths):
    print(f"{'run':30s}" + "".join(f"{f'd@{s}s':>8s}" for s in TIMES) + "".join(f"{f'yaw@{s}s':>9s}" for s in TIMES)
          + "".join(f"{f'move {a}-{b}s':>12s}" for a, b in WINDOWS) + "  verdict")
    for p in sorted(paths):
        t, x, y, dist, yaw = load(p)
        moves = [window_move(t, x, y, a, b) for a, b in WINDOWS]
        verdict = "settled" if moves[-1] < SETTLED else "still moving"
        print(f"{Path(p).stem:30s}" + "".join(f"{at(t, dist, s):8.2f}" for s in TIMES)
              + "".join(f"{at(t, yaw, s):9.1f}" for s in TIMES) + "".join(f"{m:12.2f}" for m in moves)
              + f"  {verdict}")
    print("d = distance from start (m), yaw = heading change (deg), move = net displacement in the window (m)")


if __name__ == "__main__":
    main(sys.argv[1:])
