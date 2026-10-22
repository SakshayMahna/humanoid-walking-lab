"""Q2 summary: one row per run (MuJoCo factorial or Gazebo logs), standing and walking side by side.

Pairs <name>_stand[_n].csv with <name>_walk[_n].csv when both exist, averaging repeats _1, _2, ...; any other CSV (e.g. Gazebo's stand-4-s-then-walk
logs) gives the walking columns only. Adds the pelvis height, because crouching is one of walk.onnx's symptoms.

    .venv/bin/python open_questions/q2_walk_onnx/summarize.py recordings/q2/*.csv recordings/gains/*walk*.csv
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SETTLE = 4.0        # s after the command starts before measuring the walk
FELL = 0.4          # m: pelvis below this = fell


def load(path):
    d = np.genfromtxt(path, delimiter=",", names=True)
    return d, d["t"] - d["t"][0], np.degrees(np.unwrap(d["yaw"]))


def stand_metrics(path):
    d, t, yaw = load(path)
    return {"move": np.hypot(d["x"][-1] - d["x"][0], d["y"][-1] - d["y"][0]), "rot": abs(yaw[-1] - yaw[0]),
            "z_stand": d["z"][t >= 5].mean(), "fell_s": d["z"].min() < FELL}


def walk_metrics(path):
    d, t, yaw = load(path)
    moving = d["cmd_vx"] != 0
    start = t[moving][0]
    w = t >= start + SETTLE
    fit = np.polyfit(t[w], yaw[w], 1)
    return {"vx": d["vx_body"][w].mean(), "vy": d["vy_body"][w].mean(), "turn": fit[0],
            "wander": np.std(yaw[w] - np.polyval(fit, t[w])), "z_walk": d["z"][w].mean(),
            "fell_w": d["z"].min() < FELL}


def main(paths):
    runs = defaultdict(lambda: defaultdict(list))     # name -> scenario -> [metrics per repeat]
    for p in map(Path, paths):
        if p.stem.startswith("trace_") or p.name.endswith(".odom.csv"):
            continue
        stem = re.sub(r"_\d+$", "", p.stem)              # drop the repeat number
        if stem.endswith("_stand"):
            runs[stem[:-6]]["stand"].append(stand_metrics(p))
        elif stem.endswith("_walk"):
            runs[stem[:-5]]["walk"].append(walk_metrics(p))
        else:
            runs[stem]["walk"].append(walk_metrics(p))
    print(f"{'run':40s} {'n':>3s} {'move':>6s} {'rot':>7s} {'z_st':>5s} | {'vx':>5s} {'vy':>6s} {'turn':>6s} "
          f"{'wander':>6s} {'z_wk':>5s}")
    for name in sorted(runs):
        stands, walks = runs[name]["stand"], runs[name]["walk"]
        n = f"{len(stands)}/{len(walks)}" if stands else f"{len(walks)}"
        cells = []
        if stands:
            fell = sum(s["fell_s"] for s in stands)
            ok = [s for s in stands if not s["fell_s"]]
            cells.append(f"fell {fell}/{len(stands)}".ljust(20) if not ok else
                         f"{mean(ok, 'move'):6.2f} {mean(ok, 'rot'):7.1f} {mean(ok, 'z_stand'):5.2f}"
                         + (f" (+{fell} fell)" if fell else ""))
        else:
            cells.append(" " * 20)
        if walks:
            fell = sum(m["fell_w"] for m in walks)
            ok = [m for m in walks if not m["fell_w"]]
            cells.append(f"fell {fell}/{len(walks)}" if not ok else
                         f"{mean(ok, 'vx'):5.2f} {mean(ok, 'vy'):6.2f} {mean(ok, 'turn'):6.2f} "
                         f"{mean(ok, 'wander'):6.2f} {mean(ok, 'z_walk'):5.2f}" + (f" (+{fell} fell)" if fell else ""))
        print(f"{name:40s} {n:>3s} " + " | ".join(cells))
    print("stand: 30 s zero command, move m, |rot| deg, z_st mean pelvis height after 5 s (m)\n"
          "walk: cmd vx 0.5 from 4 s, measured from 8 s: vx, vy m/s (pelvis frame), turn deg/s, wander deg, "
          "z_wk mean pelvis height (m). Repeats averaged; n = stand/walk runs")


def mean(runs, key):
    return float(np.mean([r[key] for r in runs]))


if __name__ == "__main__":
    main(sys.argv[1:])
