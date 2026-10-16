"""Summarize odometry CSVs from run_scenario.sh / gain_experiment.sh (stand STAND s, then a constant command).

    .venv/bin/python gazebo/summarize_runs.py recordings/gains/*.csv

Per run: standing rotation/drift, walking forward/lateral speed (pelvis frame), turn rate and heading wander.
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

STAND = 4.0         # s of zero command at the start
SETTLE = 4.0        # s after the command starts before measuring the walk


def metrics(path):
    d = np.genfromtxt(path, delimiter=",", names=True)
    t = d["t"] - d["t"][0]
    yaw = np.degrees(np.unwrap(d["yaw"]))
    if "cmd_vx" in d.dtype.names:   # newer logs record the command: standing = before it starts
        moving = (d["cmd_vx"] != 0) | (d["cmd_vy"] != 0) | (d["cmd_wz"] != 0)
        start = t[moving][0] if moving.any() else t[-1]
    else:                           # older logs: command timing was approximate
        start = STAND
    s = t < start
    w = t >= start + SETTLE
    if w.sum() < 2:
        raise ValueError(f"{path}: run too short, walking metrics start {SETTLE} s after the command ({start:.1f} s)")
    fit = np.polyfit(t[w], yaw[w], 1)
    return {
        "stand_rot": abs(yaw[s][-1] - yaw[s][0]),                                      # deg
        "stand_move": np.hypot(d["x"][s][-1] - d["x"][s][0], d["y"][s][-1] - d["y"][s][0]),   # m
        "vx": d["vx_body"][w].mean(),                                                  # m/s, pelvis frame
        "vy": d["vy_body"][w].mean(),
        "turn": fit[0],                                                                # deg/s
        "wander": np.std(yaw[w] - np.polyval(fit, t[w])),                              # deg around the trend
        "fell": d["z"].min() < 0.4,
    }


def main(paths):
    groups = defaultdict(list)
    for p in paths:
        groups[Path(p).stem.rsplit("_", 1)[0]].append(metrics(p))   # drop the repeat number
    keys = ["stand_rot", "stand_move", "vx", "vy", "turn", "wander"]
    print(f"{'run':28s} n  " + "  ".join(f"{k:>14s}" for k in keys) + "  falls")
    for name in sorted(groups):
        runs = groups[name]
        cells = []
        for k in keys:
            v = np.array([r[k] for r in runs])
            cells.append(f"{v.mean():7.2f} ±{v.std():5.2f}")
        print(f"{name:28s} {len(runs)}  " + "  ".join(f"{c:>14s}" for c in cells) +
              f"  {sum(r['fell'] for r in runs)}")
    print("stand_rot deg, stand_move m (first 4 s, zero command); vx, vy m/s (cmd 0.5, 0); "
          "turn deg/s; wander deg (std around the linear heading trend)")


if __name__ == "__main__":
    main(sys.argv[1:])
