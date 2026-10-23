"""Q3 summary: outcome and policy-step timing per group (policy, setup, clock, scenario), repeats listed.

Outcome: standing = distance moved in 30 s; walking = forward speed and sideways speed from 8 s (cmd vx 0.5 from 4 s).
Timing, from <name>.steps.csv (one row per policy step: wall time, sim time), in SIMULATED time, since that is what
the robot experiences:
  start    the first 1.5 s of sim time after the first step: steps taken (20 ms period -> 75), and the shortest
           and longest gap between steps
  steady   after that: share of gaps outside 19-21 ms, and the longest gap

    .venv/bin/python open_questions/q3_clock/analyze.py recordings/q3/*.csv
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

START = 1.5          # s of sim time counted as start-up


def outcome(path, scenario):
    d = np.genfromtxt(path, delimiter=",", names=True)
    t = d["t"] - d["t"][0]
    if d["z"].min() < 0.4:
        return "fell"
    if scenario == "stand":
        return f"{np.hypot(d['x'][-1] - d['x'][0], d['y'][-1] - d['y'][0]):5.2f} m"
    w = t >= t[d["cmd_vx"] != 0][0] + 4.0
    return f"{d['vx_body'][w].mean():.2f}/{d['vy_body'][w].mean():+.2f}"


def timing(path):
    d = np.genfromtxt(path, delimiter=",", names=True)
    sim = d["sim"][np.isfinite(d["sim"])]
    t = sim - sim[0]
    gaps = np.diff(sim) * 1000                                  # ms
    start = t[1:] <= START
    steady = ~start
    return {"start_steps": int((t <= START).sum()), "start_min": gaps[start].min(), "start_max": gaps[start].max(),
            "steady_off": np.mean(np.abs(gaps[steady] - 20) > 1) * 100, "steady_max": gaps[steady].max()}


def main(paths):
    groups = defaultdict(list)
    for p in map(Path, paths):
        if p.name.endswith((".steps.csv", ".odom.csv")):
            continue
        m = re.fullmatch(r"(.+)_(stand|walk)_(\d+)", p.stem)
        groups[(m.group(1), m.group(2))].append(p)
    print(f"{'run':38s} {'':5s} | {'result per repeat (stand: moved; walk: vx/vy)':46s} | "
          f"start: steps, gap min-max (ms) | steady: % off 20 ms, max gap")
    for (name, scenario), runs in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        results = [outcome(p, scenario) for p in sorted(runs)]
        times = [timing(p.with_suffix(".steps.csv")) for p in sorted(runs)]
        start = " ".join(f"{x['start_steps']}:{x['start_min']:.0f}-{x['start_max']:.0f}" for x in times)
        steady = " ".join(f"{x['steady_off']:.1f}%/{x['steady_max']:.0f}" for x in times)
        print(f"{name:38s} {scenario:5s} | {'  '.join(results):46s} | {start:30s} | {steady}")


if __name__ == "__main__":
    main(sys.argv[1:])
