"""What does walk.onnx see as its velocity (base_lin_vel), and is it right? Reads <name>.odom.csv files from Gazebo
(every odometry message, gazebo_repeats.sh) or MuJoCo (mujoco_velocity_log.py, velocimeter).

Per run, over the steady part (standing: 5-30 s; walking: from 4 s after the command):
  seen vx, vy, vz   mean of the input the policy received (pelvis frame, m/s)
  true vx, vy       mean velocity of the same point from the logged pose: (p(t) - p(t - 10 ms)) / 10 ms, rotated
                    into the pelvis frame. If the input is right, seen = true.
  slope             least-squares fit seen = slope * true, sample by sample (vx while walking, else vy);
                    1 = right scale and frame
  shake x, y, z     std of the input around its own 0.1 s moving average (m/s): fast vibration the policy sees

Repeats (_1, _2, ...) are averaged.

    .venv/bin/python open_questions/q2_walk_onnx/velocity_analysis.py recordings/q2/*.odom.csv
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

WINDOW = 0.010      # s: finite-difference window, like Gazebo's odometry average
SMOOTH = 0.1        # s: moving average that separates "motion" from "shake"


def rotation(qw, qx, qy, qz):
    """Rotation matrices (N, 3, 3), body -> world, from unit quaternions."""
    return np.stack([
        np.stack([1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)], -1),
        np.stack([2 * (qx * qy + qw * qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qw * qx)], -1),
        np.stack([2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx**2 + qy**2)], -1)], -2)


def metrics(path, scenario):
    d = np.genfromtxt(path, delimiter=",", names=True)
    t = d["t"] - d["t"][0]
    seen = np.stack([d["vx"], d["vy"], d["vz"]], -1)
    p = np.stack([d["x"], d["y"], d["z"]], -1)
    R = rotation(d["qw"], d["qx"], d["qy"], d["qz"])
    j = np.clip(np.searchsorted(t, t - WINDOW), 0, len(t) - 1)       # sample ~10 ms earlier
    dt = t - t[j]
    ok = dt > 0.5 * WINDOW
    v_world = np.zeros_like(p)
    v_world[ok] = (p[ok] - p[j[ok]]) / dt[ok, None]
    true = np.einsum("nji,nj->ni", R, v_world)                        # R^T v: into the pelvis frame
    w = ok & ((t >= 5) & (t <= 30) if scenario == "stand" else (t >= 8))

    k = max(1, round(SMOOTH / np.median(np.diff(t))))
    smooth = np.stack([np.convolve(seen[:, i], np.ones(k) / k, mode="same") for i in range(3)], -1)
    inner = w & (np.arange(len(t)) >= k) & (np.arange(len(t)) < len(t) - k)
    axis = 0 if scenario == "walk" else 1
    a, b = true[w, axis], seen[w, axis]
    return {"seen": seen[w].mean(0), "true": true[w, :2].mean(0), "slope": (a @ b) / (a @ a),
            "shake": (seen[inner] - smooth[inner]).std(0), "rate": 1 / np.median(np.diff(t))}


def main(paths):
    groups = defaultdict(list)
    for p in map(Path, paths):
        name = re.sub(r"_\d+$", "", p.name.removesuffix(".odom.csv"))   # drop the repeat number
        scenario = "stand" if name.endswith("_stand") else "walk"
        groups[(name.removesuffix(f"_{scenario}"), scenario)].append(metrics(p, scenario))
    print(f"{'run':36s} {'':5s} {'n':>2s} | {'seen vx':>7s} {'vy':>6s} {'vz':>6s} | {'true vx':>7s} {'vy':>6s} | "
          f"{'slope':>5s} | {'shake x':>7s} {'y':>6s} {'z':>6s}")
    for (name, scenario), runs in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        m = {key: np.mean([r[key] for r in runs], axis=0) for key in runs[0]}
        print(f"{name:36s} {scenario:5s} {len(runs):2d} | {m['seen'][0]:7.3f} {m['seen'][1]:6.3f} {m['seen'][2]:6.3f} | "
              f"{m['true'][0]:7.3f} {m['true'][1]:6.3f} | {m['slope']:5.2f} | "
              f"{m['shake'][0]:7.3f} {m['shake'][1]:6.3f} {m['shake'][2]:6.3f}")
    print("m/s, pelvis frame. stand: 5-30 s of zero command; walk: from 8 s (cmd vx 0.5 from 4 s). "
          "slope: seen vs true, vx (walk) / vy (stand)")


if __name__ == "__main__":
    main(sys.argv[1:])
