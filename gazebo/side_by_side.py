"""Combine recordings into one labelled side-by-side video (run on the Mac; needs ffmpeg with drawtext).

Each tile: a label at the top and, if a CSV is given, the run's result at the bottom
("moved 0.42 m, turned 12.3° in 30 s").

    .venv/bin/python gazebo/side_by_side.py OUT.mp4 --title "Standing test, repeat 1" \
        "MuJoCo (training sim)|recordings/standing/stand_mujoco_unitree_1.mp4|recordings/standing/stand_mujoco_unitree_1.csv" \
        "Gazebo: Diego's setup|recordings/standing/stand_diego_unitree_1.mp4|recordings/standing/stand_diego_unitree_1.csv"
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np

TILE_W, TILE_H = 640, 336
TITLE_H = 56
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def result_text(csv_path, seconds):
    d = np.genfromtxt(csv_path, delimiter=",", names=True)
    t = d["t"] - d["t"][0]
    j = min(np.searchsorted(t, seconds), len(t) - 1)
    moved = np.hypot(d["x"][j] - d["x"][0], d["y"][j] - d["y"][0])
    yaw = np.degrees(np.unwrap(d["yaw"]))
    return f"after {t[j]:.0f} s: moved {moved:.2f} m, turned {yaw[j] - yaw[0]:+.1f} deg"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("tiles", nargs="+", help='"label|video.mp4[|run.csv]"')
    parser.add_argument("--title", default="")
    parser.add_argument("--seconds", type=float, default=30.0)
    args = parser.parse_args()

    tmp = Path(tempfile.mkdtemp())
    inputs, filters, names = [], [], []
    for i, tile in enumerate(args.tiles):
        label, video, *csv = tile.split("|")
        inputs += ["-i", video]
        (tmp / f"label{i}.txt").write_text(label)
        chain = (f"[{i}:v]scale={TILE_W}:{TILE_H}:force_original_aspect_ratio=decrease,"
                 f"pad={TILE_W}:{TILE_H}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=25,"
                 f"drawtext=fontfile='{FONT_BOLD}':textfile='{tmp / f'label{i}.txt'}':x=12:y=10:fontsize=24:"
                 f"fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=6")
        if csv:
            (tmp / f"result{i}.txt").write_text(result_text(csv[0], args.seconds))
            chain += (f",drawtext=fontfile='{FONT}':textfile='{tmp / f'result{i}.txt'}':x=12:y=h-th-12:"
                      f"fontsize=20:fontcolor=yellow:box=1:boxcolor=black@0.6:boxborderw=6")
        filters.append(chain + f"[t{i}]")
        names.append(f"[t{i}]")

    stack = f"{''.join(names)}hstack=inputs={len(names)}[row]" if len(names) > 1 else f"{names[0]}copy[row]"
    filters.append(stack)
    (tmp / "title.txt").write_text(args.title)
    filters.append(f"[row]pad=iw:ih+{TITLE_H}:0:{TITLE_H}:color=0x101020,"
                   f"drawtext=fontfile='{FONT_BOLD}':textfile='{tmp / 'title.txt'}':x=(w-tw)/2:y=14:"
                   f"fontsize=26:fontcolor=white[out]")

    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *inputs, "-filter_complex", ";".join(filters),
                    "-map", "[out]", "-t", str(args.seconds), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    args.out], check=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
