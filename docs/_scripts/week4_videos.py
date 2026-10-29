"""Training-run clips for the Week 4 blog post, all filmed the same way.

Plain MuJoCo (Unitree's model, 5 ms physics, policy every 20 ms), the camera of training/render_policy.py following
each robot, and the same test: stand 4 s, then walk forward at 0.5 m/s for 16 s. Each clip is joined from tiles with
gazebo/side_by_side.py.

Run from the repo root:  .venv/bin/python docs/_scripts/week4_videos.py [clip ...] [--rerender] [gazebo]
(robots already filmed are reused unless --rerender)
Outputs: looping GIFs in docs/assets/images/week-4/ (tiles and MP4 clips in recordings/week4/).
`gazebo` converts the Gazebo clips recordings/week4/gazebo_{stand,walk}.mp4 into GIFs at 3x speed.
"""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "training"))
import render_policy  # noqa: E402
from policy_io import load_trained  # noqa: E402
from unitree_utils import build_unitree_model  # noqa: E402

render_policy.SEQUENCE = [(4, (0.0, 0.0, 0.0), "stand"), (16, (0.5, 0.0, 0.0), "forward 0.5 m/s")]
RUNS = REPO / "training" / "kaggle" / "output"
TILES = REPO / "recordings" / "week4"
OUT = REPO / "docs" / "assets" / "images" / "week-4"
POLICIES = {
    "ladder_0": "v5/logs/rsl_rl/g1_velocity/2026-10-25_15-26-07_ladder_0_seed1",
    "ladder_1": "v5/logs/rsl_rl/g1_velocity/2026-10-25_15-26-08_ladder_1_seed1",
    "ladder_2": "v6/logs/rsl_rl/g1_velocity/2026-10-26_03-05-28_ladder_2_seed1",
    "ladder_3": "v7/logs/rsl_rl/g1_velocity/2026-10-26_04-41-30_ladder_3_seed1",
    "ladder_4": "v7/logs/rsl_rl/g1_velocity/2026-10-26_04-41-30_ladder_4_seed1",
    "baseline": "v6/logs/rsl_rl/g1_velocity/2026-10-26_03-05-28_baseline_seed1",
    "loo_foot_gait": "v9/logs/rsl_rl/g1_velocity/2026-10-26_09-24-37_loo_foot_gait_seed1",
    "loo_pose": "v10/logs/rsl_rl/g1_velocity/2026-10-26_11-24-49_loo_pose_seed1",
    "loo_action_rate_l2": "v11/logs/rsl_rl/g1_velocity/2026-10-26_13-12-30_loo_action_rate_l2_seed1",
    "dr_armature": "v16/logs/rsl_rl/g1_velocity/2026-10-27_12-38-58_dr_armature_seed1",
    "dr_armature_height": "v18/logs/rsl_rl/g1_velocity/2026-10-27_18-47-24_dr_armature_height_seed1",
    "ladder_0_strict": "v19/logs/rsl_rl/g1_velocity/2026-10-28_09-53-30_ladder_0_strict_seed1",
    "loo_foot_gait_track_hard": "v20/logs/rsl_rl/g1_velocity/2026-10-28_11-49-18_loo_foot_gait_track_hard_seed1",
}
# clip -> (title, [(tile label, policy)])
CLIPS = {
    "step_0": ("Step 0", [("Tracking + fall penalty", "ladder_0")]),
    "step_1": ("Adding body_orientation_l2", [("Step 0 rewards", "ladder_0"),
                                              ("Step 0 rewards + body_orientation_l2", "ladder_1")]),
    "step_2": ("Adding pose", [("Step 1 rewards", "ladder_1"), ("Step 1 rewards + pose", "ladder_2")]),
    "step_3": ("Adding action_rate_l2 and joint_acc_l2", [("Step 2 rewards", "ladder_2"),
                                                          ("Step 2 rewards + smoothness", "ladder_3")]),
    "step_4": ("Adding foot_gait", [("Step 3 rewards", "ladder_3"), ("Step 3 rewards + foot_gait", "ladder_4")]),
    "steps_5_8": ("Adding the last nine terms", [("Step 4 rewards", "ladder_4"), ("All 16 rewards", "baseline")]),
    "without_foot_gait": ("Removing foot_gait", [("All 16 rewards", "baseline"),
                                                  ("All 16 minus foot_gait", "loo_foot_gait")]),
    "without_pose": ("Removing pose", [("All 16 rewards", "baseline"), ("All 16 minus pose", "loo_pose")]),
    "without_action_rate": ("Removing action_rate_l2", [("All 16 rewards", "baseline"),
                                                         ("All 16 minus action_rate_l2", "loo_action_rate_l2")]),
    "armature": ("Random armature in training", [("All 16 + random armature", "dr_armature"),
                                                  ("... + height fall check", "dr_armature_height")]),
    "strict_fall": ("Adding the strict fall check", [("Step 0 rewards", "ladder_0"),
                                                     ("Step 0 rewards + strict fall check", "ladder_0_strict")]),
    "hard_tracking": ("Adding hard speed tracking", [("All 16 minus foot_gait", "loo_foot_gait"),
                                                     ("... + hard speed tracking", "loo_foot_gait_track_hard")]),
}


def main(clips):
    TILES.mkdir(parents=True, exist_ok=True)
    model, data, joints, actuators = build_unitree_model(decorate=render_policy.scenery)
    needed = {name for clip in clips for _, name in CLIPS[clip][1]}
    seconds = 0
    for name in sorted(needed):
        if (TILES / f"{name}.mp4").exists() and "--rerender" not in sys.argv:
            seconds = sum(t for t, _, _ in render_policy.SEQUENCE)
            continue
        policy = load_trained(RUNS / POLICIES[name] / "policy.onnx", joints, actuators)
        seconds = render_policy.render(model, data, policy, TILES / f"{name}.mp4")
        print("rendered", name, flush=True)
    for clip in clips:
        title, tiles = CLIPS[clip]
        video = TILES / f"clip_{clip}.mp4"
        subprocess.run([sys.executable, str(REPO / "gazebo" / "side_by_side.py"), str(video),
                        "--title", f"{title} (told to walk at 0.5 m/s)", "--seconds", str(seconds),
                        *[f"{label}|{TILES / name}.mp4" for label, name in tiles]], check=True)
        width = 800 if len(tiles) > 1 else 640                          # a single tile is not enlarged
        to_gif(video, OUT / f"{clip}.gif", start=2.0, seconds=10.0, width=width)   # 2 s standing, then 8 s walking


def to_gif(video, gif, start=0.0, seconds=None, speed=1.0, fps=10, width=800, dither=True):
    """Looping GIF of an MP4, small enough for the blog (2-3 MB for a 10 s two-tile clip)."""
    use = "dither=bayer:bayer_scale=4" if dither else "dither=none"
    length = ["-t", str(seconds)] if seconds else []
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(start), *length, "-i", str(video), "-vf",
                    f"setpts=PTS/{speed},fps={fps},scale={width}:-1:flags=lanczos,split[a][b];"
                    f"[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse={use}:diff_mode=rectangle",
                    "-loop", "0", str(gif)], check=True)
    print("wrote", gif.name, f"{gif.stat().st_size / 1e6:.1f} MB")


def gazebo_gifs():
    """The Gazebo clips (made by gazebo/side_by_side.py from training/gazebo_repeats.sh runs, see the post) at 3x."""
    for name in ("gazebo_stand", "gazebo_walk"):
        to_gif(TILES / f"{name}.mp4", OUT / f"{name}.gif", speed=3.0, fps=8, width=900, dither=False)


if __name__ == "__main__":
    clips = [a for a in sys.argv[1:] if a not in ("--rerender", "gazebo")]
    if "gazebo" in sys.argv[1:]:
        gazebo_gifs()
    if clips or len(sys.argv) == 1 or "--rerender" in sys.argv[1:]:
        main(clips or list(CLIPS))
