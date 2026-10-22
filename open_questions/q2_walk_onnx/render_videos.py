"""Videos of Test 1 (walk.onnx, MuJoCo, 1 ms steps) as labelled 2x2 grids, standing and walking.

Same four setups as the Gazebo videos (gazebo_record.sh), so the two grids can be compared tile by tile:
original controller with / without armature, corrected controller with / without armature.
Same simulation as mujoco_factorial.py (it calls the same run() and films it), so the labels match its CSVs.
Output: recordings/q2/video_walk_<config>_<scenario>.mp4 and recordings/q2/mujoco_grid_{stand,walk}.mp4

    .venv/bin/python open_questions/q2_walk_onnx/render_videos.py      (from the repo root; needs ffmpeg)
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "q1b_controller"))
from render_videos import VideoWriter, scenery  # noqa: E402  (also puts exercises/ on sys.path)
from mujoco_controller_test import DT, REPO, SCENARIOS, run  # noqa: E402
from g1_utils import build_g1_model  # noqa: E402
from policy_utils import DIEGO_POLICY, load_policy  # noqa: E402

OUT = REPO / "recordings" / "q2"

# (label, armature, gains, limits, D term, Luis's ankle extras)
CONFIGS = [
    ("1. Original controller", "off", "luis", "luis", "err", True),
    ("2. Original + armature", "on", "luis", "luis", "err", True),
    ("3. Corrected controller", "off", "policy", "policy", "jnt", False),
    ("4. Corrected + armature", "on", "policy", "policy", "jnt", False),
]


def main():
    model, data, joints, actuators = build_g1_model(REPO / "robot" / "g1.xml", decorate=scenery)
    model.opt.timestep = DT
    model.actuator_gainprm[:, 0] = 0.0
    model.actuator_biasprm[:, 0:3] = 0.0
    armature = model.dof_armature.copy()
    policy = load_policy(DIEGO_POLICY, joints, actuators)

    for scenario in SCENARIOS:
        tiles = []
        for label, arm, gains, limits, dterm, extras in CONFIGS:
            model.dof_armature[:] = armature if arm == "on" else 0.0
            video = OUT / f"video_walk_arm{arm}_{gains}gains_{dterm}_{scenario}.mp4"
            writer = VideoWriter(model, video)
            name = run(model, data, policy, joints, gains, limits, dterm, arm, scenario, on_step=writer,
                       extras=extras, out=OUT, prefix="walk_")
            writer.close()
            tiles.append(f"{label}|{video}|{OUT / name}.csv")
            print("rendered", video.name)

        seconds = SCENARIOS[scenario][0]
        what = "zero command for 30 s" if scenario == "stand" else "stand 4 s, then walk at 0.5 m/s"
        subprocess.run([sys.executable, str(REPO / "gazebo" / "side_by_side.py"),
                        str(OUT / f"mujoco_grid_{scenario}.mp4"),
                        "--title", f"walk.onnx in MuJoCo, 1 ms steps ({what})",
                        "--seconds", str(seconds), "--cols", "2", *tiles], check=True)


if __name__ == "__main__":
    main()
