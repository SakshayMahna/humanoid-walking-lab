"""Q2: do Gazebo's feet explain why walk.onnx needs armature in Gazebo?

Gazebo's model (model.sdf) gives each foot 4 tiny spheres (radius 5 mm) at the sole's corners, so the foot touches
the ground in 4 points, and sets no friction (Gazebo's default applies). MuJoCo's model (training) has 7 capsules
(radius 1 cm) covering the sole, friction 0.6. Both soles sit at the same height (z = -0.035 in the ankle-roll frame).
Here MuJoCo's feet are replaced by Gazebo's 4 spheres, with friction 0.6 (training) and 1.0, and the corrected
controller is run with and without armature.

    .venv/bin/python open_questions/q2_walk_onnx/mujoco_gazebo_feet.py
Output: recordings/q2/gzfeet_mu<friction>_*.csv (summarize.py)
"""

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "exercises"))
sys.path.insert(0, str(HERE.parent / "q1b_controller"))
from g1_utils import FOOT_PATTERN, build_g1_model  # noqa: E402
from mujoco_controller_test import DT, SCENARIOS, run  # noqa: E402
from policy_utils import DIEGO_POLICY, load_policy  # noqa: E402

import mujoco  # noqa: E402

SDF = REPO / "3p/2025-tfg-diego-lopez/g1_sim/models/g1_29dof/model.sdf"
OUT = REPO / "recordings" / "q2"


def gazebo_feet(friction):
    """decorate(spec) hook: MuJoCo's foot capsules -> Gazebo's foot spheres (same positions, radius, link frame)."""
    root = ET.parse(SDF).getroot()

    def decorate(spec):
        for geom in [g for g in spec.geoms if re.fullmatch(FOOT_PATTERN, g.name)]:
            spec.delete(geom)
        for side in ("left", "right"):
            link = next(l for l in root.iter("link") if l.get("name") == f"{side}_ankle_roll_link")
            body = spec.body(f"{side}_ankle_roll_link")
            for k, collision in enumerate(link.findall("collision")):
                sphere = collision.find("geometry/sphere")
                pos = [float(v) for v in collision.find("pose").text.split()[:3]]
                body.add_geom(name=f"{side}_foot{k + 1}_collision", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                              size=[float(sphere.find("radius").text), 0, 0], pos=pos,
                              condim=3, priority=1, friction=[friction, 0.005, 0.0001], contype=1, conaffinity=1)
    return decorate


def main():
    for friction in (0.6, 1.0):
        model, data, joints, actuators = build_g1_model(REPO / "robot" / "g1.xml", decorate=gazebo_feet(friction))
        model.opt.timestep = DT
        model.actuator_gainprm[:, 0] = 0.0
        model.actuator_biasprm[:, 0:3] = 0.0
        policy = load_policy(DIEGO_POLICY, joints, actuators)
        armature = model.dof_armature.copy()
        for arm in ("off", "on"):
            model.dof_armature[:] = armature if arm == "on" else 0.0
            for scenario in SCENARIOS:
                print("ran", run(model, data, policy, joints, "policy", "policy", "jnt", arm, scenario, out=OUT,
                                 prefix=f"gzfeet_mu{friction}_walk_"))


if __name__ == "__main__":
    main()
