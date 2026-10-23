"""Q2: are the robot's links the same in Gazebo and MuJoCo? Mass and principal moments of inertia per link.

Gazebo: Diego's model.sdf (g1_29dof). MuJoCo: robot/g1.xml (walk.onnx's training model, mjlab's G1) and Unitree's
g1.xml. Principal moments (eigenvalues of the inertia matrix) don't depend on how each file orients its link
frames, so they can be compared directly. Also the total mass. (Armature is not in these XMLs: our model builders
add it from the motor models.)

    .venv/bin/python open_questions/q2_walk_onnx/compare_inertias.py
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "exercises"))
from unitree_utils import UNITREE_XML  # noqa: E402

SDF = REPO / "3p/2025-tfg-diego-lopez/g1_sim/models/g1_29dof/model.sdf"
MJ_MODELS = {"mjlab (walk.onnx)": REPO / "robot" / "g1.xml", "Unitree": UNITREE_XML}


def sdf_links(path):
    """link -> (mass, sorted principal moments) from the SDF."""
    links = {}
    for link in ET.parse(path).getroot().iter("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        i = {k: float(inertial.find(f"inertia/{k}").text) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")}
        matrix = np.array([[i["ixx"], i["ixy"], i["ixz"]], [i["ixy"], i["iyy"], i["iyz"]],
                           [i["ixz"], i["iyz"], i["izz"]]])
        links[link.get("name")] = (float(inertial.find("mass").text), np.sort(np.linalg.eigvalsh(matrix)))
    return links


def mujoco_links(path):
    """body -> (mass, sorted principal moments) from a MuJoCo XML."""
    model = mujoco.MjModel.from_xml_path(str(path))
    return {model.body(b).name: (model.body_mass[b], np.sort(model.body_inertia[b])) for b in range(1, model.nbody)}


def main():
    gz = sdf_links(SDF)
    for label, path in MJ_MODELS.items():
        mj = mujoco_links(path)
        common = [name for name in gz if name in mj]
        missing = sorted(set(gz) ^ set(mj))
        print(f"\n=== Gazebo SDF vs MuJoCo {label}: {len(common)} links in both; only in one: {missing or 'none'}")
        print(f"total mass: Gazebo {sum(m for m, _ in gz.values()):.3f} kg, MuJoCo {sum(m for m, _ in mj.values()):.3f} kg")
        print(f"{'link':28s} {'mass gz':>8s} {'mass mj':>8s}   principal moments gz/mj (sorted)")
        worst = []
        for name in common:
            (m_gz, i_gz), (m_mj, i_mj) = gz[name], mj[name]
            ratio = i_gz / np.maximum(i_mj, 1e-12)
            dev = max(abs(np.log(ratio)))
            worst.append((dev, name))
            flag = "  <--" if abs(m_gz / m_mj - 1) > 0.02 or dev > np.log(1.05) else ""
            print(f"{name:28s} {m_gz:8.4f} {m_mj:8.4f}   {' '.join(f'{r:6.3f}' for r in ratio)}{flag}")
        print("largest differences:", ", ".join(f"{n} ({np.exp(d):.2f}x)" for d, n in sorted(worst)[-3:]))


if __name__ == "__main__":
    main()
