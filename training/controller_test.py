"""Our trained policies under explicit vs implicit joint damping, in plain MuJoCo (Unitree's model and gains).

Week 3's controller test (open_questions/q1b_controller/mujoco_controller_test.py) ran Unitree's policy with the
joint torques computed by the script itself, so only the controller changes between runs. This runs the same test
for our trained policies, with Unitree's gains and torque limits only:

  dterm     vel   tau = kp (target - q) - kd q_dot, an explicit torque     (mjlab's IdealPdActuator)
            jnt   tau = kp (target - q), kd as joint damping, implicit      (MuJoCo's position actuator in training)
            err   tau = -kp e - kd (e - e_prev) / dt, e = q - target         (Gazebo's JointPositionController PID)
  armature  on / off
  step      1 ms (Gazebo) / 5 ms (training), policy every 20 ms

Scenarios: stand (zero command, 30 s) and walk (4 s standing, then 0.5 m/s). CSVs in recordings/controller_test/,
summary table printed and written to recordings/controller_test/summary.md.

    .venv/bin/python training/controller_test.py
"""

import itertools
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training"))
sys.path.insert(0, str(REPO / "open_questions" / "q1b_controller"))
import mujoco_controller_test as q1b  # noqa: E402
from policy_io import load_trained  # noqa: E402
from unitree_utils import build_unitree_model, load_unitree_policy  # noqa: E402

RUNS = REPO / "training" / "kaggle" / "output"
OUT = REPO / "recordings" / "controller_test"
POLICIES = {   # label -> run folder (None: Unitree's released policy)
    "unitree": None,
    "baseline_s1": "v6/logs/rsl_rl/g1_velocity/2026-10-26_03-05-28_baseline_seed1",
    "baseline_s2": "v15/logs/rsl_rl/g1_velocity/2026-10-27_10-29-22_baseline_seed2",
    "loo_foot_gait": "v9/logs/rsl_rl/g1_velocity/2026-10-26_09-24-37_loo_foot_gait_seed1",
    "dr_arm_height_s1": "v18/logs/rsl_rl/g1_velocity/2026-10-27_18-47-24_dr_armature_height_seed1",
    "dr_arm_height_s2": "v18/logs/rsl_rl/g1_velocity/2026-10-27_18-47-23_dr_armature_height_seed2",
}
STEPS = {"1ms": (0.001, 20), "5ms": (0.005, 4)}


def summarize(csv_path, scenario):
    d = np.genfromtxt(csv_path, delimiter=",", names=True)
    fell = bool((d["z"] < 0.5).any())
    if scenario == "stand":
        moved = np.hypot(d["x"][-1] - d["x"][0], d["y"][-1] - d["y"][0])
        return "falls" if fell else f"{moved:.2f} m"
    walking = d["t"] >= 8.0
    return "falls" if fell else f"{d['vx_body'][walking].mean():+.2f} m/s"


def main():
    model, data, joints, actuators = build_unitree_model()
    model.actuator_gainprm[:, 0] = 0.0      # the script applies the joint torques itself
    model.actuator_biasprm[:, 0:3] = 0.0
    armature = model.dof_armature.copy()
    results = {}
    for label, run_dir in POLICIES.items():
        policy = (load_unitree_policy(joints, actuators) if run_dir is None
                  else load_trained(RUNS / run_dir / "policy.onnx", joints, actuators))
        for step, arm, dterm in itertools.product(STEPS, ["on", "off"], ["jnt", "vel", "err"]):
            q1b.DT, q1b.DECIMATION = STEPS[step]
            model.opt.timestep = q1b.DT
            model.dof_armature[:] = armature if arm == "on" else 0.0
            for scenario in q1b.SCENARIOS:
                name = q1b.run(model, data, policy, joints, "policy", "policy", dterm, arm, scenario,
                               out=OUT, prefix=f"{label}_{step}_")
                results[(label, step, arm, dterm, scenario)] = summarize(OUT / f"{name}.csv", scenario)
            print(label, step, f"armature {arm}", dterm,
                  results[(label, step, arm, dterm, "stand")], results[(label, step, arm, dterm, "walk")], flush=True)

    names = {"jnt": "implicit", "vel": "explicit", "err": "explicit + kick"}
    lines = ["| Policy | Step | Armature | " + " | ".join(f"{names[d]}: stand / walk" for d in names) + " |",
             "|---|---|---|" + "---|" * len(names)]
    for label, step, arm in itertools.product(POLICIES, STEPS, ["on", "off"]):
        cells = [f"{results[(label, step, arm, d, 'stand')]} / {results[(label, step, arm, d, 'walk')]}"
                 for d in names]
        lines.append(f"| {label} | {step} | {arm} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    (OUT / "summary.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
