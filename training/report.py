"""Evaluation report for trained G1 policies: how they walk, not just whether they follow the command.

For each run folder (with a policy.onnx) and Unitree's released policy as a reference, in plain MuJoCo (Unitree's
model, 5 ms physics, policy every 20 ms, as evaluate_policy.py):

  tracking     forward 0.2 / 0.5 / 1.0, sideways 0.3, turn 0.5 in place, stand still (from evaluate_policy.py)
  gait         at 0.5 m/s: steps per second, time in the air per step, foot clearance, foot slip, double support
               (contact changes shorter than 20 ms are ignored: a foot brushing the floor is not a step)
  effort       at 0.5 m/s: action rate, joint acceleration, joint torque, mechanical power
  posture      at 0.5 m/s: torso tilt, pelvis height, distance from the default pose
  robustness   pushes (base velocity +-0.8 m/s, forward and sideways, while walking), floor friction 0.3 / 1.2,
               no armature (joints without the motors' reflected inertia, as in Gazebo)
  training     final mean reward, episode length and per-term rewards, from the run's training log

    .venv/bin/python training/report.py training/kaggle/output/v5/logs/rsl_rl/g1_velocity/*/
Writes recordings/training/report.md (one column per policy) and prints it.
"""

import re
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "exercises"))
sys.path.insert(0, str(REPO / "training"))
from evaluate_policy import COMMANDS, SECONDS, run as track, unstable  # noqa: E402
from policy_utils import load_policy  # noqa: E402
from unitree_utils import HOME, UNITREE_POLICY, build_unitree_model, make_unitree_observation  # noqa: E402

DECIMATION = 4
OUT = REPO / "recordings" / "training" / "report.md"


class Robot:
    """The model plus the indices the measurements need."""

    def __init__(self):
        self.model, self.data, self.joints, self.actuators = build_unitree_model()
        m = self.model
        self.floor = m.geom("floor").id
        self.foot_geoms = {side: {m.geom(f"{side}_foot{k}_collision").id for k in range(1, 8)}
                           for side in ("left", "right")}
        self.foot_site = {side: m.site(f"{side}_foot").id for side in ("left", "right")}
        self.armature = m.dof_armature.copy()
        self.friction = {g: m.geom_friction[g, 0] for side in self.foot_geoms.values() for g in side}

    def contacts(self):
        touching = {"left": False, "right": False}
        for c in self.data.contact[:self.data.ncon]:
            other = c.geom2 if c.geom1 == self.floor else c.geom1 if c.geom2 == self.floor else None
            for side, geoms in self.foot_geoms.items():
                if other in geoms:
                    touching[side] = True
        return touching

    def reset_physics(self):
        self.model.dof_armature[:] = self.armature
        for g, mu in self.friction.items():
            self.model.geom_friction[g, 0] = mu


def walk(robot, policy, command, seconds=14.0, measure_from=6.0, push=None):
    """Run the policy; return per-step logs over the measuring window, and whether it fell.
    push: (time, [dvx, dvy]) adds that velocity to the base once (in the world frame)."""
    m, d = robot.model, robot.data
    mujoco.mj_resetDataKeyframe(m, d, m.key("init_state").id)
    mujoco.mj_forward(m, d)
    last_action = np.zeros(len(policy["default_pos"]))
    log = {k: [] for k in ("t", "contact_l", "contact_r", "foot_l", "foot_r", "vfoot_l", "vfoot_r", "action_rate",
                           "qacc", "torque", "power", "tilt", "height", "pose")}
    fell, prev_action = False, None
    for step in range(round(seconds / m.opt.timestep)):
        if push is not None and abs(d.time - push[0]) < m.opt.timestep / 2:
            d.qvel[0:2] += push[1]
        if step % DECIMATION == 0:
            obs = policy["obs_fn"](m, d, policy["default_pos"], last_action, np.asarray(command, float))
            action = policy["session"].run(None, {policy["input_name"]: obs[None, :]})[0][0]
            d.ctrl[:] = (policy["default_pos"] + policy["action_scale"] * action)[policy["joint_index_of_actuator"]]
            if d.time >= measure_from and prev_action is not None:
                log["action_rate"].append(float(np.sum((action - prev_action) ** 2)))
            prev_action, last_action = action, action
        mujoco.mj_step(m, d)
        if unstable(d):                    # MuJoCo blew up and reset the robot: nothing valid to measure
            return None, True
        fell |= d.qpos[2] < 0.4
        if d.time < measure_from:
            continue
        touching = robot.contacts()
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, d.qpos[3:7])
        up = R.reshape(3, 3)[:, 2]
        vel = np.zeros(6)
        log["t"].append(d.time)
        for side, key in (("left", "l"), ("right", "r")):
            log[f"contact_{key}"].append(touching[side])
            log[f"foot_{key}"].append(d.site_xpos[robot.foot_site[side]][2])
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, robot.foot_site[side], vel, 0)
            log[f"vfoot_{key}"].append(float(np.linalg.norm(vel[3:5])))       # horizontal speed, world frame
        log["qacc"].append(float(np.sqrt(np.mean(d.qacc[6:] ** 2))))
        log["torque"].append(float(np.sqrt(np.mean(d.actuator_force ** 2))))
        log["power"].append(float(np.sum(np.abs(d.actuator_force * d.qvel[6:][policy["joint_index_of_actuator"]]))))
        log["tilt"].append(float(np.degrees(np.arccos(np.clip(up[2], -1, 1)))))
        log["height"].append(float(d.qpos[2]))
        log["pose"].append(float(np.sqrt(np.mean((d.qpos[7:] - policy["default_pos"]) ** 2))))
    return {k: np.array(v) for k, v in log.items()}, fell


def debounce(c, min_len):
    """Ignore contact changes that last less than min_len samples (a foot brushing the floor isn't a step)."""
    c = c.copy()
    for value in (0, 1):                       # remove short gaps, then short touches
        i = 0
        while i < len(c):
            if c[i] == value:
                j = i
                while j < len(c) and c[j] == value:
                    j += 1
                if 0 < i and j < len(c) and j - i < min_len:
                    c[i:j] = 1 - value
                i = j
            else:
                i += 1
    return c


def gait(log, dt):
    """Steps per second (touchdowns of both feet), air time per step, clearance, slip, double support."""
    steps, air, clearance, slip = 0, [], [], []
    for key in ("l", "r"):
        c = debounce(log[f"contact_{key}"].astype(int), round(0.02 / dt))
        z = log[f"foot_{key}"]
        stance_z = np.median(z[c == 1]) if c.any() else z.min()
        changes = np.flatnonzero(np.diff(c))
        lifts = [i + 1 for i in changes if c[i] == 1]
        downs = [i + 1 for i in changes if c[i] == 0]
        steps += len(downs)
        for start in lifts:
            ends = [j for j in downs if j > start]
            if ends:
                air.append((ends[0] - start) * dt)
                clearance.append(z[start:ends[0]].max() - stance_z)
        slip.append(log[f"vfoot_{key}"][c == 1].mean() if c.any() else np.nan)
    duration = len(log["t"]) * dt
    double = np.mean(log["contact_l"] & log["contact_r"])
    return {"steps_per_s": steps / duration, "air_time": np.mean(air) if air else np.nan,
            "clearance_cm": 100 * np.mean(clearance) if clearance else np.nan,
            "slip_cm_s": 100 * np.nanmean(slip), "double_support_pct": 100 * double}


def training_log(run_dir):
    """Final values from the run's training log (train_<run>.log, next to logs/ in the downloaded output)."""
    run = run_dir.name.split("_", 2)[2] if run_dir.name.count("_") >= 2 else run_dir.name
    candidates = list(run_dir.parents[3].glob(f"train_{run}.log")) if len(run_dir.parents) > 3 else []
    if not candidates:
        return {}
    text = candidates[0].read_text()
    last = text[text.rfind("Learning iteration"):]
    values = dict(re.findall(r"^\s*([\w/]+(?: \w+)*):\s+(-?[\d.]+)\s*$", last, flags=re.M))
    out = {"iterations": re.search(r"Learning iteration (\d+)/", last).group(1) if "Learning iteration" in last else "?",
           "mean reward": values.get("Mean reward"), "episode length": values.get("Mean episode length")}
    out.update({k.split("/", 1)[1]: v for k, v in values.items() if k.startswith("Episode_Reward/")})
    return out


def evaluate(robot, path):
    policy = load_policy(path, robot.joints, robot.actuators, pose=HOME, obs_fn=make_unitree_observation)
    m, d = robot.model, robot.data
    dt = m.opt.timestep
    rows = {}
    robot.reset_physics()
    for name in ("stand", "vx 0.2", "vx 0.5", "vx 1.0", "vy 0.3", "wz 0.5"):
        vx, vy, wz, moved, fell = track(m, d, policy, COMMANDS[name], SECONDS)
        rows[f"tracking: {name}"] = ("unstable" if fell and np.isnan(moved) else "fell" if fell else f"{moved:.2f} m" if name == "stand" else
                                     f"{wz:+.2f} rad/s" if name.startswith("wz") else
                                     f"{vy:+.2f} m/s" if name.startswith("vy") else f"{vx:+.2f} m/s")
    log, fell = walk(robot, policy, [0.5, 0, 0])
    if log is None or fell:
        rows["gait, effort, posture at 0.5 m/s"] = "unstable" if log is None else "fell"
        return rows
    g = gait(log, dt)
    rows.update({"gait: steps per second": f"{g['steps_per_s']:.2f}", "gait: air time per step (s)": f"{g['air_time']:.2f}",
                 "gait: foot clearance (cm)": f"{g['clearance_cm']:.1f}", "gait: foot slip in stance (cm/s)": f"{g['slip_cm_s']:.1f}",
                 "gait: double support (%)": f"{g['double_support_pct']:.0f}",
                 "effort: action rate": f"{np.mean(log['action_rate']):.3f}",
                 "effort: joint acceleration RMS (rad/s²)": f"{np.mean(log['qacc']):.1f}",
                 "effort: torque RMS (N·m)": f"{np.mean(log['torque']):.1f}",
                 "effort: mechanical power (W)": f"{np.mean(log['power']):.0f}",
                 "posture: torso tilt (deg)": f"{np.mean(log['tilt']):.1f}",
                 "posture: pelvis height (m)": f"{np.mean(log['height']):.3f}",
                 "posture: distance from default pose (rad RMS)": f"{np.mean(log['pose']):.3f}"})
    falls = []
    for dv in ([0.8, 0], [-0.8, 0], [0, 0.8], [0, -0.8]):
        falls.append(walk(robot, policy, [0.5, 0, 0], seconds=10.0, measure_from=9.9, push=(6.0, np.array(dv)))[1])
    rows["robustness: pushes survived (of 4)"] = f"{4 - sum(falls)}"
    for mu in (0.3, 1.2):
        robot.reset_physics()
        for g_id in robot.friction:
            m.geom_friction[g_id, 0] = mu
        vx, vy, wz, moved, fell = track(m, d, policy, COMMANDS["vx 0.5"], SECONDS)
        rows[f"robustness: friction {mu}, vx 0.5"] = ("unstable" if np.isnan(moved) else "fell") if fell else f"{vx:+.2f} m/s"
    robot.reset_physics()
    m.dof_armature[:] = 0.0
    vx, vy, wz, moved, fell = track(m, d, policy, COMMANDS["stand"], SECONDS)
    rows["robustness: no armature, stand"] = ("unstable" if np.isnan(moved) else "fell") if fell else f"{moved:.2f} m"
    vx, vy, wz, moved, fell = track(m, d, policy, COMMANDS["vx 0.5"], SECONDS)
    rows["robustness: no armature, vx 0.5"] = ("unstable" if np.isnan(moved) else "fell") if fell else f"{vx:+.2f} m/s"
    robot.reset_physics()
    return rows


def main(run_dirs):
    robot = Robot()
    columns = {"Unitree (released)": (UNITREE_POLICY, {})}
    for r in map(Path, run_dirs):
        label = r.name.split("_", 2)[2] if r.name.count("_") >= 2 else r.name
        columns[label] = (r / "policy.onnx", training_log(r))
    results = {}
    for label, (path, train) in columns.items():
        print(f"evaluating {label} ...", flush=True)
        results[label] = evaluate(robot, path)
        results[label].update({f"training: {k}": v for k, v in train.items() if v is not None})
    names = list(dict.fromkeys(k for r in results.values() for k in r))
    lines = ["| | " + " | ".join(results) + " |", "|---|" + "---|" * len(results)]
    lines += ["| " + n + " | " + " | ".join(str(results[c].get(n, "")) for c in results) + " |" for n in names]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main(sys.argv[1:])
