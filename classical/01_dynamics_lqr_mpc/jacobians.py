"""Module 1, Part E: Jacobians on the Unitree G1.

From the cart-pole (2 positions) to the G1: a floating base (the pelvis, free in space) plus 29 joints. A Jacobian
J(q) maps the robot's velocities q_dot to the velocity of one point (a foot, the centre of mass):
    v_point = J(q) q_dot
and its transpose maps a force on that point back to joint torques: tau = J^T F. Every balance, contact and IK
method later in the course is built from these two lines.

    .venv/bin/python classical/01_dynamics_lqr_mpc/jacobians.py
"""

import sys
from pathlib import Path

import matplotlib
import mujoco
import numpy as np
from PIL import Image

matplotlib.use("Agg")   # files only, no window
import matplotlib.pyplot as plt

script_dir = Path(__file__).resolve().parent
repo = script_dir.parents[1]
sys.path.insert(0, str(repo / "exercises"))
from g1_utils import build_g1_model
from render_utils import make_camera, save_gif

results_dir = script_dir / "results"
results_dir.mkdir(exist_ok=True)

FEET = ("left_foot", "right_foot")     # sites under the ankle-roll links
rng = np.random.default_rng(0)


def load_g1():
    """The G1 at its knees-bent standing keyframe (exercise 03), all derived quantities computed."""
    model, data, joint_names, _ = build_g1_model(repo / "robot" / "g1.xml")
    return model, data, joint_names


def random_velocity(model) -> np.ndarray:
    """A random q_dot (nv,): moderate base motion and joint speeds."""
    v = rng.normal(0.0, 1.0, model.nv)
    v[:3] *= 0.3                        # base linear velocity (m/s)
    v[3:6] *= 0.5                       # base angular velocity (rad/s)
    return v


# ---------------------------------------------------------------------------------------------------------
# E1: the floating base
# ---------------------------------------------------------------------------------------------------------

def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Quaternion product a * b, MuJoCo order (w, x, y, z)."""
    aw, av = a[0], a[1:]
    bw, bv = b[0], b[1:]
    return np.concatenate([[aw * bw - av @ bv], aw * bv + bw * av + np.cross(av, bv)])


def quat_from_rotvec(r: np.ndarray) -> np.ndarray:
    """Unit quaternion of a rotation by angle |r| about axis r/|r| (the exponential map)."""
    angle = np.linalg.norm(r)
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * r / angle])


def integrate_pos(model, q: np.ndarray, v: np.ndarray, dt: float) -> np.ndarray:
    """q after moving with velocity v for dt, written out by hand for the G1 (free joint + hinges).

    Why q has 36 numbers but v has 35: the base orientation is stored as a unit quaternion (4 numbers, no
    singularities), but it can only rotate in 3 independent directions, so its velocity is an angular
    velocity (3 numbers). So q_new != q + v dt; each part is updated by its own rule:
        base position (q[0:3]):    + v[0:3] dt        (MuJoCo: linear velocity in the WORLD frame)
        base orientation (q[3:7]): quat * exp(v[3:6] dt)  (angular velocity in the BODY frame: right-multiply)
        hinge angles (q[7:]):      + v[6:] dt
    """
    q_new = q.copy()
    q_new[0:3] += v[0:3] * dt
    q_new[3:7] = quat_mul(q[3:7], quat_from_rotvec(v[3:6] * dt))
    q_new[7:] += v[6:] * dt
    return q_new


def part_e1(model, data, ctx):
    print("\n=== E1: the floating base: 36 positions, 35 velocities ===")
    print(f"nq = {model.nq} (base 3 position + 4 quaternion + 29 joints), nv = {model.nv} (base 3 + 3 + 29 joints)")
    q0 = data.qpos.copy()
    worst = 0.0
    for _ in range(20):
        v, dt = random_velocity(model), 0.05
        q_mine = integrate_pos(model, q0, v, dt)
        q_mj = q0.copy()
        mujoco.mj_integratePos(model, q_mj, v, dt)
        worst = max(worst, np.abs(q_mine - q_mj).max())
    print(f"max |integrate_pos() - mj_integratePos| over 20 random velocities = {worst:.1e}")
    assert worst < 1e-12, "hand-written integration differs from MuJoCo's"

    # Going back: positions -> velocity needs the inverse rule (mj_differentiatePos), not (q2 - q1) / dt
    v, dt = random_velocity(model), 1e-6
    q1 = integrate_pos(model, q0, v, dt)
    v_back = np.zeros(model.nv)
    mujoco.mj_differentiatePos(model, v_back, dt, q0, q1)
    print(f"velocity recovered by mj_differentiatePos: max error {np.abs(v_back - v).max():.1e}")
    assert np.abs(v_back - v).max() < 1e-6


# ---------------------------------------------------------------------------------------------------------
# E2: foot Jacobians
# ---------------------------------------------------------------------------------------------------------

def site_jacobian(model, data, name: str) -> np.ndarray:
    """6 x nv: rows 0-2 linear velocity, rows 3-5 angular velocity (both in the world frame) of a site."""
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, model.site(name).id)
    return np.vstack([jacp, jacr])


def site_pose_at(model, data, q: np.ndarray, name: str):
    """(position, rotation matrix) of a site at configuration q (runs forward kinematics into data)."""
    data.qpos[:] = q
    mujoco.mj_kinematics(model, data)
    return data.site(name).xpos.copy(), data.site(name).xmat.reshape(3, 3).copy()


def part_e2(model, data, ctx):
    print("\n=== E2: foot Jacobians, v_foot = J(q) q_dot ===")
    q0 = data.qpos.copy()
    worst_lin, worst_ang = 0.0, 0.0
    for _ in range(20):
        v, eps = random_velocity(model), 1e-6
        for name in FEET:
            data.qpos[:] = q0
            mujoco.mj_forward(model, data)
            J = site_jacobian(model, data, name)
            R0 = data.site(name).xmat.reshape(3, 3).copy()
            # Finite differences: move a tiny bit forwards and backwards along v, compare foot poses
            q_plus, q_minus = q0.copy(), q0.copy()
            mujoco.mj_integratePos(model, q_plus, v, eps)
            mujoco.mj_integratePos(model, q_minus, v, -eps)
            p_plus, R_plus = site_pose_at(model, data, q_plus, name)
            p_minus, R_minus = site_pose_at(model, data, q_minus, name)
            v_lin = (p_plus - p_minus) / (2 * eps)
            W = (R_plus - R_minus) / (2 * eps) @ R0.T            # dR/dt R^T = [omega]x (skew-symmetric)
            omega = np.array([W[2, 1], W[0, 2], W[1, 0]])
            worst_lin = max(worst_lin, np.abs(J[:3] @ v - v_lin).max())
            worst_ang = max(worst_ang, np.abs(J[3:] @ v - omega).max())
    data.qpos[:] = q0
    mujoco.mj_forward(model, data)
    print(f"J q_dot vs finite differences: linear max error {worst_lin:.1e} m/s, angular {worst_ang:.1e} rad/s")
    assert worst_lin < 1e-6 and worst_ang < 1e-6

    # Structure: which velocities can move the left foot at all?
    J = site_jacobian(model, data, "left_foot")
    names = ["base"] * 6 + [model.joint(i).name.removesuffix("_joint") for i in range(1, model.njnt)]
    moves = [names[i] for i in range(model.nv) if np.abs(J[:, i]).max() > 1e-12]
    print(f"velocities that move the left foot ({len(moves)} of {model.nv}): 6 base + "
          f"{', '.join(n for n in moves if n != 'base')}")
    print(f"base-translation block of the linear rows:\n{np.round(J[:3, :3], 6)}  (identity: move the pelvis "
          f"1 m, the foot moves 1 m)")
    assert len(moves) == 12 and all(n.startswith("left_") for n in moves[6:]), "only the base and the left leg"
    assert np.allclose(J[:3, :3], np.eye(3))


# ---------------------------------------------------------------------------------------------------------
# E3: centre of mass and its Jacobian
# ---------------------------------------------------------------------------------------------------------

def com_and_jacobian(model, data):
    """CoM (3,) and CoM Jacobian (3 x nv), computed from the bodies.

    CoM = sum_i m_i p_i / M   (p_i: body i's own centre of mass, data.xipos)
    Differentiate: c_dot = sum_i m_i p_i_dot / M = (sum_i m_i J_i / M) q_dot, so the CoM Jacobian is the
    mass-weighted average of the bodies' CoM Jacobians.
    """
    masses = model.body_mass[1:]                    # body 0 is the world
    total = masses.sum()
    com = (masses[:, None] * data.xipos[1:]).sum(axis=0) / total
    J = np.zeros((3, model.nv))
    jacp = np.zeros((3, model.nv))
    for b in range(1, model.nbody):
        mujoco.mj_jacBodyCom(model, data, jacp, None, b)
        J += model.body_mass[b] * jacp
    return com, J / total


def com_and_jacobian_com_only(model, data) -> np.ndarray:
    """The CoM alone, from body positions (needs only forward kinematics)."""
    masses = model.body_mass[1:]
    return (masses[:, None] * data.xipos[1:]).sum(axis=0) / masses.sum()


def part_e3(model, data, ctx):
    print("\n=== E3: centre of mass and its Jacobian ===")
    com, J = com_and_jacobian(model, data)
    J_mj = np.zeros((3, model.nv))
    mujoco.mj_jacSubtreeCom(model, data, J_mj, model.body("pelvis").id)    # the pelvis subtree = whole robot
    print(f"total mass {model.body_mass[1:].sum():.2f} kg; CoM = {np.round(com, 4)} m "
          f"(MuJoCo subtree_com: {np.round(data.subtree_com[1], 4)})")
    print(f"max |CoM Jacobian - mj_jacSubtreeCom| = {np.abs(J - J_mj).max():.1e}")
    assert np.allclose(com, data.subtree_com[1], atol=1e-12) and np.allclose(J, J_mj, atol=1e-12)
    assert np.allclose(J[:, :3], np.eye(3)), "moving the whole robot moves its CoM by the same amount"
    # MuJoCo's mj_jacSubtreeCom does the same sum internally, so the check above confirms the bookkeeping, not
    # the maths. Independent check: move the robot a tiny step along random velocities, measure the CoM shift.
    q0, worst = data.qpos.copy(), 0.0
    for _ in range(20):
        v, eps = random_velocity(model), 1e-6
        coms = []
        for sign in (1, -1):
            q = q0.copy()
            mujoco.mj_integratePos(model, q, v, sign * eps)
            data.qpos[:] = q
            mujoco.mj_kinematics(model, data)           # xipos is all the CoM needs
            coms.append(com_and_jacobian_com_only(model, data))
        worst = max(worst, np.abs(J @ v - (coms[0] - coms[1]) / (2 * eps)).max())
    data.qpos[:] = q0
    mujoco.mj_forward(model, data)
    print(f"J_com q_dot vs finite differences of the CoM: max error {worst:.1e} m/s")
    assert worst < 1e-6

    # Which joints move the CoM the most? (size of each joint's column)
    reach = np.linalg.norm(J[:, 6:], axis=0)
    order = np.argsort(reach)[::-1]
    joint = lambda i: model.joint(i + 1).name.removesuffix("_joint")
    print("joints that move the CoM most (cm per rad): " +
          ", ".join(f"{joint(i)} {100 * reach[i]:.1f}" for i in order[:4]))
    print("joints that move it least: " + ", ".join(f"{joint(i)} {100 * reach[i]:.2f}" for i in order[-3:]))
    ctx["com0"] = com


# ---------------------------------------------------------------------------------------------------------
# E4: forces to torques, tau = J^T F
# ---------------------------------------------------------------------------------------------------------

def part_e4(model, data, ctx):
    print("\n=== E4: forces to torques: tau = J^T F ===")
    # Each foot carries half the robot's weight: the ground pushes up with F = (0, 0, M g / 2) at each foot
    weight = model.body_mass[1:].sum() * abs(model.opt.gravity[2])
    F = np.array([0.0, 0.0, weight / 2])
    q_bent = data.qpos.copy()
    q_straight = q_bent.copy()
    q_straight[7:] = 0.0                                            # all joints at zero: legs straight
    pitch_joints = [f"left_{j}_joint" for j in ("hip_pitch", "knee", "ankle_pitch")]

    for label, q in (("knees bent (keyframe)", q_bent), ("legs straight", q_straight)):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        tau = np.zeros(model.nv)
        for name in FEET:
            tau += site_jacobian(model, data, name)[:3].T @ F       # J^T F, linear rows (a pure force)
        # MuJoCo's own: apply the same forces at the same points, read the generalized force it produces.
        # (mj_applyFT computes J^T F internally too, so this checks the call; the independent check is below.)
        tau_mj = np.zeros(model.nv)
        for name in FEET:
            mujoco.mj_applyFT(model, data, F, np.zeros(3), data.site(name).xpos, model.site(name).bodyid[0], tau_mj)
        assert np.allclose(tau, tau_mj, atol=1e-9), "J^T F differs from mj_applyFT"

        # What J^T F means for one hinge: the hinge's Jacobian column is axis x (p - p_joint), so its entry of
        # J^T F is axis . ((p - p_joint) x F): the torque of F about the joint axis. For a vertical force and a
        # pitch axis that is  force x (horizontal distance from the joint to the foot, along x): a lever arm.
        foot = data.site("left_foot").xpos
        parts = []
        for jn in pitch_joints:
            jid = model.joint(jn).id
            lever = data.xanchor[jid] - foot                        # from the foot to the joint
            by_hand = data.xaxis[jid] @ np.cross(foot - data.xanchor[jid], F)
            assert np.isclose(by_hand, tau[model.joint(jn).dofadr[0]], atol=1e-9)
            parts.append(f"{jn.removeprefix('left_').removesuffix('_joint')} {by_hand:+5.1f} N m "
                         f"(lever {100 * lever[0]:+5.1f} cm)")
        print(f"  {label:<22}: " + ", ".join(parts) + f" | base force z {tau[2]:.1f} N")
        # The base rows of J^T F are the total force on the robot: the two feet together carry the whole weight
        assert np.isclose(tau[2], weight)
    print(f"  (J^T F matches mj_applyFT exactly, and each hinge's entry equals axis . (r x F) by hand;")
    print(f"   weight {weight:.1f} N, so {weight / 2:.1f} N per foot. Torque = force x horizontal lever arm: with the")
    print("   legs straight the three pitch joints sit above each other, so they share the foot's ~4 cm offset.)")
    data.qpos[:] = q_bent
    mujoco.mj_forward(model, data)


# ---------------------------------------------------------------------------------------------------------
# E5: first use: move the CoM, keep the feet planted (differential IK, kinematics only)
# ---------------------------------------------------------------------------------------------------------

def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def com_target(t: float, com0: np.ndarray) -> np.ndarray:
    """Down 8 cm (0-1 s), over the left foot (1-2 s), over the right foot (2-3 s), back up to start (3-4 s)."""
    down = -0.08 * (smoothstep(t) - smoothstep(t - 3.0))
    side = 0.06 * (smoothstep(t - 1.0) - 2 * smoothstep(t - 2.0) + smoothstep(t - 3.0))
    return com0 + np.array([0.0, side, down])


def part_e5(model, data, ctx):
    print("\n=== E5: move the CoM with the feet planted (differential IK) ===")
    # Unknown: q_dot (35). Hard requirements, stacked into C q_dot = b:
    #   both feet don't move (12 rows: their 6 x nv Jacobians = 0, plus a small pull back if they drifted),
    #   the pelvis doesn't rotate (3 rows), the waist and arms don't move (17 rows: unit rows).
    # That leaves 3 free directions; the task uses them: J_com q_dot = c_dot_target + k (c_target - c).
    # Solve: q_dot = C^+ b + N (J_com N)^+ (task - J_com C^+ b), N = I - C^+ C (the null space of C).
    dt, k, duration = 0.01, 20.0, 4.0
    q = data.qpos.copy()
    mujoco.mj_forward(model, data)
    feet0 = {n: data.site(n).xpos.copy() for n in FEET}
    upper = [model.joint(n).dofadr[0] for n in (model.joint(i).name for i in range(1, model.njnt))
             if not n.startswith(("left_hip", "left_knee", "left_ankle", "right_hip", "right_knee", "right_ankle"))]
    pelvis = model.body("pelvis").id
    com0, _ = com_and_jacobian(model, data)

    frames_q, log = [], {"t": [], "com": [], "target": [], "foot_err": [], "knee": []}
    for step_i in range(round(duration / dt) + 1):
        t = step_i * dt
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        com, J_com = com_and_jacobian(model, data)
        target = com_target(t, com0)
        target_dot = (com_target(t + 1e-4, com0) - com_target(t - 1e-4, com0)) / 2e-4
        foot_err = max(np.linalg.norm(data.site(n).xpos - feet0[n]) for n in FEET)
        for key, val in (("t", t), ("com", com), ("target", target), ("foot_err", foot_err),
                         ("knee", q[7 + 3])):
            log[key].append(val)
        if step_i % 3 == 0:
            frames_q.append(q.copy())

        rows, b = [], []
        for n in FEET:
            rows.append(site_jacobian(model, data, n))
            b.append(np.concatenate([k * (feet0[n] - data.site(n).xpos), np.zeros(3)]))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, None, jacr, pelvis)
        rows.append(jacr)
        b.append(np.zeros(3))
        sel = np.zeros((len(upper), model.nv))
        sel[np.arange(len(upper)), upper] = 1.0
        rows.append(sel)
        b.append(np.zeros(len(upper)))
        C, b = np.vstack(rows), np.concatenate(b)

        C_pinv = np.linalg.pinv(C)
        v0 = C_pinv @ b                                    # meets the hard requirements
        N = np.eye(model.nv) - C_pinv @ C                  # directions that keep meeting them
        task = target_dot + k * (target - com)
        v = v0 + N @ np.linalg.pinv(J_com @ N) @ (task - J_com @ v0)
        mujoco.mj_integratePos(model, q, v, dt)

    com_err = np.linalg.norm(np.array(log["com"]) - np.array(log["target"]), axis=1)
    print(f"CoM tracking error: max {1000 * com_err.max():.2f} mm; feet moved at most "
          f"{1000 * max(log['foot_err']):.3f} mm; knee angle {min(log['knee']):.2f} .. {max(log['knee']):.2f} rad")
    lo, hi = model.jnt_range[1:, 0], model.jnt_range[1:, 1]
    joints = np.array([fq[7:] for fq in frames_q])
    in_range = np.all((joints >= lo - 1e-6) & (joints <= hi + 1e-6))
    print(f"all joints within their limits: {in_range}")
    assert com_err.max() < 1e-3 and max(log["foot_err"]) < 1e-3 and in_range

    # Plot
    t, com, target = np.array(log["t"]), np.array(log["com"]), np.array(log["target"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    axes[0].plot(t, 100 * (com[:, 2] - com0[2]), label="CoM height")
    axes[0].plot(t, 100 * (target[:, 2] - com0[2]), "--", label="target")
    axes[0].plot(t, 100 * com[:, 1], label="CoM sideways")
    axes[0].plot(t, 100 * target[:, 1], "--", label="target")
    axes[0].set_ylabel("cm"); axes[0].set_xlabel("t (s)"); axes[0].legend(fontsize=8)
    axes[1].plot(t, log["knee"]); axes[1].set_ylabel("left knee (rad)"); axes[1].set_xlabel("t (s)")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / "e5_com_ik.png", dpi=120)
    plt.close(fig)

    # Video: kinematic replay with the CoM (red) and its ground projection (white)
    cam = make_camera(lookat=[0.0, 0.0, 0.55], distance=2.2, azimuth=150, elevation=-12)
    frames = []
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        for fq in frames_q:
            data.qpos[:] = fq
            mujoco.mj_forward(model, data)
            com, _ = com_and_jacobian(model, data)
            renderer.update_scene(data, camera=cam)
            for pos, size, rgba in ((com, 0.03, [0.95, 0.2, 0.15, 1]), ([com[0], com[1], 0.002], 0.02, [1, 1, 1, 1])):
                scn = renderer.scene
                if scn.ngeom < scn.maxgeom:
                    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, [size, 0, 0],
                                        np.asarray(pos, float), np.eye(3).ravel(), np.asarray(rgba, np.float32))
                    scn.ngeom += 1
            frames.append(Image.fromarray(renderer.render()))
    save_gif(frames, results_dir / "e5_com_ik.gif", fps=round(1 / (3 * dt)))
    print(f"saved {results_dir / 'e5_com_ik.png'} and .gif")


PARTS = [part_e1, part_e2, part_e3, part_e4, part_e5]


def main():
    model, data, _ = load_g1()
    ctx = {}
    for part in PARTS:
        part(model, data, ctx)
    print("\n=== Part E complete ===")


if __name__ == "__main__":
    main()
