"""Module 1, Part G: balancing on tip-toe with LQR (both feet, and one leg).

The same recipe as Part F; only the contact changes. On a flat foot the ankle can move the centre of pressure
around an 18 x 6 cm sole. On tip-toe the robot stands on the rounded front ends of the foot capsules: one point
per foot. A point can't take a balancing torque, so every correction has to come from moving the body above it:
an inverted pendulum on the toes, like the cart-pole.

What changes from Part F: the support point (a toe tip instead of the sole centre), the pose (toe on the floor,
foot tilted heel-up), and what counts as failing (anything but the toe tips touching the floor).

    .venv/bin/python classical/01_dynamics_lqr_mpc/tiptoe_lqr.py
"""

import mujoco
import numpy as np
from PIL import Image

import one_leg_lqr as F

results_dir = F.results_dir

# (feet on tip-toe, foot tilt in degrees). One leg at 20 deg rests on three capsule tips (2 cm wide), at 30 deg on one.
CASES = {"both feet, 30 deg": (("left", "right"), 30.0),
         "one leg, 20 deg": (("left",), 20.0),
         "one leg, 30 deg": (("left",), 30.0)}
DIRECTIONS = F.DIRECTIONS


def toe(model, side: str):
    """(local point, radius): centre of the front end of the middle foot capsule, in the ankle-roll link's frame.

    A capsule is a cylinder with rounded ends along its local z axis; the front (toe) end is at -z here. The toe
    tip touches the floor when this point is one radius above it.
    """
    g = model.geom(f"{side}_foot4_collision")
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, g.quat)
    return g.pos - R.reshape(3, 3)[:, 2] * g.size[1], g.size[0]


def support_jac(model, sides):
    """The function cost_matrices needs: Jacobian of the toe tip (or the midpoint of two toe tips)."""
    def jac(model_, data_):
        return sum(F.point_jac(model_, data_, f"{s}_ankle_roll_link", toe(model_, s)[0]) for s in sides) / len(sides)
    return jac


def tiptoe_pose(model, data, joints, q_home, sides, tilt_deg: float):
    """IK: toe tips on the floor (moved 12 cm forward), feet tilted heel-up, CoM over the toe(s), torso upright,
    arms out; with one leg, the right foot lifted as in the flamingo. Starts from q_home (standing, feet flat)."""
    home = q_home.copy()
    data.qpos[:] = home
    mujoco.mj_forward(model, data)
    toes = {s: F.point_world(data, f"{s}_ankle_roll_link", toe(model, s)[0]) for s in sides}
    targets = {s: np.array([p[0] + 0.12, p[1], toe(model, s)[1]]) for s, p in toes.items()}
    support = sum(targets.values()) / len(targets)
    q_nom = home.copy()
    roll = 1.3 if len(sides) == 1 else 0.25
    for jn, value in (("left_shoulder_roll_joint", roll), ("right_shoulder_roll_joint", -roll),
                      ("left_elbow_joint", 0.0), ("right_elbow_joint", 0.0)):
        q_nom[7 + joints.index(jn)] = value
    tasks = [("com_xy", None, support[:2], 300),
             ("body_rot", "torso_link", F.flat(data.body("torso_link").xmat.reshape(3, 3)), 10)]
    for s in sides:
        tilted = F.flat(data.site(f"{s}_foot").xmat.reshape(3, 3)) @ F.roty(np.deg2rad(tilt_deg))
        tasks += [("point_pos", (f"{s}_ankle_roll_link", toe(model, s)[0]), targets[s], 100),
                  ("site_rot", f"{s}_foot", tilted, 100)]
    if len(sides) == 1:
        tasks.append(("site_pos", "right_foot", data.site("right_foot").xpos + [0.27, -0.20, 0.30], 10))
    q = F.solve_ik(model, data, q_nom, tasks, iters=500)
    data.qpos[:] = home
    mujoco.mj_forward(model, data)
    return q, support


def floor_contacts(model, data) -> set:
    """Names of the geoms touching the floor."""
    out = set()
    for c in data.contact[:data.ncon]:
        n1, n2 = model.geom(c.geom1).name, model.geom(c.geom2).name
        if "floor" in (n1, n2):
            out.add(n2 if n1 == "floor" else n1)
    return out


def run(model, data, q0, ctrl0, K, allowed: set, duration: float = 5.0, push=None):
    """(still on tip-toe, base yaw change in degrees). Failing = any other geom touches the floor, e.g. a heel."""
    mujoco.mj_resetData(model, data)
    data.qpos[:] = q0
    mujoco.mj_forward(model, data)
    torso = model.body("torso_link").id
    for _ in range(round(duration / model.opt.timestep)):
        F.lqr_control(model, data, q0, ctrl0, K)
        data.xfrc_applied[:] = 0.0
        if push is not None and push[0] <= data.time < push[0] + 0.1:
            data.xfrc_applied[torso, :3] = push[1]
        mujoco.mj_step(model, data)
        if not floor_contacts(model, data) <= allowed or data.qpos[2] < q0[2] - 0.15:
            return False, data.time
    dq = np.zeros(model.nv)
    mujoco.mj_differentiatePos(model, dq, 1.0, q0, data.qpos)
    return True, np.degrees(dq[5])                    # base rotation about vertical: spinning on the toe


def max_push(model, data, q0, ctrl0, K, allowed, direction, hi: float = 150.0) -> float:
    lo = 0.0
    for _ in range(8):
        mid = (lo + hi) / 2
        if run(model, data, q0, ctrl0, K, allowed, 3.0, (0.5, mid * np.asarray(direction, float)))[0]:
            lo = mid
        else:
            hi = mid
    return lo


def part_g(model, data, joints):
    q_home = data.qpos.copy()                                     # standing, feet flat (before any simulation)
    print("\n=== G: tip-toe ===")
    print(f"  {'case':<18} | contact | growing modes | holding only | {'LQR (balance term)':<30} | "
          f"{'LQR (no balance term)':<30}")
    results = {}
    for name, (sides, tilt) in CASES.items():
        q, support = tiptoe_pose(model, data, joints, q_home, sides, tilt)
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        com_err = 1000 * np.linalg.norm(data.subtree_com[1][:2] - support[:2])
        assert com_err < 1 and not F.self_contacts(model, data), f"{name}: pose not balanced or limbs touching"
        q0, ctrl0, residual = F.equilibrium(model, data, q)
        allowed = floor_contacts(model, data)                     # the toe tips touching at the equilibrium
        assert all("foot" in g for g in allowed) and np.abs(residual).max() < 0.5
        A, B = F.linearize(model, data, q0, ctrl0)
        growing = int(np.sum(np.abs(np.linalg.eigvals(A)) > 1.001))
        cells = []
        for balance in (1000.0, 0.0):
            K = F.lqr_gain(A, B, *F.cost_matrices(model, data, joints, q0, balance=balance,
                                                  support_jac=support_jac(model, sides), both_legs=len(sides) == 2),
                           own=False)
            ok, yaw = run(model, data, q0, ctrl0, K, allowed)
            pushes = [max_push(model, data, q0, ctrl0, K, allowed, v) for v in DIRECTIONS.values()] if ok else []
            cells.append(f"{'/'.join(f'{p:.0f}' for p in pushes)} N, yaw {yaw:+.2f} deg" if ok else "falls")
            if balance:
                results[name] = (q0, ctrl0, K, allowed, ok, pushes)
        held, _ = run(model, data, q0, ctrl0, np.zeros_like(K), allowed)
        print(f"  {name:<18} | {len(allowed):>7} | {growing:>13} | {('holds' if held else 'falls'):<12} | "
              f"{cells[0]:<30} | {cells[1]:<30}")
        assert results[name][4] and not held
    print("  (pushes: largest 0.1 s push on the torso, forward/back/left/right; yaw: rotation about the vertical in "
          "5 s, with no push)")
    return results


def video(model, data, results):
    """All three cases side by side, each pushed sideways (6 N for 0.1 s) at t = 1.5 s."""
    F.brighten(model)
    cam = F.make_camera(lookat=[0.12, 0.05, 0.62], distance=2.6, azimuth=120, elevation=-8)
    steps_per_frame = round((1 / 30) / model.opt.timestep)
    panels = []
    for name, (q0, ctrl0, K, allowed, _, _) in results.items():
        mujoco.mj_resetData(model, data)
        data.qpos[:] = q0
        mujoco.mj_forward(model, data)
        torso, frames = model.body("torso_link").id, []
        with mujoco.Renderer(model, height=420, width=360) as renderer:
            for i in range(round(4.0 / model.opt.timestep)):
                F.lqr_control(model, data, q0, ctrl0, K)
                data.xfrc_applied[:] = 0.0
                pushing = 1.5 <= data.time < 1.6
                if pushing:
                    data.xfrc_applied[torso, :3] = [0, 6.0, 0]
                mujoco.mj_step(model, data)
                if i % steps_per_frame == 0:
                    renderer.update_scene(data, camera=cam)
                    frames.append(F.label(Image.fromarray(renderer.render()), f"{name}{'  <- push' if pushing else ''}"))
        panels.append(frames)
    F.write_mp4(F.side_by_side(panels), results_dir / "g_tiptoe.mp4", fps=1 / (steps_per_frame * model.opt.timestep))
    print(f"saved {results_dir / 'g_tiptoe.mp4'}")


def main():
    model, data, joints = F.build_torque_g1()
    results = part_g(model, data, joints)
    video(model, data, results)
    print("\n=== Part G complete ===")


if __name__ == "__main__":
    main()
