"""Module 2, Part A: contact forces on the G1.

A robot standing on the floor is held up only by contact forces, and each of them obeys three rules:
    1. it can only push       f_n >= 0                     (unilateral: the floor never pulls the foot down)
    2. it can't slip          |f_t| <= mu f_n               (the friction cone)
    3. so the push of all of them together acts at one point under the feet, the centre of pressure (CoP),
       which always lies inside the support polygon (the outline of the points in contact)
Walking control is about choosing motions whose contact forces obey these rules. This part reads the forces out
of MuJoCo and looks at all three, on the G1 standing, shifting its weight, lifting a leg, and being pulled sideways.

    A1  read the contacts: per-contact forces in the world frame, checked against Newton and MuJoCo's own solver
    A2  friction cones: how close each contact is to slipping
    A3  CoP and support polygon; standing still, the CoP is right under the CoM
    A4  weight shift and leg lift (gain-scheduled LQR, Module 1 Part F): the CoP walks into the left foot (video)
    A5  a sideways pull: which runs out first, friction or the support polygon? (video)

    .venv/bin/python classical/02_contact_zmp_lipm/contact_forces.py
    .venv/bin/mjpython classical/02_contact_zmp_lipm/contact_forces.py --viewer     # interactive (macOS: mjpython)
"""

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import mujoco
import numpy as np
from PIL import Image
import scipy.linalg
from scipy.spatial import ConvexHull

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PolygonPatch

script_dir = Path(__file__).resolve().parent
repo = script_dir.parents[1]
sys.path.insert(0, str(repo / "exercises"))
sys.path.insert(0, str(repo / "classical" / "01_dynamics_lqr_mpc"))
import g1_utils
import one_leg_lqr as F
from render_utils import make_camera

results_dir = script_dir / "results"
results_dir.mkdir(exist_ok=True)

SIDES = ("left", "right")
G = 9.81


# ---------------------------------------------------------------------------------------------------------
# A1: reading the contacts
# ---------------------------------------------------------------------------------------------------------

@dataclass
class Contact:
    side: str               # "left" / "right" foot
    geom: str               # which foot capsule
    pos: np.ndarray         # (3,) world position of the contact point
    frame: np.ndarray       # (3, 3) rows: normal (floor -> foot), tangent 1, tangent 2, in world coordinates
    f_local: np.ndarray     # (3,) force on the foot in the contact frame: (normal, tangent 1, tangent 2)
    f: np.ndarray           # (3,) the same force in world coordinates
    mu: float               # sliding friction coefficient of this contact
    body: int               # the foot body the force acts on

    @property
    def fn(self) -> float:
        return self.f_local[0]

    @property
    def ft(self) -> float:
        return float(np.hypot(self.f_local[1], self.f_local[2]))


def floor_contacts(model, data) -> list[Contact]:
    """Every foot-floor contact, with its force on the foot.

    MuJoCo stores each contact with a frame whose first row is the normal, pointing from geom1 to geom2. The
    solver's forces live in data.efc_force, in a pyramid basis (opt.cone = pyramidal); mj_contactForce converts
    them to (normal, tangent 1, tangent 2) components in that frame. frame.T @ f turns those into world
    coordinates. The force is the one geom1 exerts on geom2, so if the foot is geom1 the sign flips.
    """
    out, f6 = [], np.zeros(6)
    floor = model.geom("floor").id
    for i in range(data.ncon):
        c = data.contact[i]
        if floor not in (c.geom1, c.geom2):
            continue
        foot_geom = c.geom2 if c.geom1 == floor else c.geom1
        sign = 1.0 if c.geom1 == floor else -1.0
        mujoco.mj_contactForce(model, data, i, f6)
        frame = c.frame.reshape(3, 3).copy()
        name = model.geom(foot_geom).name
        out.append(Contact(side=name.split("_")[0], geom=name, pos=c.pos.copy(), frame=frame,
                           f_local=sign * f6[:3].copy(), f=sign * frame.T @ f6[:3], mu=c.friction[0],
                           body=model.geom_bodyid[foot_geom]))
    return out


def contact_generalized_force(model, data, contacts) -> np.ndarray:
    """sum_i J_i^T f_i: what the contact forces do to every degree of freedom (Module 1 Part E: tau = J^T F)."""
    tau = np.zeros(model.nv)
    jp = np.zeros((3, model.nv))
    for c in contacts:
        mujoco.mj_jac(model, data, jp, None, c.pos, c.body)
        tau += jp.T @ c.f
    return tau


def net_wrench(contacts, point) -> tuple[np.ndarray, np.ndarray]:
    """Total force and total moment about `point` of all contact forces: (sum f_i, sum (r_i - point) x f_i)."""
    f = sum((c.f for c in contacts), np.zeros(3))
    n = sum((np.cross(c.pos - point, c.f) for c in contacts), np.zeros(3))
    return f, n


# ---------------------------------------------------------------------------------------------------------
# A2: friction cones
# ---------------------------------------------------------------------------------------------------------

def cone_usage(c: Contact) -> float:
    """How much of the friction cone a contact uses: |f_t| / (mu f_n). 1 = on the edge of slipping.

    MuJoCo's pyramidal cone is a 4-sided pyramid inside the round cone: |f_t1| + |f_t2| <= mu f_n. This returns
    the pyramid's usage, which is >= the round cone's (they're equal when the friction is along one tangent).
    """
    return (abs(c.f_local[1]) + abs(c.f_local[2])) / (c.mu * c.fn) if c.fn > 1e-9 else 0.0


# ---------------------------------------------------------------------------------------------------------
# A3: centre of pressure, support polygon
# ---------------------------------------------------------------------------------------------------------

def cop(contacts) -> np.ndarray | None:
    """Centre of pressure on the floor: the average contact point, weighted by the vertical force.
        p = sum_i f_z,i r_i / sum_i f_z,i            (x and y; the floor is flat, at z = 0)
    """
    fz = np.array([c.f[2] for c in contacts])
    if fz.sum() < 1e-6:
        return None
    return (fz[:, None] * np.array([c.pos[:2] for c in contacts])).sum(0) / fz.sum()


def cop_from_wrench(contacts) -> np.ndarray | None:
    """The same point from the net wrench alone: the point on the floor about which the contact forces have no
    horizontal moment. With the moment n about the world origin and the total force f:
        p_x = -n_y / f_z,  p_y = n_x / f_z
    (derivation: the moment about p is n - p x f; set its x and y components to zero with p_z = 0).
    """
    f, n = net_wrench(contacts, np.zeros(3))
    if f[2] < 1e-6:
        return None
    return np.array([-n[1] / f[2], n[0] / f[2]])


def capsule_ends(model, data, side: str) -> np.ndarray:
    """(14, 3): both end centres of the foot's 7 capsules, in world coordinates."""
    pts = []
    for g in range(model.ngeom):
        name = model.geom(g).name
        if name.startswith(f"{side}_foot") and name.endswith("_collision"):
            axis = data.geom_xmat[g].reshape(3, 3)[:, 2]
            pts += [data.geom_xpos[g] + s * model.geom_size[g][1] * axis for s in (1, -1)]
    return np.array(pts)


def foot_outline(model, data, side: str, n_arc: int = 8) -> np.ndarray:
    """The foot's footprint on the floor: the convex hull of its capsules (centre segments widened by the radius).
    Returned as (k, 2) polygon corners, counter-clockwise."""
    radius = model.geom_size[model.geom(f"{side}_foot4_collision").id][0]
    angles = np.linspace(0, 2 * np.pi, n_arc, endpoint=False)
    pts = np.concatenate([e[:2] + radius * np.column_stack([np.cos(angles), np.sin(angles)])
                          for e in capsule_ends(model, data, side)])
    return pts[ConvexHull(pts).vertices]


def support_polygon(contacts) -> np.ndarray | None:
    """Convex hull of the contact points that carry force: where the CoP can be. (k, 2), counter-clockwise."""
    pts = np.array([c.pos[:2] for c in contacts if c.fn > 1e-3])
    if len(pts) < 3:
        return pts if len(pts) else None
    try:
        return pts[ConvexHull(pts).vertices]
    except Exception:                       # all in a line (a foot rolled onto its edge): the polygon is a segment
        far = np.argmax(np.linalg.norm(pts - pts[0], axis=1))
        return pts[[np.argmax(np.linalg.norm(pts - pts[far], axis=1)), far]]


def distance_inside(poly: np.ndarray, p: np.ndarray) -> float:
    """Signed distance from p to the edge of a convex polygon (counter-clockwise): > 0 inside, < 0 outside."""
    d = []
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        edge = b - a
        normal = np.array([edge[1], -edge[0]]) / np.linalg.norm(edge)      # points outwards for a CCW polygon
        d.append(-(p - a) @ normal)
    return min(d)


@dataclass
class ContactState:
    """Everything this part looks at, at one instant."""
    t: float
    contacts: list
    total: np.ndarray           # total contact force (3,)
    load: dict                  # vertical force per foot (N)
    cop: np.ndarray | None      # (2,)
    com: np.ndarray             # (3,) centre of mass
    polygon: np.ndarray | None  # support polygon (k, 2)
    usage: float                # largest cone usage of any loaded contact
    margin: float | None        # CoP distance to the polygon edge (m)


def contact_state(model, data) -> ContactState:
    contacts = floor_contacts(model, data)
    total = sum((c.f for c in contacts), np.zeros(3))
    load = {s: sum(c.f[2] for c in contacts if c.side == s) for s in SIDES}
    p, poly = cop(contacts), support_polygon(contacts)
    usage = max((cone_usage(c) for c in contacts if c.fn > 0.5), default=0.0)
    margin = distance_inside(poly, p) if p is not None and poly is not None and len(poly) >= 3 else None
    return ContactState(data.time, contacts, total, load, p, data.subtree_com[1].copy(), poly, usage, margin)


# ---------------------------------------------------------------------------------------------------------
# Equilibria with two feet down, and the weight-shift chain (Module 1 Part F's gain scheduling)
# ---------------------------------------------------------------------------------------------------------

def rotx(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def balanced_pose(model, data, nominal, tasks_for, x0, posture_weight: float, iters: int = 12):
    """A pose the floor carries exactly, and the torques that hold it: (q0, ctrl0, pelvis residual (6,)).

    MuJoCo's floor is a stiff spring (about 6 kN per mm a foot sinks). Where the weight goes is decided by how
    deep each part of each foot sinks: a foot 0.2 mm lower than the other takes most of the weight, a foot tilted
    by a hundredth of a degree moves its load to one edge. So "both feet flat on the floor" is not precise enough:
    the depths and tilts of the feet must be exactly those that put the floor's push under the CoM.

    tasks_for(x) gives IK tasks with the feet placed by three numbers x (heights and tilts, chosen by the
    caller). Newton's method on x drives the three pelvis residuals that need the floor (vertical force, roll
    and pitch torque) to zero: then no outside help is needed and the joint torques alone hold the pose.
    """
    def solve(x):
        q = F.solve_ik(model, data, nominal, tasks_for(x), iters=150, posture_weight=posture_weight)
        return q, F.inverse_dynamics(model, data, q)

    x = np.asarray(x0, float)
    for _ in range(iters):
        q, f = solve(x)
        r = f[[2, 3, 4]]
        if np.abs(r).max() < 1e-3:
            break
        J = np.column_stack([(solve(x + 1e-6 * e)[1][[2, 3, 4]] - r) / 1e-6 for e in np.eye(3)])
        x = x - np.linalg.solve(J, r)
    q, f = solve(x)
    dof = model.jnt_dofadr[model.actuator_trnid[:, 0]]
    return q, f[dof].copy(), f[:6].copy(), x


def floor_directions(model, data, q) -> np.ndarray:
    """(2nv, 3): moving the whole robot along the floor (x, y) and turning it about the vertical.

    On a flat floor nothing depends on where the robot stands or which way it faces, and friction has no spring
    sideways (it resists sliding velocity, not position). So these three directions are exact eigenvectors of A
    with eigenvalue 1 (A N = N), and no motor can change them (the robot can't push itself across the floor
    while balancing). They are uncontrollable modes exactly on the unit circle, and then the Riccati equation has
    no stabilizing solution at all: scipy rightly refuses. Part F escaped only because rounding moved them a hair
    off 1. The base's position differences are in the world frame, its rotation differences in the pelvis frame,
    so "turn about vertical" is R^T z in the rotation part.
    """
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    N = np.zeros((2 * model.nv, 3))
    N[0, 0] = N[1, 1] = 1.0
    N[3:6, 2] = data.xmat[model.body("pelvis").id].reshape(3, 3).T @ [0, 0, 1]
    return N


def lqr_gain(model, data, q0, A, B, Q, R):
    """LQR without the three floor directions. With T an orthonormal basis of everything else, write the state
    as s = T a + N b. Since A N = N:  a_next = T^T A T a + T^T B u  exactly, a model without the floor modes.
    Solve the Riccati equation for it, then K = K_a T^T (the gain ignores where on the floor the robot is)."""
    T = scipy.linalg.null_space(floor_directions(model, data, q0).T)
    K_a = F.lqr_gain(T.T @ A @ T, T.T @ B, T.T @ Q @ T, R, own=False)
    return K_a @ T.T


def build_chain(model, data, joints, n_shift: int = 12, n_lift: int = 16, shift: float = 0.9) -> dict:
    """Balanced poses from standing (CoM between the feet) to the CoM over the left foot (both feet down), then
    to Part F's flamingo (right foot lifted). Each has its equilibrium, linear model and LQR gain.

    The weight shift stops at `shift` (90 %) of the way to the left foot, so the right foot still carries some
    weight: with the CoM exactly over the left foot it would touch with zero force, on the edge between contact
    and no contact, and a linear model there wrongly counts on a foot that lifts off at the slightest sway.

    Keys "q", "u", "K", "residual", "x" (lists), "marks": {name: index} of the key poses, "foot_down": index of
    the last pose with the right foot touching the floor.
    """
    q_home = data.qpos.copy()
    flamingo = F.make_poses(model, data, joints)[0]["flamingo"]         # left foot where it is at home
    data.qpos[:] = q_home
    mujoco.mj_forward(model, data)
    feet = {s: (data.site(f"{s}_foot").xpos.copy(), F.flat(data.site(f"{s}_foot").xmat.reshape(3, 3))) for s in SIDES}
    mid = (feet["left"][0][:2] + feet["right"][0][:2]) / 2
    R_torso = F.flat(data.body("torso_link").xmat.reshape(3, 3))
    hard = 1e4                                                            # feet exactly where Newton puts them

    def both_feet(com_xy):
        """x = (left foot height, right foot height, pitch of both feet)."""
        return lambda x: [("site_pos", "left_foot", feet["left"][0] + [0, 0, x[0]], hard),
                          ("site_pos", "right_foot", feet["right"][0] + [0, 0, x[1]], hard),
                          *[("site_rot", f"{s}_foot", feet[s][1] @ F.roty(x[2]), hard) for s in SIDES],
                          ("com_xy", None, com_xy, 300), ("body_rot", "torso_link", R_torso, 10)]

    def left_foot(x):
        """x = (left foot height, roll, pitch); the CoM over the left foot."""
        return [("site_pos", "left_foot", feet["left"][0] + [0, 0, x[0]], hard),
                ("site_rot", "left_foot", feet["left"][1] @ rotx(x[1]) @ F.roty(x[2]), hard),
                ("com_xy", None, feet["left"][0][:2], 300)]

    q_flat = F.solve_ik(model, data, q_home, both_feet(mid)(np.zeros(3)), iters=150, posture_weight=0.2)
    sink = F.equilibrium(model, data, q_flat)[0][2] - q_flat[2]         # Part F's height: a starting guess

    chain = {k: [] for k in ("q", "u", "K", "residual", "x")}
    chain["marks"] = {"stand": 0, "weight on left": n_shift, "flamingo": n_shift + n_lift}
    x = np.array([sink, sink, 0.0])
    step = None
    for i in range(n_shift + n_lift + 1):
        if i <= n_shift:                            # weight shift: the CoM target slides from mid to the left foot
            a = shift * i / n_shift
            q0, u0, res, x = balanced_pose(model, data, q_home, both_feet((1 - a) * mid + a * feet["left"][0][:2]),
                                           x, posture_weight=0.2)
        else:                                       # leg lift: a straight line in joint space, IK keeps the balance
            if step is None:
                step = np.zeros(model.nv)
                mujoco.mj_differentiatePos(model, step, 1.0, chain["q"][n_shift], flamingo)
                x = np.array([x[0], 0.0, x[2]])
            between = chain["q"][n_shift].copy()
            mujoco.mj_integratePos(model, between, step, (i - n_shift) / n_lift)
            q0, u0, res, x = balanced_pose(model, data, between, left_foot, x, posture_weight=1.0)
        data.qpos[:] = q0
        mujoco.mj_forward(model, data)
        assert not F.self_contacts(model, data), f"pose {i}: {F.self_contacts(model, data)}"
        if any(c.side == "right" for c in floor_contacts(model, data)):
            chain["foot_down"] = i
        A, B = F.linearize(model, data, q0, u0)
        Q, R = F.cost_matrices(model, data, joints, q0, both_legs=i <= n_shift)
        K = lqr_gain(model, data, q0, A, B, Q, R)
        for key, val in (("q", q0), ("u", u0), ("residual", res), ("x", x.copy()), ("K", K)):
            chain.setdefault(key, []).append(val)
    return chain


class ChainController:
    """u = u0(s) - K(s) (x - x_ref(s)) at a position s along the chain (Part F's run_flow, as a reusable object).

    s is a real number: 3.4 means 40 % of the way from pose 3 to pose 4; the reference pose, the holding torques
    and the gain are blended between the two neighbours. s_dot (poses per second) gives the reference velocity.
    """

    def __init__(self, model, chain):
        self.model, self.chain, self.n = model, chain, len(chain["q"])

    def __call__(self, data, s: float, s_dot: float = 0.0):
        m, c = self.model, self.chain
        s = float(np.clip(s, 0, self.n - 1))
        a = min(int(s), self.n - 2)
        f = s - a
        dq = np.zeros(m.nv)
        mujoco.mj_differentiatePos(m, dq, 1.0, c["q"][a], c["q"][a + 1])
        q_ref = c["q"][a].copy()
        mujoco.mj_integratePos(m, q_ref, dq, f)
        dx = np.zeros(2 * m.nv)
        mujoco.mj_differentiatePos(m, dx[:m.nv], 1.0, q_ref, data.qpos)
        dx[m.nv:] = data.qvel - dq * s_dot
        K = (1 - f) * c["K"][a] + f * c["K"][a + 1]
        data.ctrl[:] = (1 - f) * c["u"][a] + f * c["u"][a + 1] - K @ dx


def schedule(plan):
    """plan = [(s_start, hold), (s_next, move time, hold), ...] -> (s(t), s_dot(t), duration). Smoothstep moves."""
    segments, t0, s_prev = [], 0.0, plan[0][0]
    segments.append((t0, plan[0][1], s_prev, s_prev))
    t0 += plan[0][1]
    for s_next, t_move, hold in plan[1:]:
        segments.append((t0, t_move, s_prev, s_next))
        segments.append((t0 + t_move, hold, s_next, s_next))
        t0, s_prev = t0 + t_move + hold, s_next

    def at(t):
        for start, length, a, b in segments:
            if t < start + length:
                x = (t - start) / length
                return a + (b - a) * x * x * (3 - 2 * x), (b - a) * 6 * x * (1 - x) / length
        return s_prev, 0.0
    return at, t0


def fallen(model, data, z0: float) -> bool:
    """Anything but a foot touches the floor, or the pelvis dropped 15 cm."""
    for c in data.contact[:data.ncon]:
        names = model.geom(c.geom1).name + model.geom(c.geom2).name
        if "floor" in names and "_foot" not in names:
            return True
    return data.qpos[2] < z0 - 0.15


# ---------------------------------------------------------------------------------------------------------
# Drawing: overlays in the 3D scene, and a top-down view of the feet
# ---------------------------------------------------------------------------------------------------------

RED, BLUE, YELLOW = [0.95, 0.15, 0.1, 1], [0.15, 0.45, 1.0, 1], [1.0, 0.85, 0.1, 1]


def _add(scn, gtype, size, pos, rgba, mat=None):
    if scn.ngeom >= scn.maxgeom:
        return None
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.asarray(size, float), np.asarray(pos, float),
                        np.eye(3).ravel() if mat is None else mat, np.asarray(rgba, np.float32))
    scn.ngeom += 1
    return g


def _line(scn, a, b, width, rgba, gtype=mujoco.mjtGeom.mjGEOM_CAPSULE):
    g = _add(scn, gtype, [0, 0, 0], [0, 0, 0], rgba)
    if g is not None:
        mujoco.mjv_connector(g, gtype, width, np.asarray(a, float), np.asarray(b, float))


def draw_overlay(scn, st: ContactState, force_scale: float = 0.001):
    """Into an MjvScene (a Renderer's scene or the viewer's user_scn): the support polygon (yellow), the CoM and
    its shadow on the floor (blue), and the total contact force as one red arrow standing on the CoP."""
    z = 0.002
    if st.polygon is not None and len(st.polygon) >= 2:
        for a, b in zip(st.polygon, np.roll(st.polygon, -1, axis=0)):
            _line(scn, [*a, z], [*b, z], 0.004, YELLOW)
    _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.025, 0, 0], st.com, BLUE)
    _line(scn, st.com, [st.com[0], st.com[1], z], 0.003, [*BLUE[:3], 0.6])
    _add(scn, mujoco.mjtGeom.mjGEOM_CYLINDER, [0.018, 0.0015, 0], [st.com[0], st.com[1], z], BLUE)
    if st.cop is not None:
        _add(scn, mujoco.mjtGeom.mjGEOM_CYLINDER, [0.014, 0.002, 0], [*st.cop, z + 0.001], RED)
        tip = np.array([*st.cop, z]) + force_scale * st.total
        _line(scn, [*st.cop, z], tip, 0.012, RED, mujoco.mjtGeom.mjGEOM_ARROW)


def scene_option(contact_forces: bool = True) -> mujoco.MjvOption:
    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = contact_forces
    return opt


def style_forces(model):
    """MuJoCo's own per-contact force arrows: thin, 3 mm per N (the weight per contact is ~10 N)."""
    model.vis.map.force = 0.003
    model.vis.scale.forcewidth = 0.015
    model.vis.rgba.contactforce[:] = [0.1, 0.9, 0.3, 1]


def top_down(model, data, st: ContactState, trail, size=(480, 480), title: str = "", view=None) -> Image.Image:
    """The feet from above: footprints, each contact (area ~ vertical force, colour = friction cone usage), the
    support polygon, the CoP (red, with its recent path) and the CoM's shadow (blue). Robot faces up the page."""
    fig = plt.figure(figsize=(size[0] / 100, size[1] / 100), dpi=100)
    ax = fig.add_axes([0.17, 0.24, 0.78, 0.66])
    for s in SIDES:
        down = st.load[s] > 1e-6
        ax.add_patch(PolygonPatch(foot_outline(model, data, s)[:, ::-1], closed=True, fc="0.85" if down else "0.96",
                                  ec="0.5", lw=1, ls="-" if down else "--"))
    if st.polygon is not None and len(st.polygon) >= 3:
        ax.add_patch(PolygonPatch(st.polygon[:, ::-1], closed=True, fill=False, ec="#e0b000", lw=2, ls="--"))
    if st.contacts:
        pts = np.array([c.pos[:2] for c in st.contacts])
        fz = np.array([max(c.f[2], 0) for c in st.contacts])
        use = np.array([cone_usage(c) for c in st.contacts])
        ax.scatter(pts[:, 1], pts[:, 0], s=4 + 400 * fz / max(st.total[2], 1.0), c=use, cmap="RdYlGn_r", vmin=0, vmax=1, ec="k", lw=0.4, zorder=3)
    if len(trail) > 1:
        tr = np.array(trail)
        ax.plot(tr[:, 1], tr[:, 0], color="#d62728", lw=1, alpha=0.6)
    ax.plot(st.com[1], st.com[0], "o", ms=11, mfc="none", mec="#1f5fd6", mew=2.5, zorder=4)
    if st.cop is not None:
        ax.plot(st.cop[1], st.cop[0], "*", ms=16, color="#d62728", mec="k", mew=0.6, zorder=5)
    (xlo, xhi), (ylo, yhi) = view if view is not None else ((-0.12, 0.22), (-0.25, 0.25))
    ax.set_xlim(yhi, ylo)                                         # robot's left on the left of the page
    ax.set_ylim(xlo, xhi)
    ax.set_aspect("equal")
    ax.set_xlabel("y (m)   <- robot's left")
    ax.set_ylabel("x (m)   forward ->")
    ax.set_title(title or f"t = {st.t:4.1f} s", fontsize=10)
    ax.tick_params(labelsize=7)
    # load bar: how the weight is shared between the feet
    bar = fig.add_axes([0.13, 0.05, 0.82, 0.05])
    w = sum(st.load.values()) or 1.0
    bar.barh(0, st.load["left"] / w, color="#4c72b0")
    bar.barh(0, st.load["right"] / w, left=st.load["left"] / w, color="#dd8452")
    bar.set_xlim(0, 1)
    bar.set_yticks([])
    bar.set_xticks([])
    bar.text(0.01, 0, f"left {st.load['left']:.0f} N", va="center", fontsize=8, color="w")
    bar.text(0.99, 0, f"right {st.load['right']:.0f} N", va="center", ha="right", fontsize=8, color="w")
    fig.text(0.13, 0.135, "red *  CoP          blue o  CoM shadow          yellow - -  support polygon", fontsize=7)
    fig.text(0.13, 0.11, "dots: contacts (size: force; colour: cone use, red = 1)", fontsize=7)
    fig.canvas.draw()
    img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    return img


def render(renderer, data, cam, st, opt):
    renderer.update_scene(data, camera=cam, scene_option=opt)
    draw_overlay(renderer.scene, st)
    return Image.fromarray(renderer.render())


def hstack(a: Image.Image, b: Image.Image) -> Image.Image:
    out = Image.new("RGB", (a.width + b.width, max(a.height, b.height)), (255, 255, 255))
    out.paste(a, (0, 0))
    out.paste(b, (a.width, 0))
    return out


# ---------------------------------------------------------------------------------------------------------
# Simulation with logging and recording
# ---------------------------------------------------------------------------------------------------------

FPS = 30


def simulate(model, data, chain, duration, s_of_t=lambda t: (0.0, 0.0), pull=None, q_start=None, record=None,
             after_fall: float = 1.0):
    """Run the chain controller; log the contact state every step; optionally record frames.

    s_of_t(t) -> (s, s_dot) along the chain. pull(t) -> force (3,) on the pelvis (at its centre of mass).
    record = (renderer, camera function(data) -> MjvCamera, panel function(st, trail) -> image or None).
    Stops `after_fall` seconds after a fall. Returns (log: dict of arrays, frames).
    """
    mujoco.mj_resetData(model, data)
    data.qpos[:] = chain["q"][0] if q_start is None else q_start
    mujoco.mj_forward(model, data)
    ctrl, pelvis = ChainController(model, chain), model.body("pelvis").id
    feet0 = {s: data.site(f"{s}_foot").xpos[:2].copy() for s in SIDES}
    z0, fell_at, frames, trail = data.qpos[2], None, [], []
    log = {k: [] for k in ("t", "left", "right", "total", "cop", "com", "usage", "margin", "min_fn", "max_pyr",
                           "slip_left", "slip_right", "pull", "s")}
    every = round(1 / (FPS * model.opt.timestep))
    opt = scene_option()
    for i in range(round(duration / model.opt.timestep)):
        s, s_dot = s_of_t(data.time)
        ctrl(data, s, s_dot)
        f_pull = np.zeros(3) if pull is None else np.asarray(pull(data.time), float)
        data.xfrc_applied[:] = 0.0
        data.xfrc_applied[pelvis, :3] = f_pull
        mujoco.mj_step(model, data)
        st = contact_state(model, data)
        for key, val in (("t", data.time), ("left", st.load["left"]), ("right", st.load["right"]), ("total", st.total),
                         ("cop", st.cop if st.cop is not None else [np.nan] * 2), ("com", st.com),
                         ("usage", st.usage), ("margin", np.nan if st.margin is None else st.margin),
                         ("min_fn", min((c.fn for c in st.contacts), default=0.0)),
                         ("max_pyr", max((cone_usage(c) for c in st.contacts), default=0.0)),
                         ("slip_left", np.linalg.norm(data.site("left_foot").xpos[:2] - feet0["left"])),
                         ("slip_right", np.linalg.norm(data.site("right_foot").xpos[:2] - feet0["right"])),
                         ("pull", f_pull), ("s", s)):
            log[key].append(val)
        if st.cop is not None:
            trail = (trail + [st.cop])[-90:]
        if record is not None and i % every == 0:
            renderer, camera, panel = record
            img = render(renderer, data, camera(data), st, opt)
            side = panel(st, trail)
            frames.append(img if side is None else hstack(img, side))
        if fell_at is None and fallen(model, data, z0):
            fell_at = data.time
        if fell_at is not None and data.time > fell_at + after_fall:
            break
    log = {k: np.array(v) for k, v in log.items()}
    log["fell_at"] = fell_at
    return log, frames


def follow_camera(distance=2.2, azimuth=155, elevation=-15):
    """A camera that keeps the robot in view (it slides across the floor in A5)."""
    def cam(data):
        return make_camera(lookat=[data.qpos[0], data.qpos[1], 0.45], distance=distance, azimuth=azimuth,
                           elevation=elevation)
    return cam


def save_video(frames, name: str, gif_scale: float = 0.5, gif_every: int = 2):
    """MP4 at full size and frame rate, plus a smaller GIF (half size, 15 fps) for READMEs and posts."""
    F.write_mp4(frames, results_dir / f"{name}.mp4", fps=FPS)
    small = [f.resize((int(f.width * gif_scale), int(f.height * gif_scale)), Image.LANCZOS) for f in frames[::gif_every]]
    small[0].save(results_dir / f"{name}.gif", save_all=True, append_images=small[1:], duration=1000 * gif_every // FPS,
                  loop=0)
    print(f"saved {results_dir / name}.mp4 and .gif")


def hold(model, data, chain, s: float, seconds: float = 1.5):
    """Stand still at chain position s (starting there) and return the contact state at the end."""
    q = chain["q"][int(round(s))]
    simulate(model, data, chain, seconds, s_of_t=lambda t: (s, 0.0), q_start=q)
    return contact_state(model, data)


# ---------------------------------------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------------------------------------

def part_a1(model, data, ctx):
    print("\n=== A1: reading the contacts ===")
    chain = ctx["chain"]
    st = hold(model, data, chain, 0.0)
    W = mujoco.mj_getTotalmass(model) * G
    print(f"standing on both feet: {len(st.contacts)} contacts "
          f"({sum(c.side == 'left' for c in st.contacts)} left, {sum(c.side == 'right' for c in st.contacts)} right):"
          f" each of the 7 capsules per foot touches at both ends")
    print(f"  {'capsule':<22} {'position (mm)':>24} | {'normal':>7} {'tang 1':>7} {'tang 2':>7} (N, contact frame)")
    for c in st.contacts[:4]:
        print(f"  {c.geom:<22} {np.array2string(1000 * c.pos, precision=1, suppress_small=True):>24} | "
              f"{c.f_local[0]:7.2f} {c.f_local[1]:7.3f} {c.f_local[2]:7.3f}")
    print(f"  ... ({len(st.contacts) - 4} more)")
    print(f"sum of all contact forces: {np.array2string(st.total, precision=2)} N; the robot weighs "
          f"{mujoco.mj_getTotalmass(model):.2f} kg x {G} = {W:.2f} N")
    err_weight = abs(st.total[2] - W) / W

    # Independent check: MuJoCo's own constraint force in joint space, sum over contacts of J^T f
    tau = contact_generalized_force(model, data, st.contacts)
    contact_rows = np.isin(data.efc_type, [mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL,
                                           mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC])
    err_tau = np.abs(tau - data.qfrc_constraint).max() / np.abs(data.qfrc_constraint).max()
    print(f"sum_i J_i^T f_i vs MuJoCo's qfrc_constraint (all {data.nefc} constraint rows are contacts: "
          f"{contact_rows.all()}): largest difference {err_tau:.1e} of the largest entry")
    print(f"  its first 3 entries (the floor's push on the floating base) {np.array2string(tau[:3], precision=2)} N "
          f"= the total contact force: no motor can supply these, only the floor")
    assert err_weight < 0.005 and err_tau < 1e-6 and contact_rows.all()


def part_a2(model, data, ctx):
    print("\n=== A2: friction cones ===")
    chain = ctx["chain"]
    st = hold(model, data, chain, 0.0)
    print(f"standing still: largest |f_t| / (mu f_n) of any contact = {max(cone_usage(c) for c in st.contacts):.3f} "
          f"(no sideways force needed, no friction used)")
    # A steady 20 N forward push on the pelvis: now friction must hold it
    push = np.array([20.0, 0, 0])
    log, _ = simulate(model, data, chain, 1.5, pull=lambda t: push)
    st = contact_state(model, data)
    use = np.array([cone_usage(c) for c in st.contacts])
    round_use = np.array([c.ft / (c.mu * c.fn) for c in st.contacts])
    fxy = st.total[:2]
    print(f"steady 20 N push forward: floor's total sideways force {np.array2string(fxy, precision=2)} N "
          f"(cancels the push); total needs |F_t| / F_n = {np.linalg.norm(fxy) / st.total[2]:.3f}, mu = "
          f"{st.contacts[0].mu}")
    print(f"  per contact: cone use median {np.median(use):.2f}, largest {use.max():.2f} (pyramid), "
          f"{round_use.max():.2f} (round cone); contacts on the cone's edge: {np.sum(use > 0.999)} of {len(use)}")
    worst = max(st.contacts, key=cone_usage)
    print(f"  most-used contact: {worst.geom} carrying {worst.fn:.1f} N (average {st.total[2] / len(st.contacts):.1f} N):"
          f" the friction isn't shared in proportion to the load. Every contact must obey its own cone, so a")
    print(f"  contact or a foot with little weight on it runs out of friction first (A4 and A5 show it)")
    assert np.all(use <= 1 + 1e-6) and np.all(round_use <= use + 1e-9) and min(c.fn for c in st.contacts) >= 0
    ctx["a2_push"] = (st, use)


def part_a3(model, data, ctx):
    print("\n=== A3: centre of pressure and support polygon ===")
    chain = ctx["chain"]
    stances = {name: hold(model, data, chain, float(i)) for name, i in chain["marks"].items()}
    images = []
    renderer = mujoco.Renderer(model, height=360, width=360)
    opt = scene_option()
    for name, st in stances.items():
        p1, p2 = cop(st.contacts), cop_from_wrench(st.contacts)
        gap = 1000 * np.linalg.norm(p1 - st.com[:2])
        print(f"  {name:<15}: {len(st.contacts):>2} contacts, load L/R {st.load['left']:5.1f} / {st.load['right']:5.1f} N;"
              f" CoP {np.array2string(1000 * p1, precision=1)} mm; two formulas differ by "
              f"{1e3 * np.linalg.norm(p1 - p2):.1e} mm; CoM shadow {gap:.2f} mm away; "
              f"{1000 * st.margin:.0f} mm inside the support polygon")
        assert np.linalg.norm(p1 - p2) < 1e-6 and gap < 2.0 and st.margin > 0
        # re-create the state for a picture
        hold(model, data, chain, float(chain["marks"][name]))
        top = top_down(model, data, st, [], size=(360, 360), title=name)
        img = render(renderer, data, make_camera([0, 0.05, 0.45], 2.2, 155, -18), st, opt)
        col = Image.new("RGB", (360, 720), "white")
        col.paste(img, (0, 0))
        col.paste(top, (0, 360))
        images.append(col)
    renderer.close()
    out = Image.new("RGB", (3 * 360 + 8, 720), "white")
    for k, im in enumerate(images):
        out.paste(im, (k * 364, 0))
    out.save(results_dir / "a3_stances.png")
    print(f"standing still the CoP is right under the CoM: the floor's push must pass through the CoM, or the robot "
          f"would start to rotate. saved {results_dir / 'a3_stances.png'}")


def part_a4(model, data, ctx):
    print("\n=== A4: weight shift and leg lift ===")
    chain, marks = ctx["chain"], ctx["chain"]["marks"]
    s_of_t, duration = schedule([(0, 1.0), (marks["weight on left"], 2.5, 1.0), (marks["flamingo"], 3.0, 2.0)])
    W = mujoco.mj_getTotalmass(model) * G
    renderer = mujoco.Renderer(model, height=480, width=480)
    panel = lambda st, trail: top_down(model, data, st, trail)
    log, frames = simulate(model, data, chain, duration, s_of_t, record=(renderer, follow_camera(), panel))
    renderer.close()
    assert log["fell_at"] is None
    lift = np.argmax(log["right"] < 1e-6)
    gap = 1000 * np.linalg.norm(log["cop"] - log["com"][:, :2], axis=1)
    print(f"{duration:.1f} s, no fall; right foot leaves the floor at t = {log['t'][lift]:.2f} s")
    shift_end = np.searchsorted(log["t"], 4.5)
    print(f"  just before the lift the right foot carries {log['right'][shift_end]:.0f} N; its contacts hit their cones "
          f"and it creeps {1000 * log['slip_right'][:lift].max():.1f} mm (the little jolt at "
          f"t = {log['t'][np.argmax(np.abs(np.diff(log['right'][:lift])))]:.2f} s)")
    print(f"  unilateral: smallest normal force of any contact {log['min_fn'].min():.2e} N (never pulls)")
    print(f"  friction: largest pyramid use of any contact {log['max_pyr'].max():.4f} (never outside the cone)")
    print(f"  CoP: always inside the support polygon; closest to its edge {1000 * np.nanmin(log['margin']):.1f} mm, at "
          f"t = {log['t'][np.nanargmin(log['margin'])]:.2f} s, just after the right foot leaves: the last of the weight "
          f"swings the CoP to the left foot's outer edge. The lift is the closest this run comes to tipping")
    print(f"  total vertical force: {log['total'][:, 2].min():.0f} .. {log['total'][:, 2].max():.0f} N "
          f"(weight {W:.0f} N): it only equals the weight when the CoM isn't accelerating")
    ends = [np.searchsorted(log["t"], t) - 1 for t in (1.0, 4.5, duration)]
    print(f"  CoP to CoM shadow at the end of each hold: " + ", ".join(f"{gap[i]:.2f}" for i in ends) +
          f" mm. In between they part, up to {gap.max():.1f} mm (t = {log['t'][np.argmax(gap)]:.2f} s, the jolt): whenever"
          f" the CoM accelerates, the floor's push has to accelerate it, so it no longer points through the CoM (Part B)")
    assert log["min_fn"].min() >= -1e-9 and log["max_pyr"].max() <= 1 + 1e-6 and np.nanmin(log["margin"]) > 0
    save_video(frames, "a4_weight_shift")
    plot_a4(log, W)


def plot_a4(log, W):
    t = log["t"]
    fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    ax[0].plot(t, log["left"], label="left foot", color="#4c72b0")
    ax[0].plot(t, log["right"], label="right foot", color="#dd8452")
    ax[0].plot(t, log["total"][:, 2], label="total", color="k", lw=1)
    ax[0].axhline(W, color="0.5", ls=":", label="weight")
    ax[0].set_ylabel("vertical force (N)")
    ax[0].legend(loc="center left", fontsize=8)
    ax[1].plot(t, 1000 * log["cop"][:, 1], label="CoP y", color="#d62728")
    ax[1].plot(t, 1000 * log["com"][:, 1], label="CoM y (its shadow)", color="#1f5fd6", ls="--")
    ax[1].set_ylabel("sideways (mm)\n<- right   left ->")
    ax[1].legend(fontsize=8)
    ax[2].plot(t, log["max_pyr"], label="largest cone use of any contact", color="#8c564b")
    ax[2].plot(t, 1000 * log["margin"] / 100, label="CoP margin inside the polygon (x 100 mm)", color="#2ca02c")
    ax[2].set_ylim(0, 1.1)
    ax[2].set_xlabel("time (s)")
    ax[2].legend(fontsize=8)
    for a in ax:
        for x, txt in ((1.0, "shift weight"), (4.5, "lift leg")):
            a.axvspan(x, x + (2.5 if x == 1.0 else 3.0), color="0.9", zorder=0)
    ax[0].set_title("A4: weight shift and leg lift, measured from the contact forces")
    fig.tight_layout()
    fig.savefig(results_dir / "a4_forces.png", dpi=110)
    plt.close(fig)
    print(f"saved {results_dir / 'a4_forces.png'}")


def part_a5(model, data, ctx):
    print("\n=== A5: a sideways pull: friction or the support polygon? ===")
    chain = ctx["chain"]
    feet = [g for g in range(model.ngeom) if re.fullmatch(g1_utils.FOOT_PATTERN, model.geom(g).name)]
    mu0 = model.geom_friction[feet[0], 0]
    W = mujoco.mj_getTotalmass(model) * G
    rate = 20.0                                                     # N per second, from t = 0.5 s
    pull = lambda t: [0.0, -rate * max(0.0, t - 0.5), 0.0]          # to the robot's right, at the pelvis

    data.qpos[:] = chain["q"][0]
    mujoco.mj_forward(model, data)
    h = data.xipos[model.body("pelvis").id][2]
    d_edge = foot_outline(model, data, "right")[:, 1].min()             # y of the right foot's outer edge
    d = data.subtree_com[1][1] - d_edge
    print(f"rigid-body predictions: slides when the pull exceeds mu W; tips over the right foot's outer edge when "
          f"pull x {h:.2f} m (pull height) > W x {d:.3f} m (CoM to that edge): {W * d / h:.0f} N")
    print(f"  so it slides first if mu < d / h = {d / h:.2f}, tips first otherwise")
    results, videos = {}, []
    for mu in (0.6, 0.1):
        model.geom_friction[feet, 0] = mu
        renderer = mujoco.Renderer(model, height=400, width=400)

        def panel(st, trail):
            c = np.mean([data.site(f"{s}_foot").xpos[:2] for s in SIDES], axis=0)
            return top_down(model, data, st, trail, size=(400, 400), title=f"mu = {mu}   pull {rate * max(0, data.time - 0.5):.0f} N",
                            view=((c[0] - 0.15, c[0] + 0.19), (c[1] - 0.25, c[1] + 0.25)))
        log, frames = simulate(model, data, chain, 7.0, pull=pull, record=(renderer, follow_camera(2.3), panel))
        renderer.close()
        force = -log["pull"][:, 1]
        n50 = round(0.05 / model.opt.timestep)                     # lift-off: no load for 50 ms in a row
        off = np.convolve(log["left"] < 1.0, np.ones(n50), "valid") == n50
        events = {"left foot scuffs (5 mm)": np.argmax(log["slip_left"] > 0.005) or None,
                  "left foot lifts off": np.argmax(off) or None,
                  "robot slides (right foot 5 mm)": np.argmax(log["slip_right"] > 0.005) or None}
        first = {k: (log["t"][i], force[i]) for k, i in events.items() if i is not None}
        print(f"  mu = {mu} (slide predicted at {mu * W:.0f} N): " + "; ".join(
            f"{k} at {f:.0f} N" for k, (t, f) in sorted(first.items(), key=lambda kv: kv[1][0])) +
            (f"; falls at {force[np.searchsorted(log['t'], log['fell_at'])]:.0f} N" if log["fell_at"] else ""))
        lift, slide = (first.get(k, (np.inf,))[0] for k in ("left foot lifts off", "robot slides (right foot 5 mm)"))
        if lift <= slide:                                           # it tipped: check the moment balance
            i = events["left foot lifts off"]
            com, edge = log["com"][i], log["cop"][i]                # at lift-off the CoP is on the edge
            line = d_edge + model.geom_size[model.geom("right_foot4_collision").id][0]   # outer capsule centres
            print(f"    at lift-off the CoP is at the right foot's outer edge: {1000 * (line - edge[1]):.1f} mm beyond the "
                  f"outer capsules' centre line, on their rounded side (radius 10 mm); the foot is rolling onto that edge"
                  f" and the support polygon has shrunk to it. The CoM had moved {1000 * (log['com'][0][1] - com[1]):.0f} mm "
                  f"toward the pull (the robot isn't rigid). The rigid formula with that CoM: "
                  f"{W * (com[1] - edge[1]) / h:.0f} N; measured {force[i]:.0f} N")
        results[mu] = (log, first)
        videos.append(frames)
    model.geom_friction[feet, 0] = mu0
    n = min(len(v) for v in videos)
    grid = [hstack(a, b) for a, b in zip(videos[0][:n], videos[1][:n])]
    save_video(grid, "a5_pull", gif_scale=0.4)
    plot_a5(results, W, rate)
    slips_first = lambda first: first.get("robot slides (right foot 5 mm)", (np.inf,))[0] < first.get("left foot lifts off", (np.inf,))[0]
    assert not slips_first(results[0.6][1]) and slips_first(results[0.1][1])


def plot_a5(results, W, rate):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for mu, (log, first) in results.items():
        force = -log["pull"][:, 1]
        end = len(force) if log["fell_at"] is None else np.searchsorted(log["t"], log["fell_at"])
        keep = (force > 1) & (np.arange(len(force)) < end)          # up to the fall (impacts after it)
        smooth = lambda x: np.convolve(x, np.ones(25) / 25, "same")  # 50 ms average: sliding feet stick and slip
        need = smooth(np.linalg.norm(log["total"][:, :2], axis=1) / np.maximum(log["total"][:, 2], 1e-9))
        line, = ax[0].plot(force[keep], need[keep], label=f"mu = {mu}: needed |F_t| / F_n")
        ax[0].axhline(mu, color=line.get_color(), ls=":", lw=1)
        ax[1].plot(force[:end], smooth(log["left"])[:end], color=line.get_color(), label=f"mu = {mu}: left foot load")
        for k, (t, f) in first.items():
            if k != "left foot scuffs (5 mm)":
                ax[0 if "slides" in k else 1].axvline(f, color=line.get_color(), lw=0.8, ls="--")
    ax[1].set_ylim(0, 250)
    ax[0].set_ylim(0, 0.7)
    ax[0].set_xlabel("pull (N)")
    ax[0].set_ylabel("friction needed (total)")
    ax[0].set_title("friction: slides when the need reaches mu (dotted)\n(dashed: the robot slides; up to the fall)",
                    fontsize=10)
    ax[1].set_xlabel("pull (N)")
    ax[1].set_ylabel("vertical force on the left foot (N)")
    fig.text(0.5, 0.005, "50 ms averages; curves end where the robot falls", ha="center", fontsize=8)
    ax[1].set_title("support polygon: tips when the left foot unloads\n(dashed: left foot lifts off)", fontsize=10)
    for a in ax:
        a.set_xlim(0, 90)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(results_dir / "a5_pull.png", dpi=110)
    plt.close(fig)
    print(f"saved {results_dir / 'a5_pull.png'}")


PARTS = [part_a1, part_a2, part_a3, part_a4, part_a5]


# ---------------------------------------------------------------------------------------------------------
# Interactive: the viewer
# ---------------------------------------------------------------------------------------------------------

def interactive(model, data, chain):
    """MuJoCo's viewer with the contact forces drawn live. Keys: 1 stand, 2 weight on the left foot, 3 flamingo,
    I toggles ice (mu 0.1), Backspace resets. Ctrl + right-drag on a body pushes it: watch the CoP run to the edge
    of the support polygon, and the friction cones fill up.
    """
    import mujoco.viewer
    targets = {49: "stand", 50: "weight on left", 51: "flamingo"}
    feet = [g for g in range(model.ngeom) if re.fullmatch(g1_utils.FOOT_PATTERN, model.geom(g).name)]
    state = {"s": 0.0, "goal": 0.0, "reset": False}

    def on_key(key):
        if key in targets:
            state["goal"] = float(chain["marks"][targets[key]])
        elif key == 73:                                              # I: ice on / off
            model.geom_friction[feet, 0] = 0.1 if model.geom_friction[feet[0], 0] > 0.3 else 0.6
        elif key == 259:                                             # Backspace
            state["reset"] = True

    ctrl, speed = ChainController(model, chain), 5.0                 # poses per second along the chain
    mujoco.mj_resetData(model, data)
    data.qpos[:] = chain["q"][0]
    mujoco.mj_forward(model, data)
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        viewer.cam.lookat[:], viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = [0, 0, 0.45], 2.4, 155, -18
        while viewer.is_running():
            start = time.time()
            with viewer.lock():
                if state["reset"]:
                    mujoco.mj_resetData(model, data)
                    data.qpos[:] = chain["q"][0]
                    state.update(s=0.0, goal=0.0, reset=False)
                    mujoco.mj_forward(model, data)
                for _ in range(round(1 / (60 * model.opt.timestep))):
                    step = np.clip(state["goal"] - state["s"], -speed * model.opt.timestep, speed * model.opt.timestep)
                    state["s"] += step
                    ctrl(data, state["s"], step / model.opt.timestep)
                    mujoco.mj_step(model, data)
                st = contact_state(model, data)
                viewer.user_scn.ngeom = 0
                draw_overlay(viewer.user_scn, st)
                labels = "contacts\nload left / right\nCoP inside polygon by\nlargest cone use\nfriction mu"
                values = (f"{len(st.contacts)}\n{st.load['left']:.0f} / {st.load['right']:.0f} N\n"
                          f"{'-' if st.margin is None else f'{1000 * st.margin:.0f} mm'}\n{st.usage:.2f}\n"
                          f"{model.geom_friction[feet[0], 0]:.1f}")
                viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, labels, values),
                                  (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                                   "keys\n\n\n\nmouse", "1 stand   2 weight on left   3 flamingo\nI ice (mu 0.1) on/off\n"
                                   "Backspace reset\n \nctrl + right-drag on a body: push it")])
            viewer.sync()
            time.sleep(max(0.0, 1 / 60 - (time.time() - start)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer", action="store_true", help="interactive viewer (macOS: run with mjpython)")
    args = parser.parse_args()
    model, data, joints = F.build_torque_g1()
    F.brighten(model)
    style_forces(model)
    t0 = time.time()
    chain = build_chain(model, data, joints)
    print(f"chain of {len(chain['q'])} balanced poses (stand -> weight on left -> flamingo) in {time.time() - t0:.0f} s; "
          f"largest pelvis residual {max(np.abs(r).max() for r in chain['residual']):.1e}")
    if args.viewer:
        interactive(model, data, chain)
        return
    ctx = {"chain": chain}
    for part in PARTS:
        part(model, data, ctx)
    print("\n=== Part A complete ===")


if __name__ == "__main__":
    main()
