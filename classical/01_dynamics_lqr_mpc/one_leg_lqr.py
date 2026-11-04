"""Module 1, Part F: the G1 balancing on one leg with LQR, in yoga poses.

The same recipe as Part B's cart-pole, on the whole robot (35 velocities, 29 motors):
    1. a pose to balance around: the standing foot flat on the floor, the centre of mass over it   (IK, Part E)
    2. the torques that hold that pose still                                                     (inverse dynamics)
    3. a linear model around it: s_next = A s + B u, with the foot's contact included             (mjd_transitionFD)
    4. a cost that says what "balanced" means, and LQR                                            (Part B's dlqr)
    5. u = u0 - K (s - s0) in simulation; pushes; and a flow between poses with blended gains     (gain scheduling)

After DeepMind's MuJoCo LQR tutorial (a humanoid on one leg), adapted to the G1.

    .venv/bin/python classical/01_dynamics_lqr_mpc/one_leg_lqr.py
"""

import re
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import scipy.linalg
from PIL import Image, ImageDraw, ImageFont

script_dir = Path(__file__).resolve().parent
repo = script_dir.parents[1]
sys.path.insert(0, str(repo / "exercises"))
import g1_utils
from render_utils import make_camera, save_gif, side_by_side

from lqr import dlqr

results_dir = script_dir / "results"
results_dir.mkdir(exist_ok=True)

STANCE, SWING = "left_foot", "right_foot"       # stand on the left foot
POSE_NAMES = ("flamingo", "tree", "warrior3")


# ---------------------------------------------------------------------------------------------------------
# F1: a torque-controlled G1
# ---------------------------------------------------------------------------------------------------------

def build_torque_g1(timestep: float = 0.002):
    """The G1 with one torque motor per joint (ctrl = joint torque, N m), instead of exercise 03's PD position
    actuators. Same motors otherwise: torque limits and armature (rotor inertia) from g1_utils, same feet and
    floor. Returns (model, data, joint_names), at the knees-bent pose with the lowest foot capsule on the floor.
    """
    spec = mujoco.MjSpec.from_file(str(repo / "robot" / "g1.xml"))
    g1_utils.apply_collision_settings(spec)
    g1_utils.add_floor(spec)
    joints = g1_utils.hinge_joint_names(spec)
    for patterns, motor, n_motors in g1_utils.ACTUATOR_GROUPS:
        _, _, effort, armature = g1_utils.group_params(motor, n_motors)
        for name in joints:
            if any(re.fullmatch(p, name) for p in patterns):
                act = spec.add_actuator(name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
                act.ctrllimited = True
                act.ctrlrange = [-effort, effort]          # a motor actuator: force = ctrl, within the limit
                spec.joint(name).armature = armature
    spec.option.timestep = timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model = spec.compile()
    data = mujoco.MjData(model)
    q = np.concatenate([[0, 0, g1_utils.BASE_HEIGHT], [1, 0, 0, 0], g1_utils.default_joint_pos(joints)])
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    data.qpos[2] -= lowest_foot_point(model, data)          # the capsules, not the foot site, touch the floor
    mujoco.mj_forward(model, data)
    return model, data, joints


def lowest_foot_point(model, data) -> float:
    """Height of the lowest point of any foot collision capsule (what actually touches the floor)."""
    z = []
    for g in range(model.ngeom):
        if re.fullmatch(g1_utils.FOOT_PATTERN, model.geom(g).name):
            axis = data.geom_xmat[g].reshape(3, 3)[:, 2]            # a capsule lies along its local z
            radius, half_length = model.geom_size[g][:2]
            z += [data.geom_xpos[g][2] + s * half_length * axis[2] - radius for s in (1, -1)]
    return min(z)


# ---------------------------------------------------------------------------------------------------------
# F2: one-leg poses by IK
# ---------------------------------------------------------------------------------------------------------

def jac(model, data, kind: str, name: str | None = None):
    """(position Jacobian, rotation Jacobian) of a site or body, or the CoM Jacobian (3 x nv)."""
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    if kind == "site":
        mujoco.mj_jacSite(model, data, jp, jr, model.site(name).id)
    elif kind == "body":
        mujoco.mj_jacBody(model, data, jp, jr, model.body(name).id)
    else:
        mujoco.mj_jacSubtreeCom(model, data, jp, model.body("pelvis").id)
    return jp, jr


def point_world(data, body: str, local: np.ndarray) -> np.ndarray:
    """World position of a point fixed in a body (given in the body's frame)."""
    b = data.body(body)
    return b.xpos + b.xmat.reshape(3, 3) @ local


def point_jac(model, data, body: str, local: np.ndarray) -> np.ndarray:
    """3 x nv position Jacobian of a point fixed in a body (mj_jac at that point, Part E)."""
    jp = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jp, None, point_world(data, body, local), model.body(body).id)
    return jp


def rotation_error(R: np.ndarray, R_target: np.ndarray) -> np.ndarray:
    """A world-frame rotation vector that turns R towards R_target (zero when they are equal)."""
    return 0.5 * sum(np.cross(R[:, i], R_target[:, i]) for i in range(3))


def flat(R: np.ndarray) -> np.ndarray:
    """R's heading, but level: z straight up. A foot with this orientation is flat on the floor."""
    x = np.array([R[0, 0], R[1, 0], 0.0])
    x /= np.linalg.norm(x)
    z = np.array([0.0, 0.0, 1.0])
    return np.column_stack([x, np.cross(z, x), z])


def solve_ik(model, data, q_nominal, tasks, iters: int = 400, posture_weight: float = 0.05, damping: float = 1e-3):
    """Weighted least-squares IK (Gauss-Newton). Each iteration solves for the q step dq that best reduces every
    task error at once, weighted:  min sum_i w_i |J_i dq - e_i|^2 + posture_weight |dq_joints - e_posture|^2
                                    + damping |dq|^2
    tasks: ("site_pos"/"site_rot"/"body_pos"/"body_rot", name, target, weight), ("com_xy", None, target, w), or
    ("point_pos", (body name, point in the body's frame), target, w) for any point on a body (e.g. a toe tip).
    The posture term pulls every joint towards q_nominal, which picks one pose among the many that meet the tasks.
    """
    q = q_nominal.copy()
    lo, hi = model.jnt_range[1:, 0], model.jnt_range[1:, 1]
    for _ in range(iters):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        rows, errors, weights = [], [], []
        for kind, name, target, w in tasks:
            if kind == "com_xy":
                J, e = jac(model, data, "com")[0][:2], target - data.subtree_com[1][:2]
            elif kind == "point_pos":
                J, e = point_jac(model, data, *name), target - point_world(data, *name)
            else:
                where, what = kind.split("_")
                frame = data.site(name) if where == "site" else data.body(name)
                jp, jr = jac(model, data, where, name)
                if what == "pos":
                    J, e = jp, target - frame.xpos
                else:
                    J, e = jr, rotation_error(frame.xmat.reshape(3, 3), target)
            rows.append(J)
            errors.append(e)
            weights.append(np.full(len(e), w))
        to_nominal = np.zeros(model.nv)
        mujoco.mj_differentiatePos(model, to_nominal, 1.0, q, q_nominal)
        rows.append(np.eye(model.nv)[6:])
        errors.append(to_nominal[6:])
        weights.append(np.full(model.nv - 6, posture_weight))
        J, e, W = np.vstack(rows), np.concatenate(errors), np.concatenate(weights)
        dq = np.linalg.solve(J.T @ (W[:, None] * J) + damping * np.eye(model.nv), J.T @ (W * e))
        mujoco.mj_integratePos(model, q, dq, 1.0)
        q[7:] = np.clip(q[7:], lo, hi)
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    return q


def self_contacts(model, data) -> list:
    """Contacts that are not with the floor (limb against limb)."""
    names = lambda c: (model.geom(c.geom1).name, model.geom(c.geom2).name)
    return [names(c) for c in data.contact[:data.ncon] if "floor" not in names(c)]


def roty(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def make_poses(model, data, joints):
    """The three yoga poses. Every pose: left foot flat in place, CoM over it (strong weights); then pose-specific
    targets for the right foot, torso, knee and arms (weaker, so they give way if they conflict)."""
    q_home = data.qpos.copy()
    F = data.site(STANCE).xpos.copy()                                  # the standing foot, where it is
    R_foot = flat(data.site(STANCE).xmat.reshape(3, 3))                # ...but exactly flat
    R_torso = flat(data.body("torso_link").xmat.reshape(3, 3))
    joint = lambda n: 7 + joints.index(n)
    specs = {
        "flamingo": ({"left_shoulder_roll_joint": 1.3, "right_shoulder_roll_joint": -1.3,
                      "left_elbow_joint": 0.0, "right_elbow_joint": 0.0},
                     [("site_pos", SWING, F + [0.15, -0.20, 0.30], 10), ("body_rot", "torso_link", R_torso, 10)]),
        "tree": ({"left_shoulder_pitch_joint": -2.6, "right_shoulder_pitch_joint": -2.6,
                  "left_shoulder_roll_joint": 0.55, "right_shoulder_roll_joint": -0.55,
                  "left_elbow_joint": 0.0, "right_elbow_joint": 0.0,
                  "right_hip_roll_joint": -0.9, "right_hip_yaw_joint": -0.6, "right_knee_joint": 2.2},
                 [("site_pos", SWING, F + [0.03, -0.09, 0.40], 10),
                  ("body_pos", "right_knee_link", F + [0.10, -0.45, 0.42], 3), ("body_rot", "torso_link", R_torso, 10)]),
        "warrior3": ({"left_shoulder_pitch_joint": -2.8, "right_shoulder_pitch_joint": -2.8,
                      "left_shoulder_roll_joint": 0.45, "right_shoulder_roll_joint": -0.45,
                      "left_elbow_joint": 0.0, "right_elbow_joint": 0.0, "right_knee_joint": 0.0, "left_knee_joint": 0.1},
                     [("body_rot", "torso_link", R_torso @ roty(np.deg2rad(75)), 10),
                      ("site_pos", SWING, F + [-0.80, -0.14, 0.55], 5)]),
    }
    poses = {}
    for name, (nominal, tasks) in specs.items():
        q_nom = q_home.copy()
        for jn, value in nominal.items():
            q_nom[joint(jn)] = value
        base = [("site_pos", STANCE, F, 100), ("site_rot", STANCE, R_foot, 300), ("com_xy", None, F[:2], 300)]
        poses[name] = solve_ik(model, data, q_nom, base + tasks)
    data.qpos[:] = q_home
    mujoco.mj_forward(model, data)
    return poses, F, R_foot


# ---------------------------------------------------------------------------------------------------------
# F3: the torques that hold a pose still
# ---------------------------------------------------------------------------------------------------------

def inverse_dynamics(model, data, q) -> np.ndarray:
    """Generalized forces (nv) that give zero acceleration at q, at rest: qfrc_inverse with qacc = 0.

    Its first 6 entries are forces/torques the pelvis would need from outside: no motor acts there, so for a real
    equilibrium they must be ~0, i.e. the floor (through the foot contact) carries the robot.
    """
    mujoco.mj_resetData(model, data)
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    data.qacc[:] = 0.0
    mujoco.mj_inverse(model, data)
    return data.qfrc_inverse.copy()


def equilibrium(model, data, q):
    """Pose height at which the floor carries the weight, and the motor torques that hold it.

    MuJoCo's contact is soft: the floor pushes back more the deeper the foot sinks (about 6 kN per mm here). Too
    high, no contact, and the pelvis would need the full weight from outside; too low, the floor pushes too hard.
    Bisect the height until the vertical force needed on the pelvis is ~0.
    """
    lo, hi = -0.02, 0.02            # metres around the IK pose's height
    for _ in range(50):
        mid = (lo + hi) / 2
        q_try = q.copy()
        q_try[2] += mid
        if inverse_dynamics(model, data, q_try)[2] > 0:
            hi = mid                    # floor not yet carrying the weight: go lower
        else:
            lo = mid
    q = q.copy()
    q[2] += (lo + hi) / 2
    f = inverse_dynamics(model, data, q)
    dof = model.jnt_dofadr[model.actuator_trnid[:, 0]]       # each motor drives one joint, gear 1
    return q, f[dof].copy(), f[:6].copy()


# ---------------------------------------------------------------------------------------------------------
# F4: LQR
# ---------------------------------------------------------------------------------------------------------

def linearize(model, data, q, ctrl0):
    """A (2nv x 2nv), B (2nv x nu) of one simulation step at (q, rest, ctrl0), contacts included.

    The state is (dq, qdot): dq is the 35-number difference from q (mj_differentiatePos), not the 36-number qpos,
    so the base orientation's quaternion never appears (Part E1).
    """
    mujoco.mj_resetData(model, data)
    data.qpos[:] = q
    data.ctrl[:] = ctrl0
    mujoco.mj_forward(model, data)
    A = np.zeros((2 * model.nv, 2 * model.nv))
    B = np.zeros((2 * model.nv, model.nu))
    mujoco.mjd_transitionFD(model, data, 1e-6, True, A, B, None, None)
    return A, B


def cost_matrices(model, data, joints, q, balance: float = 1000.0, r0: float = 1.0,
                  balance_joint: float = 3.0, other_joint: float = 0.3, scale_r: bool = True,
                  support_jac=None, both_legs: bool = False):
    """Q (2nv x 2nv), R (nu x nu): what LQR trades off.

    Balance: |(J_com - J_foot) dq|^2, the CoM moving relative to the standing foot (Part E's Jacobians). The
    robot falls when its CoM leaves the foot, so this is the term that matters; everything else just keeps the
    pose tidy: small costs on joints leaving the pose (larger for the standing leg and waist, which balance).
    Force: R = r0 * diag(1 / limit^2), a cost on each torque relative to its motor's limit. (DeepMind's humanoid
    has normalized controls; with raw N m an identity R makes torque 10^4 times more expensive and LQR too weak
    to hold the pose: scale_r=False shows it.)
    support_jac(model, data) -> 3 x nv, if given, replaces the stance foot site as the point the CoM must stay over
    (Part G: a toe tip, or the midpoint between two toes); both_legs counts both legs as balancing joints.
    """
    nv = model.nv
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    J_com, _ = jac(model, data, "com")
    J_support = jac(model, data, "site", STANCE)[0] if support_jac is None else support_jac(model, data)
    J_bal = J_com - J_support
    legs = ("left_", "right_") if both_legs else ("left_",)
    balancing = {model.jnt_dofadr[model.joint(n).id] for n in joints
                 if n.startswith("waist") or (n.startswith(legs) and re.search("hip|knee|ankle", n))}
    joint_cost = np.diag([0.0] * 6 + [balance_joint if i in balancing else other_joint for i in range(6, nv)])
    Q = 1e-8 * np.eye(2 * nv)       # tiny cost on everything: keeps the Riccati problem solvable despite modes that
    Q[:nv, :nv] += balance * J_bal.T @ J_bal + joint_cost      # sit at 1 and nothing acts on (sliding on the floor)
    limits = model.actuator_ctrlrange[:, 1]
    R = r0 * (np.diag(1 / limits**2) if scale_r else np.eye(model.nu))
    return Q, R


def lqr_gain(A, B, Q, R, own: bool = True):
    """K from Part B's dlqr (own=True), or scipy's Riccati solver.

    Part B's dlqr iterates the Riccati equation; it converges at the pace of the slowest closed-loop mode. Here
    some modes sit very close to 1 (sliding the whole robot sideways costs almost nothing), so it needs ~10^5
    iterations for the flamingo and does not converge within 2 x 10^5 for warrior III. F4 uses it (and checks it
    against scipy); everything else uses scipy, which solves the same equation directly.
    """
    if own:
        K, _ = dlqr(A, B, Q, R, tol=1e-9, max_iter=200_000)
        return K
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    return np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)


def lqr_control(model, data, q0, ctrl0, K):
    """u = u0 - K (s - s0), with s - s0 = (q difference, velocity)."""
    dx = np.zeros(2 * model.nv)
    mujoco.mj_differentiatePos(model, dx[:model.nv], 1.0, q0, data.qpos)
    dx[model.nv:] = data.qvel
    data.ctrl[:] = ctrl0 - K @ dx


def fallen(model, data, q0) -> bool:
    """Anything but the standing foot touches the floor, or the pelvis dropped 15 cm."""
    for c in data.contact[:data.ncon]:
        names = model.geom(c.geom1).name + model.geom(c.geom2).name
        if "floor" in names and "left_foot" not in names:
            return True
    return data.qpos[2] < q0[2] - 0.15


def balance_run(model, data, q0, ctrl0, K, duration: float = 4.0, push=None) -> bool:
    """True if the robot is still up after `duration`. push = (time, force (3,)): 0.1 s on the torso."""
    mujoco.mj_resetData(model, data)            # also resets data.time: the push window depends on it
    data.qpos[:] = q0
    mujoco.mj_forward(model, data)
    torso = model.body("torso_link").id
    for _ in range(round(duration / model.opt.timestep)):
        lqr_control(model, data, q0, ctrl0, K)
        data.xfrc_applied[:] = 0.0
        if push is not None and push[0] <= data.time < push[0] + 0.1:
            data.xfrc_applied[torso, :3] = push[1]
        mujoco.mj_step(model, data)
        if fallen(model, data, q0):
            return False
    return True


def max_push(model, data, q0, ctrl0, K, direction, hi: float = 300.0) -> float:
    """Largest 0.1 s push (N) on the torso the robot survives, by bisection (assumes a single threshold)."""
    lo = 0.0
    for _ in range(8):
        mid = (lo + hi) / 2
        if balance_run(model, data, q0, ctrl0, K, duration=3.0, push=(0.5, mid * np.asarray(direction, float))):
            lo = mid
        else:
            hi = mid
    return lo


DIRECTIONS = {"forward": (1, 0, 0), "back": (-1, 0, 0), "left": (0, 1, 0), "right": (0, -1, 0)}


# ---------------------------------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------------------------------

def brighten(model):
    model.vis.headlight.ambient[:] = 0.45
    model.vis.headlight.diffuse[:] = 0.65


def write_mp4(frames, path: Path, fps: float):
    """Frames (PIL images, same size) to an H.264 MP4 with ffmpeg."""
    w, h = frames[0].size
    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{w}x{h}", "-r", f"{fps:.3f}", "-i", "-", "-pix_fmt", "yuv420p",
                             "-vcodec", "libx264", "-crf", "23", str(path)], stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(f.convert("RGB").tobytes())
    proc.stdin.close()
    proc.wait()


def label(img: Image.Image, text: str) -> Image.Image:
    ImageDraw.Draw(img).text((10, 8), text, fill=(255, 255, 255), font=ImageFont.load_default(size=18))
    return img


# ---------------------------------------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------------------------------------

def part_f1(model, data, joints, ctx):
    print("\n=== F1: a torque-controlled G1 ===")
    ref_model, _, _, _ = g1_utils.build_g1_model(repo / "robot" / "g1.xml")    # exercise 03's PD version
    same_armature = np.allclose(model.dof_armature, ref_model.dof_armature)
    print(f"{model.nu} torque motors; limits {sorted(set(model.actuator_ctrlrange[:, 1]))} N m; "
          f"armature same as exercise 03's model: {same_armature}; timestep {model.opt.timestep} s")
    print(f"lowest foot capsule at z = {1000 * lowest_foot_point(model, data):.2f} mm; the foot site is at "
          f"{1000 * data.site(STANCE).xpos[2]:+.1f} mm: the capsules touch the floor, not the site")
    assert model.nu == 29 and same_armature


def part_f2(model, data, joints, ctx):
    print("\n=== F2: one-leg poses by IK (standing foot flat, CoM over it) ===")
    poses, F, R_foot = make_poses(model, data, joints)
    for name, q in poses.items():
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        com_err = 1000 * np.linalg.norm(data.subtree_com[1][:2] - F[:2])
        tilt = np.degrees(np.arccos(np.clip(data.site(STANCE).xmat.reshape(3, 3)[2, 2], -1, 1)))
        contacts = self_contacts(model, data)
        print(f"  {name:<9}: CoM {com_err:.2f} mm from the foot centre; foot tilt {tilt:.3f} deg; "
              f"swing foot at {np.round(data.site(SWING).xpos, 2)}; self-contacts: {contacts or 'none'}")
        assert com_err < 6 and tilt < 0.05 and not contacts
    cam = make_camera(lookat=[0, 0, 0.55], distance=2.4, azimuth=140, elevation=-10)
    images = []
    brighten(model)
    with mujoco.Renderer(model, height=360, width=360) as renderer:
        for name, q in poses.items():
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            images.append(label(Image.fromarray(renderer.render()), name))
    strip = Image.new("RGB", (360 * len(images), 360))
    for i, im in enumerate(images):
        strip.paste(im, (360 * i, 0))
    strip.save(results_dir / "f2_poses.png")
    print(f"saved {results_dir / 'f2_poses.png'}")
    ctx.update(poses=poses, F=F, R_foot=R_foot)


def part_f3(model, data, joints, ctx):
    print("\n=== F3: the torques that hold a pose still ===")
    poses = ctx["poses"]
    limits = model.actuator_ctrlrange[:, 1]
    eq = {}
    for name, q in poses.items():
        q0, ctrl0, residual = equilibrium(model, data, q)
        eq[name] = (q0, ctrl0)
        print(f"  {name:<9}: height changed by {1000 * (q0[2] - q[2]):+.2f} mm; force still needed on the pelvis "
              f"{np.abs(residual[:3]).max():.2f} N, {np.abs(residual[3:]).max():.2f} N m; "
              f"largest torque {np.max(np.abs(ctrl0) / limits):.0%} of its limit")
        assert np.abs(residual[:3]).max() < 0.1 and np.abs(residual[3:]).max() < 5 and np.all(np.abs(ctrl0) < limits)

    # Why the foot must be exactly flat: the same flamingo with the foot tilted like the keyframe's (0.34 deg)
    q_tilted = poses["flamingo"].copy()
    tilt = roty(np.deg2rad(-0.34))
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, (tilt @ _quat_to_mat(q_tilted[3:7])).ravel())
    q_tilted[3:7] = quat                                  # tilt the whole robot about the foot's pitch axis
    _, _, residual = equilibrium(model, data, q_tilted)
    print(f"  (tilt the whole flamingo by 0.34 deg: the pelvis would need {np.abs(residual[3:]).max():.0f} N m from "
          f"outside. Soft contact: the force follows how deep each capsule sinks, so a 0.7 mm tilt puts the weight "
          f"on one end of the foot)")
    ctx["eq"] = eq


def _quat_to_mat(quat) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, quat)
    return R.reshape(3, 3)


def part_f4(model, data, joints, ctx):
    print("\n=== F4: LQR on the flamingo ===")
    q0, ctrl0 = ctx["eq"]["flamingo"]
    A, B = linearize(model, data, q0, ctrl0)
    Q, R = cost_matrices(model, data, joints, q0)
    K = lqr_gain(A, B, Q, R, own=True)
    K_scipy = lqr_gain(A, B, Q, R, own=False)
    print(f"A {A.shape}, B {B.shape}; K {K.shape}; own dlqr vs scipy: max |K difference| / max |K| = "
          f"{np.abs(K - K_scipy).max() / np.abs(K_scipy).max():.1e}")
    assert np.abs(K - K_scipy).max() < 1e-3 * np.abs(K_scipy).max()

    # Holding still without LQR (the holding torques alone), with LQR, and with an unscaled R
    print(f"  holding torques only (no feedback): stays up 4 s: "
          f"{balance_run(model, data, q0, ctrl0, np.zeros_like(K))}")
    print(f"  LQR: stays up 4 s: {balance_run(model, data, q0, ctrl0, K)}")
    Q_u, R_u = cost_matrices(model, data, joints, q0, scale_r=False)
    K_u = lqr_gain(A, B, Q_u, R_u, own=False)
    print(f"  LQR with R = identity on raw N m (gains {np.abs(K_u).max():.0f} vs {np.abs(K).max():.0f}): "
          f"stays up 4 s: {balance_run(model, data, q0, ctrl0, K_u)}")
    assert balance_run(model, data, q0, ctrl0, K) and not balance_run(model, data, q0, ctrl0, np.zeros_like(K))

    # What the balance term buys: largest push, with and without it
    Q_nb, R_nb = cost_matrices(model, data, joints, q0, balance=0.0)
    K_nb = lqr_gain(A, B, Q_nb, R_nb, own=False)
    print(f"  {'largest 0.1 s push on the torso (N)':<38} | " + " | ".join(f"{d:>7}" for d in DIRECTIONS))
    for label_, gain in (("with the CoM-over-foot term", K), ("without it (joint costs only)", K_nb)):
        pushes = [max_push(model, data, q0, ctrl0, gain, v) for v in DIRECTIONS.values()]
        print(f"  {label_:<38} | " + " | ".join(f"{p:>7.0f}" for p in pushes))
    ctx["K"] = {"flamingo": K}


def part_f5(model, data, joints, ctx):
    print("\n=== F5: all three poses, and pushes ===")
    print(f"  {'pose':<9} | {'holds 5 s':<9} | largest 0.1 s push (N): " + ", ".join(DIRECTIONS))
    panels = []
    cam = make_camera(lookat=[0, 0, 0.55], distance=2.6, azimuth=140, elevation=-10)
    brighten(model)
    for name in POSE_NAMES:
        q0, ctrl0 = ctx["eq"][name]
        if name not in ctx["K"]:
            A, B = linearize(model, data, q0, ctrl0)
            ctx["K"][name] = lqr_gain(A, B, *cost_matrices(model, data, joints, q0), own=False)
        K = ctx["K"][name]
        holds = balance_run(model, data, q0, ctrl0, K, duration=5.0)
        pushes = [max_push(model, data, q0, ctrl0, K, v) for v in DIRECTIONS.values()]
        print(f"  {name:<9} | {str(holds):<9} | " + ", ".join(f"{p:.0f}" for p in pushes))
        assert holds and min(pushes) > 10

        # Video: hold, then a 30 N sideways push for 0.1 s at t = 1 s
        frames = []
        mujoco.mj_resetData(model, data)
        data.qpos[:] = q0
        mujoco.mj_forward(model, data)
        torso, steps_per_frame = model.body("torso_link").id, round((1 / 30) / model.opt.timestep)
        with mujoco.Renderer(model, height=360, width=360) as renderer:
            for i in range(round(3.0 / model.opt.timestep)):
                lqr_control(model, data, q0, ctrl0, K)
                data.xfrc_applied[:] = 0.0
                pushing = 1.0 <= data.time < 1.1
                if pushing:
                    data.xfrc_applied[torso, :3] = [0, 30.0, 0]
                mujoco.mj_step(model, data)
                if i % steps_per_frame == 0:
                    renderer.update_scene(data, camera=cam)
                    frames.append(label(Image.fromarray(renderer.render()), f"{name}{'   <- push' if pushing else ''}"))
        panels.append(frames)
    save_gif(side_by_side(panels), results_dir / "f5_poses_push.gif", fps=round(1 / (steps_per_frame * model.opt.timestep)))
    print(f"saved {results_dir / 'f5_poses_push.gif'}")


FLOW = ("flamingo", "tree", "warrior3", "flamingo")


def build_flow_chain(model, data, joints, ctx, sequence=FLOW, n_wp: int = 20) -> dict:
    """Poses between the yoga poses, each balanced, with its equilibrium (q0, u0), linear model (A, B) and gain K.

    Gain scheduling: LQR is only valid near the pose it was designed for, so the flow uses a chain of poses, each
    on a straight line in joint space between two yoga poses and then moved by IK to keep the foot flat and the
    CoM over it. n_wp poses per transition.
    """
    poses, F, R_foot = ctx["poses"], ctx["F"], ctx["R_foot"]
    base = [("site_pos", STANCE, F, 100), ("site_rot", STANCE, R_foot, 300), ("com_xy", None, F[:2], 300)]
    chain = {"q": [], "u": [], "K": [], "A": [], "B": [], "sequence": list(sequence), "n_wp": n_wp}
    for a, b in zip(sequence[:-1], sequence[1:]):
        step = np.zeros(model.nv)
        mujoco.mj_differentiatePos(model, step, 1.0, poses[a], poses[b])
        last = (a, b) == (sequence[-2], sequence[-1])
        for k in range(n_wp + (1 if last else 0)):
            between = poses[a].copy()
            mujoco.mj_integratePos(model, between, step, k / n_wp)            # straight line in joint space ...
            q = solve_ik(model, data, between, base, iters=150, posture_weight=1.0)   # ... moved to keep balance
            contacts = self_contacts(model, data)
            assert not contacts, f"{a} -> {b} at {k / n_wp:.2f}: {contacts}"
            q0, ctrl0, _ = equilibrium(model, data, q)
            A, B = linearize(model, data, q0, ctrl0)
            for key, val in (("q", q0), ("u", ctrl0), ("A", A), ("B", B),
                             ("K", lqr_gain(A, B, *cost_matrices(model, data, joints, q0), own=False))):
                chain[key].append(val)
    return chain


def run_flow(model, data, chain, t_move: float = 3.0, t_hold: float = 2.0, gains=None, record: bool = False):
    """Run the flow along a chain. gains(s) -> K overrides the blended gains (H6 tries other schedules).

    The reference slides along the chain on a smooth time profile (hold t_hold, move t_move, ...). At position s
    (e.g. 23.4 = 40 % from pose 23 to 24): the reference pose and its velocity, and (u0, K) blended linearly
    between the two neighbours. Returns (fell, largest joint error, standing-foot drift (mm), frames).
    """
    n, n_wp, sequence = len(chain["q"]), chain["n_wp"], chain["sequence"]

    def progress(t):
        """Position along the chain (0 .. n-1) and its rate, at time t: hold, move (smoothly), hold, ..."""
        i, tau = int(t // (t_hold + t_move)), t % (t_hold + t_move)
        if i >= len(sequence) - 1:
            return float(n - 1), 0.0
        if tau < t_hold:
            return float(i * n_wp), 0.0
        x = (tau - t_hold) / t_move
        return i * n_wp + n_wp * x * x * (3 - 2 * x), n_wp * 6 * x * (1 - x) / t_move

    def control(t):
        s, s_dot = progress(t)
        a = int(s)
        b, f = min(a + 1, n - 1), s - int(s)
        dq = np.zeros(model.nv)
        mujoco.mj_differentiatePos(model, dq, 1.0, chain["q"][a], chain["q"][b])
        q_ref = chain["q"][a].copy()
        mujoco.mj_integratePos(model, q_ref, dq, f)
        dx = np.zeros(2 * model.nv)
        mujoco.mj_differentiatePos(model, dx[:model.nv], 1.0, q_ref, data.qpos)
        dx[model.nv:] = data.qvel - dq * s_dot                                # velocity relative to the reference's
        K = (1 - f) * chain["K"][a] + f * chain["K"][b] if gains is None else gains(s)
        data.ctrl[:] = (1 - f) * chain["u"][a] + f * chain["u"][b] - K @ dx
        return q_ref, s

    duration = (t_hold + t_move) * (len(sequence) - 1) + t_hold
    mujoco.mj_resetData(model, data)
    data.qpos[:] = chain["q"][0]
    mujoco.mj_forward(model, data)
    foot0 = data.site(STANCE).xpos.copy()
    worst, fell, frames = 0.0, False, []
    cam = make_camera(lookat=[0, 0, 0.55], distance=2.6, azimuth=140, elevation=-10)
    steps_per_frame = round((1 / 30) / model.opt.timestep)
    renderer = mujoco.Renderer(model, height=480, width=640) if record else None
    for i in range(round(duration / model.opt.timestep)):
        q_ref, s = control(data.time)
        mujoco.mj_step(model, data)
        err = np.zeros(model.nv)
        mujoco.mj_differentiatePos(model, err, 1.0, q_ref, data.qpos)
        worst = max(worst, np.abs(err[6:]).max())
        if fallen(model, data, chain["q"][0]):
            fell = True
            if not record:
                break
        if record and i % steps_per_frame == 0:
            renderer.update_scene(data, camera=cam)
            frames.append(label(Image.fromarray(renderer.render()),
                                f"t = {data.time:4.1f} s   {sequence[min(round(s / n_wp), len(sequence) - 1)]}"))
    if renderer is not None:
        renderer.close()
    return fell, worst, 1000 * np.linalg.norm(data.site(STANCE).xpos - foot0), frames


def part_f6(model, data, joints, ctx):
    print("\n=== F6: yoga flow: flamingo -> tree -> warrior III -> flamingo, on one foot ===")
    chain = build_flow_chain(model, data, joints, ctx)
    print(f"{len(chain['q'])} poses in the chain; largest torque along it "
          f"{max(np.max(np.abs(u) / model.actuator_ctrlrange[:, 1]) for u in chain['u']):.0%} of its limit")
    brighten(model)
    fell, worst, moved, frames = run_flow(model, data, chain, record=True)
    duration = len(frames) * round((1 / 30) / model.opt.timestep) * model.opt.timestep
    print(f"{duration:.0f} s on one foot: fell {fell}; largest joint error from the reference {worst:.3f} rad; "
          f"the standing foot moved {moved:.1f} mm")
    write_mp4(frames, results_dir / "f6_yoga_flow.mp4", fps=1 / (round((1 / 30) / model.opt.timestep) * model.opt.timestep))
    print(f"saved {results_dir / 'f6_yoga_flow.mp4'}")
    assert not fell and moved < 2.0
    ctx["chain"] = chain


PARTS = [part_f1, part_f2, part_f3, part_f4, part_f5, part_f6]


def main():
    model, data, joints = build_torque_g1()
    ctx = {}
    for part in PARTS:
        part(model, data, joints, ctx)
    print("\n=== Part F complete ===")


if __name__ == "__main__":
    main()
