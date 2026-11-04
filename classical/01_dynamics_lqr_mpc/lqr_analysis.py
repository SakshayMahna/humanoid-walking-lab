"""Module 1, Part H: what the one-leg LQR controller does, mathematically.

Experiments on the flamingo (Part F) and the yoga-flow chain, each answering one question:

    H1  open loop: which motions run away without feedback, and how fast?          eigenvalues of A
    H2  which of them can the motors influence at all?                             controllability (PBH test)
    H3  closed loop: how does LQR move them, and how fast does it settle?          eigenvalues of A - B K
    H4  what does the gain matrix do physically?                                    K times a CoM lean
    H5  what does P (the cost-to-go) know about pushes?                            V(s) = s^T P s vs Part F's pushes
    H6  how far is the linear model right?                                          linear vs MuJoCo from a lean
    H7  why the flow needs many gains                                               cross-stability and schedules

    .venv/bin/python classical/01_dynamics_lqr_mpc/lqr_analysis.py
"""

import matplotlib
import mujoco
import numpy as np
import scipy.linalg
from PIL import Image

matplotlib.use("Agg")   # files only, no window
import matplotlib.pyplot as plt

import one_leg_lqr as F

results_dir = F.results_dir


def dof_names(model):
    """Names for the 35 velocity directions: 6 base ones, then one per joint."""
    base = ["base x", "base y", "base z", "base roll", "base pitch", "base yaw"]
    return base + [model.joint(i).name.removesuffix("_joint") for i in range(1, model.njnt)]


def top_entries(vec, names, k=4):
    """The k largest entries of a (possibly complex) vector, as 'name +0.62', scaled so the largest is +1.

    A complex eigenvector is only defined up to a complex factor: rotate it so its largest entry is real first.
    """
    big = np.argmax(np.abs(vec))
    v = np.real(vec * np.conj(vec[big]) / np.abs(vec[big])) / np.abs(vec[big])
    order = np.argsort(-np.abs(v))[:k]
    return ", ".join(f"{names[i]} {v[i]:+.2f}" for i in order)


def continuous(eig, dt):
    """A discrete eigenvalue lambda per step dt as a continuous rate s = ln(lambda) / dt (1/s).

    |lambda| > 1  <=>  Re(s) > 0 (grows);  |lambda| < 1  <=>  Re(s) < 0 (decays). For a complex pair the angle
    gives an oscillation: frequency |Im(s)| / 2 pi Hz, damping ratio zeta = -Re(s) / |s|.
    """
    return np.log(eig.astype(complex)) / dt


def setup():
    model, data, joints = F.build_torque_g1()
    ctx = {}
    poses, foot, R_foot = F.make_poses(model, data, joints)
    ctx.update(poses=poses, F=foot, R_foot=R_foot)
    q0, ctrl0, _ = F.equilibrium(model, data, poses["flamingo"])
    A, B = F.linearize(model, data, q0, ctrl0)
    Q, R = F.cost_matrices(model, data, joints, q0)
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    ctx.update(q0=q0, ctrl0=ctrl0, A=A, B=B, Q=Q, R=R, P=P, K=K)
    return model, data, joints, ctx


# ---------------------------------------------------------------------------------------------------------

def part_h1(model, data, joints, ctx):
    print("\n=== H1: open loop (holding torques only): which motions run away? eigenvalues of A ===")
    A, dt, nv, names = ctx["A"], model.opt.timestep, model.nv, dof_names(model)
    eig, vecs = np.linalg.eig(A)
    mag = np.abs(eig)
    growing, marginal = np.nonzero(mag > 1.001)[0], np.nonzero(np.abs(mag - 1) <= 1e-3)[0]
    print(f"70 eigenvalues: {len(growing)} growing (|lambda| > 1.001), {len(marginal)} at ~1 (|lambda - 1| < 0.001), "
          f"{70 - len(growing) - len(marginal)} decaying")
    print("  fastest growing modes: (doubling time; the position part of the mode, largest entries)")
    for i in growing[np.argsort(-mag[growing])][:4]:
        s = continuous(eig[i:i + 1], dt)[0]
        print(f"    |lambda| {mag[i]:.4f} per {1000 * dt:.0f} ms = rate {s.real:5.1f} /s, doubles every "
              f"{1000 * np.log(2) / s.real:4.0f} ms: {top_entries(vecs[:nv, i], names)}")
    print("  modes at ~1 (neither grow nor decay):")
    for i in marginal[:6]:
        print(f"    |lambda| {mag[i]:.6f}: {top_entries(vecs[:nv, i], names, 3)}")
    ctx.update(eig=eig, vecs=vecs, growing=growing, marginal=marginal)
    assert len(growing) > 0


def part_h2(model, data, joints, ctx):
    print("\n=== H2: can the motors move each mode? controllability (PBH rank test) ===")
    # PBH: the mode with eigenvalue lambda is controllable iff [A - lambda I, B] has full rank n. Its smallest
    # singular value says how close it is to losing rank: ~0 means some combination of the state is untouched
    # both by the dynamics at that rate and by every motor. Torques scaled by their limits (B diag(limit)), so the
    # number means "what full torque can do", and the A part divided by dt, so both halves have comparable units.
    A, B, dt, nv = ctx["A"], ctx["B"], model.opt.timestep, model.nv
    Bs = B @ np.diag(model.actuator_ctrlrange[:, 1])
    eig = ctx["eig"]

    def pbh(lam):
        M = np.hstack([(A - lam * np.eye(2 * nv)) / dt, Bs])
        return np.linalg.svd(M, compute_uv=False)[-1]

    def smallest(M):
        return np.linalg.svd(M, compute_uv=False)[-1]

    print(f"  {'mode':<26} | smallest singular value of (A - lambda I)/dt | ... of [(A - lambda I)/dt, B]")
    scores = []
    for i in ctx["growing"][np.argsort(-np.abs(eig[ctx["growing"]]))]:
        lam = eig[i]
        without, with_b = smallest((A - lam * np.eye(2 * nv)) / dt), pbh(lam)
        scores.append(with_b)
        if len(scores) <= 4:
            print(f"  growing, |lambda| {abs(lam):.4f}  | {without:44.1e} | {with_b:.3f}")
    print(f"  ... all {len(scores)} growing modes: with B the smallest is {min(scores):.3f} > 0: every runaway motion is "
          f"controllable")
    # lambda = 1 exactly: the directions where nothing changes from step to step
    U, sv, _ = np.linalg.svd((A - np.eye(2 * nv)).T)
    neutral = U[:, sv < 1e-6]
    names = dof_names(model)
    print(f"  lambda = 1 exactly: {neutral.shape[1]} neutral directions ("
          + "; ".join(top_entries(neutral[:nv, k], names, 2) for k in range(neutral.shape[1]))
          + f"); with B the smallest singular value is {pbh(1.0):.4f}")
    print("  -> the neutral directions are where the robot stands on the floor (x, y): nothing pulls it back, and the")
    print("     motors reach it only weakly (through the floor's soft give).")
    print("     They don't grow, and the cost doesn't ask for them, so LQR leaves them alone (they stay at 1 in H3).")
    print("     Weak + unasked-for + exactly 1 is also what made scipy's Riccati solver fail on tip-toe (Part G).")
    assert min(scores) > 1e-3


def part_h3(model, data, joints, ctx):
    print("\n=== H3: closed loop: eigenvalues of A - B K ===")
    A, B, K, dt, nv, names = ctx["A"], ctx["B"], ctx["K"], model.opt.timestep, model.nv, dof_names(model)
    eig_cl, vecs_cl = np.linalg.eig(A - B @ K)
    mag = np.abs(eig_cl)
    s = continuous(eig_cl, dt)
    stay = np.nonzero(mag > 0.9999)[0]
    print(f"  growing modes now: {int(np.sum(mag > 1.001))}; {len(stay)} still at ~1: " +
          "; ".join(top_entries(vecs_cl[:nv, i], names, 2) for i in stay[:4]) +
          " (where it stands and which way it faces: the cost ignores them, and the motors barely reach them)")
    real_modes = [i for i in np.argsort(-mag) if mag[i] < 0.9999][:5]
    print("  slowest decaying modes (time constant tau = -dt / ln|lambda|; complex pairs oscillate):")
    for i in real_modes:
        tau = -dt / np.log(mag[i])
        osc = f", oscillates at {abs(s[i].imag) / (2 * np.pi):.1f} Hz, damping ratio {-s[i].real / abs(s[i]):.2f}" \
            if abs(s[i].imag) > 1e-6 else ""
        print(f"    |lambda| {mag[i]:.5f}: tau {tau:6.2f} s{osc}: {top_entries(vecs_cl[:nv, i], names, 3)}")
    fast = np.sort(-np.real(s))[::-1][:3]
    print(f"  fastest decay rates: {np.round(fast, 0)} /s (gone in ~{1000 / fast[0]:.0f} ms)")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (ev, title) in zip(axes, ((ctx["eig"], "open loop: A"), (eig_cl, "closed loop: A - B K"))):
        circle = np.exp(1j * np.linspace(0, 2 * np.pi, 400))
        ax.plot(circle.real, circle.imag, color="gray", lw=0.8)
        ax.scatter(ev.real, ev.imag, s=12, c=np.where(np.abs(ev) > 1.001, "tab:red", "tab:blue"))
        ax.set_aspect("equal")
        ax.set_xlim(0.0, 1.15)
        ax.set_ylim(-0.45, 0.45)
        ax.set_title(f"{title}: {int(np.sum(np.abs(ev) > 1.001))} outside the unit circle")
        ax.set_xlabel("Re")
        ax.set_ylabel("Im")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / "h3_eigenvalues.png", dpi=120)
    plt.close(fig)
    print(f"saved {results_dir / 'h3_eigenvalues.png'}")
    ctx.update(eig_cl=eig_cl, vecs_cl=vecs_cl)
    assert mag.max() < 1.0 + 1e-6


def lean_direction(model, data, q0, direction):
    """The smallest motion dq (35) that moves the CoM 1 cm in `direction` with the standing foot held still.

    Stack the constraints: J_foot (6 rows: position and rotation) dq = 0, J_com dq = 1 cm * direction. Many dq
    solve it; the pseudo-inverse picks the smallest. (Without the foot rows, the smallest motion slides or tilts
    the foot, which the floor won't allow.)
    """
    data.qpos[:] = q0
    mujoco.mj_forward(model, data)
    J_com, _ = F.jac(model, data, "com")
    jp, jr = F.jac(model, data, "site", F.STANCE)
    J = np.vstack([jp, jr, J_com])
    return np.linalg.pinv(J) @ np.concatenate([np.zeros(6), 0.01 * np.asarray(direction, float)])


def part_h4(model, data, joints, ctx):
    print("\n=== H4: what K does: the torques for a 1 cm CoM lean (u = u0 - K s, s = [dq, 0]) ===")
    K, nv = ctx["K"], model.nv
    limits = model.actuator_ctrlrange[:, 1]
    act_names = [model.actuator(i).name.removesuffix("_joint") for i in range(model.nu)]
    for label, direction in (("sideways (to the left, +y)", (0, 1, 0)), ("forward (+x)", (1, 0, 0))):
        dq = lean_direction(model, data, ctx["q0"], direction)
        du = -K[:, :nv] @ dq
        order = np.argsort(-np.abs(du))[:6]
        print(f"  CoM leaning 1 cm {label}: " + ", ".join(
            f"{act_names[i]} {du[i]:+.1f} N m ({abs(du[i]) / limits[i]:.0%})" for i in order))
    print("  (the standing leg's ankle and hip push the CoM back, the waist and arms help: LQR finds an ankle + hip "
          "strategy by itself)")


def part_h5(model, data, joints, ctx):
    print("\n=== H5: the cost-to-go V(s) = s^T P s vs measured pushes ===")
    # Right after a push the state is s (mostly velocities). V(s) = s^T P s is what LQR expects to pay to recover.
    # V grows like (push)^2, so the push LQR can afford in a direction should scale like 1/sqrt(V per unit push).
    q0, ctrl0, K, P, nv = ctx["q0"], ctx["ctrl0"], ctx["K"], ctx["P"], model.nv
    torso = model.body("torso_link").id
    measured = {}
    rows = []
    for name, direction in F.DIRECTIONS.items():
        mujoco.mj_resetData(model, data)
        data.qpos[:] = q0
        mujoco.mj_forward(model, data)
        while data.time < 0.1 - 1e-9:                        # a 10 N push for 0.1 s, with LQR on
            F.lqr_control(model, data, q0, ctrl0, K)
            data.xfrc_applied[:] = 0.0
            data.xfrc_applied[torso, :3] = 10.0 * np.asarray(direction, float)
            mujoco.mj_step(model, data)
        s = np.zeros(2 * nv)
        mujoco.mj_differentiatePos(model, s[:nv], 1.0, q0, data.qpos)
        s[nv:] = data.qvel
        V = s @ P @ s
        measured[name] = F.max_push(model, data, q0, ctrl0, K, direction)
        rows.append((name, V, 10.0 / np.sqrt(V)))
    scale = np.mean([measured[n] for n, *_ in rows]) / np.mean([r[2] for r in rows])
    print(f"  {'direction':<9} | V after 10 N x 0.1 s | predicted max push (scaled) | measured (Part F)")
    for name, V, pred in rows:
        print(f"  {name:<9} | {V:19.4f} | {scale * pred:27.0f} | {measured[name]:7.0f}")
    corr = np.corrcoef([r[2] for r in rows], [measured[r[0]] for r in rows])[0, 1]
    print(f"  correlation between predicted and measured: {corr:+.2f}")
    print("  -> V does NOT predict which pushes are dangerous. It is the cost of recovering under the LINEAR model,"
          " with no edges anywhere; falling happens at nonlinear limits the linear model doesn't have (the foot"
          " rolling onto an edge, torque limits). A critic trained on real outcomes (with falls) would learn those.")


def part_h6(model, data, joints, ctx):
    print("\n=== H6: how far the linear model is right: lean the CoM sideways and let go ===")
    A, B, K, q0, ctrl0, nv = ctx["A"], ctx["B"], ctx["K"], ctx["q0"], ctx["ctrl0"], model.nv
    A_cl = A - B @ K
    d1 = lean_direction(model, data, q0, (0, 1, 0))         # 1 cm CoM lean
    data.qpos[:] = q0
    mujoco.mj_forward(model, data)
    J_bal = F.jac(model, data, "com")[0] - F.jac(model, data, "site", F.STANCE)[0]
    fig, ax = plt.subplots(figsize=(8, 4))
    for cm in (0.5, 1.0, 2.0, 3.0, 4.0):
        s = np.concatenate([cm * d1, np.zeros(nv)])
        steps = round(1.5 / model.opt.timestep)
        lin = [J_bal @ s[:nv]]
        x = s.copy()
        for _ in range(steps):
            x = A_cl @ x
            lin.append(J_bal @ x[:nv])
        lin = np.array(lin)[:, 1]
        mujoco.mj_resetData(model, data)
        q = q0.copy()
        mujoco.mj_integratePos(model, q, s[:nv], 1.0)
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        foot0, sim, fell = data.site(F.STANCE).xpos.copy(), [], False
        for _ in range(steps + 1):
            sim.append((data.subtree_com[1] - data.site(F.STANCE).xpos)[1] - (data.subtree_com[1] * 0)[1])
            F.lqr_control(model, data, q0, ctrl0, K)
            mujoco.mj_step(model, data)
            fell = fell or F.fallen(model, data, q0)
        sim = np.array(sim) - sim[0] + lin[0]                # same starting offset
        t = np.arange(steps + 1) * model.opt.timestep
        err = np.abs(sim - lin).max() / abs(lin[0])
        print(f"  lean {cm:3.1f} cm: largest gap MuJoCo vs linear = {err:5.0%} of the lean; falls: {fell}")
        ax.plot(t, 100 * sim, label=f"{cm} cm: MuJoCo")
        ax.plot(t, 100 * lin, "--", color=ax.lines[-1].get_color(), alpha=0.6)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("CoM - foot, sideways (cm)")
    ax.set_title("solid: MuJoCo with LQR; dashed: the linear model's prediction")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / "h6_linear_vs_mujoco.png", dpi=120)
    plt.close(fig)
    print(f"saved {results_dir / 'h6_linear_vs_mujoco.png'} (the foot is ~3 cm half-width sideways)")


def part_h7(model, data, joints, ctx):
    print("\n=== H7: gain scheduling along the yoga flow ===")
    chain = F.build_flow_chain(model, data, joints, ctx)
    n = len(chain["q"])
    open_rho = [np.abs(np.linalg.eigvals(A)).max() for A in chain["A"]]
    print(f"  {n} poses: open-loop largest |lambda| from {min(open_rho):.3f} to {max(open_rho):.3f} "
          f"(every pose falls without feedback)")
    # Cross-stability: does pose i's gain stabilize pose j's linear model? Every pose keeps its neutral modes at
    # exactly 1 (sliding on the floor, H2/H3), so "largest |eigenvalue| < 1" never holds; stable = nothing GROWS:
    # no eigenvalue of A_j - B_j K_i above 1 (+ 0.0005 for rounding).
    idx = list(range(0, n, 3))
    rho = np.array([[np.abs(np.linalg.eigvals(chain["A"][j] - chain["B"][j] @ chain["K"][i])).max() for j in idx]
                    for i in idx])
    stable = rho < 1.0005
    print(f"  pose i's gain on pose j's linear model (every 3rd pose): stable in {stable.mean():.0%} of the pairs; "
          f"own gains (diagonal) all stable: {bool(np.all(np.diag(stable)))}")
    for i in (0, len(idx) // 3, 2 * len(idx) // 3):
        ok = [idx[j] for j in range(len(idx)) if stable[i, j]]
        print(f"    gain of pose {idx[i]:2d} stabilizes {len(ok)} of {len(idx)} poses (unstable for: "
              f"{[idx[j] for j in range(len(idx)) if not stable[i, j]] or 'none'})")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(rho - 1, cmap="RdBu_r", vmin=-0.01, vmax=0.01, origin="lower", extent=[0, n - 1, 0, n - 1])
    ax.set_xlabel("pose j (linear model A_j, B_j)")
    ax.set_ylabel("pose i (gain K_i)")
    ax.set_title("largest |eig(A_j - B_j K_i)| - 1: white/blue = nothing grows, red = grows")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(results_dir / "h7_cross_stability.png", dpi=120)
    plt.close(fig)
    print(f"saved {results_dir / 'h7_cross_stability.png'}")

    n_wp = chain["n_wp"]
    yoga_idx = [i * n_wp for i in range(len(chain["sequence"]))]          # the 4 yoga poses in the chain
    schedules = {
        "blended gains (Part F)": None,
        "nearest pose's gain": lambda s: chain["K"][int(round(s))],
        "only the yoga poses' gains": lambda s: chain["K"][min(yoga_idx, key=lambda k: abs(k - s))],
        "flamingo's gain for everything": lambda s: chain["K"][0],
    }
    print(f"  {'schedule':<32} | falls | largest joint error (rad)")
    for name, gains in schedules.items():
        fell, worst, _, _ = F.run_flow(model, data, chain, gains=gains)
        print(f"  {name:<32} | {str(fell):<5} | {worst:.3f}")


def mode_video(model, data, ctx):
    """Animate mode shapes: the robot moved back and forth along a mode's position part (kinematic, not physics)."""
    nv, q0 = model.nv, ctx["q0"]
    picks = [("fastest open-loop runaway", ctx["vecs"][:, ctx["growing"][np.argmax(np.abs(ctx["eig"][ctx["growing"]]))]]),
             ("slowest decaying closed-loop mode",
              ctx["vecs_cl"][:, max((i for i in range(len(ctx["eig_cl"])) if abs(ctx["eig_cl"][i]) < 0.9999),
                                    key=lambda i: abs(ctx["eig_cl"][i]))]),
             ("a mode at ~1 (open loop)", ctx["vecs"][:, ctx["marginal"][0]])]
    F.brighten(model)
    cam = F.make_camera(lookat=[0, 0, 0.55], distance=2.4, azimuth=140, elevation=-10)
    panels = []
    with mujoco.Renderer(model, height=360, width=360) as renderer:
        for name, vec in picks:
            shape = np.real(vec[:nv])
            shape = 0.25 * shape / np.abs(shape[6:]).max()       # largest joint swings +-0.25 rad
            frames = []
            for k in range(60):
                q = q0.copy()
                mujoco.mj_integratePos(model, q, shape, np.sin(2 * np.pi * k / 60))
                data.qpos[:] = q
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=cam)
                frames.append(F.label(Image.fromarray(renderer.render()), name))
            panels.append(frames)
    F.save_gif(F.side_by_side(panels), results_dir / "h_modes.gif", fps=20)
    print(f"\nsaved {results_dir / 'h_modes.gif'} (mode shapes, exaggerated)")


def main():
    model, data, joints, ctx = setup()
    for part in (part_h1, part_h2, part_h3, part_h4, part_h5, part_h6, part_h7):
        part(model, data, joints, ctx)
    mode_video(model, data, ctx)
    print("\n=== Part H complete ===")


if __name__ == "__main__":
    main()
