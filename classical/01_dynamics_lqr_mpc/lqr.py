"""Module 1, Part B: LQR balance of the cart-pole.

Fill in the functions marked TODO, in order. Each run executes the parts in order and stops at the first
TODO that is not implemented yet, so the checks of every finished part run again each time:

    .venv/bin/python classical/01_dynamics_lqr_mpc/lqr.py

Conventions are in cartpole.py: s = [x, theta, x_dot, theta_dot], theta = 0 upright, u = cart force (N).
Allowed: numpy, scipy.linalg.expm. Not allowed in your code: scipy's Riccati / LQR solvers (used only to check).
"""

import sys
from pathlib import Path

import matplotlib
import mujoco
import numpy as np
import scipy.linalg

matplotlib.use("Agg")   # files only, no window
import matplotlib.pyplot as plt

script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir.parents[1] / "exercises"))   # render_utils
from render_utils import record_frames, record_gif, save_gif, side_by_side

from cartpole import DT, G, L, M_CART, M_POLE, get_state, make_model, set_state, simulate

results_dir = script_dir / "results"
results_dir.mkdir(exist_ok=True)

# Cost weights for Parts B4-B6: state [x, theta, x_dot, theta_dot], force u
Q = np.diag([1.0, 10.0, 0.1, 0.1])
R = np.array([[0.1]])


# ---------------------------------------------------------------------------------------------------------
# TODO functions
# ---------------------------------------------------------------------------------------------------------

def linearize_continuous() -> tuple[np.ndarray, np.ndarray]:
    """B1. A_c (4x4), B_c (4x1) of s_dot = f(s, u) linearized at s = 0, u = 0 (upright, at rest).

    From your own derivation, written with M_CART, M_POLE, L and G (from cartpole.py), not with numbers.
    """
    # Equations of motion (theta from upright, point mass at the tip), q = [x, theta]:
    #   (m_c + m_p) x_dd + m_p L cos(th) th_dd - m_p L sin(th) th_d^2 = u
    #   m_p L cos(th) x_dd + m_p L^2 th_dd - m_p g L sin(th)          = 0
    # Linearize at th = 0, th_d = 0: cos -> 1, sin -> th, th_d^2 -> 0 (second order). That leaves
    #   M0 q_dd = [u, m_p g L th],   M0 = [[m_c + m_p, m_p L], [m_p L, m_p L^2]]
    # and q_dd = M0^-1 [u, m_p g L th], with M0^-1 = 1/D [[m_p L^2, -m_p L], [-m_p L, m_c + m_p]].
    D = M_CART * M_POLE * L**2      # det(M0) = (m_c + m_p) m_p L^2 - (m_p L)^2

    # Top rows: d/dt [x, th] = [x_d, th_d]. Bottom rows: q_dd. float, or the assignments below get truncated.
    A_c = np.array([[0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]], dtype=float)
    # Gravity column (th). Times m_p g L from the right-hand side, via the second column of M0^-1:
    A_c[2, 1] = -M_POLE * L * (M_POLE * G * L) / D          # = -m_p g / m_c: the falling pole pulls the cart back
    A_c[3, 1] = (M_CART + M_POLE) * (M_POLE * G * L) / D    # = (m_c + m_p) g / (m_c L) > 0: upright is unstable
    # No velocity columns: velocities only enter through th_d^2, which vanishes to first order at rest.

    # Input column: u enters the first equation only, so it acts through the first column of M0^-1.
    B_c = np.zeros((4, 1))
    B_c[2, 0] = M_POLE * L**2 / D       # = 1 / m_c
    B_c[3, 0] = -M_POLE * L / D         # = -1 / (m_c L): pushing the cart right tips the pole left
    return A_c, B_c


def discretize(A_c: np.ndarray, B_c: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """B2. Zero-order-hold discretization: s[k+1] = A s[k] + B u[k] with u held constant over dt."""
    # Exact solution over one step with u constant:
    #   s(t + dt) = e^(A_c dt) s(t) + (integral_0^dt e^(A_c tau) dtau) B_c u
    # Trick: treat u as a state that never changes (u_d = 0). The augmented system z = [s, u] has
    #   z_d = [[A_c, B_c], [0, 0]] z,   so   z(t + dt) = expm([[A_c, B_c], [0, 0]] dt) z(t),
    # and the top-right block of that exponential is exactly the integral times B_c.
    n, m = B_c.shape
    aug = np.zeros((n + m, n + m))
    aug[:n, :n] = A_c
    aug[:n, n:] = B_c
    E = scipy.linalg.expm(aug * dt)
    A = E[:n, :n]       # e^(A_c dt): each eigenvalue lambda of A_c becomes e^(lambda dt)
    B = E[:n, n:]
    # (Forward Euler, A = I + A_c dt and B = B_c dt, is only first-order accurate and fails the 1e-5 check.)
    return A, B


def controllability_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """B3. [B, AB, A^2 B, ..., A^(n-1) B]."""
    # Column block k is A^k B: the effect on the state of an input applied k steps ago.
    # Each block is A times the previous one (A**k would be element-wise power in numpy).
    blocks = [B]
    for _ in range(A.shape[0] - 1):
        blocks.append(A @ blocks[-1])
    return np.hstack(blocks)


def dlqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray,
         tol: float = 1e-10, max_iter: int = 100_000) -> tuple[np.ndarray, np.ndarray]:
    """B4. Infinite-horizon discrete LQR by iterating the Riccati equation backwards until P stops changing.

    Returns (K, P) with u = -K s and cost-to-go V(s) = s^T P s.
    """
    # Value iteration with a quadratic value function V_k(s) = s^T P_k s. One Bellman backup:
    #   V_k(s) = min_u  s^T Q s + u^T R u + V_{k+1}(A s + B u)
    # Setting the gradient in u to zero gives the best action u = -K s with
    #   K = (R + B^T P B)^-1 B^T P A
    # and substituting it back gives the next P (Joseph form: a sum of symmetric PSD terms, so it stays
    # symmetric and PSD despite rounding):
    #   P <- Q + K^T R K + (A - B K)^T P (A - B K)
    # After k iterations P is the cost-to-go of a k-step problem; as k grows it converges to the
    # infinite-horizon P (the solution of the discrete algebraic Riccati equation).
    P = Q.copy()        # = cost-to-go with zero steps left (any PSD start converges)
    for _ in range(max_iter):
        K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)      # solve, not inv: better conditioned
        A_cl = A - B @ K                                        # closed-loop dynamics under this K
        P_next = Q + K.T @ R @ K + A_cl.T @ P @ A_cl
        P_next = 0.5 * (P_next + P_next.T)                      # remove rounding asymmetry
        converged = np.abs(P_next - P).max() <= tol * np.abs(P_next).max()   # relative: P's entries are large
        P = P_next
        if converged:
            break
    else:
        raise RuntimeError(f"Riccati iteration did not converge in {max_iter} iterations")
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)          # gain from the converged P
    return K, P


def make_lqr_controller(K: np.ndarray):
    """B5. Return controller(model, data) that reads the state from data and writes u = -K s to data.ctrl."""
    # Full state feedback: the same linear law at every step. No saturation here on purpose: with a force
    # limit, MuJoCo's ctrlrange clamps the command (Part B7), and LQR never knows about it.
    def controller(model, data):
        s = get_state(data)                 # [x, theta, x_dot, theta_dot]
        data.ctrl[0] = -(K @ s)[0]          # K is (1, 4), so K @ s has one element
    return controller


def region_of_attraction(K: np.ndarray, thetas: np.ndarray, theta_dots: np.ndarray,
                         force_limit: float | None, duration: float = 10.0) -> np.ndarray:
    """B7. Boolean grid (len(thetas), len(theta_dots)): True where LQR brings the pole back upright.

    Start each run at s0 = [0, theta, 0, theta_dot] on a model built with make_model(force_limit). Decide
    yourself what "back upright" means at the end of the run (state small, did not fall).
    """
    # "Recovered" = did not fall AND the whole state is near the origin at the end. Checked on this grid:
    # within 10 s every run either falls or ends below 1e-3, so any threshold from 1e-3 to 1 gives the same
    # map. A 5 s window does NOT: large recoveries are still bringing the cart back, and the map shrinks.
    SETTLED = 1e-2
    model, data = make_model(force_limit)       # once per limit; simulate() resets the state for each run
    controller = make_lqr_controller(K)
    grid = np.zeros((len(thetas), len(theta_dots)), dtype=bool)
    for i, theta in enumerate(thetas):
        for j, theta_dot in enumerate(theta_dots):
            s0 = np.array([0.0, theta, 0.0, theta_dot])
            _, S, _, fell = simulate(model, data, s0, controller, duration)
            grid[i, j] = (not fell) and np.abs(S[-1]).max() < SETTLED
    return grid


# ---------------------------------------------------------------------------------------------------------
# Checks (given)
# ---------------------------------------------------------------------------------------------------------

def mujoco_continuous_jacobians(model, data, eps: float = 1e-6):
    """A_c, B_c by central differences of MuJoCo's accelerations around s = 0, u = 0."""
    def f(s, u):
        set_state(model, data, s, u)
        return np.concatenate([data.qvel, data.qacc])

    A_c, B_c = np.zeros((4, 4)), np.zeros((4, 1))
    for i in range(4):
        ds = np.zeros(4)
        ds[i] = eps
        A_c[:, i] = (f(ds, 0.0) - f(-ds, 0.0)) / (2 * eps)
    B_c[:, 0] = (f(np.zeros(4), eps) - f(np.zeros(4), -eps)) / (2 * eps)
    return A_c, B_c


def mujoco_discrete_jacobians(model, data, eps: float = 1e-6):
    """A, B of one MuJoCo step (RK4) around s = 0, u = 0, by central differences.

    (mujoco.mjd_transitionFD does this for the other integrators, but not RK4.)
    """
    def step(s, u):
        set_state(model, data, s, u)
        mujoco.mj_step(model, data)
        return get_state(data)

    A, B = np.zeros((4, 4)), np.zeros((4, 1))
    for i in range(4):
        ds = np.zeros(4)
        ds[i] = eps
        A[:, i] = (step(ds, 0.0) - step(-ds, 0.0)) / (2 * eps)
    B[:, 0] = (step(np.zeros(4), eps) - step(np.zeros(4), -eps)) / (2 * eps)
    return A, B


def part_b1(ctx):
    print("\n=== B1: continuous linearization ===")
    model, data = make_model()
    A_c, B_c = linearize_continuous()
    A_mj, B_mj = mujoco_continuous_jacobians(model, data)
    np.set_printoptions(precision=4, suppress=True)
    print(f"A_c (yours)\n{A_c}\nB_c (yours)\n{B_c.ravel()}")
    print(f"max |A_c - MuJoCo| = {np.abs(A_c - A_mj).max():.2e}, max |B_c - MuJoCo| = {np.abs(B_c - B_mj).max():.2e}")
    assert A_c.shape == (4, 4) and B_c.shape == (4, 1), "shapes: A_c (4, 4), B_c (4, 1)"
    assert np.allclose(A_c, A_mj, atol=1e-4), "A_c differs from MuJoCo: check signs and the theta convention"
    assert np.allclose(B_c, B_mj, atol=1e-4), "B_c differs from MuJoCo"
    ctx.update(A_c=A_c, B_c=B_c)


def part_b2(ctx):
    print("\n=== B2: discretization ===")
    model, data = make_model()
    A, B = discretize(ctx["A_c"], ctx["B_c"], DT)

    A_mj, B_mj = mujoco_discrete_jacobians(model, data)

    print(f"max |A - MuJoCo| = {np.abs(A - A_mj).max():.2e}, max |B - MuJoCo| = {np.abs(B - B_mj).max():.2e}")
    assert A.shape == (4, 4) and B.shape == (4, 1)
    # Exact ZOH vs. one RK4 step: they differ only at O(dt^5)
    assert np.allclose(A, A_mj, atol=1e-5) and np.allclose(B, B_mj, atol=1e-5), "discrete A/B differ from MuJoCo"
    eig = np.abs(np.linalg.eigvals(A))
    print(f"|eigenvalues of A| = {np.sort(eig)}  (one > 1: the upright pole is unstable)")
    ctx.update(A=A, B=B)


def part_b3(ctx):
    print("\n=== B3: controllability ===")
    A, B = ctx["A"], ctx["B"]
    C = controllability_matrix(A, B)
    rank = np.linalg.matrix_rank(C)
    print(f"controllability matrix shape {C.shape}, rank {rank}")
    assert C.shape == (4, 4)
    assert np.allclose(C[:, :1], B) and np.allclose(C[:, 1:2], A @ B), "columns should be B, AB, ..."
    assert rank == 4, "cart-pole should be controllable"


def part_b4(ctx):
    print("\n=== B4: discrete LQR (Riccati iteration) ===")
    A, B = ctx["A"], ctx["B"]
    K, P = dlqr(A, B, Q, R)
    P_ref = scipy.linalg.solve_discrete_are(A, B, Q, R)
    K_ref = np.linalg.solve(R + B.T @ P_ref @ B, B.T @ P_ref @ A)
    closed = np.abs(np.linalg.eigvals(A - B @ K))
    print(f"K = {K.ravel()}")
    print(f"max |P - scipy| / max |P| = {np.abs(P - P_ref).max() / np.abs(P_ref).max():.2e}")
    print(f"|eigenvalues of A - BK| = {np.sort(closed)}")
    assert K.shape == (1, 4) and P.shape == (4, 4)
    assert np.allclose(P, P.T), "P should be symmetric"
    assert np.allclose(P, P_ref, rtol=1e-6), "P differs from scipy.linalg.solve_discrete_are"
    assert np.allclose(K, K_ref, rtol=1e-6), "K differs from the gain implied by scipy's P"
    assert closed.max() < 1, "closed loop should be stable"
    ctx.update(K=K, P=P)


def part_b5(ctx):
    print("\n=== B5: closed loop in MuJoCo ===")
    K, P = ctx["K"], ctx["P"]
    model, data = make_model()
    s0 = np.array([0.0, 0.2, 0.0, 0.0])
    t, S, U, fell = simulate(model, data, s0, make_lqr_controller(K), duration=10.0)
    print(f"final state = {S[-1]}, peak |u| = {np.abs(U).max():.2f} N")
    assert not fell, "pole fell"
    assert np.abs(S[-1]).max() < 1e-3, "did not settle to the origin within 10 s"
    assert np.allclose(U, -(S[:-1] @ K.T).ravel()), "applied force should be -K s at every step"

    # The cost-to-go: s0^T P s0 predicts the total cost of the whole run (exact for the linear model)
    run_cost = np.sum(np.einsum("ni,ij,nj->n", S[:-1], Q, S[:-1]) + R[0, 0] * U**2)
    predicted = s0 @ P @ s0
    print(f"V(s0) = s0^T P s0 = {predicted:.4f}, summed cost of the MuJoCo run = {run_cost:.4f}")
    # Not equal: the real pole is nonlinear (sin, cos) at 0.2 rad
    assert abs(run_cost - predicted) / predicted < 0.05, "run cost far from V(s0)"

    _plot_run(t, S, U, results_dir / "b5_balance.png", "LQR from theta0 = 0.2 rad")
    set_state(model, data, s0)
    record_gif(model, data, results_dir / "b5_balance.gif", duration=4.0, camera=model_camera(model, "fixed"),
               control=make_lqr_controller(K), capture_fps=30)
    print(f"saved {results_dir / 'b5_balance.png'} and .gif")


def part_b6(ctx):
    print("\n=== B6: the Q/R trade-off ===")
    A, B = ctx["A"], ctx["B"]
    model, data = make_model()
    s0 = np.array([0.0, 0.2, 0.0, 0.0])
    peaks, travels = [], []
    print(f"{'R':>6} | {'peak |u| (N)':>12} | {'theta settle (s)':>16} | {'max |x| (m)':>11}")
    for r in (0.01, 0.1, 1.0, 10.0):
        K, _ = dlqr(A, B, Q, np.array([[r]]))
        t, S, U, fell = simulate(model, data, s0, make_lqr_controller(K), duration=10.0)
        assert not fell
        outside = np.nonzero(np.abs(S[1:, 1]) > 0.01)[0]   # theta outside ±0.01 rad
        settle = t[outside[-1]] if len(outside) else 0.0
        peaks.append(np.abs(U).max())
        travels.append(np.abs(S[:, 0]).max())
        print(f"{r:>6} | {peaks[-1]:>12.2f} | {settle:>16.2f} | {travels[-1]:>11.3f}")
    # Questions for notes.md: why does a gentler force make the cart travel further? And why is the theta
    # settling time not monotone in R? (Plot theta and x for R = 0.1 and 1.0.)
    assert np.all(np.diff(peaks) < 0), "heavier R should mean gentler force"
    assert np.all(np.diff(travels) > 0), "heavier R should mean more cart travel"


def part_b7(ctx):
    print("\n=== B7: region of attraction, with and without a force limit ===")
    K = ctx["K"]
    thetas = np.linspace(-1.4, 1.4, 29)
    theta_dots = np.linspace(-4.0, 4.0, 33)
    limits = (None, 5.0, 2.0)
    grids = {lim: region_of_attraction(K, thetas, theta_dots, lim) for lim in limits}

    fig, axes = plt.subplots(1, len(limits), figsize=(13, 4), sharey=True)
    for ax, lim in zip(axes, limits):
        ax.pcolormesh(theta_dots, thetas, grids[lim], cmap="RdYlGn", vmin=0, vmax=1, shading="nearest")
        ax.set_title("no force limit" if lim is None else f"|u| <= {lim:g} N")
        ax.set_xlabel("theta_dot0 (rad/s)")
    axes[0].set_ylabel("theta0 (rad)")
    fig.tight_layout()
    fig.savefig(results_dir / "b7_region_of_attraction.png", dpi=120)

    counts = {lim: int(g.sum()) for lim, g in grids.items()}
    print("recovered starts: " + ", ".join(f"{'none' if k is None else f'{k:g} N'}: {v}" for k, v in counts.items()))
    print(f"saved {results_dir / 'b7_region_of_attraction.png'}")
    i0, j0 = np.argmin(np.abs(thetas)), np.argmin(np.abs(theta_dots))
    assert all(g[i0, j0] for g in grids.values()), "the upright state itself must count as recovered"
    assert counts[None] > counts[5.0] > counts[2.0], "a tighter force limit should shrink the region"

    # Video: one start outside both limited regions, same K, three force limits side by side
    s0 = np.array([0.0, 0.5, 0.0, 0.0])
    panels = []
    for lim in limits:
        model, data = make_model(lim)
        set_state(model, data, s0)
        # The camera rides with the cart: once the pole is lost, the clamped force keeps pushing and the cart
        # runs away down the rail
        cam = model_camera(model, "follow")
        name = "no limit" if lim is None else f"|u| <= {lim:g} N"
        panels.append(record_frames(model, data, 4.0, camera=cam, control=make_lqr_controller(K), capture_fps=30,
                                    label=lambda d, name=name: f"{name}   u = {d.actuator_force[0]:+6.1f} N"))
    save_gif(side_by_side(panels), results_dir / "b7_force_limits.gif", fps=30)
    print(f"saved {results_dir / 'b7_force_limits.gif'} (theta0 = 0.5 rad, same LQR gain)")

    # Question for notes.md: LQR doesn't know about the limit. What would a controller that does know do
    # differently near the edge of the region? (Parts C-D: MPC.)


def model_camera(model, name: str) -> mujoco.MjvCamera:
    """One of the cameras defined in cartpole.py: "fixed" (world) or "follow" (rides with the cart)."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = model.camera(name).id
    return cam


def _plot_run(t, S, U, path, title):
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(t, S[1:, 1]); axes[0].set_ylabel("theta (rad)")
    axes[1].plot(t, S[1:, 0]); axes[1].set_ylabel("x (m)")
    axes[2].plot(t, U); axes[2].set_ylabel("u (N)"); axes[2].set_xlabel("t (s)")
    axes[0].set_title(title)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


PARTS = [part_b1, part_b2, part_b3, part_b4, part_b5, part_b6, part_b7]


def main():
    ctx = {}
    for part in PARTS:
        try:
            part(ctx)
        except NotImplementedError as todo:
            print(f"\n>>> Next TODO: {todo}")
            return
    print("\n=== Part B complete ===")


if __name__ == "__main__":
    main()
