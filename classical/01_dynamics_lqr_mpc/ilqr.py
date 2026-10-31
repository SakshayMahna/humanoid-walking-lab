"""Module 1, Part C: swing-up of the cart-pole with iLQR.

Part B could only hold the pole near upright, because LQR's linear model is only true there. A swing-up goes
through every angle, so here the controller plans a whole trajectory instead:

    1. guess a force sequence u_0 .. u_{N-1} and simulate it              (a trajectory)
    2. backward pass: linearize the dynamics along that trajectory and solve a time-varying LQR problem
       backwards in time                                                (how to improve it, and feedback gains)
    3. forward pass: simulate the improved forces, keep them if the cost went down, repeat

Run the parts in order (each run re-checks the earlier ones):

    .venv/bin/python classical/01_dynamics_lqr_mpc/ilqr.py

Conventions as in cartpole.py: s = [x, theta, x_dot, theta_dot], theta = 0 upright, theta = pi hanging down.
"""

import sys
from pathlib import Path

import matplotlib
import mujoco
import numpy as np

matplotlib.use("Agg")   # files only, no window
import matplotlib.pyplot as plt

script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir.parents[1] / "exercises"))   # render_utils
from render_utils import record_frames, save_gif, side_by_side

from cartpole import DT, G, L, M_CART, M_POLE, get_state, make_model, set_state, step_jacobians
from lqr import Q as Q_LQR, R as R_LQR, discretize, dlqr, linearize_continuous, model_camera

results_dir = script_dir / "results"
results_dir.mkdir(exist_ok=True)

# The task: from hanging down, at rest, to upright and still at x = 0, in T seconds, with |u| <= U_MAX.
S_START = np.array([0.0, np.pi, 0.0, 0.0])
S_GOAL = np.zeros(4)
T = 3.0                     # s, planning horizon
N = round(T / DT)           # 300 control steps
U_MAX = 5.0                 # N, force limit: tight enough that the plan rides it (4 N cannot swing up in 3 s)

# Cost (C3). Running cost per step: dt * (1/2 r u^2 + 1/2 q_x x^2): use little force, keep the cart near the
# middle. Terminal cost: 1/2 (s - goal)^T Q_F (s - goal): be upright, centred and still at the end.
R_RUN = 0.1
Q_X_RUN = 1.0
Q_F = np.diag([50.0, 500.0, 50.0, 50.0])


# ---------------------------------------------------------------------------------------------------------
# C1: the model the planner uses
# ---------------------------------------------------------------------------------------------------------

def f_continuous(s: np.ndarray, u) -> np.ndarray:
    """s_dot for the full nonlinear cart-pole (no small-angle approximations).

    Equations of motion (the same ones linearized in Part B, before linearizing):
        (m_c + m_p) x_dd + m_p L cos(th) th_dd = u + m_p L sin(th) th_d^2
        m_p L cos(th) x_dd + m_p L^2 th_dd     = m_p g L sin(th)
    i.e. M(q) q_dd = rhs. M is 2x2, so it is inverted in closed form:
        [[a, b], [b, d]]^-1 = 1/(a d - b^2) [[d, -b], [-b, a]]
    which also lets s be a batch of states (n, 4) with u of shape (n,): one call evaluates many states
    (used to linearize a whole trajectory at once, Part D).
    """
    th, x_d, th_d = s[..., 1], s[..., 2], s[..., 3]
    c, sn = np.cos(th), np.sin(th)
    a = M_CART + M_POLE                                 # M = [[a, b], [b, d]]
    b = M_POLE * L * c
    d = M_POLE * L**2
    rhs_x = u + M_POLE * L * sn * th_d**2               # th_d^2 term: the pole's centripetal pull on the cart
    rhs_th = M_POLE * G * L * sn                        # gravity torque about the hinge
    det = a * d - b**2                                  # = m_c m_p L^2 + m_p^2 L^2 sin^2(th) > 0 always
    x_dd = (d * rhs_x - b * rhs_th) / det
    th_dd = (a * rhs_th - b * rhs_x) / det
    return np.stack([x_d, th_d, x_dd, th_dd], axis=-1)


def step(s: np.ndarray, u, dt: float = DT) -> np.ndarray:
    """One control step with u held constant, integrated with classic RK4 (the integrator MuJoCo uses here).
    Works on one state (4,) or a batch (n, 4) with u (n,).

    RK4 samples the slope four times across the step (start, two midpoints, end) and averages them with
    weights 1-2-2-1, which is accurate to O(dt^4) per step versus O(dt) for Euler.
    """
    k1 = f_continuous(s, u)
    k2 = f_continuous(s + 0.5 * dt * k1, u)
    k3 = f_continuous(s + 0.5 * dt * k2, u)
    k4 = f_continuous(s + dt * k3, u)
    return s + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def rollout(s0: np.ndarray, U: np.ndarray, step_fn=step) -> np.ndarray:
    """States S (N+1, 4) from s0 under the force sequence U (N,), with our model (or another step_fn)."""
    S = np.zeros((len(U) + 1, 4))
    S[0] = s0
    for t, u in enumerate(U):
        S[t + 1] = step_fn(S[t], u)
    return S


# ---------------------------------------------------------------------------------------------------------
# C2: derivatives of the model along a trajectory
# ---------------------------------------------------------------------------------------------------------

def linearize_step(s: np.ndarray, u: float, step_fn=step, eps: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
    """A = d step / d s (4x4), B = d step / d u (4x1) at (s, u), by central differences of our own step().

    Part B linearized once, at upright. iLQR linearizes at every point of the current trajectory: near the
    bottom the pole is stable (gravity pulls it back), near the top it is unstable, and in between the force's
    effect on the pole changes with cos(theta). Central differences are accurate to O(eps^2); analytic
    derivatives would be faster but give the same numbers.
    """
    A = np.zeros((4, 4))
    for i in range(4):
        ds = np.zeros(4)
        ds[i] = eps
        A[:, i] = (step_fn(s + ds, u) - step_fn(s - ds, u)) / (2 * eps)
    B = ((step_fn(s, u + eps) - step_fn(s, u - eps)) / (2 * eps)).reshape(4, 1)
    return A, B


def linearize_trajectory(S: np.ndarray, U: np.ndarray, step_fn=step, eps: float = 1e-5):
    """A_t (n, 4, 4), B_t (n, 4, 1) at every (S[t], U[t]): linearize_step for all n points in ONE batched call.

    Same central differences, but the 10 perturbed copies of all n points (8 for the states, 2 for the force)
    go through step_fn together as one (10 n, 4) array, so NumPy does the arithmetic in C instead of 10 n
    separate Python calls. Same numbers, much faster (Part D needs many solves). step_fn must accept batches.
    """
    n = len(U)
    E = np.eye(4) * eps
    S_pert = np.concatenate([S[:, None, :] + E[None], S[:, None, :] - E[None],     # (n, 8, 4): +eps and -eps
                             S[:, None, :], S[:, None, :]], axis=1)                 # (n, 10, 4): + 2 for u
    U_pert = np.concatenate([np.repeat(U[:, None], 8, axis=1), U[:, None] + eps, U[:, None] - eps], axis=1)
    out = step_fn(S_pert.reshape(-1, 4), U_pert.reshape(-1)).reshape(n, 10, 4)
    A = ((out[:, :4] - out[:, 4:8]) / (2 * eps)).transpose(0, 2, 1)    # column i = d s_next / d s_i
    B = ((out[:, 8] - out[:, 9]) / (2 * eps))[:, :, None]
    return A, B


# ---------------------------------------------------------------------------------------------------------
# C3: the cost and its derivatives
# ---------------------------------------------------------------------------------------------------------

def quadratic_terminal(s: np.ndarray):
    """Default terminal cost 1/2 (s - goal)^T Q_F (s - goal): returns (value, gradient (4,), Hessian (4, 4)).

    Any function with this signature can replace it (ilqr(..., terminal=...)); Part D5 uses value-function
    estimates.
    """
    e = s - S_GOAL
    return 0.5 * e @ Q_F @ e, Q_F @ e, Q_F.copy()


def total_cost(S: np.ndarray, U: np.ndarray, terminal=quadratic_terminal) -> float:
    running = DT * np.sum(0.5 * R_RUN * U**2 + 0.5 * Q_X_RUN * S[:-1, 0]**2)
    return float(running + terminal(S[-1])[0])


def running_cost_derivatives(s: np.ndarray, u: float):
    """l_x (4,), l_u (1,), l_xx (4,4), l_uu (1,1), l_ux (1,4) of the running cost at (s, u).

    The cost is quadratic, so these are exact and constant except l_x and l_u (linear in s and u).
    """
    l_x = np.array([DT * Q_X_RUN * s[0], 0.0, 0.0, 0.0])
    l_u = np.array([DT * R_RUN * u])
    l_xx = np.diag([DT * Q_X_RUN, 0.0, 0.0, 0.0])
    l_uu = np.array([[DT * R_RUN]])
    l_ux = np.zeros((1, 4))         # no cross term between force and state in this cost
    return l_x, l_u, l_xx, l_uu, l_ux


# ---------------------------------------------------------------------------------------------------------
# C4: backward pass
# ---------------------------------------------------------------------------------------------------------

def backward_pass(S: np.ndarray, U: np.ndarray, mu: float, step_fn=step, u_max: float = U_MAX,
                  terminal=quadratic_terminal):
    """Time-varying LQR around the trajectory (S, U). Returns (k, K, dV) or None if mu is too small.

    We look for small changes du_t that improve the trajectory, as a function of the small state change
    dx_t = s_t - S[t] they will cause. Going backwards from the end, keep a quadratic model of the
    cost-to-go around the trajectory:  V_t(S[t] + dx) ~ V + V_x dx + 1/2 dx^T V_xx dx.

    At each step t, the Q-function (cost of taking du now, then acting optimally) to second order is:
        Q_x  = l_x  + A^T V_x'          Q_u  = l_u  + B^T V_x'
        Q_xx = l_xx + A^T V_xx' A       Q_uu = l_uu + B^T V_xx' B       Q_ux = l_ux + B^T V_xx' A
    (' = at step t+1). This is the same algebra as dlqr() in Part B, but around a trajectory and with the
    linear terms V_x, Q_x, Q_u because the trajectory is not yet optimal.

    Minimizing over du gives  du = k + K dx  with  k = -Q_uu^-1 Q_u  (feedforward: the improvement) and
    K = -Q_uu^-1 Q_ux  (feedback: how to react when the state is off the trajectory).

    Force limit (box constraint): u + du must stay in [-U_MAX, U_MAX]. With one input that is exact and easy:
    clamp k to the allowed range, and where it is clamped set K = 0 (feedback could not act there anyway).

    mu (regularization) is added to Q_uu: it makes the step smaller and safer when the quadratic model is
    not trustworthy (Q_uu small or negative far from the optimum).

    dV = (sum k^T Q_u, sum 1/2 k^T Q_uu k): the cost change the quadratic model predicts, -(alpha dV[0] +
    alpha^2 dV[1]) for step size alpha (Tassa et al. 2012). It is exact at alpha = 1 when the dynamics are linear
    and the cost quadratic (check C4). It is not the exact slope at alpha -> 0 (V_x already includes the
    improvement of later steps), so the line search only asks for a fraction of it.
    """
    _, V_x, V_xx = terminal(S[-1])  # terminal cost gradient and Hessian start the recursion
    n_steps = len(U)
    k = np.zeros(n_steps)
    K = np.zeros((n_steps, 1, 4))
    dV = np.zeros(2)
    A_all, B_all = linearize_trajectory(S[:-1], U, step_fn)     # all n_steps linearizations at once
    for t in reversed(range(n_steps)):
        A, B = A_all[t], B_all[t]
        l_x, l_u, l_xx, l_uu, l_ux = running_cost_derivatives(S[t], U[t])
        Q_x = l_x + A.T @ V_x
        Q_u = l_u + B.T @ V_x
        Q_xx = l_xx + A.T @ V_xx @ A
        Q_uu = l_uu + B.T @ V_xx @ B + mu * np.eye(1)
        Q_ux = l_ux + B.T @ V_xx @ A
        if Q_uu[0, 0] <= 0:
            return None             # not a minimum in du: caller raises mu and retries

        # Feedforward, clamped to the force limit; no feedback where the limit is active
        k_free = -Q_u[0] / Q_uu[0, 0]
        k_t = np.clip(k_free, -u_max - U[t], u_max - U[t])
        K_t = np.zeros((1, 4)) if k_t != k_free else -Q_ux / Q_uu[0, 0]
        k[t], K[t] = k_t, K_t

        # Cost-to-go at step t, with du = k_t + K_t dx substituted into the Q-function
        kv = np.array([k_t])
        V_x = Q_x + K_t.T @ Q_uu @ kv + K_t.T @ Q_u + Q_ux.T @ kv
        V_xx = Q_xx + K_t.T @ Q_uu @ K_t + K_t.T @ Q_ux + Q_ux.T @ K_t
        V_xx = 0.5 * (V_xx + V_xx.T)
        dV += np.array([kv @ Q_u, 0.5 * kv @ Q_uu @ kv])
    return k, K, dV


# ---------------------------------------------------------------------------------------------------------
# C5: forward pass and the main loop
# ---------------------------------------------------------------------------------------------------------

def forward_pass(S: np.ndarray, U: np.ndarray, k: np.ndarray, K: np.ndarray, alpha: float, step_fn=step,
                 u_max: float = U_MAX):
    """New trajectory: u_t = U[t] + alpha k_t + K_t (s_t - S[t]), simulated with our model.

    alpha scales only the feedforward step (line search). The feedback term is what keeps the new trajectory
    close to the region where the linearization was valid: if the state drifts from the old trajectory,
    K pulls it back instead of letting the error grow through the unstable dynamics.
    """
    S_new = np.zeros_like(S)
    U_new = np.zeros_like(U)
    S_new[0] = S[0]
    for t in range(len(U)):
        u = U[t] + alpha * k[t] + (K[t] @ (S_new[t] - S[t]))[0]
        U_new[t] = np.clip(u, -u_max, u_max)
        S_new[t + 1] = step_fn(S_new[t], U_new[t])
    return S_new, U_new


def ilqr(s0: np.ndarray, U0: np.ndarray, max_iter: int = 200, tol: float = 1e-7, verbose: bool = True,
         u_max: float = U_MAX, terminal=quadratic_terminal):
    """Optimize the force sequence from initial guess U0 (its length is the horizon). Returns (S, U, K, costs)."""
    U = np.clip(U0.astype(float), -u_max, u_max)
    S = rollout(s0, U)
    cost = total_cost(S, U, terminal)
    costs = [cost]
    mu, mu_min, mu_factor = 1e-6, 1e-8, 10.0
    K = np.zeros((len(U), 1, 4))
    for it in range(max_iter):
        # Backward pass; if Q_uu is not positive somewhere, regularize more and try again
        while (result := backward_pass(S, U, mu, u_max=u_max, terminal=terminal)) is None:
            mu *= mu_factor
        k, K, dV = result

        # Line search: try the full step, then smaller ones, until the cost drops by a reasonable fraction of
        # what the quadratic model predicted (the model is only trustworthy near the old trajectory)
        accepted = False
        for alpha in (1.0, 0.5, 0.25, 0.1, 0.05, 0.01):
            S_new, U_new = forward_pass(S, U, k, K, alpha, u_max=u_max)
            new_cost = total_cost(S_new, U_new, terminal)
            expected = -(alpha * dV[0] + alpha**2 * dV[1])     # predicted decrease (positive)
            if expected > 0 and (cost - new_cost) / expected > 1e-4:
                accepted = True
                break
        if not accepted:
            mu *= mu_factor                 # model not trusted: take more careful steps next time
            if mu > 1e10:
                break
            continue
        mu = max(mu_min, mu / mu_factor)    # model worked: allow bigger steps

        rel_change = (cost - new_cost) / cost
        S, U, cost = S_new, U_new, new_cost
        costs.append(cost)
        if verbose and (it < 5 or it % 10 == 0):
            print(f"  iter {it:3d}: cost {cost:10.4f}  alpha {alpha:<5}  mu {mu:.0e}")
        if rel_change < tol:
            break
    # Feedback gains consistent with the final trajectory
    _, K, _ = backward_pass(S, U, mu, u_max=u_max, terminal=terminal)
    return S, U, K, np.array(costs)


def make_plan_controller(S_plan, U_plan, K_plan, feedback: bool = True, u_max: float = U_MAX):
    """Controller that follows a fixed plan from time 0, then holds the pole up with Part B's LQR.

    During the plan: u = U_plan[t] + K_t (s - S_plan[t]) (or the plan's forces alone, feedback=False).
    After it: Part B's LQR gain, designed for the nominal model at upright.
    """
    A, B = discretize(*linearize_continuous(), DT)
    K_hold, _ = dlqr(A, B, Q_LQR, R_LQR)
    n = len(U_plan)

    def controller(model, data):
        t = min(round(data.time / DT), n)
        s = get_state(data)
        if t < n:
            u = U_plan[t] + ((K_plan[t] @ (s - S_plan[t]))[0] if feedback else 0.0)
        else:
            u = -(K_hold @ s)[0]
        data.ctrl[0] = np.clip(u, -u_max, u_max)
    return controller


# ---------------------------------------------------------------------------------------------------------
# Checks and experiments
# ---------------------------------------------------------------------------------------------------------

def part_c1(ctx):
    print("\n=== C1: our nonlinear model vs MuJoCo ===")
    model, data = make_model()
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        s = np.array([rng.uniform(-1, 1), rng.uniform(-np.pi, np.pi), rng.uniform(-3, 3), rng.uniform(-8, 8)])
        u = rng.uniform(-U_MAX, U_MAX)
        set_state(model, data, s, u)
        mujoco.mj_step(model, data)
        worst = max(worst, np.abs(step(s, u) - get_state(data)).max())
    print(f"max |step() - mj_step| over 200 random states and forces = {worst:.2e}")
    # Same equations, same integrator: only the pole's 1e-9 rotational inertia and rounding differ
    assert worst < 1e-6, "our model disagrees with MuJoCo"


def part_c2(ctx):
    print("\n=== C2: linearization anywhere along a trajectory ===")
    model, data = make_model()
    worst = 0.0
    for th in (np.pi, 2.0, 1.0, 0.0):     # hanging, on the way up, upright
        s, u = np.array([0.2, th, 0.5, 1.0]), 3.0
        A, B = linearize_step(s, u)
        A_mj, B_mj = step_jacobians(model, data, s, u)
        worst = max(worst, np.abs(A - A_mj).max(), np.abs(B - B_mj).max())
        print(f"  theta = {th:4.2f}: dtheta_dot/du per step = {B[3, 0]:+.4f}  (sign and size depend on cos theta)")
    print(f"max |A, B - MuJoCo| = {worst:.2e}")
    assert worst < 1e-5


def part_c4(ctx):
    print("\n=== C4: backward pass on a linear-quadratic problem (where it must be exact) ===")
    # With linear dynamics and a quadratic cost, the quadratic model in the backward pass is the problem
    # itself: one full step (alpha = 1) must land exactly on the optimum, and the predicted cost drop must
    # equal the actual one. Use Part B's discrete model of the upright cart-pole, starting tilted.
    A, B = discretize(*linearize_continuous(), DT)
    linear_step = lambda s, u: s @ A.T + np.multiply.outer(u, B[:, 0])     # works on batches too
    # No force limit here (a clamp is not linear). Initial guess: the forces Part B's LQR applies from s0. It is
    # not optimal for this cost (different weights, a terminal cost) but keeps every number moderate; from
    # u = 0 the tilt would grow ~e^(5.1 t) to a cost of ~1e13, and rounding would swamp the check.
    s0 = np.array([0.0, 0.1, 0.0, 0.0])
    K_lqr, _ = dlqr(A, B, Q_LQR, R_LQR)
    S0, U0 = np.zeros((N + 1, 4)), np.zeros(N)
    S0[0] = s0
    for t in range(N):
        U0[t] = -(K_lqr @ S0[t])[0]
        S0[t + 1] = linear_step(S0[t], U0[t])
    k, K, dV = backward_pass(S0, U0, mu=0.0, step_fn=linear_step, u_max=np.inf)
    S1, U1 = forward_pass(S0, U0, k, K, 1.0, linear_step, u_max=np.inf)
    predicted, actual = -(dV[0] + dV[1]), total_cost(S0, U0) - total_cost(S1, U1)
    print(f"cost {total_cost(S0, U0):.6f} -> {total_cost(S1, U1):.6f}: predicted drop {predicted:.6f}, actual {actual:.6f}")
    assert abs(predicted - actual) < 1e-6 * actual, "the quadratic model should be exact here"
    # A second backward pass from the result finds nothing left to improve
    k2, _, dV2 = backward_pass(S1, U1, mu=0.0, step_fn=linear_step, u_max=np.inf)
    print(f"second iteration: max |k| = {np.abs(k2).max():.1e} N, predicted drop {-(dV2[0] + dV2[1]):.1e}")
    assert np.abs(k2).max() < 1e-6, "one iteration should reach the optimum"


def part_c5(ctx):
    print("\n=== C5: iLQR swing-up ===")
    S, U, K, costs = ilqr(S_START, np.zeros(N))
    print(f"converged in {len(costs) - 1} accepted iterations: cost {costs[0]:.2f} -> {costs[-1]:.4f}")
    print(f"final state = {np.round(S[-1], 4)}, max |u| = {np.abs(U).max():.2f} N, max |x| = {np.abs(S[:, 0]).max():.2f} m")
    assert np.all(np.diff(costs) < 0), "cost must decrease at every accepted iteration"
    assert np.abs(S[-1] - S_GOAL).max() < 0.05, "did not end upright and still"
    at_limit = int(np.sum(np.abs(U) > U_MAX - 1e-6))
    print(f"steps at the force limit: {at_limit} of {N}")
    assert np.abs(U).max() <= U_MAX + 1e-9
    assert at_limit > 0, "expected the plan to use the full force (box constraint active)"
    ctx.update(S=S, U=U, K=K)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes[0, 0].semilogy(costs); axes[0, 0].set_title("cost per iteration"); axes[0, 0].set_xlabel("iteration")
    t = np.arange(N + 1) * DT
    axes[0, 1].plot(t, S[:, 1]); axes[0, 1].axhline(0, color="gray", lw=0.5); axes[0, 1].set_title("theta (rad), 0 = upright")
    axes[1, 0].plot(t, S[:, 0]); axes[1, 0].set_title("x (m)"); axes[1, 0].set_xlabel("t (s)")
    axes[1, 1].plot(t[:-1], U); axes[1, 1].axhline(U_MAX, color="gray", ls="--"); axes[1, 1].axhline(-U_MAX, color="gray", ls="--")
    axes[1, 1].set_title("u (N)"); axes[1, 1].set_xlabel("t (s)")
    for ax in axes.flat:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / "c5_swingup_plan.png", dpi=120)
    plt.close(fig)
    print(f"saved {results_dir / 'c5_swingup_plan.png'}")


def part_c6(ctx):
    print("\n=== C6: the plan in MuJoCo, with a pole 20% heavier than the planner assumed ===")
    S_plan, U_plan, K_plan = ctx["S"], ctx["U"], ctx["K"]
    PLANT_MASS = 1.2 * M_POLE   # the real system differs from the model, as it always does
    # Both runs hand over to Part B's LQR (designed for the nominal model) to hold the pole up after the plan

    wrap = lambda th: (th + np.pi) % (2 * np.pi) - np.pi     # angle in [-pi, pi): upright = 0
    worst_hold = {}     # largest |theta| during the 3 s after the plan ends
    panels = []
    for name, feedback in (("open loop: plan's forces only", False), ("plan + feedback gains K_t", True)):
        model, data = make_model(U_MAX, pole_mass=PLANT_MASS)
        controller = make_plan_controller(S_plan, U_plan, K_plan, feedback)
        set_state(model, data, S_START)
        worst = 0.0
        while data.time < T + 3.0 - 1e-9:       # the plan, then 3 s of holding
            controller(model, data)
            mujoco.mj_step(model, data)
            if data.time > T:
                worst = max(worst, abs(wrap(data.qpos[1])))
        worst_hold[feedback] = worst
        print(f"  {name:<32}: max |theta| while holding = {worst:.3f} rad, final x = {data.qpos[0]:+.2f} m")

        set_state(model, data, S_START)
        panels.append(record_frames(model, data, T + 3.0, camera=model_camera(model, "follow"), control=controller,
                                    capture_fps=30, label=lambda d, name=name: f"{name}   u = {d.actuator_force[0]:+5.1f} N"))
    save_gif(side_by_side(panels), results_dir / "c6_swingup.gif", fps=30)
    print(f"saved {results_dir / 'c6_swingup.gif'}")

    assert worst_hold[True] < 0.1, "with feedback the pole should arrive upright and be held"
    assert worst_hold[False] > np.pi / 2, "open loop was expected to miss (pole below horizontal) with the heavier pole"


PARTS = [part_c1, part_c2, part_c4, part_c5, part_c6]


def main():
    ctx = {}
    for part in PARTS:
        part(ctx)
    print("\n=== Part C complete ===")


if __name__ == "__main__":
    main()
