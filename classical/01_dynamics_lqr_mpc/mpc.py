"""Module 1, Part D: iLQR as Model Predictive Control (MPC).

Part C planned once, before moving, and then followed the plan with fixed feedback gains. Here the controller
re-plans while it moves: every few ticks it measures the state, re-solves iLQR from there (starting from the
previous plan, shifted), and switches to the new plan. That is MPC.

    .venv/bin/python classical/01_dynamics_lqr_mpc/mpc.py          (all parts, a few minutes)
    .venv/bin/python classical/01_dynamics_lqr_mpc/mpc.py d3       (one part; D2-D5 run D1 first)
"""

import sys
import time
from pathlib import Path

import matplotlib
import mujoco
import numpy as np

matplotlib.use("Agg")   # files only, no window
import matplotlib.pyplot as plt

script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir.parents[1] / "exercises"))   # render_utils
from render_utils import record_frames, save_gif, side_by_side

from cartpole import DT, M_POLE, get_state, make_model, set_state
from ilqr import N, S_START, U_MAX, ilqr, make_plan_controller, quadratic_terminal, step
from cartpole import G, L
from lqr import Q as Q_LQR, R as R_LQR, discretize, dlqr, linearize_continuous, model_camera

results_dir = script_dir / "results"
results_dir.mkdir(exist_ok=True)


def wrap(th):
    """Angle in [-pi, pi): upright = 0, whichever way round the pole went."""
    return (th + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------------------------------------------------
# D1: the MPC controller
# ---------------------------------------------------------------------------------------------------------

class MPCController:
    """Receding-horizon iLQR, called once per tick like every controller here: controller(model, data).

    Every plan covers the next `horizon` seconds; every `replan` ticks a new one is solved. The goal is the
    same at every moment (be upright, centred, still), so one controller does the swing-up and then keeps
    balancing for as long as it runs.

    Latency: a solve needs time, so the new plan is only switched on `replan` ticks later (latency=True): the
    computer solves while the robot executes the previous plan. Real-time is then possible if a solve takes
    less than replan * DT. Until the new plan is ready, the previous plan's feedback gains K_t correct the
    state (the Talos scheme in the survey, V-A). With latency=False the new plan is used at once, as if
    solving took no time.

    Delay compensation: the new plan must start from where the robot WILL BE when it is switched on, not from
    where it is now, because during the solve the robot keeps executing the old plan. So the solve starts from
    a prediction: the measured state, run forward `replan` ticks through our model under the old plan's
    policy (compensate=False plans from the current state instead; D1 compares).

    Warm start: the initial guess for each solve is the old plan's policy, u = U_t + K_t (s - S_t), run
    forward through our model from the (predicted) start state, CLOSED-LOOP: the gains steer the guess back
    towards the old plan when the state is off it. Replaying only the old forces, OPEN-LOOP (warm_start=
    "open"), looks equivalent but is not: the upright pole is unstable (errors grow ~e^(5 t)), so any gap
    between the real state and the old plan, which a wrong model guarantees, blows up over a 3 s replay and
    every solve starts from a ruined guess (D3). Beyond the end of the old plan the guess continues with u = 0.
    A good guess means a few iterations per solve, instead of the 67 a cold start needs.

    The first plan is solved fully before the run starts (the robot is at rest; there is time).
    """

    def __init__(self, horizon: float = 3.0, replan: int = 5, iters: int = 3, u_max: float = U_MAX,
                 latency: bool = True, compensate: bool = True, warm_start: str = "closed",
                 terminal=quadratic_terminal):
        self.H = round(horizon / DT)        # ticks in each plan
        self.replan = replan
        self.iters = iters
        self.u_max = u_max
        self.latency = latency
        self.compensate = compensate        # with latency: plan from the predicted state (see above)
        self.warm_start = warm_start        # "closed" (old plan with its gains) or "open" (old forces only)
        self.terminal = terminal            # terminal cost of every plan (D5: value-function estimates)
        self.active = None                  # (start_tick, S, U, K): the plan in use
        self.pending = None                 # (activation_tick, plan): solved, waiting for its time
        self.solve_times, self.solve_iters = [], []

    def start(self, s0: np.ndarray) -> None:
        """Full solve before the run (no time limit), from zero force."""
        S, U, K, _ = ilqr(s0, np.zeros(self.H), verbose=False, u_max=self.u_max, terminal=self.terminal)
        self.active, self.pending = (0, S, U, K), None
        self.solve_times, self.solve_iters = [], []

    def _policy(self, plan, tick: int, s: np.ndarray) -> float:
        """The plan's feedback policy at a tick: u = U_i + K_i (s - S_i), clipped to the force limit."""
        start, S, U, K = plan
        i = tick - start                                            # index into the plan
        return float(np.clip(U[i] + (K[i] @ (s - S[i]))[0], -self.u_max, self.u_max))

    def _warm_start(self, plan_start: int, s: np.ndarray) -> np.ndarray:
        start, _, U_prev, _ = self.active
        U0 = np.zeros(self.H)
        n_old = min(self.H, len(U_prev) - (plan_start - start))     # ticks the old plan still covers
        if self.warm_start == "open":
            U0[:n_old] = U_prev[plan_start - start:][:n_old]        # the old forces, replayed as they were
        else:
            for j in range(n_old):                                  # the old policy, with its feedback
                U0[j] = self._policy(self.active, plan_start + j, s)
                s = step(s, U0[j])
        return U0

    def _solve(self, tick: int, s: np.ndarray):
        t0 = time.perf_counter()
        plan_start = tick
        if self.latency and self.compensate:
            # Predict where the old plan takes us by the time the new one is switched on
            for t in range(tick, tick + self.replan):
                s = step(s, self._policy(self.active, t, s))
            plan_start = tick + self.replan
        U0 = self._warm_start(plan_start, s)
        S, U, K, costs = ilqr(s, U0, max_iter=self.iters, verbose=False, u_max=self.u_max, terminal=self.terminal)
        self.solve_times.append(time.perf_counter() - t0)
        self.solve_iters.append(len(costs) - 1)                     # accepted iterations
        return (plan_start, S, U, K)

    def __call__(self, model, data):
        tick = round(data.time / DT)
        s = get_state(data)
        if self.pending is not None and tick >= self.pending[0]:
            self.active, self.pending = self.pending[1], None       # the plan solved earlier is now ready
        if tick > 0 and tick % self.replan == 0:
            plan = self._solve(tick, s)
            if self.latency:
                self.pending = (tick + self.replan, plan)           # usable one interval later
            else:
                self.active = plan
        data.ctrl[0] = self._policy(self.active, tick, s)


def run(model, data, s0, controller, duration: float, kick=None):
    """Simulate; returns (S, U) per tick. kick = (time, d_theta_dot): a sudden push on the pole."""
    set_state(model, data, s0)
    if isinstance(controller, MPCController):
        controller.start(s0)
    S, U = [get_state(data)], []
    for _ in range(round(duration / DT)):
        apply_kick(data, kick)
        controller(model, data)
        mujoco.mj_step(model, data)
        S.append(get_state(data))
        U.append(data.actuator_force[0])
    return np.array(S), np.array(U)


def apply_kick(data, kick) -> None:
    if kick is not None and round(data.time / DT) == round(kick[0] / DT):
        data.qvel[1] += kick[1]


def held_upright(S, last: float = 2.0, tol: float = 0.1) -> bool:
    """True if |theta| stayed under tol (rad) during the last `last` seconds."""
    return bool(np.abs(wrap(S[-round(last / DT):, 1])).max() < tol)


def film(make, s0, controller, duration, label, kick=None):
    """Frames of one run with the cart-following camera. make() builds a fresh (model, data)."""
    model, data = make()
    set_state(model, data, s0)
    if isinstance(controller, MPCController):
        controller.start(s0)

    def control(m, d):
        apply_kick(d, kick)
        controller(m, d)
    return record_frames(model, data, duration, camera=model_camera(model, "follow"), control=control,
                         capture_fps=30, label=lambda d: f"{label}   u = {d.actuator_force[0]:+5.1f} N")


# ---------------------------------------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------------------------------------

def upright_from(S, tol: float = 0.1):
    """First time (s) after which |theta| stays under tol for the rest of the run, or None."""
    th = np.abs(wrap(S[:, 1]))
    outside = np.nonzero(th >= tol)[0]
    i = 0 if len(outside) == 0 else outside[-1] + 1
    return None if i >= len(th) else i * DT


def part_d1(ctx):
    print("\n=== D1: MPC swing-up and balance (3 s horizon, re-plan every 5 ticks, 3 iterations) ===")
    # Latency and delay compensation. Measured while building this part: with the closed-loop warm start,
    # planning from the current state works as well as planning from the predicted one, here and with
    # delays up to 300 ms, a 30% heavier pole and a push. The plan's feedback gains absorb the gap. Delay
    # compensation stays on (it is the standard practice, and costs one short rollout), but this system
    # does not need it. (An earlier version of this part claimed it did: that was the open-loop warm start
    # failing, see D3.)
    variants = {"no latency (solve takes no time)": dict(latency=False),
                "latency, planning from the current state": dict(compensate=False),
                "latency, planning from the predicted state": dict()}
    for name, kw in variants.items():
        model, data = make_model(U_MAX)
        S, _ = run(model, data, S_START, MPCController(**kw), duration=6.0)
        print(f"  {name:<44}: upright from t = {upright_from(S)} s, RMS theta over the last 2 s = "
              f"{np.sqrt(np.mean(wrap(S[-200:, 1])**2)):.4f} rad")
        assert held_upright(S), f"{name}: should swing up and hold"

    model, data = make_model(U_MAX)
    mpc = MPCController()
    S, U = run(model, data, S_START, mpc, duration=6.0)
    times = np.array(mpc.solve_times) * 1e3
    print(f"{len(times)} solves: {np.mean(mpc.solve_iters):.2f} accepted iterations on average, "
          f"solve time mean {times.mean():.0f} ms, max {times.max():.0f} ms (budget {mpc.replan * DT * 1e3:.0f} ms)")
    ctx["d1"] = (S, U)


def part_d2(ctx):
    print("\n=== D2: what it costs per decision: horizon and iterations vs. solve time ===")
    budget = 5 * DT * 1e3
    rows = []
    print(f"{'horizon (s)':>11} | {'iters':>5} | {'mean (ms)':>9} | {'95% (ms)':>8} | {'max (ms)':>8} | "
          f"{'real-time?':>10} | swing-up")
    for horizon in (1.0, 2.0, 3.0):
        for iters in (1, 3):
            model, data = make_model(U_MAX)
            mpc = MPCController(horizon=horizon, iters=iters)
            S, _ = run(model, data, S_START, mpc, duration=6.0)
            t = np.array(mpc.solve_times) * 1e3
            ok = held_upright(S)
            rows.append((horizon, iters, t.mean(), t.max(), ok))
            print(f"{horizon:>11} | {iters:>5} | {t.mean():>9.1f} | {np.percentile(t, 95):>8.1f} | {t.max():>8.1f} | "
                  f"{'yes' if t.max() < budget else 'no':>10} | {'held' if ok else 'FAILED'}")
    print(f"(budget: a solve must finish within one re-plan interval, {budget:.0f} ms. Real-time means EVERY solve")
    print(" does, so one slow solve fails it; the max varies between runs on a laptop, the 95% column much less.)")
    # Timing depends on the machine, so only the planning result is asserted
    assert all(ok for h, _, _, _, ok in rows if h == 3.0), "the full 3 s horizon should always swing up"
    ctx["d2"] = rows


def part_d3(ctx):
    print("\n=== D3: wrong model and pushes: fixed plan vs MPC (open-loop vs closed-loop warm start) ===")
    # The planner always uses the nominal model; the simulated pole is heavier and/or gets pushed
    down, tilted = S_START, np.array([0.0, 0.1, 0.0, 0.0])
    scenarios = [("swing-up, exact model", down, 1.0, None), ("swing-up, pole 20% heavier", down, 1.2, None),
                 ("swing-up, pole 30% heavier", down, 1.3, None), ("swing-up, pole 50% heavier", down, 1.5, None),
                 ("swing-up, push -2 rad/s", down, 1.0, (1.6, -2.0)), ("swing-up, push -4 rad/s", down, 1.0, (1.6, -4.0)),
                 ("balance from 0.1 rad, 20% heavier", tilted, 1.2, None)]
    S_plan, U_plan, K_plan, _ = ilqr(S_START, np.zeros(N), verbose=False)      # Part C's offline plan
    controllers = {
        "fixed plan (C)": lambda: make_plan_controller(S_plan, U_plan, K_plan),
        "MPC, open-loop": lambda: MPCController(warm_start="open"),
        "MPC, closed-loop": lambda: MPCController(),
    }
    held = {}
    print(f"{'scenario':<34} | " + " | ".join(f"{c:<16}" for c in controllers))
    for name, s0, mass, kick in scenarios:
        for c, make_c in controllers.items():
            if c == "fixed plan (C)" and s0 is tilted:
                continue                    # Part C's plan is a swing-up from hanging, not for this start
            model, data = make_model(U_MAX, pole_mass=mass * M_POLE)
            S, _ = run(model, data, s0, make_c(), duration=8.0, kick=kick)
            held[(name, c)] = held_upright(S)
        print(f"{name:<34} | " + " | ".join(
            f"{('held' if held[(name, c)] else 'FAILED') if (name, c) in held else '-':<16}" for c in controllers))

    # The video: the case that separates all three
    make = lambda: make_model(U_MAX, pole_mass=1.3 * M_POLE)
    panels = [film(make, S_START, make_c(), 8.0, c) for c, make_c in controllers.items()]
    save_gif(side_by_side(panels), results_dir / "d3_fixed_vs_mpc.gif", fps=30)
    print(f"saved {results_dir / 'd3_fixed_vs_mpc.gif'} (pole 30% heavier)")

    assert all(held[(name, "MPC, closed-loop")] for name, *_ in scenarios), "closed-loop MPC should handle all"
    assert not held[("swing-up, pole 20% heavier", "MPC, open-loop")], "expected the open-loop warm start to fail"
    assert not held[("swing-up, push -2 rad/s", "fixed plan (C)")], "expected the fixed plan to fail the push"


def part_d4(ctx):
    print("\n=== D4: force limit 2 N: LQR (does not know the limit) vs MPC (plans with it) ===")
    U_LIM = 2.0
    thetas = np.linspace(-1.2, 1.2, 9)
    theta_dots = np.linspace(-4.0, 4.0, 9)
    A, B = discretize(*linearize_continuous(), DT)
    K_lqr, _ = dlqr(A, B, Q_LQR, R_LQR)

    def lqr_controller(model, data):
        data.ctrl[0] = np.clip(-(K_lqr @ get_state(data))[0], -U_LIM, U_LIM)

    grids = {}
    t0 = time.perf_counter()
    for name in ("LQR", "MPC"):
        grid = np.zeros((len(thetas), len(theta_dots)), dtype=bool)
        for i, th in enumerate(thetas):
            for j, thd in enumerate(theta_dots):
                model, data = make_model(U_LIM)
                # 1 s horizon is enough to catch the pole; re-plan every 10 ticks to keep the grid quick
                controller = lqr_controller if name == "LQR" else MPCController(horizon=1.0, replan=10, iters=2,
                                                                                 u_max=U_LIM)
                S, _ = run(model, data, np.array([0.0, th, 0.0, thd]), controller, duration=5.0)
                grid[i, j] = held_upright(S, last=1.0)
        grids[name] = grid
        print(f"  {name}: recovered from {grid.sum()} of {grid.size} starts")
    print(f"  ({time.perf_counter() - t0:.0f} s)")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, name in zip(axes, ("LQR", "MPC")):
        ax.pcolormesh(theta_dots, thetas, grids[name], cmap="RdYlGn", vmin=0, vmax=1, shading="nearest")
        ax.set_title(f"{name}, |u| <= {U_LIM:g} N: {grids[name].sum()} of {grids[name].size}")
        ax.set_xlabel("theta_dot0 (rad/s)")
    axes[0].set_ylabel("theta0 (rad)")
    fig.tight_layout()
    fig.savefig(results_dir / "d4_region_lqr_vs_mpc.png", dpi=120)
    plt.close(fig)
    print(f"saved {results_dir / 'd4_region_lqr_vs_mpc.png'}")
    assert grids["MPC"].sum() > grids["LQR"].sum(), "planning with the limit should recover from more starts"


def fd_terminal(V):
    """Turn a terminal cost V(s) -> float into what iLQR needs: (value, gradient, Hessian), by central
    differences. The Hessian is made positive semi-definite (negative curvature clipped to 0), so the backward
    pass always sees a bowl to roll into, never a saddle."""
    def terminal(s):
        s = np.asarray(s, dtype=float)
        h = 1e-4
        g, H = np.zeros(4), np.zeros((4, 4))
        for i in range(4):
            e = np.zeros(4)
            e[i] = h
            g[i] = (V(s + e) - V(s - e)) / (2 * h)
        for i in range(4):
            for j in range(i, 4):
                ei, ej = np.zeros(4), np.zeros(4)
                ei[i], ej[j] = h, h
                H[i, j] = H[j, i] = (V(s + ei + ej) - V(s + ei - ej) - V(s - ei + ej) + V(s - ei - ej)) / (4 * h * h)
        w, vecs = np.linalg.eigh(H)
        return V(s), g, (vecs * np.maximum(w, 0.0)) @ vecs.T
    return terminal


def energy_lqr_value():
    """A hand-made estimate of the cost-to-go, to use as the terminal cost.

    Far from upright: (E - E*)^2, where E is the pole's energy about the hinge, 1/2 m L^2 th_d^2 + m g L cos(th),
    and E* = m g L is the energy of standing still at the top. A pole with exactly E* swings up to the top on
    its own, so "right energy" is a state from which finishing is easy, wherever the pole is.
    Near upright: Part B's LQR cost-to-go s^T P s, which is the true cost-to-go of balancing there.
    Plus terms that keep the cart centred and slow. The weights are hand-picked.
    """
    A, B = discretize(*linearize_continuous(), DT)
    _, P = dlqr(A, B, Q_LQR, R_LQR)
    e_top = M_POLE * G * L

    def V(s):
        energy = 0.5 * M_POLE * L**2 * s[3]**2 + M_POLE * G * L * np.cos(s[1])
        s_wrapped = np.array([s[0], wrap(s[1]), s[2], s[3]])
        near_top = ((1 + np.cos(s[1])) / 2)**4                     # 1 upright, 0 hanging, smooth in between
        return (20.0 * (energy - e_top)**2 + 0.5 * 10.0 * s[0]**2 + 0.5 * 5.0 * s[2]**2
                + near_top * 0.5 * 10.0 * s_wrapped @ P @ s_wrapped)
    return fd_terminal(V)


def part_d5(ctx):
    print("\n=== D5: the terminal cost, and what it says ===")
    # 1. The cost decides the behaviour. From 0.1 rad off upright, with 3 s of horizon, the cost here only
    # charges force and cart position along the way, and only asks for upright at the end. Falling all the
    # way round and swinging back up turns out cheaper than catching the pole:
    s0 = np.array([0.0, 0.1, 0.0, 0.0])
    S_zero, _, _, c_zero = ilqr(s0, np.zeros(N), verbose=False)
    print(f"  plan from 0.1 rad, starting guess u = 0: pole goes to {np.abs(S_zero[:, 1]).max():.2f} rad and back, "
          f"cost {c_zero[-1]:.4f}")
    A, B = discretize(*linearize_continuous(), DT)
    K_lqr, _ = dlqr(A, B, Q_LQR, R_LQR)
    U_guess, s = np.zeros(N), s0.copy()
    for t in range(N):                                  # starting guess: Part B's LQR catching the pole
        U_guess[t] = np.clip(-(K_lqr @ s)[0], -U_MAX, U_MAX)
        s = step(s, U_guess[t])
    S_catch, _, _, c_catch = ilqr(s0, U_guess, verbose=False)
    print(f"  plan from 0.1 rad, starting guess = catch it:  pole goes to {np.abs(S_catch[:, 1]).max():.2f} rad, "
          f"cost {c_catch[-1]:.4f}")
    print("  -> iLQR finds a local optimum near its starting guess, and here the detour is the cheaper one.")
    assert c_zero[-1] < c_catch[-1] and np.abs(S_zero[:, 1]).max() > np.pi

    # 2. A better terminal cost = a better estimate of the cost-to-go. With a perfect one, even a short
    # horizon would plan as if it saw to the end. Does a hand-made one let a short (cheap) horizon swing up?
    terminals = {"quadratic (D1-D4)": quadratic_terminal, "energy + LQR value": energy_lqr_value()}
    print(f"  {'horizon':>7} | {'terminal cost':<18} | exact model | pole 30% heavier | solve mean (ms)")
    results = {}
    for horizon in (1.0, 1.5):
        for name, terminal in terminals.items():
            ok, times = [], []
            for mass in (1.0, 1.3):
                model, data = make_model(U_MAX, pole_mass=mass * M_POLE)
                mpc = MPCController(horizon=horizon, iters=1, terminal=terminal)
                S, _ = run(model, data, S_START, mpc, duration=10.0)
                ok.append(held_upright(S))
                times += mpc.solve_times
            results[(horizon, name)] = ok
            print(f"  {horizon:>7} | {name:<18} | {'held' if ok[0] else 'FAILED':<11} | "
                  f"{'held' if ok[1] else 'FAILED':<16} | {1e3 * np.mean(times):.1f}")
    assert all(results[(1.5, "quadratic (D1-D4)")]), "1.5 s with 1 iteration should be enough (real-time)"
    # Recorded result: the hand-made value estimate does not help at 1 s and makes 1.5 s worse. A terminal
    # cost only helps if it is a GOOD estimate of the cost-to-go; writing one by hand is hard. Learning one
    # from data is what an RL critic does.


PARTS = {"d1": part_d1, "d2": part_d2, "d3": part_d3, "d4": part_d4, "d5": part_d5}


def main():
    chosen = sys.argv[1:] or list(PARTS)
    ctx = {}
    for name in PARTS:
        if name in chosen or (name == "d1" and chosen != ["d1"]):
            PARTS[name](ctx)
    print("\n=== done ===")


if __name__ == "__main__":
    main()
