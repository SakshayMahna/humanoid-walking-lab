"""PD gain experiments on the pendulum: simulation vs. linear control theory."""

import sys
from pathlib import Path

import mujoco
import numpy as np

script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir.parent))  # make exercises/render_utils.py importable
from render_utils import make_camera, record_gif

from pendulum import L, MOTOR, make_pd, metrics, pendulum_xml, simulate

TARGET = 0.5      # desired angle (rad)
DURATION = 6.0    # simulated seconds per run (slowest runs need ~4 s to settle)

# Physical parameters, read from the model rather than hardcoded
model = mujoco.MjModel.from_xml_string(pendulum_xml(L, MOTOR))
data = mujoco.MjData(model)
m = model.body("pole").mass[0]       # kg (rod is massless, so this is the bob)
r = model.geom("bob").size[0]        # bob radius (m)
g = abs(model.opt.gravity[2])        # m/s²
dt = model.opt.timestep              # s
mujoco.mj_forward(model, data)
I_mujoco = data.M[0]                 # joint-space inertia MuJoCo uses (1 DoF -> 1 number)

def pendulum_inertia(m, L, r):
    """Rotational inertia about the hinge (kg·m²): point mass at L + solid sphere of radius r."""
    return (m * L**2) + (2/5) * m * r**2


def predicted_steady_state(kp, target, m, g, L):
    """Angle where the PD torque balances gravity (rad)."""
    # Small-angle approximation: kp*(target - q) = m*g*L*sin(q) with sin(q) ≈ q, solved for q.
    # sin(q) < q, so it overestimates gravity and lands ~0.002 rad low; the error is
    # divided by the total stiffness (kp + m*g*L), so stiffer gains hide it.
    return kp * target / (m * g * L + kp)


def effective_stiffness(kp, m, g, L, q_eq):
    """Total stiffness around q_eq (N·m/rad): PD spring + gravity's restoring effect."""
    return kp + m * g * L * np.cos(q_eq)


def natural_frequency(k_eff, I):
    """Undamped natural frequency ω_n (rad/s)."""
    return np.sqrt(k_eff / I)


def damping_ratio(kd, k_eff, I):
    """Damping ratio ζ (dimensionless)."""
    return kd / (2 * np.sqrt(k_eff * I))


def critical_kd(k_eff, I):
    """KD that gives ζ = 1 (N·m·s/rad)."""
    return 2 * np.sqrt(k_eff * I)


def predicted_overshoot(zeta, step):
    """Peak overshoot (rad) for a step of size `step`. 0 if ζ >= 1."""
    if zeta < 1:
        return step * np.exp(-np.pi * zeta / np.sqrt(1 - zeta**2))
    else:
        return 0.0


def predicted_settling_time(zeta, wn):
    """Time to stay within 2% of the final value (s)."""
    if zeta < 1:
        return 4 / (zeta * wn)
    else:
        return 4 / (wn * (zeta - np.sqrt(zeta**2 - 1)))


def stability_limits(I, dt):
    """(KP_max, KD_max) beyond which a single dt step overcorrects and the sim diverges."""
    kd_max = 2 * I / dt
    kp_max = I * (2 / dt)**2
    return (kp_max, kd_max)


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

cam = make_camera(lookat=[0, 0, -0.25], distance=1.5, azimuth=90, elevation=0)  # side view of the swing plane
gif_dir = script_dir / "gifs"
gif_dir.mkdir(exist_ok=True)


def fmt(x, spec=".3f"):
    """Format a number for the table; None (not implemented) -> '—'."""
    if x is None:
        return "—"
    return format(x, spec)


def theory(kp, kd, q_final_sim):
    """All predictions for one (kp, kd). Missing pieces stay None."""
    I = pendulum_inertia(m, L, r)
    q_eq = predicted_steady_state(kp, TARGET, m, g, L)
    q_lin = q_eq if q_eq is not None else q_final_sim   # linearize around theory, else sim
    k_eff = effective_stiffness(kp, m, g, L, q_lin)
    wn = natural_frequency(k_eff, I) if None not in (k_eff, I) else None
    zeta = damping_ratio(kd, k_eff, I) if None not in (k_eff, I) else None
    overshoot = predicted_overshoot(zeta, q_lin) if zeta is not None else None
    settling = predicted_settling_time(zeta, wn) if None not in (zeta, wn) else None
    return dict(final=q_eq, wn=wn, zeta=zeta, overshoot=overshoot, settling_time=settling)


header = (f"{'KP':>8} {'KD':>6} | {'final sim':>9} {'theory':>7} | {'overshoot':>9} {'theory':>7} | "
          f"{'settle s':>8} {'theory':>7} | {'ω_n':>6} {'ζ':>6} | {'max τ':>8} | status")


def run_experiment(name, gains):
    print(f"\n=== {name} ===")
    print(header)
    print("-" * len(header))
    for kp, kd in gains:
        t, q, tau, diverged = simulate(make_pd(kp, kd, TARGET), DURATION)
        sim = metrics(t, q, tau, diverged)
        th = theory(kp, kd, sim["final"])
        if diverged:
            status = f"DIVERGED at t={t[-1]:.3f}s"
        elif np.isnan(sim["settling_time"]):
            status = "not settled"
        else:
            status = "ok"
        print(f"{kp:>8g} {kd:>6g} | {fmt(sim['final']):>9} {fmt(th['final']):>7} | "
              f"{fmt(sim['overshoot']):>9} {fmt(th['overshoot']):>7} | "
              f"{fmt(sim['settling_time'], '.2f'):>8} {fmt(th['settling_time'], '.2f'):>7} | "
              f"{fmt(th['wn'], '.2f'):>6} {fmt(th['zeta'], '.2f'):>6} | "
              f"{fmt(sim['max_torque'], '.1f'):>8} | {status}")

        # GIF of the same run, labeled with its gains
        label = f"KP={kp:g}  KD={kd:g}"
        if th["zeta"] is not None:
            label += f"  zeta={th['zeta']:.2f}"
        mujoco.mj_resetData(model, data)
        record_gif(model, data, gif_dir / f"{name.split(':')[0].lower().replace(' ', '')}_kp{kp:g}_kd{kd:g}.gif",
                   duration=DURATION, camera=cam, control=make_pd(kp, kd, TARGET),
                   capture_fps=20, width=320, height=240, label=label)


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

print("=== Physical parameters ===")
print(f"m = {m} kg | L = {L} m | r = {r} m | g = {g:.3f} m/s² | dt = {dt} s")
print(f"inertia: MuJoCo = {I_mujoco:.4f} | theory = {fmt(pendulum_inertia(m, L, r), '.4f')} kg·m²")

run_experiment("Exp1: stiffness", [(5, 2), (20, 2), (80, 2), (320, 2)])
run_experiment("Exp2: damping", [(20, 0.5), (20, 2), (20, 5), (20, 20)])

# Exp3 needs your theory: KD for ζ = 1 at KP = 20
I = pendulum_inertia(m, L, r)
q_eq = predicted_steady_state(20, TARGET, m, g, L)
k_eff = effective_stiffness(20, m, g, L, q_eq) if q_eq is not None else None
kd_crit = critical_kd(k_eff, I) if None not in (k_eff, I) else None
if kd_crit is None:
    print("\n=== Exp3: critical damping === skipped (implement the theory functions first)")
else:
    run_experiment("Exp3: critical damping", [(20, kd_crit), (20, 0.8 * kd_crit), (20, 1.2 * kd_crit)])

limits = stability_limits(I, dt) if I is not None else None
print(f"\nstability limits (theory): {limits if limits is not None else '—'}")
run_experiment("Exp4: breaking it", [(1000, 2), (30000, 2), (38000, 2), (42000, 2),
                                     (20, 50), (20, 99), (20, 101), (20, 120)])

print(f"\nGIFs saved to {gif_dir}")
