from typing import Callable

import mujoco
import numpy as np

L = 0.5                          # default rod length (m)
MOTOR = '<motor joint="hinge"/>'  # motor: ctrl is applied directly as torque (N·m)


def pendulum_xml(L: float, actuator: str = "") -> str:
    pendulum = f"""
    <mujoco>
        <option timestep="0.005"/>  <!-- same dt as the G1 policy -->
        <worldbody>
            <body name="pole">  <!-- no pos: pivot at world origin -->
                <joint name="hinge" type="hinge" axis="0 1 0"/>  <!-- swings in x-z plane -->
                <geom name="rod" type="capsule" fromto="0 0 0 0 0 {-L}" size="0.01" density="0"/>  <!-- massless rod -->
                <geom name="bob" type="sphere" pos="0 0 {-L}" size="0.05" mass="1"/>  <!-- 1 kg tip mass -->
            </body>
        </worldbody>
        <actuator>
            {actuator}
        </actuator>
    </mujoco>
    """
    return pendulum


def make_pd(kp: float, kd: float, target: float) -> Callable[[mujoco.MjModel, mujoco.MjData], None]:
    def control(model, data):
        q, q_dot = data.qpos[0], data.qvel[0]
        data.ctrl[0] = kp * (target - q) - kd * q_dot
    return control


def simulate(control, duration: float, L: float = L):
    """Run from rest (hanging down) with a torque motor.

    Returns (t, q, tau, diverged) where t, q, tau are arrays (s, rad, N·m) and
    diverged is True if the simulation blew up.
    """
    model = mujoco.MjModel.from_xml_string(pendulum_xml(L, MOTOR))
    data = mujoco.MjData(model)

    # Count steps instead of looping on data.time: when a simulation blows up,
    # MuJoCo auto-resets data (time back to 0), so a time-based loop never ends.
    n_steps = round(duration / model.opt.timestep)
    bad = (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL, mujoco.mjtWarning.mjWARN_BADQACC)

    t, q, tau = [], [], []
    diverged = False
    for _ in range(n_steps):
        control(model, data)
        mujoco.mj_step(model, data)
        t.append(data.time)
        q.append(data.qpos[0])
        tau.append(data.ctrl[0])
        # MuJoCo's own instability warning, or the pendulum spun past π (lost control)
        if any(data.warning[w].number for w in bad) or abs(data.qpos[0]) > np.pi:
            diverged = True
            break

    return np.array(t), np.array(q), np.array(tau), diverged


def metrics(t, q, tau, diverged: bool, band: float = 0.02) -> dict:
    """Summarize a step response (start at q = 0). nan = not meaningful for this run."""
    nan = float("nan")
    if diverged:
        return dict(final=nan, overshoot=nan, settling_time=nan, max_torque=nan)

    final = q[-1]
    # Slow overdamped runs creep toward the target and look "settled" relative to their
    # own last sample. Catch that: compare the mean of the last 0.5 s to the 0.5 s before.
    n = int(round(0.5 / (t[1] - t[0])))
    drift = abs(q[-n:].mean() - q[-2 * n:-n].mean())

    # Settling time: last moment |q - final| was outside ±band of the step size (0 -> final).
    outside = np.abs(q - final) > band * abs(final)
    if drift > 0.25 * band * abs(final):
        settling_time = nan                       # still drifting at the end: not settled
    elif not outside.any():
        settling_time = 0.0
    elif outside[-1]:
        settling_time = nan                       # still moving at the end: not settled
    else:
        settling_time = t[np.nonzero(outside)[0][-1] + 1]

    return dict(
        final=final,
        overshoot=max(q.max() - final, 0.0),      # rad above the final value (0 if none)
        settling_time=settling_time,
        max_torque=np.abs(tau).max(),             # peak motor effort
    )
