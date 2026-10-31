"""Cart-pole model and simulation helpers for Module 1.

Conventions (match these in your derivation):
    state s = [x, theta, x_dot, theta_dot]
    x      cart position (m), +x to the right
    theta  pole angle from UPRIGHT (rad); theta > 0 tilts the tip towards +x
    u      horizontal force on the cart (N), +x

Pole: massless rod of length L with a point mass M_POLE at the tip (no rotational inertia about the tip).
Underactuated: u acts on x only.
"""

from typing import Callable

import mujoco
import numpy as np

M_CART = 1.0    # kg
M_POLE = 0.3    # kg, point mass at the tip
L = 0.5         # m, hinge to tip
G = 9.81        # m/s², magnitude
DT = 0.01       # s, physics step = control step (RK4, so one step is accurate)

Controller = Callable[[mujoco.MjModel, mujoco.MjData], None]


def cartpole_xml(force_limit: float | None = None, pole_mass: float = M_POLE, pole_stop: bool = False) -> str:
    """MJCF string.

    force_limit (N), if given, clamps the cart force to ±force_limit. pole_mass overrides the tip mass (to make the
    simulated plant differ from the model a controller was designed with). pole_stop adds a joint stop at ±100°
    (MJCF ranges are in degrees) so a fallen pole lies flat instead of spinning round in videos; leave it off for
    anything that swings through the bottom.
    """
    limit = f'ctrllimited="true" ctrlrange="{-force_limit} {force_limit}"' if force_limit else ""
    stop = 'range="-100 100"' if pole_stop else ""
    return f"""
    <mujoco model="cartpole">
        <option timestep="{DT}" integrator="RK4" gravity="0 0 {-G}"/>
        <visual><global offwidth="1280" offheight="720"/></visual>
        <default><geom contype="0" conaffinity="0"/></default>  <!-- no contacts anywhere -->
        <worldbody>
            <light directional="true" pos="0 -2 3" dir="0 1 -1.5"/>  <!-- directional: lights the cart anywhere on the rail -->
            <geom name="rail" type="box" size="100 0.01 0.01" pos="0 0 -0.05" rgba="0.5 0.5 0.5 1"/>
            <camera name="fixed" pos="0.15 -2.2 0.25" xyaxes="1 0 0 0 0 1"/>  <!-- world-fixed: shows cart travel -->
            <body name="cart">
                <joint name="slider" type="slide" axis="1 0 0"/>
                <inertial pos="0 0 0" mass="{M_CART}" diaginertia="0.01 0.01 0.01"/>
                <geom type="box" size="0.12 0.06 0.04" rgba="0.22 0.53 0.9 1"/>
                <camera name="follow" mode="track" pos="0 -2.5 0.25" xyaxes="1 0 0 0 0 1"/>  <!-- follows the cart -->
                <body name="pole">
                    <joint name="hinge" type="hinge" axis="0 1 0" {stop}/>  <!-- +theta: tip towards +x -->
                    <inertial pos="0 0 {L}" mass="{pole_mass}" diaginertia="1e-9 1e-9 1e-9"/>  <!-- point mass -->
                    <geom type="capsule" fromto="0 0 0 0 0 {L}" size="0.012" rgba="0.85 0.35 0.15 1"/>
                    <geom type="sphere" pos="0 0 {L}" size="0.04" rgba="0.85 0.35 0.15 1"/>
                </body>
            </body>
        </worldbody>
        <actuator>
            <motor name="push" joint="slider" {limit}/>  <!-- ctrl = force on the cart (N) -->
        </actuator>
    </mujoco>
    """


def make_model(force_limit: float | None = None, pole_mass: float = M_POLE,
               pole_stop: bool = False) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_string(cartpole_xml(force_limit, pole_mass, pole_stop))
    return model, mujoco.MjData(model)


def get_state(data: mujoco.MjData) -> np.ndarray:
    """s = [x, theta, x_dot, theta_dot] (a copy)."""
    return np.concatenate([data.qpos, data.qvel]).copy()


def set_state(model: mujoco.MjModel, data: mujoco.MjData, s, u: float = 0.0) -> None:
    """Reset, then set state and control, and compute derived quantities (qacc etc.)."""
    mujoco.mj_resetData(model, data)
    data.qpos[:] = s[:2]
    data.qvel[:] = s[2:]
    data.ctrl[0] = u
    mujoco.mj_forward(model, data)


def simulate(model, data, s0, controller: Controller | None, duration: float):
    """Run from state s0. controller(model, data) sets data.ctrl before every step.

    Returns (t, S, U, fell): times (N,), states (N+1, 4) including s0, applied forces (N,) after any clamping,
    and fell = True if |theta| passed pi/2 (the run stops there).
    """
    set_state(model, data, s0)
    t, S, U = [], [get_state(data)], []
    for _ in range(round(duration / model.opt.timestep)):
        if controller is not None:
            controller(model, data)
        mujoco.mj_step(model, data)
        t.append(data.time)
        S.append(get_state(data))
        U.append(data.actuator_force[0])
        if abs(data.qpos[1]) > np.pi / 2:
            return np.array(t), np.array(S), np.array(U), True
    return np.array(t), np.array(S), np.array(U), False


def step_jacobians(model, data, s, u: float, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """A = d s_next / d s, B = d s_next / d u of one MuJoCo step at (s, u), by central differences.

    (mujoco.mjd_transitionFD does this for the other integrators, but not RK4.)
    """
    def step(s_, u_):
        set_state(model, data, s_, u_)
        mujoco.mj_step(model, data)
        return get_state(data)

    s = np.asarray(s, dtype=float)
    A, B = np.zeros((4, 4)), np.zeros((4, 1))
    for i in range(4):
        ds = np.zeros(4)
        ds[i] = eps
        A[:, i] = (step(s + ds, u) - step(s - ds, u)) / (2 * eps)
    B[:, 0] = (step(s, u + eps) - step(s, u - eps)) / (2 * eps)
    return A, B
