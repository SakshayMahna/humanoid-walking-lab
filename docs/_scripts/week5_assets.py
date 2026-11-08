"""Images and GIFs for the Week 5 blog post (classical control: LQR, iLQR, MPC, the one-leg G1).

Everything comes from the code in classical/01_dynamics_lqr_mpc/, imported unchanged: same models, same controllers.
Only the scenery is swapped for the blog's look (sky, floor), and neither takes part in the physics.

Run from the repo root:  .venv/bin/python docs/_scripts/week5_assets.py [clips] [charts] [banner] [name ...]
Outputs: docs/assets/images/week-5/ (intermediate data and frames in recordings/week5/).
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
CLASSICAL = REPO / "classical" / "01_dynamics_lqr_mpc"
sys.path[:0] = [str(REPO / "docs" / "_scripts"), str(CLASSICAL), str(REPO / "exercises")]

import mujoco  # noqa: E402

import cartpole  # noqa: E402
import g1_utils  # noqa: E402
from week3_assets import (C_AQUA, C_BLUE, C_ORANGE, C_YELLOW, SVG_GRID, SVG_MUTED, SVG_TEXT, mujoco_look,  # noqa: E402
                          svg_doc, svg_text)
from render_utils import make_camera, record_frames, side_by_side  # noqa: E402

OUT = REPO / "docs" / "assets" / "images" / "week-5"
WORK = REPO / "recordings" / "week5"
OUT.mkdir(parents=True, exist_ok=True)
WORK.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------------------------------------
# Scenery (visual only: the cart-pole's geoms have no contacts, the G1's floor plane is the same plane)
# ---------------------------------------------------------------------------------------------------------

_plain_cartpole_xml = cartpole.cartpole_xml
_plain_add_floor = g1_utils.add_floor


def styled_cartpole_xml(*args, **kwargs) -> str:
    """The cart-pole with the blog's navy sky and a checker floor below the rail."""
    xml = _plain_cartpole_xml(*args, **kwargs)
    asset = """<asset>
        <texture name="sky" type="skybox" builtin="gradient" rgb1="0.30 0.40 0.85" rgb2="0.03 0.03 0.09" width="512" height="512"/>
        <texture name="grid" type="2d" builtin="checker" rgb1="0.20 0.24 0.42" rgb2="0.11 0.13 0.25" width="512" height="512"/>
        <material name="grid" texture="grid" texrepeat="2 2" texuniform="true"/>
    </asset>"""
    xml = xml.replace("<worldbody>", asset + '<worldbody><geom name="floor" type="plane" size="60 60 0.1" pos="0 0 -0.75" material="grid"/>', 1)
    return xml.replace('rgba="0.5 0.5 0.5 1"', 'rgba="0.75 0.77 0.85 1"')


def styled_add_floor(spec) -> None:
    _plain_add_floor(spec)
    mujoco_look(spec, 1280, 720)


cartpole.cartpole_xml = styled_cartpole_xml
g1_utils.add_floor = styled_add_floor

# These import from the (patched) modules above
import ilqr as C  # noqa: E402
import jacobians as E  # noqa: E402
import lqr as B  # noqa: E402
import lqr_analysis as H  # noqa: E402
import mpc as D  # noqa: E402
import one_leg_lqr as F  # noqa: E402

for module in (B, C, D, E, F, H):
    module.results_dir = WORK         # the scripts' own plots go here, not over classical/.../results/
F.brighten = lambda model: None       # keep the blog's lighting

DT = cartpole.DT


# ---------------------------------------------------------------------------------------------------------
# Frames -> looping GIF, small enough for the blog
# ---------------------------------------------------------------------------------------------------------

def to_gif(frames, name: str, fps_in: float, fps: int = 12, width: int | None = None, speed: float = 1.0,
           colors: int = 96, dither: bool = True) -> None:
    """Looping GIF from PIL frames. width=None keeps the size: a clip is never enlarged."""
    gif = OUT / f"{name}.gif"
    width = min(width or frames[0].width, frames[0].width)
    with tempfile.TemporaryDirectory() as tmp:
        video = Path(tmp) / "clip.mp4"
        F.write_mp4(frames, video, fps=fps_in)
        use = "dither=bayer:bayer_scale=4" if dither else "dither=none"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(video), "-vf",
                        f"setpts=PTS/{speed},fps={fps},scale={width}:-2:flags=lanczos,split[a][b];"
                        f"[a]palettegen=max_colors={colors}:stats_mode=diff[p];"
                        f"[b][p]paletteuse={use}:diff_mode=rectangle", "-loop", "0", str(gif)], check=True)
    print(f"wrote {gif.name}: {gif.stat().st_size / 1e6:.1f} MB", flush=True)


# ---------------------------------------------------------------------------------------------------------
# Cart-pole clips
# ---------------------------------------------------------------------------------------------------------

def _cart_frames(make, s0, controller, duration, name):
    model, data = make()
    cartpole.set_state(model, data, s0)
    if isinstance(controller, D.MPCController):
        controller.start(s0)
    return record_frames(model, data, duration, camera=B.model_camera(model, "follow"), control=controller,
                         capture_fps=30, label=lambda d: f"{name}   u = {d.actuator_force[0]:+5.1f} N")


def clip_lqr_balance():
    A_, B_ = B.discretize(*B.linearize_continuous(), DT)
    K, _ = B.dlqr(A_, B_, B.Q, B.R)
    s0 = np.array([0.0, 0.2, 0.0, 0.0])
    frames = _cart_frames(cartpole.make_model, s0, B.make_lqr_controller(K), 4.0, "LQR, from 0.2 rad")
    to_gif(frames, "lqr_balance", 30, fps=15)


def clip_force_limits():
    A_, B_ = B.discretize(*B.linearize_continuous(), DT)
    K, _ = B.dlqr(A_, B_, B.Q, B.R)
    s0 = np.array([0.0, 0.5, 0.0, 0.0])
    panels = []
    for lim, name in ((None, "no limit"), (5.0, "limit 5 N"), (2.0, "limit 2 N")):
        make = lambda lim=lim: cartpole.make_model(lim, pole_stop=True)
        panels.append(_cart_frames(make, s0, B.make_lqr_controller(K), 3.5, name))
    to_gif(side_by_side(panels), "lqr_force_limits", 30, width=900, colors=64)


def clip_swingup():
    S, U, K, _ = C.ilqr(C.S_START, np.zeros(C.N), verbose=False)
    make = lambda: cartpole.make_model(C.U_MAX, pole_mass=1.2 * cartpole.M_POLE)
    panels = [_cart_frames(make, C.S_START, C.make_plan_controller(S, U, K, fb), C.T + 3.0, name)
              for fb, name in ((False, "plan only"), (True, "plan + gains"))]
    to_gif(side_by_side(panels), "ilqr_swingup", 30, width=800, colors=64)


def clip_mpc():
    S, U, K, _ = C.ilqr(C.S_START, np.zeros(C.N), verbose=False)
    make = lambda: cartpole.make_model(C.U_MAX, pole_mass=1.3 * cartpole.M_POLE)
    panels = [_cart_frames(make, C.S_START, c, 8.0, name) for c, name in
              ((C.make_plan_controller(S, U, K), "fixed plan"),
               (D.MPCController(warm_start="open"), "MPC, old forces"),
               (D.MPCController(), "MPC, old policy"))]
    to_gif(side_by_side(panels), "mpc_heavy_pole", 30, fps=10, width=900, colors=64)


# ---------------------------------------------------------------------------------------------------------
# G1 clips
# ---------------------------------------------------------------------------------------------------------

def clip_com_ik():
    """Part E5, run with the blog's scenery; its GIF is converted to a smaller one."""
    model, data, _, _ = g1_utils.build_g1_model(REPO / "robot" / "g1.xml")
    E.part_e5(model, data, {})
    with Image.open(WORK / "e5_com_ik.gif") as gif:
        frames = []
        for i in range(gif.n_frames):
            gif.seek(i)
            frames.append(gif.convert("RGB").copy())
    to_gif(frames, "null_space_squat", 33.3, fps=15, width=640)


def _flamingo_setup():
    model, data, joints = F.build_torque_g1()
    poses, foot, R_foot = F.make_poses(model, data, joints)
    return model, data, joints, dict(poses=poses, F=foot, R_foot=R_foot)


def clip_poses_push():
    model, data, joints, ctx = _flamingo_setup()
    cam = make_camera(lookat=[0, 0, 0.55], distance=2.6, azimuth=140, elevation=-10)
    torso = model.body("torso_link").id
    panels = []
    per_frame = round((1 / 30) / model.opt.timestep)
    with mujoco.Renderer(model, height=360, width=360) as renderer:
        for name, shown in (("flamingo", "Flamingo"), ("tree", "Tree"), ("warrior3", "Warrior III")):
            q0, u0, _ = F.equilibrium(model, data, ctx["poses"][name])
            A_, B_ = F.linearize(model, data, q0, u0)
            K = F.lqr_gain(A_, B_, *F.cost_matrices(model, data, joints, q0), own=False)
            mujoco.mj_resetData(model, data)
            data.qpos[:] = q0
            mujoco.mj_forward(model, data)
            frames = []
            for i in range(round(3.0 / model.opt.timestep)):
                F.lqr_control(model, data, q0, u0, K)
                data.xfrc_applied[:] = 0.0
                pushing = 1.0 <= data.time < 1.1
                if pushing:
                    data.xfrc_applied[torso, :3] = [0, 30.0, 0]
                mujoco.mj_step(model, data)
                if i % per_frame == 0:
                    renderer.update_scene(data, camera=cam)
                    frames.append(F.label(Image.fromarray(renderer.render()), f"{shown}{'   <- 30 N push' if pushing else ''}"))
            panels.append(frames)
    to_gif(side_by_side(panels), "yoga_push", 1 / (per_frame * model.opt.timestep), fps=15, width=900)


def clip_flow():
    model, data, joints, ctx = _flamingo_setup()
    chain = F.build_flow_chain(model, data, joints, ctx)
    chain["sequence"] = ["flamingo", "tree", "warrior III", "flamingo"]      # only used for the clip's labels here
    fell, worst, moved, frames = F.run_flow(model, data, chain, record=True)
    print(f"flow: fell {fell}, largest joint error {worst:.3f} rad, foot moved {moved:.2f} mm")
    per_frame = round((1 / 30) / model.opt.timestep)
    to_gif(frames, "yoga_flow", 1 / (per_frame * model.opt.timestep), fps=12, speed=1.5, width=640)


def clip_gain_schedule():
    """The same flow with blended gains and with flamingo's gain throughout, until shortly after the second falls."""
    model, data, joints, ctx = _flamingo_setup()
    chain = F.build_flow_chain(model, data, joints, ctx)
    chain["sequence"] = ["flamingo", "tree", "warrior III", "flamingo"]
    one_gain = lambda s: chain["K"][0]
    fell, *_ = F.run_flow(model, data, chain, gains=one_gain)
    t_fall = data.time                                  # run_flow stops at the fall when it isn't recording
    assert fell
    print(f"flamingo's gain falls at t = {t_fall:.2f} s")
    per_frame = round((1 / 30) / model.opt.timestep)
    fps_in = 1 / (per_frame * model.opt.timestep)
    panels = []
    for name, gains in (("blended gains", None), ("flamingo's gain only", one_gain)):
        *_, frames = F.run_flow(model, data, chain, gains=gains, record=True)
        frames = frames[:round((t_fall + 1.5) * fps_in)]
        for frame in frames:
            ImageDraw.Draw(frame).text((10, 34), name, fill=(255, 255, 255), font=ImageFont.load_default(size=22))
        panels.append(frames)
    to_gif(side_by_side(panels), "gain_schedule", fps_in, fps=12, speed=1.5, width=900, colors=80)


def clip_modes():
    model, data, joints, ctx = H.setup()
    H.part_h1(model, data, joints, ctx)
    H.part_h3(model, data, joints, ctx)
    H.mode_video(model, data, ctx)
    with Image.open(WORK / "h_modes.gif") as gif:
        frames = []
        for i in range(gif.n_frames):
            gif.seek(i)
            frames.append(gif.convert("RGB").copy())
    to_gif(frames, "modes", 20, fps=15, width=900)


CLIPS = {"lqr_balance": clip_lqr_balance, "force_limits": clip_force_limits, "swingup": clip_swingup,
         "mpc": clip_mpc, "com_ik": clip_com_ik, "poses_push": clip_poses_push, "flow": clip_flow,
         "gain_schedule": clip_gain_schedule, "modes": clip_modes}


# ---------------------------------------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------------------------------------

def save_svg(name: str, width: int, height: int, body: list) -> None:
    (OUT / f"{name}.svg").write_text(svg_doc(width, height, "\n".join(body)))
    print(f"saved {name}.svg", flush=True)


def polyline(xs, ys, color, width=2, dash=None) -> str:
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round"{extra}/>'


class Axes:
    """A plot area with recessive grid lines: maps data to pixels and draws its own ticks."""

    def __init__(self, x0, y0, x1, y1, xlim, ylim):
        self.x0, self.y0, self.x1, self.y1, self.xlim, self.ylim = x0, y0, x1, y1, xlim, ylim

    def x(self, v):
        return self.x0 + (np.asarray(v, float) - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * (self.x1 - self.x0)

    def y(self, v):
        return self.y1 - (np.asarray(v, float) - self.ylim[0]) / (self.ylim[1] - self.ylim[0]) * (self.y1 - self.y0)

    def grid(self, xticks, yticks, xlabel="", ylabel="", fmt="{:g}") -> list:
        b = []
        for t in yticks:
            y = self.y(t)
            b.append(f'<line x1="{self.x0}" y1="{y:.1f}" x2="{self.x1}" y2="{y:.1f}" stroke="{SVG_GRID}" stroke-width="1"/>')
            b.append(svg_text(self.x0 - 8, y + 4, fmt.format(t), 12, SVG_MUTED, "end"))
        for t in xticks:
            x = self.x(t)
            b.append(f'<line x1="{x:.1f}" y1="{self.y0}" x2="{x:.1f}" y2="{self.y1}" stroke="{SVG_GRID}" stroke-width="1"/>')
            b.append(svg_text(x, self.y1 + 18, fmt.format(t), 12, SVG_MUTED, "middle"))
        if xlabel:
            b.append(svg_text((self.x0 + self.x1) / 2, self.y1 + 38, xlabel, 13, SVG_MUTED, "middle"))
        if ylabel:
            b.append(svg_text(self.x0, self.y0 - 10, ylabel, 13, SVG_MUTED))
        return b


def legend(items, x, y) -> list:
    """Coloured mark + text in the text colour. items: (name, color, dashed)."""
    b = []
    for name, color, dashed in items:
        dash = ' stroke-dasharray="6,4"' if dashed else ""
        b.append(f'<line x1="{x}" y1="{y - 4}" x2="{x + 22}" y2="{y - 4}" stroke="{color}" stroke-width="2.5"{dash}/>')
        b.append(svg_text(x + 28, y, name, 13))
        x += 28 + 7.4 * len(name) + 24
    return b


def grid_cells(x0, y0, cell_w, cell_h, on, on_color=C_BLUE, off_color=SVG_GRID, gap=1.5) -> list:
    """Boolean matrix as rounded cells; row 0 at the TOP."""
    b = []
    for i, row in enumerate(on):
        for j, v in enumerate(row):
            b.append(f'<rect x="{x0 + j * cell_w:.1f}" y="{y0 + i * cell_h:.1f}" width="{cell_w - gap:.1f}" '
                     f'height="{cell_h - gap:.1f}" rx="1.5" fill="{on_color if v else off_color}"/>')
    return b


# ---------------------------------------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------------------------------------

def chart_riccati():
    """Each Riccati step adds one step of look-ahead; the gains settle to LQR's constant K."""
    A_, B_ = B.discretize(*B.linearize_continuous(), DT)
    K_inf, _ = B.dlqr(A_, B_, B.Q, B.R)
    P, gains = B.Q.copy(), []
    steps = 500
    for _ in range(steps):
        K = np.linalg.solve(B.R + B_.T @ P @ B_, B_.T @ P @ A_)
        gains.append(K[0])
        A_cl = A_ - B_ @ K
        P = B.Q + K.T @ B.R @ K + A_cl.T @ P @ A_cl
    gains = np.array(gains) / K_inf[0]
    print("K inf", np.round(K_inf[0], 3), "gain ratio range", gains.min().round(2), gains.max().round(2))
    t = np.arange(steps) * DT
    W, H_ = 900, 400
    ax = Axes(70, 90, W - 30, H_ - 60, (0, t[-1]), (-0.2, max(1.2, float(np.ceil(gains.max() * 5) / 5))))
    b = [svg_text(30, 34, "The Riccati iteration: gains as the look-ahead grows", 19, SVG_TEXT, weight="bold")]
    series = [("cart position", 0, C_BLUE), ("pole angle", 1, C_ORANGE), ("cart speed", 2, C_AQUA), ("pole speed", 3, C_YELLOW)]
    b += legend([(n, c, False) for n, _, c in series], 30, 62)
    yticks = np.arange(0.0, ax.ylim[1] + 1e-9, 0.5)
    b += ax.grid([0, 1, 2, 3, 4, 5], yticks, "look-ahead (seconds of future the value function has seen)", "gain / final LQR gain")
    for _, i, c in series:
        b.append(polyline(ax.x(t), ax.y(gains[:, i]), c))
    save_svg("riccati_gains", W, H_, b)


def chart_force_limits():
    data_file = WORK / "regions_b7.npz"
    thetas, theta_dots = np.linspace(-1.4, 1.4, 29), np.linspace(-4.0, 4.0, 33)
    if data_file.exists():
        grids = dict(np.load(data_file))
    else:
        A_, B_ = B.discretize(*B.linearize_continuous(), DT)
        K, _ = B.dlqr(A_, B_, B.Q, B.R)
        grids = {str(lim): B.region_of_attraction(K, thetas, theta_dots, lim) for lim in (None, 5.0, 2.0)}
        np.savez(data_file, **grids)
    panels = (("None", "no limit"), ("5.0", "limit 5 N"), ("2.0", "limit 2 N"))
    print("B7 counts:", {k: int(grids[k].sum()) for k, _ in panels}, "of", grids["None"].size)
    W, H_ = 1000, 400
    cw, ch = 8.2, 9.2
    b = [svg_text(30, 34, "Starts LQR recovers from, same gain, three force limits", 19, SVG_TEXT, weight="bold")]
    for p, (key, name) in enumerate(panels):
        x0, y0 = 70 + p * (33 * cw + 34), 80
        g = grids[key][::-1]                                        # largest angle at the top
        b += grid_cells(x0, y0, cw, ch, g, gap=1)
        b.append(svg_text(x0, y0 - 10, f"{name}: {int(g.sum())} of {g.size}", 14, SVG_TEXT))
        for t in (-4, 0, 4):
            b.append(svg_text(x0 + (t + 4) / 8 * 32 * cw + cw / 2, y0 + 29 * ch + 16, f"{t:g}", 12, SVG_MUTED, "middle"))
        b.append(svg_text(x0 + 16 * cw, y0 + 29 * ch + 34, "starting pole speed (rad/s)", 12, SVG_MUTED, "middle"))
        if p == 0:
            for th in (1.4, 0, -1.4):
                b.append(svg_text(x0 - 8, y0 + (1.4 - th) / 2.8 * 28 * ch + ch / 2 + 4, f"{th:g}", 12, SVG_MUTED, "end"))
            b.append(svg_text(30, y0 - 10, "rad", 12, SVG_MUTED))
    save_svg("lqr_force_limits_map", W, H_, b)


def chart_swingup():
    S, U, _, costs = C.ilqr(C.S_START, np.zeros(C.N), verbose=False)
    print(f"swing-up: {len(costs) - 1} iterations, cost {costs[0]:.2f} -> {costs[-1]:.4f}, "
          f"at the limit {int(np.sum(np.abs(U) > C.U_MAX - 1e-6))} of {C.N} steps")
    t = np.arange(C.N + 1) * DT
    W, H_ = 900, 340
    a1 = Axes(70, 70, 440, H_ - 60, (0, 3), (-0.5, 5.2))
    a2 = Axes(520, 70, W - 30, H_ - 60, (0, 3), (-6, 6))
    b = [svg_text(30, 34, "The swing-up iLQR plans, with a 5 N limit", 19, SVG_TEXT, weight="bold")]
    b += a1.grid([0, 1, 2, 3], [0, 3.14], "time (s)", "pole angle (rad): 0 is upright, 3.14 is hanging", "{:g}")
    b.append(polyline(a1.x(t), a1.y(S[:, 1]), C_BLUE))
    b += a2.grid([0, 1, 2, 3], [-5, 0, 5], "time (s)", "force on the cart (N)")
    b.append(polyline(a2.x(t[:-1]), a2.y(U), C_ORANGE))
    save_svg("ilqr_swingup_plan", W, H_, b)


def chart_lqr_vs_mpc():
    data_file = WORK / "regions_d4.npz"
    thetas, theta_dots = np.linspace(-1.2, 1.2, 9), np.linspace(-4.0, 4.0, 9)
    if data_file.exists():
        grids = dict(np.load(data_file))
    else:
        U_LIM = 2.0
        A_, B_ = B.discretize(*B.linearize_continuous(), DT)
        K, _ = B.dlqr(A_, B_, B.Q, B.R)

        def lqr_controller(model, data):
            data.ctrl[0] = np.clip(-(K @ cartpole.get_state(data))[0], -U_LIM, U_LIM)
        grids = {}
        for name in ("LQR", "MPC"):
            grid = np.zeros((9, 9), dtype=bool)
            for i, th in enumerate(thetas):
                for j, thd in enumerate(theta_dots):
                    model, data = cartpole.make_model(U_LIM)
                    c = lqr_controller if name == "LQR" else D.MPCController(horizon=1.0, replan=10, iters=2, u_max=U_LIM)
                    S, _ = D.run(model, data, np.array([0.0, th, 0.0, thd]), c, duration=5.0)
                    grid[i, j] = D.held_upright(S, last=1.0)
            grids[name] = grid
        np.savez(data_file, **grids)
    W, H_ = 800, 390
    c = 30
    b = [svg_text(30, 34, "Same 2 N limit: LQR ignores it, MPC plans with it", 19, SVG_TEXT, weight="bold")]
    for p, name in enumerate(("LQR", "MPC")):
        x0, y0 = 90 + p * (9 * c + 70), 80
        g = grids[name][::-1]
        b += grid_cells(x0, y0, c, c, g, gap=2)
        b.append(svg_text(x0, y0 - 10, f"{name}: {int(g.sum())} of {g.size} starts", 14, SVG_TEXT))
        for t in (-4, 0, 4):
            b.append(svg_text(x0 + (t + 4) / 8 * 8 * c + c / 2, y0 + 9 * c + 16, f"{t:g}", 12, SVG_MUTED, "middle"))
        b.append(svg_text(x0 + 4.5 * c, y0 + 9 * c + 34, "starting pole speed (rad/s)", 12, SVG_MUTED, "middle"))
        if p == 0:
            for th in (1.2, 0, -1.2):
                b.append(svg_text(x0 - 8, y0 + (1.2 - th) / 2.4 * 8 * c + c / 2 + 4, f"{th:g}", 12, SVG_MUTED, "end"))
            b.append(svg_text(40, y0 - 10, "rad", 12, SVG_MUTED))
    save_svg("lqr_vs_mpc_region", W, H_, b)
    print("D4:", {k: int(v.sum()) for k, v in grids.items()})


def chart_null_space():
    """Two joints, one hard requirement, one task: the formula from the post on numbers small enough to draw."""
    C_, b_ = np.array([[1.0, 1.0]]), np.array([1.0])     # hard requirement: qd1 + qd2 = 1
    J, task = np.array([[1.0, 0.0]]), np.array([1.5])    # task: qd1 = 1.5
    C_pinv = np.linalg.pinv(C_)
    v0 = C_pinv @ b_
    N = np.eye(2) - C_pinv @ C_
    v = v0 + N @ np.linalg.pinv(J @ N) @ (task - J @ v0)
    assert np.allclose(C_ @ v, b_) and np.allclose(J @ v, task)
    print("null space example: v0", v0, "v", v, "N", N.round(3).tolist())
    W, H_ = 760, 470
    ax = Axes(90, 60, W - 150, H_ - 60, (-0.75, 2.25), (-1, 1.5))
    b = [svg_text(30, 34, "Null-space projection, for two joints", 19, SVG_TEXT, weight="bold")]
    b += ax.grid([-0.5, 0, 0.5, 1, 1.5, 2], [-1, -0.5, 0, 0.5, 1, 1.5], "speed of joint 1", "speed of joint 2")
    # the two lines
    x = np.array([-0.5, 2.0])
    b.append(polyline(ax.x(x), ax.y(1 - x), C_BLUE, 2.5))
    b.append(polyline(ax.x([1.5, 1.5]), ax.y([-1, 1.5]), C_ORANGE, 2.5, "7,5"))
    b.append(svg_text(ax.x(0.32), ax.y(1.22), "hard requirement:", 13))
    b.append(svg_text(ax.x(0.32), ax.y(1.22) + 17, "every point on this line meets it", 13))
    b.append(svg_text(ax.x(1.5) + 8, ax.y(1.35), "task", 13))
    # arrows
    b.append(f'<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
             f'<path d="M0,0 L10,5 L0,10 z" fill="{SVG_TEXT}"/></marker></defs>')
    def arrow(p, q):
        return f'<line x1="{ax.x(p[0]):.1f}" y1="{ax.y(p[1]):.1f}" x2="{ax.x(q[0]):.1f}" y2="{ax.y(q[1]):.1f}" ' \
               f'stroke="{SVG_TEXT}" stroke-width="2" marker-end="url(#arrow)"/>'
    b += [arrow((0, 0), v0), arrow(v0, v)]
    b.append(svg_text(ax.x(-0.68), ax.y(0.62), "v0: the smallest speeds", 13))
    b.append(svg_text(ax.x(-0.68), ax.y(0.62) + 17, "that meet it", 13))
    b.append(svg_text(ax.x(0.32), ax.y(-0.32), "along the null space", 13))
    for p, name, color in (((0, 0), "no motion", SVG_MUTED), (v, "v: meets both", C_ORANGE)):
        b.append(f'<circle cx="{ax.x(p[0]):.1f}" cy="{ax.y(p[1]):.1f}" r="5" fill="{color}" stroke="#121421" stroke-width="2"/>')
        b.append(svg_text(ax.x(p[0]) - 10, ax.y(p[1]) + 22, name, 13, anchor="end"))
    save_svg("null_space", W, H_, b)


def chart_jacobian_structure():
    model, data, _, _ = g1_utils.build_g1_model(REPO / "robot" / "g1.xml")
    mujoco.mj_forward(model, data)
    J_l, J_r = E.site_jacobian(model, data, "left_foot"), E.site_jacobian(model, data, "right_foot")
    _, J_c = E.com_and_jacobian(model, data)
    rows = [("left foot: speed", J_l[:3]), ("left foot: turning", J_l[3:]), ("right foot: speed", J_r[:3]),
            ("right foot: turning", J_r[3:]), ("centre of mass", J_c)]
    names = [model.joint(i).name for i in range(1, model.njnt)]
    groups = [("base", 6)]
    for prefix, label in (("left_hip|left_knee|left_ankle", "left leg"), ("right_hip|right_knee|right_ankle", "right leg"),
                          ("waist", "waist"), ("left_shoulder|left_elbow|left_wrist", "left arm"),
                          ("right_shoulder|right_elbow|right_wrist", "right arm")):
        n = sum(n_.startswith(tuple(prefix.split("|"))) for n_ in names)
        groups.append((label, n))
    assert sum(n for _, n in groups) == model.nv
    print("groups", groups, "first joints", names[:7], "| waist at", [i for i, n_ in enumerate(names) if n_.startswith("waist")])
    print("left foot moves", int(np.sum(np.abs(J_l).max(axis=0) > 1e-12)), "columns; right foot",
          int(np.sum(np.abs(J_r).max(axis=0) > 1e-12)), "; CoM", int(np.sum(np.abs(J_c).max(axis=0) > 1e-12)))
    c, top, left = 17.5, 96, 170
    W, H_ = int(left + model.nv * c + 40 + 18), top + 15 * c + 34 + 8 * 5 + 16
    b = [svg_text(30, 34, "Which velocities can move what: the G1's Jacobians", 19, SVG_TEXT, weight="bold")]
    x = left
    for label, n in groups:
        b.append(svg_text(x + n * c / 2, top - 10, label, 12.5, SVG_TEXT, "middle"))
        x += n * c + 3
    y = top
    for label, J in rows:
        b.append(svg_text(left - 10, y + J.shape[0] * c / 2 + 4, label, 12.5, SVG_TEXT, "end"))
        x = left
        col = 0
        for _, n in groups:
            b += grid_cells(x, y, c, c, np.abs(J[:, col:col + n]) > 1e-12, gap=1.5)
            x += n * c + 3
            col += n
        y += J.shape[0] * c + 8
    b.append(f'<rect x="{left}" y="{y + 2}" width="13" height="13" rx="2" fill="{C_BLUE}"/>')
    b.append(svg_text(left + 20, y + 13, "this velocity moves this quantity", 12.5))
    b.append(f'<rect x="{left + 270}" y="{y + 2}" width="13" height="13" rx="2" fill="{SVG_GRID}"/>')
    b.append(svg_text(left + 290, y + 13, "exactly zero", 12.5))
    save_svg("jacobian_structure", W, int(H_), b)


def chart_eigenvalues():
    model, data, joints, ctx = H.setup()
    A_, B_, K, dt = ctx["A"], ctx["B"], ctx["K"], model.opt.timestep
    panels = (("Without feedback", np.linalg.eigvals(A_)), ("With LQR", np.linalg.eigvals(A_ - B_ @ K)))
    W, H_ = 1000, 410
    b = [svg_text(30, 34, "The flamingo's 70 modes: how fast each one grows or dies away", 19, SVG_TEXT, weight="bold")]
    b += [f'<circle cx="36" cy="58" r="5" fill="{C_ORANGE}"/>', svg_text(48, 63, "grows", 13),
          f'<circle cx="116" cy="58" r="5" fill="{C_BLUE}"/>', svg_text(128, 63, "dies away, or stays", 13)]
    for p, (name, eig) in enumerate(panels):
        s = np.log(eig.astype(complex)) / dt
        grows = np.abs(eig) > 1.001
        print(f"{name}: {int(grows.sum())} growing; fastest {s.real.max():.1f} /s; "
              f"most negative {s.real.min():.1f} /s; largest oscillation {np.abs(s.imag).max() / (2 * np.pi):.1f} Hz")
        ax = Axes(80 + p * 470, 105, 460 + p * 470, H_ - 60, (-130, 30), (-10, 10))
        b.append(svg_text(ax.x0, 95, name + f": {int(grows.sum())} modes grow", 14, SVG_TEXT))
        b += ax.grid([-120, -80, -40, 0, 30], [-10, 0, 10], "growth rate (1/s): right of 0 grows, left dies away")
        b.append(f'<line x1="{ax.x(0):.1f}" y1="{ax.y0}" x2="{ax.x(0):.1f}" y2="{ax.y1}" stroke="{SVG_MUTED}" stroke-width="1.5"/>')
        if p == 0:
            b.append(svg_text(30, 95, "Hz", 12, SVG_MUTED))
        for g, si in sorted(zip(grows, s), key=lambda z: z[0]):
            y = np.clip(si.imag / (2 * np.pi), -10, 10)
            b.append(f'<circle cx="{ax.x(si.real):.1f}" cy="{ax.y(y):.1f}" r="4.5" fill="{C_ORANGE if g else C_BLUE}" '
                     f'stroke="#121421" stroke-width="1.5"/>')
    save_svg("eigenvalues", W, H_, b)


def _capture_figure(run):
    """Run a function that builds a matplotlib figure and closes it; returns that figure instead."""
    figures, close = [], H.plt.close
    H.plt.close = figures.append
    try:
        run()
    finally:
        H.plt.close = close
    return figures[-1]


def chart_linear_vs_mujoco():
    model, data, joints, ctx = H.setup()
    fig = _capture_figure(lambda: H.part_h6(model, data, joints, ctx))
    lines = fig.axes[0].lines                      # solid MuJoCo, dashed linear, per lean: 0.5, 1, 2, 3, 4 cm
    W, H_ = 900, 400
    ax = Axes(70, 90, W - 30, H_ - 60, (0, 1.5), (-1, 5))
    b = [svg_text(30, 34, "Lean the robot sideways and let LQR recover", 19, SVG_TEXT, weight="bold")]
    b += legend([("1 cm", C_BLUE, False), ("2 cm", C_ORANGE, False), ("4 cm", C_AQUA, False),
                 ("the linear model's prediction", SVG_MUTED, True)], 30, 62)
    b += ax.grid([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5], [-1, 0, 1, 2, 3, 4, 5], "time (s)", "CoM relative to the foot, sideways (cm)")
    for i, color in ((1, C_BLUE), (2, C_ORANGE), (4, C_AQUA)):
        sim, lin = lines[2 * i], lines[2 * i + 1]
        t = sim.get_xdata()
        b.append(polyline(ax.x(t), ax.y(lin.get_ydata()), color, 1.8, "6,4"))
        b.append(polyline(ax.x(t), ax.y(sim.get_ydata()), color, 2.2))
    save_svg("linear_vs_mujoco", W, H_, b)


def chart_cross_stability():
    model, data, joints, ctx = H.setup()
    fig = _capture_figure(lambda: H.part_h7(model, data, joints, ctx))
    rho = np.asarray(fig.axes[0].images[0].get_array()) + 1          # [gain i, pose j], every 3rd pose of 61
    stable = rho < 1.0005
    n = stable.shape[0]
    c = 19
    W, H_ = 740, 90 + n * c + 70
    x0, y0 = 150, 70
    b = [svg_text(30, 34, "Which gain can hold which pose (linear model)", 19, SVG_TEXT, weight="bold")]
    b += grid_cells(x0, y0, c, c, stable[::-1], on_color=C_BLUE, off_color=C_ORANGE, gap=1.5)
    ticks = ((0, "flamingo"), (20, "tree"), (40, "warrior III"), (60, "flamingo"))
    for pose, name in ticks:
        k = pose / 3
        b.append(svg_text(x0 + k * c + c / 2, y0 + n * c + 16, name, 12.5, SVG_TEXT, "middle"))
        b.append(svg_text(x0 - 10, y0 + (n - 1 - k) * c + c / 2 + 4, name, 12.5, SVG_TEXT, "end"))
    b.append(svg_text(x0 + n * c / 2, y0 + n * c + 36, "the pose it has to hold", 12.5, SVG_MUTED, "middle"))
    b.append(svg_text(30, y0 - 10, "the gain designed for", 12.5, SVG_MUTED))
    lx = x0 + n * c + 24
    b.append(f'<rect x="{lx}" y="{y0}" width="13" height="13" rx="2" fill="{C_BLUE}"/>')
    b.append(svg_text(lx + 20, y0 + 11, "nothing grows", 12.5))
    b.append(f'<rect x="{lx}" y="{y0 + 24}" width="13" height="13" rx="2" fill="{C_ORANGE}"/>')
    b.append(svg_text(lx + 20, y0 + 35, "unstable", 12.5))
    save_svg("cross_stability", W, H_, b)
    print(f"stable pairs: {stable.mean():.0%}, {int(stable.sum())} of {stable.size}")


CHARTS = {"riccati": chart_riccati, "force_map": chart_force_limits, "plan": chart_swingup, "lqr_mpc": chart_lqr_vs_mpc,
          "null": chart_null_space, "jac": chart_jacobian_structure, "eig": chart_eigenvalues,
          "lin": chart_linear_vs_mujoco, "cross": chart_cross_stability}


# ---------------------------------------------------------------------------------------------------------
# Banner: from standing on two feet to warrior III, as ghosts (the Week 3-4 camera: level and side-on)
# ---------------------------------------------------------------------------------------------------------

def banner(preview_dir=None):
    W, H_ = 1280, 400
    model, data, joints = F.build_torque_g1()
    home = data.qpos.copy()
    poses, *_ = F.make_poses(model, data, joints)
    sequence = [("stand", home, 0.46, 0.5), ("flamingo", poses["flamingo"], 0.60, 0.6),
                ("tree", poses["tree"], 0.73, 0.75), ("warrior3", poses["warrior3"], 0.90, 1.0)]
    floor = model.geom("floor").id
    cam = make_camera(lookat=[0.0, 0.0, 0.62], distance=3.8, azimuth=90, elevation=-4)
    model.vis.global_.offwidth, model.vis.global_.offheight = W, H_
    with mujoco.Renderer(model, height=H_, width=W) as r:
        data.qpos[:] = home
        data.qpos[0] += 100.0
        mujoco.mj_forward(model, data)
        r.update_scene(data, camera=cam)
        img = np.asarray(r.render()).astype(float)
        for name, q, x, alpha in sequence:
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            cam.lookat[0] = data.qpos[0]
            r.update_scene(data, camera=cam)
            shot = np.asarray(r.render()).astype(float)
            r.enable_segmentation_rendering()
            seg = r.render()
            r.disable_segmentation_rendering()
            mask = (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (seg[..., 0] != floor)
            dx = round((x - 0.5) * W)
            shot, mask = np.roll(shot, dx, axis=1), np.roll(mask, dx, axis=1)
            if dx > 0:
                mask[:, :dx] = False
            img = np.where(mask[..., None], img * (1 - alpha) + shot * alpha, img)
    out = Image.fromarray(img.clip(0, 255).astype(np.uint8))
    out.save(OUT / "banner.png")
    print("saved banner.png")
    if preview_dir:
        prev = Image.fromarray((img * 0.6).clip(0, 255).astype(np.uint8))
        d = ImageDraw.Draw(prev)
        d.text((60, 140), "Week 5 - Classical Control for Humanoid Robots", fill=(250, 250, 255), font=ImageFont.load_default(size=44))
        d.text((60, 205), "LQR, iLQR and MPC on a cart-pole, then LQR on the whole Unitree G1,\nbalancing on one foot.",
               fill=(225, 225, 235), font=ImageFont.load_default(size=19))
        prev.save(Path(preview_dir) / "banner_week5_preview.png")


if __name__ == "__main__":
    args = sys.argv[1:] or ["clips", "charts", "banner"]
    preview = next((a.split("=", 1)[1] for a in args if a.startswith("preview=")), None)
    for a in args:
        if a == "clips":
            for fn in CLIPS.values():
                fn()
        elif a == "charts":
            for fn in CHARTS.values():
                fn()
        elif a == "banner":
            banner(preview)
        elif a in CLIPS:
            CLIPS[a]()
        elif a in CHARTS:
            CHARTS[a]()
        elif not a.startswith("preview="):
            raise SystemExit(f"unknown target {a}")
