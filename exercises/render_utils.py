"""Rendering helpers shared by the exercises."""

from pathlib import Path
from typing import Callable

import mujoco
from PIL import Image, ImageDraw, ImageFont


def make_camera(lookat, distance, azimuth=90.0, elevation=-20.0) -> mujoco.MjvCamera:
    """Free camera looking at `lookat` (m) from `distance` (m). Angles in degrees."""
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


def render_image(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    path: Path | None = None,
    camera: mujoco.MjvCamera | None = None,
    width: int = 640,
    height: int = 480,
    show: bool = False,
) -> Image.Image:
    """Render the current state without stepping. Optionally save and/or show it."""
    mujoco.mj_forward(model, data)  # make body poses consistent with qpos
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        renderer.update_scene(data, camera=camera if camera is not None else -1)  # -1 = default free camera
        image = Image.fromarray(renderer.render())
    if path is not None:
        image.save(path)
    if show:
        image.show()
    return image


def record_gif(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    path: Path,
    duration: float,
    camera: mujoco.MjvCamera | None = None,
    control: Callable[[mujoco.MjModel, mujoco.MjData], None] | None = None,
    capture_fps: int = 60,
    playback_fps: int | None = None,
    width: int = 640,
    height: int = 480,
    label: str | None = None,
) -> int:
    """Simulate `duration` seconds from the current state and save a GIF.

    control(model, data), if given, runs before every mj_step (e.g. to set data.ctrl).
    playback_fps < capture_fps gives slow motion; defaults to real time.
    label, if given, is drawn in the top-left corner of every frame.
    Returns the number of frames saved.
    """
    playback_fps = playback_fps or capture_fps
    font = ImageFont.load_default(size=height // 20)
    # Count steps instead of looping on data.time: if the simulation blows up,
    # MuJoCo auto-resets data (time back to 0) and a time-based loop never ends.
    n_steps = round(duration / model.opt.timestep)
    frames = []
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for i in range(n_steps):
            if control is not None:
                control(model, data)
            mujoco.mj_step(model, data)
            elapsed = (i + 1) * model.opt.timestep
            if len(frames) < elapsed * capture_fps:
                renderer.update_scene(data, camera=camera if camera is not None else -1)
                frame = Image.fromarray(renderer.render())
                if label:
                    ImageDraw.Draw(frame).text((10, 10), label, fill="white", font=font)
                frames.append(frame)

    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=1000 // playback_fps,  # ms per frame
        loop=0,                         # repeat forever
    )
    return len(frames)
