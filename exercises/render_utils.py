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


def record_frames(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    duration: float,
    camera: mujoco.MjvCamera | None = None,
    control: Callable[[mujoco.MjModel, mujoco.MjData], None] | None = None,
    capture_fps: int = 60,
    width: int = 640,
    height: int = 480,
    label: str | Callable[[mujoco.MjData], str] | None = None,
) -> list[Image.Image]:
    """Simulate `duration` seconds from the current state and return the rendered frames.

    control(model, data), if given, runs before every mj_step (e.g. to set data.ctrl).
    label, if given, is drawn in the top-left corner of every frame; it can be a function of data
    (e.g. to show the current command or time).
    """
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
                    text = label(data) if callable(label) else label
                    ImageDraw.Draw(frame).text((10, 10), text, fill="white", font=font)
                frames.append(frame)
    return frames


def save_gif(frames: list[Image.Image], path: Path, fps: int) -> None:
    """Save frames as a looping GIF played at fps."""
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=1000 // fps, loop=0)


def side_by_side(frame_lists: list[list[Image.Image]], gap: int = 4) -> list[Image.Image]:
    """Tile several equally long frame lists horizontally, one panel per list."""
    w, h = frame_lists[0][0].size
    n = min(len(f) for f in frame_lists)
    tiled = []
    for i in range(n):
        canvas = Image.new("RGB", (len(frame_lists) * w + (len(frame_lists) - 1) * gap, h), (0, 0, 0))
        for k, frames in enumerate(frame_lists):
            canvas.paste(frames[i], (k * (w + gap), 0))
        tiled.append(canvas)
    return tiled


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
    label: str | Callable[[mujoco.MjData], str] | None = None,
) -> int:
    """Simulate `duration` seconds from the current state and save a GIF (see record_frames).

    playback_fps < capture_fps gives slow motion; defaults to real time.
    Returns the number of frames saved.
    """
    frames = record_frames(model, data, duration, camera, control, capture_fps, width, height, label)
    save_gif(frames, path, playback_fps or capture_fps)
    return len(frames)
