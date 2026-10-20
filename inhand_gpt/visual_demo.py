"""Target-free visual demonstrations for anonymous motion primitives."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import mujoco
from PIL import Image, ImageDraw

from .policy import GPTPolicy, PolicyError
from .primitives import get as get_primitive
from .primitives import menu as primitive_menu


TRANSIENT_TOKENS = (
    "timed out", "time-out", "timeout", "connecterror", "connection",
    "unexpected_eof", "temporarily unavailable", "bad gateway",
    "gateway time-out", "gateway timeout", "connection reset",
    "winerror 10054", "远程主机强迫关闭", "502", "503", "504",
)


def _is_transient_error(exc: BaseException) -> bool:
    """Recognize transport failures even when Windows localizes the text."""
    current: BaseException | None = exc
    while current is not None:
        name = type(current).__name__.lower()
        message = str(current).lower()
        if (name in {"connecterror", "readtimeout", "writetimeout",
                     "connecttimeout", "pooltimeout"} or
                any(token in message for token in TRANSIENT_TOKENS)):
            return True
        current = current.__cause__ or current.__context__
    return False


def _contact_sheet(frames: list[np.ndarray]) -> np.ndarray:
    """Arrange the 16 repetition-boundary frames in a labelled 4x4 grid."""
    if len(frames) != 16:
        raise ValueError("15 repetitions require exactly 16 boundary frames")
    tiles = []
    for index, frame in enumerate(frames):
        image = Image.fromarray(frame).convert("RGB")
        image.thumbnail((240, 240), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (256, 272), (0, 0, 0))
        x = (256 - image.width) // 2
        y = 16 + (240 - image.height) // 2
        tile.paste(image, (x, y))
        ImageDraw.Draw(tile).text(
            (7, 3), "before" if index == 0 else f"after repeat {index}",
            fill=(255, 255, 255))
        tiles.append(np.asarray(tile))
    rows = [np.concatenate(tiles[row:row + 4], axis=1)
            for row in range(0, 16, 4)]
    return np.concatenate(rows, axis=0)


def _summarize_with_retry(policy: GPTPolicy, alias: str,
                          sheet: np.ndarray, repetitions: int,
                          retries: int,
                          calibration_surface: bool = False
                          ) -> dict[str, Any]:
    for attempt in range(retries + 1):
        try:
            return policy.summarize_visual_demo(
                alias, sheet, repetitions, calibration_surface)
        except PolicyError as exc:
            transient = _is_transient_error(exc)
            if not transient or attempt >= retries:
                raise
            delay = 2 * (attempt + 1)
            print(f"warning: visual demo summary retry {attempt + 1}/"
                  f"{retries} after {delay}s: {exc}")
            time.sleep(delay)
    raise AssertionError("unreachable")


def _sphere_summary_with_retry(policy: GPTPolicy, atlas: np.ndarray,
                               view_count: int, retries: int
                               ) -> dict[str, Any]:
    for attempt in range(retries + 1):
        try:
            return policy.summarize_sphere_visual_atlas(atlas, view_count)
        except PolicyError as exc:
            transient = _is_transient_error(exc)
            if not transient or attempt >= retries:
                raise
            delay = 2 * (attempt + 1)
            print(f"warning: sphere atlas summary retry {attempt + 1}/"
                  f"{retries} after {delay}s: {exc}")
            time.sleep(delay)
    raise AssertionError("unreachable")


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.asarray([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float64)


def _euler_quat(pitch_deg: float, yaw_deg: float) -> np.ndarray:
    pitch = np.deg2rad(pitch_deg) / 2.0
    yaw = np.deg2rad(yaw_deg) / 2.0
    q_pitch = np.asarray([np.cos(pitch), 0.0, np.sin(pitch), 0.0])
    q_yaw = np.asarray([np.cos(yaw), 0.0, 0.0, np.sin(yaw)])
    return _quat_mul(q_yaw, q_pitch)


def _crop_object(frame: np.ndarray) -> Image.Image:
    """Crop the only visible geom from a constant MuJoCo background."""
    image = Image.fromarray(frame).convert("RGB")
    pixels = np.asarray(image)
    corners = np.asarray([
        pixels[0, 0], pixels[0, -1], pixels[-1, 0], pixels[-1, -1]],
        dtype=np.float32)
    background = np.median(corners, axis=0)
    mask = np.linalg.norm(pixels.astype(np.float32) - background, axis=2) > 14
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return image
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    side = max(x1 - x0, y1 - y0) + 28
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    left = max(0, cx - side // 2)
    top = max(0, cy - side // 2)
    right = min(image.width, left + side)
    bottom = min(image.height, top + side)
    left = max(0, right - side)
    top = max(0, bottom - side)
    return image.crop((left, top, right, bottom))


def _atlas_sheet(frames: list[np.ndarray]) -> np.ndarray:
    if len(frames) != 24:
        raise ValueError("sphere atlas requires exactly 24 views")
    tiles = []
    for index, frame in enumerate(frames, start=1):
        image = _crop_object(frame)
        image.thumbnail((232, 232), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (256, 264), (0, 0, 0))
        tile.paste(image, ((256 - image.width) // 2,
                           24 + (232 - image.height) // 2))
        ImageDraw.Draw(tile).text(
            (7, 5), f"view_{index:02d}", fill=(255, 255, 255))
        tiles.append(np.asarray(tile))
    rows = [np.concatenate(tiles[row:row + 6], axis=1)
            for row in range(0, 24, 6)]
    return np.concatenate(rows, axis=0)


def _calibration_texture(width: int, height: int) -> np.ndarray:
    """Build a text-free, asymmetric black/white six-region texture."""
    image = Image.new("RGB", (width, height), (238, 238, 238))
    draw = ImageDraw.Draw(image)
    face_h = height // 6
    for face in range(6):
        y0 = face * face_h
        y1 = height if face == 5 else (face + 1) * face_h
        margin = max(5, min(width, face_h) // 12)
        box = (margin, y0 + margin, width - margin - 1, y1 - margin - 1)
        if face == 0:
            draw.ellipse(box, fill=(20, 20, 20))
            inner = margin * 3
            draw.ellipse((inner, y0 + inner, width - inner - 1,
                          y1 - inner - 1), fill=(238, 238, 238))
        elif face == 1:
            cells = 5
            for row in range(cells):
                for col in range(cells):
                    if (row + col) % 2 == 0:
                        draw.rectangle((col * width // cells,
                                        y0 + row * face_h // cells,
                                        (col + 1) * width // cells,
                                        y0 + (row + 1) * face_h // cells),
                                       fill=(20, 20, 20))
        elif face == 2:
            band = max(8, width // 12)
            for offset in range(-face_h, width, band * 3):
                draw.polygon([(offset, y1), (offset + band, y1),
                              (offset + face_h + band, y0),
                              (offset + face_h, y0)], fill=(20, 20, 20))
        elif face == 3:
            draw.rectangle((0, y0, width, y1), fill=(20, 20, 20))
            arm = max(10, width // 7)
            cx, cy = width // 2, (y0 + y1) // 2
            draw.rectangle((cx - arm, y0 + margin, cx + arm, y1 - margin),
                           fill=(238, 238, 238))
            draw.rectangle((margin, cy - arm, width - margin, cy + arm),
                           fill=(238, 238, 238))
        elif face == 4:
            for inset in range(margin, min(width, face_h) // 2, margin * 2):
                color = (20, 20, 20) if (inset // (margin * 2)) % 2 == 0 else (238, 238, 238)
                draw.rectangle((inset, y0 + inset, width - inset - 1,
                                y1 - inset - 1), fill=color)
        else:
            radius = max(8, width // 11)
            for x_frac, y_frac, scale in ((0.25, 0.28, 1.0),
                                          (0.68, 0.43, 1.35),
                                          (0.42, 0.76, 0.75)):
                r = int(radius * scale)
                cx, cy = int(width * x_frac), y0 + int(face_h * y_frac)
                draw.ellipse((cx - r, cy - r, cx + r, cy + r),
                             fill=(20, 20, 20))
    return np.asarray(image, dtype=np.uint8)


def apply_sphere_demo_surface(env) -> None:
    """Replace only the demo sphere's appearance; physics stays unchanged."""
    model = env.unwrapped.model
    material_id = int(model.geom("object").matid[0])
    texture_id = int(model.mat_texid[material_id, 1])
    if texture_id < 0:
        raise RuntimeError("demo sphere material has no RGB texture")
    width = int(model.tex_width[texture_id])
    height = int(model.tex_height[texture_id])
    address = int(model.tex_adr[texture_id])
    texture = _calibration_texture(width, height).reshape(-1)
    count = width * height * 3
    model.tex_data[address:address + count] = texture


def run_sphere_visual_atlas(
        env, policy: GPTPolicy, seed: int, output_dir: str | Path,
        api_retries: int = 2) -> dict[str, Any]:
    """Let the Agent map sphere markings from image-only multi-view evidence."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model = env.unwrapped.model
    data = env.unwrapped.data
    env.reset(seed=seed)
    rgba = model.geom_rgba.copy()
    qpos = data.qpos.copy()
    qvel = data.qvel.copy()
    object_geom_id = model.geom("object").id
    joint = model.joint("object:joint")
    qpos_address = int(joint.qposadr[0])
    dof_address = int(joint.dofadr[0])
    initial_quat = data.qpos[qpos_address + 3:qpos_address + 7].copy()
    frames: list[np.ndarray] = []
    try:
        model.geom_rgba[:, 3] = 0.0
        model.geom_rgba[object_geom_id] = rgba[object_geom_id]
        for pitch in (-75.0, -45.0, -15.0, 15.0, 45.0, 75.0):
            for yaw in (0.0, 90.0, 180.0, 270.0):
                quat = _quat_mul(_euler_quat(pitch, yaw), initial_quat)
                quat /= np.linalg.norm(quat)
                data.qpos[qpos_address + 3:qpos_address + 7] = quat
                data.qvel[dof_address:dof_address + 6] = 0.0
                mujoco.mj_forward(model, data)
                frame = env.render()
                if frame is None:
                    raise RuntimeError("failed to render sphere atlas view")
                frames.append(frame.copy())
    finally:
        model.geom_rgba[:] = rgba
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        mujoco.mj_forward(model, data)
    atlas = _atlas_sheet(frames)
    atlas_path = output / "sphere_surface_atlas.png"
    Image.fromarray(atlas).save(atlas_path)
    print(f"sphere visual atlas written: {atlas_path} (24 image-only views)")
    summary = _sphere_summary_with_retry(
        policy, atlas, len(frames), api_retries)
    record = {
        "protocol": {
            "view_count": len(frames),
            "agent_received_pose_truth": False,
            "atlas": str(atlas_path),
        },
        "agent_summary": summary,
    }
    (output / "sphere_visual_notebook.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def run_visual_action_demos(
        env, policy: GPTPolicy, object_shape: str, seed: int,
        repetitions: int, output_dir: str | Path,
        api_retries: int = 2, resume: bool = False
        ) -> dict[str, Any]:
    """Execute every primitive fully ``repetitions`` times from one reset.

    Each primitive gets an independent reset to the same seed. Within one
    primitive's demonstration, its complete waveform is run consecutively 15
    times. Only boundary RGB frames are sent to the Agent; no target or state
    truth is included.
    """
    if repetitions != 15:
        raise ValueError("the visual demo protocol requires 15 repetitions")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "visual_demo_manifest.json").write_text(
        json.dumps({
            "agent": policy.agent,
            "api_protocol": policy.api_protocol,
            "model": policy.model,
            "repetitions_per_action": repetitions,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
    calibration_surface = object_shape == "sphere"
    if calibration_surface:
        apply_sphere_demo_surface(env)
    entries = primitive_menu(object_shape)
    notebook_path = output / "visual_action_notebook.json"
    notebook: dict[str, Any] = {}
    if resume and notebook_path.exists():
        loaded = json.loads(notebook_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise RuntimeError("visual action notebook must be a JSON object")
        notebook = loaded
    target_geom = env.unwrapped.model.geom("target")
    original_target_alpha = float(target_geom.rgba[3])
    target_geom.rgba[3] = 0.0
    try:
        for index, item in enumerate(entries, start=1):
            alias = f"action_{index:02d}"
            primitive = get_primitive(item["name"])
            if (alias in notebook and
                    isinstance(notebook[alias], dict) and
                    isinstance(notebook[alias].get("agent_summary"), dict)):
                print(f"visual demo {alias}: reusing completed summary")
                continue
            sheet_path = output / f"{alias}_15_repetitions.png"
            if resume and sheet_path.exists():
                sheet = np.asarray(Image.open(sheet_path).convert("RGB"))
                print(f"visual demo {alias}: reusing rendered contact sheet")
            else:
                env.reset(seed=seed)
                first = env.render()
                if first is None:
                    raise RuntimeError("failed to render visual demo frame")
                frames = [first.copy()]
                executed_steps = 0
                for repetition in range(1, repetitions + 1):
                    for step in range(primitive.n_steps):
                        _, _, terminated, truncated, _ = env.step(
                            primitive.action(step))
                        executed_steps += 1
                        if terminated or truncated:
                            raise RuntimeError(
                                f"{alias} demo ended during repetition "
                                f"{repetition}")
                    frame = env.render()
                    if frame is None:
                        raise RuntimeError("failed to render visual demo frame")
                    frames.append(frame.copy())
                sheet = _contact_sheet(frames)
                Image.fromarray(sheet).save(sheet_path)
                print(f"visual demo {alias}: 15 complete executions, "
                      f"{executed_steps} simulation steps")
            summary = _summarize_with_retry(
                policy, alias, sheet, repetitions, api_retries,
                calibration_surface)
            notebook[alias] = {
                "demonstration": {
                    "repetitions": repetitions,
                    "amplitude_scale": 1.0,
                    "complete_primitive_steps_each": primitive.n_steps,
                    "contact_sheet": str(sheet_path),
                    "surface": (
                        "black_white_asymmetric_calibration"
                        if calibration_surface else "formal_object"),
                    "physics_changed_for_surface": False,
                },
                "agent_summary": summary,
            }
            notebook_path.write_text(
                json.dumps(notebook, indent=2, ensure_ascii=False),
                encoding="utf-8")
    finally:
        target_geom.rgba[3] = original_target_alpha
    # The formal policy receives only its own visual summaries. Paths, step
    # counts, and internal execution metadata remain in the audit file.
    return {alias: record["agent_summary"]
            for alias, record in notebook.items()}
