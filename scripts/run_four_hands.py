"""Run four independent Shadow Hand environments and make a 2x2 video.

Panel A reproduces the ordinary seed reset.  Panels B-D share A's hand pose,
object position, and target quaternion, but start with different XYZ object
orientation offsets.  Blocks and spheres use separate preset tables.  Each
panel owns an independent policy and history.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inhand_gpt.loop import run_episode
from inhand_gpt.policy import GPTPolicy, PolicyError, RuleBaselinePolicy
from inhand_gpt.record import Recorder
from run_sim import (MAX_OBJECT_SCALE, MAX_SPHERE_SCALE, VideoWriter,
                     build_env, object_scale_arg)


BLOCK_PRESETS = (
    ("A", (0.0, 0.0, 0.0), "original seed state"),
    ("B", (15.0, -10.0, 35.0), "X=+15 Y=-10 Z=+35 deg"),
    ("C", (-15.0, 12.0, -45.0), "X=-15 Y=+12 Z=-45 deg"),
    ("D", (20.0, -18.0, 120.0), "X=+20 Y=-18 Z=+120 deg"),
)

# A keeps the single-sphere experiment unchanged.  B-D deliberately combine
# substantial rotations on all three axes; because a sphere's collision shape
# is rotation invariant, these offsets change its marked orientation without
# changing the initial geometric fit in the hand.
SPHERE_PRESETS = (
    ("A", (0.0, 0.0, 0.0), "original 80 mm sphere state"),
    ("B", (35.0, -30.0, 55.0), "X=+35 Y=-30 Z=+55 deg"),
    ("C", (-40.0, 35.0, -70.0), "X=-40 Y=+35 Z=-70 deg"),
    ("D", (-55.0, -70.0, 90.0), "X=-55 Y=-70 Z=+90 deg"),
)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product for MuJoCo wxyz quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.asarray([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float64)


def euler_xyz_quat_deg(xyz_deg: tuple[float, float, float]) -> np.ndarray:
    """Return qx*qy*qz for fixed, documented XYZ preset offsets."""
    x, y, z = (math.radians(value) for value in xyz_deg)
    qx = np.asarray([math.cos(x / 2), math.sin(x / 2), 0.0, 0.0])
    qy = np.asarray([math.cos(y / 2), 0.0, math.sin(y / 2), 0.0])
    qz = np.asarray([math.cos(z / 2), 0.0, 0.0, math.sin(z / 2)])
    q = quat_mul(quat_mul(qx, qy), qz)
    return q / np.linalg.norm(q)


class PresetResetEnv:
    """Delegate to a Gym env while replacing block pose and shared goal."""

    def __init__(self, env, block_pos: np.ndarray, block_quat: np.ndarray,
                 target_goal: np.ndarray):
        self.env = env
        self.block_pos = np.asarray(block_pos, dtype=np.float64).copy()
        self.block_quat = np.asarray(block_quat, dtype=np.float64).copy()
        self.target_goal = np.asarray(target_goal, dtype=np.float64).copy()

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def reset(self, *, seed=None, options=None):
        _, info = self.env.reset(seed=seed, options=options)
        u = self.env.unwrapped
        joint_id = u.model.joint("object:joint").id
        address = int(u.model.jnt_qposadr[joint_id])
        u.data.qpos[address:address + 3] = self.block_pos
        u.data.qpos[address + 3:address + 7] = self.block_quat
        u.data.qvel[int(u.model.jnt_dofadr[joint_id]):
                    int(u.model.jnt_dofadr[joint_id]) + 6] = 0.0
        u.goal = self.target_goal.copy()
        import mujoco
        mujoco.mj_forward(u.model, u.data)
        return u._get_obs(), info

    def step(self, action):
        return self.env.step(action)

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()


class LimitedPolicy:
    """Limit concurrent API requests while keeping per-panel policy state."""

    def __init__(self, policy, semaphore: threading.Semaphore,
                 retries: int):
        self.policy = policy
        self.semaphore = semaphore
        self.retries = retries
        self.name = policy.name

    def choose(self, *args, **kwargs):
        if self.name != "gpt":
            return self.policy.choose(*args, **kwargs)
        for attempt in range(self.retries + 1):
            try:
                with self.semaphore:
                    return self.policy.choose(*args, **kwargs)
            except PolicyError as exc:
                message = str(exc).lower()
                transient = any(token in message for token in (
                    "timed out", "timeout", "connecterror", "connection",
                    "unexpected_eof", "temporarily unavailable"))
                if not transient or attempt >= self.retries:
                    raise
                delay = 2 * (attempt + 1)
                print(f"warning: transient GPT API failure; retry "
                      f"{attempt + 1}/{self.retries} after {delay}s: {exc}")
                time.sleep(delay)
        raise AssertionError("unreachable")


def reference_state(seed: int, object_scale: float, object_shape: str):
    env = build_env(seed, need_video=False, object_scale=object_scale,
                    object_shape=object_shape)
    try:
        obs, _ = env.reset(seed=seed)
        return (
            np.asarray(obs["achieved_goal"][:3], dtype=np.float64).copy(),
            np.asarray(obs["achieved_goal"][3:7], dtype=np.float64).copy(),
            np.asarray(obs["desired_goal"], dtype=np.float64).copy(),
        )
    finally:
        env.close()


def run_panel(panel: str, offset: tuple[float, float, float], label: str,
              args, initial_pos: np.ndarray, initial_quat: np.ndarray,
              target_goal: np.ndarray, api_semaphore: threading.Semaphore,
              render_lock: threading.Lock,
              output_dir: Path) -> dict[str, Any]:
    env_base = build_env(args.seed, need_video=True,
                         object_scale=args.object_scale,
                         object_shape=args.object_shape)
    offset_quat = euler_xyz_quat_deg(offset)
    block_quat = quat_mul(offset_quat, initial_quat)
    block_quat /= np.linalg.norm(block_quat)
    env = PresetResetEnv(
        env_base, initial_pos, block_quat, target_goal)
    policy = (RuleBaselinePolicy(deadband_deg=args.deadband)
              if args.policy == "baseline" else GPTPolicy(
                  timeout_s=args.api_timeout))
    policy = LimitedPolicy(policy, api_semaphore, args.api_retries)
    recorder = Recorder(
        policy_name=policy.name, seed=args.seed, max_cycles=args.max_cycles)
    recorder.meta.update({
        "panel": panel,
        "initial_pose_label": label,
        "initial_xyz_offset_deg": list(offset),
        "shared_target_goal": target_goal.tolist(),
        "object_scale": args.object_scale,
        "object_shape": args.object_shape,
        "four_hand_run": True,
        "lock_error_deg": args.lock_error_deg,
        "primitive_chunk_steps": args.primitive_chunk_steps,
        "api_timeout_s": args.api_timeout,
        "api_retries": args.api_retries,
    })
    if args.policy == "gpt":
        recorder.meta.update({
            "action_names_anonymized": True,
            "action_interface": "anonymous_name_with_kinematic_description",
            "action_selection": "uniform_full_menu_chunked_v1",
            "memory_mode": "recent5_plus_local_recent3_xyz_effect_stats",
        })
    panel_video = output_dir / f"hand_{panel.lower()}.mp4"
    writer = VideoWriter(str(panel_video), fps=args.fps)

    def frame_fn():
        # GLFW context creation/rendering is not concurrently thread-safe on
        # Windows.  Physics and API calls remain parallel; only frame capture
        # is serialized.
        with render_lock:
            frame = env.render()
        if frame is not None:
            writer.append(frame)
        return frame

    api_error = None
    try:
        summary = run_episode(
            env, policy, max_cycles=args.max_cycles, seed=args.seed,
            recorder=recorder, frame_fn=frame_fn, verbose=not args.quiet,
            lock_error_deg=args.lock_error_deg,
            primitive_chunk_steps=args.primitive_chunk_steps,
            object_shape=args.object_shape)
    except PolicyError as exc:
        api_error = str(exc)
        last = recorder.cycles[-1] if recorder.cycles else {}
        summary = {
            "outcome": "failure_api",
            "n_cycles": len(recorder.cycles),
            "final_err_deg": last.get("err_after_deg"),
            "final_rot_dist_rad": last.get("rot_dist_after_rad"),
            "final_drift_xy_m": last.get("drift_xy_m"),
            "wall_time_s": None,
            "error": api_error,
        }
        recorder.summary = summary
        print(f"panel {panel} stopped after API retries: {exc}")
    finally:
        writer.close()
        with render_lock:
            env.close()
    json_path = recorder.export(output_dir / f"hand_{panel.lower()}.json")
    return {
        "panel": panel,
        "label": label,
        "offset_xyz_deg": list(offset),
        "video": str(writer.path),
        "json": str(json_path),
        "summary": summary,
        "api_error": api_error,
    }


def annotate(frame: np.ndarray, title: str) -> np.ndarray:
    from PIL import Image, ImageDraw
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    box_height = 26
    draw.rectangle((0, 0, image.width, box_height), fill=(0, 0, 0))
    draw.text((7, 6), title, fill=(255, 255, 255))
    return np.asarray(image)


def compose_grid(results: list[dict[str, Any]], output: Path,
                 fps: int) -> Path:
    import imageio.v2 as imageio
    ordered = sorted(results, key=lambda item: item["panel"])
    readers = [imageio.get_reader(item["video"]) for item in ordered]
    writer = VideoWriter(str(output), fps=fps)
    last_frames: list[np.ndarray | None] = [None] * 4
    ended = [False] * 4
    try:
        while not all(ended):
            for index, reader in enumerate(readers):
                if ended[index]:
                    continue
                try:
                    last_frames[index] = reader.get_next_data()
                except (IndexError, StopIteration, RuntimeError):
                    ended[index] = True
            if all(frame is None for frame in last_frames):
                break
            panels = []
            for index, frame in enumerate(last_frames):
                if frame is None:
                    reference = next(item for item in last_frames
                                     if item is not None)
                    frame = np.zeros_like(reference)
                title = f"{ordered[index]['panel']}  {ordered[index]['label']}"
                panels.append(annotate(frame, title))
            height = min(frame.shape[0] for frame in panels)
            width = min(frame.shape[1] for frame in panels)
            panels = [frame[:height, :width, :3] for frame in panels]
            grid = np.concatenate([
                np.concatenate(panels[:2], axis=1),
                np.concatenate(panels[2:], axis=1),
            ], axis=0)
            writer.append(grid)
    finally:
        writer.close()
        for reader in readers:
            reader.close()
    return writer.path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Four independent hands with a shared target and 2x2 video")
    parser.add_argument("--policy", choices=["baseline", "gpt"], default="gpt")
    parser.add_argument(
        "--execution-mode", choices=["parallel", "serial"],
        default="parallel",
        help="run four environments concurrently or complete A-D in order")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-cycles", type=int, default=50)
    parser.add_argument(
        "--object-shape", choices=["block", "sphere"], default="block")
    parser.add_argument("--deadband", type=float, default=5.0)
    parser.add_argument("--object-scale", type=object_scale_arg, default=1.03)
    parser.add_argument("--api-concurrency", type=int, default=1)
    parser.add_argument("--api-timeout", type=float, default=600.0)
    parser.add_argument("--api-retries", type=int, default=2)
    parser.add_argument("--lock-error-deg", type=float, default=0.5)
    parser.add_argument("--primitive-chunk-steps", type=int, default=6)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--video", required=True)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.object_shape == "block" and args.object_scale > MAX_OBJECT_SCALE:
        parser.error(
            f"block object scale must not exceed {MAX_OBJECT_SCALE:.2f}; "
            f"sphere scale may be up to {MAX_SPHERE_SCALE:.2f}")
    if not 1 <= args.api_concurrency <= 4:
        parser.error("--api-concurrency must be between 1 and 4")
    if args.api_timeout <= 0:
        parser.error("--api-timeout must be positive")
    if args.api_retries < 0:
        parser.error("--api-retries cannot be negative")
    if args.lock_error_deg <= 0:
        parser.error("--lock-error-deg must be positive")
    if args.primitive_chunk_steps < 1:
        parser.error("--primitive-chunk-steps must be at least 1")

    output_dir = Path(args.export_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_pos, initial_quat, target_goal = reference_state(
        args.seed, args.object_scale, args.object_shape)
    presets = (SPHERE_PRESETS if args.object_shape == "sphere"
               else BLOCK_PRESETS)
    semaphore = threading.Semaphore(args.api_concurrency)
    render_lock = threading.Lock()
    results = []
    if args.execution_mode == "serial":
        for panel, offset, label in presets:
            print(f"\n=== starting panel {panel}: {label} ===")
            results.append(run_panel(
                panel, offset, label, args,
                initial_pos, initial_quat, target_goal,
                semaphore, render_lock, output_dir))
    else:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(
                    run_panel, panel, offset, label, args,
                    initial_pos, initial_quat, target_goal,
                    semaphore, render_lock, output_dir)
                for panel, offset, label in presets
            ]
            for future in as_completed(futures):
                results.append(future.result())

    combined_video = compose_grid(results, Path(args.video), args.fps)
    ordered = sorted(results, key=lambda item: item["panel"])
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps({
        "seed": args.seed,
        "execution_mode": args.execution_mode,
        "object_shape": args.object_shape,
        "preset_table": ("sphere_xyz_v1" if args.object_shape == "sphere"
                         else "block_xyz_v1"),
        "shared_target_goal": target_goal.tolist(),
        "combined_video": str(combined_video),
        "panels": ordered,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"combined video written: {combined_video}")
    print(f"summary written: {summary_path}")
    for item in ordered:
        summary = item["summary"]
        final_rot = summary.get("final_rot_dist_rad")
        final_rot_text = (f"{final_rot:.4f}rad"
                          if final_rot is not None else "n/a")
        print(f"panel {item['panel']}: {summary['outcome']}, "
              f"final_rot_dist={final_rot_text}, "
              f"cycles={summary['n_cycles']}")
    return 2 if any(item.get("api_error") for item in ordered) else 0


if __name__ == "__main__":
    sys.exit(main())
