"""Closed loop: observe -> decide -> execute primitive -> observe -> record."""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .observe import block_pose, pack_observation, signed_z_error_rad
from .policy import Policy
from .primitives import get as get_primitive
from .primitives import menu as primitive_menu
from .record import Recorder

FrameFn = Optional[Callable[[], Optional[np.ndarray]]]


def run_episode(
    env,
    policy: Policy,
    max_cycles: int,
    seed: int,
    recorder: Recorder,
    frame_fn: FrameFn = None,
    verbose: bool = True,
    lock_error_deg: float = 2.0,
    primitive_chunk_steps: int = 6,
    object_shape: str = "block",
) -> Dict[str, Any]:
    """Run one episode. Returns the summary dict (also stored on recorder)."""
    obs, _ = env.reset(seed=seed)
    initial_pos, _ = block_pose(env)
    if frame_fn is not None:
        frame_fn()  # ensure the video has at least one frame even if the
                    # policy finishes at cycle 0 without executing a primitive
    t0 = time.time()

    history: List[Dict[str, Any]] = []
    menu = primitive_menu(object_shape)
    outcome = "failure_max_cycles"
    final_obs = pack_observation(obs, initial_pos, cycle=0)
    final_obs["object_shape"] = object_shape
    active_choice: Optional[str] = None
    active_next_step = 0

    if verbose:
        print(f"[seed {seed}] initial err={final_obs['err_deg']:+.2f}deg "
              f"rot_dist={final_obs['rot_dist_rad']:.3f}rad")

    for cycle in range(max_cycles):
        obs_pack = pack_observation(obs, initial_pos, cycle)
        obs_pack["object_shape"] = object_shape
        # Use the requested strict three-axis tolerance instead of the
        # environment's much looser 0.1 rad success flag.  Once reached, the
        # panel ends without issuing another action; the grid compositor holds
        # its final frame while the remaining panels continue.
        obs_pack["success"] = all(
            abs(float(value)) <= lock_error_deg
            for value in obs_pack["rotation_error_xyz_deg"])

        if obs_pack["success"]:
            decision = {
                "choice": "finish",
                "rationale": (
                    f"all XYZ errors are within +/-{lock_error_deg:g} deg; "
                    "lock the current pose"),
                "selection_mode": "strict_xyz_lock",
                "recommended_phase": "strict_xyz_lock",
                "model_reported_phase": None,
                "follows_rules": True,
            }
        elif policy.name == "gpt":
            decision = policy.choose(obs_pack, history, menu)  # type: ignore[arg-type]
        else:
            decision = policy.choose(obs_pack, history)
        choice = decision["choice"]
        err_before = obs_pack["err_deg"]
        rot_dist_before = obs_pack["rot_dist_rad"]
        tilt_before = obs_pack["tilt_error_deg"]
        rotation_error_before = obs_pack["rotation_error_xyz_deg"]

        if choice == "finish":
            final_obs = obs_pack
            outcome = "success" if obs_pack["success"] else "failure_max_cycles"
            recorder.log_cycle({
                "cycle": cycle, "choice": "finish", "rationale": decision["rationale"],
                "selection_mode": decision.get("selection_mode"),
                "recommended_phase": decision.get("recommended_phase"),
                "model_reported_phase": decision.get("model_reported_phase"),
                "follows_rules": decision.get("follows_rules"),
                "err_before_deg": err_before, "err_after_deg": err_before,
                "measured_delta_deg": 0.0,
                "rot_dist_before_rad": rot_dist_before,
                "rot_dist_after_rad": rot_dist_before,
                "rot_dist_gain_rad": 0.0,
                "tilt_before_deg": tilt_before, "tilt_after_deg": tilt_before,
                "tilt_gain_deg": 0.0,
                "rotation_error_before_xyz_deg": rotation_error_before,
                "rotation_error_after_xyz_deg": rotation_error_before,
                "rotation_error_gain_xyz_deg": [0.0, 0.0, 0.0],
                "drift_xy_m": obs_pack["drift_xy_m"],
                "elapsed_s": round(time.time() - t0, 2),
            })
            if verbose:
                print(f"[seed {seed}] cycle {cycle}: finish "
                      f"(err={err_before:+.2f}deg) -> {outcome}")
            break

        prim = get_primitive(choice)
        continuing_primitive = (
            choice == active_choice and active_next_step < prim.n_steps)
        primitive_step_start = active_next_step if continuing_primitive else 0
        # The legacy double little-finger primitive contains two independent
        # 30-step cycles.  Even when the caller requests a larger chunk, stop
        # after one cycle so the policy receives measured feedback before it
        # decides whether the second cycle is still useful.
        decision_chunk_steps = primitive_chunk_steps
        if choice == "sphere_double_little_finger_cycle":
            decision_chunk_steps = min(decision_chunk_steps, 30)
        primitive_step_end = min(
            primitive_step_start + decision_chunk_steps, prim.n_steps)
        replaced_incomplete_primitive = (
            active_choice is not None and not continuing_primitive)
        for t in range(primitive_step_start, primitive_step_end):
            action = prim.action(t)
            obs, _, terminated, truncated, _ = env.step(action)
            if frame_fn is not None:
                frame_fn()
            if terminated or truncated:
                break
        primitive_completed = primitive_step_end >= prim.n_steps
        if primitive_completed or terminated or truncated:
            active_choice = None
            active_next_step = 0
        else:
            active_choice = choice
            active_next_step = primitive_step_end

        after = pack_observation(obs, initial_pos, cycle)
        after["object_shape"] = object_shape
        after["success"] = all(
            abs(float(value)) <= lock_error_deg
            for value in after["rotation_error_xyz_deg"])
        delta = after["err_deg"] - err_before  # signed change of the error angle
        rot_dist_gain = rot_dist_before - after["rot_dist_rad"]
        tilt_gain = tilt_before - after["tilt_error_deg"]
        rotation_error_after = after["rotation_error_xyz_deg"]
        rotation_error_gain = [
            round(float(before) - float(after_value), 3)
            for before, after_value in zip(
                rotation_error_before, rotation_error_after)
        ]
        record = {
            "cycle": cycle,
            "choice": choice,
            "rationale": decision["rationale"],
            "selection_mode": decision.get("selection_mode"),
            "recommended_phase": decision.get("recommended_phase"),
            "model_reported_phase": decision.get("model_reported_phase"),
            "follows_rules": decision.get("follows_rules"),
            "err_before_deg": err_before,
            "err_after_deg": after["err_deg"],
            "measured_delta_deg": round(delta, 3),
            "rot_dist_before_rad": rot_dist_before,
            "rot_dist_after_rad": after["rot_dist_rad"],
            "rot_dist_gain_rad": round(rot_dist_gain, 4),
            "tilt_before_deg": tilt_before,
            "tilt_after_deg": after["tilt_error_deg"],
            "tilt_gain_deg": round(tilt_gain, 3),
            "rotation_error_before_xyz_deg": rotation_error_before,
            "rotation_error_after_xyz_deg": rotation_error_after,
            "rotation_error_gain_xyz_deg": rotation_error_gain,
            "primitive_total_steps": prim.n_steps,
            "primitive_step_start": primitive_step_start,
            "primitive_step_end": primitive_step_end,
            "primitive_completed": primitive_completed,
            "primitive_steps_remaining": max(
                0, prim.n_steps - primitive_step_end),
            "continued_previous_primitive": continuing_primitive,
            "replaced_incomplete_primitive": replaced_incomplete_primitive,
            "drift_xy_m": after["drift_xy_m"],
            "drift_z_m": after["drift_z_m"],
            "elapsed_s": round(time.time() - t0, 2),
        }
        recorder.log_cycle(record)
        history.append({
            "cycle": cycle, "choice": choice,
            "selection_mode": decision.get("selection_mode"),
            "recommended_phase": decision.get("recommended_phase"),
            "follows_rules": decision.get("follows_rules"),
            "err_before_deg": err_before,
            "err_deg": after["err_deg"], "measured_delta_deg": record["measured_delta_deg"],
            "rot_dist_before_rad": rot_dist_before,
            "rot_dist_rad": after["rot_dist_rad"],
            "rot_dist_gain_rad": record["rot_dist_gain_rad"],
            "tilt_before_deg": tilt_before,
            "tilt_error_deg": after["tilt_error_deg"],
            "tilt_gain_deg": record["tilt_gain_deg"],
            "rotation_error_before_xyz_deg": rotation_error_before,
            "rotation_error_xyz_deg": rotation_error_after,
            "rotation_error_gain_xyz_deg": rotation_error_gain,
            "primitive_total_steps": prim.n_steps,
            "primitive_step_start": primitive_step_start,
            "primitive_step_end": primitive_step_end,
            "primitive_completed": primitive_completed,
            "primitive_steps_remaining": max(
                0, prim.n_steps - primitive_step_end),
            "continued_previous_primitive": continuing_primitive,
            "replaced_incomplete_primitive": replaced_incomplete_primitive,
            "drift_xy_m": after["drift_xy_m"],
        })
        if verbose:
            print(f"[seed {seed}] cycle {cycle}: {choice:9s} "
                  f"err {err_before:+.2f} -> {after['err_deg']:+.2f}deg "
                  f"rot_dist={after['rot_dist_rad']:.3f}rad "
                  f"tilt={after['tilt_error_deg']:.1f}deg "
                  f"steps={primitive_step_start}:{primitive_step_end}/"
                  f"{prim.n_steps} "
                  f"(drift_xy={after['drift_xy_m'] * 100:.2f}cm)")

        final_obs = after
        if after["success"]:
            outcome = "success"
            break
        if after["dropped"]:
            outcome = "failure_dropped"
            break
        if terminated or truncated:
            break

    summary = recorder.finish(outcome, final_obs, time.time() - t0)
    if verbose:
        print(f"[seed {seed}] episode done: {outcome}, "
              f"final_err={final_obs['err_deg']:+.2f}deg, cycles={len(recorder.cycles)}")
    return summary
