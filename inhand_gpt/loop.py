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
CheckpointFn = Optional[Callable[[Dict[str, Any]], None]]


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
    observation_mode: str = "state",
    target_reference_frame: Optional[np.ndarray] = None,
    resume_state: Optional[Dict[str, Any]] = None,
    checkpoint_fn: CheckpointFn = None,
) -> Dict[str, Any]:
    """Run one episode. Returns the summary dict (also stored on recorder)."""
    obs, _ = env.reset(seed=seed)
    initial_pos, _ = block_pose(env)
    current_frame = None
    previous_decision_frame = None
    if resume_state is not None:
        import mujoco
        u = env.unwrapped
        u.data.qpos[:] = resume_state["qpos"]
        u.data.qvel[:] = resume_state["qvel"]
        if u.data.act.size and "act" in resume_state:
            u.data.act[:] = resume_state["act"]
        if "ctrl" in resume_state:
            u.data.ctrl[:] = resume_state["ctrl"]
        if "time" in resume_state:
            u.data.time = float(resume_state["time"])
        mujoco.mj_forward(u.model, u.data)
        obs = u._get_obs()
        initial_pos = np.asarray(resume_state["initial_pos"]).copy()
        current_frame = resume_state.get("current_frame")
        previous_decision_frame = resume_state.get(
            "previous_decision_frame")
    if frame_fn is not None and current_frame is None:
        current_frame = frame_fn()  # video seed and visual current view
    if observation_mode == "vision" and target_reference_frame is None:
        raise ValueError("vision mode requires a target reference frame")
    t0 = time.time()

    history: List[Dict[str, Any]] = (
        list(resume_state.get("history", []))
        if resume_state is not None else [])
    menu = primitive_menu(object_shape)
    outcome = "failure_max_cycles"
    final_obs = pack_observation(obs, initial_pos, cycle=0)
    final_obs["object_shape"] = object_shape
    active_choice: Optional[str] = (
        resume_state.get("active_choice") if resume_state else None)
    active_amplitude_scale = float(
        resume_state.get("active_amplitude_scale", 1.0)
        if resume_state else 1.0)
    active_next_step = int(
        resume_state.get("active_next_step", 0) if resume_state else 0)
    start_cycle = int(resume_state.get("next_cycle", 0) if resume_state else 0)

    def save_checkpoint(next_cycle: int) -> None:
        if checkpoint_fn is None:
            return
        u = env.unwrapped
        checkpoint_fn({
            "next_cycle": next_cycle,
            "qpos": u.data.qpos.copy(),
            "qvel": u.data.qvel.copy(),
            "act": u.data.act.copy(),
            "ctrl": u.data.ctrl.copy(),
            "time": float(u.data.time),
            "initial_pos": np.asarray(initial_pos).copy(),
            "history": history,
            "active_choice": active_choice,
            "active_amplitude_scale": active_amplitude_scale,
            "active_next_step": active_next_step,
            "current_frame": (None if current_frame is None
                              else current_frame.copy()),
            "previous_decision_frame": (
                None if previous_decision_frame is None
                else previous_decision_frame.copy()),
        })

    if verbose and start_cycle == 0:
        print(f"[seed {seed}] initial err={final_obs['err_deg']:+.2f}deg "
              f"rot_dist={final_obs['rot_dist_rad']:.3f}rad")

    elif verbose:
        print(f"[seed {seed}] resuming at cycle {start_cycle}/{max_cycles}")

    for cycle in range(start_cycle, max_cycles):
        obs_pack = pack_observation(obs, initial_pos, cycle)
        obs_pack["object_shape"] = object_shape
        # Use the requested strict three-axis tolerance instead of the
        # environment's much looser 0.1 rad success flag.  Once reached, the
        # panel ends without issuing another action; the grid compositor holds
        # its final frame while the remaining panels continue.
        obs_pack["success"] = all(
            abs(float(value)) <= lock_error_deg
            for value in obs_pack["rotation_error_xyz_deg"])

        # Persist the exact state before the network request. If all API
        # retries fail, rerunning can continue this same undecided cycle.
        save_checkpoint(cycle)

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
            visual_frames = None
            if observation_mode == "vision":
                visual_frames = [current_frame, target_reference_frame]
                if previous_decision_frame is not None:
                    visual_frames.append(previous_decision_frame)
            if observation_mode == "vision" and current_frame is None:
                raise RuntimeError("vision mode could not render current frame")
            decision = policy.choose(  # type: ignore[arg-type]
                obs_pack, history, menu, visual_frames=visual_frames)
            if observation_mode == "vision":
                previous_decision_frame = current_frame.copy()
        else:
            decision = policy.choose(obs_pack, history)
        choice = decision["choice"]
        amplitude_scale = float(decision.get("amplitude_scale", 1.0))
        err_before = obs_pack["err_deg"]
        rot_dist_before = obs_pack["rot_dist_rad"]
        tilt_before = obs_pack["tilt_error_deg"]
        rotation_error_before = obs_pack["rotation_error_xyz_deg"]

        if choice == "finish":
            final_obs = obs_pack
            outcome = "success" if obs_pack["success"] else "failure_max_cycles"
            recorder.log_cycle({
                "cycle": cycle, "choice": "finish", "rationale": decision["rationale"],
                "amplitude_scale": 1.0,
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
            choice == active_choice and
            abs(amplitude_scale - active_amplitude_scale) < 1e-9 and
            active_next_step < prim.n_steps)
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
            action = prim.action(t, amplitude_scale=amplitude_scale)
            obs, _, terminated, truncated, _ = env.step(action)
            if frame_fn is not None:
                rendered = frame_fn()
                if rendered is not None:
                    current_frame = rendered
            if terminated or truncated:
                break
        primitive_completed = primitive_step_end >= prim.n_steps
        if primitive_completed or terminated or truncated:
            active_choice = None
            active_amplitude_scale = 1.0
            active_next_step = 0
        else:
            active_choice = choice
            active_amplitude_scale = amplitude_scale
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
            "amplitude_scale": amplitude_scale,
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
            "amplitude_scale": amplitude_scale,
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
        save_checkpoint(cycle + 1)
        if verbose:
            print(f"[seed {seed}] cycle {cycle}: {choice:9s} "
                  f"amp={amplitude_scale:.2f} "
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
