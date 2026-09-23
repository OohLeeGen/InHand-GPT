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
) -> Dict[str, Any]:
    """Run one episode. Returns the summary dict (also stored on recorder)."""
    obs, _ = env.reset(seed=seed)
    initial_pos, _ = block_pose(env)
    if frame_fn is not None:
        frame_fn()  # ensure the video has at least one frame even if the
                    # policy finishes at cycle 0 without executing a primitive
    t0 = time.time()

    history: List[Dict[str, Any]] = []
    menu = primitive_menu()
    outcome = "failure_max_cycles"
    final_obs = pack_observation(obs, initial_pos, cycle=0)

    if verbose:
        print(f"[seed {seed}] initial err={final_obs['err_deg']:+.2f}deg "
              f"rot_dist={final_obs['rot_dist_rad']:.3f}rad")

    for cycle in range(max_cycles):
        obs_pack = pack_observation(obs, initial_pos, cycle)

        if policy.name == "gpt":
            decision = policy.choose(obs_pack, history, menu)  # type: ignore[arg-type]
        else:
            decision = policy.choose(obs_pack, history)
        choice = decision["choice"]
        err_before = obs_pack["err_deg"]

        if choice == "finish":
            final_obs = obs_pack
            outcome = "success" if obs_pack["success"] else "failure_max_cycles"
            recorder.log_cycle({
                "cycle": cycle, "choice": "finish", "rationale": decision["rationale"],
                "err_before_deg": err_before, "err_after_deg": err_before,
                "measured_delta_deg": 0.0, "drift_xy_m": obs_pack["drift_xy_m"],
                "elapsed_s": round(time.time() - t0, 2),
            })
            if verbose:
                print(f"[seed {seed}] cycle {cycle}: finish "
                      f"(err={err_before:+.2f}deg) -> {outcome}")
            break

        prim = get_primitive(choice)
        for t in range(prim.n_steps):
            action = prim.action(t)
            obs, _, terminated, truncated, _ = env.step(action)
            if frame_fn is not None:
                frame_fn()
            if terminated or truncated:
                break

        after = pack_observation(obs, initial_pos, cycle)
        delta = after["err_deg"] - err_before  # signed change of the error angle
        record = {
            "cycle": cycle,
            "choice": choice,
            "rationale": decision["rationale"],
            "err_before_deg": err_before,
            "err_after_deg": after["err_deg"],
            "measured_delta_deg": round(delta, 3),
            "drift_xy_m": after["drift_xy_m"],
            "drift_z_m": after["drift_z_m"],
            "elapsed_s": round(time.time() - t0, 2),
        }
        recorder.log_cycle(record)
        history.append({
            "cycle": cycle, "choice": choice,
            "err_before_deg": err_before,
            "err_deg": after["err_deg"], "measured_delta_deg": record["measured_delta_deg"],
            "drift_xy_m": after["drift_xy_m"],
        })
        if verbose:
            print(f"[seed {seed}] cycle {cycle}: {choice:9s} "
                  f"err {err_before:+.2f} -> {after['err_deg']:+.2f}deg "
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
