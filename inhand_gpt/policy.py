"""Policies: rule baseline + GPT (OpenAI-compatible) decision makers.

Research-integrity rule: GPTPolicy NEVER silently falls back to the rule
baseline. A missing INHAND_API_KEY or a failed/malformed response is a hard
error, so every logged decision is traceable to the declared policy.

Run the offline parser self-test with:
    python -m inhand_gpt.policy
"""
from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

ENV_API_BASE = "INHAND_API_BASE"
ENV_API_KEY = "INHAND_API_KEY"
ENV_MODEL = "INHAND_MODEL"
DEFAULT_API_BASE = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-6"


class PolicyError(RuntimeError):
    pass


class Policy(ABC):
    name = "abstract"

    @abstractmethod
    def choose(self, obs: Dict[str, Any], history: List[Dict[str, Any]]) -> Dict[str, str]:
        """Return {"choice": <primitive name or 'finish'>, "rationale": str}."""


class RuleBaselinePolicy(Policy):
    """Sign-prior pair rotation with measured-helpfulness feedback.

    err = signed Z error (block - goal). The candidate pool is restricted to
    gaits whose measured prior sign matches the needed rotation (exploring
    opposite-prior gaits cost up to -18deg in closed-loop tests):

      err>0 (need -yaw): [roll_cw, roll_th, roll_wr]  (+ nudge_cw if |err|<18)
      err<0 (need +yaw): [roll_ccw, roll_lf, roll_wr]  (+ nudge_ccw if |err|<18)

    Selection (window = uses since the last "regrasp"):
      1. |err| <= deadband -> "finish" (loop judges success at 0.1 rad).
      2. Exploit the candidate with the best mean helpfulness
         (|err_before| - |err_after|) over its last 2 window uses, if >= 0.3.
      3. Otherwise pick the first candidate with < 2 window uses (explore).
      4. Otherwise "regrasp": open/close reset that clears the window and
         typically buys a few degrees plus fresh contact for the next round.
    """

    name = "baseline"
    SCORE_MIN = 0.3
    PAIRS = {
        +1: ["roll_cw", "roll_th", "roll_wr"],
        -1: ["roll_ccw", "roll_lf", "roll_wr"],
    }
    NUDGE = {+1: "nudge_cw", -1: "nudge_ccw"}

    def __init__(self, deadband_deg: float = 5.0, fine_deg: float = 18.0):
        self.deadband_deg = deadband_deg
        self.fine_deg = fine_deg

    def _candidates(self, err: float) -> List[str]:
        sign = +1 if err > 0 else -1
        cands = list(self.PAIRS[sign])
        if abs(err) < self.fine_deg:
            cands.insert(0, self.NUDGE[sign])
        return cands

    @staticmethod
    def _window(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for h in history:
            if h["choice"] == "regrasp":
                out = []
            else:
                out.append(h)
        return out

    @classmethod
    def _uses(cls, window: List[Dict[str, Any]], name: str) -> List[float]:
        return [abs(h["err_before_deg"]) - abs(h["err_deg"])
                for h in window if h["choice"] == name]

    def choose(self, obs: Dict[str, Any], history: List[Dict[str, Any]]) -> Dict[str, str]:
        err = obs["err_deg"]
        if abs(err) <= self.deadband_deg:
            return {"choice": "finish",
                    "rationale": f"|err|={abs(err):.2f}deg within deadband {self.deadband_deg}deg"}

        cands = self._candidates(err)
        window = self._window(history)
        best_score, best = None, None
        for c in cands:
            uses = self._uses(window, c)[-2:]
            if len(uses) >= 2:
                score = sum(uses) / len(uses)
                if best_score is None or score > best_score:
                    best_score, best = score, c
        if best is not None and best_score >= self.SCORE_MIN:
            return {"choice": best,
                    "rationale": f"err={err:+.2f}deg -> {best} (measured help {best_score:+.2f}deg/use)"}

        for c in cands:
            if len(self._uses(window, c)) < 2:
                return {"choice": c,
                        "rationale": f"err={err:+.2f}deg -> explore {c}"}

        return {"choice": "regrasp",
                "rationale": f"pair stalled at err={err:+.2f}deg; regrasp resets contact"}


# --- GPT policy ---------------------------------------------------------------

PROMPT_TEMPLATE = """You are the high-level decision policy for a Shadow Dexterous Hand
performing in-hand reorientation of a marked object (block or sphere) in a
MuJoCo simulation.

Each cycle you receive a structured observation and must pick exactly ONE
motion primitive from the menu (or "finish").

Execution is closed-loop and chunked: only a short segment of the selected
primitive runs before the next observation. Selecting the same action again
continues its remaining waveform; selecting another action abandons the
unfinished waveform and starts the new one. The recent history reports step
start/end, remaining steps, completion, continuation, and replacement. These
are observations, not constraints: every menu action remains selectable.

Observation semantics:
- err_deg: signed achieved-minus-desired rotation error about +Z in degrees.
  Positive error requires negative yaw; negative error requires positive yaw.
  A small yaw error alone does NOT mean success.
- rotation_error_xyz_deg: shortest full 3-axis rotation-error vector.
- tilt_error_deg: magnitude of the X/Y components of that vector.
- object_shape: block or sphere. A sphere has no geometric faces; its coloured
  directional markings and tracked quaternion define orientation.
- object_angular_velocity_xyz_rad_s / object_angular_speed_rad_s: measured
  object angular velocity. For the sphere, prefer a brief hold when residual
  spin is high before making a fine correction.
- rot_dist_rad: unsigned full-orientation geodesic distance. Success requires
  rot_dist_rad < 0.1 rad, not merely |err_deg| < 5.73 deg.
- drift_xy_m / drift_z_m: block-center displacement from episode start.
  Horizontal drift above 0.05 m or vertical drop below -0.05 m ends the
  episode as dropped. Large drift means the grasp is unstable.
- last_action / last_delta_deg: what was executed last cycle and the measured
  signed yaw change it produced (deg). Use this feedback to adapt.

Current observation (JSON):
{observation}

Recent history (oldest first, JSON):
{history}

Episode-long measured action statistics (JSON):
{action_stats}

Untried actions in the full menu (JSON):
{untried_actions}

Primitive menu (JSON):
{menu}

The only soft rule (advisory; it never removes an action):
- Treat the block as potentially outside the finger region when
  drift_xy_m >= 0.02 or drift_z_m <= -0.01. In that situation, normally choose
  {wrist_lower_action}, the wrist-lowering reseat motion, so the block can
  slide back toward the finger region when that action is available. If it is
  listed as unavailable, use measured feedback to select a regrasp or other
  contact-changing motion. This is advice, not a hard constraint.

Important:
- Every action in the full menu is always legal. category, use_when,
  avoid_when, suggested_scenario, and the rule are not constraints.
- Positive/negative joint directions in descriptions are actuator-coordinate
  directions only; they do NOT predict block yaw direction.
- Do not keep repeating one primitive after its effect has decayed. If the
  same action was used three or more times and its latest full-orientation
  gain is below 0.01 rad, change to an action from a different contact group,
  preferably an untried contact-escape action. This is advisory, not a gate.
- Treat episode-long mean gains as background evidence only. Prioritize the
  last gain, the recent weighted gain, and the current XYZ/contact state. A
  positive early mean must not override a negative latest result.
- If an action's latest full-orientation gain is negative, normally switch to
  a different contact group instead of immediately repeating it. This remains
  advice: every action stays legal.
- When the current orientation is close to the best state reached in this
  episode, preserve that progress: prefer a sphere_micro action, stabilization,
  or a small contact adjustment over a large-orientation escape.
- If regression_from_best_rad exceeds 0.15, treat recovery of the lost
  orientation as the immediate priority. Avoid repeating the action/contact
  group that caused the latest regression; use recent per-axis gains to choose
  a smaller corrective action. This is advisory and does not remove choices.
- For each XYZ component, a useful measured gain normally has the same sign
  as the current signed error component. Compare the episode statistics and
  prefer actions that improve the largest-magnitude component without making
  rot_dist_rad worse.
- Choose finish only when observation.success is true.

Reply with ONLY a JSON object, no markdown, no extra text:
{{"choice": "<primitive name or 'finish'>", "rationale": "<one sentence>",
  "follows_rules": true}}
"""


def build_messages(obs: Dict[str, Any], history: List[Dict[str, Any]],
                   menu: List[Dict[str, Any]], model: str,
                   selection_mode: str = "normal_rotation",
                   action_stats: Dict[str, Any] | None = None,
                   untried_actions: List[str] | None = None,
                   wrist_lower_action: str = "") -> List[Dict[str, str]]:
    prompt = PROMPT_TEMPLATE.format(
        observation=json.dumps(obs, ensure_ascii=False),
        history=json.dumps(history[-5:], ensure_ascii=False),
        selection_mode=selection_mode,
        wrist_lower_action=wrist_lower_action,
        action_stats=json.dumps(action_stats or {}, ensure_ascii=False),
        untried_actions=json.dumps(untried_actions or [], ensure_ascii=False),
        menu=json.dumps(menu, ensure_ascii=False),
    )
    return [
        {"role": "system", "content": "You are a careful robot control policy. Always answer with strict JSON."},
        {"role": "user", "content": prompt},
    ]


def anonymize_action_interface(
        history: List[Dict[str, Any]], menu: List[Dict[str, Any]]
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, str]]:
    """Replace internal primitive names with stable, episode-local aliases."""
    alias_by_name = {
        item["name"]: f"action_{index:02d}"
        for index, item in enumerate(menu, start=1)
    }
    name_by_alias = {alias: name for name, alias in alias_by_name.items()}
    prompt_menu = [
        {**item, "name": alias_by_name[item["name"]]}
        for item in menu
    ]
    prompt_history = []
    for item in history:
        prompt_item = dict(item)
        choice = prompt_item.get("choice")
        if choice in alias_by_name:
            prompt_item["choice"] = alias_by_name[choice]
        prompt_history.append(prompt_item)
    return prompt_history, prompt_menu, name_by_alias


MICRO_ACTIONS = {
    "micro_forefinger_pulse", "micro_middle_finger_pulse",
    "micro_ring_finger_pulse", "micro_little_finger_pulse",
    "micro_thumb_index_pinch", "micro_thumb_middle_pinch",
    "micro_thumb_sweep", "micro_finger_walk",
}
LATERAL_SWEEP_ACTIONS = {
    "ring_lateral_sweep_a", "ring_lateral_sweep_b",
    "ring_lateral_sweep_b_strong", "ring_lateral_ratchet_b",
}
TILT_PROBE_ACTIONS = {
    "tilt_wrist_pitch_a", "tilt_wrist_pitch_b",
    "tilt_wrist_side_a", "tilt_wrist_side_b",
    "tilt_index_little_transfer_a", "tilt_index_little_transfer_b",
    "tilt_thumb_fingers_transfer_a", "tilt_thumb_fingers_transfer_b",
    "tilt_wrj0_reseat_a", "tilt_wrj0_reseat_b",
    "tilt_wrj1_reseat_a", "tilt_wrj1_reseat_b",
    "axis_fine_wrj0_a", "axis_fine_wrj0_b",
    "axis_fine_wrj1_a", "axis_fine_wrj1_b",
    "axis_fine_diagonal_a", "axis_fine_diagonal_b",
    "escape_little_thumb_a", "escape_little_thumb_b",
    "escape_wrist_finger_a", "escape_wrist_finger_b",
    "escape_wrj0_reseat_a", "escape_wrj0_reseat_b",
    "regrasp_then_roll_wr", "little_regrasp_then_roll_th",
    "regrasp_then_roll_cw", "regrasp_then_roll_ccw",
}
NORMAL_ACTIONS = {
    "hold", "roll_cw", "roll_ccw", "roll_lf", "roll_th", "roll_wr",
    "nudge_cw", "nudge_ccw",
} | MICRO_ACTIONS | LATERAL_SWEEP_ACTIONS | TILT_PROBE_ACTIONS
CONTACT_RECOVERY_ACTIONS = {
    "regrasp", "forefinger_regrasp", "middle_finger_regrasp",
    "ring_finger_regrasp", "little_finger_regrasp", "thumb_regrasp",
    "multi_finger_dither",
}
WRIST_RECOVERY_ACTIONS = {
    "wrist_lower_reseat", "wrist_raise_reseat",
    "wrist_three_finger_regrasp", "recenter_palmward",
    "recenter_fingertips", "recenter_forefinger_side",
    "recenter_little_side",
}
RECOVERY_ACTIONS = CONTACT_RECOVERY_ACTIONS | WRIST_RECOVERY_ACTIONS
COARSE_ROTATION_ACTIONS = {
    "roll_cw", "roll_ccw", "roll_lf", "roll_th", "roll_wr",
}
GENTLE_CONTACT_SHIFT_ACTIONS = {
    "ring_lateral_sweep_a", "ring_lateral_sweep_b",
}
FINE_ORIENTATION_ACTIONS = (
    {"hold", "nudge_cw", "nudge_ccw"}
    | MICRO_ACTIONS | GENTLE_CONTACT_SHIFT_ACTIONS | TILT_PROBE_ACTIONS
    | RECOVERY_ACTIONS
)
TILT_CORRECTION_ACTIONS = (
    {"hold"} | MICRO_ACTIONS | GENTLE_CONTACT_SHIFT_ACTIONS
    | TILT_PROBE_ACTIONS | RECOVERY_ACTIONS
)


def action_guidance(name: str) -> Dict[str, str]:
    """Advisory conditions shown to GPT; never used as a hard filter."""
    if name == "sphere_hold":
        return {
            "category": "sphere_stabilization",
            "use_when": "sphere motion or angular velocity should settle",
            "avoid_when": "a measured tangential motion is improving orientation",
            "suggested_scenario": "brief sphere stabilization",
        }
    if name.startswith("sphere_micro_"):
        return {
            "category": "sphere_micro_rotation",
            "use_when": "full orientation error is small",
            "avoid_when": "a larger measured motion is still useful",
            "suggested_scenario": "near-target sphere correction",
        }
    if name.startswith("sphere_recontact"):
        return {
            "category": "sphere_contact_refresh",
            "use_when": "recent tangential motions have stalled",
            "avoid_when": "the sphere is already moving usefully",
            "suggested_scenario": "sphere contact refresh",
        }
    if name == "sphere_recenter":
        return {
            "category": "sphere_position_recovery",
            "use_when": "the sphere is moving away from the finger region",
            "avoid_when": "position and contact are stable",
            "suggested_scenario": "sphere recentering",
        }
    if name == "sphere_double_little_finger_cycle":
        return {
            "category": "sphere_large_orientation_escape",
            "use_when": "full orientation error is very large or recent motions stalled",
            "avoid_when": "near the episode best, position is unstable, or a smaller correction is sufficient",
            "suggested_scenario": "large sphere orientation escape",
        }
    if name.startswith("sphere_y_"):
        return {
            "category": "sphere_y_orientation_probe",
            "use_when": "the measured Y rotation error is significant",
            "avoid_when": "its latest Y-axis gain was negative",
            "suggested_scenario": "sphere Y-orientation adjustment",
        }
    if name.startswith("sphere_"):
        return {
            "category": "sphere_tangential_rotation",
            "use_when": "measured XYZ feedback supports this contact motion",
            "avoid_when": "its latest full-orientation gain was negative",
            "suggested_scenario": "sphere orientation adjustment",
        }
    if name == "hold":
        return {
            "category": "stabilization",
            "use_when": "motion is unstable",
            "avoid_when": "large error is improving",
            "suggested_scenario": "brief stabilization",
        }
    if name in {"nudge_cw", "nudge_ccw"}:
        return {
            "category": "fine_rotation",
            "use_when": "error is small or coarse motion overshoots",
            "avoid_when": "large error has effective coarse motion",
            "suggested_scenario": "near-target correction",
        }
    if name == "roll_lf":
        return {
            "category": "alternate_contact_rotation",
            "use_when": "contact pattern must change",
            "avoid_when": "two consecutive uses or gain below 0.5 deg",
            "suggested_scenario": "alternate-contact rotation",
        }
    if name in LATERAL_SWEEP_ACTIONS:
        return {
            "category": "contact_shift",
            "use_when": "ring-thumb rotation has stalled",
            "avoid_when": "object position is unstable",
            "suggested_scenario": "ring-contact relocation",
        }
    if name in TILT_PROBE_ACTIONS:
        return {
            "category": "three_axis_orientation_probe",
            "use_when": "X or Y rotation error is significant",
            "avoid_when": "the same action just increased full orientation error",
            "suggested_scenario": "X/Y orientation adjustment",
        }
    if name in MICRO_ACTIONS:
        return {
            "category": "small_object_micro",
            "use_when": "small object needs fine adjustment",
            "avoid_when": "coarse motion is still effective",
            "suggested_scenario": "small-object fine control",
        }
    if name in NORMAL_ACTIONS:
        return {
            "category": "coarse_rotation",
            "use_when": "large error with stable contact",
            "avoid_when": "repeated gain below 0.5 deg",
            "suggested_scenario": "early coarse rotation",
        }
    if name in CONTACT_RECOVERY_ACTIONS:
        return {
            "category": "contact_recovery",
            "use_when": "two recent rotation actions stalled",
            "avoid_when": "rotation is improving or recovery just ran",
            "suggested_scenario": "contact refresh",
        }
    return {
        "category": "position_recovery",
        "use_when": "drift exceeds 0.02 m with stalled rotation",
        "avoid_when": "drift is low or improving",
        "suggested_scenario": "object recentering",
    }


def add_action_guidance(
        prompt_menu: List[Dict[str, Any]], name_by_alias: Dict[str, str]
) -> List[Dict[str, Any]]:
    return [
        {**item, **action_guidance(name_by_alias[item["name"]])}
        for item in prompt_menu
    ]


def measured_gain(item: Dict[str, Any]) -> float:
    """Positive when the action reduced absolute rotation error."""
    before = float(item["err_before_deg"])
    after = float(item.get("err_deg", item.get("err_after_deg", before)))
    return abs(before) - abs(after)


def select_action_mode(history: List[Dict[str, Any]]) -> tuple[str, set[str]]:
    """Gate recovery menus using three consecutive low-progress cycles."""
    if not history or history[-1].get("choice") in RECOVERY_ACTIONS:
        return "normal_rotation", NORMAL_ACTIONS
    if len(history) < 3 or not all(
            item.get("choice") in NORMAL_ACTIONS and measured_gain(item) < 0.5
            for item in history[-3:]):
        return "normal_rotation", NORMAL_ACTIONS

    last_recovery = next(
        (item.get("choice") for item in reversed(history)
         if item.get("choice") in RECOVERY_ACTIONS),
        None,
    )
    if last_recovery in CONTACT_RECOVERY_ACTIONS:
        return "wrist_recovery", WRIST_RECOVERY_ACTIONS
    return "contact_recovery", CONTACT_RECOVERY_ACTIONS


def select_orientation_phase(
        obs: Dict[str, Any], history: List[Dict[str, Any]],
        all_names: set[str]) -> tuple[str, set[str], bool]:
    """Select a safety-bounded action set from full-orientation state."""
    if bool(obs.get("success")):
        return "success_ready", all_names, False

    yaw_abs = abs(float(obs.get("err_deg", 180.0)))
    rot_dist = float(obs.get("rot_dist_rad", 3.1416))
    if yaw_abs <= 5.73 and rot_dist >= 0.1:
        return "tilt_correction", TILT_CORRECTION_ACTIONS & all_names, True
    if yaw_abs <= 10.0:
        return "fine_orientation", FINE_ORIENTATION_ACTIONS & all_names, True
    if yaw_abs <= 30.0 and history:
        if history[-1].get("choice") in COARSE_ROTATION_ACTIONS:
            return "transition_break", all_names - COARSE_ROTATION_ACTIONS, True

    advisory_mode, _ = select_action_mode(history)
    return advisory_mode, all_names, False


def build_action_stats(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize all measured action outcomes in the current episode."""
    grouped: Dict[str, List[tuple[Dict[str, Any], float]]] = {}
    previous_drift = 0.0
    for item in history:
        choice = item.get("choice")
        current_drift = float(item.get("drift_xy_m", previous_drift))
        position_gain = previous_drift - current_drift
        if isinstance(choice, str):
            grouped.setdefault(choice, []).append((item, position_gain))
        previous_drift = current_drift

    stats: Dict[str, Any] = {}
    best_rot_dist = min((
        float(item.get("rot_dist_after_rad", item.get("rot_dist_rad", 99.0)))
        for item in history
    ), default=None)
    current_rot_dist = (float(history[-1].get(
        "rot_dist_after_rad", history[-1].get("rot_dist_rad", 99.0)))
        if history else None)
    if best_rot_dist is not None and current_rot_dist is not None:
        best_index = min(
            range(len(history)),
            key=lambda index: float(history[index].get(
                "rot_dist_after_rad",
                history[index].get("rot_dist_rad", 99.0))))
        stats["_episode_context"] = {
            "best_rot_dist_rad": round(best_rot_dist, 4),
            "current_rot_dist_rad": round(current_rot_dist, 4),
            "regression_from_best_rad": round(
                current_rot_dist - best_rot_dist, 4),
            "cycles_since_best": len(history) - 1 - best_index,
            "near_episode_best": current_rot_dist <= best_rot_dist + 0.1,
        }
    for choice, samples in grouped.items():
        items = [sample[0] for sample in samples]
        position_gains = [sample[1] for sample in samples]
        gains = [measured_gain(item) for item in items]
        deltas = [float(item.get("measured_delta_deg", 0.0)) for item in items]
        rotation_gains = [
            float(item.get("rot_dist_gain_rad", 0.0)) for item in items]
        tilt_gains = [float(item.get("tilt_gain_deg", 0.0)) for item in items]
        axis_gains = [
            item.get("rotation_error_gain_xyz_deg", [0.0, 0.0, 0.0])
            for item in items
        ]
        recent_items = items[-3:]
        recent_rotation_gains = rotation_gains[-3:]
        recent_axis_gains = axis_gains[-3:]
        recent_weights = list(range(1, len(recent_rotation_gains) + 1))
        recent_weighted_gain = sum(
            weight * gain for weight, gain in zip(
                recent_weights, recent_rotation_gains)) / sum(recent_weights)
        recent_negative_streak = 0
        for gain in reversed(rotation_gains):
            if gain >= 0.0:
                break
            recent_negative_streak += 1
        last_history_index = max(
            index for index, history_item in enumerate(history)
            if history_item.get("choice") == choice)
        consecutive_recent_uses = 0
        for history_item in reversed(history):
            if history_item.get("choice") != choice:
                break
            consecutive_recent_uses += 1
        stats[choice] = {
            "uses": len(items),
            "helpful_uses": sum(gain > 0.0 for gain in gains),
            "mean_error_gain_deg": round(sum(gains) / len(gains), 3),
            "mean_yaw_delta_deg": round(sum(deltas) / len(deltas), 3),
            "last_error_gain_deg": round(gains[-1], 3),
            "mean_rot_dist_gain_rad": round(
                sum(rotation_gains) / len(rotation_gains), 4),
            "last_rot_dist_gain_rad": round(rotation_gains[-1], 4),
            "recent3_mean_rot_dist_gain_rad": round(
                sum(recent_rotation_gains) / len(recent_rotation_gains), 4),
            "recent_weighted_rot_dist_gain_rad": round(
                recent_weighted_gain, 4),
            "last_use_harmful": rotation_gains[-1] < 0.0,
            "recent_negative_streak": recent_negative_streak,
            "cycles_since_last_use": len(history) - 1 - last_history_index,
            "rot_dist_gain_trend": round(
                rotation_gains[-1] - rotation_gains[-2], 4)
                if len(rotation_gains) >= 2 else None,
            "consecutive_recent_uses": consecutive_recent_uses,
            "stalled_after_repetition": (
                len(items) >= 3 and rotation_gains[-1] < 0.01),
            "mean_tilt_gain_deg": round(
                sum(tilt_gains) / len(tilt_gains), 3),
            "last_tilt_gain_deg": round(tilt_gains[-1], 3),
            "mean_rotation_error_gain_xyz_deg": [
                round(sum(float(gain[axis]) for gain in axis_gains)
                      / len(axis_gains), 3)
                for axis in range(3)
            ],
            "last_rotation_error_gain_xyz_deg": axis_gains[-1],
            "recent3_mean_rotation_error_gain_xyz_deg": [
                round(sum(float(gain[axis]) for gain in recent_axis_gains)
                      / len(recent_axis_gains), 3)
                for axis in range(3)
            ],
            "helpful_position_uses": sum(gain > 0.0 for gain in position_gains),
            "mean_position_gain_m": round(
                sum(position_gains) / len(position_gains), 4),
            "last_position_gain_m": round(position_gains[-1], 4),
            "last_drift_xy_m": items[-1].get("drift_xy_m"),
        }
    return stats


def parse_gpt_response(content: str, valid_choices: List[str]) -> Dict[str, Any]:
    """Parse + validate a GPT reply. Pure function (offline-testable).

    Accepts raw JSON or JSON inside a ``` fence. Rejects unknown choices and
    malformed payloads with PolicyError.
    """
    text = content.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as e:
        raise PolicyError(f"GPT response is not valid JSON: {e}") from e
    if not isinstance(payload, dict):
        raise PolicyError("GPT response JSON must be an object")
    choice = payload.get("choice")
    if not isinstance(choice, str) or choice not in valid_choices:
        raise PolicyError(
            f"invalid choice {choice!r}; valid: {sorted(valid_choices)}")
    rationale = payload.get("rationale", "")
    if not isinstance(rationale, str):
        rationale = str(rationale)
    reported_phase = payload.get("recommended_phase", "")
    if not isinstance(reported_phase, str):
        reported_phase = str(reported_phase)
    follows_rules = payload.get("follows_rules")
    if not isinstance(follows_rules, bool):
        follows_rules = None
    return {
        "choice": choice,
        "rationale": rationale,
        "model_reported_phase": reported_phase,
        "follows_rules": follows_rules,
    }


class GPTPolicy(Policy):
    """OpenAI-compatible /chat/completions policy (httpx, synchronous).

    Config via environment variables only:
      INHAND_API_BASE (default https://api.openai.com/v1)
      INHAND_API_KEY  (required; missing -> immediate PolicyError, no fallback)
      INHAND_MODEL    (default gpt-6)
    """

    name = "gpt"

    def __init__(self, dry_run: bool = False, timeout_s: float = 300.0,
                 deadband_deg: float = 3.0):
        self.api_base = os.environ.get(ENV_API_BASE, DEFAULT_API_BASE).rstrip("/")
        self.model = os.environ.get(ENV_MODEL, DEFAULT_MODEL)
        self.api_key = os.environ.get(ENV_API_KEY)
        self.dry_run = dry_run
        self.timeout_s = timeout_s
        self.deadband_deg = deadband_deg
        if not self.dry_run and not self.api_key:
            raise PolicyError(
                f"{ENV_API_KEY} is not set. GPT policy requires an API key; "
                "refusing to fall back to the rule baseline (research integrity). "
                f"Set {ENV_API_KEY} (and optionally {ENV_API_BASE}, {ENV_MODEL}) or "
                "use --policy baseline / --dry-run.")

    def valid_choices(self, menu: List[Dict[str, Any]]) -> List[str]:
        return [m["name"] for m in menu] + ["finish"]

    def choose(self, obs: Dict[str, Any], history: List[Dict[str, Any]],
               menu: List[Dict[str, Any]]) -> Dict[str, str]:
        prompt_history, prompt_menu, name_by_alias = anonymize_action_interface(
            history, menu)
        selection_mode = "uniform_full_menu"
        prompt_menu = add_action_guidance(prompt_menu, name_by_alias)
        wrist_lower_action = next((
            alias for alias, name in name_by_alias.items()
            if name == "wrist_lower_reseat"), "not available")
        action_stats = build_action_stats(prompt_history)
        tried = set(action_stats)
        untried_actions = [
            item["name"] for item in prompt_menu if item["name"] not in tried
        ]
        messages = build_messages(
            obs, prompt_history, prompt_menu, self.model,
            selection_mode=selection_mode,
            action_stats=action_stats,
            untried_actions=untried_actions,
            wrist_lower_action=wrist_lower_action,
        )
        if self.dry_run:
            print("=== GPT dry-run: request NOT sent. Full prompt follows ===")
            for m in messages:
                print(f"--- [{m['role']}] ---")
                print(m["content"])
            print("=== end of dry-run prompt ===")
            return {
                "choice": "finish", "rationale": "dry-run: no request sent",
                "selection_mode": f"soft:{selection_mode}",
                "recommended_phase": selection_mode,
                "model_reported_phase": selection_mode,
                "follows_rules": True,
            }

        import httpx  # local import: baseline runs must not require httpx

        body = {
            "model": self.model,
            "messages": messages,
            "temperature": 1.0,
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        valid_choices = [item["name"] for item in prompt_menu] + ["finish"]
        try:
            resp = httpx.post(
                f"{self.api_base}/chat/completions",
                json=body, headers=headers, timeout=self.timeout_s)
            resp.raise_for_status()
        except Exception as e:
            raise PolicyError(f"GPT API request failed: {e}") from e
        try:
            content = resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            raise PolicyError(f"unexpected GPT API payload: {e}") from e
        decision = parse_gpt_response(content, valid_choices)
        if decision["choice"] != "finish":
            decision["choice"] = name_by_alias[decision["choice"]]
        decision["selection_mode"] = f"soft:{selection_mode}"
        decision["recommended_phase"] = selection_mode
        return decision


# --- offline self-test ---------------------------------------------------------

def _self_test() -> int:
    valid = ["roll_cw", "roll_ccw", "hold", "finish"]
    ok = 0

    good = '{"choice": "roll_cw", "rationale": "err is negative"}'
    assert parse_gpt_response(good, valid)["choice"] == "roll_cw"; ok += 1

    fenced = '```json\n{"choice": "hold", "rationale": "stabilize"}\n```'
    assert parse_gpt_response(fenced, valid)["choice"] == "hold"; ok += 1

    bad_json = "I think you should roll clockwise."
    try:
        parse_gpt_response(bad_json, valid)
        raise AssertionError("malformed text was accepted")
    except PolicyError:
        ok += 1

    bad_choice = '{"choice": "spin_fast", "rationale": "x"}'
    try:
        parse_gpt_response(bad_choice, valid)
        raise AssertionError("invalid choice was accepted")
    except PolicyError:
        ok += 1

    not_obj = '["roll_cw"]'
    try:
        parse_gpt_response(not_obj, valid)
        raise AssertionError("non-object JSON was accepted")
    except PolicyError:
        ok += 1

    # missing key must raise immediately (no silent fallback), even in dry-run? no:
    # dry-run is allowed without a key; non-dry-run must fail fast.
    saved = os.environ.pop(ENV_API_KEY, None)
    try:
        GPTPolicy(dry_run=False)
        raise AssertionError("missing API key was accepted")
    except PolicyError:
        ok += 1
    GPTPolicy(dry_run=True)  # must work without key
    ok += 1
    if saved is not None:
        os.environ[ENV_API_KEY] = saved

    print(f"policy self-test: {ok}/7 checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
