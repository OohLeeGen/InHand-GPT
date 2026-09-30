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
performing in-hand reorientation of a block about the world Z axis (MuJoCo sim).

Each cycle you receive a structured observation and must pick exactly ONE
motion primitive from the menu (or "finish").

Observation semantics:
- err_deg: signed achieved-minus-desired rotation error about +Z in degrees.
  Positive error requires negative yaw; negative error requires positive yaw.
  Success requires |err| < 5.73 deg (0.1 rad).
- rot_dist_rad: unsigned geodesic rotation distance to the goal (rad).
- drift_xy_m / drift_z_m: block-center displacement from episode start.
  Horizontal drift above 0.05 m or vertical drop below -0.05 m ends the
  episode as dropped. Large drift means the grasp is unstable.
- last_action / last_delta_deg: what was executed last cycle and the measured
  signed yaw change it produced (deg). Use this feedback to adapt.

Current observation (JSON):
{observation}

Recent history (oldest first, JSON):
{history}

Recommended mode (advisory only; all menu actions remain legal): {selection_mode}

Episode-long measured action statistics (JSON):
{action_stats}

Untried actions in the active menu (JSON):
{untried_actions}

Primitive menu (JSON):
{menu}

Important:
- All listed actions are legal. Treat category, use_when, and avoid_when as
  guidance rather than guaranteed effects or hard constraints.
- Positive/negative joint directions in descriptions are actuator-coordinate
  directions only; they do NOT predict block yaw direction.
- Trust measured episode statistics over action wording. Avoid consecutive
  recovery actions unless the first produced a measured position improvement.

Reply with ONLY a JSON object, no markdown, no extra text:
{{"choice": "<primitive name or 'finish'>", "rationale": "<one sentence>"}}
"""


def build_messages(obs: Dict[str, Any], history: List[Dict[str, Any]],
                   menu: List[Dict[str, Any]], model: str,
                   selection_mode: str = "normal_rotation",
                   action_stats: Dict[str, Any] | None = None,
                   untried_actions: List[str] | None = None) -> List[Dict[str, str]]:
    prompt = PROMPT_TEMPLATE.format(
        observation=json.dumps(obs, ensure_ascii=False),
        history=json.dumps(history[-5:], ensure_ascii=False),
        selection_mode=selection_mode,
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
NORMAL_ACTIONS = {
    "hold", "roll_cw", "roll_ccw", "roll_lf", "roll_th", "roll_wr",
    "nudge_cw", "nudge_ccw",
} | MICRO_ACTIONS | LATERAL_SWEEP_ACTIONS
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


def action_guidance(name: str) -> Dict[str, str]:
    """Advisory conditions shown to GPT; never used as a hard filter."""
    if name == "hold":
        return {
            "category": "stabilization",
            "use_when": "immediately after an abrupt motion or unstable drift",
            "avoid_when": "large error remains and another action is improving it",
        }
    if name in {"nudge_cw", "nudge_ccw"}:
        return {
            "category": "fine_rotation",
            "use_when": "rotation error is small or a coarse action overshoots",
            "avoid_when": "large error remains and a coarse action is effective",
        }
    if name == "roll_lf":
        return {
            "category": "alternate_contact_rotation",
            "use_when": "a different contact pattern is needed and its most recent measured gain is positive",
            "avoid_when": "it ran twice consecutively or its recent measured gain is below 0.5 deg",
        }
    if name in LATERAL_SWEEP_ACTIONS:
        return {
            "category": "contact_shift",
            "use_when": "repeated ring/thumb rotation gains have decayed while object drift remains stable",
            "avoid_when": "the object is positionally unstable or the same sweep direction recently worsened contact",
        }
    if name in MICRO_ACTIONS:
        return {
            "category": "small_object_micro",
            "use_when": "the object/contact scale is small, a coarse motion is excessive, or a fine local contact adjustment is needed",
            "avoid_when": "large error remains and a coarse action is producing stable progress",
        }
    if name in NORMAL_ACTIONS:
        return {
            "category": "coarse_rotation",
            "use_when": "contact is stable and rotation error remains large",
            "avoid_when": "its recent measured gain is below 0.5 deg repeatedly",
        }
    if name in CONTACT_RECOVERY_ACTIONS:
        return {
            "category": "contact_recovery",
            "use_when": "at least two recent rotation actions stalled or worsened",
            "avoid_when": "a rotation action is still improving or recovery just ran",
        }
    return {
        "category": "position_recovery",
        "use_when": "drift exceeds 0.02 m and rotation recently stalled, or contact recovery failed",
        "avoid_when": "drift is low, rotation is improving, or its measured position gain is negative",
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
    for choice, samples in grouped.items():
        items = [sample[0] for sample in samples]
        position_gains = [sample[1] for sample in samples]
        gains = [measured_gain(item) for item in items]
        deltas = [float(item.get("measured_delta_deg", 0.0)) for item in items]
        stats[choice] = {
            "uses": len(items),
            "helpful_uses": sum(gain > 0.0 for gain in gains),
            "mean_error_gain_deg": round(sum(gains) / len(gains), 3),
            "mean_yaw_delta_deg": round(sum(deltas) / len(deltas), 3),
            "last_error_gain_deg": round(gains[-1], 3),
            "helpful_position_uses": sum(gain > 0.0 for gain in position_gains),
            "mean_position_gain_m": round(
                sum(position_gains) / len(position_gains), 4),
            "last_position_gain_m": round(position_gains[-1], 4),
            "last_drift_xy_m": items[-1].get("drift_xy_m"),
        }
    return stats


def parse_gpt_response(content: str, valid_choices: List[str]) -> Dict[str, str]:
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
    return {"choice": choice, "rationale": rationale}


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
        selection_mode, _ = select_action_mode(history)
        prompt_menu = add_action_guidance(prompt_menu, name_by_alias)
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
        )
        if self.dry_run:
            print("=== GPT dry-run: request NOT sent. Full prompt follows ===")
            for m in messages:
                print(f"--- [{m['role']}] ---")
                print(m["content"])
            print("=== end of dry-run prompt ===")
            return {"choice": "finish", "rationale": "dry-run: no request sent",
                    "selection_mode": f"soft:{selection_mode}"}

        import httpx  # local import: baseline runs must not require httpx

        body = {
            "model": self.model,
            "messages": messages,
            "temperature": 1.0,
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            resp = httpx.post(f"{self.api_base}/chat/completions",
                              json=body, headers=headers, timeout=self.timeout_s)
            resp.raise_for_status()
        except Exception as e:
            raise PolicyError(f"GPT API request failed: {e}") from e
        try:
            content = resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            raise PolicyError(f"unexpected GPT API payload: {e}") from e
        decision = parse_gpt_response(
            content, [item["name"] for item in prompt_menu] + ["finish"])
        if decision["choice"] != "finish":
            decision["choice"] = name_by_alias[decision["choice"]]
        decision["selection_mode"] = f"soft:{selection_mode}"
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
