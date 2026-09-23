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
- err_deg: signed rotation error about +Z in degrees. Positive means the block
  must rotate further about +Z. Success requires |err| < 5.73 deg (0.1 rad).
- rot_dist_rad: unsigned geodesic rotation distance to the goal (rad).
- drift_xy_m / drift_z_m: block-center displacement from episode start. Large
  drift means the grasp is unstable; prefer "hold" to re-stabilize.
- last_action / last_delta_deg: what was executed last cycle and the measured
  signed yaw change it produced (deg). Use this feedback to adapt.

Current observation (JSON):
{observation}

Recent history (oldest first, JSON):
{history}

Primitive menu (JSON):
{menu}

Reply with ONLY a JSON object, no markdown, no extra text:
{{"choice": "<primitive name or 'finish'>", "rationale": "<one sentence>"}}
"""


def build_messages(obs: Dict[str, Any], history: List[Dict[str, Any]],
                   menu: List[Dict[str, Any]], model: str) -> List[Dict[str, str]]:
    prompt = PROMPT_TEMPLATE.format(
        observation=json.dumps(obs, ensure_ascii=False),
        history=json.dumps(history[-5:], ensure_ascii=False),
        menu=json.dumps(menu, ensure_ascii=False),
    )
    return [
        {"role": "system", "content": "You are a careful robot control policy. Always answer with strict JSON."},
        {"role": "user", "content": prompt},
    ]


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

    def __init__(self, dry_run: bool = False, timeout_s: float = 60.0,
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
        messages = build_messages(obs, history, menu, self.model)
        if self.dry_run:
            print("=== GPT dry-run: request NOT sent. Full prompt follows ===")
            for m in messages:
                print(f"--- [{m['role']}] ---")
                print(m["content"])
            print("=== end of dry-run prompt ===")
            return {"choice": "finish", "rationale": "dry-run: no request sent"}

        import httpx  # local import: baseline runs must not require httpx

        body = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
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
        return parse_gpt_response(content, self.valid_choices(menu))


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
