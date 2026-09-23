"""Shadow Dexterous Hand motion primitives for in-hand block rotation (RotateZ).

Each primitive is a short open-loop sequence of *relative* joint-delta actions
(the env applies `ctrl = current_qpos + action * actuation_range` each step, so
an action entry behaves like a normalized joint velocity in [-1, 1]; a constant
"bias" ramps the joint toward its limit, which is how the deep-flexion drag
works).

Sign convention: "yaw" = signed block rotation about world +Z (CCW seen from
above). roll_cw aims at negative yaw, roll_ccw at positive yaw.

DESIGN NOTE (closed-loop evidence, 2026-09-22): bias-only "deep drag" waveforms
saturate the dragged joints at their caps after 1-3 executions and then stall
completely (roll_cw stalled at block yaw == +54deg mod 90 on two independent
seeds). The primary roll primitives are therefore COMPOUND: a drag segment
(bias + wave) followed by a joint-return segment that brings the dragged
joints back to mid-range, so every execution starts unsaturated and stays
effective on repeated use. The return segment may refund a little yaw; the
frozen numbers below are NET effects.

MEASURED (scripts/tune_primitives.py --scan pkg, seeds 0-4, frozen registry;
rolls = 30 steps, nudges = 12 steps, hold/regrasp = 20 steps; yaw deg / max
drift cm, single execution from reset). The sign map is contact-geometry
dependent — the decision layer must use measured feedback (see DESIGN.md):

  hold     : -3.3/0.2  -0.2/0.1  -1.1/0.1  -0.4/0.1  +2.5/0.1
  roll_cw  : -15.0/0.8 -11.9/1.3 -10.1/1.5 +0.3/1.5  +0.9/0.4
  roll_ccw : +6.3/1.1  -2.8/0.9  -3.4/0.9  +10.1/0.3 +2.2/0.8
  roll_lf  : -1.6/0.9  +15.7/1.0 +16.1/1.1 +14.5/0.6 -0.0/0.4
  roll_th  : -9.8/0.8  -6.4/0.5  -7.0/0.6  +1.6/0.9  +3.5/0.3
  roll_wr  : -13.7/1.8 -13.2/1.4 -13.5/1.4 +4.0/1.7  -2.2/1.4
  regrasp  : -1.9/1.2  -2.6/1.0  -4.7/1.1  +9.3/1.1  +2.0/0.5
  nudge_cw : -2.9/0.1  -4.5/0.7  -4.8/0.8  +6.6/0.6  +2.2/0.2
  nudge_ccw: +1.5/0.9  -1.1/0.5  -1.4/0.5  +4.5/0.3  +1.8/0.4

Closed-loop findings that shaped this design (2026-09-22):
- Bias-only "deep drag" waveforms saturate the dragged joints at their caps
  after 1-3 executions and then stall completely; the compound drag+return
  shape above stays effective on repeated use.
- Block yaw == +54deg (mod 90) is a contact-mode attractor ("pocket") where
  every gait decays to zero net rotation; see DESIGN.md section 5.
Do not edit waveforms without re-running the tuner.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np

# Actuator order of HandManipulateBlock* (without the "robot0:A_" prefix).
ACTUATOR_NAMES = [
    "WRJ1", "WRJ0",
    "FFJ3", "FFJ2", "FFJ1",
    "MFJ3", "MFJ2", "MFJ1",
    "RFJ3", "RFJ2", "RFJ1",
    "LFJ4", "LFJ3", "LFJ2", "LFJ1",
    "THJ4", "THJ3", "THJ2", "THJ1", "THJ0",
]
IDX = {name: i for i, name in enumerate(ACTUATOR_NAMES)}
N_ACTIONS = len(ACTUATOR_NAMES)

_INF = 10 ** 9  # "hold bias for the whole segment"


@dataclass
class Primitive:
    """A named, fixed-length open-loop action generator."""

    name: str
    n_steps: int
    action_fn: Callable[[int], np.ndarray]
    description: str = ""
    tags: List[str] = field(default_factory=list)

    def action(self, t: int) -> np.ndarray:
        return np.clip(self.action_fn(t), -1.0, 1.0).astype(np.float32)


def _pattern(
    entries: Dict[str, tuple],
    period: int,
) -> Callable[[int], np.ndarray]:
    """entries: {actuator: (amplitude, phase_rad, bias, bias_steps)}."""

    def fn(t: int) -> np.ndarray:
        a = np.zeros(N_ACTIONS, dtype=np.float64)
        for name, (amp, phase, bias, bias_steps) in entries.items():
            a[IDX[name]] = (bias if t < bias_steps else 0.0) + \
                amp * math.sin(2.0 * math.pi * t / period + phase)
        return a

    return fn


def _compound(
    roll_entries: Dict[str, tuple],
    period: int,
    roll_steps: int,
    return_joints: Dict[str, float],
) -> Callable[[int], np.ndarray]:
    """Drag segment (roll_entries) + joint-return segment (return_joints).

    return_joints maps actuator -> constant action applied for the rest of the
    primitive; amplitudes are chosen to bring the dragged (capped) joints back
    to mid-range so the primitive is repeatable.
    """
    roll_fn = _pattern(roll_entries, period)

    def fn(t: int) -> np.ndarray:
        if t < roll_steps:
            return roll_fn(t)
        a = np.zeros(N_ACTIONS, dtype=np.float64)
        for name, v in return_joints.items():
            a[IDX[name]] = v
        return a

    return fn


def _zeros(t: int) -> np.ndarray:
    return np.zeros(N_ACTIONS, dtype=np.float64)


# --- primitive registry -------------------------------------------------------

PRIMITIVES: Dict[str, Primitive] = {}


def register(p: Primitive) -> None:
    PRIMITIVES[p.name] = p


def get(name: str) -> Primitive:
    if name not in PRIMITIVES:
        raise KeyError(f"unknown primitive {name!r}; available: {sorted(PRIMITIVES)}")
    return PRIMITIVES[name]


def menu() -> List[Dict[str, str]]:
    """Candidate menu handed to the GPT policy."""
    return [
        {"name": p.name, "description": p.description, "n_steps": p.n_steps}
        for p in PRIMITIVES.values()
    ]


# --- built-in primitives (waveforms frozen from tuner measurements) -----------

register(Primitive(
    name="hold",
    n_steps=20,
    action_fn=_zeros,
    description="Zero action: keep the current grasp pose (passive settling only).",
    tags=["stable"],
))

# -yaw primary: forefinger deep-flexion drag + joint return (repeatable).
register(Primitive(
    name="roll_cw",
    n_steps=30,
    action_fn=_compound(
        {"FFJ1": (0.5, 0.0, 0.35, _INF), "FFJ2": (0.5, 0.0, 0.35, _INF)},
        period=9, roll_steps=18,
        return_joints={"FFJ1": -0.085, "FFJ2": -0.085},
    ),
    description="Forefinger flexion-drag then joint return; rolls the block "
                "clockwise (negative yaw about +Z), repeatable without joint "
                "saturation.",
    tags=["roll", "primary"],
))

# +yaw primary: middle+ring deep-flexion drag, thumb base opened + return.
register(Primitive(
    name="roll_ccw",
    n_steps=30,
    action_fn=_compound(
        {
            "MFJ1": (0.0, 0.0, 0.45, _INF), "MFJ2": (0.0, 0.0, 0.45, _INF),
            "RFJ1": (0.0, 0.0, 0.45, _INF), "RFJ2": (0.0, 0.0, 0.45, _INF),
            "THJ4": (0.0, 0.0, 0.4, _INF),
        },
        period=10, roll_steps=18,
        return_joints={"MFJ1": -0.085, "MFJ2": -0.085,
                       "RFJ1": -0.085, "RFJ2": -0.085, "THJ4": -0.08},
    ),
    description="Middle+ring deep-flexion drag with thumb base opened, then "
                "joint return; rolls the block counter-clockwise (positive "
                "yaw), repeatable without joint saturation.",
    tags=["roll", "primary"],
))

# Wildcard gait: little-finger base joint drive against a ring-finger cage
# (zero net excursion on the driver joint; sustainable, weaker per cycle).
register(Primitive(
    name="roll_lf",
    n_steps=30,
    action_fn=_pattern({
        "LFJ4": (0.6, 0.0, 0.0, 0),
        "LFJ1": (0.4, -1.57, 0.0, 0),
        "LFJ2": (0.4, -1.57, 0.0, 0),
        "RFJ1": (0.0, 0.0, 0.3, _INF),
        "RFJ2": (0.0, 0.0, 0.3, _INF),
    }, period=10),
    description="Little-finger base joint (LFJ4) drive against a ring-finger "
                "cage; sustainable rolling gait, positive yaw on seed-1/2/3 "
                "geometry (+15deg/30steps), near zero on seeds 0/4.",
    tags=["roll", "wildcard"],
))

# Wildcard gait: thumb base sweep with a two-finger cage.
register(Primitive(
    name="roll_th",
    n_steps=30,
    action_fn=_pattern({
        "THJ0": (0.8, 0.0, 0.0, 0),
        "THJ3": (0.5, -1.57, 0.0, 0),
        "FFJ1": (0.0, 0.0, 0.2, _INF),
        "FFJ2": (0.0, 0.0, 0.2, _INF),
        "MFJ1": (0.0, 0.0, 0.2, _INF),
        "MFJ2": (0.0, 0.0, 0.2, _INF),
    }, period=12),
    description="Thumb base sweep against a fore+middle finger cage; "
                "alternative rolling gait (sign is geometry-dependent: "
                "-7..-10deg on seeds 0-2, +3.5deg on seed 4 per 30 steps).",
    tags=["roll", "wildcard"],
))

# Wildcard gait: wrist ulnar/radial rock with a two-finger cage (pocket escape).
register(Primitive(
    name="roll_wr",
    n_steps=30,
    action_fn=_pattern({
        "WRJ1": (0.5, 0.0, 0.0, 0),
        "FFJ1": (0.0, 0.0, 0.2, _INF),
        "FFJ2": (0.0, 0.0, 0.2, _INF),
        "MFJ1": (0.0, 0.0, 0.2, _INF),
        "MFJ2": (0.0, 0.0, 0.2, _INF),
    }, period=12),
    description="Wrist WRJ1 rock against a fore+middle finger cage; changes the "
                "relative block-finger angle, used to escape contact-mode "
                "pockets (measured -3deg/use from the seed-1 pocket, drift ~1cm "
                "per 3 uses).",
    tags=["roll", "wildcard"],
))

# Circuit breaker: open-then-close regrasp to change the contact mode when
# every rolling gait has stalled. One full sine period: extend then re-flex.
register(Primitive(
    name="regrasp",
    n_steps=20,
    action_fn=_pattern({
        "FFJ1": (0.6, math.pi, 0.0, 0), "FFJ2": (0.6, math.pi, 0.0, 0),
        "MFJ1": (0.6, math.pi, 0.0, 0), "MFJ2": (0.6, math.pi, 0.0, 0),
        "RFJ1": (0.6, math.pi, 0.0, 0), "RFJ2": (0.6, math.pi, 0.0, 0),
    }, period=20),
    description="Open then re-close FF/MF/RF to reset the contact mode; small "
                "yaw disturbance, ~1cm drift. Use when rolling gaits stall.",
    tags=["recovery"],
))

# Fine mode near the goal: same mechanisms, short bursts with joint return.
register(Primitive(
    name="nudge_cw",
    n_steps=12,
    action_fn=_compound(
        {"FFJ1": (0.3, 0.0, 0.3, _INF), "FFJ2": (0.3, 0.0, 0.3, _INF)},
        period=6, roll_steps=6,
        return_joints={"FFJ1": -0.16, "FFJ2": -0.16},
    ),
    description="Short, weak roll_cw for fine corrections near the goal.",
    tags=["nudge"],
))
register(Primitive(
    name="nudge_ccw",
    n_steps=12,
    action_fn=_compound(
        {"MFJ1": (0.0, 0.0, 0.45, _INF), "MFJ2": (0.0, 0.0, 0.45, _INF),
         "THJ4": (0.0, 0.0, 0.4, _INF)},
        period=6, roll_steps=6,
        return_joints={"MFJ1": -0.16, "MFJ2": -0.16, "THJ4": -0.16},
    ),
    description="Short, weak roll_ccw for fine corrections near the goal.",
    tags=["nudge"],
))
