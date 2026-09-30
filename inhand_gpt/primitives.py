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


def _staged(
    segments: List[tuple[int, Dict[str, float]]],
) -> Callable[[int], np.ndarray]:
    """Piecewise-constant relative commands for release/transport/regrasp."""
    boundaries: List[tuple[int, Dict[str, float]]] = []
    end = 0
    for n_steps, commands in segments:
        end += n_steps
        boundaries.append((end, commands))

    def fn(t: int) -> np.ndarray:
        a = np.zeros(N_ACTIONS, dtype=np.float64)
        for end_step, commands in boundaries:
            if t < end_step:
                for name, value in commands.items():
                    a[IDX[name]] = value
                break
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
    description="Send zero commands to all controlled joints for 20 steps.",
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
    description="Flex forefinger joints FFJ1 and FFJ2 with a forward-biased "
                "periodic command, then apply a small reverse command to the "
                "same joints.",
    tags=["roll", "primary"],
))

# +yaw primary variant: ring-finger deep-flexion drag with thumb-base motion;
# middle-finger channels intentionally remain at zero for ablation testing.
register(Primitive(
    name="roll_ccw",
    n_steps=30,
    action_fn=_compound(
        {
            "RFJ1": (0.0, 0.0, 0.45, _INF), "RFJ2": (0.0, 0.0, 0.45, _INF),
            "THJ4": (0.0, 0.0, 0.4, _INF),
        },
        period=10, roll_steps=18,
        return_joints={"RFJ1": -0.085, "RFJ2": -0.085, "THJ4": -0.08},
    ),
    description="Flex ring-finger joints RFJ1/RFJ2 while moving THJ4 in the "
                "same command direction, keep the middle-finger channels at "
                "zero, then apply small reverse commands to RFJ1/RFJ2 and "
                "THJ4.",
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
    description="Oscillate little-finger joint LFJ4 while LFJ1/LFJ2 follow "
                "phase-shifted oscillations, with a small flexion bias on "
                "ring-finger joints RFJ1/RFJ2.",
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
    description="Oscillate thumb joints THJ0 and THJ3 with a phase offset "
                "while applying a small flexion bias to forefinger and "
                "middle-finger joints FFJ1/FFJ2 and MFJ1/MFJ2.",
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
    description="Oscillate wrist joint WRJ1 while applying a small flexion "
                "bias to forefinger and middle-finger joints FFJ1/FFJ2 and "
                "MFJ1/MFJ2.",
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
    description="Extend and then flex forefinger, middle-finger, and "
                "ring-finger joints FFJ1/FFJ2, MFJ1/MFJ2, and RFJ1/RFJ2 over "
                "one movement cycle.",
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
    description="Briefly flex forefinger joints FFJ1 and FFJ2, then apply a "
                "reverse command to the same joints.",
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
    description="Briefly flex middle-finger joints MFJ1/MFJ2 while moving "
                "THJ4 in the same command direction, then apply reverse "
                "commands to those joints.",
    tags=["nudge"],
))

# Staged wrist-flexion reseat: release, tilt, settle, re-close, return.
register(Primitive(
    name="wrist_lower_reseat",
    n_steps=30,
    action_fn=_staged([
        (5, {"FFJ1": -0.10, "FFJ2": -0.10,
             "MFJ1": -0.10, "MFJ2": -0.10}),
        (6, {"WRJ0": -0.12, "RFJ1": 0.04, "RFJ2": 0.04}),
        (6, {}),
        (7, {"FFJ1": 0.11, "FFJ2": 0.11, "MFJ1": 0.11, "MFJ2": 0.11,
             "RFJ1": 0.04, "RFJ2": 0.04}),
        (6, {"WRJ0": 0.12, "FFJ1": 0.02, "FFJ2": 0.02,
             "MFJ1": 0.02, "MFJ2": 0.02}),
    ]),
    description="Extend the forefinger and middle-finger pairs, move WRJ0 in "
                "one direction, hold briefly, flex the fingers at the new "
                "contact, then return WRJ0 while maintaining flexion.",
    tags=["recovery", "wrist"],
))

# Mirrored staged WRJ0 motion.
register(Primitive(
    name="wrist_raise_reseat",
    n_steps=34,
    action_fn=_staged([
        (5, {"FFJ1": -0.10, "FFJ2": -0.10,
             "MFJ1": -0.10, "MFJ2": -0.10}),
        (8, {"WRJ0": 0.16, "RFJ1": 0.04, "RFJ2": 0.04}),
        (8, {}),
        (7, {"FFJ1": 0.11, "FFJ2": 0.11, "MFJ1": 0.11, "MFJ2": 0.11,
             "RFJ1": 0.04, "RFJ2": 0.04}),
        (6, {"WRJ0": -0.16, "FFJ1": 0.02, "FFJ2": 0.02,
             "MFJ1": 0.02, "MFJ2": 0.02}),
    ]),
    description="Extend the forefinger and middle-finger pairs, move WRJ0 in "
                "the opposite direction, hold briefly, flex the fingers at "
                "the new contact, then return WRJ0 while maintaining flexion.",
    tags=["recovery", "wrist"],
))

# Independent contact-release cycles for each finger group.
register(Primitive(
    name="forefinger_regrasp",
    n_steps=20,
    action_fn=_pattern({
        "FFJ1": (0.45, math.pi, 0.0, 0),
        "FFJ2": (0.45, math.pi, 0.0, 0),
    }, period=19),
    description="Extend and then flex forefinger joints FFJ1 and FFJ2 while "
                "the other controlled joints receive zero commands.",
    tags=["recovery", "finger"],
))
register(Primitive(
    name="middle_finger_regrasp",
    n_steps=20,
    action_fn=_pattern({
        "MFJ1": (0.45, math.pi, 0.0, 0),
        "MFJ2": (0.45, math.pi, 0.0, 0),
    }, period=19),
    description="Extend and then flex middle-finger joints MFJ1 and MFJ2 "
                "while the other controlled joints receive zero commands.",
    tags=["recovery", "finger"],
))
register(Primitive(
    name="ring_finger_regrasp",
    n_steps=20,
    action_fn=_pattern({
        "RFJ1": (0.45, math.pi, 0.0, 0),
        "RFJ2": (0.45, math.pi, 0.0, 0),
    }, period=19),
    description="Extend and then flex ring-finger joints RFJ1 and RFJ2 while "
                "the other controlled joints receive zero commands.",
    tags=["recovery", "finger"],
))
register(Primitive(
    name="little_finger_regrasp",
    n_steps=20,
    action_fn=_pattern({
        "LFJ1": (0.45, math.pi, 0.0, 0),
        "LFJ2": (0.45, math.pi, 0.0, 0),
    }, period=19),
    description="Extend and then flex little-finger joints LFJ1 and LFJ2 "
                "while the other controlled joints receive zero commands.",
    tags=["recovery", "finger"],
))
register(Primitive(
    name="thumb_regrasp",
    n_steps=20,
    action_fn=_pattern({
        "THJ0": (0.50, math.pi, 0.0, 0),
        "THJ3": (0.35, math.pi, 0.0, 0),
    }, period=19),
    description="Move thumb joints THJ0 and THJ3 through a negative "
                "half-cycle and then return them while the other controlled "
                "joints receive zero commands.",
    tags=["recovery", "finger"],
))

# Coupled staged wrist tilt and three-finger release/re-flex.
register(Primitive(
    name="wrist_three_finger_regrasp",
    n_steps=30,
    action_fn=_staged([
        (5, {"FFJ1": -0.10, "FFJ2": -0.10, "MFJ1": -0.10,
             "MFJ2": -0.10, "RFJ1": -0.10, "RFJ2": -0.10}),
        (6, {"WRJ0": -0.10}),
        (6, {}),
        (7, {"FFJ1": 0.12, "FFJ2": 0.12, "MFJ1": 0.12,
             "MFJ2": 0.12, "RFJ1": 0.12, "RFJ2": 0.12}),
        (6, {"WRJ0": 0.10, "FFJ1": 0.02, "FFJ2": 0.02,
             "MFJ1": 0.02, "MFJ2": 0.02, "RFJ1": 0.02, "RFJ2": 0.02}),
    ]),
    description="Extend the forefinger, middle-finger, and ring-finger pairs, "
                "move WRJ0 and hold, flex all three pairs at the new contact, "
                "then return WRJ0 while maintaining flexion.",
    tags=["recovery", "wrist", "finger"],
))

# Small phase-offset motions to perturb several contacts without a net bias.
register(Primitive(
    name="multi_finger_dither",
    n_steps=22,
    action_fn=_pattern({
        "FFJ1": (0.18, 0.0, 0.0, 0),
        "FFJ2": (0.18, 0.0, 0.0, 0),
        "MFJ1": (0.18, 2 * math.pi / 3, 0.0, 0),
        "MFJ2": (0.18, 2 * math.pi / 3, 0.0, 0),
        "RFJ1": (0.18, 4 * math.pi / 3, 0.0, 0),
        "RFJ2": (0.18, 4 * math.pi / 3, 0.0, 0),
        "LFJ1": (0.16, math.pi, 0.0, 0),
        "LFJ2": (0.16, math.pi, 0.0, 0),
        "THJ0": (0.16, math.pi / 2, 0.0, 0),
    }, period=11),
    description="Apply small phase-offset oscillations to separate "
                "forefinger, middle-finger, ring-finger, little-finger, and "
                "thumb joint groups for two cycles.",
    tags=["recovery", "finger"],
))

# Four staged position-recovery variants. Their names encode implementation
# intent only; GPT receives anonymous aliases and kinematic descriptions.
register(Primitive(
    name="recenter_palmward",
    n_steps=30,
    action_fn=_staged([
        (5, {"FFJ1": -0.10, "FFJ2": -0.10, "MFJ1": -0.10,
             "MFJ2": -0.10, "RFJ1": -0.10, "RFJ2": -0.10}),
        (6, {"WRJ0": -0.14}),
        (6, {}),
        (7, {"FFJ1": 0.12, "FFJ2": 0.12, "MFJ1": 0.12,
             "MFJ2": 0.12, "RFJ1": 0.12, "RFJ2": 0.12}),
        (6, {"WRJ0": 0.14, "FFJ1": 0.02, "FFJ2": 0.02,
             "MFJ1": 0.02, "MFJ2": 0.02, "RFJ1": 0.02, "RFJ2": 0.02}),
    ]),
    description="Extend three finger pairs, move WRJ0 in direction A and hold, "
                "flex the finger pairs, then return WRJ0 while maintaining "
                "the new finger flexion.",
    tags=["recovery", "recenter"],
))
register(Primitive(
    name="recenter_fingertips",
    n_steps=30,
    action_fn=_staged([
        (5, {"FFJ1": -0.10, "FFJ2": -0.10, "MFJ1": -0.10,
             "MFJ2": -0.10, "RFJ1": -0.10, "RFJ2": -0.10}),
        (6, {"WRJ0": 0.14}),
        (6, {}),
        (7, {"FFJ1": 0.12, "FFJ2": 0.12, "MFJ1": 0.12,
             "MFJ2": 0.12, "RFJ1": 0.12, "RFJ2": 0.12}),
        (6, {"WRJ0": -0.14, "FFJ1": 0.02, "FFJ2": 0.02,
             "MFJ1": 0.02, "MFJ2": 0.02, "RFJ1": 0.02, "RFJ2": 0.02}),
    ]),
    description="Extend three finger pairs, move WRJ0 in direction B and hold, "
                "flex the finger pairs, then return WRJ0 while maintaining "
                "the new finger flexion.",
    tags=["recovery", "recenter"],
))
register(Primitive(
    name="recenter_forefinger_side",
    n_steps=30,
    action_fn=_staged([
        (5, {"FFJ1": -0.11, "FFJ2": -0.11}),
        (6, {"LFJ4": 0.12, "LFJ1": 0.08, "LFJ2": 0.08}),
        (6, {}),
        (7, {"FFJ1": 0.13, "FFJ2": 0.13}),
        (6, {"LFJ4": -0.12, "FFJ1": 0.02, "FFJ2": 0.02}),
    ]),
    description="Extend the forefinger pair, move LFJ4 with light LFJ1/LFJ2 "
                "flexion and hold, re-flex the forefinger, then return LFJ4 "
                "while maintaining the new forefinger flexion.",
    tags=["recovery", "recenter"],
))
register(Primitive(
    name="recenter_little_side",
    n_steps=30,
    action_fn=_staged([
        (5, {"LFJ1": -0.11, "LFJ2": -0.11}),
        (6, {"FFJ3": 0.12, "FFJ1": 0.08, "FFJ2": 0.08}),
        (6, {}),
        (7, {"LFJ1": 0.13, "LFJ2": 0.13}),
        (6, {"FFJ3": -0.12, "LFJ1": 0.02, "LFJ2": 0.02}),
    ]),
    description="Extend the little-finger pair, move FFJ3 with light "
                "FFJ1/FFJ2 flexion and hold, re-flex the little finger, then "
                "return FFJ3 while maintaining the new little-finger flexion.",
    tags=["recovery", "recenter"],
))

# Low-amplitude, short-horizon primitives for fine contact changes on objects
# that are small relative to the hand.  Keep these registrations after the
# original 22 primitives so their anonymous action_01..action_22 aliases stay
# stable in prompts and experiment logs.
register(Primitive(
    name="micro_forefinger_pulse",
    n_steps=10,
    action_fn=_staged([
        (4, {"FFJ1": 0.07, "FFJ2": 0.07}),
        (2, {}),
        (4, {"FFJ1": -0.05, "FFJ2": -0.05}),
    ]),
    description="Briefly flex the forefinger's FFJ1/FFJ2 pair, pause, then "
                "partly return that pair.",
    tags=["micro", "small_object"],
))
register(Primitive(
    name="micro_middle_finger_pulse",
    n_steps=10,
    action_fn=_staged([
        (4, {"MFJ1": 0.07, "MFJ2": 0.07}),
        (2, {}),
        (4, {"MFJ1": -0.05, "MFJ2": -0.05}),
    ]),
    description="Briefly flex the middle finger's MFJ1/MFJ2 pair, pause, "
                "then partly return that pair.",
    tags=["micro", "small_object"],
))
register(Primitive(
    name="micro_ring_finger_pulse",
    n_steps=10,
    action_fn=_staged([
        (4, {"RFJ1": 0.07, "RFJ2": 0.07}),
        (2, {}),
        (4, {"RFJ1": -0.05, "RFJ2": -0.05}),
    ]),
    description="Briefly flex the ring finger's RFJ1/RFJ2 pair, pause, then "
                "partly return that pair.",
    tags=["micro", "small_object"],
))
register(Primitive(
    name="micro_little_finger_pulse",
    n_steps=10,
    action_fn=_staged([
        (4, {"LFJ1": 0.07, "LFJ2": 0.07}),
        (2, {}),
        (4, {"LFJ1": -0.05, "LFJ2": -0.05}),
    ]),
    description="Briefly flex the little finger's LFJ1/LFJ2 pair, pause, "
                "then partly return that pair.",
    tags=["micro", "small_object"],
))
register(Primitive(
    name="micro_thumb_index_pinch",
    n_steps=12,
    action_fn=_staged([
        (4, {"THJ0": 0.06, "THJ3": 0.05,
             "FFJ1": 0.05, "FFJ2": 0.05}),
        (4, {}),
        (4, {"THJ0": -0.04, "THJ3": -0.04,
             "FFJ1": -0.04, "FFJ2": -0.04}),
    ]),
    description="Move the thumb and forefinger toward one another by small "
                "joint increments, pause, then partly reopen both digits.",
    tags=["micro", "small_object"],
))
register(Primitive(
    name="micro_thumb_middle_pinch",
    n_steps=12,
    action_fn=_staged([
        (4, {"THJ0": 0.06, "THJ3": 0.05,
             "MFJ1": 0.05, "MFJ2": 0.05}),
        (4, {}),
        (4, {"THJ0": -0.04, "THJ3": -0.04,
             "MFJ1": -0.04, "MFJ2": -0.04}),
    ]),
    description="Move the thumb and middle finger toward one another by small "
                "joint increments, pause, then partly reopen both digits.",
    tags=["micro", "small_object"],
))
register(Primitive(
    name="micro_thumb_sweep",
    n_steps=10,
    action_fn=_staged([
        (4, {"THJ0": 0.06, "THJ3": 0.04}),
        (2, {}),
        (4, {"THJ0": -0.05, "THJ3": -0.03}),
    ]),
    description="Move THJ0 and THJ3 together through a short thumb sweep, "
                "pause, then mostly return them.",
    tags=["micro", "small_object"],
))
register(Primitive(
    name="micro_finger_walk",
    n_steps=16,
    action_fn=_staged([
        (4, {"FFJ1": 0.05, "FFJ2": 0.05}),
        (4, {"MFJ1": 0.05, "MFJ2": 0.05}),
        (4, {"RFJ1": 0.05, "RFJ2": 0.05}),
        (4, {"FFJ1": -0.03, "FFJ2": -0.03,
             "MFJ1": -0.03, "MFJ2": -0.03,
             "RFJ1": -0.03, "RFJ2": -0.03}),
    ]),
    description="Apply short low-amplitude flexion increments to the "
                "forefinger, middle finger, and ring finger in sequence, "
                "then partly return all three finger pairs together.",
    tags=["micro", "small_object"],
))

# Contact-then-sweep pair for changing the ring-finger contact patch without
# coupling in the middle finger.  RFJ3 directions are deliberately exposed as
# two alternatives because actuator direction does not guarantee object yaw.
register(Primitive(
    name="ring_lateral_sweep_a",
    n_steps=20,
    action_fn=_staged([
        (4, {"RFJ1": 0.08, "RFJ2": 0.08, "THJ4": 0.03}),
        (5, {"RFJ3": 0.10, "RFJ1": 0.02, "RFJ2": 0.02,
             "THJ4": 0.02}),
        (3, {"RFJ1": -0.07, "RFJ2": -0.07, "THJ4": -0.02}),
        (5, {"RFJ3": -0.10}),
        (3, {"RFJ1": 0.04, "RFJ2": 0.04}),
    ]),
    description="Flex RFJ1/RFJ2 toward contact, move RFJ3 in direction A "
                "with light THJ4 support, release RFJ1/RFJ2, return RFJ3, "
                "then lightly re-flex RFJ1/RFJ2.",
    tags=["roll", "contact_shift", "small_object"],
))
register(Primitive(
    name="ring_lateral_sweep_b",
    n_steps=20,
    action_fn=_staged([
        (4, {"RFJ1": 0.08, "RFJ2": 0.08, "THJ4": 0.03}),
        (5, {"RFJ3": -0.10, "RFJ1": 0.02, "RFJ2": 0.02,
             "THJ4": 0.02}),
        (3, {"RFJ1": -0.07, "RFJ2": -0.07, "THJ4": -0.02}),
        (5, {"RFJ3": 0.10}),
        (3, {"RFJ1": 0.04, "RFJ2": 0.04}),
    ]),
    description="Flex RFJ1/RFJ2 toward contact, move RFJ3 in direction B "
                "with light THJ4 support, release RFJ1/RFJ2, return RFJ3, "
                "then lightly re-flex RFJ1/RFJ2.",
    tags=["roll", "contact_shift", "small_object"],
))

# Higher-authority variants of the empirically useful RFJ3-B contact motion.
# They retain the release-before-return ordering so the RFJ3 return is less
# likely to undo the tangential contact displacement.
register(Primitive(
    name="ring_lateral_sweep_b_strong",
    n_steps=24,
    action_fn=_staged([
        (5, {"RFJ1": 0.10, "RFJ2": 0.10, "THJ4": 0.04}),
        (7, {"RFJ3": -0.14, "RFJ1": 0.025, "RFJ2": 0.025,
             "THJ4": 0.03}),
        (4, {"RFJ1": -0.08, "RFJ2": -0.08, "THJ4": -0.03}),
        (6, {"RFJ3": 0.14}),
        (2, {"RFJ1": 0.05, "RFJ2": 0.05}),
    ]),
    description="Flex RFJ1/RFJ2 toward contact, move RFJ3 farther in "
                "direction B with THJ4 support, release RFJ1/RFJ2, return "
                "RFJ3, then lightly re-flex RFJ1/RFJ2.",
    tags=["roll", "contact_shift", "small_object"],
))
register(Primitive(
    name="ring_lateral_ratchet_b",
    n_steps=24,
    action_fn=_staged([
        (3, {"RFJ1": 0.08, "RFJ2": 0.08, "THJ4": 0.03}),
        (3, {"RFJ3": -0.12, "RFJ1": 0.025, "RFJ2": 0.025}),
        (3, {"RFJ1": -0.07, "RFJ2": -0.07, "THJ4": -0.02}),
        (3, {"RFJ3": 0.12}),
        (3, {"RFJ1": 0.08, "RFJ2": 0.08, "THJ4": 0.03}),
        (3, {"RFJ3": -0.12, "RFJ1": 0.025, "RFJ2": 0.025}),
        (3, {"RFJ1": -0.07, "RFJ2": -0.07, "THJ4": -0.02}),
        (3, {"RFJ3": 0.12}),
    ]),
    description="Perform two low-amplitude RFJ1/RFJ2 contact cycles; in each "
                "cycle move RFJ3 in direction B, release RFJ1/RFJ2, and "
                "return RFJ3 before making contact again, with light THJ4 "
                "support during contact formation.",
    tags=["roll", "contact_shift", "small_object", "ratchet"],
))
