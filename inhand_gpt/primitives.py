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


def menu(object_shape: str = "block") -> List[Dict[str, str]]:
    """Candidate menu handed to the GPT policy."""
    if object_shape == "block":
        allowed = globals().get("BLOCK_PRIMITIVE_NAMES", set(PRIMITIVES))
    elif object_shape == "sphere":
        allowed = globals().get("SPHERE_PRIMITIVE_NAMES", set())
    else:
        raise ValueError(f"unknown object shape: {object_shape}")
    return [
        {"name": p.name, "description": p.description, "n_steps": p.n_steps}
        for p in PRIMITIVES.values() if p.name in allowed
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

# Low-amplitude paired probes for correcting non-Z orientation error.  The
# names identify commanded joint groups, not guaranteed world-axis effects;
# contact geometry determines the measured X/Y response at runtime.
register(Primitive(
    name="tilt_wrist_pitch_a",
    n_steps=16,
    action_fn=_staged([
        (4, {"FFJ1": 0.04, "FFJ2": 0.04, "MFJ1": 0.04,
             "MFJ2": 0.04, "RFJ1": 0.04, "RFJ2": 0.04}),
        (4, {"WRJ0": 0.10, "THJ4": 0.035}),
        (4, {"WRJ0": -0.10, "THJ4": -0.025}),
        (4, {"FFJ1": -0.03, "FFJ2": -0.03, "MFJ1": -0.03,
             "MFJ2": -0.03, "RFJ1": -0.03, "RFJ2": -0.03}),
    ]),
    description="Lightly flex three finger pairs, move WRJ0 in direction A "
                "with small THJ4 support, return WRJ0, then partly release "
                "the three finger pairs.",
    tags=["orientation", "tilt_probe", "small_object"],
))
register(Primitive(
    name="tilt_wrist_pitch_b",
    n_steps=16,
    action_fn=_staged([
        (4, {"FFJ1": 0.04, "FFJ2": 0.04, "MFJ1": 0.04,
             "MFJ2": 0.04, "RFJ1": 0.04, "RFJ2": 0.04}),
        (4, {"WRJ0": -0.10, "THJ4": -0.035}),
        (4, {"WRJ0": 0.10, "THJ4": 0.025}),
        (4, {"FFJ1": -0.03, "FFJ2": -0.03, "MFJ1": -0.03,
             "MFJ2": -0.03, "RFJ1": -0.03, "RFJ2": -0.03}),
    ]),
    description="Lightly flex three finger pairs, move WRJ0 in direction B "
                "with reverse THJ4 support, return WRJ0, then partly release "
                "the three finger pairs.",
    tags=["orientation", "tilt_probe", "small_object"],
))
register(Primitive(
    name="tilt_wrist_side_a",
    n_steps=16,
    action_fn=_staged([
        (4, {"FFJ1": 0.05, "FFJ2": 0.05,
             "LFJ1": -0.03, "LFJ2": -0.03}),
        (4, {"WRJ1": 0.10, "MFJ1": 0.04, "MFJ2": 0.04}),
        (4, {"FFJ1": -0.04, "FFJ2": -0.04,
             "LFJ1": 0.05, "LFJ2": 0.05}),
        (4, {"WRJ1": -0.10}),
    ]),
    description="Shift contact toward the forefinger side, move WRJ1 in "
                "direction A with light middle-finger support, shift contact "
                "toward the little-finger side, then return WRJ1.",
    tags=["orientation", "tilt_probe", "small_object"],
))
register(Primitive(
    name="tilt_wrist_side_b",
    n_steps=16,
    action_fn=_staged([
        (4, {"LFJ1": 0.05, "LFJ2": 0.05,
             "FFJ1": -0.03, "FFJ2": -0.03}),
        (4, {"WRJ1": -0.10, "MFJ1": 0.04, "MFJ2": 0.04}),
        (4, {"LFJ1": -0.04, "LFJ2": -0.04,
             "FFJ1": 0.05, "FFJ2": 0.05}),
        (4, {"WRJ1": 0.10}),
    ]),
    description="Shift contact toward the little-finger side, move WRJ1 in "
                "direction B with light middle-finger support, shift contact "
                "toward the forefinger side, then return WRJ1.",
    tags=["orientation", "tilt_probe", "small_object"],
))
register(Primitive(
    name="tilt_index_little_transfer_a",
    n_steps=14,
    action_fn=_staged([
        (5, {"FFJ1": 0.07, "FFJ2": 0.07,
             "LFJ1": -0.05, "LFJ2": -0.05}),
        (4, {"FFJ3": 0.05, "LFJ4": -0.04}),
        (5, {"FFJ1": -0.05, "FFJ2": -0.05,
             "LFJ1": 0.04, "LFJ2": 0.04,
             "FFJ3": -0.04, "LFJ4": 0.03}),
    ]),
    description="Flex the forefinger pair while extending the little-finger "
                "pair, move FFJ3 and LFJ4 in opposite directions, then "
                "partly reverse all commanded joints.",
    tags=["orientation", "tilt_probe", "small_object"],
))
register(Primitive(
    name="tilt_index_little_transfer_b",
    n_steps=14,
    action_fn=_staged([
        (5, {"LFJ1": 0.07, "LFJ2": 0.07,
             "FFJ1": -0.05, "FFJ2": -0.05}),
        (4, {"LFJ4": 0.05, "FFJ3": -0.04}),
        (5, {"LFJ1": -0.05, "LFJ2": -0.05,
             "FFJ1": 0.04, "FFJ2": 0.04,
             "LFJ4": -0.04, "FFJ3": 0.03}),
    ]),
    description="Flex the little-finger pair while extending the forefinger "
                "pair, move LFJ4 and FFJ3 in opposite directions, then "
                "partly reverse all commanded joints.",
    tags=["orientation", "tilt_probe", "small_object"],
))
register(Primitive(
    name="tilt_thumb_fingers_transfer_a",
    n_steps=14,
    action_fn=_staged([
        (5, {"THJ0": 0.06, "THJ3": 0.05,
             "MFJ1": -0.04, "MFJ2": -0.04,
             "RFJ1": -0.04, "RFJ2": -0.04}),
        (4, {"THJ4": 0.05, "MFJ3": -0.04, "RFJ3": -0.04}),
        (5, {"THJ0": -0.05, "THJ3": -0.04, "THJ4": -0.04,
             "MFJ1": 0.03, "MFJ2": 0.03, "MFJ3": 0.03,
             "RFJ1": 0.03, "RFJ2": 0.03, "RFJ3": 0.03}),
    ]),
    description="Move the thumb inward while partly extending the middle "
                "and ring fingers, add opposed THJ4 and MFJ3/RFJ3 motion, "
                "then partly reverse the commanded joints.",
    tags=["orientation", "tilt_probe", "small_object"],
))
register(Primitive(
    name="tilt_thumb_fingers_transfer_b",
    n_steps=14,
    action_fn=_staged([
        (5, {"MFJ1": 0.06, "MFJ2": 0.06,
             "RFJ1": 0.06, "RFJ2": 0.06,
             "THJ0": -0.04, "THJ3": -0.04}),
        (4, {"MFJ3": 0.04, "RFJ3": 0.04, "THJ4": -0.05}),
        (5, {"MFJ1": -0.05, "MFJ2": -0.05, "MFJ3": -0.03,
             "RFJ1": -0.05, "RFJ2": -0.05, "RFJ3": -0.03,
             "THJ0": 0.03, "THJ3": 0.03, "THJ4": 0.04}),
    ]),
    description="Flex the middle and ring fingers while partly moving the "
                "thumb outward, add opposed MFJ3/RFJ3 and THJ4 motion, then "
                "partly reverse the commanded joints.",
    tags=["orientation", "tilt_probe", "small_object"],
))

register(Primitive(
    name="tilt_wrj0_reseat_a",
    n_steps=34,
    action_fn=_staged([
        (5, {"FFJ1": -0.10, "FFJ2": -0.10,
             "MFJ1": -0.10, "MFJ2": -0.10}),
        (8, {"WRJ0": 0.20, "RFJ1": 0.05, "RFJ2": 0.05}),
        (7, {}),
        (7, {"FFJ1": 0.12, "FFJ2": 0.12,
             "MFJ1": 0.12, "MFJ2": 0.12,
             "RFJ1": 0.05, "RFJ2": 0.05}),
        (7, {"WRJ0": -0.20, "FFJ1": 0.02, "FFJ2": 0.02,
             "MFJ1": 0.02, "MFJ2": 0.02}),
    ]),
    description="Extend the forefinger and middle-finger pairs, move WRJ0 "
                "farther in direction A with ring-finger support, hold, "
                "re-flex the fingers, then return WRJ0.",
    tags=["orientation", "tilt_adjust", "wrist"],
))
register(Primitive(
    name="tilt_wrj0_reseat_b",
    n_steps=34,
    action_fn=_staged([
        (5, {"FFJ1": -0.10, "FFJ2": -0.10,
             "MFJ1": -0.10, "MFJ2": -0.10}),
        (8, {"WRJ0": -0.20, "RFJ1": 0.05, "RFJ2": 0.05}),
        (7, {}),
        (7, {"FFJ1": 0.12, "FFJ2": 0.12,
             "MFJ1": 0.12, "MFJ2": 0.12,
             "RFJ1": 0.05, "RFJ2": 0.05}),
        (7, {"WRJ0": 0.20, "FFJ1": 0.02, "FFJ2": 0.02,
             "MFJ1": 0.02, "MFJ2": 0.02}),
    ]),
    description="Extend the forefinger and middle-finger pairs, move WRJ0 "
                "farther in direction B with ring-finger support, hold, "
                "re-flex the fingers, then return WRJ0.",
    tags=["orientation", "tilt_adjust", "wrist"],
))
register(Primitive(
    name="tilt_wrj1_reseat_a",
    n_steps=26,
    action_fn=_staged([
        (5, {"FFJ1": -0.07, "FFJ2": -0.07,
             "LFJ1": 0.06, "LFJ2": 0.06}),
        (7, {"WRJ1": 0.16, "MFJ1": 0.04, "MFJ2": 0.04}),
        (5, {}),
        (5, {"FFJ1": 0.09, "FFJ2": 0.09,
             "LFJ1": -0.05, "LFJ2": -0.05}),
        (4, {"WRJ1": -0.16}),
    ]),
    description="Shift finger contact asymmetrically, move WRJ1 farther in "
                "direction A with middle-finger support, hold, exchange the "
                "side contacts, then return WRJ1.",
    tags=["orientation", "tilt_adjust", "wrist"],
))
register(Primitive(
    name="tilt_wrj1_reseat_b",
    n_steps=26,
    action_fn=_staged([
        (5, {"LFJ1": -0.07, "LFJ2": -0.07,
             "FFJ1": 0.06, "FFJ2": 0.06}),
        (7, {"WRJ1": -0.16, "MFJ1": 0.04, "MFJ2": 0.04}),
        (5, {}),
        (5, {"LFJ1": 0.09, "LFJ2": 0.09,
             "FFJ1": -0.05, "FFJ2": -0.05}),
        (4, {"WRJ1": 0.16}),
    ]),
    description="Shift finger contact asymmetrically, move WRJ1 farther in "
                "direction B with middle-finger support, hold, exchange the "
                "side contacts, then return WRJ1.",
    tags=["orientation", "tilt_adjust", "wrist"],
))

# Short paired orientation motions for late-stage, direction-dependent
# correction.  Descriptions state only commanded kinematics; the policy uses
# measured XYZ gains to learn which member helps in the current contact state.
register(Primitive(
    name="axis_fine_wrj0_a", n_steps=12,
    action_fn=_staged([
        (3, {"FFJ1": 0.035, "MFJ1": 0.035, "RFJ1": 0.035}),
        (4, {"WRJ0": 0.055, "THJ4": 0.025}),
        (2, {"FFJ1": 0.02, "MFJ1": 0.02, "RFJ1": 0.02}),
        (3, {"WRJ0": -0.055, "THJ4": -0.02,
             "FFJ1": -0.025, "MFJ1": -0.025, "RFJ1": -0.025}),
    ]),
    description="Lightly brace three fingers, move WRJ0 in direction A with "
                "thumb support, then partly return the commanded joints.",
    tags=["orientation", "axis_fine", "small_object"],
))
register(Primitive(
    name="axis_fine_wrj0_b", n_steps=12,
    action_fn=_staged([
        (3, {"FFJ1": 0.035, "MFJ1": 0.035, "RFJ1": 0.035}),
        (4, {"WRJ0": -0.055, "THJ4": -0.025}),
        (2, {"FFJ1": 0.02, "MFJ1": 0.02, "RFJ1": 0.02}),
        (3, {"WRJ0": 0.055, "THJ4": 0.02,
             "FFJ1": -0.025, "MFJ1": -0.025, "RFJ1": -0.025}),
    ]),
    description="Lightly brace three fingers, move WRJ0 in direction B with "
                "reverse thumb support, then partly return the commanded joints.",
    tags=["orientation", "axis_fine", "small_object"],
))
register(Primitive(
    name="axis_fine_wrj1_a", n_steps=12,
    action_fn=_staged([
        (3, {"FFJ1": 0.04, "FFJ2": 0.03,
             "LFJ1": -0.025, "LFJ2": -0.02}),
        (4, {"WRJ1": 0.055, "MFJ1": 0.025}),
        (2, {"LFJ1": 0.035, "LFJ2": 0.025}),
        (3, {"WRJ1": -0.055, "FFJ1": -0.03, "FFJ2": -0.02,
             "LFJ1": -0.025, "LFJ2": -0.02}),
    ]),
    description="Shift light contact toward the forefinger side, move WRJ1 "
                "in direction A, exchange side support, then return WRJ1.",
    tags=["orientation", "axis_fine", "small_object"],
))
register(Primitive(
    name="axis_fine_wrj1_b", n_steps=12,
    action_fn=_staged([
        (3, {"LFJ1": 0.04, "LFJ2": 0.03,
             "FFJ1": -0.025, "FFJ2": -0.02}),
        (4, {"WRJ1": -0.055, "MFJ1": 0.025}),
        (2, {"FFJ1": 0.035, "FFJ2": 0.025}),
        (3, {"WRJ1": 0.055, "LFJ1": -0.03, "LFJ2": -0.02,
             "FFJ1": -0.025, "FFJ2": -0.02}),
    ]),
    description="Shift light contact toward the little-finger side, move WRJ1 "
                "in direction B, exchange side support, then return WRJ1.",
    tags=["orientation", "axis_fine", "small_object"],
))
register(Primitive(
    name="axis_fine_diagonal_a", n_steps=12,
    action_fn=_staged([
        (4, {"THJ0": 0.04, "THJ3": 0.035,
             "FFJ1": 0.035, "FFJ2": 0.03,
             "RFJ1": -0.025, "LFJ1": -0.03}),
        (4, {"THJ4": 0.035, "FFJ3": 0.03,
             "RFJ3": -0.025, "LFJ3": -0.03}),
        (4, {"THJ0": -0.035, "THJ3": -0.03, "THJ4": -0.03,
             "FFJ1": -0.03, "FFJ2": -0.025, "FFJ3": -0.025,
             "RFJ1": 0.02, "RFJ3": 0.02,
             "LFJ1": 0.025, "LFJ3": 0.025}),
    ]),
    description="Move thumb and forefinger inward while lightly releasing the "
                "ring/little side, add opposed lateral motion, then return.",
    tags=["orientation", "axis_fine", "small_object"],
))
register(Primitive(
    name="axis_fine_diagonal_b", n_steps=12,
    action_fn=_staged([
        (4, {"RFJ1": 0.035, "LFJ1": 0.04,
             "THJ0": -0.03, "THJ3": -0.025,
             "FFJ1": -0.025, "FFJ2": -0.02}),
        (4, {"RFJ3": 0.03, "LFJ3": 0.035,
             "THJ4": -0.035, "FFJ3": -0.025}),
        (4, {"RFJ1": -0.03, "RFJ3": -0.025,
             "LFJ1": -0.035, "LFJ3": -0.03,
             "THJ0": 0.025, "THJ3": 0.02, "THJ4": 0.03,
             "FFJ1": 0.02, "FFJ2": 0.02, "FFJ3": 0.02}),
    ]),
    description="Move the ring/little side inward while partly releasing the "
                "thumb/forefinger side, add opposed lateral motion, then return.",
    tags=["orientation", "axis_fine", "small_object"],
))

# Contact-changing compound motions derived from the four-panel dead-zone
# scan.  They combine a release/reseat phase with a different actuator group,
# avoiding repeated commands against already saturated finger joints.
register(Primitive(
    name="escape_little_thumb_a", n_steps=32,
    action_fn=_staged([
        (5, {"LFJ1": -0.10, "LFJ2": -0.10, "THJ0": -0.04}),
        (6, {"LFJ4": 0.08, "RFJ1": 0.05, "RFJ2": 0.05}),
        (6, {"LFJ1": 0.13, "LFJ2": 0.13, "THJ0": 0.05}),
        (7, {"THJ0": 0.10, "THJ3": 0.09,
             "FFJ1": 0.04, "MFJ1": 0.04}),
        (8, {"THJ0": -0.09, "THJ3": -0.08, "LFJ4": -0.07,
             "LFJ1": -0.04, "LFJ2": -0.04}),
    ]),
    description="Release and laterally move the little finger, re-form its "
                "contact, sweep the thumb with light fore/middle support, "
                "then partly return the commanded joints.",
    tags=["orientation", "contact_escape", "small_object"],
))
register(Primitive(
    name="escape_little_thumb_b", n_steps=32,
    action_fn=_staged([
        (5, {"LFJ1": -0.10, "LFJ2": -0.10, "THJ0": 0.04}),
        (6, {"LFJ4": -0.08, "RFJ1": 0.05, "RFJ2": 0.05}),
        (6, {"LFJ1": 0.13, "LFJ2": 0.13, "THJ0": -0.05}),
        (7, {"THJ0": -0.10, "THJ3": -0.09,
             "FFJ1": 0.04, "MFJ1": 0.04}),
        (8, {"THJ0": 0.09, "THJ3": 0.08, "LFJ4": 0.07,
             "LFJ1": -0.04, "LFJ2": -0.04}),
    ]),
    description="Release and move the little finger in the opposite lateral "
                "direction, re-form contact, reverse the thumb sweep, then "
                "partly return the commanded joints.",
    tags=["orientation", "contact_escape", "small_object"],
))
register(Primitive(
    name="escape_wrist_finger_a", n_steps=34,
    action_fn=_staged([
        (6, {"FFJ1": -0.10, "FFJ2": -0.10,
             "MFJ1": -0.07, "MFJ2": -0.07}),
        (7, {"WRJ1": 0.14, "RFJ1": 0.06, "LFJ1": 0.05}),
        (6, {"FFJ1": 0.14, "FFJ2": 0.12,
             "MFJ1": 0.09, "MFJ2": 0.08}),
        (7, {"FFJ3": 0.08, "WRJ1": -0.08, "THJ4": 0.05}),
        (8, {"WRJ1": -0.06, "FFJ3": -0.07, "THJ4": -0.04,
             "FFJ1": -0.04, "FFJ2": -0.04}),
    ]),
    description="Release the fore/middle pairs, move WRJ1 with ring/little "
                "support, re-flex the released fingers, add a short forefinger "
                "lateral sweep, then return the wrist and forefinger.",
    tags=["orientation", "contact_escape", "wrist"],
))
register(Primitive(
    name="escape_wrist_finger_b", n_steps=34,
    action_fn=_staged([
        (6, {"RFJ1": -0.09, "RFJ2": -0.09,
             "LFJ1": -0.08, "LFJ2": -0.08}),
        (7, {"WRJ1": -0.14, "FFJ1": 0.06, "MFJ1": 0.05}),
        (6, {"RFJ1": 0.13, "RFJ2": 0.11,
             "LFJ1": 0.12, "LFJ2": 0.10}),
        (7, {"RFJ3": -0.08, "WRJ1": 0.08, "THJ4": -0.05}),
        (8, {"WRJ1": 0.06, "RFJ3": 0.07, "THJ4": 0.04,
             "RFJ1": -0.04, "RFJ2": -0.04}),
    ]),
    description="Release the ring/little pairs, move WRJ1 oppositely with "
                "fore/middle support, re-flex the released fingers, add a "
                "short ring-finger lateral sweep, then return.",
    tags=["orientation", "contact_escape", "wrist"],
))
register(Primitive(
    name="escape_wrj0_reseat_a", n_steps=38,
    action_fn=_staged([
        (6, {"FFJ1": -0.12, "FFJ2": -0.12,
             "MFJ1": -0.10, "MFJ2": -0.10,
             "RFJ1": 0.04, "RFJ2": 0.04}),
        (8, {"WRJ0": 0.22, "LFJ1": 0.07, "LFJ2": 0.07}),
        (6, {"RFJ1": -0.06, "RFJ2": -0.06,
             "LFJ1": 0.08, "LFJ2": 0.08}),
        (8, {"FFJ1": 0.15, "FFJ2": 0.14,
             "MFJ1": 0.13, "MFJ2": 0.12,
             "RFJ1": 0.09, "RFJ2": 0.09}),
        (10, {"WRJ0": -0.20, "FFJ1": 0.02, "MFJ1": 0.02,
              "LFJ1": -0.05, "LFJ2": -0.05}),
    ]),
    description="Release the fore/middle pairs, move WRJ0 farther in direction "
                "A with little-finger support, exchange ring/little contact, "
                "re-flex three fingers, then return WRJ0.",
    tags=["orientation", "contact_escape", "wrist"],
))
register(Primitive(
    name="escape_wrj0_reseat_b", n_steps=38,
    action_fn=_staged([
        (6, {"FFJ1": -0.12, "FFJ2": -0.12,
             "MFJ1": -0.10, "MFJ2": -0.10,
             "RFJ1": 0.04, "RFJ2": 0.04}),
        (8, {"WRJ0": -0.22, "LFJ1": 0.07, "LFJ2": 0.07}),
        (6, {"RFJ1": -0.06, "RFJ2": -0.06,
             "LFJ1": 0.08, "LFJ2": 0.08}),
        (8, {"FFJ1": 0.15, "FFJ2": 0.14,
             "MFJ1": 0.13, "MFJ2": 0.12,
             "RFJ1": 0.09, "RFJ2": 0.09}),
        (10, {"WRJ0": 0.20, "FFJ1": 0.02, "MFJ1": 0.02,
              "LFJ1": -0.05, "LFJ2": -0.05}),
    ]),
    description="Release the fore/middle pairs, move WRJ0 farther in direction "
                "B with little-finger support, exchange ring/little contact, "
                "re-flex three fingers, then return WRJ0.",
    tags=["orientation", "contact_escape", "wrist"],
))


def _chain_registered(names: List[str]) -> tuple[int, Callable[[int], np.ndarray]]:
    """Concatenate existing primitives without changing their waveforms."""
    parts = [get(name) for name in names]
    total = sum(part.n_steps for part in parts)

    def fn(t: int) -> np.ndarray:
        offset = 0
        for part in parts:
            if t < offset + part.n_steps:
                return part.action(t - offset)
            offset += part.n_steps
        return np.zeros(N_ACTIONS, dtype=np.float64)

    return total, fn


_n_steps, _action_fn = _chain_registered(["regrasp", "roll_wr"])
register(Primitive(
    name="regrasp_then_roll_wr", n_steps=_n_steps, action_fn=_action_fn,
    description="Complete a three-finger extend-and-regrasp cycle, then run "
                "the WRJ1 oscillation with fore/middle support.",
    tags=["orientation", "contact_escape", "compound"],
))
_n_steps, _action_fn = _chain_registered(["little_finger_regrasp", "roll_th"])
register(Primitive(
    name="little_regrasp_then_roll_th", n_steps=_n_steps, action_fn=_action_fn,
    description="Complete an independent little-finger regrasp, then run the "
                "thumb oscillation with fore/middle support.",
    tags=["orientation", "contact_escape", "compound"],
))
_n_steps, _action_fn = _chain_registered(["regrasp", "roll_cw"])
register(Primitive(
    name="regrasp_then_roll_cw", n_steps=_n_steps, action_fn=_action_fn,
    description="Complete a three-finger extend-and-regrasp cycle, then run "
                "the forefinger-biased periodic motion and its return phase.",
    tags=["orientation", "contact_escape", "compound"],
))
_n_steps, _action_fn = _chain_registered(["regrasp", "roll_ccw"])
register(Primitive(
    name="regrasp_then_roll_ccw", n_steps=_n_steps, action_fn=_action_fn,
    description="Complete a three-finger extend-and-regrasp cycle, then run "
                "the ring-finger/thumb periodic motion and its return phase.",
    tags=["orientation", "contact_escape", "compound"],
))


# Preserve the original 62 block primitives exactly. Sphere-specific motions
# are registered separately and are only exposed by menu("sphere").
BLOCK_PRIMITIVE_NAMES = set(PRIMITIVES)
SPHERE_PRIMITIVE_NAMES: set[str] = set()


def _register_sphere(name: str, drive: Dict[str, float], description: str,
                     n_steps: int = 18, return_ratio: float = 0.55) -> None:
    """Sustained tangential motion with support and a partial joint return."""
    reverse = {joint: -return_ratio * value for joint, value in drive.items()}
    register(Primitive(
        name=name, n_steps=n_steps,
        action_fn=_staged([(8, drive), (4, {}), (6, reverse)]),
        description=description,
        tags=["sphere", "tangential_contact", "closed_loop"],
    ))
    SPHERE_PRIMITIVE_NAMES.add(name)


_register_sphere(
    "sphere_hold", {},
    "Keep all finger and wrist commands at zero while preserving the current grasp.")

# Opposed tangential motions. Directions A/B are actuator-coordinate
# directions; the measured XYZ feedback remains authoritative.
_SPHERE_MOTIONS = {
    "sphere_x_a": ({"WRJ0": 0.13, "FFJ1": 0.08, "MFJ1": 0.06,
                    "RFJ1": -0.05, "LFJ1": -0.05, "THJ0": 0.05},
                   "Move WRJ0 in direction A while fore/middle and ring/little contacts move oppositely."),
    "sphere_x_b": ({"WRJ0": -0.13, "FFJ1": -0.06, "MFJ1": -0.05,
                    "RFJ1": 0.08, "LFJ1": 0.08, "THJ0": -0.05},
                   "Move WRJ0 in direction B while the two finger sides exchange flexion."),
    "sphere_y_a": ({"WRJ1": 0.11, "FFJ3": 0.10, "LFJ3": -0.10,
                    "MFJ1": 0.05, "RFJ1": 0.05, "THJ4": 0.05},
                   "Move WRJ1 in direction A with opposed forefinger/little-finger lateral motion."),
    "sphere_y_b": ({"WRJ1": -0.11, "FFJ3": -0.10, "LFJ3": 0.10,
                    "MFJ1": 0.05, "RFJ1": 0.05, "THJ4": -0.05},
                   "Move WRJ1 in direction B with reversed forefinger/little-finger lateral motion."),
    "sphere_z_a": ({"FFJ3": 0.12, "MFJ3": 0.07, "RFJ3": -0.09,
                    "LFJ3": -0.11, "THJ4": 0.10, "THJ0": 0.05},
                   "Sweep the radial and ulnar finger contacts in opposed lateral directions with thumb support."),
    "sphere_z_b": ({"FFJ3": -0.12, "MFJ3": -0.07, "RFJ3": 0.09,
                    "LFJ3": 0.11, "THJ4": -0.10, "THJ0": 0.05},
                   "Reverse the opposed lateral finger sweep while maintaining thumb support."),
    "sphere_thumb_fore_a": ({"THJ4": 0.12, "THJ0": 0.10,
                             "FFJ3": -0.11, "FFJ1": 0.08},
                            "Sweep the thumb and forefinger tangentially in opposite directions."),
    "sphere_thumb_fore_b": ({"THJ4": -0.12, "THJ0": 0.10,
                             "FFJ3": 0.11, "FFJ1": 0.08},
                            "Reverse the opposed thumb and forefinger tangential sweep."),
    "sphere_thumb_middle_a": ({"THJ4": 0.11, "THJ3": 0.08,
                               "MFJ3": -0.10, "MFJ1": 0.08},
                              "Sweep the thumb and middle-finger contacts in opposite directions."),
    "sphere_thumb_middle_b": ({"THJ4": -0.11, "THJ3": 0.08,
                               "MFJ3": 0.10, "MFJ1": 0.08},
                              "Reverse the opposed thumb and middle-finger contact sweep."),
    "sphere_ring_little_a": ({"RFJ3": 0.11, "LFJ3": -0.12,
                              "RFJ1": 0.08, "LFJ1": 0.08, "THJ0": 0.04},
                             "Move ring and little-finger lateral contacts oppositely with light thumb support."),
    "sphere_ring_little_b": ({"RFJ3": -0.11, "LFJ3": 0.12,
                              "RFJ1": 0.08, "LFJ1": 0.08, "THJ0": 0.04},
                             "Reverse the opposed ring and little-finger lateral contact motion."),
}
for _name, (_drive, _description) in _SPHERE_MOTIONS.items():
    _register_sphere(_name, _drive, _description)

# Directional contact-transfer actions for the 80 mm sphere.  Unlike the
# sinusoidal roll_th gait, each action contains one transport stroke followed
# by a smaller reset stroke, so the policy can select either actuator direction
# from measured Y-axis feedback.
register(Primitive(
    name="sphere_y_thumb_arc_a", n_steps=24,
    action_fn=_staged([
        (5, {"FFJ1": 0.16, "FFJ2": 0.12, "MFJ1": 0.14,
             "MFJ2": 0.10, "RFJ1": 0.07, "LFJ1": 0.06}),
        (9, {"THJ0": 0.52, "THJ3": -0.32, "THJ4": 0.12,
             "FFJ1": 0.08, "MFJ1": 0.08}),
        (4, {"THJ0": 0.16, "THJ3": -0.10,
             "FFJ1": 0.10, "MFJ1": 0.10}),
        (6, {"THJ0": -0.30, "THJ3": 0.20, "THJ4": -0.07,
             "FFJ1": 0.06, "MFJ1": 0.06}),
    ]),
    description="Brace the fore/middle contacts, sweep THJ0 and THJ3 in "
                "opposed direction A with light THJ4 motion, then apply a "
                "smaller reverse thumb stroke.",
    tags=["sphere", "orientation", "y_axis_probe", "directional"],
))
register(Primitive(
    name="sphere_y_thumb_arc_b", n_steps=24,
    action_fn=_staged([
        (5, {"FFJ1": 0.16, "FFJ2": 0.12, "MFJ1": 0.14,
             "MFJ2": 0.10, "RFJ1": 0.07, "LFJ1": 0.06}),
        (9, {"THJ0": -0.52, "THJ3": 0.32, "THJ4": -0.12,
             "FFJ1": 0.08, "MFJ1": 0.08}),
        (4, {"THJ0": -0.16, "THJ3": 0.10,
             "FFJ1": 0.10, "MFJ1": 0.10}),
        (6, {"THJ0": 0.30, "THJ3": -0.20, "THJ4": 0.07,
             "FFJ1": 0.06, "MFJ1": 0.06}),
    ]),
    description="Brace the fore/middle contacts, sweep THJ0 and THJ3 in "
                "opposed direction B with reverse THJ4 motion, then apply a "
                "smaller reverse thumb stroke.",
    tags=["sphere", "orientation", "y_axis_probe", "directional"],
))
register(Primitive(
    name="sphere_y_thumb_cycle_reverse", n_steps=30,
    action_fn=_pattern({
        "THJ0": (-0.80, 0.0, 0.0, 0),
        "THJ3": (-0.50, -1.57, 0.0, 0),
        "FFJ1": (0.0, 0.0, 0.20, _INF),
        "FFJ2": (0.0, 0.0, 0.20, _INF),
        "MFJ1": (0.0, 0.0, 0.20, _INF),
        "MFJ2": (0.0, 0.0, 0.20, _INF),
    }, period=12),
    description="Oscillate THJ0 and THJ3 along the reverse trajectory of the "
                "standard thumb cycle while forefinger and middle-finger "
                "pairs maintain a light flexion bias.",
    tags=["sphere", "orientation", "y_axis_probe", "directional"],
))
register(Primitive(
    name="sphere_y_contact_transfer_a", n_steps=24,
    action_fn=_staged([
        (5, {"RFJ1": 0.16, "RFJ2": 0.12, "LFJ1": 0.18,
             "LFJ2": 0.13, "FFJ1": -0.07, "MFJ1": -0.06,
             "THJ0": 0.08}),
        (8, {"FFJ3": 0.24, "MFJ3": 0.15, "RFJ3": -0.16,
             "LFJ4": -0.20, "WRJ1": 0.14, "THJ4": 0.08}),
        (5, {"FFJ1": 0.15, "MFJ1": 0.13, "RFJ1": -0.05,
             "LFJ1": -0.06, "THJ0": 0.10}),
        (6, {"FFJ3": -0.13, "MFJ3": -0.08, "RFJ3": 0.09,
             "LFJ4": 0.11, "WRJ1": -0.08}),
    ]),
    description="Shift support to the ring/little side, sweep the radial and "
                "ulnar lateral joints oppositely with WRJ1 direction A, "
                "transfer support, then partly reverse the sweep.",
    tags=["sphere", "orientation", "y_axis_probe", "contact_transfer"],
))
register(Primitive(
    name="sphere_y_contact_transfer_b", n_steps=24,
    action_fn=_staged([
        (5, {"FFJ1": 0.18, "FFJ2": 0.13, "MFJ1": 0.16,
             "MFJ2": 0.12, "RFJ1": -0.07, "LFJ1": -0.06,
             "THJ0": 0.08}),
        (8, {"FFJ3": -0.24, "MFJ3": -0.15, "RFJ3": 0.16,
             "LFJ4": 0.20, "WRJ1": -0.14, "THJ4": -0.08}),
        (5, {"RFJ1": 0.15, "RFJ2": 0.12, "LFJ1": 0.17,
             "LFJ2": 0.13, "FFJ1": -0.05, "MFJ1": -0.05,
             "THJ0": 0.10}),
        (6, {"FFJ3": 0.13, "MFJ3": 0.08, "RFJ3": -0.09,
             "LFJ4": -0.11, "WRJ1": 0.08}),
    ]),
    description="Shift support to the fore/middle side, reverse the opposed "
                "radial/ulnar lateral sweep with WRJ1 direction B, transfer "
                "support, then partly reverse the sweep.",
    tags=["sphere", "orientation", "y_axis_probe", "contact_transfer"],
))

# Lower-amplitude counterparts for corrections near the target.
for _axis in ("x", "y", "z"):
    for _direction in ("a", "b"):
        _source = _SPHERE_MOTIONS[f"sphere_{_axis}_{_direction}"][0]
        _register_sphere(
            f"sphere_micro_{_axis}_{_direction}",
            {joint: value * 0.42 for joint, value in _source.items()},
            f"Apply a low-amplitude version of the {_axis.upper()}-axis {_direction.upper()} contact motion.",
            n_steps=12, return_ratio=0.65)

_register_sphere(
    "sphere_recontact_radial",
    {"FFJ1": -0.10, "FFJ2": -0.10, "MFJ1": -0.08,
     "MFJ2": -0.08, "RFJ1": 0.07, "LFJ1": 0.07},
    "Release the fore/middle contacts while ring/little contacts support the sphere, then partly return.",
    n_steps=20)
_register_sphere(
    "sphere_recontact_ulnar",
    {"RFJ1": -0.10, "RFJ2": -0.10, "LFJ1": -0.10,
     "LFJ2": -0.10, "FFJ1": 0.07, "MFJ1": 0.07},
    "Release the ring/little contacts while fore/middle contacts support the sphere, then partly return.",
    n_steps=20)
_register_sphere(
    "sphere_recontact_thumb",
    {"THJ0": -0.10, "THJ3": -0.08, "THJ4": -0.06,
     "FFJ1": 0.06, "MFJ1": 0.06, "RFJ1": 0.05},
    "Release the thumb while three fingers support the sphere, then partly return.",
    n_steps=20)
_register_sphere(
    "sphere_recenter",
    {"WRJ0": 0.12, "FFJ1": 0.09, "MFJ1": 0.09,
     "RFJ1": 0.08, "LFJ1": 0.08, "THJ0": 0.08},
    "Move the wrist and all opposing fingertips inward together, then partly return.",
    n_steps=22, return_ratio=0.45)

# The existing wrist-lowering primitive remains useful for a sphere outside
# the finger region and keeps the policy's soft recovery advice available.
SPHERE_PRIMITIVE_NAMES.add("wrist_lower_reseat")

# The first sphere trial showed that the small sphere-specific commands were
# visually subtle and led the model to repeat two wrist-dominated actions.
# Keep those definitions available for later tuning, but expose the original
# nine well-established primitives for the next sphere comparison.
SPHERE_EXPERIMENTAL_PRIMITIVE_NAMES = set(SPHERE_PRIMITIVE_NAMES)

# Preserve the exact roll_lf waveform while extending its contact transport
# over two complete cycles.  This sequence was selected by a zero-command
# controlled scan at the difficult four-sphere panel-C terminal state.
_n_steps, _action_fn = _chain_registered(["roll_lf", "roll_lf"])
register(Primitive(
    name="sphere_double_little_finger_cycle",
    n_steps=_n_steps,
    action_fn=_action_fn,
    description="Run two consecutive complete little-finger LFJ4 oscillation "
                "cycles with their LFJ1/LFJ2 phase motion and ring-finger "
                "support unchanged.",
    tags=["sphere", "orientation", "contact_escape", "compound"],
))

SPHERE_PRIMITIVE_NAMES = {
    "hold", "roll_cw", "roll_ccw", "roll_lf", "roll_th", "roll_wr",
    "regrasp", "nudge_cw", "nudge_ccw",
    # Additional actions selected by zero-command-controlled scans on the
    # 80 mm sphere across four distinct initial orientations.
    "forefinger_regrasp", "middle_finger_regrasp",
    "ring_finger_regrasp", "little_finger_regrasp",
    "multi_finger_dither", "escape_wrist_finger_b",
    "escape_little_thumb_b", "recenter_fingertips",
    "tilt_wrj0_reseat_b",
    # Direction-paired Y-axis probes calibrated for the 80 mm sphere.
    "sphere_y_thumb_arc_a", "sphere_y_thumb_arc_b",
    "sphere_y_thumb_cycle_reverse",
    # Terminal-state scans on the four 80 mm sphere presets selected these
    # complementary orientation/contact motions.
    "tilt_wrj0_reseat_a", "wrist_raise_reseat",
    "recenter_forefinger_side", "sphere_double_little_finger_cycle",
}
