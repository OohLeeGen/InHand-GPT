"""Primitive tuning harness: measure net block yaw + drift for candidate waveforms.

Protocol (no rendering, fast):
  1. reset(seed=SEED), record initial block quat/pos
  2. apply candidate pattern (actuator -> sine amplitude A, phase, bias)
     for N_STEPS env steps
  3. every SEG steps and at the end, read block quat -> signed yaw delta about
     world Z relative to reset, plus horizontal/vertical drift
  4. selection rule: stable direction, max |yaw| per 20 steps, drift_xy < 1 cm

Usage:
  python scripts/tune_primitives.py [--seed 0] [--steps 40] [--scan SET]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import gymnasium_robotics  # noqa: F401

from inhand_gpt.primitives import IDX
from inhand_gpt.observe import quat_conjugate, quat_mul, wrap_angle

P = math.pi


def signed_yaw(q: np.ndarray, q0: np.ndarray) -> float:
    """Signed yaw of q relative to q0 about world Z (rad)."""
    q_rel = quat_mul(q / np.linalg.norm(q), quat_conjugate(q0 / np.linalg.norm(q0)))
    return wrap_angle(2.0 * math.atan2(q_rel[3], q_rel[0]))


def _object_qpos(env) -> np.ndarray:
    u = env.unwrapped
    adr = u.model.jnt_qposadr[u.model.joint("object:joint").id]
    return np.array(u.data.qpos[adr:adr + 7], dtype=np.float64)


def run_pattern(seed: int, pattern: dict, n_steps: int, period: int, seg: int = 20):
    """pattern: {act_name: (amp, phase, bias, bias_steps)}. Returns metrics dict."""
    env = gym.make("HandManipulateBlockRotateZ-v1", max_episode_steps=500)
    env.reset(seed=seed)
    q0 = _object_qpos(env)[3:7]
    p0 = _object_qpos(env)[0:3]

    yaw_at = {}
    drift_at = {}
    max_drift = 0.0
    for t in range(n_steps):
        a = np.zeros(20)
        for name, (amp, phase, bias, bias_steps) in pattern.items():
            a[IDX[name]] = (bias if t < bias_steps else 0.0) + \
                amp * math.sin(2 * math.pi * t / period + phase)
        env.step(np.clip(a, -1.0, 1.0).astype(np.float32))
        qpos = _object_qpos(env)
        d = np.linalg.norm(qpos[:2] - p0[:2])
        max_drift = max(max_drift, d)
        if (t + 1) % seg == 0 or t == n_steps - 1:
            yaw_at[t + 1] = signed_yaw(qpos[3:7], q0)
            drift_at[t + 1] = d

    qpos = _object_qpos(env)
    drift_xy = float(np.linalg.norm(qpos[:2] - p0[:2]))
    drift_z = float(qpos[2] - p0[2])
    env.close()
    segs = sorted(yaw_at)
    return {
        "yaw_deg_total": round(math.degrees(yaw_at[n_steps]), 2),
        "yaw_deg_per20": [round(math.degrees(yaw_at[s]), 2) for s in segs],
        "drift_cm_per20": [round(drift_at[s] * 100, 2) for s in segs],
        "drift_xy_cm": round(drift_xy * 100, 2),
        "drift_z_cm": round(drift_z * 100, 2),
        "max_drift_cm": round(max_drift * 100, 2),
    }


def _sq(fingers, bias=0.25, steps=4):
    """squeeze entries: flex J1+J2 of given fingers for `steps` steps then hold"""
    out = {}
    for f in fingers:
        out[f"{f}J1"] = (0.0, 0.0, bias, steps)
        out[f"{f}J2"] = (0.0, 0.0, bias, steps)
    return out


# pattern values: (amp_scale, phase_rad, bias, bias_steps)
def candidate_patterns(scan: str):
    pats = []

    if scan in ("control", "all"):
        pats.append(("hold_zero", {}, 10))

    if scan in ("pos", "all"):
        sq3 = lambda b=0.25, s=4: _sq(["FF", "MF", "RF"], b, s)
        # squeeze then thumb wave (THJ0 base + THJ3 knuckle quadrature)
        pats.append(("thumbroll_a", {**sq3(),
                                     "THJ0": (1.0, 0.0, 0.0, 0), "THJ3": (0.7, -P / 2, 0.0, 0)}, 12))
        pats.append(("thumbroll_a_rev", {**sq3(),
                                         "THJ0": (1.0, P, 0.0, 0), "THJ3": (0.7, P / 2, 0.0, 0)}, 12))
        # squeeze then thumb distal wave (THJ2+THJ1)
        pats.append(("thumbroll_b", {**sq3(),
                                     "THJ2": (1.0, 0.0, 0.0, 0), "THJ1": (1.0, 0.0, 0.0, 0)}, 12))
        pats.append(("thumbroll_b_rev", {**sq3(),
                                         "THJ2": (1.0, P, 0.0, 0), "THJ1": (1.0, P, 0.0, 0)}, 12))
        # squeeze then rf-lead wave (the +yaw family from scan v3, bounded bias)
        pats.append(("sq_wave_rf", {**sq3(0.25, 4),
                                   "RFJ1": (1.0, 0.0, 0.0, 0), "RFJ2": (1.0, 0.0, 0.0, 0),
                                   "MFJ1": (1.0, -2.1, 0.0, 0), "MFJ2": (1.0, -2.1, 0.0, 0),
                                   "FFJ1": (1.0, -4.2, 0.0, 0), "FFJ2": (1.0, -4.2, 0.0, 0)}, 12))
        pats.append(("sq_wave_rf_rev", {**sq3(0.25, 4),
                                       "RFJ1": (1.0, 0.0, 0.0, 0), "RFJ2": (1.0, 0.0, 0.0, 0),
                                       "MFJ1": (1.0, 2.1, 0.0, 0), "MFJ2": (1.0, 2.1, 0.0, 0),
                                       "FFJ1": (1.0, 4.2, 0.0, 0), "FFJ2": (1.0, 4.2, 0.0, 0)}, 12))
        # extension-drag wave: squeeze first, then small-amplitude reversed wave
        pats.append(("sq_wave_ff", {**sq3(0.25, 4),
                                   "FFJ1": (1.0, 0.0, 0.0, 0), "FFJ2": (1.0, 0.0, 0.0, 0),
                                   "MFJ1": (1.0, -2.1, 0.0, 0), "MFJ2": (1.0, -2.1, 0.0, 0),
                                   "RFJ1": (1.0, -4.2, 0.0, 0), "RFJ2": (1.0, -4.2, 0.0, 0)}, 12))
        # pinch roll: FF wave vs TH wave antiphase
        pats.append(("pinch_ff_th", {
            "FFJ1": (1.0, 0.0, 0.0, 0), "FFJ2": (1.0, 0.0, 0.0, 0),
            "THJ2": (1.0, P, 0.0, 0), "THJ1": (1.0, P, 0.0, 0),
            "MFJ1": (0.0, 0.0, 0.25, 4), "MFJ2": (0.0, 0.0, 0.25, 4)}, 12))
        pats.append(("pinch_ff_th_rev", {
            "FFJ1": (1.0, P, 0.0, 0), "FFJ2": (1.0, P, 0.0, 0),
            "THJ2": (1.0, 0.0, 0.0, 0), "THJ1": (1.0, 0.0, 0.0, 0),
            "MFJ1": (0.0, 0.0, 0.25, 4), "MFJ2": (0.0, 0.0, 0.25, 4)}, 12))

    if scan in ("bias", "all"):
        INF = 10 ** 9  # constant bias (relative control ramps to joint cap)
        for b in (0.2, 0.3, 0.4):
            pats.append((f"squeeze_only_b{b}", {
                "FFJ1": (0.0, 0.0, b, INF), "FFJ2": (0.0, 0.0, b, INF),
                "MFJ1": (0.0, 0.0, b, INF), "MFJ2": (0.0, 0.0, b, INF),
                "RFJ1": (0.0, 0.0, b, INF), "RFJ2": (0.0, 0.0, b, INF),
            }, 12))
        # biased rf-lead wave (the +yaw family), tracked per 10 steps
        pats.append(("bias_wave_rf", {
            "RFJ1": (1.0, 0.0, 0.3, INF), "RFJ2": (1.0, 0.0, 0.3, INF),
            "MFJ1": (1.0, -2.1, 0.3, INF), "MFJ2": (1.0, -2.1, 0.3, INF),
            "FFJ1": (1.0, -4.2, 0.3, INF), "FFJ2": (1.0, -4.2, 0.3, INF),
        }, 12))
        pats.append(("bias_wave_ff", {
            "FFJ1": (1.0, 0.0, 0.3, INF), "FFJ2": (1.0, 0.0, 0.3, INF),
            "MFJ1": (1.0, -2.1, 0.3, INF), "MFJ2": (1.0, -2.1, 0.3, INF),
            "RFJ1": (1.0, -4.2, 0.3, INF), "RFJ2": (1.0, -4.2, 0.3, INF),
        }, 12))
        # squeeze incl. little finger + thumb abduction to cage the block
        pats.append(("bias_wave_rf_cage", {
            "RFJ1": (1.0, 0.0, 0.3, INF), "RFJ2": (1.0, 0.0, 0.3, INF),
            "MFJ1": (1.0, -2.1, 0.3, INF), "MFJ2": (1.0, -2.1, 0.3, INF),
            "FFJ1": (1.0, -4.2, 0.3, INF), "FFJ2": (1.0, -4.2, 0.3, INF),
            "LFJ1": (0.0, 0.0, 0.3, INF), "LFJ2": (0.0, 0.0, 0.3, INF),
        }, 12))

    if scan in ("s2hunt", "all"):
        INF = 10 ** 9
        # deep-bias wave at higher amplitude
        pats.append(("plus_wave_hi", {
            "MFJ1": (0.8, 0.0, 0.45, INF), "MFJ2": (0.8, 0.0, 0.45, INF),
            "RFJ1": (0.8, -2.1, 0.45, INF), "RFJ2": (0.8, -2.1, 0.45, INF),
            "THJ4": (0.0, 0.0, 0.4, INF),
        }, 12))
        # 4-finger gait riding on a deep grip
        pats.append(("gait4_bias", {
            "FFJ1": (0.5, 0.0, 0.25, INF), "FFJ2": (0.5, 0.0, 0.25, INF),
            "MFJ1": (0.5, -1.57, 0.25, INF), "MFJ2": (0.5, -1.57, 0.25, INF),
            "RFJ1": (0.5, -3.14, 0.25, INF), "RFJ2": (0.5, -3.14, 0.25, INF),
            "LFJ1": (0.5, -4.71, 0.25, INF), "LFJ2": (0.5, -4.71, 0.25, INF),
        }, 16))
        pats.append(("gait4_bias_rev", {
            "FFJ1": (0.5, 0.0, 0.25, INF), "FFJ2": (0.5, 0.0, 0.25, INF),
            "MFJ1": (0.5, 1.57, 0.25, INF), "MFJ2": (0.5, 1.57, 0.25, INF),
            "RFJ1": (0.5, 3.14, 0.25, INF), "RFJ2": (0.5, 3.14, 0.25, INF),
            "LFJ1": (0.5, 4.71, 0.25, INF), "LFJ2": (0.5, 4.71, 0.25, INF),
        }, 16))
        # full-hand deep squeeze with thumb cage opened (s0 gave +15.6)
        pats.append(("squeeze_open_th4", {
            "FFJ1": (0.0, 0.0, 0.4, INF), "FFJ2": (0.0, 0.0, 0.4, INF),
            "MFJ1": (0.0, 0.0, 0.4, INF), "MFJ2": (0.0, 0.0, 0.4, INF),
            "RFJ1": (0.0, 0.0, 0.4, INF), "RFJ2": (0.0, 0.0, 0.4, INF),
            "THJ4": (0.0, 0.0, 0.4, INF),
        }, 10))
        # little-finger base joint (LFJ4) driving against RF cage
        pats.append(("lf4_drive", {
            "LFJ4": (0.6, 0.0, 0.0, 0),
            "LFJ1": (0.4, -1.57, 0.0, 0), "LFJ2": (0.4, -1.57, 0.0, 0),
            "RFJ1": (0.0, 0.0, 0.3, INF), "RFJ2": (0.0, 0.0, 0.3, INF),
        }, 10))
        # thumb base sweep with cage (thumb may engage on these geometries)
        pats.append(("th0_sweep_cage", {
            "THJ0": (0.8, 0.0, 0.0, 0), "THJ3": (0.5, -1.57, 0.0, 0),
            "FFJ1": (0.0, 0.0, 0.2, INF), "FFJ2": (0.0, 0.0, 0.2, INF),
            "MFJ1": (0.0, 0.0, 0.2, INF), "MFJ2": (0.0, 0.0, 0.2, INF),
        }, 12))
        # regrasp: open then close (contact-mode reset); measured for drift/yaw
        pats.append(("regrasp_oc", {
            "FFJ1": (0.6, P, 0.0, 0), "FFJ2": (0.6, P, 0.0, 0),
            "MFJ1": (0.6, P, 0.0, 0), "MFJ2": (0.6, P, 0.0, 0),
            "RFJ1": (0.6, P, 0.0, 0), "RFJ2": (0.6, P, 0.0, 0),
        }, 20))

    if scan in ("gait", "all"):
        INF = 10 ** 9
        # 4-finger walking gait (full phase cycle around the block)
        pats.append(("gait4", {
            "FFJ1": (0.5, 0.0, 0.0, 0), "FFJ2": (0.5, 0.0, 0.0, 0),
            "MFJ1": (0.5, -1.57, 0.0, 0), "MFJ2": (0.5, -1.57, 0.0, 0),
            "RFJ1": (0.5, -3.14, 0.0, 0), "RFJ2": (0.5, -3.14, 0.0, 0),
            "LFJ1": (0.5, -4.71, 0.0, 0), "LFJ2": (0.5, -4.71, 0.0, 0),
        }, 16))
        pats.append(("gait4_rev", {
            "FFJ1": (0.5, 0.0, 0.0, 0), "FFJ2": (0.5, 0.0, 0.0, 0),
            "MFJ1": (0.5, 1.57, 0.0, 0), "MFJ2": (0.5, 1.57, 0.0, 0),
            "RFJ1": (0.5, 3.14, 0.0, 0), "RFJ2": (0.5, 3.14, 0.0, 0),
            "LFJ1": (0.5, 4.71, 0.0, 0), "LFJ2": (0.5, 4.71, 0.0, 0),
        }, 16))
        # diagonal opposition: (FF+RF) vs (MF+LF) antiphase
        pats.append(("opp_pairs", {
            "FFJ1": (0.5, 0.0, 0.0, 0), "FFJ2": (0.5, 0.0, 0.0, 0),
            "RFJ1": (0.5, 0.0, 0.0, 0), "RFJ2": (0.5, 0.0, 0.0, 0),
            "MFJ1": (0.5, P, 0.0, 0), "MFJ2": (0.5, P, 0.0, 0),
            "LFJ1": (0.5, P, 0.0, 0), "LFJ2": (0.5, P, 0.0, 0),
        }, 12))
        pats.append(("opp_pairs_rev", {
            "FFJ1": (0.5, P, 0.0, 0), "FFJ2": (0.5, P, 0.0, 0),
            "RFJ1": (0.5, P, 0.0, 0), "RFJ2": (0.5, P, 0.0, 0),
            "MFJ1": (0.5, 0.0, 0.0, 0), "MFJ2": (0.5, 0.0, 0.0, 0),
            "LFJ1": (0.5, 0.0, 0.0, 0), "LFJ2": (0.5, 0.0, 0.0, 0),
        }, 12))
        # thumb-base rocking against a light two-finger cage
        pats.append(("th4_rock", {
            "THJ4": (0.6, 0.0, 0.0, 0),
            "MFJ1": (0.0, 0.0, 0.15, INF), "MFJ2": (0.0, 0.0, 0.15, INF),
            "RFJ1": (0.0, 0.0, 0.15, INF), "RFJ2": (0.0, 0.0, 0.15, INF),
        }, 10))
        pats.append(("th4_rock_rev", {
            "THJ4": (0.6, P, 0.0, 0),
            "MFJ1": (0.0, 0.0, 0.15, INF), "MFJ2": (0.0, 0.0, 0.15, INF),
            "RFJ1": (0.0, 0.0, 0.15, INF), "RFJ2": (0.0, 0.0, 0.15, INF),
        }, 10))
        # gentle wrist tilt assist with caged fingers
        pats.append(("wrist_tilt_a", {
            "WRJ0": (0.3, 0.0, 0.0, 0),
            "FFJ1": (0.0, 0.0, 0.15, INF), "FFJ2": (0.0, 0.0, 0.15, INF),
            "MFJ1": (0.0, 0.0, 0.15, INF), "MFJ2": (0.0, 0.0, 0.15, INF),
        }, 14))
        pats.append(("wrist_tilt_b", {
            "WRJ0": (0.3, P, 0.0, 0),
            "FFJ1": (0.0, 0.0, 0.15, INF), "FFJ2": (0.0, 0.0, 0.15, INF),
            "MFJ1": (0.0, 0.0, 0.15, INF), "MFJ2": (0.0, 0.0, 0.15, INF),
        }, 14))

    if scan in ("final", "all"):
        INF = 10 ** 9
        # +yaw finalists
        pats.append(("plus_mf_rf_th4_b35", {
            "MFJ1": (0.0, 0.0, 0.35, INF), "MFJ2": (0.0, 0.0, 0.35, INF),
            "RFJ1": (0.0, 0.0, 0.35, INF), "RFJ2": (0.0, 0.0, 0.35, INF),
            "THJ4": (0.0, 0.0, 0.4, INF),
        }, 10))
        pats.append(("plus_mf_rf_th4_b45", {
            "MFJ1": (0.0, 0.0, 0.45, INF), "MFJ2": (0.0, 0.0, 0.45, INF),
            "RFJ1": (0.0, 0.0, 0.45, INF), "RFJ2": (0.0, 0.0, 0.45, INF),
            "THJ4": (0.0, 0.0, 0.4, INF),
        }, 10))
        pats.append(("plus_mf_rf_th4_wave", {
            "MFJ1": (0.3, 0.0, 0.35, INF), "MFJ2": (0.3, 0.0, 0.35, INF),
            "RFJ1": (0.3, -2.1, 0.35, INF), "RFJ2": (0.3, -2.1, 0.35, INF),
            "THJ4": (0.0, 0.0, 0.4, INF),
        }, 12))
        pats.append(("plus_mf_rf_th43", {
            "MFJ1": (0.0, 0.0, 0.35, INF), "MFJ2": (0.0, 0.0, 0.35, INF),
            "RFJ1": (0.0, 0.0, 0.35, INF), "RFJ2": (0.0, 0.0, 0.35, INF),
            "THJ4": (0.0, 0.0, 0.4, INF), "THJ3": (0.0, 0.0, -0.3, INF),
        }, 10))
        # -yaw finalists
        pats.append(("minus_wave_ff_mf_rf", {
            "FFJ1": (0.6, 0.0, 0.0, 0), "FFJ2": (0.6, 0.0, 0.0, 0),
            "MFJ1": (0.6, -2.1, 0.0, 0), "MFJ2": (0.6, -2.1, 0.0, 0),
            "RFJ1": (0.6, -4.2, 0.0, 0), "RFJ2": (0.6, -4.2, 0.0, 0),
        }, 12))
        pats.append(("minus_steer_ff_wave", {
            "FFJ1": (0.5, 0.0, 0.3, INF), "FFJ2": (0.5, 0.0, 0.3, INF),
        }, 12))

    if scan in ("fmap", "all"):
        INF = 10 ** 9
        for f in ("FF", "MF", "RF", "LF"):
            pats.append((f"solo_{f}", {
                f"{f}J1": (0.0, 0.0, 0.35, INF), f"{f}J2": (0.0, 0.0, 0.35, INF),
            }, 10))
        # positive-side combo with thumb cage opened
        pats.append(("steer_mf_rf_th4", {
            "MFJ1": (0.0, 0.0, 0.35, INF), "MFJ2": (0.0, 0.0, 0.35, INF),
            "RFJ1": (0.0, 0.0, 0.35, INF), "RFJ2": (0.0, 0.0, 0.35, INF),
            "THJ4": (0.0, 0.0, 0.4, INF),
        }, 10))
        pats.append(("steer_mf_rf_lf_th4", {
            "MFJ1": (0.0, 0.0, 0.35, INF), "MFJ2": (0.0, 0.0, 0.35, INF),
            "RFJ1": (0.0, 0.0, 0.35, INF), "RFJ2": (0.0, 0.0, 0.35, INF),
            "LFJ1": (0.0, 0.0, 0.35, INF), "LFJ2": (0.0, 0.0, 0.35, INF),
            "THJ4": (0.0, 0.0, 0.4, INF),
        }, 10))
        # RF-only with thumb opened
        pats.append(("steer_rf_th4", {
            "RFJ1": (0.0, 0.0, 0.35, INF), "RFJ2": (0.0, 0.0, 0.35, INF),
            "THJ4": (0.0, 0.0, 0.4, INF),
        }, 10))
        # MF+RF flexion while FF cages lightly (hold block against translation)
        pats.append(("steer_mf_rf_ffcage", {
            "MFJ1": (0.0, 0.0, 0.35, INF), "MFJ2": (0.0, 0.0, 0.35, INF),
            "RFJ1": (0.0, 0.0, 0.35, INF), "RFJ2": (0.0, 0.0, 0.35, INF),
            "FFJ1": (0.0, 0.0, 0.12, INF), "FFJ2": (0.0, 0.0, 0.12, INF),
        }, 10))

    if scan in ("pulse", "all"):
        # in-phase squeeze pulse: flex all three fingers then release (sine),
        # friction asymmetry may leave net yaw without the constant-bias climb
        pats.append(("pulse3_t16", {
            "FFJ1": (1.0, 0.0, 0.0, 0), "FFJ2": (1.0, 0.0, 0.0, 0),
            "MFJ1": (1.0, 0.0, 0.0, 0), "MFJ2": (1.0, 0.0, 0.0, 0),
            "RFJ1": (1.0, 0.0, 0.0, 0), "RFJ2": (1.0, 0.0, 0.0, 0),
        }, 16))
        pats.append(("pulse3_t20", {
            "FFJ1": (1.0, 0.0, 0.0, 0), "FFJ2": (1.0, 0.0, 0.0, 0),
            "MFJ1": (1.0, 0.0, 0.0, 0), "MFJ2": (1.0, 0.0, 0.0, 0),
            "RFJ1": (1.0, 0.0, 0.0, 0), "RFJ2": (1.0, 0.0, 0.0, 0),
        }, 20))
        # squeeze pulse starting from a mildly pre-flexed pose (one-time bias 4 steps)
        pats.append(("pulse3_pre", {
            "FFJ1": (1.0, 0.0, 0.2, 4), "FFJ2": (1.0, 0.0, 0.2, 4),
            "MFJ1": (1.0, 0.0, 0.2, 4), "MFJ2": (1.0, 0.0, 0.2, 4),
            "RFJ1": (1.0, 0.0, 0.2, 4), "RFJ2": (1.0, 0.0, 0.2, 4),
        }, 16))
        # one-sided pulse (ulnar): RF+LF only
        pats.append(("pulse_rf_lf", {
            "RFJ1": (1.0, 0.0, 0.0, 0), "RFJ2": (1.0, 0.0, 0.0, 0),
            "LFJ1": (1.0, 0.0, 0.0, 0), "LFJ2": (1.0, 0.0, 0.0, 0),
        }, 16))
        pats.append(("pulse_ff_mf", {
            "FFJ1": (1.0, 0.0, 0.0, 0), "FFJ2": (1.0, 0.0, 0.0, 0),
            "MFJ1": (1.0, 0.0, 0.0, 0), "MFJ2": (1.0, 0.0, 0.0, 0),
        }, 16))

    if scan in ("steer", "all"):
        INF = 10 ** 9
        # one-sided deep flexion: steering-wheel roll, translation should cancel
        pats.append(("steer_rf_lf", {
            "RFJ1": (0.0, 0.0, 0.35, INF), "RFJ2": (0.0, 0.0, 0.35, INF),
            "LFJ1": (0.0, 0.0, 0.35, INF), "LFJ2": (0.0, 0.0, 0.35, INF),
        }, 10))
        pats.append(("steer_ff", {
            "FFJ1": (0.0, 0.0, 0.35, INF), "FFJ2": (0.0, 0.0, 0.35, INF),
        }, 10))
        pats.append(("steer_mf_rf", {
            "MFJ1": (0.0, 0.0, 0.35, INF), "MFJ2": (0.0, 0.0, 0.35, INF),
            "RFJ1": (0.0, 0.0, 0.35, INF), "RFJ2": (0.0, 0.0, 0.35, INF),
        }, 10))
        pats.append(("steer_ff_mf", {
            "FFJ1": (0.0, 0.0, 0.35, INF), "FFJ2": (0.0, 0.0, 0.35, INF),
            "MFJ1": (0.0, 0.0, 0.35, INF), "MFJ2": (0.0, 0.0, 0.35, INF),
        }, 10))
        # one-sided flexion + wave on same fingers to keep rolling contact
        pats.append(("steer_rf_lf_wave", {
            "RFJ1": (1.0, 0.0, 0.3, INF), "RFJ2": (1.0, 0.0, 0.3, INF),
            "LFJ1": (1.0, -2.1, 0.3, INF), "LFJ2": (1.0, -2.1, 0.3, INF),
        }, 12))
        pats.append(("steer_ff_wave", {
            "FFJ1": (1.0, 0.0, 0.3, INF), "FFJ2": (1.0, 0.0, 0.3, INF),
        }, 12))
        # deep flexion with thumb cage opened (THJ4 abduct + THJ3)
        pats.append(("squeeze_open_th4", {
            "FFJ1": (0.0, 0.0, 0.4, INF), "FFJ2": (0.0, 0.0, 0.4, INF),
            "MFJ1": (0.0, 0.0, 0.4, INF), "MFJ2": (0.0, 0.0, 0.4, INF),
            "RFJ1": (0.0, 0.0, 0.4, INF), "RFJ2": (0.0, 0.0, 0.4, INF),
            "THJ4": (0.0, 0.0, 0.4, INF),
        }, 10))
        # deep flexion + slight wrist pronation to keep block centered
        pats.append(("squeeze_wrist", {
            "FFJ1": (0.0, 0.0, 0.4, INF), "FFJ2": (0.0, 0.0, 0.4, INF),
            "MFJ1": (0.0, 0.0, 0.4, INF), "MFJ2": (0.0, 0.0, 0.4, INF),
            "RFJ1": (0.0, 0.0, 0.4, INF), "RFJ2": (0.0, 0.0, 0.4, INF),
            "WRJ1": (0.4, 0.0, 0.0, 0),
        }, 12))

    return pats


def pkg_patterns():
    """Evaluate the frozen inhand_gpt.primitives registry directly."""
    from inhand_gpt.primitives import PRIMITIVES
    return [(name, p) for name, p in PRIMITIVES.items()]


def run_primitive(seed: int, prim, n_steps: int = None):
    """Run a frozen Primitive object from the package."""
    env = gym.make("HandManipulateBlockRotateZ-v1", max_episode_steps=500)
    env.reset(seed=seed)
    q0 = _object_qpos(env)[3:7]
    p0 = _object_qpos(env)[0:3]
    n = n_steps or prim.n_steps
    max_drift = 0.0
    for t in range(n):
        env.step(prim.action(t))
        qpos = _object_qpos(env)
        max_drift = max(max_drift, float(np.linalg.norm(qpos[:2] - p0[:2])))
    qpos = _object_qpos(env)
    yaw = signed_yaw(qpos[3:7], q0)
    env.close()
    return {
        "yaw_deg": round(math.degrees(yaw), 2),
        "drift_xy_cm": round(float(np.linalg.norm(qpos[:2] - p0[:2])) * 100, 2),
        "max_drift_cm": round(max_drift * 100, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="if given, overrides --seed and loops over seeds")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--scan", default="all")
    ap.add_argument("--amps", type=float, nargs="+", default=[0.2, 0.4, 0.6])
    args = ap.parse_args()

    if args.scan == "pkg":
        seeds = args.seeds if args.seeds else [args.seed]
        print(f"pkg primitive validation: seeds={seeds}")
        for name, prim in pkg_patterns():
            yaws, drifts = [], []
            for seed in seeds:
                m = run_primitive(seed, prim)
                yaws.append(m["yaw_deg"])
                drifts.append(m["max_drift_cm"])
                print(f"  {name:10s} seed {seed}: yaw={m['yaw_deg']:+8.2f}deg "
                      f"drift_xy={m['drift_xy_cm']:6.2f}cm max_drift={m['max_drift_cm']:6.2f}cm",
                      flush=True)
            print(f"  {name:10s} mean_yaw={sum(yaws) / len(yaws):+8.2f}deg "
                  f"yaws={yaws} max_drift={max(drifts):.2f}cm")
        return

    print(f"tune: seed={args.seed} steps={args.steps} amps={args.amps} scan={args.scan}")
    print(f"{'candidate':22s} {'A':>4s} {'yaw_total_deg':>13s} {'yaw@segs':>24s} "
          f"{'drift_xy_cm':>11s} {'drift_z_cm':>10s} {'max_drift_cm':>12s}")
    results = []
    seeds = args.seeds if args.seeds else [args.seed]
    for seed in seeds:
        if len(seeds) > 1:
            print(f"-- seed {seed} --")
        for label, pattern, period in candidate_patterns(args.scan):
            for A in (args.amps if pattern else [0.0]):
                scaled = {k: (amp * A, ph, b, bs) for k, (amp, ph, b, bs) in pattern.items()}
                m = run_pattern(seed, scaled, args.steps, period)
                results.append((f"{label}@s{seed}" if len(seeds) > 1 else label, A, m))
                print(f"{label:22s} {A:4.1f} {m['yaw_deg_total']:13.2f} "
                      f"{str(m['yaw_deg_per20']):>24s} {m['drift_xy_cm']:11.2f} "
                      f"{m['drift_z_cm']:10.2f} {m['max_drift_cm']:12.2f} "
                      f"drift@segs={m['drift_cm_per20']}", flush=True)

    print("\nwinners (|yaw_total|>=2deg AND max_drift<1cm), sorted by |yaw|:")
    ok = [r for r in results if abs(r[2]["yaw_deg_total"]) >= 2.0 and r[2]["max_drift_cm"] < 1.0]
    for label, A, m in sorted(ok, key=lambda r: -abs(r[2]["yaw_deg_total"])):
        print(f"  {label:22s} A={A:.1f} yaw={m['yaw_deg_total']:+.2f}deg "
              f"drift_xy={m['drift_xy_cm']:.2f}cm max_drift={m['max_drift_cm']:.2f}cm")
    if not ok:
        print("  (none passed the filter)")


if __name__ == "__main__":
    main()
