"""Observation packing: quaternions -> signed Z-angle error, structured dicts.

All functions are pure (no env state kept here); the loop owns episode state.

Conventions
-----------
- Quaternions are [w, x, y, z] (MuJoCo convention).
- The signed Z error is the policy-facing signal:
      q_rel = q_achieved * conj(q_desired)
      err   = wrap(2 * atan2(q_rel.z, q_rel.w))        # radians, [-pi, pi]
  Positive err means the block must rotate further about +Z (world) to reach
  the goal.
- The success criterion mirrors the Gymnasium-Robotics env:
      d_rot = 2 * acos(|<q_ach, q_des>|) < rotation_threshold (0.1 rad)
  (RotateZ ignores position; we still track drift for drop detection.)
"""
from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

ROTATION_THRESHOLD_RAD = 0.1  # env's rotation_threshold
DROP_DRIFT_XY_M = 0.025       # loop-level drop rule: horizontal drift
DROP_Z_M = 0.05               # loop-level drop rule: block fell this far below start


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    return np.asarray(q, dtype=np.float64) / (np.linalg.norm(q) + 1e-12)


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def signed_z_error_rad(q_ach: np.ndarray, q_des: np.ndarray) -> float:
    """Signed rotation error about world Z, radians in [-pi, pi]."""
    q_rel = quat_mul(quat_normalize(q_ach), quat_conjugate(quat_normalize(q_des)))
    return wrap_angle(2.0 * math.atan2(q_rel[3], q_rel[0]))


def rotation_distance_rad(q_ach: np.ndarray, q_des: np.ndarray) -> float:
    """Env-style unsigned geodesic rotation distance (radians)."""
    q_rel = quat_mul(quat_normalize(q_ach), quat_conjugate(quat_normalize(q_des)))
    return 2.0 * math.acos(max(-1.0, min(1.0, abs(q_rel[0]))))


def block_pose(env) -> tuple[np.ndarray, np.ndarray]:
    """Return (pos[3], quat[4]) of the manipulated block.

    Raw MjData in mujoco>=3 has no get_joint_qpos helper, so we read qpos
    directly at the free joint's address.
    """
    u = env.unwrapped
    jid = u.model.joint("object:joint").id
    adr = u.model.jnt_qposadr[jid]
    qpos = u.data.qpos[adr:adr + 7]
    # np.array copies: qpos is a live view into MjData
    return np.array(qpos[:3], dtype=np.float64), np.array(qpos[3:7], dtype=np.float64)


def pack_observation(
    obs: Dict[str, np.ndarray],
    initial_block_pos: np.ndarray,
    cycle: int,
) -> Dict[str, Any]:
    """Build the structured observation dict shared by both policies and records."""
    q_ach = np.asarray(obs["achieved_goal"][3:7], dtype=np.float64)
    q_des = np.asarray(obs["desired_goal"][3:7], dtype=np.float64)
    block_pos = np.asarray(obs["achieved_goal"][0:3], dtype=np.float64)

    err_rad = signed_z_error_rad(q_ach, q_des)
    d_rot = rotation_distance_rad(q_ach, q_des)
    drift = block_pos - np.asarray(initial_block_pos, dtype=np.float64)
    drift_xy = float(np.linalg.norm(drift[:2]))

    return {
        "cycle": cycle,
        "err_deg": round(math.degrees(err_rad), 3),
        "rot_dist_rad": round(d_rot, 4),
        "success": bool(d_rot < ROTATION_THRESHOLD_RAD),
        "block_pos": [round(float(v), 4) for v in block_pos],
        "drift_xy_m": round(drift_xy, 4),
        "drift_z_m": round(float(drift[2]), 4),
        "dropped": bool(drift_xy > DROP_DRIFT_XY_M or drift[2] < -DROP_Z_M),
    }
