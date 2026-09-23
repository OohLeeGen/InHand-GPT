"""Per-cycle recording and JSON export."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class Recorder:
    def __init__(self, policy_name: str, seed: int, max_cycles: int):
        self.meta: Dict[str, Any] = {
            "policy": policy_name,
            "seed": seed,
            "max_cycles": max_cycles,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self.cycles: List[Dict[str, Any]] = []
        self.summary: Optional[Dict[str, Any]] = None

    def log_cycle(self, record: Dict[str, Any]) -> None:
        self.cycles.append(record)

    def finish(self, outcome: str, final_obs: Dict[str, Any], wall_time_s: float) -> Dict[str, Any]:
        """outcome: 'success' | 'failure_max_cycles' | 'failure_dropped'"""
        self.summary = {
            "outcome": outcome,
            "n_cycles": len(self.cycles),
            "final_err_deg": final_obs["err_deg"],
            "final_rot_dist_rad": final_obs["rot_dist_rad"],
            "final_drift_xy_m": final_obs["drift_xy_m"],
            "wall_time_s": round(wall_time_s, 2),
        }
        return self.summary

    def export(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"meta": self.meta, "cycles": self.cycles, "summary": self.summary}
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path
