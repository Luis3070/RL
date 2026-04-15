from __future__ import annotations

import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional


def _to_jsonable(value: Any) -> Any:
    # Evitar dependencias pesadas: conversión defensiva.
    try:
        import numpy as np  # type: ignore

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.integer,)):
            return int(value)
    except Exception:
        pass

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return str(value)


class IncidentDumper:
    """
    Captura automática de incidentes: ring-buffer + trigger.

    Es intencionalmente simple: el objetivo es anclar el bug a un artifact JSON
    sin depender del visualizer, incluso si el episodio/step “se mueve”.
    """

    def __init__(
        self,
        log_dir: str,
        *,
        buffer_steps: int = 220,
        overlap_sep_mm: float = 8.0,
        overlap_streak: int = 8,
        cooldown_steps: int = 120,
    ) -> None:
        self._dir = Path(log_dir) / "incidents"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._buf: Deque[Dict[str, Any]] = deque(maxlen=max(50, int(buffer_steps)))
        self._overlap_sep_mm = float(overlap_sep_mm)
        self._overlap_streak_req = int(overlap_streak)
        self._cooldown_steps = int(cooldown_steps)
        self._cooldown = 0
        self._streak = 0

    @classmethod
    def maybe_create(cls, log_dir: str) -> Optional["IncidentDumper"]:
        raw = os.environ.get("INCIDENT_DUMP_ENABLED", "").strip().lower()
        if raw in {"0", "false", "no", "off"}:
            return None
        return cls(log_dir)

    def _should_trigger(self, info: Dict[str, Any]) -> bool:
        owner = str(info.get("control_owner", "") or "")
        if owner not in {"HANDOFF", "BHUMAN"}:
            self._streak = 0
            return False

        try:
            sep = float(info.get("diag_ball_agent_sep_mm", 9999.0) or 9999.0)
        except Exception:
            sep = 9999.0

        # “Severe overlap”: sep muy pequeño sostenido.
        if sep < self._overlap_sep_mm:
            self._streak += 1
        else:
            self._streak = 0

        if self._streak >= self._overlap_streak_req:
            return True

        # Otros triggers puntuales (si existen en la telemetría):
        try:
            if int(info.get("diag_post_kick_inversion_flag", 0) or 0) != 0:
                return True
            if int(info.get("diag_rapid_kick_proxy_flag", 0) or 0) != 0:
                return True
            if int(info.get("diag_theta_snap_prep_flag", 0) or 0) != 0:
                return True
            if int(info.get("diag_theta_snap_motion_prep_flag", 0) or 0) != 0:
                return True
        except Exception:
            pass

        return False

    def observe(self, *, episode: int, stage_id: int, stage_name: str, info: Dict[str, Any]) -> None:
        if not info:
            return
        if str(info.get("sidecar_backend", "")) != "real":
            return

        if self._cooldown > 0:
            self._cooldown -= 1

        row = dict(info)
        row["_meta"] = {
            "episode": int(episode),
            "stage_id": int(stage_id),
            "stage_name": str(stage_name),
            "wall_ts": time.time(),
        }
        self._buf.append(row)

        if self._cooldown > 0:
            return
        if not self._should_trigger(info):
            return

        env_step = int(float(info.get("step", 0) or 0))
        bridge_frame_id = int(float(info.get("bridge_frame_id", -1) or -1))
        owner = str(info.get("control_owner", "") or "")
        reason = "severe_overlap" if self._streak >= self._overlap_streak_req else "flag_trigger"

        payload = {
            "reason": reason,
            "episode": int(episode),
            "env_step": env_step,
            "bridge_frame_id": bridge_frame_id,
            "control_owner": owner,
            "buffer": [_to_jsonable(x) for x in list(self._buf)],
        }
        name = f"incident_ep{int(episode):04d}_st{env_step:04d}_bf{bridge_frame_id}_{reason}.json"
        (self._dir / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        # reset / cooldown
        self._cooldown = max(10, self._cooldown_steps)
        self._streak = 0

