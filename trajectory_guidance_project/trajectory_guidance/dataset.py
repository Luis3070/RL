from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import pickle

import numpy as np


@dataclass
class DemoSnapshot:
    stage_id: int
    episode_id: int
    t: int
    obs_local: np.ndarray
    state_global: np.ndarray
    teacher_action: np.ndarray
    next_obs_local: np.ndarray
    done: bool
    event: Dict[str, Any]
    trajectory_id: str
    quality_score: float

    # Fix #5: Campos para reproducibilidad exacta
    reset_seed: Optional[int] = None
    initial_agent_pos: Optional[np.ndarray] = None
    initial_ball_pos: Optional[np.ndarray] = None

    # BC metadata: intención del episodio y estado de la fase de kick
    intent: str = "dribble"                      # "dribble" | "shoot" | "pass"
    kick_phase_active: bool = False              # True cuando la fase kick está activa
    kick_target: Optional[np.ndarray] = None    # posición absoluta del target de kick/pase

    @classmethod
    def from_transition(cls, **kwargs) -> "DemoSnapshot":
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetIndex:
    by_stage: Dict[int, List[str]] = field(default_factory=dict)

    def add(self, stage_id: int, path: str) -> None:
        self.by_stage.setdefault(stage_id, []).append(path)


def export_snapshots_pkl(snapshots: Iterable[DemoSnapshot], out_path: str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump([s.to_dict() for s in snapshots], f)


def load_snapshots_pkl(path: str) -> List[dict]:
    with Path(path).open("rb") as f:
        return pickle.load(f)