"""
wrapper.py — NaoSoccerEnvWrapper: adapta NaoSoccerEnv para skrl MAPPO.

Alineado con contrato BHuman:
  - Curriculum advancement basado en takeover metrics, no touch/goal/win legacy
  - Métricas BHuman-aware: takeover_ready_rate, takeover_success_rate, etc.
  - Estado global para critic centralizado (CTDE)
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import (
    CURRICULUM, CurriculumStageConfig,
    GLOBAL_STATE, OBS, ACTION,
)
from environment import NaoSoccerEnv


# ─────────────────────────────────────────────────────────────────────────────
# CURRICULUM MANAGER — BHuman-aware advancement
# ─────────────────────────────────────────────────────────────────────────────

class CurriculumManager:
    """
    Manages curriculum advancement with BHuman-aware metrics.

    Advancement metrics:
      - takeover_rate: fraction of episodes where handoff_ready was achieved
      - takeover_success_rate: fraction of takeovers that succeeded
      - win_rate: fraction of episodes won
      - support_quality: fraction with good spacing metrics

    Early takeover stages can additionally require semantic guard rails:
      - minimum playBall entry rate
      - minimum takeover success rate
      - maximum ready_but_walk rates
    """

    def __init__(self, stages: List[CurriculumStageConfig] = CURRICULUM):
        self._stages = stages
        self._current_idx = 0
        self._window: deque = deque(maxlen=stages[0].advance_window)
        self._playball_window: deque = deque(maxlen=stages[0].advance_window)
        self._takeover_success_window: deque = deque(maxlen=stages[0].advance_window)
        self._ready_but_walk_window: deque = deque(maxlen=stages[0].advance_window)
        self._ready_touch_but_walk_window: deque = deque(maxlen=stages[0].advance_window)

    @property
    def current_stage(self) -> CurriculumStageConfig:
        return self._stages[self._current_idx]

    @property
    def stage_id(self) -> int:
        return self._current_idx

    @property
    def is_final_stage(self) -> bool:
        return self._current_idx == len(self._stages) - 1

    def record_episode(self,
                       own_score: int,
                       opp_score: int,
                       takeover_count: int = 0,
                       takeover_success_count: int = 0,
                       handoff_achieved: bool = False,
                       playball_achieved: bool = False,
                       ready_but_walk_rate: float = 0.0,
                       ready_touch_but_walk_rate: float = 0.0,
                       support_quality: float = 0.0) -> bool:
        """
        Record episode result and evaluate advancement.

        Returns True if advanced to next stage.
        """
        stage = self.current_stage
        metric = stage.advance_metric

        if metric == "takeover_rate":
            success = float(handoff_achieved)
        elif metric == "takeover_success_rate":
            if takeover_count > 0:
                success = float(takeover_success_count / takeover_count)
            else:
                success = 0.0
        elif metric == "win_rate":
            success = float(own_score > opp_score)
        elif metric == "support_quality":
            success = support_quality
        else:
            success = float(own_score > opp_score)

        self._window.append(float(success))
        self._playball_window.append(float(playball_achieved))
        self._takeover_success_window.append(
            float(takeover_success_count / takeover_count) if takeover_count > 0 else 0.0
        )
        self._ready_but_walk_window.append(float(ready_but_walk_rate))
        self._ready_touch_but_walk_window.append(float(ready_touch_but_walk_rate))

        if self._window.maxlen != stage.advance_window:
            self._window = deque(self._window, maxlen=stage.advance_window)
            self._playball_window = deque(self._playball_window, maxlen=stage.advance_window)
            self._takeover_success_window = deque(
                self._takeover_success_window, maxlen=stage.advance_window
            )
            self._ready_but_walk_window = deque(
                self._ready_but_walk_window, maxlen=stage.advance_window
            )
            self._ready_touch_but_walk_window = deque(
                self._ready_touch_but_walk_window, maxlen=stage.advance_window
            )

        if (len(self._window) == stage.advance_window
                and not self.is_final_stage):
            rate = float(np.mean(self._window))
            playball_rate = float(np.mean(self._playball_window)) if self._playball_window else 0.0
            takeover_success_rate = (
                float(np.mean(self._takeover_success_window))
                if self._takeover_success_window else 0.0
            )
            ready_but_walk_mean = (
                float(np.mean(self._ready_but_walk_window))
                if self._ready_but_walk_window else 0.0
            )
            ready_touch_but_walk_mean = (
                float(np.mean(self._ready_touch_but_walk_window))
                if self._ready_touch_but_walk_window else 0.0
            )
            guard_ok = (
                playball_rate >= stage.min_playball_entry_rate
                and takeover_success_rate >= stage.min_takeover_success_rate
                and ready_but_walk_mean <= stage.max_ready_but_walk_rate
                and ready_touch_but_walk_mean <= stage.max_ready_touch_but_walk_rate
            )
            if rate >= stage.advance_threshold and guard_ok:
                self._advance()
                return True

        return False

    def _advance(self):
        self._current_idx += 1
        new_stage = self.current_stage
        self._window = deque(maxlen=new_stage.advance_window)
        self._playball_window = deque(maxlen=new_stage.advance_window)
        self._takeover_success_window = deque(maxlen=new_stage.advance_window)
        self._ready_but_walk_window = deque(maxlen=new_stage.advance_window)
        self._ready_touch_but_walk_window = deque(maxlen=new_stage.advance_window)
        print(f"\n[Curriculum] Advanced to Stage {self._current_idx}: {new_stage.name}")

    def jump_to_stage(self, stage_idx: int):
        if not (0 <= stage_idx < len(self._stages)):
            raise ValueError(f"stage_idx={stage_idx} out of range [0, {len(self._stages)-1}]")
        self._current_idx = stage_idx
        new_stage = self.current_stage
        self._window = deque(maxlen=new_stage.advance_window)
        self._playball_window = deque(maxlen=new_stage.advance_window)
        self._takeover_success_window = deque(maxlen=new_stage.advance_window)
        self._ready_but_walk_window = deque(maxlen=new_stage.advance_window)
        self._ready_touch_but_walk_window = deque(maxlen=new_stage.advance_window)
        print(f"[Curriculum] jump_to_stage -> Stage {stage_idx}: {new_stage.name}")

    def get_progress_stats(self) -> Dict[str, Any]:
        stage = self.current_stage
        rate = float(np.mean(self._window)) if self._window else 0.0
        return {
            "stage_id": self._current_idx,
            "stage_name": stage.name,
            "advance_metric": stage.advance_metric,
            "current_rate": rate,
            "playball_entry_rate": (
                float(np.mean(self._playball_window)) if self._playball_window else 0.0
            ),
            "takeover_success_rate": (
                float(np.mean(self._takeover_success_window))
                if self._takeover_success_window else 0.0
            ),
            "ready_but_walk_rate": (
                float(np.mean(self._ready_but_walk_window))
                if self._ready_but_walk_window else 0.0
            ),
            "ready_touch_but_walk_rate": (
                float(np.mean(self._ready_touch_but_walk_window))
                if self._ready_touch_but_walk_window else 0.0
            ),
            "advance_threshold": stage.advance_threshold,
            "window_fill": f"{len(self._window)}/{stage.advance_window}",
        }


# ─────────────────────────────────────────────────────────────────────────────
# BHUMAN METRICS AGGREGATOR
# ─────────────────────────────────────────────────────────────────────────────

class BHumanMetricsAggregator:
    """
    Aggregates BHuman-specific metrics across episodes.

    Reports:
      - takeover_ready_rate: how often handoff_ready is achieved
      - takeover_success_rate: success rate of BHuman executions
      - bhuman_exec_success_rate: = takeover_success_rate (alias)
      - support_quality_rate: spacing quality
      - anti_blocking_rate: how often agents avoid blocking
    """

    def __init__(self, window_size: int = 100):
        self._window_size = window_size
        self._takeover_ready: deque = deque(maxlen=window_size)
        self._takeover_success: deque = deque(maxlen=window_size)
        self._playball_entry: deque = deque(maxlen=window_size)
        self._ready_but_walk: deque = deque(maxlen=window_size)
        self._ready_touch_but_walk: deque = deque(maxlen=window_size)
        self._support_quality: deque = deque(maxlen=window_size)
        self._anti_blocking: deque = deque(maxlen=window_size)
        self._goals: deque = deque(maxlen=window_size)
        self._wins: deque = deque(maxlen=window_size)

    def record(self,
               handoff_achieved: bool,
               takeover_count: int,
               takeover_success_count: int,
               playball_achieved: bool,
               ready_but_walk_rate: float,
               ready_touch_but_walk_rate: float,
               own_score: int,
               opp_score: int,
               support_quality: float = 0.5,
               blocking_events: int = 0):
        self._takeover_ready.append(float(handoff_achieved))
        if takeover_count > 0:
            self._takeover_success.append(float(takeover_success_count / takeover_count))
        else:
            self._takeover_success.append(0.0)
        self._playball_entry.append(float(playball_achieved))
        self._ready_but_walk.append(float(ready_but_walk_rate))
        self._ready_touch_but_walk.append(float(ready_touch_but_walk_rate))
        self._support_quality.append(support_quality)
        self._anti_blocking.append(1.0 if blocking_events == 0 else 0.0)
        self._goals.append(float(own_score > 0))
        self._wins.append(float(own_score > opp_score))

    def get_metrics(self) -> Dict[str, float]:
        return {
            "takeover_ready_rate": self._safe_mean(self._takeover_ready),
            "takeover_success_rate": self._safe_mean(self._takeover_success),
            "bhuman_exec_success_rate": self._safe_mean(self._takeover_success),
            "playball_entry_rate": self._safe_mean(self._playball_entry),
            "ready_but_walk_rate": self._safe_mean(self._ready_but_walk),
            "ready_touch_but_walk_rate": self._safe_mean(self._ready_touch_but_walk),
            "support_quality_rate": self._safe_mean(self._support_quality),
            "anti_blocking_rate": self._safe_mean(self._anti_blocking),
            "goal_rate": self._safe_mean(self._goals),
            "win_rate": self._safe_mean(self._wins),
        }

    @staticmethod
    def _safe_mean(d: deque) -> float:
        return float(np.mean(d)) if d else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# WRAPPER PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

class NaoSoccerEnvWrapper:
    """
    Wrapper over NaoSoccerEnv for skrl MAPPO compatibility.

    Adds:
      - "state": global state vector for centralized critic
      - BHuman-aware curriculum management
      - BHuman metrics aggregation
    """

    def __init__(self,
                 stage_cfg: Optional[CurriculumStageConfig] = None,
                 curriculum_stages: Optional[List[CurriculumStageConfig]] = None,
                 use_fov: bool = True,
                 seed: Optional[int] = None):

        self._curriculum_stages = curriculum_stages or CURRICULUM

        if stage_cfg is None:
            stage_cfg = self._curriculum_stages[0]

        self._env = NaoSoccerEnv(stage_cfg=stage_cfg, use_fov=use_fov, seed=seed)
        self.curriculum_manager = CurriculumManager(self._curriculum_stages)
        self.metrics = BHumanMetricsAggregator()

        # PettingZoo interface
        self.possible_agents = self._env.possible_agents
        self.observation_spaces = self._env.observation_spaces
        self.action_spaces = self._env.action_spaces
        self.metadata = self._env.metadata

        # Shared observation space for centralized critic
        import gymnasium as gym
        self.shared_observation_spaces = {
            aid: gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(GLOBAL_STATE.total,), dtype=np.float32
            )
            for aid in self.possible_agents
        }

    @property
    def agents(self) -> List[str]:
        return self._env.agents

    def observation_space(self, agent):
        return self._env.observation_space(agent)

    def action_space(self, agent):
        return self._env.action_space(agent)

    def reset(self, seed=None, options=None):
        self._sync_stage()
        obs, infos = self._env.reset(seed=seed, options=options)
        self._inject_global_state(infos)
        return obs, infos

    def step(self, actions):
        obs, rewards, terminateds, truncateds, infos = self._env.step(actions)
        self._inject_global_state(infos)
        return obs, rewards, terminateds, truncateds, infos

    def record_episode_metrics(self,
                               own_score: int,
                               opp_score: int,
                               infos: Dict[str, Dict]) -> bool:
        """
        Record episode metrics and check for curriculum advancement.

        Call this at the end of each episode from train.py.
        Returns True if curriculum advanced.
        """
        # Extract BHuman metrics from last info
        sample_info = next(iter(infos.values())) if infos else {}
        takeover_count = sample_info.get("episode_takeover_count", 0)
        takeover_success = sample_info.get("episode_takeover_success", 0)
        # handoff_achieved = True if handoff_ready envelope was reached at any point
        handoff_achieved = sample_info.get("episode_handoff_achieved", False)
        playball_achieved = sample_info.get("episode_playball_achieved", False)
        ready_but_walk_rate = float(sample_info.get("episode_ready_but_walk_rate", 0.0))
        ready_touch_but_walk_rate = float(
            sample_info.get("episode_ready_touch_but_walk_rate", 0.0)
        )

        # Record in metrics aggregator
        self.metrics.record(
            handoff_achieved=handoff_achieved,
            takeover_count=takeover_count,
            takeover_success_count=takeover_success,
            playball_achieved=playball_achieved,
            ready_but_walk_rate=ready_but_walk_rate,
            ready_touch_but_walk_rate=ready_touch_but_walk_rate,
            own_score=own_score,
            opp_score=opp_score,
        )

        # Record in curriculum manager
        return self.curriculum_manager.record_episode(
            own_score=own_score,
            opp_score=opp_score,
            takeover_count=takeover_count,
            takeover_success_count=takeover_success,
            handoff_achieved=handoff_achieved,
            playball_achieved=playball_achieved,
            ready_but_walk_rate=ready_but_walk_rate,
            ready_touch_but_walk_rate=ready_touch_but_walk_rate,
        )

    def set_curriculum_stage(self, stage_idx: int):
        if 0 <= stage_idx < len(self._curriculum_stages):
            self.curriculum_manager.jump_to_stage(stage_idx)
            self._env.set_stage(self._curriculum_stages[stage_idx])

    def _sync_stage(self):
        self._env.set_stage(self.curriculum_manager.current_stage)

    def _inject_global_state(self, infos):
        global_state = self._env.get_global_state()
        for aid in infos:
            infos[aid]["state"] = global_state

    def collect_episode(self, policy_fn=None, seed=None) -> Dict[str, Any]:
        obs, _ = self.reset(seed=seed)
        total_rewards = {aid: 0.0 for aid in self.agents}
        steps = 0

        while self.agents:
            if policy_fn is not None:
                actions = policy_fn(obs)
            else:
                actions = {
                    aid: self.action_space(aid).sample()
                    for aid in self.agents
                }

            obs, rewards, terminateds, truncateds, infos = self.step(actions)
            steps += 1

            for aid, r in rewards.items():
                total_rewards[aid] = total_rewards.get(aid, 0.0) + r

            if truncateds.get("__all__") or terminateds.get("__all__"):
                break

        bhuman_metrics = self.metrics.get_metrics()

        return {
            "steps": steps,
            "own_score": self._env._game.own_score,
            "opp_score": self._env._game.opp_score,
            "total_rewards": total_rewards,
            "mean_reward": float(np.mean(list(total_rewards.values()))) if total_rewards else 0.0,
            "stage_id": self.curriculum_manager.stage_id,
            "stage_name": self.curriculum_manager.current_stage.name,
            **bhuman_metrics,
        }

    def close(self):
        self._env.close()
