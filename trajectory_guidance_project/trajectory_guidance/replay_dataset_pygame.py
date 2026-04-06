from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np

# Allow running this file directly (Run Python File) without -m.
if __package__ in (None, ""):
    script_dir = Path(__file__).resolve().parent
    package_dir = script_dir
    project_dir = package_dir.parent
    workspace_root = project_dir.parent

    sys.path = [p for p in sys.path if Path(p).resolve() != script_dir]
    sys.path.insert(0, str(workspace_root))
    sys.path.insert(1, str(project_dir))

import pygame

from config import CURRICULUM
from visualizer import CoordTransform, EnvStateReader, Renderer
from wrapper import NaoSoccerEnvWrapper

from trajectory_guidance.settings import DATASET_CFG
from trajectory_guidance.dataset import load_snapshots_pkl


@dataclass
class TrajectorySlice:
    stage_id: int
    episode_id: int
    trajectory_id: str
    actions: List[np.ndarray]
    quality_score: float
    reset_seed: int | None = None


def _group_trajectories(records: List[dict]) -> List[TrajectorySlice]:
    groups: "OrderedDict[Tuple[int, int, str], List[dict]]" = OrderedDict()
    for rec in records:
        key = (int(rec["stage_id"]), int(rec["episode_id"]), str(rec["trajectory_id"]))
        groups.setdefault(key, []).append(rec)

    result: List[TrajectorySlice] = []
    for (stage_id, episode_id, traj_id), rows in groups.items():
        rows.sort(key=lambda x: int(x["t"]))
        actions = [np.asarray(r["teacher_action"], dtype=np.float32) for r in rows]
        quality = float(rows[0].get("quality_score", 0.0))
        # Extraer semilla de reseteo si existe
        seed = rows[0].get("reset_seed", None)
        if seed is not None:
            seed = int(seed)
        result.append(TrajectorySlice(
            stage_id=stage_id,
            episode_id=episode_id,
            trajectory_id=traj_id,
            actions=actions,
            quality_score=quality,
            reset_seed=seed,
        ))
    return result


class DatasetTrajectoryPlayer:
    W = 1200
    H = 700
    FPS = 30

    def __init__(self, dataset_path: str, use_fov: bool = True, seed: int = 42):
        self._dataset_path = Path(dataset_path)
        if not self._dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self._dataset_path}")

        raw = load_snapshots_pkl(str(self._dataset_path))
        self._trajectories = _group_trajectories(raw)
        if not self._trajectories:
            raise RuntimeError(f"No trajectories found in {self._dataset_path}")

        pygame.init()
        pygame.display.set_caption("Trajectory Replay (dataset) - pygame")
        self._screen = pygame.display.set_mode((self.W, self.H))
        self._clock = pygame.time.Clock()

        self._env = NaoSoccerEnvWrapper(use_fov=use_fov, seed=seed)
        self._reader = EnvStateReader(self._env._env)
        self._renderer = Renderer(self._screen, CoordTransform(self.W, self.H))

        self._running = True
        self._paused = False
        self._show_fov = use_fov
        self._auto_advance = True

        self._traj_idx = 0
        self._step_idx = 0
        self._episode = 0
        self._ep_reward = 0.0
        self._step_rewards: Dict[str, float] = {}
        self._trace: List[Tuple[float, float]] = []

        self._load_trajectory(reset_episode=True)

    @property
    def _current(self) -> TrajectorySlice:
        return self._trajectories[self._traj_idx]

    def _load_trajectory(self, reset_episode: bool) -> None:
        t = self._current
        self._env.set_curriculum_stage(t.stage_id)
        # Usar la semilla almacenada para reproducir el estado exacto
        # de la generación. Sin esto, las acciones apuntan a donde
        # estaba el balón en otro mundo, no en este.
        self._obs, _ = self._env.reset(seed=t.reset_seed)
        self._step_idx = 0
        self._ep_reward = 0.0
        self._step_rewards = {}
        self._trace = []
        if reset_episode:
            self._episode += 1

    def _make_actions(self, teacher_action: np.ndarray) -> Dict[str, np.ndarray]:
        actions: Dict[str, np.ndarray] = {}
        for aid in self._env.agents:
            if aid == DATASET_CFG.controlled_agent_id:
                actions[aid] = np.asarray(teacher_action, dtype=np.float32)
            else:
                actions[aid] = np.zeros(3, dtype=np.float32)
        return actions

    def _step(self) -> None:
        cur = self._current
        if self._step_idx >= len(cur.actions):
            if self._auto_advance:
                self._next_trajectory()
            return

        if not self._env.agents:
            self._load_trajectory(reset_episode=False)
            return

        action = cur.actions[self._step_idx]
        actions = self._make_actions(action)
        self._obs, rewards, terms, truncs, _ = self._env.step(actions)
        self._step_idx += 1

        self._step_rewards = {aid: float(r) for aid, r in rewards.items()}
        self._ep_reward += float(sum(rewards.values()))

        aid = DATASET_CFG.controlled_agent_id
        if aid in self._reader.agents:
            pos = self._reader.agents[aid].pos
            self._trace.append((float(pos[0]), float(pos[1])))

        if terms.get("__all__") or truncs.get("__all__"):
            if self._auto_advance:
                self._next_trajectory()

    def _draw_trace(self) -> None:
        if len(self._trace) < 2:
            return
        points = [self._renderer._t.to_px(x, y) for x, y in self._trace]
        pygame.draw.lines(self._screen, (255, 255, 0), False, points, 2)

    def _render_overlay(self) -> None:
        font = pygame.font.SysFont("monospace", 14)
        cur = self._current
        lines = [
            f"dataset: {self._dataset_path}",
            f"traj: {self._traj_idx + 1}/{len(self._trajectories)}",
            f"stage: {cur.stage_id} ({CURRICULUM[cur.stage_id].name})",
            f"episode_id: {cur.episode_id}",
            f"trajectory_id: {cur.trajectory_id}",
            f"quality_score: {cur.quality_score:.3f}",
            f"step: {self._step_idx}/{len(cur.actions)}",
            f"auto_advance: {'ON' if self._auto_advance else 'OFF'}",
            "keys: SPACE pause | RIGHT step | N next | B prev | R reset | A auto | F fov | Q quit",
        ]
        y = self.H - (len(lines) * 18 + 10)
        bg = pygame.Surface((self.W - 16, len(lines) * 18 + 8), pygame.SRCALPHA)
        bg.fill((20, 20, 20, 170))
        self._screen.blit(bg, (8, y - 4))
        for i, text in enumerate(lines):
            self._screen.blit(font.render(text, True, (240, 240, 240)), (14, y + i * 18))

    def _render(self) -> None:
        cur = self._current
        self._renderer.draw_field()
        if self._show_fov:
            for agent in self._reader.agents.values():
                self._renderer.draw_fov_cone(agent)
        for opp in self._reader.opponents:
            self._renderer.draw_opponent(opp)
        for agent in self._reader.agents.values():
            self._renderer.draw_agent(agent)
        self._draw_trace()
        self._renderer.draw_ball(self._reader.ball, self._reader.nearest_agent_dist())
        self._renderer.draw_hud(
            game=self._reader.game,
            stage=CURRICULUM[cur.stage_id],
            episode=self._episode,
            ep_reward=self._ep_reward,
            show_fov=self._show_fov,
            paused=self._paused,
            step_rewards=self._step_rewards,
        )
        self._render_overlay()
        pygame.display.flip()

    def _next_trajectory(self) -> None:
        self._traj_idx = (self._traj_idx + 1) % len(self._trajectories)
        self._load_trajectory(reset_episode=True)

    def _prev_trajectory(self) -> None:
        self._traj_idx = (self._traj_idx - 1) % len(self._trajectories)
        self._load_trajectory(reset_episode=True)

    def _handle_key(self, key: int) -> None:
        if key in (pygame.K_q, pygame.K_ESCAPE):
            self._running = False
        elif key == pygame.K_SPACE:
            self._paused = not self._paused
        elif key == pygame.K_RIGHT and self._paused:
            self._step()
        elif key == pygame.K_n:
            self._next_trajectory()
        elif key == pygame.K_b:
            self._prev_trajectory()
        elif key == pygame.K_r:
            self._load_trajectory(reset_episode=False)
        elif key == pygame.K_a:
            self._auto_advance = not self._auto_advance
        elif key == pygame.K_f:
            self._show_fov = not self._show_fov
            self._env._env._use_fov = self._show_fov
            self._env._env._fov_filter._use_fov = self._show_fov

    def run(self) -> None:
        while self._running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    self._running = False
                elif ev.type == pygame.KEYDOWN:
                    self._handle_key(ev.key)

            if not self._paused:
                self._step()
            self._render()
            self._clock.tick(self.FPS)
        pygame.quit()


def main() -> None:
    # Quick edit config: adjust these values and press Run Python File.
    DATASET_PATH = r"C:\ALEJO\trajectory_demos\demos_fixed0_partial.pkl"
    USE_FOV = True
    SEED = 42

    player = DatasetTrajectoryPlayer(
        dataset_path=DATASET_PATH,
        use_fov=USE_FOV,
        seed=SEED,
    )
    player.run()


if __name__ == "__main__":
    main()