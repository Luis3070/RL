"""
game.py — GameController: control humano del agente vía pygame.

Diagnóstico: si el humano puede driblear/hacer goles usando el mismo
pipeline de acciones que A*, el problema está en A*. Si el humano
también sufre comportamientos erráticos, el problema es el environment.

Ejecución directa:
    python game.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

# ── Path setup (permite darle Run directamente) ──────────────────────────
if __package__ in (None, ""):
    _here = Path(__file__).resolve().parent           # gamecontroller/
    _tg   = _here.parent                               # trajectory_guidance/
    _proj = _tg.parent                                  # trajectory_guidance_project/
    _ws   = _proj.parent                                # RL/

    sys.path = [p for p in sys.path if Path(p).resolve() != _here]
    sys.path.insert(0, str(_ws))
    sys.path.insert(1, str(_proj))
    sys.path.insert(2, str(_here))

import numpy as np
import pygame

from config import CURRICULUM, FIELD, AGENT
from visualizer import CoordTransform, Renderer

from env_bridge import GameEnvBridge
from controller import HumanController


def main():
    # ═══════════════════════════════════════════════════════════════════════
    # CONFIG — editar aquí y darle Run
    # ═══════════════════════════════════════════════════════════════════════
    INITIAL_STAGE = 3      # stage inicial (0-3)
    USE_FOV       = True
    SEED          = 42
    W, H          = 1200, 700
    FPS           = 10      # 10 = tiempo real (dt=0.1s por step)
    # ═══════════════════════════════════════════════════════════════════════

    pygame.init()
    pygame.display.set_caption("GameController — Human Agent Control")
    screen = pygame.display.set_mode((W, H))
    clock  = pygame.time.Clock()

    coord    = CoordTransform(W, H)
    renderer = Renderer(screen, coord)
    bridge   = GameEnvBridge(use_fov=USE_FOV, seed=SEED)
    ctrl     = HumanController()

    # ── Reverse coord transform (px → mm) ────────────────────────────────
    def px_to_mm(px_x: int, px_y: int) -> Tuple[float, float]:
        x_mm = (px_x - coord.cx) / coord.scale
        y_mm = -(px_y - coord.cy) / coord.scale
        return x_mm, y_mm

    # ── Estado de juego ──────────────────────────────────────────────────
    stage_id: int    = INITIAL_STAGE
    episode:  int    = 0
    ep_reward: float = 0.0
    step_count: int  = 0
    step_rewards: Dict[str, float] = {}
    trace: List[Tuple[float, float]] = []
    running:  bool = True
    paused:   bool = False
    show_fov: bool = USE_FOV

    def reset_episode(new_stage: int | None = None, seed_val=None):
        nonlocal stage_id, episode, ep_reward, step_count, step_rewards, trace
        if new_stage is not None:
            stage_id = new_stage
        episode   += 1
        ep_reward  = 0.0
        step_count = 0
        step_rewards = {}
        trace = []
        ctrl.target_mm = None
        bridge.reset(stage_id, seed=seed_val)

    reset_episode(stage_id, SEED)

    # ── Game Loop ────────────────────────────────────────────────────────
    while running:
        # 1. Eventos
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                mx, my = px_to_mm(*ev.pos)
                ctrl.set_move_target(mx, my)

            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

                elif ev.key == pygame.K_SPACE:
                    ctrl.set_shoot(bridge.get_ball_pos())

                elif ev.key == pygame.K_p:
                    mx, my = px_to_mm(*pygame.mouse.get_pos())
                    ctrl.set_pass(
                        bridge.get_ball_pos(),
                        np.array([mx, my], dtype=np.float32),
                    )

                elif ev.key == pygame.K_r:
                    reset_episode()

                elif ev.key == pygame.K_n:
                    reset_episode(min(stage_id + 1, 3))

                elif ev.key in (pygame.K_0, pygame.K_1, pygame.K_2, pygame.K_3):
                    reset_episode(ev.key - pygame.K_0)

                elif ev.key == pygame.K_f:
                    show_fov = not show_fov
                    bridge.env._env._use_fov = show_fov
                    bridge.env._env._fov_filter._use_fov = show_fov

                elif ev.key == pygame.K_F1:
                    paused = not paused

        # 2. Step del environment
        if not paused and bridge.has_agents:
            agent_pos = bridge.get_agent_pos()
            max_disp  = bridge.get_max_displacement()
            action    = ctrl.compute_action(agent_pos, max_disp)

            obs, rewards, terms, truncs, infos = bridge.step(action)
            step_count += 1
            step_rewards = {aid: float(r) for aid, r in rewards.items()}
            ep_reward   += float(sum(rewards.values()))

            new_pos = bridge.get_agent_pos()
            trace.append((float(new_pos[0]), float(new_pos[1])))

            if terms.get("__all__") or truncs.get("__all__"):
                reset_episode()

        # 3. Render
        renderer.draw_field()

        reader = bridge.reader
        if show_fov:
            for agent in reader.agents.values():
                renderer.draw_fov_cone(agent)

        for opp in reader.opponents:
            renderer.draw_opponent(opp)

        for agent in reader.agents.values():
            renderer.draw_agent(agent)

        # Trace del agente (amarillo)
        if len(trace) >= 2:
            pts = [coord.to_px(x, y) for x, y in trace]
            pygame.draw.lines(screen, (255, 255, 0), False, pts, 2)

        # Target indicator
        if ctrl.target_mm is not None:
            tx, ty = coord.to_px(
                float(ctrl.target_mm[0]), float(ctrl.target_mm[1])
            )
            if ctrl.mode == "shoot":
                color = (255, 50, 50)
            elif ctrl.mode == "pass":
                color = (50, 150, 255)
            else:
                color = (50, 255, 50)
            pygame.draw.circle(screen, color, (tx, ty), 8, 2)
            pygame.draw.line(screen, color, (tx - 6, ty), (tx + 6, ty), 1)
            pygame.draw.line(screen, color, (tx, ty - 6), (tx, ty + 6), 1)

        # Kick range indicator (naranja si en rango, amarillo tenue si cerca)
        agent_pos = bridge.get_agent_pos()
        ball_pos  = bridge.get_ball_pos()
        dist_ball = float(np.linalg.norm(agent_pos - ball_pos))
        if dist_ball < AGENT.ball_kick_range * 1.5:
            apx   = coord.to_px(float(agent_pos[0]), float(agent_pos[1]))
            kr_px = coord.mm_to_px(AGENT.ball_kick_range)
            surf  = pygame.Surface((kr_px * 2, kr_px * 2), pygame.SRCALPHA)
            c = (255, 140, 0, 40) if dist_ball < AGENT.ball_kick_range \
                else (255, 255, 0, 20)
            pygame.draw.circle(surf, c, (kr_px, kr_px), kr_px)
            screen.blit(surf, (apx[0] - kr_px, apx[1] - kr_px))

        renderer.draw_ball(reader.ball, reader.nearest_agent_dist())

        # HUD del visualizer
        renderer.draw_hud(
            game=reader.game,
            stage=CURRICULUM[stage_id],
            episode=episode,
            ep_reward=ep_reward,
            show_fov=show_fov,
            paused=paused,
            step_rewards=step_rewards,
        )

        # Overlay del gamecontroller
        font = pygame.font.SysFont("monospace", 14)
        mode_str = ctrl.mode.upper()
        lines = [
            f"GAMECONTROLLER | stage: {stage_id} ({CURRICULUM[stage_id].name})",
            f"episode: {episode} | step: {step_count} | reward: {ep_reward:.2f}",
            f"mode: {mode_str} | dist_ball: {dist_ball:.0f}mm",
            f"agent: ({agent_pos[0]:.0f}, {agent_pos[1]:.0f})"
            f" | ball: ({ball_pos[0]:.0f}, {ball_pos[1]:.0f})",
            "CLICK=move  SPACE=shoot  P=pass  R=reset  N=next  0-3=stage"
            "  F=fov  F1=pause  Q=quit",
        ]
        y0 = H - (len(lines) * 18 + 10)
        bg = pygame.Surface((W - 16, len(lines) * 18 + 8), pygame.SRCALPHA)
        bg.fill((20, 20, 20, 170))
        screen.blit(bg, (8, y0 - 4))
        for i, text in enumerate(lines):
            tc = (240, 240, 240)
            if i == 2:
                tc = {
                    "SHOOT": (255, 80, 80),
                    "PASS":  (80, 150, 255),
                }.get(mode_str, (240, 240, 240))
            screen.blit(font.render(text, True, tc), (14, y0 + i * 18))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
