"""
visualizer.py — Visualizador en tiempo real del NaoSoccerEnv.

Renderiza el entorno con pygame para validación visual del comportamiento
de agentes, física del balón, conos de FOV y curriculum.

Uso:
    python visualizer.py                         # Stage 0, policy aleatoria
    python visualizer.py --stage 3               # Stage 3 (2v2 scripted)
    python visualizer.py --stage 5 --no-fov      # 4v4 sin FOV
    python visualizer.py --checkpoint ruta.pt    # con policy entrenada

Controles:
    SPACE   — pausa / continúa
    R       — reiniciar episodio
    N       — siguiente stage del curriculum
    F       — toggle conos de FOV
    →       — avanzar un paso (solo en pausa)
    Q/ESC   — salir

Leyenda:
    Azul oscuro  — defender   |  Azul claro — midfielder / forward
    Rojo         — oponentes  |  Blanco     — balón
    Verde trans  — cono FOV   |  Amarillo   — walk target del agente
    Naranja      — balón en kick range
"""

from __future__ import annotations

import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pygame
    import pygame.gfxdraw
except ImportError:
    print("[visualizer] ERROR: pygame no instalado. Ejecuta: pip install pygame")
    sys.exit(1)

from config import FIELD, AGENT, FOV, CURRICULUM, CurriculumStageConfig, ROLES
from environment import (
    NaoSoccerEnv, AgentState, OpponentState, StaticObstacle, BallState, GameState,
)


# ─────────────────────────────────────────────────────────────────────────────
# PALETA DE COLORES
# ─────────────────────────────────────────────────────────────────────────────

class Color:
    # Campo
    FIELD            = ( 34, 139,  34)
    FIELD_DARK       = ( 22,  90,  22)
    LINE             = (255, 255, 255)
    GOAL             = (255, 215,   0)

    # Agentes RL por rol
    DEFENDER         = ( 30,  90, 200)
    MIDFIELDER       = ( 60, 140, 220)
    FORWARD          = ( 90, 180, 240)
    FALLEN           = (100, 100, 150)

    # Oponentes
    OPPONENT         = (200,  40,  40)
    OPP_FALLEN       = (140,  80,  80)
    OBSTACLE         = (170, 170, 170)
    OBSTACLE_EDGE    = (220, 220, 220)

    # Balón
    BALL             = (240, 240, 240)
    BALL_MOVING      = (255, 200,  50)
    BALL_IN_RANGE    = (255, 140,   0)

    # FOV (RGBA)
    FOV_FILL         = ( 80, 200,  80,  25)
    FOV_EDGE         = ( 80, 200,  80, 100)

    # HUD / texto
    TXT_WHITE        = (255, 255, 255)
    TXT_YELLOW       = (255, 215,   0)
    TXT_GREEN        = ( 80, 220,  80)
    TXT_RED          = (255,  80,  80)
    TXT_CYAN         = ( 90, 220, 255)
    TXT_ORANGE       = (255, 165,  60)
    TXT_GRAY         = (160, 160, 160)
    HUD_BG           = ( 20,  20,  20, 185)

    # Walk target
    TARGET           = (255, 255,   0)
    SHADOW           = (  0,   0,   0)


ROLE_COLOR = {
    "defender":   Color.DEFENDER,
    "midfielder": Color.MIDFIELDER,
    "forward":    Color.FORWARD,
}


# ─────────────────────────────────────────────────────────────────────────────
# SISTEMA DE COORDENADAS
# BHuman: mm, origen en el centro, +y hacia arriba.
# pygame: px, origen en la esquina superior izquierda, +y hacia abajo.
# ─────────────────────────────────────────────────────────────────────────────

class CoordTransform:
    """Convierte coordenadas mundo (mm) ↔ pantalla (px)."""

    def __init__(self, screen_w: int, screen_h: int, margin: int = 60):
        avail_w = screen_w - 2 * margin
        avail_h = screen_h - 2 * margin
        self.scale = min(avail_w / FIELD.length, avail_h / FIELD.width)
        self.cx = screen_w // 2
        self.cy = screen_h // 2

    def to_px(self, x_mm: float, y_mm: float) -> Tuple[int, int]:
        return (int(self.cx + x_mm * self.scale),
                int(self.cy - y_mm * self.scale))   # y invertido

    def mm_to_px(self, mm: float) -> int:
        return max(1, int(mm * self.scale))


# ─────────────────────────────────────────────────────────────────────────────
# RENDERER
# Cada método dibuja exactamente UNA entidad.
# ─────────────────────────────────────────────────────────────────────────────

AGENT_R_MM  = 180    # radio visual del NAO en mm
BALL_R_MM   = 105    # radio visual del balón en mm
ARROW_MM    = 340    # longitud de la flecha de orientación


class Renderer:

    def __init__(self, surf: pygame.Surface, t: CoordTransform):
        self._surf = surf
        self._t    = t

    # ── Campo ─────────────────────────────────────────────────────────────────

    def draw_field(self) -> None:
        """Rectángulo verde con líneas blancas y áreas de penalti."""
        s, t = self._surf, self._t
        s.fill(Color.FIELD)

        # Rectángulo principal del campo
        tl = t.to_px(FIELD.x_own_ground,      FIELD.y_left_sideline)
        br = t.to_px(FIELD.x_opponent_ground,  FIELD.y_right_sideline)
        rect = pygame.Rect(tl, (br[0]-tl[0], br[1]-tl[1]))
        pygame.draw.rect(s, Color.FIELD_DARK, rect)
        pygame.draw.rect(s, Color.LINE,       rect, 2)

        # Línea central
        pygame.draw.line(s, Color.LINE,
                         t.to_px(0, FIELD.y_left_sideline),
                         t.to_px(0, FIELD.y_right_sideline), 2)

        # Círculo y punto central
        center = t.to_px(0, 0)
        pygame.draw.circle(s, Color.LINE, center, t.mm_to_px(FIELD.center_circle_r), 2)
        pygame.draw.circle(s, Color.LINE, center, 4)

        # Áreas de penalti
        self._draw_box(FIELD.x_opp_penalty, FIELD.x_opponent_ground,  2000.0, -2000.0)
        self._draw_box(FIELD.x_own_ground,  -FIELD.x_opp_penalty,     2000.0, -2000.0)

        # Arcos
        self._draw_goal( FIELD.x_opponent_ground,  600.0)
        self._draw_goal( FIELD.x_own_ground,       -600.0)

    def _draw_box(self, x1: float, x2: float, y_top: float, y_bot: float) -> None:
        tl = self._t.to_px(min(x1,x2), y_top)
        br = self._t.to_px(max(x1,x2), y_bot)
        pygame.draw.rect(self._surf, Color.LINE,
                         pygame.Rect(tl, (br[0]-tl[0], br[1]-tl[1])), 1)

    def _draw_goal(self, x_line: float, depth: float) -> None:
        """Dibuja el rectángulo del arco."""
        x_back = x_line + depth
        tl = self._t.to_px(min(x_line, x_back), FIELD.y_left_goal)
        br = self._t.to_px(max(x_line, x_back), FIELD.y_right_goal)
        pygame.draw.rect(self._surf, Color.GOAL,
                         pygame.Rect(tl, (abs(br[0]-tl[0]), abs(br[1]-tl[1]))), 2)

    # ── Cono de FOV ───────────────────────────────────────────────────────────

    def draw_fov_cone(self, agent: AgentState) -> None:
        """Cono de visión del agente. Usa SRCALPHA para transparencia."""
        if not agent.is_upright:
            return

        t = self._t
        cx, cy    = t.to_px(agent.pos[0], agent.pos[1])
        radius_px = t.mm_to_px(FOV.max_distance)

        # Ángulo center en coordenadas pygame (eje y invertido)
        theta_pg  = -agent.theta             # invertir por eje y
        half_deg  = math.degrees(FOV.half_angle)
        a_start   = math.degrees(theta_pg) - half_deg
        a_end     = math.degrees(theta_pg) + half_deg

        # Construir polígono del cono
        N = 24
        pts = [(cx, cy)]
        for i in range(N + 1):
            a_rad = math.radians(a_start + (a_end - a_start) * i / N)
            pts.append((int(cx + radius_px * math.cos(a_rad)),
                        int(cy + radius_px * math.sin(a_rad))))

        fov_surf = pygame.Surface(self._surf.get_size(), pygame.SRCALPHA)
        if len(pts) > 2:
            pygame.draw.polygon(fov_surf, Color.FOV_FILL, pts)
            # Borde del cono (las dos líneas radiales)
            pygame.draw.line(fov_surf, Color.FOV_EDGE, (cx, cy), pts[1],  1)
            pygame.draw.line(fov_surf, Color.FOV_EDGE, (cx, cy), pts[-1], 1)
        self._surf.blit(fov_surf, (0, 0))

    # ── Agente RL ─────────────────────────────────────────────────────────────

    def draw_agent(self, agent: AgentState) -> None:
        """Círculo con flecha de orientación, walk target y etiqueta."""
        t  = self._t
        cx, cy = t.to_px(agent.pos[0], agent.pos[1])
        r      = t.mm_to_px(AGENT_R_MM)
        color  = Color.FALLEN if not agent.is_upright else ROLE_COLOR.get(agent.role, Color.DEFENDER)

        # Sombra
        pygame.draw.circle(self._surf, Color.SHADOW, (cx+2, cy+2), r)
        # Cuerpo
        pygame.draw.circle(self._surf, color, (cx, cy), r)
        # Contorno
        pygame.draw.circle(self._surf, Color.LINE, (cx, cy), r, 2)

        if agent.is_upright:
            # Flecha de orientación
            arr = t.mm_to_px(ARROW_MM)
            ex  = int(cx + arr * math.cos(-agent.theta))
            ey  = int(cy + arr * math.sin(-agent.theta))
            pygame.draw.line(self._surf, Color.LINE, (cx, cy), (ex, ey), 2)
            pygame.draw.circle(self._surf, Color.LINE, (ex, ey), 4)

            # Walk target (línea punteada)
            if agent.walk_target is not None:
                tx, ty = t.to_px(agent.walk_target[0], agent.walk_target[1])
                self._dashed_line(Color.TARGET, (cx, cy), (tx, ty))
                pygame.draw.circle(self._surf, Color.TARGET, (tx, ty), 4)

        # Etiqueta
        short_role = agent.role[:3].upper()
        self._label(f"{agent.agent_id[-1]}:{short_role}", cx, cy - r - 13, Color.TXT_WHITE)

    # ── Oponente ──────────────────────────────────────────────────────────────

    def draw_opponent(self, opp: OpponentState) -> None:
        """Círculo rojo con flecha de orientación."""
        t  = self._t
        cx, cy = t.to_px(opp.pos[0], opp.pos[1])
        r      = t.mm_to_px(AGENT_R_MM)
        color  = Color.OPPONENT if opp.is_upright else Color.OPP_FALLEN

        pygame.draw.circle(self._surf, Color.SHADOW, (cx+2, cy+2), r)
        pygame.draw.circle(self._surf, color, (cx, cy), r)
        pygame.draw.circle(self._surf, Color.LINE, (cx, cy), r, 2)

        if opp.is_upright:
            arr = t.mm_to_px(ARROW_MM)
            ex  = int(cx + arr * math.cos(-opp.theta))
            ey  = int(cy + arr * math.sin(-opp.theta))
            pygame.draw.line(self._surf, Color.LINE, (cx, cy), (ex, ey), 2)

        self._label("OPP", cx, cy - r - 13, Color.TXT_RED)

    def draw_static_obstacle(self, obstacle: StaticObstacle) -> None:
        t = self._t
        cx, cy = t.to_px(obstacle.pos[0], obstacle.pos[1])
        r = max(6, t.mm_to_px(obstacle.radius))
        pygame.draw.circle(self._surf, Color.SHADOW, (cx + 2, cy + 2), r)
        pygame.draw.circle(self._surf, Color.OBSTACLE, (cx, cy), r)
        pygame.draw.circle(self._surf, Color.OBSTACLE_EDGE, (cx, cy), r, 2)
        self._label("OBS", cx, cy - r - 13, Color.TXT_GRAY, size=11)

    # ── Balón ─────────────────────────────────────────────────────────────────

    def draw_ball(self, ball: BallState, nearest_agent_dist: float) -> None:
        """Círculo blanco. Cambia de color según velocidad y kick range."""
        t  = self._t
        cx, cy = t.to_px(ball.pos[0], ball.pos[1])
        r      = max(5, t.mm_to_px(BALL_R_MM))
        speed  = float(np.linalg.norm(ball.vel))

        interaction_range = getattr(
            AGENT,
            "ball_interaction_range",
            getattr(AGENT, "ball_kick_range", 500.0),
        )

        if nearest_agent_dist < interaction_range:
            color = Color.BALL_IN_RANGE
        elif speed > 300:
            color = Color.BALL_MOVING
        else:
            color = Color.BALL

        pygame.draw.circle(self._surf, Color.SHADOW, (cx+2, cy+2), r)
        pygame.draw.circle(self._surf, color, (cx, cy), r)

        # Línea de velocidad
        if speed > 80:
            vn = ball.vel / speed
            vlen = min(speed / 600.0, 1.5) * t.mm_to_px(700)
            ex = int(cx + vlen * vn[0])
            ey = int(cy - vlen * vn[1])
            pygame.draw.line(self._surf, Color.BALL_MOVING, (cx, cy), (ex, ey), 3)

    # ── HUD ───────────────────────────────────────────────────────────────────

    def draw_hud(self,
                  game:         GameState,
                  stage:        CurriculumStageConfig,
                  episode:      int,
                  ep_reward:    float,
                  show_fov:     bool,
                  paused:       bool,
                  step_rewards: Dict[str, float],
                  control_debug: Optional[Dict[str, str]] = None) -> None:
        """Dibuja marcador, estadísticas y controles."""
        f_lg = pygame.font.SysFont("monospace", 18, bold=True)
        f_md = pygame.font.SysFont("monospace", 14)
        f_sm = pygame.font.SysFont("monospace", 11)

        # Marcador (top-center)
        score = f"  {game.own_score}  :  {game.opp_score}  "
        s_surf = f_lg.render(score, True, Color.TXT_YELLOW)
        sw = self._surf.get_width()
        self._surf.blit(s_surf, (sw // 2 - s_surf.get_width() // 2, 6))

        # Panel izquierdo — stats del episodio
        left = [
            (f"[Stage {stage.stage_id}] {stage.name}",    Color.TXT_YELLOW),
            (f"Episode:   {episode}",                       Color.TXT_WHITE),
            (f"Step:      {game.step_count}/{stage.episode_steps}", Color.TXT_WHITE),
            (f"Time:      {game.time_elapsed:.1f}s",        Color.TXT_WHITE),
            (f"Ep Reward: {ep_reward:+.3f}",
             Color.TXT_GREEN if ep_reward >= 0 else Color.TXT_RED),
            (f"FOV:       {'ON' if show_fov else 'OFF'}",
             Color.TXT_GREEN if show_fov else Color.TXT_GRAY),
            (f"Shaping:   {stage.shaping_weight:.2f}",      Color.TXT_WHITE),
            (f"Theta free:{stage.theta_free}",               Color.TXT_WHITE),
            (f"Obstacles: {stage.num_static_obstacles}",     Color.TXT_WHITE),
            ("",                                             Color.TXT_WHITE),
            (f"{'  ⏸  PAUSED  ' if paused else '  ▶  RUNNING'}",
             Color.TXT_YELLOW if paused else Color.TXT_GREEN),
        ]
        if control_debug:
            owner_color = {
                "green": Color.TXT_GREEN,
                "orange": Color.TXT_ORANGE,
                "cyan": Color.TXT_CYAN,
            }.get(control_debug.get("color", "green"), Color.TXT_WHITE)
            left.extend([
                ("", Color.TXT_WHITE),
                (f"Control:   {control_debug.get('owner', 'RL')}", owner_color),
                (f"Actor:     {control_debug.get('active_agent', '-')}", Color.TXT_WHITE),
                (f"Macro:     {control_debug.get('macro_type', 'none')}", Color.TXT_WHITE),
                (f"End:       {control_debug.get('macro_end_reason', 'none')}", Color.TXT_WHITE),
                (f"BHuman:    {control_debug.get('skill_request', 'none')}", Color.TXT_WHITE),
                (f"PB entry:  {control_debug.get('playball_entries', 0)}", Color.TXT_WHITE),
                (f"RBW rate:  {control_debug.get('ready_walk_rate', 0.0):.2f}", Color.TXT_WHITE),
                (f"RTBW rate: {control_debug.get('ready_touch_walk_rate', 0.0):.2f}", Color.TXT_WHITE),
                (f"TO rate:   {control_debug.get('takeover_timeout_rate', 0.0):.2f}", Color.TXT_WHITE),
                (f"CLR rate:  {control_debug.get('takeover_skill_cleared_rate', 0.0):.2f}", Color.TXT_WHITE),
            ])
        self._panel(left, x=8, y=8, font=f_md)

        # Panel derecho — reward por agente en este paso
        if step_rewards:
            right = [("Step rewards:", Color.TXT_YELLOW)]
            for aid, r in sorted(step_rewards.items()):
                c = Color.TXT_GREEN if r >= 0 else Color.TXT_RED
                right.append((f"  {aid}: {r:+.4f}", c))
            self._panel(right, x=sw - 215, y=8, font=f_md)

        # Controles (parte inferior)
        ctrl = "SPACE:pausa  R:reset  N:sig.stage  F:FOV  →:step  Q:salir"
        c_s  = f_sm.render(ctrl, True, Color.TXT_GRAY)
        self._surf.blit(c_s, (8, self._surf.get_height() - 18))

    # ── Utilidades ────────────────────────────────────────────────────────────

    def _panel(self, lines: List[Tuple[str, tuple]],
               x: int, y: int, font: pygame.font.Font,
               pad: int = 7) -> None:
        """Panel con fondo semitransparente."""
        if not lines:
            return
        lh = font.get_height() + 3
        pw = max((font.size(t)[0] for t, _ in lines), default=10) + pad * 2
        ph = len(lines) * lh + pad * 2
        bg = pygame.Surface((pw, ph), pygame.SRCALPHA)
        bg.fill(Color.HUD_BG)
        self._surf.blit(bg, (x, y))
        for i, (text, color) in enumerate(lines):
            self._surf.blit(font.render(text, True, color),
                            (x + pad, y + pad + i * lh))

    def _label(self, text: str, cx: int, y: int,
               color: tuple, size: int = 13) -> None:
        font = pygame.font.SysFont("monospace", size, bold=True)
        s = font.render(text, True, color)
        sh = font.render(text, True, Color.SHADOW)
        x  = cx - s.get_width() // 2
        self._surf.blit(sh, (x + 1, y + 1))
        self._surf.blit(s,  (x,     y))

    def _dashed_line(self, color: tuple,
                      p0: Tuple[int,int], p1: Tuple[int,int],
                      dash: int = 7) -> None:
        dx, dy = p1[0]-p0[0], p1[1]-p0[1]
        d = math.hypot(dx, dy)
        if d < 1:
            return
        steps = int(d / (dash * 2))
        for i in range(steps):
            t0, t1 = (2*i*dash)/d, min(((2*i+1)*dash)/d, 1.0)
            a = (int(p0[0]+t0*dx), int(p0[1]+t0*dy))
            b = (int(p0[0]+t1*dx), int(p0[1]+t1*dy))
            pygame.draw.line(self._surf, color, a, b, 1)


# ─────────────────────────────────────────────────────────────────────────────
# LECTOR DE ESTADO INTERNO DEL ENTORNO
# ─────────────────────────────────────────────────────────────────────────────

class EnvStateReader:
    """Acceso centralizado al estado interno del NaoSoccerEnv."""

    def __init__(self, env: NaoSoccerEnv):
        self._env = env

    @property
    def agents(self) -> Dict[str, AgentState]:
        return self._env._agents_state

    @property
    def opponents(self) -> List[OpponentState]:
        return self._env._opponents_state

    @property
    def static_obstacles(self) -> List[StaticObstacle]:
        return self._env._static_obstacles

    @property
    def ball(self) -> BallState:
        return self._env._ball

    @property
    def game(self) -> GameState:
        return self._env._game

    def sample_info(self, infos: Optional[Dict[str, Dict]] = None) -> Dict:
        """Devuelve un info representativo del step actual."""
        infos = infos or {}
        return next(iter(infos.values()), {}) if infos else {}

    def control_debug(self, infos: Optional[Dict[str, Dict]] = None) -> Dict[str, object]:
        """Resume si manda RL, handoff o BHuman takeover."""
        info = self.sample_info(infos)
        takeover_active = bool(info.get("takeover_active", False))
        handoff_ready = bool(info.get("handoff_ready", False))

        if takeover_active:
            owner = "BHUMAN"
            color = "orange"
        elif handoff_ready:
            owner = "HANDOFF"
            color = "cyan"
        else:
            owner = "RL"
            color = "green"

        return {
            "owner": owner,
            "color": color,
            "active_agent": str(info.get("active_agent_id", "-")),
            "macro_type": str(info.get("macro_type", "none")),
            "macro_end_reason": str(info.get("last_macro_end_reason", "none")),
            "skill_request": str(info.get("sidecar_skill_request", "none")),
            "playball_entries": int(info.get("episode_playball_entry_count", 0)),
            "ready_walk_rate": float(info.get("episode_ready_but_walk_rate", 0.0)),
            "ready_touch_walk_rate": float(
                info.get("episode_ready_touch_but_walk_rate", 0.0)
            ),
            "takeover_timeout_rate": float(
                info.get("episode_takeover_timeout_rate", 0.0)
            ),
            "takeover_skill_cleared_rate": float(
                info.get("episode_takeover_skill_cleared_rate", 0.0)
            ),
        }

    def nearest_agent_dist(self) -> float:
        """Distancia del agente RL más cercano al balón."""
        if not self._env._agents_state:
            return float("inf")
        ball = self._env._ball.pos
        dists = [float(np.linalg.norm(a.pos - ball))
                 for a in self._env._agents_state.values()
                 if a.is_upright]
        return min(dists) if dists else float("inf")


# ─────────────────────────────────────────────────────────────────────────────
# POLICY ALEATORIA
# ─────────────────────────────────────────────────────────────────────────────

class RandomPolicy:
    """Acciones uniformes en [-1, 1] — baseline sin entrenamiento."""

    def act(self, agents: List[str]) -> Dict[str, np.ndarray]:
        return {aid: np.random.uniform(-1, 1, 3).astype(np.float32)
                for aid in agents}


def load_policy(checkpoint_path: str):
    """
    Carga la policy desde un checkpoint de skrl MAPPO.

    skrl guarda con agent.save() un dict con estructura variable según versión:
      Formato A: {"policy": state_dict, "value": ..., "optimizer": ...}
      Formato B: {"policy": {"policy": state_dict, ...}, ...}  (anidado)
      Formato C: state_dict plano (torch.save directo)

    Los keys del state_dict de PolicyNetwork son:
      "log_std_param", "net.0.weight", "net.0.bias", "net.2.weight", ...

    Esta función navega la estructura hasta encontrar esos keys.
    """
    try:
        import torch
        import gymnasium as gym
        from config import OBS, ACTION
        from train import PolicyNetwork

        obs_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                   shape=(OBS.total,), dtype=np.float32)
        act_space = gym.spaces.Box(low=-1.0, high=1.0,
                                   shape=(ACTION.size,), dtype=np.float32)

        policy = PolicyNetwork(obs_space, act_space, "cpu")
        ckpt   = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Estrategia: probar múltiples niveles de anidamiento
        def _has_net_keys(d: dict) -> bool:
            return isinstance(d, dict) and ("net.0.weight" in d or "log_std_param" in d)

        def _strip_prefix(d: dict) -> dict:
            """Quita prefijos comunes de skrl (policy., module., etc.)."""
            stripped = {}
            for k, v in d.items():
                new_k = k
                for prefix in ["policy.", "module.", "net_container."]:
                    if k.startswith(prefix):
                        new_k = k[len(prefix):]
                        break
                stripped[new_k] = v
            return stripped

        # Candidatos en orden de prioridad
        candidates = []

        if isinstance(ckpt, dict):
            # Nivel 0: checkpoint directo
            candidates.append(ckpt)
            # Nivel 1: ckpt["policy"]
            if "policy" in ckpt and isinstance(ckpt["policy"], dict):
                candidates.append(ckpt["policy"])
                # Nivel 2: ckpt["policy"]["policy"] (anidado)
                if "policy" in ckpt["policy"] and isinstance(ckpt["policy"]["policy"], dict):
                    candidates.append(ckpt["policy"]["policy"])

        for candidate in candidates:
            if _has_net_keys(candidate):
                policy.load_state_dict(candidate)
                policy.eval()
                print(f"[visualizer] Policy cargada: {checkpoint_path}")
                return policy
            # Probar con prefijos removidos
            stripped = _strip_prefix(candidate)
            if _has_net_keys(stripped):
                policy.load_state_dict(stripped)
                policy.eval()
                print(f"[visualizer] Policy cargada (prefijos removidos): {checkpoint_path}")
                return policy

        # Último recurso: listar keys para diagnóstico
        if isinstance(ckpt, dict):
            top_keys = list(ckpt.keys())[:10]
            if "policy" in ckpt and isinstance(ckpt["policy"], dict):
                inner_keys = list(ckpt["policy"].keys())[:10]
                print(f"[visualizer] Checkpoint keys: top={top_keys}, policy={inner_keys}")
            else:
                print(f"[visualizer] Checkpoint keys: {top_keys}")
        print(f"[visualizer] No se encontró state_dict compatible en {checkpoint_path}")
        return None

    except Exception as e:
        print(f"[visualizer] No se pudo cargar policy: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# APLICACIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

class SoccerVisualizer:
    """
    Aplicación pygame que renderiza NaoSoccerEnv en tiempo real.

    Loop principal:
        eventos → simulación → render → flip

    El render ocurre siempre (incluso en pausa) para que el HUD quede vivo.
    La simulación solo avanza cuando no está en pausa.
    """

    W, H = 1200, 700   # resolución de la ventana
    FPS  = 30

    def __init__(self, stage_idx: int, use_fov: bool,
                 checkpoint: Optional[str], seed: int):

        pygame.init()
        pygame.display.set_caption("NAO Soccer Visualizer — MAPPO Fase 1")
        self._screen  = pygame.display.set_mode((self.W, self.H))
        self._clock   = pygame.time.Clock()

        self._running  = True
        self._paused   = False
        self._show_fov = use_fov
        self._stage_id = stage_idx
        self._episode  = 0
        self._ep_reward  = 0.0
        self._step_rews: Dict[str, float] = {}

        # Entorno y lector
        self._env    = NaoSoccerEnv(CURRICULUM[stage_idx], use_fov=use_fov, seed=seed)
        self._reader = EnvStateReader(self._env)

        # Renderer
        t = CoordTransform(self.W, self.H)
        self._renderer = Renderer(self._screen, t)

        # Policies
        self._trained = load_policy(checkpoint) if checkpoint else None
        self._random  = RandomPolicy()

        self._obs, self._infos = self._env.reset(seed=seed)

    # ── Loop ──────────────────────────────────────────────────────────────────

    def run(self) -> None:
        while self._running:
            self._events()
            if not self._paused:
                self._step()
            self._render()
            self._clock.tick(self.FPS)
        pygame.quit()

    # ── Simulación ────────────────────────────────────────────────────────────

    def _step(self) -> None:
        if not self._env.agents:
            self._new_episode()
            return

        actions = self._actions()
        next_obs, rews, terms, truncs, infos = self._env.step(actions)

        self._obs       = next_obs
        self._infos     = infos
        self._step_rews = {aid: float(r) for aid, r in rews.items()}
        self._ep_reward += sum(rews.values())

        if truncs.get("__all__") or terms.get("__all__"):
            self._new_episode()

    def _actions(self) -> Dict[str, np.ndarray]:
        active = list(self._env.agents)
        if self._trained is None:
            return self._random.act(active)
        try:
            import torch
            acts = {}
            with torch.no_grad():
                for aid in active:
                    obs_t = torch.FloatTensor(
                        self._obs.get(aid, np.zeros(68, dtype=np.float32))
                    ).unsqueeze(0)
                    out      = self._trained.act({"observations": obs_t}, role="policy")
                    acts[aid] = out[0].squeeze(0).numpy()
            return acts
        except Exception as e:
            print(f"[visualizer] Error en policy: {e}")
            self._trained = None
            return self._random.act(active)

    def _new_episode(self) -> None:
        self._episode  += 1
        self._ep_reward = 0.0
        self._step_rews = {}
        self._obs, self._infos = self._env.reset()

    # ── Render ────────────────────────────────────────────────────────────────

    def _render(self) -> None:
        r = self._renderer
        control_debug = self._reader.control_debug(self._infos)

        # 1. Campo
        r.draw_field()

        # 2. Conos de FOV (debajo de los agentes)
        if self._show_fov:
            for agent in self._reader.agents.values():
                r.draw_fov_cone(agent)

        # 3. Oponentes
        for obstacle in self._reader.static_obstacles:
            r.draw_static_obstacle(obstacle)

        # 4. Oponentes
        for opp in self._reader.opponents:
            r.draw_opponent(opp)

        # 5. Agentes RL
        for agent in self._reader.agents.values():
            r.draw_agent(agent)

        # 6. Balón
        r.draw_ball(self._reader.ball, self._reader.nearest_agent_dist())

        # 7. HUD
        r.draw_hud(
            game         = self._reader.game,
            stage        = CURRICULUM[self._stage_id],
            episode      = self._episode,
            ep_reward    = self._ep_reward,
            show_fov     = self._show_fov,
            paused       = self._paused,
            step_rewards = self._step_rews,
            control_debug= control_debug,
        )

        pygame.display.flip()

    # ── Teclado ───────────────────────────────────────────────────────────────

    def _events(self) -> None:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self._running = False
            elif ev.type == pygame.KEYDOWN:
                self._key(ev.key)

    def _key(self, key: int) -> None:
        if key in (pygame.K_ESCAPE, pygame.K_q):
            self._running = False

        elif key == pygame.K_SPACE:
            self._paused = not self._paused

        elif key == pygame.K_r:
            self._new_episode()

        elif key == pygame.K_f:
            self._show_fov = not self._show_fov
            self._env._use_fov = self._show_fov
            self._env._fov_filter._use_fov = self._show_fov
            print(f"[visualizer] FOV → {'ON' if self._show_fov else 'OFF'}")

        elif key == pygame.K_n:
            new = min(self._stage_id + 1, len(CURRICULUM) - 1)
            if new != self._stage_id:
                self._stage_id = new
                self._env.set_stage(CURRICULUM[new])
                self._new_episode()
                print(f"[visualizer] Stage → {CURRICULUM[new].name}")

        elif key == pygame.K_RIGHT and self._paused:
            self._step()


def main() -> None:
    # Configuración rápida: solo editar estos valores y presionar Play.
    STAGE_IDX  = 0
    USE_FOV    = True
    SEED       = 42
    MODEL_PATH = os.environ.get("RL_MODEL_PATH", "C:\ALEJO\runs\practical_stage0_fovTrue_1775100472").strip()
    USE_POLICY = bool(MODEL_PATH)

    checkpoint = MODEL_PATH if USE_POLICY else None

    n = len(CURRICULUM)
    if not (0 <= STAGE_IDX < n):
        print(f"[visualizer] Stage inválido: {STAGE_IDX}. Rango válido: 0–{n-1}")
        sys.exit(1)

    if USE_POLICY and not MODEL_PATH:
        print("[visualizer] ERROR: USE_POLICY=True requiere una ruta válida en MODEL_PATH")
        sys.exit(1)

    print(f"[visualizer] Stage:  [{STAGE_IDX}] {CURRICULUM[STAGE_IDX].name}")
    print(f"[visualizer] FOV:    {'ON' if USE_FOV else 'OFF'}")
    print(f"[visualizer] Mode:   {'policy' if USE_POLICY else 'random'}")
    print(f"[visualizer] Policy: {checkpoint if checkpoint else 'aleatoria'}")
    print()

    SoccerVisualizer(
        stage_idx  = STAGE_IDX,
        use_fov    = USE_FOV,
        checkpoint = checkpoint,
        seed       = SEED,
    ).run()


if __name__ == "__main__":
    main()
