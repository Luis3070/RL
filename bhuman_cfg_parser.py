"""
bhuman_cfg_parser.py — Parser de archivos .cfg de BHuman y contract loader.

Fuente de verdad: Documentacion/PLAN/BHUMANPARSER_GUIDE.md

Arquitectura:
  1. BHumanCfgParser: tokeniza y parsea archivos .cfg en dicts/listas Python
  2. Dataclasses contractuales: tipadas, inmutables, con unidades documentadas
  3. ContractLoader: produce AllConfigs desde CFGs reales o desde defaults embebidos
  4. AllConfigs: estructura runtime consumible por env, reward, wrapper

Unidades: mm, mm/s, deg (en cfg) → rad donde se use en runtime, ms, frames.

Rutas reales (bajo cfg_root = Config/):
  Locations/4v4_Scaled/fieldDimensions.cfg
  Behavior/Strategies/s4v4_scaled.cfg
  Behavior/Tactics/t121_4v4_scaled.cfg
  Behavior/Tactics/t13_4v4_scaled.cfg
  Scenarios/4v4_Scaled/BehaviorControl/kickAtGoalImpl.cfg
  Scenarios/4v4_Scaled/BehaviorControl/passToTeammateImpl.cfg
  Scenarios/4v4_Scaled/BehaviorControl/zweikampfImpl.cfg
  Scenarios/4v4_Scaled/kickInfo.cfg
  Scenarios/4v4_Scaled/skillBehaviorControl.cfg
  Scenarios/4v4_Scaled/strategyBehaviorControl.cfg
  Scenarios/4v4_Scaled/behaviorParameters.cfg
  Scenarios/4v4_Scaled/libWalkProvider.cfg
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


# ─────────────────────────────────────────────────────────────────────────────
# 1. FIELD SYMBOLS — 4v4 Scaled (fuente: fieldDimensions.cfg)
# ─────────────────────────────────────────────────────────────────────────────

FIELD_SYMBOLS_4V4_SCALED: Dict[str, float] = {
    # X axis (field length 5626mm)
    "xPosOpponentFieldBorder":  3313.0,
    "xPosOpponentGoal":         3368.0,
    "xPosOpponentGoalPost":     2838.0,
    "xPosOpponentGroundLine":   2813.0,
    "xPosOpponentGoalArea":     2438.0,
    "xPosOpponentPenaltyMark":  2000.0,
    "xPosOpponentPenaltyArea":  1781.0,
    "xPosPenaltyStrikerStartPosition": 1781.0,
    "xPosHalfWayLine":          0.0,
    "xPosOwnPenaltyArea":      -1781.0,
    "xPosOwnPenaltyMark":      -2000.0,
    "xPosOwnGoalArea":         -2438.0,
    "xPosOwnGroundLine":       -2813.0,
    "xPosOwnGoalPost":         -2838.0,
    "xPosOwnGoal":             -3368.0,
    "xPosOwnFieldBorder":      -3313.0,
    # Y axis (field width 3750mm)
    "yPosLeftFieldBorder":      1960.0,
    "yPosLeftSideline":         1875.0,
    "yPosLeftPenaltyArea":      1250.0,
    "yPosLeftGoalArea":         688.0,
    "yPosLeftGoal":             800.0,
    "yPosRightGoal":           -800.0,
    "yPosRightGoalArea":       -688.0,
    "yPosRightPenaltyArea":   -1250.0,
    "yPosRightSideline":      -1875.0,
    "yPosRightFieldBorder":   -1960.0,
    # Others
    "centerCircleRadius":       469.0,
    "fieldLinesWidth":          50.0,
    "goalPostRadius":           50.0,
    "penaltyMarkSize":         100.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. CFG PARSER — Tokeniza y parsea archivos .cfg de BHuman
# ─────────────────────────────────────────────────────────────────────────────

class BHumanCfgParser:
    """
    Parser para archivos .cfg de BHuman (formato InExprMapFile).

    Soporta:
      - Comentarios: //, /* */, # (strategy files)
      - Tipos: int, float (con sufijo deg), bool (true/false), string
      - Estructuras: { key = value; }
      - Arrays: [ value, value, ... ]
      - Expresiones aritméticas con FIELD_SYMBOLS
      - Unary minus, symbol-to-symbol references
    """

    def __init__(self, symbols: Optional[Dict[str, float]] = None):
        self._symbols = symbols or FIELD_SYMBOLS_4V4_SCALED.copy()
        self._text = ""
        self._pos = 0

    def parse_file(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """Parsea un archivo .cfg y retorna un dict."""
        text = Path(filepath).read_text(encoding="utf-8", errors="replace")
        return self.parse_string(text)

    def parse_string(self, text: str) -> Dict[str, Any]:
        """Parsea un string .cfg y retorna un dict."""
        self._text = text
        self._pos = 0
        return self._parse_map()

    # ── Internal ──────────────────────────────────────────────────────────

    def _peek(self) -> str:
        if self._pos >= len(self._text):
            return ""
        return self._text[self._pos]

    def _advance(self, n: int = 1) -> str:
        result = self._text[self._pos:self._pos + n]
        self._pos += n
        return result

    def _skip_ws_and_comments(self) -> None:
        while self._pos < len(self._text):
            c = self._text[self._pos]
            # Whitespace
            if c in " \t\n\r":
                self._pos += 1
                continue
            # Line comment //
            if c == "/" and self._pos + 1 < len(self._text) and self._text[self._pos + 1] == "/":
                while self._pos < len(self._text) and self._text[self._pos] != "\n":
                    self._pos += 1
                continue
            # Block comment /* */
            if c == "/" and self._pos + 1 < len(self._text) and self._text[self._pos + 1] == "*":
                self._pos += 2
                while self._pos + 1 < len(self._text):
                    if self._text[self._pos] == "*" and self._text[self._pos + 1] == "/":
                        self._pos += 2
                        break
                    self._pos += 1
                continue
            # Hash comment (strategy files)
            if c == "#":
                while self._pos < len(self._text) and self._text[self._pos] != "\n":
                    self._pos += 1
                continue
            break

    def _parse_map(self) -> Dict[str, Any]:
        """Parsea un mapa de key = value; pares (top level o dentro de {})."""
        result: Dict[str, Any] = {}
        while True:
            self._skip_ws_and_comments()
            if self._pos >= len(self._text):
                break
            if self._peek() == "}":
                break
            # Parse key
            key = self._parse_identifier()
            if not key:
                break
            self._skip_ws_and_comments()
            # Expect =
            if self._peek() == "=":
                self._advance()
            self._skip_ws_and_comments()
            # Parse value
            value = self._parse_value()
            result[key] = value
            self._skip_ws_and_comments()
            # Optional ;
            if self._peek() == ";":
                self._advance()
        return result

    def _parse_identifier(self) -> str:
        self._skip_ws_and_comments()
        start = self._pos
        while self._pos < len(self._text) and (
            self._text[self._pos].isalnum() or self._text[self._pos] in "_."
        ):
            self._pos += 1
        return self._text[start:self._pos]

    def _parse_value(self) -> Any:
        self._skip_ws_and_comments()
        c = self._peek()

        if c == "{":
            return self._parse_struct()
        if c == "[":
            return self._parse_array()
        if c == '"':
            return self._parse_string_literal()

        # Try boolean
        rest = self._text[self._pos:self._pos + 5]
        if rest.startswith("true"):
            if self._pos + 4 >= len(self._text) or not self._text[self._pos + 4].isalnum():
                self._pos += 4
                return True
        if rest.startswith("false"):
            if self._pos + 5 >= len(self._text) or not self._text[self._pos + 5].isalnum():
                self._pos += 5
                return False

        # Expression (number, symbol, arithmetic)
        return self._parse_expression()

    def _parse_struct(self) -> Any:
        """Parsea { ... }. Puede ser un struct (key=value) o un array de structs."""
        self._advance()  # skip {
        self._skip_ws_and_comments()

        # Check if it's key=value or just values (array of structs)
        saved_pos = self._pos
        has_key = False
        depth = 0
        while saved_pos < len(self._text):
            ch = self._text[saved_pos]
            if ch in "{}[]":
                if ch in "{[":
                    depth += 1
                else:
                    depth -= 1
                if depth < 0:
                    break
            if depth == 0 and ch == "=":
                has_key = True
                break
            if depth == 0 and ch in ";,}":
                break
            saved_pos += 1

        if has_key:
            result = self._parse_map()
        else:
            # Array of values/structs inside {}
            items = []
            while True:
                self._skip_ws_and_comments()
                if self._peek() == "}":
                    break
                items.append(self._parse_value())
                self._skip_ws_and_comments()
                if self._peek() in ",;":
                    self._advance()
            result = items

        self._skip_ws_and_comments()
        if self._peek() == "}":
            self._advance()
        return result

    def _parse_array(self) -> List[Any]:
        self._advance()  # skip [
        items: List[Any] = []
        while True:
            self._skip_ws_and_comments()
            if self._peek() == "]":
                self._advance()
                break
            if self._pos >= len(self._text):
                break
            items.append(self._parse_value())
            self._skip_ws_and_comments()
            if self._peek() in ",;":
                self._advance()
        return items

    def _parse_string_literal(self) -> str:
        self._advance()  # skip opening "
        result = []
        while self._pos < len(self._text):
            c = self._text[self._pos]
            if c == '"':
                self._advance()
                break
            if c == "\\":
                self._advance()
                result.append(self._advance())
            else:
                result.append(self._advance())
        return "".join(result)

    def _parse_expression(self) -> Any:
        """Parsea una expresión numérica con posibles símbolos y operadores."""
        self._skip_ws_and_comments()
        result = self._parse_term()
        while True:
            self._skip_ws_and_comments()
            c = self._peek()
            if c == "+":
                self._advance()
                right = self._parse_term()
                result = self._num(result) + self._num(right)
            elif c == "-":
                # Check it's not start of a new key (letter after -)
                next_pos = self._pos + 1
                if next_pos < len(self._text) and self._text[next_pos].isalpha():
                    break
                self._advance()
                right = self._parse_term()
                result = self._num(result) - self._num(right)
            else:
                break
        return result

    def _parse_term(self) -> Any:
        self._skip_ws_and_comments()
        result = self._parse_factor()
        while True:
            self._skip_ws_and_comments()
            c = self._peek()
            if c == "*":
                self._advance()
                right = self._parse_factor()
                result = self._num(result) * self._num(right)
            elif c == "/":
                self._advance()
                right = self._parse_factor()
                d = self._num(right)
                result = self._num(result) / d if d != 0 else 0.0
            else:
                break
        return result

    def _parse_factor(self) -> Any:
        self._skip_ws_and_comments()
        c = self._peek()

        # Unary minus
        if c == "-":
            self._advance()
            val = self._parse_factor()
            return -self._num(val)

        # Parentheses
        if c == "(":
            self._advance()
            val = self._parse_expression()
            self._skip_ws_and_comments()
            if self._peek() == ")":
                self._advance()
            return val

        # Number or identifier
        if c.isdigit() or c == ".":
            return self._parse_number()

        # Identifier (symbol or enum value)
        ident = self._parse_identifier()
        if not ident:
            return 0.0

        # Check for deg suffix on numbers parsed as ident continuation
        self._skip_ws_and_comments()

        # Resolve symbol
        if ident in self._symbols:
            val = self._symbols[ident]
            # Check for deg suffix
            self._skip_ws_and_comments()
            rest = self._text[self._pos:self._pos + 3]
            if rest == "deg":
                self._pos += 3
                return val  # Already in degrees from symbol
            return val

        return ident  # Return as string (enum value, kick name, etc.)

    def _parse_number(self) -> float:
        start = self._pos
        has_dot = False
        if self._peek() == "-":
            self._advance()
        while self._pos < len(self._text):
            c = self._text[self._pos]
            if c.isdigit():
                self._pos += 1
            elif c == "." and not has_dot:
                has_dot = True
                self._pos += 1
            elif c == "e" or c == "E":
                self._pos += 1
                if self._pos < len(self._text) and self._text[self._pos] in "+-":
                    self._pos += 1
            else:
                break
        num_str = self._text[start:self._pos]
        try:
            val = float(num_str)
        except ValueError:
            val = 0.0

        # Check for deg suffix
        self._skip_ws_and_comments()
        rest = self._text[self._pos:self._pos + 3]
        if rest == "deg":
            self._pos += 3
            return val  # Keep in degrees (converted to rad at consumer level)

        # Check for unit suffixes to skip (mm, ms, etc.)
        rest2 = self._text[self._pos:self._pos + 2]
        if rest2 in ("mm", "ms"):
            self._pos += 2

        return val

    def _num(self, val: Any) -> float:
        """Coerce value to float for arithmetic."""
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str) and val in self._symbols:
            return self._symbols[val]
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. DATACLASSES CONTRACTUALES — Tipadas, con unidades documentadas
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FieldConfig4v4:
    """Dimensiones del campo 4v4 Scaled en milímetros."""
    x_opponent_ground: float = 2813.0    # xPosOpponentGroundLine
    x_own_ground:      float = -2813.0   # xPosOwnGroundLine
    y_left_sideline:   float = 1875.0    # yPosLeftSideline
    y_right_sideline:  float = -1875.0   # yPosRightSideline
    y_left_goal:       float = 800.0     # yPosLeftGoal
    y_right_goal:      float = -800.0    # yPosRightGoal
    center_circle_r:   float = 469.0     # centerCircleRadius
    x_opp_penalty:     float = 1781.0    # xPosOpponentPenaltyArea
    x_own_penalty:     float = -1781.0   # xPosOwnPenaltyArea
    x_opp_goal_area:   float = 2438.0    # xPosOpponentGoalArea
    x_own_goal_area:   float = -2438.0   # xPosOwnGoalArea
    y_left_penalty:    float = 1250.0    # yPosLeftPenaltyArea
    y_right_penalty:   float = -1250.0   # yPosRightPenaltyArea
    goal_post_radius:  float = 50.0
    field_border_exp:  float = 300.0     # field_border_expansion from libWalkProvider

    @property
    def length(self) -> float:
        return self.x_opponent_ground - self.x_own_ground  # 5626 mm

    @property
    def width(self) -> float:
        return self.y_left_sideline - self.y_right_sideline  # 3750 mm

    @property
    def goal_width(self) -> float:
        return self.y_left_goal - self.y_right_goal  # 1600 mm

    @property
    def diagonal(self) -> float:
        return math.hypot(self.length, self.width)


@dataclass(frozen=True)
class PoseInfo:
    """Posición de un rol en una táctica (mm, deg)."""
    x: float        # mm
    y: float        # mm
    rotation: float  # deg


@dataclass(frozen=True)
class RolePosition:
    role_type: str   # "goalkeeper", "defenderL", "defenderR", "forwardL", "forwardR"
    pose: PoseInfo


@dataclass(frozen=True)
class PriorityGroup:
    """Grupo de prioridad para asignación de roles en una táctica."""
    positions: Tuple[str, ...]    # role names
    priorities: Tuple[int, ...]   # priority numbers


@dataclass(frozen=True)
class TacticConfig:
    name: str
    positions: Tuple[RolePosition, ...]
    priority_groups: Tuple[PriorityGroup, ...] = ()


@dataclass(frozen=True)
class TransitionCondition:
    num_field_players_ge: Optional[int] = None
    num_field_players_le: Optional[int] = None
    ball_x_threshold: Optional[float] = None       # mm
    time_ball_ahead_ge: Optional[int] = None        # ms
    time_ball_behind_ge: Optional[int] = None       # ms


@dataclass(frozen=True)
class Transition:
    to: str  # tactic name
    conditions: Tuple[TransitionCondition, ...]


@dataclass(frozen=True)
class TacticState:
    tactic: str
    transitions: Tuple[Transition, ...]


@dataclass(frozen=True)
class SetPlayEntry:
    """A set play entry (kickoff, penalty, free kick)."""
    type: str
    weight: float = 1.0


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    tactics: Tuple[TacticState, ...]
    own_kickoffs: Tuple[SetPlayEntry, ...] = ()
    opponent_kickoffs: Tuple[SetPlayEntry, ...] = ()
    own_penalty_kicks: Tuple[SetPlayEntry, ...] = ()
    opponent_penalty_kicks: Tuple[SetPlayEntry, ...] = ()
    own_free_kicks: Tuple[SetPlayEntry, ...] = ()
    opponent_free_kicks: Tuple[SetPlayEntry, ...] = ()


@dataclass(frozen=True)
class SmashOrPassParams:
    """PlayBall.h defaults — decision thresholds. Fase 2 learnable."""
    shoot_threshold: float = 0.8          # L20
    min_pass_distance: float = 1500.0     # L22 mm
    max_pass_distance: float = 4500.0     # L23 mm
    max_free_kick_distance: float = 6000.0  # L24 mm (HARD)
    max_goal_kick_distance: float = 9000.0  # L25 mm (HARD)
    min_penalty: float = 0.1              # L26
    max_penalty: float = 0.8              # L27
    ball_dist_min_penalty: float = 300.0  # L28 mm (HARD)
    ball_dist_max_penalty: float = 200.0  # L29 mm (HARD)
    min_rating: float = 0.01             # L30 (HARD)
    sample_distance: float = 500.0        # L33 mm
    sample_steps_min: int = -1            # L34
    sample_steps_max: int = 2             # L34


@dataclass(frozen=True)
class KickAtGoalConfig:
    """kickAtGoalImpl.cfg parameters."""
    hysteresis_number: int = 300            # ms
    hysteresis_kick_range_ext: float = 500.0  # mm
    hysteresis_angle: float = 10.0          # deg
    min_opening_angle: float = 6.0          # deg
    kick_inaccuracy: float = 10.0           # deg
    hysteresis_decision_penalty: int = 500  # ms
    allowed_kicks: Tuple[str, ...] = (
        "forwardFastLeftLong", "forwardFastRightLong",
        "walkForwardsLeftLong", "walkForwardsRightLong",
    )


@dataclass(frozen=True)
class PassToTeammateConfig:
    """passToTeammateImpl.cfg parameters."""
    min_kick_length_for_long: float = 2813.0    # mm
    rating_threshold: float = 0.7
    pass_ahead_distance: float = 0.0            # mm
    min_angle_offset: float = 7.0               # deg
    max_angle_offset: float = 25.0              # deg
    max_angle_offset_behind: float = 10.0       # deg
    max_precision_offset: float = 5.0           # deg
    min_time_waiting: int = 3000                # ms
    max_time_waiting: int = 8000                # ms
    ignore_obstacles_threshold: float = 250.0   # mm
    kick_range_offset: float = 313.0            # mm
    average_kick_execution_time: int = 800      # ms
    target_offset_to_field_border: float = 500.0  # mm
    min_rating: float = 0.01
    allowed_kicks: Tuple[str, ...] = (
        "forwardFastRightLong", "forwardFastLeftLong",
        "forwardFastRight", "forwardFastLeft",
        "forwardFastRightPass", "forwardFastLeftPass",
        "walkForwardsRightLong", "walkForwardsLeftLong",
        "walkForwardsRightAlternative", "walkForwardsLeftAlternative",
    )


@dataclass(frozen=True)
class ZweikampfSearchParams:
    num_angles_near_best: int = 3
    range_of_best_duel_pose: float = 10.0       # deg
    num_overall_search: int = 5
    overall_search_range: float = 180.0         # deg
    goal_sector_width: float = 1.0              # deg


@dataclass(frozen=True)
class ZweikampfDuelRatings:
    rating_pose_blocked: float = 1.0
    rating_opponent_faster: float = 0.1
    rating_steal_ball: float = -0.4
    rating_goal_shot: float = -1.5
    rating_steal_ball_kick: float = -0.9


@dataclass(frozen=True)
class ZweikampfTimings:
    max_time_doing_nothing: int = 1000      # ms
    deactive_after: int = 5000              # ms
    no_kick_min_time: int = 200             # ms
    no_kick_steal_min_time: int = 250       # ms


@dataclass(frozen=True)
class ZweikampfObstacleHandling:
    obstacle_blocking_radius: float = 200.0       # mm
    max_obstacle_dist_for_walk_steal: float = 100.0  # mm
    max_obstacle_dist_for_side_steal: float = 450.0  # mm
    obstacle_min_dist_for_back_pass: float = 600.0   # mm


@dataclass(frozen=True)
class ZweikampfStealBallParams:
    steal_ball_min_robot_rotation: float = 70.0      # deg
    steal_ball_min_position_rotation: float = 45.0   # deg


@dataclass(frozen=True)
class ZweikampfLastKickHysterese:
    kick_length_hysterese: float = 500.0    # mm
    kick_direction_hysterese: float = 20.0  # deg


@dataclass(frozen=True)
class ZweikampfConfig:
    search: ZweikampfSearchParams = field(default_factory=ZweikampfSearchParams)
    ratings: ZweikampfDuelRatings = field(default_factory=ZweikampfDuelRatings)
    timings: ZweikampfTimings = field(default_factory=ZweikampfTimings)
    obstacle_handling: ZweikampfObstacleHandling = field(default_factory=ZweikampfObstacleHandling)
    steal_ball: ZweikampfStealBallParams = field(default_factory=ZweikampfStealBallParams)
    last_kick_hysterese: ZweikampfLastKickHysterese = field(default_factory=ZweikampfLastKickHysterese)
    field_border_safe_distance: float = 625.0    # mm
    allow_pass_angle_min: float = -45.0          # deg
    allow_pass_angle_max: float = 45.0           # deg
    allowed_kicks: Tuple[str, ...] = (
        "walkForwardsLeft", "walkForwardsRight",
        "walkForwardsLeftLong", "walkForwardsRightLong",
        "walkForwardsRightAlternative", "walkForwardsLeftAlternative",
        "walkForwardStealBallLeft", "walkForwardStealBallRight",
    )


@dataclass(frozen=True)
class WalkConfig:
    """walkingEngine.cfg + libWalkProvider.cfg parameters."""
    max_forward: float = 250.0         # mm/s
    max_sideways: float = 200.0        # mm/s
    max_backwards: float = 150.0       # mm/s
    max_rotation: float = 120.0        # deg/s
    # Avoidance (libWalkProvider.cfg)
    obstacle_avoidance_distance: float = 480.0     # mm
    obstacle_avoidance_min_radius: float = 220.0   # mm
    obstacle_avoidance_max_radius: float = 600.0   # mm
    ball_avoidance_distance: float = 465.0         # mm
    field_border_expansion: float = 300.0          # mm
    field_border_avoidance_distance: float = 400.0 # mm


@dataclass(frozen=True)
class KickInfoEntry:
    """A single kick from kickInfo.cfg."""
    kick_type: str
    range_min: float        # mm
    range_max: float        # mm
    ball_velocity_min: float  # mm/s
    ball_velocity_max: float  # mm/s
    execution_time: int     # ms
    motion: str             # "kick" or "walk"


@dataclass(frozen=True)
class KickInfo:
    """Rangos de kick (kickInfo.cfg)."""
    walk_forward_min: float = 400.0      # mm
    walk_forward_max: float = 1000.0     # mm
    forward_fast_max: float = 5400.0     # mm
    ball_velocity_max: float = 1800.0    # mm/s
    long_kick_threshold: float = 2813.0  # mm (= xPosOpponentGroundLine)
    kicks: Tuple[KickInfoEntry, ...] = ()  # all parsed kicks


@dataclass(frozen=True)
class HandoffConfig:
    """Handoff/takeover envelope thresholds (BHUMANPARSER_GUIDE 2.2-2.3)."""
    # handoff_ready conditions
    ball_seen_age_max: float = 300.0      # ms
    ball_rel_x_min: float = 0.0           # mm
    ball_rel_x_max: float = 500.0         # mm
    ball_rel_y_abs_max: float = 300.0     # mm
    angle_to_ball_abs_max: float = 80.0   # deg
    # takeover
    takeover_timeout_frames: int = 90     # ~3s at 33ms/frame
    takeover_grace_frames: int = 2        # frames without evidence before soft exit


@dataclass(frozen=True)
class CognitionTimingConfig:
    """Temporal budget constraints."""
    cognition_cycle_ms: float = 33.0    # ms (~30 FPS)
    rl_max_inference_ms: float = 3.0    # ms
    bridge_max_latency_ms: float = 2.0  # ms
    team_message_rate_hz: float = 10.0  # Hz


@dataclass(frozen=True)
class SkillBehaviorControlConfig:
    """skillBehaviorControl.cfg — gating options."""
    options: Tuple[str, ...] = (
        "HandlePhysicalRobot", "HandlePlayerState",
        "HandleReturnFromSideline", "HandlePenaltyShootout",
        "HandlePenaltyKick", "HandleRefereeSignal",
        "HandleGameState", "HandleIllegalAreas",
    )
    playing_options: Tuple[str, ...] = (
        "HandleGoalkeeperCatchBall", "HandleStrikerLostBall",
        "HandleCatchBall",
    )


@dataclass(frozen=True)
class StrategyBehaviorControlConfig:
    """strategyBehaviorControl.cfg — active strategy reference."""
    strategy: str = "s4v4_scaled"


@dataclass(frozen=True)
class BehaviorParams:
    """behaviorParameters.cfg — keeper and penalty params."""
    keeper_jumping_on: bool = True
    penalty_striker_walk_speed: float = 0.5
    ball_catch_max_walk_distance: float = 1200.0  # mm
    stand_radius: float = 60.0       # mm
    walk_radius: float = 125.0       # mm
    genuflect_radius: float = 200.0  # mm
    jump_radius: float = 600.0       # mm


# ─────────────────────────────────────────────────────────────────────────────
# 4. AllConfigs — Estructura runtime consumible por env/reward/wrapper
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AllConfigs:
    """
    Estructura runtime completa derivada del parser.

    Trazabilidad: parser → AllConfigs → consumer runtime.
    Cada campo es la fuente de verdad para su dominio.
    """
    field: FieldConfig4v4
    strategy: StrategyConfig
    tactics: Dict[str, TacticConfig]
    smash_or_pass: SmashOrPassParams
    kick_at_goal: KickAtGoalConfig
    pass_to_teammate: PassToTeammateConfig
    zweikampf: ZweikampfConfig
    walk: WalkConfig
    kick_info: KickInfo
    handoff: HandoffConfig
    timing: CognitionTimingConfig
    skill_behavior_control: SkillBehaviorControlConfig
    strategy_behavior_control: StrategyBehaviorControlConfig
    behavior: BehaviorParams
    # Source tracking
    loaded_from: str = "defaults"  # "defaults" or "cfg_files"


# ─────────────────────────────────────────────────────────────────────────────
# 5. CONTRACT LOADER — Produce AllConfigs
# ─────────────────────────────────────────────────────────────────────────────

# Expected file layout under cfg_root (= Config/ directory)
_REQUIRED_CFG_FILES = {
    "field":         "Locations/4v4_Scaled/fieldDimensions.cfg",
    "strategy":      "Behavior/Strategies/s4v4_scaled.cfg",
    "tactic_t121":   "Behavior/Tactics/t121_4v4_scaled.cfg",
    "tactic_t13":    "Behavior/Tactics/t13_4v4_scaled.cfg",
    "kick_at_goal":  "Scenarios/4v4_Scaled/BehaviorControl/kickAtGoalImpl.cfg",
    "pass":          "Scenarios/4v4_Scaled/BehaviorControl/passToTeammateImpl.cfg",
    "zweikampf":     "Scenarios/4v4_Scaled/BehaviorControl/zweikampfImpl.cfg",
    "kick_info":     "Scenarios/4v4_Scaled/kickInfo.cfg",
    "skill_bc":      "Scenarios/4v4_Scaled/skillBehaviorControl.cfg",
    "strategy_bc":   "Scenarios/4v4_Scaled/strategyBehaviorControl.cfg",
    "behavior":      "Scenarios/4v4_Scaled/behaviorParameters.cfg",
    "walk":          "Scenarios/4v4_Scaled/libWalkProvider.cfg",
}


# Default tactic positions (resolved from CFG expressions)
_T121_POSITIONS = (
    RolePosition("goalkeeper", PoseInfo(-2532.0, 0.0, 0.0)),      # xPosOwnGoalArea - 94
    RolePosition("defenderL", PoseInfo(-1468.0, 0.0, 0.0)),       # xPosOwnPenaltyArea + 313
    RolePosition("defenderR", PoseInfo(-234.5, -625.0, 0.0)),     # -centerCircleRadius*0.5
    RolePosition("forwardL",  PoseInfo(-234.5, 625.0, 0.0)),      # -centerCircleRadius*0.5
    RolePosition("forwardR",  PoseInfo(703.5, 0.0, 0.0)),         # centerCircleRadius*1.5
)

_T121_PRIORITY_GROUPS = (
    PriorityGroup(positions=("defenderR",), priorities=(1,)),
    PriorityGroup(positions=("forwardR",), priorities=(2,)),
    PriorityGroup(positions=("forwardL",), priorities=(3,)),
    PriorityGroup(positions=("defenderL",), priorities=(4,)),
)

_T13_POSITIONS = (
    RolePosition("goalkeeper", PoseInfo(-2532.0, 0.0, 0.0)),
    RolePosition("defenderL", PoseInfo(-1938.0, 0.0, 0.0)),       # xPosOwnGoalArea + 500
    RolePosition("forwardL",  PoseInfo(234.5, 500.0, 0.0)),       # centerCircleRadius*0.5
    RolePosition("forwardR",  PoseInfo(234.5, -500.0, 0.0)),      # centerCircleRadius*0.5
)

_T13_PRIORITY_GROUPS = (
    PriorityGroup(positions=("defenderL",), priorities=(1,)),
    PriorityGroup(positions=("forwardL",), priorities=(2,)),
    PriorityGroup(positions=("forwardR",), priorities=(3,)),
)


# Strategy FSM: s4v4_scaled with real transition values from CFG
_DEFAULT_STRATEGY = StrategyConfig(
    name="s4v4_scaled",
    tactics=(
        TacticState(
            tactic="t121_4v4_scaled",
            transitions=(
                Transition(
                    to="t13_4v4_scaled",
                    conditions=(
                        TransitionCondition(
                            num_field_players_ge=3,
                            ball_x_threshold=500.0,
                            time_ball_ahead_ge=280,
                        ),
                    ),
                ),
            ),
        ),
        TacticState(
            tactic="t13_4v4_scaled",
            transitions=(
                Transition(
                    to="t121_4v4_scaled",
                    conditions=(
                        TransitionCondition(
                            num_field_players_le=2,
                        ),
                    ),
                ),
                Transition(
                    to="t121_4v4_scaled",
                    conditions=(
                        TransitionCondition(
                            num_field_players_ge=3,
                            ball_x_threshold=-500.0,
                            time_ball_behind_ge=280,
                        ),
                    ),
                ),
            ),
        ),
    ),
    own_kickoffs=(SetPlayEntry("attackKickOff_scaled"),),
    opponent_kickoffs=(SetPlayEntry("kiteKickOff_scaled"),),
    own_penalty_kicks=(SetPlayEntry("theOneTrueOwnPenaltyKick_scaled"),),
    opponent_penalty_kicks=(SetPlayEntry("theOneTrueOpponentPenaltyKick_scaled"),),
    own_free_kicks=(
        SetPlayEntry("ownCornerKick_scaled"),
        SetPlayEntry("ownGoalKick_scaled"),
    ),
    opponent_free_kicks=(SetPlayEntry("opponentCornerKick_scaled"),),
)


def load_contract(cfg_root: Optional[str] = None) -> AllConfigs:
    """
    Produce AllConfigs from CFG files or defaults.

    If cfg_root is provided, attempts to parse all required .cfg files.
    Fails hard if cfg_root is given but required files are missing.
    If cfg_root is None, uses embedded defaults (documentation-audited).

    Args:
        cfg_root: Path to BHuman Config/ directory, or None for defaults.

    Returns:
        AllConfigs with all parameters for the 4v4 Scaled contract.

    Raises:
        FileNotFoundError: if cfg_root is given but required CFGs are missing.
    """
    if cfg_root:
        root = Path(cfg_root)
        if not root.is_dir():
            raise FileNotFoundError(
                f"cfg_root={cfg_root} does not exist or is not a directory"
            )
        # Check all required files exist
        missing = []
        for label, rel_path in _REQUIRED_CFG_FILES.items():
            if not (root / rel_path).exists():
                missing.append(f"  {label}: {rel_path}")
        if missing:
            raise FileNotFoundError(
                f"Missing required CFG files under {cfg_root}:\n"
                + "\n".join(missing)
            )
        parser = BHumanCfgParser(FIELD_SYMBOLS_4V4_SCALED.copy())
        return _load_from_files(root, parser)

    return _load_defaults()


def _load_defaults() -> AllConfigs:
    """Load from embedded defaults (documentation-audited values)."""
    return AllConfigs(
        field=FieldConfig4v4(),
        strategy=_DEFAULT_STRATEGY,
        tactics={
            "t121_4v4_scaled": TacticConfig(
                name="t121_4v4_scaled",
                positions=_T121_POSITIONS,
                priority_groups=_T121_PRIORITY_GROUPS,
            ),
            "t13_4v4_scaled": TacticConfig(
                name="t13_4v4_scaled",
                positions=_T13_POSITIONS,
                priority_groups=_T13_PRIORITY_GROUPS,
            ),
        },
        smash_or_pass=SmashOrPassParams(),
        kick_at_goal=KickAtGoalConfig(),
        pass_to_teammate=PassToTeammateConfig(),
        zweikampf=ZweikampfConfig(),
        walk=WalkConfig(),
        kick_info=KickInfo(),
        handoff=HandoffConfig(),
        timing=CognitionTimingConfig(),
        skill_behavior_control=SkillBehaviorControlConfig(),
        strategy_behavior_control=StrategyBehaviorControlConfig(),
        behavior=BehaviorParams(),
        loaded_from="defaults",
    )


def _load_from_files(root: Path, parser: BHumanCfgParser) -> AllConfigs:
    """
    Load from actual CFG files. All files verified present before calling.

    Real file layout under root (= Config/ directory):
      Locations/4v4_Scaled/fieldDimensions.cfg
      Behavior/Strategies/s4v4_scaled.cfg
      Behavior/Tactics/t121_4v4_scaled.cfg
      Behavior/Tactics/t13_4v4_scaled.cfg
      Scenarios/4v4_Scaled/BehaviorControl/kickAtGoalImpl.cfg
      Scenarios/4v4_Scaled/BehaviorControl/passToTeammateImpl.cfg
      Scenarios/4v4_Scaled/BehaviorControl/zweikampfImpl.cfg
      Scenarios/4v4_Scaled/kickInfo.cfg
      Scenarios/4v4_Scaled/skillBehaviorControl.cfg
      Scenarios/4v4_Scaled/strategyBehaviorControl.cfg
      Scenarios/4v4_Scaled/behaviorParameters.cfg
      Scenarios/4v4_Scaled/libWalkProvider.cfg
    """
    def _p(label: str) -> Path:
        return root / _REQUIRED_CFG_FILES[label]

    # 1. Parse field dimensions first — updates symbol table
    fd_data = parser.parse_file(_p("field"))
    for key, val in fd_data.items():
        if isinstance(val, (int, float)):
            parser._symbols[key] = float(val)
    field_cfg = _build_field_config(fd_data, parser._symbols)

    # 2. Parse tactics (need updated symbols for expressions)
    t121_data = parser.parse_file(_p("tactic_t121"))
    t13_data = parser.parse_file(_p("tactic_t13"))
    t121_cfg = _build_tactic_config("t121_4v4_scaled", t121_data)
    t13_cfg = _build_tactic_config("t13_4v4_scaled", t13_data)

    # 3. Parse strategy
    strategy_data = parser.parse_file(_p("strategy"))
    strategy_cfg = _build_strategy_config(strategy_data)

    # 4. Parse action configs
    kag_data = parser.parse_file(_p("kick_at_goal"))
    kag_cfg = _build_kick_at_goal(kag_data)

    pass_data = parser.parse_file(_p("pass"))
    pass_cfg = _build_pass_to_teammate(pass_data)

    zk_data = parser.parse_file(_p("zweikampf"))
    zk_cfg = _build_zweikampf(zk_data)

    # 5. Parse kick info
    ki_data = parser.parse_file(_p("kick_info"))
    ki_cfg = _build_kick_info(ki_data)

    # 6. Parse walk/avoidance
    walk_data = parser.parse_file(_p("walk"))
    walk_cfg = _build_walk_config(walk_data)

    # 7. Parse behavior control configs
    sbc_data = parser.parse_file(_p("skill_bc"))
    sbc_cfg = _build_skill_behavior_control(sbc_data)

    stbc_data = parser.parse_file(_p("strategy_bc"))
    stbc_cfg = StrategyBehaviorControlConfig(
        strategy=str(stbc_data.get("strategy", "s4v4_scaled"))
    )

    bp_data = parser.parse_file(_p("behavior"))
    bp_cfg = _build_behavior_params(bp_data)

    return AllConfigs(
        field=field_cfg,
        strategy=strategy_cfg,
        tactics={
            "t121_4v4_scaled": t121_cfg,
            "t13_4v4_scaled": t13_cfg,
        },
        smash_or_pass=SmashOrPassParams(),  # from PlayBall.h, not a CFG file
        kick_at_goal=kag_cfg,
        pass_to_teammate=pass_cfg,
        zweikampf=zk_cfg,
        walk=walk_cfg,
        kick_info=ki_cfg,
        handoff=HandoffConfig(),  # contract constants, not from CFG
        timing=CognitionTimingConfig(),  # contract constants, not from CFG
        skill_behavior_control=sbc_cfg,
        strategy_behavior_control=stbc_cfg,
        behavior=bp_cfg,
        loaded_from="cfg_files",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6. BUILDER FUNCTIONS — Parse raw dicts into typed dataclasses
# ─────────────────────────────────────────────────────────────────────────────

def _g(d: dict, key: str, default: Any = 0.0) -> Any:
    """Safe get from parsed dict with default."""
    return d.get(key, default)


def _gf(d: dict, key: str, default: float = 0.0) -> float:
    """Safe get float."""
    v = d.get(key, default)
    return float(v) if isinstance(v, (int, float)) else default


def _gi(d: dict, key: str, default: int = 0) -> int:
    """Safe get int."""
    v = d.get(key, default)
    return int(v) if isinstance(v, (int, float)) else default


def _build_field_config(data: dict, symbols: dict) -> FieldConfig4v4:
    s = symbols  # Use resolved symbols (parser updates them during parse)
    return FieldConfig4v4(
        x_opponent_ground=s.get("xPosOpponentGroundLine", 2813.0),
        x_own_ground=s.get("xPosOwnGroundLine", -2813.0),
        y_left_sideline=s.get("yPosLeftSideline", 1875.0),
        y_right_sideline=s.get("yPosRightSideline", -1875.0),
        y_left_goal=s.get("yPosLeftGoal", 800.0),
        y_right_goal=s.get("yPosRightGoal", -800.0),
        center_circle_r=s.get("centerCircleRadius", 469.0),
        x_opp_penalty=s.get("xPosOpponentPenaltyArea", 1781.0),
        x_own_penalty=s.get("xPosOwnPenaltyArea", -1781.0),
        x_opp_goal_area=s.get("xPosOpponentGoalArea", 2438.0),
        x_own_goal_area=s.get("xPosOwnGoalArea", -2438.0),
        y_left_penalty=s.get("yPosLeftPenaltyArea", 1250.0),
        y_right_penalty=s.get("yPosRightPenaltyArea", -1250.0),
        goal_post_radius=s.get("goalPostRadius", 50.0),
        field_border_exp=300.0,  # from libWalkProvider, set later if available
    )


def _build_tactic_config(name: str, data: dict) -> TacticConfig:
    positions_raw = data.get("positions", [])
    positions = []
    for p in positions_raw:
        if not isinstance(p, dict):
            continue
        role_type = str(p.get("type", "unknown"))
        pose_d = p.get("pose", {})
        if isinstance(pose_d, dict):
            rotation = _gf(pose_d, "rotation", 0.0)
            trans = pose_d.get("translation", {})
            if isinstance(trans, dict):
                x = _gf(trans, "x", 0.0)
                y = _gf(trans, "y", 0.0)
            else:
                x, y = 0.0, 0.0
        else:
            rotation, x, y = 0.0, 0.0, 0.0
        positions.append(RolePosition(role_type, PoseInfo(x, y, rotation)))

    # Parse priority groups
    pg_raw = data.get("priorityGroups", [])
    priority_groups = []
    for pg in pg_raw:
        if not isinstance(pg, dict):
            continue
        pos_list = pg.get("positions", [])
        pri_list = pg.get("priorities", [])
        if isinstance(pos_list, list) and isinstance(pri_list, list):
            pos_names = tuple(str(p) for p in pos_list)
            pri_nums = tuple(int(p) if isinstance(p, (int, float)) else 0 for p in pri_list)
            priority_groups.append(PriorityGroup(pos_names, pri_nums))

    return TacticConfig(
        name=name,
        positions=tuple(positions),
        priority_groups=tuple(priority_groups),
    )


def _build_strategy_config(data: dict) -> StrategyConfig:
    tactics_raw = data.get("tactics", [])
    tactic_states = []
    for ts in tactics_raw:
        if not isinstance(ts, dict):
            continue
        tactic_name = str(ts.get("tactic", ""))
        transitions_raw = ts.get("transitions", [])
        transitions = []
        for tr in transitions_raw:
            if not isinstance(tr, dict):
                continue
            to = str(tr.get("to", ""))
            conditions_raw = tr.get("conditions", [])
            conditions = []
            for cond in conditions_raw:
                if not isinstance(cond, dict):
                    continue
                conditions.append(TransitionCondition(
                    num_field_players_ge=_gi(cond, "numOfFieldPlayersGE", None) if "numOfFieldPlayersGE" in cond else None,
                    num_field_players_le=_gi(cond, "numOfFieldPlayersLE", None) if "numOfFieldPlayersLE" in cond else None,
                    ball_x_threshold=_gf(cond, "ballXThreshold", None) if "ballXThreshold" in cond else None,
                    time_ball_ahead_ge=_gi(cond, "timeSinceBallAheadOfThresholdGE", None) if "timeSinceBallAheadOfThresholdGE" in cond else None,
                    time_ball_behind_ge=_gi(cond, "timeSinceBallBehindThresholdGE", None) if "timeSinceBallBehindThresholdGE" in cond else None,
                ))
            transitions.append(Transition(to=to, conditions=tuple(conditions)))
        tactic_states.append(TacticState(tactic=tactic_name, transitions=tuple(transitions)))

    # Parse set plays
    def _parse_set_plays(key: str) -> Tuple[SetPlayEntry, ...]:
        raw = data.get(key, [])
        entries = []
        for e in raw:
            if isinstance(e, dict):
                entries.append(SetPlayEntry(
                    type=str(e.get("type", "")),
                    weight=_gf(e, "weight", 1.0),
                ))
        return tuple(entries)

    return StrategyConfig(
        name="s4v4_scaled",
        tactics=tuple(tactic_states),
        own_kickoffs=_parse_set_plays("ownKickOffs"),
        opponent_kickoffs=_parse_set_plays("opponentKickOffs"),
        own_penalty_kicks=_parse_set_plays("ownPenaltyKicks"),
        opponent_penalty_kicks=_parse_set_plays("opponentPenaltyKicks"),
        own_free_kicks=_parse_set_plays("ownFreeKicks"),
        opponent_free_kicks=_parse_set_plays("opponentFreeKicks"),
    )


def _build_kick_at_goal(data: dict) -> KickAtGoalConfig:
    allowed = data.get("allowedKicks", [])
    kick_names = tuple(str(k) for k in allowed if isinstance(k, str))
    return KickAtGoalConfig(
        hysteresis_number=_gi(data, "hysteresisNumber", 300),
        hysteresis_kick_range_ext=_gf(data, "hysteresisKickRangeExtension", 500.0),
        hysteresis_angle=_gf(data, "hysteresisAngle", 10.0),
        min_opening_angle=_gf(data, "minOpeningAngle", 6.0),
        kick_inaccuracy=_gf(data, "kickInaccuracy", 10.0),
        hysteresis_decision_penalty=_gi(data, "hysteresisDecisionPenalty", 500),
        allowed_kicks=kick_names if kick_names else KickAtGoalConfig.allowed_kicks,
    )


def _build_pass_to_teammate(data: dict) -> PassToTeammateConfig:
    allowed = data.get("allowedKicks", [])
    kick_names = tuple(str(k) for k in allowed if isinstance(k, str))
    return PassToTeammateConfig(
        min_kick_length_for_long=_gf(data, "minKickLengthForLong", 2813.0),
        rating_threshold=_gf(data, "ratingThreshold", 0.7),
        pass_ahead_distance=_gf(data, "passAheadDistance", 0.0),
        min_angle_offset=_gf(data, "minAngleOffset", 7.0),
        max_angle_offset=_gf(data, "maxAngleOffset", 25.0),
        max_angle_offset_behind=_gf(data, "maxAngleOffsetBehind", 10.0),
        max_precision_offset=_gf(data, "maxPrecisionOffset", 5.0),
        min_time_waiting=_gi(data, "minTimeWaiting", 3000),
        max_time_waiting=_gi(data, "maxTimeWaiting", 8000),
        ignore_obstacles_threshold=_gf(data, "ignoreObstaclesThreshold", 250.0),
        kick_range_offset=_gf(data, "kickRangeOffset", 313.0),
        average_kick_execution_time=_gi(data, "averageKickExecutionTime", 800),
        target_offset_to_field_border=_gf(data, "targetOffsetToFieldBorder", 500.0),
        min_rating=_gf(data, "minRating", 0.01),
        allowed_kicks=kick_names if kick_names else PassToTeammateConfig.allowed_kicks,
    )


def _build_zweikampf(data: dict) -> ZweikampfConfig:
    sp = data.get("searchParameters", {})
    dr = data.get("duelRatings", {})
    dt = data.get("duelTimings", {})
    oh = data.get("obstacleHandling", {})
    sb = data.get("stealBallParameters", {})
    lk = data.get("lastKickHysterese", {})

    allowed = data.get("allowedKicks", [])
    kick_names = tuple(str(k) for k in allowed if isinstance(k, str))

    return ZweikampfConfig(
        search=ZweikampfSearchParams(
            num_angles_near_best=_gi(sp, "numOfAnglesNearBestDuelPose", 3),
            range_of_best_duel_pose=_gf(sp, "rangeOfBestDuelPose", 10.0),
            num_overall_search=_gi(sp, "numOfOverallSearch", 5),
            overall_search_range=_gf(sp, "overallSearchRange", 180.0),
            goal_sector_width=_gf(sp, "goalSectorWidth", 1.0),
        ),
        ratings=ZweikampfDuelRatings(
            rating_pose_blocked=_gf(dr, "ratingPoseBlocked", 1.0),
            rating_opponent_faster=_gf(dr, "ratingOpponentFaster", 0.1),
            rating_steal_ball=_gf(dr, "ratingStealBall", -0.4),
            rating_goal_shot=_gf(dr, "ratingGoalShot", -1.5),
            rating_steal_ball_kick=_gf(dr, "ratingStealBallKick", -0.9),
        ),
        timings=ZweikampfTimings(
            max_time_doing_nothing=_gi(dt, "maxTimeDoingNothing", 1000),
            deactive_after=_gi(dt, "deactiveAfterDoingNothing", 5000),
            no_kick_min_time=_gi(dt, "noKickMinTime", 200),
            no_kick_steal_min_time=_gi(dt, "noKickStealMinTime", 250),
        ),
        obstacle_handling=ZweikampfObstacleHandling(
            obstacle_blocking_radius=_gf(oh, "obstacleBlockingRadius", 200.0),
            max_obstacle_dist_for_walk_steal=_gf(oh, "maxObstacleDistanceForWalkStealBallKick", 100.0),
            max_obstacle_dist_for_side_steal=_gf(oh, "maxObstacleDistanceForSideStealBallKick", 450.0),
            obstacle_min_dist_for_back_pass=_gf(oh, "obstacleMinDistanceForBackPass", 600.0),
        ),
        steal_ball=ZweikampfStealBallParams(
            steal_ball_min_robot_rotation=_gf(sb, "stealBallMinRobotRotation", 70.0),
            steal_ball_min_position_rotation=_gf(sb, "stealBallMinPositionRotation", 45.0),
        ),
        last_kick_hysterese=ZweikampfLastKickHysterese(
            kick_length_hysterese=_gf(lk, "kickLengthHysterese", 500.0),
            kick_direction_hysterese=_gf(lk, "kickDirectionHysterese", 20.0),
        ),
        field_border_safe_distance=_gf(data, "fieldBorderSafeDistance", 625.0),
        allow_pass_angle_min=_gf(data.get("allowPassAngle", {}), "min", -45.0) if isinstance(data.get("allowPassAngle"), dict) else -45.0,
        allow_pass_angle_max=_gf(data.get("allowPassAngle", {}), "max", 45.0) if isinstance(data.get("allowPassAngle"), dict) else 45.0,
        allowed_kicks=kick_names if kick_names else ZweikampfConfig.allowed_kicks,
    )


def _build_kick_info(data: dict) -> KickInfo:
    kicks_raw = data.get("kicks", {})
    entries = []
    min_range = 99999.0
    max_range = 0.0
    max_vel = 0.0

    for name, kd in kicks_raw.items():
        if not isinstance(kd, dict):
            continue
        r = kd.get("range", {})
        v = kd.get("ballVelocity", {})
        r_min = _gf(r, "min", 0.0) if isinstance(r, dict) else 0.0
        r_max = _gf(r, "max", 0.0) if isinstance(r, dict) else 0.0
        v_min = _gf(v, "min", 0.0) if isinstance(v, dict) else 0.0
        v_max = _gf(v, "max", 0.0) if isinstance(v, dict) else 0.0
        entry = KickInfoEntry(
            kick_type=str(kd.get("kickType", name)),
            range_min=r_min,
            range_max=r_max,
            ball_velocity_min=v_min,
            ball_velocity_max=v_max,
            execution_time=_gi(kd, "executionTime", 500),
            motion=str(kd.get("motion", "walk")),
        )
        entries.append(entry)
        if r_min < min_range and r_min > 0:
            min_range = r_min
        if r_max > max_range:
            max_range = r_max
        if v_max > max_vel:
            max_vel = v_max

    walk_min = min_range if min_range < 99999 else 400.0
    walk_max = 1000.0  # walkForwards range max
    for e in entries:
        if "walkForwards" in e.kick_type and "Long" not in e.kick_type and "Steal" not in e.kick_type:
            if e.range_max > walk_max:
                walk_max = e.range_max

    return KickInfo(
        walk_forward_min=walk_min,
        walk_forward_max=walk_max,
        forward_fast_max=max_range if max_range > 0 else 5400.0,
        ball_velocity_max=max_vel if max_vel > 0 else 1800.0,
        long_kick_threshold=2813.0,
        kicks=tuple(entries),
    )


def _build_walk_config(data: dict) -> WalkConfig:
    return WalkConfig(
        max_forward=250.0,   # from walkingEngine.cfg (not in libWalkProvider)
        max_sideways=200.0,
        max_backwards=150.0,
        max_rotation=120.0,
        obstacle_avoidance_distance=_gf(data, "obstacleAvoidanceDistance", 480.0),
        obstacle_avoidance_min_radius=_gf(data, "obstacleAvoidanceMinRadius", 220.0),
        obstacle_avoidance_max_radius=_gf(data, "obstacleAvoidanceMaxRadius", 600.0),
        ball_avoidance_distance=_gf(data, "ballAvoidanceDistance", 465.0),
        field_border_expansion=_gf(data, "fieldBorderExpansion", 300.0),
        field_border_avoidance_distance=_gf(data, "fieldBorderAvoidanceDistance", 400.0),
    )


def _build_skill_behavior_control(data: dict) -> SkillBehaviorControlConfig:
    options_raw = data.get("options", [])
    playing_raw = data.get("playingOptions", [])
    options = tuple(str(o) for o in options_raw if isinstance(o, str))
    playing = tuple(str(o) for o in playing_raw if isinstance(o, str))
    return SkillBehaviorControlConfig(
        options=options if options else SkillBehaviorControlConfig.options,
        playing_options=playing if playing else SkillBehaviorControlConfig.playing_options,
    )


def _build_behavior_params(data: dict) -> BehaviorParams:
    return BehaviorParams(
        keeper_jumping_on=data.get("keeperJumpingOn", True),
        penalty_striker_walk_speed=_gf(data, "penaltyStrikerWalkSpeed", 0.5),
        ball_catch_max_walk_distance=_gf(data, "ballCatchMaxWalkDistance", 1200.0),
        stand_radius=_gf(data, "standRadius", 60.0),
        walk_radius=_gf(data, "walkRadius", 125.0),
        genuflect_radius=_gf(data, "genuflectRadius", 200.0),
        jump_radius=_gf(data, "jumpRadius", 600.0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 7. VALID TACTIC NAMES — For validation
# ─────────────────────────────────────────────────────────────────────────────

VALID_TACTIC_NAMES = ("t121_4v4_scaled", "t13_4v4_scaled")
VALID_STRATEGY_NAME = "s4v4_scaled"

# ─────────────────────────────────────────────────────────────────────────────
# 8. HOME POSE LOOKUP — For env spawning
# ─────────────────────────────────────────────────────────────────────────────

def get_home_poses(tactic_name: str, configs: AllConfigs) -> Dict[str, Tuple[float, float]]:
    """
    Returns {role_type: (x, y)} for a given tactic from AllConfigs.
    Excludes goalkeeper for RL field player spawning.
    """
    tactic = configs.tactics.get(tactic_name)
    if tactic is None:
        tactic = configs.tactics.get("t121_4v4_scaled")
    if tactic is None:
        return {}
    return {
        rp.role_type: (rp.pose.x, rp.pose.y)
        for rp in tactic.positions
        if rp.role_type != "goalkeeper"
    }


# ─────────────────────────────────────────────────────────────────────────────
# 9. MACRO-ACTION TYPES — For macrostate tracking
# ─────────────────────────────────────────────────────────────────────────────

MACRO_TYPES = (
    "none",
    "playBall_shoot",
    "playBall_pass",
    "playBall_dribble",
    "playBall_duel",
    "playBall_goalpost",
)

MACRO_END_REASONS = (
    "none",
    "completed",
    "role_changed",
    "game_state_changed",
    "skill_request_cleared",
    "motion_request_cleared",
    "active_agent_changed",
    "ball_lost",
    "goal",
    "out",
    "timeout",
    "aborted_integrity",
    "aborted_bridge_fallback",
)

TAKEOVER_SKILLS = ("shoot", "pass", "dribble", "duel")
