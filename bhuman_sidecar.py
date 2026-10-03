"""
bhuman_sidecar.py -- BHuman Decision Sidecar for RL integration.

Architecture:
  - RealSidecarBackend: real C++ bridge orchestration across the active
    own-team players.
  - PythonContractStub: Python re-implementation of BHuman decision logic
    using parsed CFG parameters. ONLY for smoke tests and debug.
  - BHumanDecisionSidecar: dispatcher that selects backend based on mode.

In mode "bhuman_sidecar" (normal training):
  - If no real C++ backend is available, the sidecar enters
    "contract_validation_mode" — it runs the stub but marks output
    explicitly as non-authoritative. The env MUST NOT use
    BHumanTakeoverSimulator in this mode.

In mode "python_stub_smoke_only":
  - Uses PythonContractStub directly. Clearly marked.

Output schema aligned with BHuman real representations:
  - skill_request: uses real SkillRequest enum (no "duel")
  - motion_request: uses real MotionRequest.Motion enum
  - activation_graph: structured nodes (option, depth, state, etc.)
  - last_kick_type: "unavailable" unless real MotionInfo evidence exists
  - macro_type = playBall_duel comes from activation_graph zweikampf state

Source of truth: BHUMAN_IMPLEMENTATION_CORRECTION_GUIDE.md
                 BHUMAN_IN_LOOP_SIDECAR_GUIDE.md
"""

from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from bhuman_cfg_parser import AllConfigs, get_home_poses
from runtime_paths import preferred_bridge_mode


# ---------------------------------------------------------------------------
# ACTIVATION GRAPH NODE — structured, mirrors ActivationGraph.h
# ---------------------------------------------------------------------------

@dataclass
class ActivationNode:
    """
    Mirrors BHuman ActivationGraph node structure.
    See: Representations/BehaviorControl/ActivationGraph.h
    """
    option: str = ""
    depth: int = 0
    state: str = ""
    option_time: int = 0   # ms
    state_time: int = 0    # ms
    parameters: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "option": self.option,
            "depth": self.depth,
            "state": self.state,
            "option_time": self.option_time,
            "state_time": self.state_time,
            "parameters": self.parameters,
        }


# ---------------------------------------------------------------------------
# SKILL REQUEST — mirrors real SkillRequest.h enum
# ---------------------------------------------------------------------------

# Real BHuman SkillRequest skills (from SkillRequest.h):
#   none, stand, walk, shoot, pass, dribble, block, mark, observe
# NOTE: "duel" is NOT a skill. It is a sub-state of PlayBall.
VALID_SKILL_REQUESTS = frozenset({
    "none", "stand", "walk", "shoot", "pass", "dribble",
    "block", "mark", "observe",
})

# Real BHuman MotionRequest.Motion enum (from MotionRequest.h):
VALID_MOTION_REQUESTS = frozenset({
    "playDead", "stand", "walkAtAbsoluteSpeed", "walkAtRelativeSpeed",
    "walkToPose", "walkToBallAndKick", "dribble", "dive",
    "special", "replayWalk", "calibration",
})

DUEL_ACTIVATION_STATES = frozenset({"zweikampf"})
DUEL_ACTIVATION_OPTIONS = frozenset({"Zweikampf"})
DRIBBLE_ACTIVATION_STATES = frozenset({"dribbleCloseRange", "dribbleFarRange"})
DRIBBLE_ACTIVATION_OPTIONS = frozenset({"DribbleToGoal", "GoToBallAndDribble", "Dribble"})

CANONICAL_AGENT_IDS = tuple(f"agent_{i}" for i in range(4))
CANONICAL_ROLE_ORDER = ("defenderL", "defenderR", "forwardL", "forwardR")
TACTIC_NAME_NORMALIZATION = {
    "t121": "t121_4v4_scaled",
    "t13": "t13_4v4_scaled",
}


def normalize_tactic_name(name: Optional[str]) -> str:
    if name in (None, "", "none"):
        return "t121_4v4_scaled"
    return TACTIC_NAME_NORMALIZATION.get(str(name), str(name))


def infer_real_macro_type(skill: str,
                          motion_request: str,
                          activation_graph: List["ActivationNode"]) -> str:
    """
    Derive the *current* play-ball macro from real BHuman evidence.

    In the real bridge, `MotionRequest` can lag one sub-state behind the
    high-level `SkillRequest` near goal. Treating any transient dribble
    motion as `playBall_dribble` was contaminating takeover semantics:
    BHuman was already asking to `shoot` / `pass`, but our macro label
    kept flipping back to dribble. For contract fidelity we keep duel as
    highest priority, then preserve the high-level intent (`shoot/pass`)
    when it is explicitly present, and only fall back to dribble when
    there is no stronger play-ball intent.
    """
    states = {node.state for node in activation_graph if getattr(node, "state", "")}
    options = {node.option for node in activation_graph if getattr(node, "option", "")}

    if states & DUEL_ACTIVATION_STATES or options & DUEL_ACTIVATION_OPTIONS:
        return "playBall_duel"
    if skill == "shoot":
        return "playBall_shoot"
    if skill == "pass":
        return "playBall_pass"
    if (states & DRIBBLE_ACTIVATION_STATES or
            options & DRIBBLE_ACTIVATION_OPTIONS or
            motion_request == "dribble"):
        return "playBall_dribble"
    if skill == "dribble":
        return "playBall_dribble"
    return "none"


# ---------------------------------------------------------------------------
# OUTPUT STRUCTURE
# ---------------------------------------------------------------------------

@dataclass
class SidecarOutput:
    """Complete output from one sidecar decision frame."""
    # Core decisions
    active_agent: Optional[str] = None
    macro_type: str = "none"  # playBall_shoot/pass/dribble/duel/none

    # Strategy layer
    strategy_status: str = "s4v4_scaled"
    current_tactic: str = "t121_4v4_scaled"

    # Skill layer (real BHuman enum — never "duel")
    skill_request: str = "none"
    skill_target: Optional[Tuple[float, float]] = None
    skill_target_direction: Optional[float] = None

    # Behavior layer
    behavior_status: str = "idle"
    behavior_speed: float = 0.0
    behavior_walking_to: Optional[Tuple[float, float]] = None
    behavior_shooting_to: Optional[Tuple[float, float]] = None
    pass_target_player_number: Optional[int] = None
    pass_origin_player_number: Optional[int] = None

    # Motion layer (real BHuman enum)
    motion_request: str = "stand"
    motion_speed: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    motion_walk_target: Optional[Tuple[float, float, float]] = None
    motion_target_direction: Optional[float] = None
    motion_executed_phase: str = "unavailable"
    motion_executed_speed: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    motion_ball_target_field: Optional[Tuple[float, float]] = None
    motion_contact_point_field: Optional[Tuple[float, float]] = None
    robot_pose_field: Optional[Tuple[float, float, float]] = None
    field_ball_position: Optional[Tuple[float, float]] = None
    field_ball_intercepted_position: Optional[Tuple[float, float]] = None

    # Kick metadata
    kick_type: str = "none"
    kick_length: float = 0.0
    kick_length_raw: float = 0.0
    # last_kick_type: comes from MotionInfo evidence ONLY.
    # "unavailable" = no real execution evidence exists.
    last_kick_type: str = "unavailable"
    last_kick_timestamp: int = 0  # ms, 0 = no evidence

    # ActivationGraph — structured nodes
    activation_graph: List[ActivationNode] = field(default_factory=list)

    # Backend provenance
    backend: str = "unknown"  # "real", "contract_validation", "python_stub"
    contract_validation_mode: bool = False

    # Bridge provenance/debug metadata (only populated by real backend)
    bridge_mode: str = "persistent_2d_cognition"
    bridge_frame_id: int = -1
    bridge_player_number: int = -1
    bridge_metadata: Dict[str, Any] = field(default_factory=dict)

    # Timing
    decision_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# GROUND TRUTH WORLD STATE (conceptual input)
# ---------------------------------------------------------------------------

@dataclass
class GroundTruthFrame:
    """
    Minimal frame input mimicking GroundTruthWorldState + OracledWorldModelProvider.
    """
    ball_pos: np.ndarray
    ball_vel: np.ndarray
    ball_seen_age_ms: float = 0.0

    own_poses: Dict[str, np.ndarray] = field(default_factory=dict)
    own_thetas: Dict[str, float] = field(default_factory=dict)
    own_upright: Dict[str, bool] = field(default_factory=dict)
    own_roles: Dict[str, str] = field(default_factory=dict)

    opp_poses: List[np.ndarray] = field(default_factory=list)
    opp_upright: List[bool] = field(default_factory=list)

    is_playing: bool = True
    own_score: int = 0
    opp_score: int = 0


# ---------------------------------------------------------------------------
# REAL SIDECAR BACKEND
# ---------------------------------------------------------------------------

class RealSidecarBackend:
    """
    Real BHuman C++ decision sidecar via subprocess bridge.

    Uses bridge/bhuman_bridge.exe (compiled C++ executable) that creates
    a headless BHuman Robot in 2D mode, accepts GroundTruthWorldState
    frames as JSON on stdin, runs one cognition cycle per frame, and
    returns decisions (SkillRequest, MotionRequest, ActivationGraph,
    MotionInfo) as JSON on stdout.

    If the compiled bridge is not found, instantiation raises
    NotImplementedError (triggering fallback to contract_validation_mode).

    See bridge/bridge_protocol.py for the JSON protocol.
    See bridge/bhuman_bridge.cpp for the C++ implementation.
    """

    def __init__(self, contract: AllConfigs):
        from bridge.bridge_protocol import BHumanBridge, BridgeConfig
        self._contract = contract
        self._bridge_cls = BHumanBridge
        self._bridge_config_cls = BridgeConfig
        self._bridge_mode = preferred_bridge_mode()
        self._bridges: Dict[int, Any] = {}
        self._bridge_player_by_position: Dict[str, int] = {}
        self._player_count = 0
        self._frame_counter = 0
        self._prewarmed_player_count = 4
        self._last_active_player_number: Optional[int] = None
        self._executor = ThreadPoolExecutor(
            max_workers=self._prewarmed_player_count,
            thread_name_prefix="bhuman_sidecar",
        )
        self._ensure_bridges(self._prewarmed_player_count)
        self.reset()

    def decide(self, frame: GroundTruthFrame,
               active_agent_hint: Optional[str] = None,
               prefer_single_bridge_fast_path: bool = False) -> SidecarOutput:
        """Run one BHuman cognition cycle across the active own-team players."""
        self._frame_counter += 1
        agent_order = self._sorted_agent_ids(frame.own_poses.keys())
        player_count = max(1, len(agent_order))
        self._ensure_bridges(player_count)

        own_team = []
        for number, aid in enumerate(agent_order, start=1):
            pos = frame.own_poses[aid]
            own_team.append({
                "number": number,
                "x": float(pos[0]),
                "y": float(pos[1]),
                "theta": float(frame.own_thetas.get(aid, 0.0)),
                "upright": frame.own_upright.get(aid, True),
            })

        opp_team = []
        for i, opp_pos in enumerate(frame.opp_poses):
            opp_team.append({
                "number": i + 1,
                "x": float(opp_pos[0]),
                "y": float(opp_pos[1]),
                "theta": 0.0,
                "upright": frame.opp_upright[i] if i < len(frame.opp_upright) else True,
            })

        game_state = "playing" if frame.is_playing else "initial"
        timestamp_ms = self._frame_counter * 33
        team_t0 = time.perf_counter()
        results: Dict[int, dict]
        candidate_player: Optional[int] = None
        query_mode = "team_parallel"
        queried_players: List[int]
        fallback_reason = "none"

        use_single_bridge = prefer_single_bridge_fast_path and active_agent_hint in frame.own_roles
        if use_single_bridge:
            hinted_role = frame.own_roles.get(active_agent_hint, "")
            candidate_player = self._choose_role_mapped_player(hinted_role)
            if candidate_player is None:
                fallback_reason = "role_map_unavailable"
                use_single_bridge = False
            else:
                query_mode = "single_bridge_role_map"
                queried_players = [candidate_player]
                try:
                    candidate_result = self._step_single_bridge(
                        candidate_player,
                        frame,
                        own_team,
                        opp_team,
                        game_state,
                        timestamp_ms,
                    )
                except Exception as exc:
                    self.shutdown()
                    raise NotImplementedError(
                        f"C++ bridge failed during single bridge step: {exc}"
                    ) from exc
                results = {candidate_player: candidate_result}
                active_player = candidate_result.get("active_agent_number")
                if active_player is None:
                    active_player = candidate_player
        if not use_single_bridge:
            try:
                results = self._step_team_parallel(
                    frame,
                    own_team,
                    opp_team,
                    game_state,
                    timestamp_ms,
                )
            except Exception as exc:
                self.shutdown()
                raise NotImplementedError(
                    f"C++ bridge failed during team step: {exc}"
                ) from exc
            queried_players = list(sorted(results.keys()))
            self._update_position_map(results)
            active_player = self._select_active_player(results)

        canonical_player = active_player if active_player is not None else 1
        canonical = dict(results.get(canonical_player, {}))
        total_time_us = int((time.perf_counter() - team_t0) * 1_000_000.0)
        canonical["frame_id"] = self._frame_counter
        canonical["player_number"] = canonical_player
        canonical["active_agent_number"] = active_player
        canonical["debug_team_bridge_count"] = player_count
        canonical["debug_team_bridges_queried"] = len(queried_players)
        canonical["debug_team_query_mode"] = query_mode
        canonical["debug_team_query_fallback_reason"] = fallback_reason
        canonical["debug_team_decision_time_us"] = total_time_us
        canonical["debug_team_active_candidates"] = [
            int(v["active_agent_number"])
            for v in results.values()
            if v.get("active_agent_number") is not None
        ]
        canonical["debug_team_bridge_players"] = queried_players
        canonical["debug_active_agent_source"] = (
            "real_team_consensus" if active_player is not None else "unavailable"
        )
        if canonical_player in results:
            active_result = results[canonical_player]
            canonical["decision_time_us"] = active_result.get(
                "decision_time_us",
                canonical.get("decision_time_us", 0),
            )
            canonical["debug_active_agent_decision_time_us"] = active_result.get(
                "decision_time_us", 0
            )
        else:
            canonical["decision_time_us"] = canonical.get("decision_time_us", total_time_us)
        self._last_active_player_number = (
            active_player
            if active_player is not None
            else (candidate_player if candidate_player is not None else self._last_active_player_number)
        )

        return self._parse_bridge_result(canonical, agent_order)

    def reset(self):
        """Reset BHuman state for a new episode."""
        for bridge in self._bridges.values():
            bridge.reset()
        self._frame_counter = 0
        self._last_active_player_number = None

    def shutdown(self, close_executor: bool = True):
        """Gracefully shut down all bridge subprocesses."""
        for bridge in self._bridges.values():
            bridge.shutdown()
        self._bridges.clear()
        self._player_count = 0
        if close_executor:
            self._executor.shutdown(wait=True, cancel_futures=False)

    def _ensure_bridges(self, player_count: int) -> None:
        from bridge.bridge_protocol import BridgeNotFoundError

        if self._player_count == player_count and len(self._bridges) == player_count:
            return

        self.shutdown(close_executor=False)
        started: Dict[int, Any] = {}
        try:
            for player_number in range(1, player_count + 1):
                config = self._bridge_config_cls(
                    player_number=player_number,
                    num_players=player_count,
                    bridge_mode=self._bridge_mode,
                )
                bridge = self._bridge_cls(config)
                bridge.start()
                started[player_number] = bridge
        except BridgeNotFoundError as exc:
            for bridge in started.values():
                bridge.shutdown()
            raise NotImplementedError(
                f"C++ bridge executable not found. Build it first. {exc}"
            ) from exc
        except Exception as exc:
            for bridge in started.values():
                bridge.shutdown()
            raise NotImplementedError(
                f"C++ bridge failed to start for {player_count} players: {exc}"
            ) from exc

        self._bridges = started
        self._player_count = player_count

    def _sorted_agent_ids(self, agent_ids) -> List[str]:
        def key(agent_id: str) -> tuple[int, str]:
            try:
                return int(agent_id.rsplit("_", 1)[-1]), agent_id
            except ValueError:
                return 10_000, agent_id

        return sorted(agent_ids, key=key)

    def _step_single_bridge(
        self,
        player_number: int,
        frame: GroundTruthFrame,
        own_team: List[Dict[str, Any]],
        opp_team: List[Dict[str, Any]],
        game_state: str,
        timestamp_ms: int,
    ) -> dict:
        return self._bridges[player_number].step(
            ball_pos=frame.ball_pos,
            ball_vel=frame.ball_vel,
            own_team=own_team,
            opp_team=opp_team,
            game_state=game_state,
            own_score=frame.own_score,
            opp_score=frame.opp_score,
            timestamp_ms=timestamp_ms,
        )

    def _step_team_parallel(
        self,
        frame: GroundTruthFrame,
        own_team: List[Dict[str, Any]],
        opp_team: List[Dict[str, Any]],
        game_state: str,
        timestamp_ms: int,
    ) -> Dict[int, dict]:
        results: Dict[int, dict] = {}
        future_map = {}
        for player_number, bridge in self._bridges.items():
            future = self._executor.submit(
                bridge.step,
                ball_pos=frame.ball_pos,
                ball_vel=frame.ball_vel,
                own_team=own_team,
                opp_team=opp_team,
                game_state=game_state,
                own_score=frame.own_score,
                opp_score=frame.opp_score,
                timestamp_ms=timestamp_ms,
            )
            future_map[future] = player_number

        for future, player_number in future_map.items():
            results[player_number] = future.result()
        return results

    def _choose_candidate_player(
        self,
        agent_order: List[str],
        active_agent_hint: Optional[str],
    ) -> int:
        if active_agent_hint in agent_order:
            return agent_order.index(active_agent_hint) + 1
        if self._last_active_player_number is not None and 1 <= self._last_active_player_number <= len(agent_order):
            return self._last_active_player_number
        return 1

    def _choose_role_mapped_player(self, role_name: str) -> Optional[int]:
        if role_name in self._bridge_player_by_position:
            return self._bridge_player_by_position[role_name]
        return self._last_active_player_number

    def _update_position_map(self, results: Dict[int, dict]) -> None:
        for player_number, result in results.items():
            position = result.get("strategy_position")
            if position in (None, "", "none", "goalkeeper"):
                continue
            self._bridge_player_by_position[str(position)] = player_number

    def _select_active_player(self, results: Dict[int, dict]) -> Optional[int]:
        candidates: List[int] = []
        for player_number, result in results.items():
            active_number = result.get("active_agent_number")
            if active_number is not None:
                try:
                    active_number = int(active_number)
                except (TypeError, ValueError):
                    active_number = None
                if active_number is not None and active_number >= 1:
                    candidates.append(active_number)
                    continue

            skill = result.get("skill_request", "none")
            if skill in {"shoot", "pass", "dribble", "block"}:
                candidates.append(player_number)
                continue

            role = str(result.get("strategy_role", ""))
            if "active" in role.lower():
                candidates.append(player_number)

        if not candidates:
            return None

        most_common = Counter(candidates).most_common()
        best_count = most_common[0][1]
        best_candidates = [player for player, count in most_common if count == best_count]
        return min(best_candidates)

    def _active_agent_from_number(self, active_number: Any,
                                  agent_order: List[str]) -> Optional[str]:
        try:
            index = int(active_number) - 1
        except (TypeError, ValueError):
            return None
        if 0 <= index < len(agent_order):
            return agent_order[index]
        return None

    def _parse_xy(self, value: Any) -> Optional[Tuple[float, float]]:
        if not isinstance(value, dict):
            return None
        if "x_mm" not in value or "y_mm" not in value:
            return None
        return (float(value["x_mm"]), float(value["y_mm"]))

    def _parse_pose(self, value: Any) -> Optional[Tuple[float, float, float]]:
        if not isinstance(value, dict):
            return None
        if "x_mm" not in value or "y_mm" not in value:
            return None
        theta = float(value.get("theta_rad", 0.0))
        return (float(value["x_mm"]), float(value["y_mm"]), theta)

    def _parse_speed(self, value: Any) -> Tuple[float, float, float]:
        if isinstance(value, (list, tuple)) and len(value) == 3:
            return (float(value[0]), float(value[1]), float(value[2]))
        return (0.0, 0.0, 0.0)

    def _parse_bridge_result(self, result: dict,
                             agent_order: Optional[List[str]] = None) -> SidecarOutput:
        """Convert bridge JSON response to SidecarOutput."""
        # Parse activation graph
        activation_graph = []
        for node_dict in result.get("activation_graph", []):
            activation_graph.append(ActivationNode(
                option=node_dict.get("option", ""),
                depth=node_dict.get("depth", 0),
                state=node_dict.get("state", ""),
                option_time=node_dict.get("option_time", 0),
                state_time=node_dict.get("state_time", 0),
                parameters=node_dict.get("parameters", []),
            ))

        skill = result.get("skill_request", "none")
        motion_request = result.get("motion_request", "stand")
        macro_type = infer_real_macro_type(
            skill=skill,
            motion_request=motion_request,
            activation_graph=activation_graph,
        )

        # Determine active_agent from player number
        active_number = result.get("active_agent_number")
        active_agent = self._active_agent_from_number(active_number, agent_order or [])
        motion_speed = self._parse_speed(result.get("motion_speed", [0.0, 0.0, 0.0]))
        motion_executed_speed = self._parse_speed(
            result.get("motion_executed_speed", [0.0, 0.0, 0.0])
        )

        # Preserve bridge debug/provenance data for end-to-end env smoke tests.
        bridge_metadata = {
            key: value
            for key, value in result.items()
            if key.startswith("debug_") or key == "decision_time_us"
        }

        return SidecarOutput(
            active_agent=active_agent,
            macro_type=macro_type,
            strategy_status=result.get("strategy_status", "s4v4_scaled"),
            current_tactic=normalize_tactic_name(
                result.get("current_tactic", "t121_4v4_scaled")
            ),
            skill_request=skill,
            skill_target=self._parse_xy(result.get("skill_target_pose")),
            skill_target_direction=result.get("skill_target_direction_rad"),
            behavior_status=result.get("behavior_status", "playing"),
            behavior_speed=float(result.get("behavior_speed_mm_s", 0.0)),
            behavior_walking_to=self._parse_xy(result.get("behavior_walking_to")),
            behavior_shooting_to=self._parse_xy(result.get("behavior_shooting_to")),
            pass_target_player_number=result.get("pass_target_player_number"),
            pass_origin_player_number=result.get("pass_origin_player_number"),
            motion_request=motion_request,
            motion_speed=motion_speed,
            motion_walk_target=self._parse_pose(result.get("motion_walk_target")),
            motion_target_direction=result.get("motion_target_direction_rad"),
            motion_executed_phase=result.get("motion_executed_phase", "unavailable"),
            motion_executed_speed=motion_executed_speed,
            motion_ball_target_field=self._parse_xy(result.get("motion_ball_target_field")),
            motion_contact_point_field=self._parse_xy(result.get("motion_contact_point_field")),
            robot_pose_field=self._parse_pose(result.get("robot_pose_field")),
            field_ball_position=self._parse_xy(result.get("field_ball_position")),
            field_ball_intercepted_position=self._parse_xy(result.get("field_ball_intercepted_position")),
            kick_type=result.get("kick_type", "none"),
            kick_length=result.get("kick_length_mm", 0.0),
            kick_length_raw=result.get("kick_length_mm_raw", result.get("kick_length_mm", 0.0)),
            last_kick_type=result.get("last_kick_type", "unavailable"),
            last_kick_timestamp=result.get("last_kick_timestamp_ms", 0),
            activation_graph=activation_graph,
            backend="real",
            contract_validation_mode=False,
            bridge_mode=result.get("debug_bridge_mode", self._bridge_mode),
            bridge_frame_id=result.get("frame_id", -1),
            bridge_player_number=result.get("player_number", -1),
            bridge_metadata=bridge_metadata,
            decision_time_ms=result.get("decision_time_us", 0) / 1000.0,
        )


# ---------------------------------------------------------------------------
# STRATEGY EVALUATION — tactic selection (s4v4_scaled FSM)
# ---------------------------------------------------------------------------

class StrategyEvaluator:
    """Evaluates strategy FSM transitions. Source: s4v4_scaled.cfg."""

    def __init__(self, contract: AllConfigs):
        self._strategy = contract.strategy
        self._current_tactic = "t121_4v4_scaled"
        self._time_ball_ahead_ms = 0
        self._time_ball_behind_ms = 0

    def reset(self):
        self._current_tactic = "t121_4v4_scaled"
        self._time_ball_ahead_ms = 0
        self._time_ball_behind_ms = 0

    def evaluate(self, frame: GroundTruthFrame, dt_ms: float = 33.0) -> str:
        ball_x = float(frame.ball_pos[0])
        num_field_players = sum(1 for u in frame.own_upright.values() if u)

        for ts in self._strategy.tactics:
            if ts.tactic != self._current_tactic:
                continue
            for tr in ts.transitions:
                for cond in tr.conditions:
                    if cond.ball_x_threshold is not None:
                        if ball_x > cond.ball_x_threshold:
                            self._time_ball_ahead_ms += dt_ms
                            self._time_ball_behind_ms = 0
                        else:
                            self._time_ball_behind_ms += dt_ms
                            self._time_ball_ahead_ms = 0

        for ts in self._strategy.tactics:
            if ts.tactic != self._current_tactic:
                continue
            for tr in ts.transitions:
                if self._check_transition(tr, num_field_players):
                    self._current_tactic = tr.to
                    self._time_ball_ahead_ms = 0
                    self._time_ball_behind_ms = 0
                    break
            break

        return self._current_tactic

    def _check_transition(self, tr, num_field_players: int) -> bool:
        for cond in tr.conditions:
            if cond.num_field_players_ge is not None:
                if num_field_players < cond.num_field_players_ge:
                    return False
            if cond.num_field_players_le is not None:
                if num_field_players > cond.num_field_players_le:
                    return False
            if cond.time_ball_ahead_ge is not None:
                if self._time_ball_ahead_ms < cond.time_ball_ahead_ge:
                    return False
            if cond.time_ball_behind_ge is not None:
                if self._time_ball_behind_ms < cond.time_ball_behind_ge:
                    return False
        return True


# ---------------------------------------------------------------------------
# PLAY BALL DECISION — smashOrPass logic (STUB ONLY)
# ---------------------------------------------------------------------------

class PlayBallDecider:
    """
    Python re-implementation of PlayBall::smashOrPass using parsed CFGs.
    THIS IS A STUB for smoke/debug. Not authoritative for real BHuman.

    Key semantic rules:
      - skill_request never contains "duel"
      - duel is detected via Zweikampf sub-state and reported in activation_graph
      - macro_type = playBall_duel comes from activation_graph, not skill_request
    """

    def __init__(self, contract: AllConfigs):
        self._sop = contract.smash_or_pass
        self._kag = contract.kick_at_goal
        self._pass = contract.pass_to_teammate
        self._zk = contract.zweikampf
        self._kick_info = contract.kick_info
        self._field = contract.field

        self._last_decision: str = "none"
        self._last_decision_time_ms: float = 0.0
        self._current_time_ms: float = 0.0

    def reset(self):
        self._last_decision = "none"
        self._last_decision_time_ms = 0.0
        self._current_time_ms = 0.0

    def decide(self, frame: GroundTruthFrame, active_agent_id: str,
               dt_ms: float = 33.0
               ) -> Tuple[str, str, str, float, List[ActivationNode]]:
        """
        Decide macro action for the active agent.

        Returns: (macro_type, skill_request, kick_type, kick_length, activation_nodes)

        NOTE: skill_request is a real BHuman enum value (never "duel").
              macro_type = playBall_duel is derived from activation_graph zweikampf state.
        """
        self._current_time_ms += dt_ms
        graph: List[ActivationNode] = [
            ActivationNode(option="PlayBall", depth=0, state="execute",
                           option_time=int(self._current_time_ms)),
        ]

        if active_agent_id not in frame.own_poses:
            return "none", "none", "none", 0.0, graph

        agent_pos = frame.own_poses[active_agent_id]

        # 1. Check for duel conditions (opponent close — Zweikampf sub-state)
        duel_needed = self._check_duel_condition(frame, agent_pos)

        if duel_needed:
            # Zweikampf is a SUB-STATE of PlayBall, not a SkillRequest.
            # skill_request remains "dribble" (the skill used during duel).
            # macro_type = playBall_duel from activation_graph state.
            graph.append(ActivationNode(
                option="Zweikampf", depth=1, state="zweikampf",
                state_time=int(dt_ms),
                parameters=["opponent_blocking=true"],
            ))
            return ("playBall_duel", "dribble", "walkForwardStealBallLeft",
                    0.0, graph)

        # 2. Compute expected goals (goal opening angle)
        goal_rating = self._compute_goal_rating(agent_pos)

        # 3. Compute pass rating
        pass_rating, pass_target = self._compute_pass_rating(
            frame, active_agent_id, agent_pos
        )

        # 4. Hysteresis check
        time_since_last = self._current_time_ms - self._last_decision_time_ms
        hysteresis_active = time_since_last < self._kag.hysteresis_number

        # 5. Decision logic (mirrors smashOrPass)
        if goal_rating >= self._sop.shoot_threshold:
            macro = "playBall_shoot"
            skill = "shoot"
            kick = "forwardFastLeftLong"
            length = self._kick_info.forward_fast_max
            graph.append(ActivationNode(
                option="KickAtGoal", depth=1, state="shoot",
                parameters=[f"goalRating={goal_rating:.3f}"],
            ))
        elif pass_rating >= self._pass.rating_threshold:
            if hysteresis_active and self._last_decision == "playBall_shoot":
                macro = "playBall_shoot"
                skill = "shoot"
                kick = "forwardFastLeftLong"
                length = self._kick_info.forward_fast_max
                graph.append(ActivationNode(
                    option="KickAtGoal", depth=1, state="shoot_hysteresis",
                ))
            else:
                macro = "playBall_pass"
                skill = "pass"
                kick = "forwardFastRightPass"
                length = min(
                    float(np.linalg.norm(
                        pass_target - agent_pos
                    )) if pass_target is not None else 2000.0,
                    self._sop.max_pass_distance
                )
                graph.append(ActivationNode(
                    option="PassToTeammate", depth=1, state="pass",
                    parameters=[f"passRating={pass_rating:.3f}"],
                ))
        else:
            macro = "playBall_dribble"
            skill = "dribble"
            kick = "walkForwardsLeft"
            length = self._kick_info.walk_forward_max
            graph.append(ActivationNode(
                option="HandleBall", depth=1, state="dribble",
                parameters=["fallback=true"],
            ))

        if macro != self._last_decision:
            self._last_decision_time_ms = self._current_time_ms
        self._last_decision = macro

        return macro, skill, kick, length, graph

    def _compute_goal_rating(self, agent_pos: np.ndarray) -> float:
        goal_left = np.array([self._field.x_opponent_ground, self._field.y_left_goal])
        goal_right = np.array([self._field.x_opponent_ground, self._field.y_right_goal])

        angle_left = math.atan2(goal_left[1] - agent_pos[1],
                                goal_left[0] - agent_pos[0])
        angle_right = math.atan2(goal_right[1] - agent_pos[1],
                                 goal_right[0] - agent_pos[0])
        opening_angle = abs(angle_left - angle_right)
        opening_deg = math.degrees(opening_angle)

        if opening_deg < self._kag.min_opening_angle:
            return 0.0
        return min(1.0, opening_deg / 45.0)

    def _compute_pass_rating(self, frame: GroundTruthFrame,
                             active_id: str,
                             agent_pos: np.ndarray
                             ) -> Tuple[float, Optional[np.ndarray]]:
        best_rating = 0.0
        best_target = None

        for aid, pos in frame.own_poses.items():
            if aid == active_id:
                continue
            if not frame.own_upright.get(aid, False):
                continue

            dist = float(np.linalg.norm(pos - agent_pos))
            if dist < self._sop.min_pass_distance or dist > self._sop.max_pass_distance:
                continue

            to_tm = pos - agent_pos
            angle_to_tm = math.degrees(math.atan2(to_tm[1], to_tm[0]))
            agent_theta_deg = math.degrees(frame.own_thetas.get(active_id, 0.0))
            angle_offset = abs(angle_to_tm - agent_theta_deg)
            if angle_offset > 180:
                angle_offset = 360 - angle_offset

            if angle_offset < self._pass.min_angle_offset:
                continue
            if angle_offset > self._pass.max_angle_offset:
                continue

            blocked = False
            for opp_pos in frame.opp_poses:
                opp_dist = self._point_line_distance(agent_pos, pos, opp_pos)
                if opp_dist < self._pass.ignore_obstacles_threshold:
                    blocked = True
                    break

            if blocked:
                continue

            forward_progress = (pos[0] - agent_pos[0]) / self._field.length
            dist_rating = 1.0 - abs(dist - 2500.0) / self._sop.max_pass_distance
            rating = 0.5 * max(0, forward_progress) + 0.5 * max(0, dist_rating)

            if rating > best_rating:
                best_rating = rating
                best_target = pos.copy()

        return best_rating, best_target

    def _check_duel_condition(self, frame: GroundTruthFrame,
                              agent_pos: np.ndarray) -> bool:
        for opp_pos in frame.opp_poses:
            if not isinstance(opp_pos, np.ndarray):
                continue
            dist = float(np.linalg.norm(opp_pos - agent_pos))
            if dist < self._zk.obstacle_handling.obstacle_blocking_radius * 2:
                return True
        return False

    @staticmethod
    def _point_line_distance(line_start: np.ndarray, line_end: np.ndarray,
                             point: np.ndarray) -> float:
        line_vec = line_end - line_start
        line_len = float(np.linalg.norm(line_vec))
        if line_len < 1e-6:
            return float(np.linalg.norm(point - line_start))
        line_unit = line_vec / line_len
        proj = float(np.dot(point - line_start, line_unit))
        proj = max(0, min(line_len, proj))
        closest = line_start + proj * line_unit
        return float(np.linalg.norm(point - closest))


# ---------------------------------------------------------------------------
# ACTIVE AGENT SELECTOR (STUB — nearest-to-ball heuristic)
# ---------------------------------------------------------------------------

class ActiveAgentSelector:
    """
    STUB: Selects active agent by nearest-to-ball heuristic.
    Real BHuman uses Behavior::determineActiveAgent() which considers
    time-to-reach, role priorities, and team coordination.
    This stub is acceptable ONLY in python_stub_smoke_only or
    contract_validation_mode.
    """

    def __init__(self, contract: AllConfigs):
        self._contract = contract

    def select(self, frame: GroundTruthFrame) -> Optional[str]:
        if not frame.is_playing:
            return None

        best_id = None
        best_dist = float("inf")
        for aid, pos in frame.own_poses.items():
            if not frame.own_upright.get(aid, False):
                continue
            dist = float(np.linalg.norm(pos - frame.ball_pos))
            if dist < best_dist:
                best_dist = dist
                best_id = aid
        return best_id


# ---------------------------------------------------------------------------
# SKILL -> MOTION MAPPER (STUB)
# ---------------------------------------------------------------------------

class SkillToMotionMapper:
    """
    Maps skill requests to MotionRequest using real BHuman enum values.
    See: MotionRequest.h
    """

    def map(self, macro_type: str, skill_request: str,
            agent_pos: np.ndarray, ball_pos: np.ndarray,
            agent_theta: float
            ) -> Tuple[str, Tuple[float, float, float]]:
        """Returns (motion_request, (vx, vy, vtheta)) using real enum."""
        if skill_request == "shoot":
            return "walkToBallAndKick", (0.0, 0.0, 0.0)
        elif skill_request == "pass":
            return "walkToBallAndKick", (0.0, 0.0, 0.0)
        elif skill_request == "dribble":
            to_ball = ball_pos - agent_pos
            dist = float(np.linalg.norm(to_ball))
            if dist > 1e-6:
                direction = to_ball / dist
                vx = direction[0] * 250.0
                vy = direction[1] * 250.0
            else:
                vx, vy = 0.0, 0.0
            return "dribble", (vx, vy, 0.0)
        elif macro_type == "playBall_duel":
            # Zweikampf uses dribble motion
            to_ball = ball_pos - agent_pos
            dist = float(np.linalg.norm(to_ball))
            if dist > 1e-6:
                direction = to_ball / dist
                vx = direction[0] * 250.0
                vy = direction[1] * 250.0
            else:
                vx, vy = 0.0, 0.0
            return "dribble", (vx, vy, 0.0)
        else:
            return "stand", (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# PYTHON CONTRACT STUB — for smoke tests and debug ONLY
# ---------------------------------------------------------------------------

class PythonContractStub:
    """
    Python re-implementation of BHuman decision chain using parsed CFGs.
    THIS IS NOT BHuman IN LOOP. It is a contractual stub for:
      - smoke tests
      - debug
      - contract validation mode (when no real backend exists)

    It MUST NOT be treated as authoritative BHuman execution.
    """

    def __init__(self, contract: AllConfigs):
        self._contract = contract
        self._strategy_eval = StrategyEvaluator(contract)
        self._play_ball = PlayBallDecider(contract)
        self._agent_selector = ActiveAgentSelector(contract)
        self._skill_mapper = SkillToMotionMapper()

    def reset(self):
        self._strategy_eval.reset()
        self._play_ball.reset()

    def decide(self, frame: GroundTruthFrame,
               backend_label: str = "python_stub") -> SidecarOutput:
        t0 = time.perf_counter()

        if not frame.is_playing:
            return SidecarOutput(
                behavior_status="not_playing",
                activation_graph=[
                    ActivationNode(option="GameState", depth=0,
                                   state="not_playing"),
                ],
                backend=backend_label,
                contract_validation_mode=(backend_label == "contract_validation"),
                bridge_mode="persistent_2d_cognition",
                decision_time_ms=(time.perf_counter() - t0) * 1000,
            )

        current_tactic = self._strategy_eval.evaluate(frame)
        active_agent = self._agent_selector.select(frame)

        macro_type = "none"
        kick_type = "none"
        kick_length = 0.0
        skill_request = "none"
        activation_graph: List[ActivationNode] = [
            ActivationNode(option="Strategy", depth=0, state=current_tactic),
        ]

        if active_agent is not None:
            macro_type, skill_request, kick_type, kick_length, pb_graph = \
                self._play_ball.decide(frame, active_agent)
            activation_graph.extend(pb_graph)

        # Skill -> Motion
        motion_type = "stand"
        motion_speed = (0.0, 0.0, 0.0)
        if active_agent is not None and active_agent in frame.own_poses:
            motion_type, motion_speed = self._skill_mapper.map(
                macro_type, skill_request,
                frame.own_poses[active_agent],
                frame.ball_pos,
                frame.own_thetas.get(active_agent, 0.0),
            )

        activation_graph.append(ActivationNode(
            option="SkillToMotion", depth=1,
            state=f"{skill_request}_to_{motion_type}",
        ))

        dt_ms = (time.perf_counter() - t0) * 1000

        return SidecarOutput(
            active_agent=active_agent,
            macro_type=macro_type,
            strategy_status=self._contract.strategy.name,
            current_tactic=current_tactic,
            skill_request=skill_request,
            behavior_status="playing",
            motion_request=motion_type,
            motion_speed=motion_speed,
            kick_type=kick_type,
            kick_length=kick_length,
            # last_kick_type: NO real MotionInfo evidence exists in stub.
            last_kick_type="unavailable",
            last_kick_timestamp=0,
            activation_graph=activation_graph,
            backend=backend_label,
            contract_validation_mode=(backend_label == "contract_validation"),
            bridge_mode="persistent_2d_cognition",
            decision_time_ms=dt_ms,
        )


# ---------------------------------------------------------------------------
# MAIN SIDECAR CLASS — dispatcher
# ---------------------------------------------------------------------------

class BHumanDecisionSidecar:
    """
    BHuman decision sidecar dispatcher.

    In "bhuman_sidecar" mode:
      - Tries RealSidecarBackend first.
      - If unavailable, falls into "contract_validation_mode" using
        PythonContractStub but with explicit marking.
      - The env MUST NOT use BHumanTakeoverSimulator.

    In "python_stub_smoke_only" mode:
      - Uses PythonContractStub directly, marked as "python_stub".
    """

    def __init__(self, contract: AllConfigs,
                 execution_mode: str = "bhuman_sidecar"):
        self._contract = contract
        self._execution_mode = execution_mode
        self._real_backend = None
        self._stub = PythonContractStub(contract)
        self._last_real_backend_error: Optional[str] = None
        self._last_known_tactic: str = "t121_4v4_scaled"

        # Try to load real backend
        if execution_mode == "bhuman_sidecar":
            try:
                self._real_backend = RealSidecarBackend(contract)
            except NotImplementedError:
                # No real backend yet — contract_validation_mode
                self._real_backend = None

        # Determine effective backend label
        if self._real_backend is not None:
            self._backend_label = "real"
        elif execution_mode == "bhuman_sidecar":
            self._backend_label = "contract_validation"
        else:
            self._backend_label = "python_stub"

    @property
    def backend_label(self) -> str:
        return self._backend_label

    @property
    def is_real_backend(self) -> bool:
        return self._real_backend is not None

    @property
    def is_contract_validation_mode(self) -> bool:
        return (self._backend_label == "contract_validation")

    @property
    def uses_takeover_simulator(self) -> bool:
        """Real sidecar and contract_validation mode do NOT use takeover sim."""
        return False

    def reset(self):
        if self._real_backend is not None:
            self._real_backend.reset()
        self._stub.reset()
        self._last_known_tactic = "t121_4v4_scaled"

    def shutdown(self):
        """Shutdown real backend subprocess if running."""
        if self._real_backend is not None:
            self._real_backend.shutdown()

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

    def decide(self,
               agents: Dict[str, Any],
               ball: Any,
               opponents: list,
               game: Any,
               active_agent_hint: Optional[str] = None,
               prefer_single_bridge_fast_path: bool = False) -> SidecarOutput:
        frame = self._build_frame(agents, ball, opponents, game)
        return self.decide_from_frame(
            frame,
            active_agent_hint=active_agent_hint,
            prefer_single_bridge_fast_path=prefer_single_bridge_fast_path,
        )

    def decide_from_frame(self, frame: GroundTruthFrame,
                          active_agent_hint: Optional[str] = None,
                          prefer_single_bridge_fast_path: bool = False) -> SidecarOutput:
        if self._real_backend is not None:
            try:
                output = self._real_backend.decide(
                    frame,
                    active_agent_hint=active_agent_hint,
                    prefer_single_bridge_fast_path=prefer_single_bridge_fast_path,
                )
                if output.current_tactic not in (None, "", "none"):
                    self._last_known_tactic = str(output.current_tactic)
                return output
            except NotImplementedError as exc:
                self._last_real_backend_error = str(exc)
                self._real_backend = None
                self._backend_label = "contract_validation"
        output = self._stub.decide(frame, backend_label=self._backend_label)
        if output.current_tactic not in (None, "", "none"):
            self._last_known_tactic = str(output.current_tactic)
        return output

    def _build_frame(self, agents, ball, opponents, game) -> GroundTruthFrame:
        own_poses = {}
        own_thetas = {}
        own_upright = {}
        own_roles = {}

        for aid, agent in agents.items():
            own_poses[aid] = agent.pos.copy()
            own_thetas[aid] = agent.theta
            own_upright[aid] = agent.is_upright
            own_roles[aid] = agent.role

        self._augment_missing_teammates(own_poses, own_thetas, own_upright, own_roles)

        opp_poses = []
        opp_upright = []
        for opp in opponents:
            opp_poses.append(opp.pos.copy())
            opp_upright.append(opp.is_upright)

        return GroundTruthFrame(
            ball_pos=ball.pos.copy(),
            ball_vel=ball.vel.copy(),
            ball_seen_age_ms=ball.seen_age_ms,
            own_poses=own_poses,
            own_thetas=own_thetas,
            own_upright=own_upright,
            own_roles=own_roles,
            opp_poses=opp_poses,
            opp_upright=opp_upright,
            is_playing=game.is_playing,
            own_score=game.own_score,
            opp_score=game.opp_score,
        )

    def _augment_missing_teammates(self,
                                   own_poses: Dict[str, np.ndarray],
                                   own_thetas: Dict[str, float],
                                   own_upright: Dict[str, bool],
                                   own_roles: Dict[str, str]) -> None:
        """
        Keep sidecar context in canonical 4-player form even in early curriculum.

        Curriculum can still train 1/2/3 RL-controlled agents, but the BHuman
        sidecar should see a stable 4v4-style team snapshot. Missing teammates
        are represented with tactic home poses so tactical reasoning does not
        collapse to a reduced-team artifact.
        """
        tactic_name = self._resolve_padding_tactic()
        tactic_home_poses = get_home_poses(tactic_name, self._contract)
        fallback_home_poses = get_home_poses("t121_4v4_scaled", self._contract)

        for idx, aid in enumerate(CANONICAL_AGENT_IDS):
            if aid in own_poses:
                continue
            role = CANONICAL_ROLE_ORDER[idx]
            home = tactic_home_poses.get(role, fallback_home_poses.get(role, (0.0, 0.0)))
            own_poses[aid] = np.array(home, dtype=np.float32)
            own_thetas[aid] = 0.0
            own_upright[aid] = True
            own_roles[aid] = role

    def _resolve_padding_tactic(self) -> str:
        current = normalize_tactic_name(self._last_known_tactic or "t121_4v4_scaled")
        if current in self._contract.tactics:
            return current
        return "t121_4v4_scaled"
