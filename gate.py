"""Unified, stateful skill gate (dynamic forbid-invalid masking).

One place decides which skills are *valid* this step. Its output (armed flags +
debounce progress) is baked into the observation vector by the environment, so the
per-step mask the policy applies is a pure function of the stored observation and
MAPPO's update-time log-prob recompute reproduces it exactly (Design A — no rollout
buffer surgery). The same gate is meant to drive the teacher, the dataset labels and
the on-robot guardrail, which collapses the natural/exported perception split into a
single channel.

Semantics are strictly **forbid-invalid**: the gate only ever *removes* a skill that
is unsafe/impossible right now (shoot when not set up, dribble when not on the ball).
`walk`/`stand` are never removed, so the policy always has a safe fallback and the
mask can never trap it. The actual trigger pull for the kick comes from the BC prior,
not from the gate.

Robustness against sensor noise / chattering is built in:
  * hysteresis  — separate (tight) enter and (loose) exit envelopes per skill;
  * temporal buffering — the enter condition must hold N consecutive frames to arm;
  * uncertainty gating — arming requires fresh *natural* ball perception (Option B),
    not the exported/estimated ball the navigator may be chasing.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from action import SKILL_ORDER
from config import GOAL_X
from pybh_backend import BackendObservation
from reward import BALL_TRUSTED_MAX_AGE_MS, features_from_obs, goal_ball_distance_mm

# Skills this gate dynamically arms/disarms. Everything else (walk/stand/pass/...) is
# left to the stage mask; the gate never removes the walk/stand fallback.
_GATED_SKILLS = ("shoot", "dribble")


def _wrap_angle(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else int(default)


@dataclass(frozen=True)
class GateConfig:
    # --- shoot: robot "on the ball" envelope (enter tight, exit loose) ---
    shoot_enter_ball_mm: float = 380.0
    shoot_exit_ball_mm: float = 460.0
    shoot_enter_angle_rad: float = math.radians(25.0)
    shoot_exit_angle_rad: float = math.radians(35.0)
    # --- shoot: ball-to-goal range that permits a kick (mid-distance shots) ---
    # Conservative for NAO v6 kick lethality vs. weak RoboCup keepers: arm only
    # within 2.5 m, disarm past 2.9 m (hysteresis so a ball drifting near the edge
    # does not flip the kick on/off).
    shoot_goal_enter_mm: float = 2500.0
    shoot_goal_exit_mm: float = 2900.0
    # --- shoot: clear lane proxy (shot_opening_with_obstacles); score_now overrides ---
    shoot_open_enter: float = 0.6
    shoot_open_exit: float = 0.45
    # --- shoot: robot heading aligned toward the goal mouth (lethality safeguard) ---
    shoot_align_enter_rad: float = math.radians(30.0)
    shoot_align_exit_rad: float = math.radians(45.0)
    # --- shoot: temporal debounce (consecutive in-zone frames before arming) ---
    shoot_arm_frames: int = 3
    # --- dribble (takeover) envelope; wide exit == drive-commit latch ---
    dribble_enter_ball_mm: float = 450.0
    dribble_exit_ball_mm: float = 600.0
    dribble_enter_angle_rad: float = math.radians(35.0)
    dribble_exit_angle_rad: float = math.radians(55.0)
    dribble_arm_frames: int = 1
    # --- dribble grace latch (tiered trust) ---
    # walkTo's head does not reliably keep a ground ball in the camera, so the
    # robot goes ball-blind on the NATURAL channel while approaching. Requiring
    # *fresh* natural vision for dribble created a see->engage->see deadlock. The
    # grace window lets the robot navigate to / stay engaged with a *remembered*
    # ball (BHuman's BallModel propagates the last sighting): enter needs a natural
    # sighting within grace_enter; once armed, hold tolerates the longer grace_hold
    # (drive-commit latch surviving a head-scan gap). Sim-to-real safe — it trusts a
    # real natural sighting for a bounded age, NOT the GT-corrected channel. SHOOT
    # stays strict-fresh (trusted_max_age_ms) below: the kick must truly see the ball.
    dribble_grace_enter_age_ms: float = 1500.0
    dribble_grace_hold_age_ms: float = 2500.0
    # --- natural-perception trust (closes the exported/natural split) ---
    trusted_max_age_ms: float = BALL_TRUSTED_MAX_AGE_MS

    @classmethod
    def from_env(cls) -> "GateConfig":
        """Build with RL_GATE_* overrides for reproducible overnight runs."""
        base = cls()
        return cls(
            shoot_enter_ball_mm=_env_float("RL_GATE_SHOOT_ENTER_BALL_MM", base.shoot_enter_ball_mm),
            shoot_exit_ball_mm=_env_float("RL_GATE_SHOOT_EXIT_BALL_MM", base.shoot_exit_ball_mm),
            shoot_enter_angle_rad=base.shoot_enter_angle_rad,
            shoot_exit_angle_rad=base.shoot_exit_angle_rad,
            shoot_goal_enter_mm=_env_float("RL_GATE_SHOOT_GOAL_ENTER_MM", base.shoot_goal_enter_mm),
            shoot_goal_exit_mm=_env_float("RL_GATE_SHOOT_GOAL_EXIT_MM", base.shoot_goal_exit_mm),
            shoot_open_enter=_env_float("RL_GATE_SHOOT_OPEN_ENTER", base.shoot_open_enter),
            shoot_open_exit=_env_float("RL_GATE_SHOOT_OPEN_EXIT", base.shoot_open_exit),
            shoot_align_enter_rad=math.radians(
                _env_float("RL_GATE_SHOOT_ALIGN_ENTER_DEG", math.degrees(base.shoot_align_enter_rad))
            ),
            shoot_align_exit_rad=math.radians(
                _env_float("RL_GATE_SHOOT_ALIGN_EXIT_DEG", math.degrees(base.shoot_align_exit_rad))
            ),
            shoot_arm_frames=_env_int("RL_GATE_SHOOT_ARM_FRAMES", base.shoot_arm_frames),
            dribble_enter_ball_mm=_env_float("RL_GATE_DRIBBLE_ENTER_BALL_MM", base.dribble_enter_ball_mm),
            dribble_exit_ball_mm=_env_float("RL_GATE_DRIBBLE_EXIT_BALL_MM", base.dribble_exit_ball_mm),
            dribble_arm_frames=_env_int("RL_GATE_DRIBBLE_ARM_FRAMES", base.dribble_arm_frames),
            dribble_grace_enter_age_ms=_env_float(
                "RL_GATE_DRIBBLE_GRACE_ENTER_AGE_MS", base.dribble_grace_enter_age_ms
            ),
            dribble_grace_hold_age_ms=_env_float(
                "RL_GATE_DRIBBLE_GRACE_HOLD_AGE_MS", base.dribble_grace_hold_age_ms
            ),
            trusted_max_age_ms=_env_float("RL_GATE_TRUSTED_MAX_AGE_MS", base.trusted_max_age_ms),
        )


@dataclass
class GateState:
    shoot_armed: bool = False
    dribble_armed: bool = False
    shoot_streak: int = 0
    dribble_streak: int = 0


@dataclass(frozen=True)
class GateDecision:
    shoot_armed: bool
    dribble_armed: bool
    shoot_arm_progress: float  # [0,1]: 1.0 when armed, else debounce fill toward arming
    allowed_skills: tuple[str, ...]

    def obs_features(self) -> list[float]:
        """Ordered to match ObsIndex.{shoot_armed, dribble_armed, shoot_arm_progress}."""
        return [
            1.0 if self.shoot_armed else 0.0,
            1.0 if self.dribble_armed else 0.0,
            float(self.shoot_arm_progress),
        ]


def _allowed_skills(*, shoot_armed: bool, dribble_armed: bool) -> tuple[str, ...]:
    forbidden = set()
    if not shoot_armed:
        forbidden.add("shoot")
    if not dribble_armed:
        forbidden.add("dribble")
    # walk/stand are never forbidden: there is always a safe fallback.
    return tuple(name for name in SKILL_ORDER if name not in forbidden)


class SkillGate:
    """Per-episode stateful gate. Hold one per environment; reset() on episode reset."""

    def __init__(self, cfg: GateConfig | None = None):
        self.cfg = cfg or GateConfig.from_env()
        self.state = GateState()

    def reset(self) -> None:
        self.state = GateState()

    def step(self, obs: BackendObservation) -> GateDecision:
        cfg = self.cfg
        st = self.state
        feats = features_from_obs(obs)
        d_ball = feats.d_ball_mm
        a_ball = abs(feats.a_ball_rad)
        shot_open = feats.shot_open
        score_now = feats.score_now
        goal_dist = goal_ball_distance_mm(obs)

        # Uncertainty gate on the NATURAL channel (Option B). The exported/estimated
        # ball can be stale or hallucinated; arming on it is what caused the split.
        nat_age_ms = float(obs.natural_time_since_ball_seen)
        # SHOOT requires *fresh* natural vision: the kick only pulls when the robot
        # truly sees the ball right now.
        fresh = nat_age_ms <= cfg.trusted_max_age_ms
        # DRIBBLE / engagement uses the wider GRACE window so the robot can commit to
        # a *remembered* ball across brief perception dropouts (breaks the see->engage
        # ->see deadlock once the ball has been seen at least once). Enter is tighter
        # than hold so a single sighting commits the drive and survives the gap.
        engage_enter_fresh = nat_age_ms <= cfg.dribble_grace_enter_age_ms
        engage_hold_fresh = nat_age_ms <= cfg.dribble_grace_hold_age_ms

        # Robot heading vs. the goal-mouth centre (y=0): a long shot only arms when the
        # robot is actually pointed at goal, not merely near the ball.
        goal_bearing = math.atan2(0.0 - float(obs.robot_y), GOAL_X - float(obs.robot_x))
        align = abs(_wrap_angle(float(obs.robot_theta) - goal_bearing))

        # ---- shoot: hysteresis (enter tight / exit loose) + temporal debounce ----
        on_ball_enter = d_ball <= cfg.shoot_enter_ball_mm and a_ball <= cfg.shoot_enter_angle_rad
        on_ball_exit = d_ball <= cfg.shoot_exit_ball_mm and a_ball <= cfg.shoot_exit_angle_rad
        lane_enter = shot_open >= cfg.shoot_open_enter or score_now
        lane_exit = shot_open >= cfg.shoot_open_exit or score_now
        range_enter = goal_dist <= cfg.shoot_goal_enter_mm
        range_exit = goal_dist <= cfg.shoot_goal_exit_mm
        align_enter = align <= cfg.shoot_align_enter_rad
        align_exit = align <= cfg.shoot_align_exit_rad
        enter_cond = fresh and on_ball_enter and lane_enter and range_enter and align_enter
        hold_cond = fresh and on_ball_exit and lane_exit and range_exit and align_exit
        if not st.shoot_armed:
            st.shoot_streak = st.shoot_streak + 1 if enter_cond else 0
            if st.shoot_streak >= cfg.shoot_arm_frames:
                st.shoot_armed = True
        elif not hold_cond:
            st.shoot_armed = False
            st.shoot_streak = 0

        # ---- dribble (takeover): wide exit envelope + grace window == the drive-commit latch ----
        db_enter = engage_enter_fresh and d_ball <= cfg.dribble_enter_ball_mm and a_ball <= cfg.dribble_enter_angle_rad
        db_hold = engage_hold_fresh and d_ball <= cfg.dribble_exit_ball_mm and a_ball <= cfg.dribble_exit_angle_rad
        if not st.dribble_armed:
            st.dribble_streak = st.dribble_streak + 1 if db_enter else 0
            if st.dribble_streak >= cfg.dribble_arm_frames:
                st.dribble_armed = True
        elif not db_hold:
            st.dribble_armed = False
            st.dribble_streak = 0

        if st.shoot_armed:
            progress = 1.0
        else:
            progress = min(1.0, st.shoot_streak / max(1, cfg.shoot_arm_frames))

        return GateDecision(
            shoot_armed=st.shoot_armed,
            dribble_armed=st.dribble_armed,
            shoot_arm_progress=progress,
            allowed_skills=_allowed_skills(shoot_armed=st.shoot_armed, dribble_armed=st.dribble_armed),
        )
