"""
LEGACY REDIRECT

Archivo legado. La version viva esta en:
`smoke_legacy.smoke_test_real_env_semantic_variants`.
Usar la superficie oficial: `smoke_diagnostic_extended.py` o `smoke_suite_option1.py`.
"""

"""
smoke_test_real_env_semantic_variants.py -- Real env.step semantic variants.

Purpose:
  1. Prove a stable positive duel takeover inside env.step().
  2. Prove stable real dribble semantics inside env.step() without any
     contract-validation fallback or info leakage.
  3. Prove stable real pass semantics inside env.step().

This smoke complements:
  - smoke_test_real_takeover_env_positive.py   (positive shoot takeover)
  - smoke_test_real_takeover_playball.py       (real bridge playBall probes)
  - smoke_test_semantic_strong.py              (contract semantics)

Important nuance:
  - duel is validated here as a full positive takeover overlap
  - dribble and pass are validated here as real env.step semantic outputs
    from the bridge, even when the handoff envelope is not active on the
    same frame
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

os.environ["RL_EXECUTION_MODE"] = "bhuman_sidecar"

from bridge.bridge_protocol import BridgeConfig
from bhuman_sidecar import GroundTruthFrame, normalize_tactic_name
from config import CURRICULUM, resolve_cfg_root
from environment import NaoSoccerEnv
from smoke_test_real_takeover_playball import _make_real_pass_frame


ROLES = {
    "agent_0": "forwardR",
    "agent_1": "forwardL",
    "agent_2": "defenderR",
    "agent_3": "defenderL",
}

DUEL_REPLAY_SEED = 314
DUEL_REPLAY_STEPS = 173

DRIBBLE_REPLAY_SEED = 314
DRIBBLE_TARGET_INDEX = 185
DRIBBLE_AGENT_OFFSET = (-140.0, -300.0)


def _assert(condition: bool, msg: str) -> None:
    if not condition:
        print(f"  FAIL: {msg}")
        raise AssertionError(msg)
    print(f"  OK: {msg}")


def _zero_actions(env: NaoSoccerEnv):
    return {
        aid: np.zeros(env.action_space(aid).shape, dtype=np.float32)
        for aid in env.agents
    }


def _generate_replay_frames(seed: int, count: int) -> list[GroundTruthFrame]:
    rng = np.random.default_rng(seed)
    frames: list[GroundTruthFrame] = []
    for _ in range(count):
        ball = np.array(
            [rng.uniform(-2500, 2500), rng.uniform(-1800, 1800)],
            dtype=np.float32,
        )
        own_poses = {
            aid: np.array(
                [rng.uniform(-2500, 2500), rng.uniform(-1800, 1800)],
                dtype=np.float32,
            )
            for aid in ROLES
        }
        frames.append(
            GroundTruthFrame(
                ball_pos=ball,
                ball_vel=np.zeros(2, dtype=np.float32),
                ball_seen_age_ms=0.0,
                own_poses=own_poses,
                own_thetas={aid: 0.0 for aid in ROLES},
                own_upright={aid: True for aid in ROLES},
                own_roles=ROLES,
                opp_poses=[
                    np.array(
                        [rng.uniform(-2500, 2500), rng.uniform(-1800, 1800)],
                        dtype=np.float32,
                    )
                    for _ in range(4)
                ],
                opp_upright=[True] * 4,
                is_playing=True,
                own_score=0,
                opp_score=0,
            )
        )
    return frames


def _build_dribble_replay_frames() -> list[GroundTruthFrame]:
    frames = _generate_replay_frames(DRIBBLE_REPLAY_SEED, DRIBBLE_TARGET_INDEX + 1)
    base = frames[DRIBBLE_TARGET_INDEX]
    own_poses = {aid: pos.copy() for aid, pos in base.own_poses.items()}
    own_poses["agent_1"] = base.ball_pos + np.array(DRIBBLE_AGENT_OFFSET, dtype=np.float32)
    frames[DRIBBLE_TARGET_INDEX] = GroundTruthFrame(
        ball_pos=base.ball_pos.copy(),
        ball_vel=base.ball_vel.copy(),
        ball_seen_age_ms=base.ball_seen_age_ms,
        own_poses=own_poses,
        own_thetas=base.own_thetas.copy(),
        own_upright=base.own_upright.copy(),
        own_roles=base.own_roles.copy(),
        opp_poses=[pos.copy() for pos in base.opp_poses],
        opp_upright=list(base.opp_upright),
        is_playing=True,
        own_score=0,
        opp_score=0,
    )
    return frames


def _install_frame(env: NaoSoccerEnv, frame: GroundTruthFrame) -> None:
    env._ball.pos[:] = frame.ball_pos.astype(np.float32)
    env._ball.vel[:] = frame.ball_vel.astype(np.float32)
    env._ball.seen_age_ms = float(frame.ball_seen_age_ms)

    for aid in env.agents:
        agent = env._agents_state[aid]
        agent.pos = frame.own_poses[aid].astype(np.float32).copy()
        agent.theta = float(frame.own_thetas[aid])
        agent.vel = np.zeros(2, dtype=np.float32)
        agent.omega = 0.0
        agent.is_upright = bool(frame.own_upright[aid])
        agent.walk_target = agent.pos.copy()
        agent.speed_scalar = 0.0
        agent.role = frame.own_roles[aid]

    for idx, opp in enumerate(env._opponents_state):
        opp.pos = frame.opp_poses[idx].astype(np.float32).copy()
        opp.theta = math.pi
        opp.vel = np.zeros(2, dtype=np.float32)
        opp.omega = 0.0
        opp.is_upright = True

    env._game.is_playing = True
    env._game.own_score = 0
    env._game.opp_score = 0


def _run_replay(frames: list[GroundTruthFrame], seed: int) -> dict:
    env = NaoSoccerEnv(stage_cfg=CURRICULUM[12], use_fov=False, seed=seed)
    try:
        env.reset(seed=seed)
        last_info = None
        for frame in frames:
            _install_frame(env, frame)
            _, _, _, _, infos = env.step(_zero_actions(env))
            last_info = next(iter(infos.values()))
        return last_info or {}
    finally:
        env.close()


def _run_pass_trial(seed: int) -> dict:
    env = NaoSoccerEnv(stage_cfg=CURRICULUM[12], use_fov=False, seed=seed)
    try:
        env.reset(seed=seed)
        _install_frame(env, _make_real_pass_frame())
        last_info = None
        for _ in range(3):
            _, _, _, _, infos = env.step(_zero_actions(env))
            last_info = next(iter(infos.values()))
            if last_info.get("sidecar_macro_type") == "playBall_pass":
                return last_info
        return last_info or {}
    finally:
        env.close()


def test_real_env_duel_takeover() -> None:
    print("\n[TEST] Real duel takeover inside env.step()")
    frames = _generate_replay_frames(DUEL_REPLAY_SEED, DUEL_REPLAY_STEPS)

    for trial_idx in range(3):
        info = _run_replay(frames, seed=DUEL_REPLAY_SEED)
        _assert(info.get("execution_mode") == "bhuman_sidecar",
                f"execution_mode = bhuman_sidecar on trial {trial_idx + 1}")
        _assert(info.get("sidecar_backend") == "real",
                f"sidecar_backend = real on trial {trial_idx + 1} "
                f"(got {info.get('sidecar_backend')})")
        _assert(info.get("sidecar_current_tactic") == normalize_tactic_name(
                    info.get("sidecar_current_tactic")
                ),
                f"sidecar_current_tactic is normalized on trial {trial_idx + 1} "
                f"(got {info.get('sidecar_current_tactic')})")
        _assert(info.get("sidecar_macro_type") == "playBall_duel",
                f"sidecar_macro_type = playBall_duel on trial {trial_idx + 1} "
                f"(got {info.get('sidecar_macro_type')})")
        _assert(info.get("macro_type") == "playBall_duel",
                f"macro_type = playBall_duel on trial {trial_idx + 1} "
                f"(got {info.get('macro_type')})")
        _assert(info.get("handoff_ready") is True,
                f"handoff_ready = True on trial {trial_idx + 1}")
        _assert(info.get("takeover_active") is True,
                f"takeover_active = True on trial {trial_idx + 1}")
        _assert(info.get("active_agent_id") == "agent_2",
                f"active_agent_id = agent_2 on trial {trial_idx + 1} "
                f"(got {info.get('active_agent_id')})")
        _assert(info.get("sidecar_motion_request") == "walkToBallAndKick",
                f"sidecar_motion_request = walkToBallAndKick on trial {trial_idx + 1} "
                f"(got {info.get('sidecar_motion_request')})")


def test_real_env_dribble_semantics() -> None:
    print("\n[TEST] Real dribble semantics inside env.step()")
    frames = _build_dribble_replay_frames()

    for trial_idx in range(3):
        info = _run_replay(frames, seed=DRIBBLE_REPLAY_SEED)
        _assert(info.get("execution_mode") == "bhuman_sidecar",
                f"execution_mode = bhuman_sidecar on trial {trial_idx + 1}")
        _assert(info.get("sidecar_backend") == "real",
                f"sidecar_backend = real on trial {trial_idx + 1} "
                f"(got {info.get('sidecar_backend')})")
        _assert(info.get("sidecar_current_tactic") == normalize_tactic_name(
                    info.get("sidecar_current_tactic")
                ),
                f"sidecar_current_tactic is normalized on trial {trial_idx + 1} "
                f"(got {info.get('sidecar_current_tactic')})")
        _assert(info.get("sidecar_macro_type") == "playBall_dribble",
                f"sidecar_macro_type = playBall_dribble on trial {trial_idx + 1} "
                f"(got {info.get('sidecar_macro_type')})")
        _assert(info.get("macro_type") == "playBall_dribble",
                f"macro_type = playBall_dribble on trial {trial_idx + 1} "
                f"(got {info.get('macro_type')})")
        _assert(info.get("sidecar_motion_request") == "dribble",
                f"sidecar_motion_request = dribble on trial {trial_idx + 1} "
                f"(got {info.get('sidecar_motion_request')})")
        _assert(info.get("active_agent_id") == "agent_1",
                f"active_agent_id = agent_1 on trial {trial_idx + 1} "
                f"(got {info.get('active_agent_id')})")
        _assert(info.get("takeover_active") is False,
                f"takeover_active = False on trial {trial_idx + 1}")
        _assert(info.get("handoff_ready") is False,
                f"handoff_ready = False on trial {trial_idx + 1}")


def test_real_env_pass_semantics() -> None:
    print("\n[TEST] Real pass semantics inside env.step()")

    for trial_idx in range(3):
        info = _run_pass_trial(seed=21)
        _assert(info.get("execution_mode") == "bhuman_sidecar",
                f"execution_mode = bhuman_sidecar on trial {trial_idx + 1}")
        _assert(info.get("sidecar_backend") == "real",
                f"sidecar_backend = real on trial {trial_idx + 1} "
                f"(got {info.get('sidecar_backend')})")
        _assert(info.get("sidecar_current_tactic") == normalize_tactic_name(
                    info.get("sidecar_current_tactic")
                ),
                f"sidecar_current_tactic is normalized on trial {trial_idx + 1} "
                f"(got {info.get('sidecar_current_tactic')})")
        _assert(info.get("sidecar_macro_type") == "playBall_pass",
                f"sidecar_macro_type = playBall_pass on trial {trial_idx + 1} "
                f"(got {info.get('sidecar_macro_type')})")
        _assert(info.get("macro_type") == "playBall_pass",
                f"macro_type = playBall_pass on trial {trial_idx + 1} "
                f"(got {info.get('macro_type')})")
        _assert(info.get("active_agent_id") == "agent_2",
                f"active_agent_id = agent_2 on trial {trial_idx + 1} "
                f"(got {info.get('active_agent_id')})")
        _assert(info.get("takeover_active") is False,
                f"takeover_active = False on trial {trial_idx + 1}")
        _assert(info.get("handoff_ready") is False,
                f"handoff_ready = False on trial {trial_idx + 1}")


def main() -> None:
    print("[TEST] Real env.step semantic variants smoke")

    bridge_exe = BridgeConfig().bridge_executable
    _assert(Path(bridge_exe).exists(), f"bridge executable exists at {bridge_exe}")

    cfg_root = resolve_cfg_root()
    _assert(cfg_root is not None, f"real CFG root resolved (got {cfg_root})")

    test_real_env_duel_takeover()
    test_real_env_dribble_semantics()
    test_real_env_pass_semantics()

    print("\nRESULT: PASS")
    print("  env.step() preserves real bridge semantics for duel, dribble, and pass.")
    print("  duel reaches takeover overlap; dribble/pass stay semantically real without fallback.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nRESULT: FAIL")
        print(f"  {type(exc).__name__}: {exc}")
        raise SystemExit(1)
