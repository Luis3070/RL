"""
smoke_test_real_takeover_env_positive.py -- Positive end-to-end takeover smoke.

Purpose:
  Validate a true positive path inside the real environment loop:

    handoff_ready -> takeover_active -> playBall_* -> real BHuman skill

  This complements smoke_test_real_takeover_playball.py:
  - that smoke proves real playBall semantics exist and false takeovers are blocked
  - this smoke proves the positive overlap exists inside env.step()

The fixture below was found by searching a full 4v4 real bridge snapshot and is
executed with a fresh environment per trial for stability.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

os.environ["RL_EXECUTION_MODE"] = "bhuman_sidecar"

from bridge.bridge_protocol import BridgeConfig
from config import CURRICULUM, resolve_cfg_root
from bhuman_sidecar import normalize_tactic_name
from environment import NaoSoccerEnv


POSITIVE_FIXTURE = {
    "ball": [1823.082877532614, -569.9870937327996],
    "own": {
        "agent_0": [1649.5728759765625, -691.7743530273438],
        "agent_1": [1492.6826171875, -492.23931884765625],
        "agent_2": [1474.525634765625, -989.142822265625],
        "agent_3": [-1854.65478515625, -329.1419372558594],
    },
    "opp": [
        [1349.82080078125, 12.852691650390625],
        [1042.45751953125, -722.7896118164062],
        [2039.8150634765625, 107.85955047607422],
        [1352.7049560546875, 214.23764038085938],
    ],
}


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


def _install_positive_fixture(env: NaoSoccerEnv) -> None:
    env._ball.pos[:] = np.array(POSITIVE_FIXTURE["ball"], dtype=np.float32)
    env._ball.vel[:] = 0.0
    env._ball.seen_age_ms = 0.0

    for aid in env.agents:
        agent = env._agents_state[aid]
        agent.pos = np.array(POSITIVE_FIXTURE["own"][aid], dtype=np.float32)
        agent.theta = 0.0
        agent.vel = np.zeros(2, dtype=np.float32)
        agent.omega = 0.0
        agent.is_upright = True
        agent.walk_target = agent.pos.copy()
        agent.speed_scalar = 0.0

    for idx, opp in enumerate(env._opponents_state):
        opp.pos = np.array(POSITIVE_FIXTURE["opp"][idx], dtype=np.float32)
        opp.theta = math.pi
        opp.vel = np.zeros(2, dtype=np.float32)
        opp.is_upright = True

    env._game.is_playing = True
    env._game.own_score = 0
    env._game.opp_score = 0


def _run_one_positive_trial(seed: int) -> dict:
    env = NaoSoccerEnv(stage_cfg=CURRICULUM[12], use_fov=False, seed=seed)
    try:
        obs, infos = env.reset(seed=seed)
        _install_positive_fixture(env)
        last_info = None
        for _ in range(3):
            obs, rewards, terms, truncs, infos = env.step(_zero_actions(env))
            last_info = next(iter(infos.values()))
            if (last_info.get("handoff_ready") is True and
                    last_info.get("takeover_active") is True and
                    last_info.get("macro_type") == "playBall_shoot"):
                return last_info
        return last_info or {}
    finally:
        env.close()


def main() -> None:
    print("[TEST] Real positive takeover env-loop smoke")

    bridge_exe = BridgeConfig().bridge_executable
    _assert(Path(bridge_exe).exists(), f"bridge executable exists at {bridge_exe}")

    cfg_root = resolve_cfg_root()
    _assert(cfg_root is not None, f"real CFG root resolved (got {cfg_root})")

    for trial_idx in range(3):
        info = _run_one_positive_trial(seed=42)
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
        _assert(info.get("sidecar_contract_validation_mode") is False,
                f"sidecar_contract_validation_mode = False on trial {trial_idx + 1}")
        _assert(info.get("takeover_simulator_active") is False,
                f"takeover_simulator_active = False on trial {trial_idx + 1}")
        _assert(info.get("bridge_debug_game_state") == "playing",
                f"bridge_debug_game_state = playing on trial {trial_idx + 1} "
                f"(got {info.get('bridge_debug_game_state')})")
        _assert(info.get("handoff_ready") is True,
                f"handoff_ready = True on trial {trial_idx + 1}")
        _assert(info.get("takeover_active") is True,
                f"takeover_active = True on trial {trial_idx + 1}")
        _assert(info.get("sidecar_macro_type") == "playBall_shoot",
                f"sidecar_macro_type = playBall_shoot on trial {trial_idx + 1} "
                f"(got {info.get('sidecar_macro_type')})")
        _assert(info.get("macro_type") == "playBall_shoot",
                f"macro_type = playBall_shoot on trial {trial_idx + 1} "
                f"(got {info.get('macro_type')})")
        _assert(info.get("sidecar_skill_request") == "shoot",
                f"sidecar_skill_request = shoot on trial {trial_idx + 1} "
                f"(got {info.get('sidecar_skill_request')})")
        _assert(info.get("active_agent_id") == "agent_1",
                f"active_agent_id = agent_1 on trial {trial_idx + 1} "
                f"(got {info.get('active_agent_id')})")

    print("\nRESULT: PASS")
    print("  Real env.step() reaches a true positive takeover path.")
    print("  The overlap between handoff_ready and playBall_shoot is stable with fresh env instances.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nRESULT: FAIL")
        print(f"  {type(exc).__name__}: {exc}")
        raise SystemExit(1)
