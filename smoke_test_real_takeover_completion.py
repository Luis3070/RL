"""
smoke_test_real_takeover_completion.py

Strict real-backend smoke for takeover completion.

Purpose:
  Prove that at least one real BHuman-in-loop takeover can reach:

    macro_end_reason = completed
    macro_success = True

inside `env.step()`, not just overlap/start of takeover.

Current stable positive path:
  - dribble replay found from the real bridge
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ["RL_EXECUTION_MODE"] = "bhuman_sidecar"

from bridge.bridge_protocol import BridgeConfig
from bhuman_sidecar import normalize_tactic_name
from config import CURRICULUM, resolve_cfg_root
from environment import NaoSoccerEnv
from smoke_test_real_env_semantic_variants import (
    DRIBBLE_REPLAY_SEED,
    _build_dribble_replay_frames,
    _install_frame,
    _zero_actions,
)


def _assert(condition: bool, msg: str) -> None:
    if not condition:
        print(f"  FAIL: {msg}")
        raise AssertionError(msg)
    print(f"  OK: {msg}")


def _run_dribble_completion_trial(seed: int) -> tuple[dict, list[dict]]:
    frames = _build_dribble_replay_frames()
    env = NaoSoccerEnv(stage_cfg=CURRICULUM[12], use_fov=False, seed=seed)
    try:
        env.reset(seed=seed)
        last_info = {}
        for frame in frames:
            _install_frame(env, frame)
            _, _, _, _, infos = env.step(_zero_actions(env))
            last_info = next(iter(infos.values()))
        return last_info, list(env._takeover_events)
    finally:
        env.close()


def main() -> None:
    print("[TEST] Real takeover completion smoke")

    bridge_exe = BridgeConfig().bridge_executable
    _assert(Path(bridge_exe).exists(), f"bridge executable exists at {bridge_exe}")

    cfg_root = resolve_cfg_root()
    _assert(cfg_root is not None, f"real CFG root resolved (got {cfg_root})")

    for trial_idx in range(3):
        info, events = _run_dribble_completion_trial(seed=DRIBBLE_REPLAY_SEED)
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
        _assert(info.get("episode_takeover_count", 0) >= 1,
                f"episode_takeover_count >= 1 on trial {trial_idx + 1} "
                f"(got {info.get('episode_takeover_count')})")
        _assert(info.get("episode_takeover_success", 0) >= 1,
                f"episode_takeover_success >= 1 on trial {trial_idx + 1} "
                f"(got {info.get('episode_takeover_success')})")
        _assert(info.get("macro_end_reason") == "completed",
                f"macro_end_reason = completed on trial {trial_idx + 1} "
                f"(got {info.get('macro_end_reason')})")
        _assert(info.get("macro_success") is True,
                f"macro_success = True on trial {trial_idx + 1} "
                f"(got {info.get('macro_success')})")

        successful_events = [ev for ev in events if ev.get("success")]
        _assert(bool(successful_events),
                f"successful takeover event recorded on trial {trial_idx + 1}")
        latest_success = successful_events[-1]
        _assert(latest_success.get("type") == "playBall_dribble",
                f"successful event type = playBall_dribble on trial {trial_idx + 1} "
                f"(got {latest_success.get('type')})")
        _assert(latest_success.get("end_reason") == "completed",
                f"successful event end_reason = completed on trial {trial_idx + 1} "
                f"(got {latest_success.get('end_reason')})")

    print("\nRESULT: PASS")
    print("  Real env.step() can reach takeover completion with success=True.")
    print("  The stable completion path is currently a real dribble takeover.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nRESULT: FAIL")
        print(f"  {type(exc).__name__}: {exc}")
        raise SystemExit(1)
