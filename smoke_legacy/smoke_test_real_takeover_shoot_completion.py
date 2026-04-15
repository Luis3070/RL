"""
LEGACY REDIRECT

Archivo legado. La version viva esta en:
`smoke_legacy.smoke_test_real_takeover_shoot_completion`.
Usar la superficie oficial: `smoke_takeover_real.py` o `smoke_suite_option1.py`.



Strict real-backend smoke for shoot takeover completion.

Purpose:
  Prove that a real playBall_shoot takeover can reach:

    macro_end_reason = completed
    macro_success = True

inside env.step(), with real bridge evidence and without relying on the
python takeover simulator.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ["RL_EXECUTION_MODE"] = "bhuman_sidecar"

from bridge.bridge_protocol import BridgeConfig
from config import CURRICULUM, resolve_cfg_root
from environment import NaoSoccerEnv
from smoke_legacy.smoke_test_real_takeover_env_positive import _install_positive_fixture
from smoke_legacy.smoke_test_real_takeover_playball import _zero_actions


def _assert(condition: bool, msg: str) -> None:
    if not condition:
        print(f"  FAIL: {msg}")
        raise AssertionError(msg)
    print(f"  OK: {msg}")


def _run_shoot_completion_trial(seed: int, max_steps: int = 80) -> tuple[dict, list[dict]]:
    env = NaoSoccerEnv(stage_cfg=CURRICULUM[12], use_fov=False, seed=seed)
    try:
        env.reset(seed=seed)
        _install_positive_fixture(env)
        last_info = {}
        for _ in range(max_steps):
            _, _, _, _, infos = env.step(_zero_actions(env))
            last_info = next(iter(infos.values()))
            if last_info.get("episode_takeover_success", 0) > 0:
                break
        return last_info, list(env._takeover_events)
    finally:
        env.close()


def main() -> None:
    print("[TEST] Real shoot takeover completion smoke")

    bridge_exe = BridgeConfig().bridge_executable
    _assert(Path(bridge_exe).exists(), f"bridge executable exists at {bridge_exe}")

    cfg_root = resolve_cfg_root()
    _assert(cfg_root is not None, f"real CFG root resolved (got {cfg_root})")

    info, events = _run_shoot_completion_trial(seed=42)
    _assert(info.get("execution_mode") == "bhuman_sidecar", "execution_mode = bhuman_sidecar")
    _assert(info.get("sidecar_backend") == "real", f"sidecar_backend = real (got {info.get('sidecar_backend')})")
    _assert(info.get("sidecar_macro_type") == "playBall_shoot",
            f"sidecar_macro_type = playBall_shoot (got {info.get('sidecar_macro_type')})")
    _assert(info.get("sidecar_skill_request") == "shoot",
            f"sidecar_skill_request = shoot (got {info.get('sidecar_skill_request')})")
    _assert(
        bool(info.get("sidecar_last_kick_timestamp", 0)) or bool(info.get("takeover_kick_proxy", False)),
        "real kick evidence or proxy evidence present",
    )
    _assert(info.get("episode_takeover_success", 0) >= 1,
            f"episode_takeover_success >= 1 (got {info.get('episode_takeover_success')})")
    _assert(info.get("macro_end_reason") == "completed",
            f"macro_end_reason = completed (got {info.get('macro_end_reason')})")
    _assert(info.get("macro_success") is True,
            f"macro_success = True (got {info.get('macro_success')})")

    successful_events = [ev for ev in events if ev.get("success")]
    _assert(bool(successful_events), "successful takeover event recorded")
    latest_success = successful_events[-1]
    _assert(latest_success.get("type") == "playBall_shoot",
            f"successful event type = playBall_shoot (got {latest_success.get('type')})")
    _assert(latest_success.get("end_reason") == "completed",
            f"successful event end_reason = completed (got {latest_success.get('end_reason')})")

    print("\nRESULT: PASS")
    print("  Real env.step() can reach shoot takeover completion with success=True.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nRESULT: FAIL")
        print(f"  {type(exc).__name__}: {exc}")
        raise SystemExit(1)
