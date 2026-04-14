"""
LEGACY REDIRECT

Archivo legado. La version viva para diagnostico extendido esta en:
`smoke_legacy.audit_motioninfo_takeover_evidence`.
La superficie oficial del proyecto entra por `smoke_suite_option1.py`.
"""

"""
audit_motioninfo_takeover_evidence.py

Non-gating audit for real takeover evidence in shoot/pass fixtures.

Why this exists:
  The project already proves real bridge semantics and at least one successful
  takeover completion (dribble). What still needs explicit auditing is whether
  real `shoot` / `pass` takeovers expose execution evidence via:

    - MotionInfo.last_kick_type
    - MotionInfo.last_kick_timestamp
    - takeover completion success in the env

This script is intentionally diagnostic, not a strict smoke gate. It reports
whether those signals are present under the current real fixtures.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ["RL_EXECUTION_MODE"] = "bhuman_sidecar"

from bridge.bridge_protocol import BridgeConfig
from config import CURRICULUM, resolve_cfg_root
from environment import NaoSoccerEnv
from smoke_test_real_env_semantic_variants import _install_frame, _zero_actions
from smoke_test_real_takeover_env_positive import _install_positive_fixture
from smoke_test_real_takeover_playball import _make_real_pass_frame


def _probe_shoot_fixture(max_steps: int = 80) -> dict:
    env = NaoSoccerEnv(stage_cfg=CURRICULUM[12], use_fov=False, seed=42)
    try:
        env.reset(seed=42)
        _install_positive_fixture(env)

        overlap_seen = False
        kick_evidence_seen = False
        success_seen = False
        final_info = {}

        for _ in range(max_steps):
            _, _, _, _, infos = env.step(_zero_actions(env))
            final_info = next(iter(infos.values()))
            overlap_seen = overlap_seen or bool(
                final_info.get("handoff_ready") and final_info.get("takeover_active")
            )
            kick_evidence_seen = kick_evidence_seen or bool(
                final_info.get("sidecar_last_kick_timestamp", 0)
            )
            success_seen = success_seen or bool(final_info.get("episode_takeover_success", 0) > 0)
            if final_info.get("episode_takeover_count", 0) > 0:
                break

        return {
            "scenario": "shoot",
            "overlap_seen": overlap_seen,
            "kick_evidence_seen": kick_evidence_seen,
            "success_seen": success_seen,
            "final_info": final_info,
            "events": list(env._takeover_events),
        }
    finally:
        env.close()


def _probe_pass_fixture(max_steps: int = 80) -> dict:
    env = NaoSoccerEnv(stage_cfg=CURRICULUM[12], use_fov=False, seed=21)
    try:
        env.reset(seed=21)
        _install_frame(env, _make_real_pass_frame())

        overlap_seen = False
        kick_evidence_seen = False
        success_seen = False
        final_info = {}

        for _ in range(max_steps):
            _, _, _, _, infos = env.step(_zero_actions(env))
            final_info = next(iter(infos.values()))
            overlap_seen = overlap_seen or bool(
                final_info.get("handoff_ready") and final_info.get("takeover_active")
            )
            kick_evidence_seen = kick_evidence_seen or bool(
                final_info.get("sidecar_last_kick_timestamp", 0)
            )
            success_seen = success_seen or bool(final_info.get("episode_takeover_success", 0) > 0)

        return {
            "scenario": "pass",
            "overlap_seen": overlap_seen,
            "kick_evidence_seen": kick_evidence_seen,
            "success_seen": success_seen,
            "final_info": final_info,
            "events": list(env._takeover_events),
        }
    finally:
        env.close()


def _print_probe(result: dict) -> None:
    final_info = result["final_info"]
    print(f"\n[SCENARIO] {result['scenario']}")
    print(f"  overlap_seen={result['overlap_seen']}")
    print(f"  kick_evidence_seen={result['kick_evidence_seen']}")
    print(f"  success_seen={result['success_seen']}")
    print(f"  final_macro_type={final_info.get('macro_type')}")
    print(f"  final_sidecar_macro_type={final_info.get('sidecar_macro_type')}")
    print(f"  final_skill_request={final_info.get('sidecar_skill_request')}")
    print(f"  final_last_kick_type={final_info.get('sidecar_last_kick_type')}")
    print(f"  final_last_kick_timestamp={final_info.get('sidecar_last_kick_timestamp')}")
    print(f"  final_macro_end_reason={final_info.get('macro_end_reason')}")
    print(f"  final_macro_success={final_info.get('macro_success')}")
    print(f"  episode_takeover_count={final_info.get('episode_takeover_count')}")
    print(f"  episode_takeover_success={final_info.get('episode_takeover_success')}")
    print(f"  takeover_events_tail={result['events'][-3:]}")


def main() -> None:
    print("[AUDIT] MotionInfo takeover evidence")

    bridge_exe = BridgeConfig().bridge_executable
    if not Path(bridge_exe).exists():
        raise SystemExit(f"Bridge executable not found: {bridge_exe}")

    cfg_root = resolve_cfg_root()
    if cfg_root is None:
        raise SystemExit("Real CFG root could not be resolved")

    shoot = _probe_shoot_fixture()
    pass_result = _probe_pass_fixture()

    _print_probe(shoot)
    _print_probe(pass_result)

    print("\nRESULT")
    if (
        shoot["overlap_seen"] and not shoot["kick_evidence_seen"] and not shoot["success_seen"]
        and not pass_result["kick_evidence_seen"] and not pass_result["success_seen"]
    ):
        print("  ATTENTION: real shoot/pass semantics exist, but MotionInfo kick evidence")
        print("  is not yet appearing robustly enough to confirm successful shoot/pass")
        print("  takeover completion in the environment.")
    else:
        print("  MotionInfo evidence is present in at least one audited kick scenario.")


if __name__ == "__main__":
    main()
