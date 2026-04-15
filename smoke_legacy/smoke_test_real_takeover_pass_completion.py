"""
LEGACY REDIRECT

Archivo legado y fixture especializada.
La version viva esta en:
`smoke_legacy.smoke_test_real_takeover_pass_completion`.
No forma parte del gate canonico de takeover.



Strict real-backend smoke for the pass telemetry fixture.

Purpose:
  1. Prove that the dedicated pass fixture is loaded from real BHuman
     telemetry rather than a hand-tuned synthetic frame.
  2. Prove that the fixture still reproduces real `playBall_pass` semantics
     inside env.step() under the current baseline.
  3. Report, but do not falsely over-claim, whether the backend also provides
     enough execution evidence to complete the pass takeover.

Why this smoke is semantic rather than completion-gated:
  The current real pass snapshot yields `playBall_pass` intent reliably, but
  the real backend still reports `last_kick_timestamp = 0` and never trips the
  kick proxy, so strict "completed" gating would conflate fixture provenance
  with a deeper execution-evidence limitation.
"""

from __future__ import annotations

import os
import json
from pathlib import Path

os.environ["RL_EXECUTION_MODE"] = "bhuman_sidecar"

from bridge.bridge_protocol import BridgeConfig
from bhuman_cfg_parser import load_contract
from bhuman_sidecar import BHumanDecisionSidecar
from config import CURRICULUM, resolve_cfg_root
from environment import NaoSoccerEnv
from smoke_legacy.smoke_test_real_env_semantic_variants import _install_frame
from smoke_legacy.smoke_test_real_takeover_playball import PASS_SNAPSHOT_JSON, _make_real_pass_frame, _zero_actions


def _assert(condition: bool, msg: str) -> None:
    if not condition:
        print(f"  FAIL: {msg}")
        raise AssertionError(msg)
    print(f"  OK: {msg}")


def _load_snapshot_payload() -> dict:
    return json.loads(PASS_SNAPSHOT_JSON.read_text(encoding="utf-8"))


def _run_direct_snapshot_semantics() -> dict:
    sidecar = BHumanDecisionSidecar(load_contract(cfg_root=resolve_cfg_root()), execution_mode="bhuman_sidecar")
    try:
        output = sidecar.decide_from_frame(_make_real_pass_frame())
        return {
            "backend": output.backend,
            "macro_type": output.macro_type,
            "skill_request": output.skill_request,
            "active_agent": output.active_agent,
            "pass_target_player_number": output.pass_target_player_number,
            "motion_request": output.motion_request,
            "last_kick_timestamp": output.last_kick_timestamp,
        }
    finally:
        sidecar.shutdown()


def _run_pass_semantic_trial(seed: int, max_steps: int = 20) -> tuple[list[dict], list[dict]]:
    env = NaoSoccerEnv(stage_cfg=CURRICULUM[12], use_fov=False, seed=seed)
    try:
        env.reset(seed=seed)
        _install_frame(env, _make_real_pass_frame())
        timeline: list[dict] = []
        for _ in range(max_steps):
            _, _, _, _, infos = env.step(_zero_actions(env))
            info = next(iter(infos.values()))
            timeline.append({
                "sidecar_backend": info.get("sidecar_backend"),
                "sidecar_macro_type": info.get("sidecar_macro_type"),
                "sidecar_skill_request": info.get("sidecar_skill_request"),
                "sidecar_motion_request": info.get("sidecar_motion_request"),
                "sidecar_last_kick_timestamp": info.get("sidecar_last_kick_timestamp", 0),
                "takeover_active": info.get("takeover_active"),
                "takeover_kick_proxy": info.get("takeover_kick_proxy", False),
                "macro_end_reason": info.get("macro_end_reason"),
                "macro_success": info.get("macro_success"),
                "episode_takeover_success": info.get("episode_takeover_success", 0),
            })
        return timeline, list(env._takeover_events)
    finally:
        env.close()


def main() -> None:
    print("[TEST] Real pass telemetry fixture smoke")

    bridge_exe = BridgeConfig().bridge_executable
    _assert(Path(bridge_exe).exists(), f"bridge executable exists at {bridge_exe}")

    cfg_root = resolve_cfg_root()
    _assert(cfg_root is not None, f"real CFG root resolved (got {cfg_root})")

    _assert(PASS_SNAPSHOT_JSON.exists(), f"pass snapshot exists at {PASS_SNAPSHOT_JSON}")
    payload = _load_snapshot_payload()
    provenance = payload.get("provenance", {})
    _assert(
        provenance.get("source") == "real_bhuman_bridge_search",
        f"snapshot provenance source is real_bhuman_bridge_search (got {provenance.get('source')})",
    )
    _assert(
        provenance.get("macro_type") == "playBall_pass",
        f"snapshot provenance macro_type = playBall_pass (got {provenance.get('macro_type')})",
    )
    _assert(
        provenance.get("skill_request") == "pass",
        f"snapshot provenance skill_request = pass (got {provenance.get('skill_request')})",
    )

    direct = _run_direct_snapshot_semantics()
    _assert(direct.get("backend") == "real", f"direct sidecar backend = real (got {direct.get('backend')})")
    _assert(direct.get("macro_type") == "playBall_pass",
            f"direct sidecar macro_type = playBall_pass (got {direct.get('macro_type')})")
    _assert(direct.get("skill_request") == "pass",
            f"direct sidecar skill_request = pass (got {direct.get('skill_request')})")

    timeline, events = _run_pass_semantic_trial(seed=21)
    _assert(bool(timeline), "env produced pass-fixture timeline")
    first = timeline[0]
    _assert(first.get("sidecar_backend") == "real",
            f"first env frame backend = real (got {first.get('sidecar_backend')})")
    _assert(first.get("sidecar_macro_type") == "playBall_pass",
            f"first env frame macro_type = playBall_pass (got {first.get('sidecar_macro_type')})")
    _assert(first.get("sidecar_skill_request") == "pass",
            f"first env frame skill_request = pass (got {first.get('sidecar_skill_request')})")
    _assert(first.get("takeover_active") is True,
            f"first env frame takeover_active = True (got {first.get('takeover_active')})")

    saw_pass = any(step.get("sidecar_macro_type") == "playBall_pass" for step in timeline)
    _assert(saw_pass, "timeline contains at least one real playBall_pass frame")

    saw_completion = any(step.get("episode_takeover_success", 0) > 0 for step in timeline)
    saw_kick_evidence = any(
        bool(step.get("sidecar_last_kick_timestamp", 0)) or bool(step.get("takeover_kick_proxy", False))
        for step in timeline
    )

    print("\nRESULT: PASS")
    print("  The dedicated pass fixture now comes from real BHuman telemetry.")
    print("  The fixture reproduces real playBall_pass semantics inside env.step().")
    if saw_completion:
        print("  The rollout also reached takeover completion with success=True.")
    else:
        print("  Completion is still blocked: no real kick evidence emerged during the replay.")
    if not saw_kick_evidence:
        print("  Note: sidecar_last_kick_timestamp stayed at 0 and takeover_kick_proxy never triggered.")
    if events:
        print(f"  Recorded takeover events: {events}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nRESULT: FAIL")
        print(f"  {type(exc).__name__}: {exc}")
        print("  NOTE: if the first env frame is not playBall_pass, the dedicated pass fixture")
        print("  is stale or not actually sourced from current real BHuman telemetry.")
        raise SystemExit(1)
