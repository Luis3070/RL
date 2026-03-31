"""
smoke_suite_option1.py

Canonical smoke runner for the current BHuman-in-loop Option-1 baseline.

Why this exists:
  The project accumulated several focused smoke tests. They are still useful,
  but day-to-day validation should not require remembering 8+ filenames.

This runner groups the existing tests into a few operational suites:
  - quick: fastest baseline confidence
  - gating: recommended pre-training gate for Option 1
  - extended: broader semantic/regression coverage
  - diagnostic: non-gating audits
  - all: extended + diagnostic

Important:
  Heavy real-backend smokes are executed SERIALIZED on purpose. Running them
  in parallel can destabilize the bridge and force misleading fallbacks.
"""

from __future__ import annotations

import argparse
import importlib
import time
import traceback
from dataclasses import dataclass
from typing import Callable, Dict, List


@dataclass(frozen=True)
class SmokeEntry:
    module_name: str
    description: str


SMOKE_CATALOG: Dict[str, SmokeEntry] = {
    "persistent": SmokeEntry(
        module_name="smoke_test_rlbridge_persistent_basic",
        description="Transporte persistente RLBridge y reset básico",
    ),
    "env_bridge": SmokeEntry(
        module_name="smoke_test_environment_bridge",
        description="Integración env <-> RLBridge real",
    ),
    "semantic": SmokeEntry(
        module_name="smoke_test_semantic_strong",
        description="Contrato semántico fuerte + provenance real",
    ),
    "sidecar": SmokeEntry(
        module_name="smoke_test_sidecar",
        description="Cobertura sidecar, budget temporal e info plumbing",
    ),
    "takeover_completion": SmokeEntry(
        module_name="smoke_test_real_takeover_completion",
        description="Completion real de takeover dentro de env.step()",
    ),
    "playball_alignment": SmokeEntry(
        module_name="smoke_test_real_takeover_playball",
        description="Alineación playBall real, contexto 4v4 y no falsos takeovers",
    ),
    "env_semantic_variants": SmokeEntry(
        module_name="smoke_test_real_env_semantic_variants",
        description="Variantes reales de semántica en env.step()",
    ),
    "motioninfo_audit": SmokeEntry(
        module_name="audit_motioninfo_takeover_evidence",
        description="Auditoría diagnóstica de MotionInfo para shoot/pass",
    ),
}


SMOKE_SUITES: Dict[str, List[str]] = {
    "quick": [
        "persistent",
        "env_bridge",
        "semantic",
    ],
    "gating": [
        "persistent",
        "env_bridge",
        "semantic",
        "sidecar",
        "takeover_completion",
    ],
    "extended": [
        "persistent",
        "env_bridge",
        "semantic",
        "sidecar",
        "takeover_completion",
        "playball_alignment",
        "env_semantic_variants",
    ],
    "diagnostic": [
        "motioninfo_audit",
    ],
    "all": [
        "persistent",
        "env_bridge",
        "semantic",
        "sidecar",
        "takeover_completion",
        "playball_alignment",
        "env_semantic_variants",
        "motioninfo_audit",
    ],
}


def _load_main(module_name: str) -> Callable[[], None]:
    module = importlib.import_module(module_name)
    main = getattr(module, "main", None)
    if main is None or not callable(main):
        raise AttributeError(f"{module_name} does not expose a callable main()")
    return main


def _run_entry(entry_key: str) -> tuple[bool, float, str | None]:
    entry = SMOKE_CATALOG[entry_key]
    print(f"\n[RUN] {entry_key}: {entry.description}")
    started = time.perf_counter()
    try:
        main = _load_main(entry.module_name)
        main()
        elapsed = time.perf_counter() - started
        print(f"[PASS] {entry_key} ({elapsed:.2f}s)")
        return True, elapsed, None
    except Exception:
        elapsed = time.perf_counter() - started
        error_text = traceback.format_exc()
        print(f"[FAIL] {entry_key} ({elapsed:.2f}s)")
        print(error_text)
        return False, elapsed, error_text


def _print_suite_list() -> None:
    print("Suites disponibles:\n")
    for suite_name, entry_keys in SMOKE_SUITES.items():
        print(f"- {suite_name}")
        for entry_key in entry_keys:
            entry = SMOKE_CATALOG[entry_key]
            print(f"  - {entry_key}: {entry.description}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Canonical smoke runner for BHuman-in-loop Option 1",
    )
    parser.add_argument(
        "--suite",
        default="gating",
        choices=tuple(SMOKE_SUITES.keys()),
        help="Smoke suite to execute. Default: gating",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List suites and exit",
    )
    args = parser.parse_args()

    if args.list:
        _print_suite_list()
        return

    selected = SMOKE_SUITES[args.suite]
    print(f"[SUITE] option1/{args.suite}")
    print("Execution mode: serialized")
    print("Baseline: persistent_2d_cognition")

    results = []
    for entry_key in selected:
        results.append((entry_key, *_run_entry(entry_key)))

    passed = sum(1 for _, ok, _, _ in results if ok)
    failed = len(results) - passed
    total_seconds = sum(elapsed for _, _, elapsed, _ in results)

    print("\n[SUMMARY]")
    print(f"  suite={args.suite}")
    print(f"  passed={passed}")
    print(f"  failed={failed}")
    print(f"  total={len(results)}")
    print(f"  wall_s={total_seconds:.2f}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
