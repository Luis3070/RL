"""
smoke_diagnostic_extended.py

Consolidated diagnostic smoke. Not intended to be the primary daily gate, but
kept as the official extended/regression entrypoint.

Important:
  `smoke_test_real_env_semantic_variants` remains preserved in `smoke_legacy`
  as a drift detector. It is intentionally not part of the official wrapper
  because its fixture stability is weaker than the canonical diagnostic surface.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class SmokeEntry:
    module_name: str
    description: str


CATALOG: tuple[SmokeEntry, ...] = (
    SmokeEntry("smoke_legacy.smoke_test_rlbridge_persistent_stress", "persistent stress"),
    SmokeEntry("smoke_legacy.audit_motioninfo_takeover_evidence", "motioninfo audit"),
)


def _load_main(module_name: str) -> Callable[[], None]:
    module = importlib.import_module(module_name)
    main = getattr(module, "main", None)
    if main is None or not callable(main):
        raise AttributeError(f"{module_name} does not expose a callable main()")
    return main


def main() -> None:
    print("[SMOKE] diagnostic_extended")
    for entry in CATALOG:
        print(f"  -> {entry.module_name}: {entry.description}")
        _load_main(entry.module_name)()


if __name__ == "__main__":
    main()
