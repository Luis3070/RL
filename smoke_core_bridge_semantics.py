"""
smoke_core_bridge_semantics.py

Canonical consolidated smoke for the core Option-1 baseline:

- persistent transport
- env <-> bridge integration
- strong semantic contract
- sidecar plumbing

This wrapper exists to reduce the public smoke surface without deleting the
older focused tests that remain useful for diagnostics.
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
    SmokeEntry("smoke_legacy.smoke_test_rlbridge_persistent_basic", "persistent transport"),
    SmokeEntry("smoke_legacy.smoke_test_environment_bridge", "environment bridge integration"),
    SmokeEntry("smoke_legacy.smoke_test_semantic_strong", "strong semantic contract"),
    SmokeEntry("smoke_legacy.smoke_test_sidecar", "sidecar plumbing"),
)


def _load_main(module_name: str) -> Callable[[], None]:
    module = importlib.import_module(module_name)
    main = getattr(module, "main", None)
    if main is None or not callable(main):
        raise AttributeError(f"{module_name} does not expose a callable main()")
    return main


def main() -> None:
    print("[SMOKE] core_bridge_semantics")
    for entry in CATALOG:
        print(f"  -> {entry.module_name}: {entry.description}")
        _load_main(entry.module_name)()


if __name__ == "__main__":
    main()
