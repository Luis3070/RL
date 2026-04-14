"""
smoke_takeover_real.py

Canonical consolidated smoke for real takeover behavior under the Option-1
baseline. This keeps a small public surface while preserving legacy focused
fixtures underneath.

Important:
  The dedicated pass completion fixture stays in `smoke_legacy` as a specialist
  semantic replay. It is preserved, but not promoted to canonical gating
  because its own contract is not a stable completion gate.
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
    SmokeEntry("smoke_legacy.smoke_test_real_takeover_completion", "real takeover completion"),
    SmokeEntry("smoke_legacy.smoke_test_real_takeover_playball", "playBall alignment"),
    SmokeEntry("smoke_legacy.smoke_test_real_takeover_env_positive", "positive env fixture"),
    SmokeEntry("smoke_legacy.smoke_test_real_takeover_shoot_completion", "real shoot completion"),
)


def _load_main(module_name: str) -> Callable[[], None]:
    module = importlib.import_module(module_name)
    main = getattr(module, "main", None)
    if main is None or not callable(main):
        raise AttributeError(f"{module_name} does not expose a callable main()")
    return main


def main() -> None:
    print("[SMOKE] takeover_real")
    for entry in CATALOG:
        print(f"  -> {entry.module_name}: {entry.description}")
        _load_main(entry.module_name)()


if __name__ == "__main__":
    main()
