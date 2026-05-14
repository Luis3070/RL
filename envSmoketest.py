from __future__ import annotations

from dataclasses import replace

from train import IDE_RUN, ScriptRunConfig, main


SMOKE_RUN = replace(
    IDE_RUN,
    mode="smoke",
    telemetry_tag="ide_smoke",
    smoke_steps=8,
    stage_id=0,
    max_stage_id=0,
)


def run() -> None:
    main(SMOKE_RUN)


if __name__ == "__main__":
    run()
