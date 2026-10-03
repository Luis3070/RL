from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import onnx
import torch
from onnx import numpy_helper

from rlfootball.core.action import SKILL_ORDER
from rlfootball.teachers.defender_policy import ACTIVE_DEFENDER_SKILLS, DefenderActor


ONNX_TO_STATE = {
    "w0": "trunk.0.weight",
    "b0": "trunk.0.bias",
    "w2": "trunk.2.weight",
    "b2": "trunk.2.bias",
    "ws": "skill_head.weight",
    "bs": "skill_head.bias",
    "wp": "param_head.weight",
    "bp": "param_head.bias",
}


def recover(onnx_path: Path, output_path: Path, manifest_path: Path | None = None) -> None:
    model = onnx.load(onnx_path)
    initializers = {tensor.name: torch.from_numpy(numpy_helper.to_array(tensor).copy()).float() for tensor in model.graph.initializer}
    missing = sorted(set(ONNX_TO_STATE) - set(initializers))
    if missing:
        raise KeyError(f"Missing ONNX initializers required for checkpoint recovery: {missing}")

    recovered_state = {state_key: initializers[onnx_key] for onnx_key, state_key in ONNX_TO_STATE.items()}
    actor = DefenderActor(recovered_state)
    state = actor.state_dict()
    value_state = _fresh_value_state(seed=260611)
    meta = {
        "role": "defender",
        "recovered_from_onnx": str(onnx_path),
        "onnx_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
        "obs_size": 26,
        "skill_order": list(SKILL_ORDER),
        "active_skills": list(ACTIVE_DEFENDER_SKILLS),
        "normalization_contract": "ppo_obs_v26_hsl2026",
        "decoder_contract": "ppo_defender_obs26_skill_selector_v1",
        "note": "Recovered dense actor weights from ONNX initializers. The ONNX graph also contains runtime masking nodes; this checkpoint stores the underlying actor weights.",
    }
    if manifest_path and manifest_path.exists():
        meta["source_manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "timestep": 0,
            "stage_id": 4,
            "policy_state_dict": state,
            "value_state_dict": value_state,
            "eval_summary": {
                "source": "recovered_from_onnx",
                "onnx_sha256": meta["onnx_sha256"],
                "value_state_initialized": True,
            },
            "meta": meta,
        },
        output_path,
    )
    print(output_path)
    print(hashlib.sha256(output_path.read_bytes()).hexdigest())


def _fresh_value_state(seed: int) -> dict[str, torch.Tensor]:
    generator_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(26, 128),
        torch.nn.ELU(),
        torch.nn.Linear(128, 128),
        torch.nn.ELU(),
        torch.nn.Linear(128, 1),
    )
    state = {f"net.{key}": value for key, value in net.state_dict().items()}
    torch.random.set_rng_state(generator_state)
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover obs26 defender .pt checkpoint from exported ONNX initializers.")
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--manifest", default="")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    recover(Path(args.onnx).resolve(), Path(args.output).resolve(), Path(args.manifest).resolve() if args.manifest else None)
