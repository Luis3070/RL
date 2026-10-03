from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from action import SKILL_ORDER
from defender_policy import ACTIVE_DEFENDER_SKILLS, DefenderActor, PASS_FRONT_CLEAR_THRESHOLDS


def export(checkpoint: Path, output: Path) -> None:
    ckpt = torch.load(checkpoint, map_location="cpu")
    actor = DefenderActor(ckpt["policy_state_dict"]).eval()
    dummy = torch.zeros(1, 26, dtype=torch.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        actor,
        dummy,
        str(output),
        input_names=["obs"],
        output_names=["skill_logits", "param_mean"],
        opset_version=11,
        dynamic_axes={"obs": {0: "batch"}, "skill_logits": {0: "batch"}, "param_mean": {0: "batch"}},
        dynamo=False,
    )
    manifest = {
        "format": "onnx",
        "role": "defender",
        "decoder_contract": "ppo_defender_obs26_skill_selector_v1",
        "obs_size_input": 26,
        "obs_size_env": 26,
        "obs_size_model_trained": 26,
        "skill_order": list(SKILL_ORDER),
        "active_skills": list(ACTIVE_DEFENDER_SKILLS),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "onnx_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "normalization_contract": "ppo_obs_v26_hsl2026",
        "deterministic": True,
        "pass_commit_mask": {
            "location": "onnx_graph_and_bhuman_runtime",
            "obs_index": 23,
            "engage_obs_index": 24,
            "progress_obs_index": 25,
            "traffic_clear": {
                "teammate_front_mm": PASS_FRONT_CLEAR_THRESHOLDS["teammate"],
                "opponent_front_mm": PASS_FRONT_CLEAR_THRESHOLDS["opponent"],
                "uncertain_front_mm": PASS_FRONT_CLEAR_THRESHOLDS["uncertain"],
            },
            "effect": "when defender pass_armed is baked into obs[23] and front traffic is clear, all non-pass logits are set to -1e9 before argmax",
        },
        "engage_mask": {
            "location": "onnx_graph_and_bhuman_runtime",
            "obs_index": 24,
            "effect": "when pass_armed is false and engage_armed is false, only stand/walk remain selectable",
        },
        "runtime_note": "For defender obs26, obs[23] is pass_armed, obs[24] is engage_armed, and obs[25] is pass_arm_progress.",
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(output)
    print(output.with_suffix(".json"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export obs26 defender BC checkpoint to ONNX.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="models/defender/ppo_defender_hsl2026_param_repair.onnx")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export(Path(args.checkpoint).resolve(), Path(args.output).resolve())
