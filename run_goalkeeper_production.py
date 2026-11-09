from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from config import ROOT_DIR
from teacher_goalkeeper_dataset import write_dataset
from train_goalkeeper_bc import GKBCConfig, train
from train_goalkeeper_ppo import GKPPOConfig, train as train_ppo
from tools.export_goalkeeper_policy import export
from tools.verify_goalkeeper_dataset_quality import verify_dataset
from tools.validate_goalkeeper_teacher import validate_diversity, validate_teacher


def main() -> int:
    parser = argparse.ArgumentParser(description="One-shot GK teacher audit -> dataset -> dataset audit -> warm BC -> ONNX export")
    parser.add_argument("--synthetic", action="store_true", default=True)
    parser.add_argument("--episodes", type=int, default=48)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--ppo-steps", type=int, default=0)
    parser.add_argument("--ppo-output-onnx", default=str(ROOT_DIR / "ppo_goalkeeper_hsl2026_ppo.onnx"))
    parser.add_argument("--seed", type=int, default=42042)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--teacher-samples", type=int, default=3000)
    parser.add_argument("--min-skill-count", type=int, default=100)
    parser.add_argument("--min-rows", type=int, default=5000)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-onnx", default=str(ROOT_DIR / "ppo_goalkeeper_hsl2026.onnx"))
    parser.add_argument("--verify-action-space", action="store_true")
    parser.add_argument("--skip-teacher-verify", action="store_true")
    parser.add_argument("--skip-dataset-verify", action="store_true")
    args = parser.parse_args()

    teacher_report = None
    if not args.skip_teacher_verify:
        teacher_report = {
            "teacher_cases": validate_teacher(),
            "diversity": validate_diversity(int(args.teacher_samples), int(args.seed)),
        }
        failures = [
            *teacher_report["teacher_cases"]["failures"],
            *teacher_report["diversity"]["failures"],
        ]
        if failures:
            print(json.dumps({"teacher_report": teacher_report, "failures": failures}, indent=2, sort_keys=True), flush=True)
            return 2

    action_space_report = None
    if args.verify_action_space:
        cmd = [
            sys.executable,
            str(ROOT_DIR / "tools" / "verify_goalkeeper_action_space_truth.py"),
            "--launch-simrobot",
            "--reads-per-action",
            "8",
        ]
        completed = subprocess.run(cmd, cwd=ROOT_DIR, text=True)
        action_space_report = {"command": cmd, "returncode": completed.returncode}
        if completed.returncode != 0:
            return completed.returncode

    # Reuse teacher_goalkeeper_dataset's argparse-shaped config object.
    dataset_args = argparse.Namespace(
        synthetic=bool(args.synthetic),
        episodes=int(args.episodes),
        steps=int(args.steps),
        seed=int(args.seed),
        output_dir="",
        backend="simrobot_visible",
        simrobot_profile="3d_visible",
        launch_simrobot=False,
        frames_per_step=3,
        player_number=1,
    )
    dataset_path = write_dataset(dataset_args)
    dataset_report = None
    if not args.skip_dataset_verify:
        dataset_report = verify_dataset(
            Path(dataset_path),
            min_skill_count=int(args.min_skill_count),
            min_rows=int(args.min_rows),
        )
        print(json.dumps({"dataset_quality": dataset_report}, indent=2, sort_keys=True), flush=True)
        if not dataset_report["passed"]:
            return 3

    ckpt = train(
        GKBCConfig(
            dataset=dataset_path,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            device=str(args.device),
        )
    )
    onnx = export(str(ckpt), str(Path(args.output_onnx)))
    ppo_ckpt = None
    ppo_onnx = None
    if int(args.ppo_steps) > 0:
        ppo_ckpt = train_ppo(
            GKPPOConfig(
                dataset=dataset_path,
                bc_checkpoint=ckpt,
                total_steps=int(args.ppo_steps),
                device=str(args.device),
            )
        )
        ppo_onnx = export(str(ppo_ckpt), str(Path(args.ppo_output_onnx)))
    print(
        json.dumps(
            {
                "dataset": str(dataset_path),
                "checkpoint": str(ckpt),
                "onnx": str(onnx),
                "ppo_checkpoint": None if ppo_ckpt is None else str(ppo_ckpt),
                "ppo_onnx": None if ppo_onnx is None else str(ppo_onnx),
                "teacher_report": teacher_report,
                "dataset_quality": dataset_report,
                "action_space_report": action_space_report,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
