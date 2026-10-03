from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from rlfootball.core.goalkeeper_action import GK_SKILL_ORDER
from rlfootball.teachers.goalkeeper_teacher import GK_OBS_SIZE


class ActorExport(torch.nn.Module):
    def __init__(self, state_dict, no_head: bool = False):
        super().__init__()
        self.no_head = no_head
        self.w0 = torch.nn.Parameter(state_dict["trunk.0.weight"])
        self.b0 = torch.nn.Parameter(state_dict["trunk.0.bias"])
        self.w2 = torch.nn.Parameter(state_dict["trunk.2.weight"])
        self.b2 = torch.nn.Parameter(state_dict["trunk.2.bias"])
        self.ws = torch.nn.Parameter(state_dict["skill_head.weight"])
        self.bs = torch.nn.Parameter(state_dict["skill_head.bias"])
        self.wp = torch.nn.Parameter(state_dict["param_head.weight"])
        self.bp = torch.nn.Parameter(state_dict["param_head.bias"])

    def forward(self, obs):
        import torch.nn.functional as F
        h = F.elu(F.linear(obs, self.w0, self.b0))
        h = F.elu(F.linear(h, self.w2, self.b2))
        logits = F.linear(h, self.ws, self.bs)
        params = torch.tanh(F.linear(h, self.wp, self.bp))
        if self.no_head:
            return logits, params
        speed = torch.sqrt(torch.clamp(obs[:, 7] * obs[:, 7] + obs[:, 8] * obs[:, 8], min=0.0))
        slow_corner = (
            (speed <= 0.30)
            & (torch.abs(obs[:, 21]) >= 0.22)
            & (torch.abs(obs[:, 4]) <= 0.12)
            & (obs[:, 19] >= 0.5)
            & (obs[:, 27] >= 0.5)
            & (obs[:, 37] <= 0.4)
            & (obs[:, 3] > 0.0)
        )
        controlled = obs[:, 25] >= 0.5
        walk = slow_corner & (~controlled)
        clear = slow_corner & controlled
        logits = logits.clone()
        params = params.clone()
        logits[:, 1] = torch.where(walk, logits[:, 1] + 80.0, logits[:, 1])
        logits[:, 9] = torch.where(clear, logits[:, 9] + 80.0, logits[:, 9])
        params[:, 0] = torch.where(walk, torch.full_like(params[:, 0], -3700.0 / 4500.0), params[:, 0])
        params[:, 1] = torch.where(walk, torch.clamp(obs[:, 21] * (1200.0 / 3000.0), -1.0, 1.0), params[:, 1])
        params[:, 2] = torch.where(walk, torch.zeros_like(params[:, 2]), params[:, 2])
        return logits, params


def _resolve(path: str) -> Path:
    p = Path(path).expanduser()
    return (_REPO / p).resolve() if not p.is_absolute() else p.resolve()


def export(checkpoint: str, output: str, no_head: bool = False) -> Path:
    ckpt_path = _resolve(checkpoint)
    out = _resolve(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["policy_state_dict"]
    actor = ActorExport(sd, no_head=no_head).eval()
    obs_size = int(sd["trunk.0.weight"].shape[1])
    dummy = torch.zeros(1, obs_size)
    torch.onnx.export(
        actor,
        dummy,
        str(out),
        input_names=["obs"],
        output_names=["skill_logits", "param_mean"],
        dynamic_axes={"obs": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    manifest = {
        "format": "onnx",
        "specialization": "goalkeeper",
        "obs_size_input": obs_size,
        "obs_size_contract": GK_OBS_SIZE,
        "skill_order": list(GK_SKILL_ORDER),
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": hashlib.sha256(ckpt_path.read_bytes()).hexdigest(),
        "onnx_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "bridge_note": "intercept_* actions require the GK C++ SkillRequest bridge for native execution",
        "slow_corner_dribble_head": (
            "DISABLED (head-free policy — dribble defense is learned, not scripted)" if no_head
            else "boosts walk_goalie_pose/clear for slow fresh corner-bound dribble threats; obs/action contract unchanged"
        ),
    }
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"onnx": str(out), "manifest": str(out.with_suffix(".json"))}, indent=2), flush=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="models/goalkeeper/ppo_goalkeeper_hsl2026.onnx")
    parser.add_argument("--no-head", action="store_true", help="export head-free (for closed-loop-trained policies)")
    args = parser.parse_args()
    export(args.checkpoint, args.output, no_head=args.no_head)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
