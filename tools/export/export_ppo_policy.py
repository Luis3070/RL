"""
Exporta el actor PPO a ONNX para inferencia en robot.
Uso:
    python tools/export/export_ppo_policy.py \
        --checkpoint runs/production_stability_repair_v1/checkpoint_stage4_step5000.pt \
        --output models/striker/ppo_striker_hsl2026.onnx
"""
import argparse, hashlib, json, torch
from pathlib import Path
from rlfootball.core.action import SKILL_ORDER
from rlfootball.core.observation import OBS_SIZE


class ActorExport(torch.nn.Module):
    def __init__(self, state_dict):
        super().__init__()
        # Registrar pesos directamente — sin reconstruir capas
        self.w0 = torch.nn.Parameter(state_dict["trunk.0.weight"])  # [128, 26]
        self.b0 = torch.nn.Parameter(state_dict["trunk.0.bias"])
        self.w2 = torch.nn.Parameter(state_dict["trunk.2.weight"])  # [128, 128]
        self.b2 = torch.nn.Parameter(state_dict["trunk.2.bias"])
        self.ws = torch.nn.Parameter(state_dict["skill_head.weight"])  # [8, 128]
        self.bs = torch.nn.Parameter(state_dict["skill_head.bias"])
        self.wp = torch.nn.Parameter(state_dict["param_head.weight"])  # [4, 128]
        self.bp = torch.nn.Parameter(state_dict["param_head.bias"])

    def forward(self, obs):
        import torch.nn.functional as F
        h = F.elu(F.linear(obs, self.w0, self.b0))
        h = F.elu(F.linear(h,   self.w2, self.b2))
        skill_logits = F.linear(h, self.ws, self.bs)
        param_mean   = torch.tanh(F.linear(h, self.wp, self.bp))
        return skill_logits, param_mean


def export(checkpoint_path: str, output_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    sd   = ckpt["policy_state_dict"]

    print("trunk.0.weight:", sd["trunk.0.weight"].shape)
    print("skill_head.weight:", sd["skill_head.weight"].shape)
    print("param_head.weight:", sd["param_head.weight"].shape)

    actor = ActorExport(sd)
    actor.eval()

    dummy = torch.zeros(1, 26)
    skill_logits, param_mean = actor(dummy)
    print("sanity — skill_logits:", skill_logits.shape, "param_mean:", param_mean.shape)

    out_path = Path(output_path)
    torch.onnx.export(
        actor, dummy, str(out_path),
        input_names=["obs"],
        output_names=["skill_logits", "param_mean"],
        dynamic_axes={"obs": {0: "batch"}},
        opset_version=17,
    )

    ckpt_sha = hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest()
    onnx_sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    manifest = {
        "format": "onnx",
        "obs_size_input": 26,        # lo que entra al ONNX
        "obs_size_env": OBS_SIZE,    # 41 — lo que produce el entorno (15 son padding cero)
        "obs_size_model_trained": 26,# con lo que se entrenó el trunk
        "skill_order": list(SKILL_ORDER),
        "active_skills": ["stand", "walk", "dribble", "shoot"],
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": ckpt_sha,
        "onnx_sha256": onnx_sha,
        "export_commit": "ed8ffdc",
        "normalization_contract": "ppo_obs_v26_hsl2026",
        "deterministic": True,
    }
    Path(out_path.with_suffix(".json")).write_text(json.dumps(manifest, indent=2))
    print(f"ONNX:     {out_path}  ({onnx_sha[:16]}...)")
    print(f"Manifest: {out_path.with_suffix('.json')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default="models/striker/ppo_striker_hsl2026.onnx")
    args = p.parse_args()
    export(args.checkpoint, args.output)