"""
Exporta el actor PPO multi-agente (47-dim, role-conditioned) a ONNX.

Sibling de export_ppo_policy.py (que queda intacto para el contrato obs26 de
producción). Este script INFIERE obs_size del checkpoint (trunk.0.weight) en vez
de hard-codearlo, así sirve para 26, 41 o 47.

Uso:
    python tools/export/export_ppo_policy_team.py \
        --checkpoint runs/multiagent_stage7_v1/checkpoint_stage7_step15000.pt \
        --output models/team/ppo_team_hsl2026.onnx
"""
import argparse, hashlib, json, torch
from pathlib import Path
from action import SKILL_ORDER
from observation import OBS_SIZE

_REPO = Path(__file__).resolve().parents[2]


def _resolve_repo_path(path_str: str) -> Path:
    """Resolve a CLI path, treating relative paths as relative to the RL repo root.

    Importing pybh (pulled in transitively via train/env modules) changes the
    process CWD to SabanaHerons2026/Make/Python, so a relative `runs/...` arg typed
    from the RL repo would otherwise resolve against the wrong root.
    """
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = _REPO / p
    return p.resolve()


class ActorExport(torch.nn.Module):
    def __init__(self, state_dict):
        super().__init__()
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
        skill_logits = F.linear(h, self.ws, self.bs)
        param_mean = torch.tanh(F.linear(h, self.wp, self.bp))
        return skill_logits, param_mean


def export(checkpoint_path: str, output_path: str, active_skills: list[str], specialization: str):
    checkpoint_path = str(_resolve_repo_path(checkpoint_path))
    out_path = _resolve_repo_path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    sd = ckpt["policy_state_dict"]

    obs_size = int(sd["trunk.0.weight"].shape[1])
    n_skills = int(sd["skill_head.weight"].shape[0])
    print(f"obs_size (inferred): {obs_size} | skill logits: {n_skills}")
    print("trunk.0.weight:", sd["trunk.0.weight"].shape)
    print("skill_head.weight:", sd["skill_head.weight"].shape)
    print("param_head.weight:", sd["param_head.weight"].shape)

    actor = ActorExport(sd)
    actor.eval()

    dummy = torch.zeros(1, obs_size)
    skill_logits, param_mean = actor(dummy)
    print("sanity — skill_logits:", skill_logits.shape, "param_mean:", param_mean.shape)

    # dynamo=False: the new dynamo exporter requires onnxscript (not in RLNAO);
    # the legacy exporter handles this 2-layer MLP fine and matches what the
    # production obs26 export used.
    torch.onnx.export(
        actor, dummy, str(out_path),
        input_names=["obs"],
        output_names=["skill_logits", "param_mean"],
        dynamic_axes={"obs": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )

    ckpt_sha = hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest()
    onnx_sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    manifest = {
        "format": "onnx",
        "obs_size_input": obs_size,
        "obs_size_env": OBS_SIZE,
        "obs_size_model_trained": obs_size,
        "skill_order": list(SKILL_ORDER),
        "active_skills": active_skills,
        "specialization": specialization,
        # Gate thresholds the policy was trained under — the deployment-side gate
        # bits (obs[23:26], obs[41:44]) MUST be computed with these values or the
        # policy sees an out-of-distribution gate signal. Mirrors the dataset's
        # summary.json gate_config.
        "gate_config": {
            "shoot_open_enter": 0.30,
            "shoot_open_exit": 0.20,
        },
        # Role one-hot contract (obs[44:46]): exactly one bit set per RL agent,
        # driven by the BHuman role election at deployment
        # (playBall -> is_striker; supports per the team mapping).
        "role_onehot_indices": {"is_striker": 44, "is_open_support": 45, "is_off_ball_support": 46},
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": ckpt_sha,
        "onnx_sha256": onnx_sha,
        "normalization_contract": f"ppo_obs_v{obs_size}_team_hsl2026",
        "deterministic": True,
    }
    Path(out_path.with_suffix(".json")).write_text(json.dumps(manifest, indent=2))
    print(f"ONNX:     {out_path}  ({onnx_sha[:16]}...)")
    print(f"Manifest: {out_path.with_suffix('.json')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default="models/team/ppo_team_hsl2026.onnx")
    p.add_argument(
        "--active-skills",
        default="stand,walk,dribble,shoot,pass",
        help="Comma-separated active skill names recorded in the manifest.",
    )
    p.add_argument(
        "--specialization",
        default="team",
        help="Per-role specialization recorded in the manifest, e.g. striker, defender, supporter.",
    )
    args = p.parse_args()
    export(
        args.checkpoint,
        args.output,
        [s.strip() for s in args.active_skills.split(",") if s.strip()],
        args.specialization,
    )
