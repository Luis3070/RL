from __future__ import annotations

import argparse
from dataclasses import dataclass, replace

import numpy as np

import torch
import torch.nn as nn

from action import StructuredSkillAction, action_to_numpy
from config import ACTION, ENV, TRAIN
from environment import RealSimRobotEnv
from wrapper import RealSimRobotEnvWrapper


@dataclass
class PolicyOutput:
    skill_logits: torch.Tensor
    target_x: torch.Tensor
    target_y: torch.Tensor
    target_theta: torch.Tensor
    pass_target_logits: torch.Tensor


class StructuredPolicy(nn.Module):
    def __init__(self, obs_size: int, hidden_size: int, num_skills: int, max_pass_target: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
        )
        self.skill_head = nn.Linear(hidden_size, num_skills)
        self.target_x_head = nn.Linear(hidden_size, 1)
        self.target_y_head = nn.Linear(hidden_size, 1)
        self.target_theta_head = nn.Linear(hidden_size, 1)
        self.pass_head = nn.Linear(hidden_size, max_pass_target + 1)

    def forward(self, obs: torch.Tensor) -> PolicyOutput:
        latent = self.backbone(obs)
        return PolicyOutput(
            skill_logits=self.skill_head(latent),
            target_x=torch.tanh(self.target_x_head(latent)).squeeze(-1) * ACTION.x_range,
            target_y=torch.tanh(self.target_y_head(latent)).squeeze(-1) * ACTION.y_range,
            target_theta=torch.tanh(self.target_theta_head(latent)).squeeze(-1) * ACTION.theta_range,
            pass_target_logits=self.pass_head(latent),
        )


def sample_action(policy: StructuredPolicy, obs: np.ndarray) -> StructuredSkillAction:
    with torch.no_grad():
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        out = policy(obs_t)
        skill_type = int(torch.argmax(out.skill_logits, dim=-1).item())
        pass_target = int(torch.argmax(out.pass_target_logits, dim=-1).item())
        return StructuredSkillAction(
            skill_type=skill_type,
            target_x=float(out.target_x.item()),
            target_y=float(out.target_y.item()),
            target_theta=float(out.target_theta.item()),
            pass_target=pass_target,
        )

def run_smoke(steps: int, backend: str) -> None:
    env_cfg = replace(ENV, backend=replace(ENV.backend, mode=backend))
    env = RealSimRobotEnvWrapper(RealSimRobotEnv(env_cfg=env_cfg))
    obs, infos = env.reset(seed=TRAIN.seed)
    policy = StructuredPolicy(
        obs_size=env.observation_spaces["player_1"].shape[0],
        hidden_size=TRAIN.hidden_size,
        num_skills=len(ACTION.skill_names),
        max_pass_target=ACTION.max_pass_target,
    )
    total_reward = 0.0
    for step in range(steps):
        action = sample_action(policy, obs["player_1"])
        if action.skill_name not in env._env.cfg.active_skill_names:
            action = StructuredSkillAction(skill_type=0)
        obs, rewards, terminateds, truncateds, infos = env.step({"player_1": action_to_numpy(action)})
        total_reward += rewards["player_1"]
        print(
            f"[smoke] step={step + 1} skill={action.skill_name} "
            f"reward={rewards['player_1']:.3f} frame={infos['player_1']['bridge_frame']}"
        )
        if terminateds.get("__all__") or truncateds.get("__all__"):
            break
    env.close()
    print(f"[smoke] total_reward={total_reward:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Structured SkillRequest smoke runner")
    parser.add_argument("--smoke-steps", type=int, default=TRAIN.smoke_steps)
    parser.add_argument("--backend", choices=("simrobot", "cpp", "python"), default="simrobot")
    args = parser.parse_args()
    run_smoke(args.smoke_steps, backend=args.backend)


if __name__ == "__main__":
    main()
