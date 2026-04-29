from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.multi_agents.torch.mappo import MAPPO, MAPPO_CFG
from skrl.trainers.torch import SequentialTrainerCfg

from action import StructuredSkillAction, action_to_numpy
from config import ACTION, ENV, TRAIN, BackendConfig
from environment import RealSimRobotEnv

RUNS_DIR = str((Path(__file__).resolve().parent / "runs").resolve())


class PolicyNetwork(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(
            self,
            clip_actions=True,
            clip_log_std=True,
            min_log_std=-4.0,
            max_log_std=0.0,
        )
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, TRAIN.hidden_size),
            nn.ELU(),
            nn.Linear(TRAIN.hidden_size, TRAIN.hidden_size),
            nn.ELU(),
            nn.Linear(TRAIN.hidden_size, self.num_actions),
        )
        self.log_std_parameter = nn.Parameter(torch.full((self.num_actions,), -0.5, dtype=torch.float32))

    def compute(self, inputs, role=""):
        observations = inputs["observations"]
        mean_actions = self.net(observations)
        return mean_actions, {"log_std": self.log_std_parameter.expand_as(mean_actions)}


class ValueNetwork(DeterministicMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self, clip_actions=False)
        self.net = nn.Sequential(
            nn.Linear(self.num_states, TRAIN.hidden_size),
            nn.ELU(),
            nn.Linear(TRAIN.hidden_size, TRAIN.hidden_size),
            nn.ELU(),
            nn.Linear(TRAIN.hidden_size, 1),
        )

    def compute(self, inputs, role=""):
        states = inputs.get("states", inputs["observations"])
        return self.net(states), {}


def build_env(backend_cfg: BackendConfig):
    env_cfg = replace(ENV, backend=backend_cfg)
    env = RealSimRobotEnv(env_cfg=env_cfg)
    return env, wrap_env(env, wrapper="pettingzoo")


def build_mappo(env, device: str) -> MAPPO:
    agent_id = env.possible_agents[0]
    observation_space = env.observation_space(agent_id)
    action_space = env.action_space(agent_id)
    state_space = env.state_space(agent_id)

    shared_policy = PolicyNetwork(observation_space, action_space, device)
    shared_value = ValueNetwork(observation_space, state_space, action_space, device)

    models = {
        uid: {
            "policy": shared_policy,
            "value": shared_value,
        }
        for uid in env.possible_agents
    }
    memories = {
        uid: RandomMemory(memory_size=TRAIN.rollout_steps, num_envs=env.num_envs, device=device)
        for uid in env.possible_agents
    }

    cfg = MAPPO_CFG(
        experiment={"directory": RUNS_DIR, "experiment_name": f"pybh_mappo_{env.possible_agents[0]}"},
        rollouts=TRAIN.rollout_steps,
        learning_epochs=TRAIN.learning_epochs,
        mini_batches=TRAIN.mini_batches,
        learning_rate=1e-3,
        learning_rate_scheduler_kwargs={agent_id: ({}, {})},
        discount_factor=0.99,
        gae_lambda=0.95,
        observation_preprocessor_kwargs={agent_id: {}},
        state_preprocessor_kwargs={agent_id: {}},
        value_preprocessor_kwargs={agent_id: {}},
        entropy_loss_scale=0.0,
        value_loss_scale=1.0,
        grad_norm_clip=0.5,
        ratio_clip=0.2,
    )

    return MAPPO(
        possible_agents=env.possible_agents,
        models=models,
        memories=memories,
        observation_spaces=env.observation_spaces,
        state_spaces=env.state_spaces,
        action_spaces=env.action_spaces,
        device=device,
        cfg=cfg,
    )


def sample_smoke_action(agent: MAPPO, wrapped_env, observations, timestep: int, timesteps: int) -> StructuredSkillAction:
    agent_id = wrapped_env.possible_agents[0]
    with torch.no_grad():
        actions, _ = agent.act(
            observations=observations,
            states=wrapped_env.state(),
            timestep=timestep,
            timesteps=timesteps,
        )
    action_tensor = actions[agent_id].detach().cpu().reshape(-1).numpy()
    action = StructuredSkillAction(
        skill_type=int(round(float(action_tensor[0]))),
        target_x=float(action_tensor[1]),
        target_y=float(action_tensor[2]),
        target_theta=float(action_tensor[3]),
        pass_target=int(round(float(action_tensor[4]))),
    )
    if action.skill_name not in ENV.active_skill_names:
        action = StructuredSkillAction(skill_type=0)
    return action


def run_smoke(steps: int, backend: str, device: str, simrobot_profile: str) -> None:
    backend_cfg = replace(ENV.backend, mode=backend, simrobot_profile=simrobot_profile)
    raw_env, wrapped_env = build_env(backend_cfg)
    observations, infos = wrapped_env.reset()
    agent = build_mappo(wrapped_env, device)
    total_reward = 0.0
    for step in range(steps):
        action = sample_smoke_action(agent, wrapped_env, observations, step, steps)
        observations, rewards, terminated, truncated, infos = wrapped_env.step(
            {"player_1": torch.as_tensor(action_to_numpy(action), device=wrapped_env.device).view(1, -1)}
        )
        reward_value = float(rewards["player_1"].item())
        total_reward += reward_value
        print(
            f"[smoke] step={step + 1} skill={action.skill_name} "
            f"reward={reward_value:.3f} frame={infos['player_1']['bridge_frame']}"
        )
        if bool(terminated["player_1"].item()) or bool(truncated["player_1"].item()):
            break
    wrapped_env.close()
    raw_env.close()
    print(f"[smoke] total_reward={total_reward:.3f}")


def run_training(timesteps: int, backend: str, device: str, simrobot_profile: str) -> None:
    backend_cfg = replace(ENV.backend, mode=backend, simrobot_profile=simrobot_profile)
    raw_env, wrapped_env = build_env(backend_cfg)
    agent = build_mappo(wrapped_env, device)
    trainer_cfg = SequentialTrainerCfg(
        timesteps=timesteps,
        headless=True,
        disable_progressbar=True,
        close_environment_at_exit=True,
    )
    agent.init(trainer_cfg=trainer_cfg)
    agent.enable_training_mode(True)

    observations, infos = wrapped_env.reset()
    states = wrapped_env.state()

    for timestep in range(timesteps):
        agent.pre_interaction(timestep=timestep, timesteps=timesteps)
        with torch.no_grad():
            actions, _ = agent.act(
                observations=observations,
                states=states,
                timestep=timestep,
                timesteps=timesteps,
            )
            next_observations, rewards, terminated, truncated, infos = wrapped_env.step(actions)
            next_states = wrapped_env.state()
            agent.record_transition(
                observations=observations,
                states=states,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
                infos=infos,
                timestep=timestep,
                timesteps=timesteps,
            )
        agent.post_interaction(timestep=timestep, timesteps=timesteps)

        if not wrapped_env.agents:
            observations, infos = wrapped_env.reset()
            states = wrapped_env.state()
        else:
            observations = next_observations
            states = next_states

    wrapped_env.close()
    raw_env.close()
    print(f"[train] completed timesteps={timesteps}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or smoke-test the pybh SkillRequest environment")
    parser.add_argument("--backend", choices=("simrobot", "cpp", "python"), default="simrobot")
    parser.add_argument("--simrobot-profile", choices=("2d", "3d"), default=ENV.backend.simrobot_profile)
    parser.add_argument("--mode", choices=("smoke", "train"), default="smoke")
    parser.add_argument("--smoke-steps", type=int, default=TRAIN.smoke_steps)
    parser.add_argument("--train-steps", type=int, default=TRAIN.trainer_timesteps)
    parser.add_argument("--device", default=TRAIN.device)
    args = parser.parse_args()

    if args.mode == "smoke":
        run_smoke(args.smoke_steps, backend=args.backend, device=args.device, simrobot_profile=args.simrobot_profile)
    else:
        run_training(
            args.train_steps,
            backend=args.backend,
            device=args.device,
            simrobot_profile=args.simrobot_profile,
        )


if __name__ == "__main__":
    main()
