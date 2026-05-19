from __future__ import annotations

import json
import importlib
import os
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from types import MethodType
from typing import Any, Dict, Mapping

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, Model
from skrl.multi_agents.torch.mappo import MAPPO, MAPPO_CFG
from skrl.trainers.torch import SequentialTrainerCfg

from action import SKILL_ORDER, StructuredSkillAction, action_to_numpy, skill_mask
from config import ENV, TRAIN, BackendConfig, EnvConfig, TrainConfig
from curriculum import get_stage_scene_path, get_stage_spec, promote_stage
from environment import RealSimRobotEnv
from parallel_environment import ParallelSlotSimRobotEnv
from visible_parallel_runtime import VisibleParallelRuntime
from wrapper import RealSimRobotEnvWrapper
from wire_audit import build_wire_audit

RUNS_DIR = Path(__file__).resolve().parent / "runs"


@dataclass(frozen=True)
class ScriptRunConfig:
    mode: str = "train"
    backend: str = "simrobot_visible"
    obs_mode: str = ENV.backend.obs_mode
    simrobot_profile: str = "3d_visible"
    simrobot_launch: bool = False
    embedded_qt_platform: str | None = ENV.backend.embedded_qt_platform
    embedded_3d_preflight: bool = ENV.backend.embedded_3d_preflight
    smoke_steps: int = TRAIN.smoke_steps
    train_steps: int = 800_000
    learner_device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    stage_id: int = -1
    max_stage_id: int = 4
    save_telemetry: bool = True
    save_plots: bool = True
    capture_eval_steps: bool = True
    telemetry_tag: str = "overnight_stage0_to_3"
    experiment_name: str = TRAIN.experiment_name
    max_episode_steps: int = 220
    rollout_steps: int = 256
    learning_epochs: int = 4
    mini_batches: int = 4
    eval_interval: int = 5_000
    checkpoint_interval: int = 5_000
    eval_episodes: int = 4
    abort_on_wire_fail: bool = False
    parallel_player_numbers: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    visible_parallel_display_base: int = 410


@dataclass(frozen=True)
class TelemetryPaths:
    session_dir: Path
    csv_dir: Path
    plots_dir: Path
    run_config_path: Path
    smoke_steps_csv: Path
    smoke_summary_csv: Path
    eval_summary_csv: Path
    eval_episodes_csv: Path


IDE_RUN = ScriptRunConfig()


def _env_bool(name: str, fallback: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return fallback
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_player_numbers(value: str | None, fallback: tuple[int, ...]) -> tuple[int, ...]:
    if not value:
        return fallback
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _run_cfg_from_env(base: ScriptRunConfig = IDE_RUN) -> ScriptRunConfig:
    cfg = base
    if mode := os.environ.get("RL_RUN_MODE"):
        cfg = replace(cfg, mode=mode)
    if train_steps := os.environ.get("RL_TRAIN_STEPS"):
        cfg = replace(cfg, train_steps=int(train_steps))
    if learner_device := os.environ.get("RL_LEARNER_DEVICE"):
        cfg = replace(cfg, learner_device=learner_device)
    if smoke_steps := os.environ.get("RL_SMOKE_STEPS"):
        cfg = replace(cfg, smoke_steps=int(smoke_steps))
    if stage_id := os.environ.get("RL_STAGE_ID"):
        cfg = replace(cfg, stage_id=int(stage_id))
    if max_stage_id := os.environ.get("RL_MAX_STAGE_ID"):
        cfg = replace(cfg, max_stage_id=int(max_stage_id))
    if max_episode_steps := os.environ.get("RL_MAX_EPISODE_STEPS"):
        cfg = replace(cfg, max_episode_steps=int(max_episode_steps))
    if rollout_steps := os.environ.get("RL_ROLLOUT_STEPS"):
        cfg = replace(cfg, rollout_steps=int(rollout_steps))
    if learning_epochs := os.environ.get("RL_LEARNING_EPOCHS"):
        cfg = replace(cfg, learning_epochs=int(learning_epochs))
    if mini_batches := os.environ.get("RL_MINI_BATCHES"):
        cfg = replace(cfg, mini_batches=int(mini_batches))
    if eval_interval := os.environ.get("RL_EVAL_INTERVAL"):
        cfg = replace(cfg, eval_interval=int(eval_interval))
    if checkpoint_interval := os.environ.get("RL_CHECKPOINT_INTERVAL"):
        cfg = replace(cfg, checkpoint_interval=int(checkpoint_interval))
    if eval_episodes := os.environ.get("RL_EVAL_EPISODES"):
        cfg = replace(cfg, eval_episodes=int(eval_episodes))
    cfg = replace(
        cfg,
        save_telemetry=_env_bool("RL_SAVE_TELEMETRY", cfg.save_telemetry),
        save_plots=_env_bool("RL_SAVE_PLOTS", cfg.save_plots),
        capture_eval_steps=_env_bool("RL_CAPTURE_EVAL_STEPS", cfg.capture_eval_steps),
        abort_on_wire_fail=_env_bool("RL_ABORT_ON_WIRE_FAIL", cfg.abort_on_wire_fail),
    )
    if telemetry_tag := os.environ.get("RL_TELEMETRY_TAG"):
        cfg = replace(cfg, telemetry_tag=telemetry_tag)
    if experiment_name := os.environ.get("RL_EXPERIMENT_NAME"):
        cfg = replace(cfg, experiment_name=experiment_name)
    cfg = replace(
        cfg,
        parallel_player_numbers=_parse_player_numbers(
            os.environ.get("RL_PARALLEL_PLAYER_NUMBERS"),
            cfg.parallel_player_numbers,
        ),
    )
    if display_base := os.environ.get("RL_VISIBLE_DISPLAY_BASE"):
        cfg = replace(cfg, visible_parallel_display_base=int(display_base))
    return cfg


def _pd():
    return importlib.import_module("pandas")


def _plt():
    import matplotlib
    matplotlib.use("Agg")  # must be set before pyplot import; prevents TkAgg objects in worker threads
    return importlib.import_module("matplotlib.pyplot")


def _coerce_scalar_metric(value: Any) -> float:
    if isinstance(value, (list, tuple)):
        if not value:
            return float("nan")
        return _coerce_scalar_metric(value[-1])
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return float("nan")
        return _coerce_scalar_metric(value.reshape(-1)[-1].item())
    if torch.is_tensor(value):
        if value.numel() == 0:
            return float("nan")
        return float(value.detach().reshape(-1)[-1].item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


class HybridPolicyDistribution:
    def __init__(self, skill_distribution: Categorical, param_distribution: Normal):
        self.skill_distribution = skill_distribution
        self.param_distribution = param_distribution
        self.stddev = param_distribution.stddev

    def entropy(self) -> torch.Tensor:
        return self.skill_distribution.entropy().unsqueeze(-1) + self.param_distribution.entropy().sum(dim=-1, keepdim=True)


class PolicyNetwork(Model):
    def __init__(self, observation_space, action_space, device, hidden_size: int):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        self.num_skills = len(SKILL_ORDER)
        self.num_params = self.num_actions - 1
        if self.num_params != 4:
            raise ValueError(f"Hybrid policy expects 4 continuous params, got {self.num_params}")
        self.trunk = nn.Sequential(
            nn.Linear(self.num_observations, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
        )
        self.skill_head = nn.Linear(hidden_size, self.num_skills)
        self.param_head = nn.Linear(hidden_size, self.num_params)
        self.log_std_parameter = nn.Parameter(torch.full((self.num_params,), -0.5, dtype=torch.float32))
        self._distribution: HybridPolicyDistribution | None = None
        action_low = torch.as_tensor(action_space.low, dtype=torch.float32, device=self.device)
        action_high = torch.as_tensor(action_space.high, dtype=torch.float32, device=self.device)
        self.register_buffer("param_low", action_low[1:].clone())
        self.register_buffer("param_high", action_high[1:].clone())
        self.register_buffer("active_skill_mask", torch.ones(self.num_skills, dtype=torch.float32, device=self.device))
        with torch.no_grad():
            # Separate heads keep discrete skill selection gradients from being
            # entangled with BHuman's continuous target parameters.
            nn.init.orthogonal_(self.skill_head.weight, gain=0.01)
            nn.init.zeros_(self.skill_head.bias)
            nn.init.orthogonal_(self.param_head.weight, gain=0.01)
            nn.init.zeros_(self.param_head.bias)
        self.to(device)

    def set_active_skill_names(self, active_skill_names: tuple[str, ...]) -> None:
        mask = torch.as_tensor(skill_mask(active_skill_names), dtype=torch.float32, device=self.device)
        if float(mask.sum().item()) <= 0.0:
            raise ValueError("Hybrid policy skill mask must enable at least one skill")
        self.active_skill_mask.copy_(mask)

    def _masked_skill_logits(self, skill_logits: torch.Tensor) -> torch.Tensor:
        mask = self.active_skill_mask.to(device=skill_logits.device, dtype=skill_logits.dtype)
        return skill_logits.masked_fill(mask.unsqueeze(0) <= 0.0, -1.0e9)

    def compute(self, inputs, role=""):
        observations = inputs["observations"]
        features = self.trunk(observations)
        skill_logits = self.skill_head(features)
        param_mean = self.param_head(features)
        return skill_logits, {
            "param_mean": param_mean,
            "log_std": self.log_std_parameter.expand_as(param_mean),
        }

    def act(self, inputs, role=""):
        skill_logits, outputs = self.compute(inputs, role)
        param_mean = outputs["param_mean"]
        log_std = torch.clamp(outputs["log_std"], min=-2.0, max=0.0)
        outputs["log_std"] = log_std

        masked_skill_logits = self._masked_skill_logits(skill_logits)
        skill_distribution = Categorical(logits=masked_skill_logits)
        param_distribution = Normal(param_mean, log_std.exp())
        self._distribution = HybridPolicyDistribution(skill_distribution, param_distribution)

        taken_actions = inputs.get("taken_actions")
        if taken_actions is None:
            skill_ids = skill_distribution.sample()
            params = param_distribution.rsample()
            params = torch.clamp(params, min=self.param_low, max=self.param_high)
        else:
            flat = taken_actions.reshape(taken_actions.shape[0], -1)
            skill_ids = torch.clamp(flat[:, 0].round().long(), min=0, max=self.num_skills - 1)
            params = torch.clamp(flat[:, 1:], min=self.param_low, max=self.param_high)

        skill_log_prob = skill_distribution.log_prob(skill_ids).unsqueeze(-1)
        param_log_prob = param_distribution.log_prob(params).sum(dim=-1, keepdim=True)
        actions = torch.cat((skill_ids.to(dtype=param_mean.dtype).unsqueeze(-1), params), dim=-1)
        skill_probs = torch.softmax(masked_skill_logits, dim=-1)

        outputs.update(
            {
                "log_prob": skill_log_prob + param_log_prob,
                "skill_log_prob": skill_log_prob,
                "param_log_prob": param_log_prob,
                "skill_logits": skill_logits,
                "masked_skill_logits": masked_skill_logits,
                "skill_probs": skill_probs,
                "skill_entropy": skill_distribution.entropy().unsqueeze(-1),
                "param_entropy": param_distribution.entropy().sum(dim=-1, keepdim=True),
                "mean_actions": torch.cat(
                    (torch.argmax(masked_skill_logits, dim=-1, keepdim=True).to(dtype=param_mean.dtype), param_mean),
                    dim=-1,
                ),
            }
        )
        return actions, outputs

    def random_act(self, inputs, role=""):
        return self.act(inputs, role=role)

    def get_entropy(self, *, role: str = "") -> torch.Tensor:
        if self._distribution is None:
            return torch.tensor(0.0, device=self.device)
        return self._distribution.entropy()

    def distribution(self, *, role: str = "") -> HybridPolicyDistribution:
        return self._distribution


class ValueNetwork(DeterministicMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device, hidden_size: int):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self, clip_actions=False)
        self.net = nn.Sequential(
            nn.Linear(self.num_states, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 1),
        )
        self.to(device)

    def compute(self, inputs, role=""):
        states = inputs.get("states", inputs["observations"])
        return self.net(states), {}


def make_env_config(stage_id: int, backend_cfg: BackendConfig) -> EnvConfig:
    stage_spec = get_stage_spec(stage_id)
    stage_backend_cfg = replace(
        backend_cfg,
        simrobot_scene_path=get_stage_scene_path(stage_id, backend_cfg.simrobot_profile),
    )
    return replace(
        ENV,
        stage_id=stage_id,
        backend=stage_backend_cfg,
        active_skill_names=stage_spec.active_skill_names,
    )


def build_env(env_cfg: EnvConfig):
    if len(env_cfg.parallel_player_numbers) > 1:
        env = ParallelSlotSimRobotEnv(env_cfg=env_cfg)
    else:
        env = RealSimRobotEnv(env_cfg=env_cfg)
    wrapped_env = wrap_env(env, wrapper="pettingzoo")
    if hasattr(wrapped_env, "_device"):
        wrapped_env._device = torch.device("cpu")
    if hasattr(wrapped_env, "_seed"):
        wrapped_env._seed = int(env_cfg.seed)
    eval_wrapper = RealSimRobotEnvWrapper(env)
    return env, wrapped_env, eval_wrapper


def _visible_runtime_for_run(run_cfg: ScriptRunConfig, env_cfg: EnvConfig) -> VisibleParallelRuntime | None:
    if env_cfg.backend.mode != "simrobot_visible" or len(env_cfg.parallel_player_numbers) <= 1:
        return None
    shm_name = os.environ.get("PYBH_RL_SHM_NAME")
    if not shm_name:
        shm_name = f"/pybh_rl_{os.getpid()}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        os.environ["PYBH_RL_SHM_NAME"] = shm_name
    print(f"[shm] PYBH_RL_SHM_NAME={shm_name}", flush=True)
    # RL_SCENE_NAME overrides the SimRobot scene. ci=60 ("RL1v0_3D_visible_ci60") was tested
    # and abandoned: BallPerceptor fails at 60fps. Default ci=30 gives ~27 steps/sec.
    scene_name = os.environ.get("RL_SCENE_NAME", "RL1v0_3D_visible")
    return VisibleParallelRuntime(
        player_numbers=env_cfg.parallel_player_numbers,
        shm_name=shm_name,
        display_base=run_cfg.visible_parallel_display_base,
        simrobot_bin=env_cfg.backend.simrobot_bin,
        scene_name=scene_name,
    )


def build_mappo(
    env,
    device: str,
    train_cfg: TrainConfig,
    active_skill_names: tuple[str, ...],
) -> tuple[MAPPO, Dict[str, torch.nn.Module]]:
    agent_id = env.possible_agents[0]
    observation_space = env.observation_space(agent_id)
    action_space = env.action_space(agent_id)
    state_space = env.state_space(agent_id)

    shared_policy = PolicyNetwork(observation_space, action_space, device, train_cfg.hidden_size)
    shared_policy.set_active_skill_names(active_skill_names)
    shared_value = ValueNetwork(observation_space, state_space, action_space, device, train_cfg.hidden_size)
    shared_policy.to(device)
    shared_value.net = shared_value.net.to(device)
    shared_value.to(device)

    models = {
        uid: {
            "policy": shared_policy,
            "value": shared_value,
        }
        for uid in env.possible_agents
    }
    memories = {
        uid: RandomMemory(memory_size=train_cfg.rollout_steps, num_envs=env.num_envs, device=device)
        for uid in env.possible_agents
    }

    empty_agent_kwargs = {uid: ({}, {}) for uid in env.possible_agents}
    empty_preprocessor_kwargs = {uid: {} for uid in env.possible_agents}
    cfg = MAPPO_CFG(
        experiment={
            "directory": str(RUNS_DIR),
            "experiment_name": train_cfg.experiment_name,
            "write_interval": max(1, min(train_cfg.rollout_steps, train_cfg.eval_interval)),
            "checkpoint_interval": 0,
        },
        rollouts=train_cfg.rollout_steps,
        learning_epochs=train_cfg.learning_epochs,
        mini_batches=train_cfg.mini_batches,
        learning_rate=train_cfg.learning_rate,
        learning_rate_scheduler_kwargs=empty_agent_kwargs,
        discount_factor=train_cfg.discount_factor,
        gae_lambda=train_cfg.gae_lambda,
        observation_preprocessor_kwargs=empty_preprocessor_kwargs,
        state_preprocessor_kwargs=empty_preprocessor_kwargs,
        value_preprocessor_kwargs=empty_preprocessor_kwargs,
        entropy_loss_scale=train_cfg.entropy_loss_scale,
        value_loss_scale=train_cfg.value_loss_scale,
        grad_norm_clip=train_cfg.grad_norm_clip,
        ratio_clip=train_cfg.ratio_clip,
    )

    agent = MAPPO(
        possible_agents=env.possible_agents,
        models=models,
        memories=memories,
        observation_spaces=env.observation_spaces,
        state_spaces=env.state_spaces,
        action_spaces=env.action_spaces,
        device=device,
        cfg=cfg,
    )
    agent._last_tracking_snapshot = {}

    original_write_tracking_data = agent.write_tracking_data

    def _write_tracking_data_with_snapshot(self, *, timestep: int, timesteps: int) -> None:
        # Start from previous snapshot so values persist across skrl's internal clear()
        snapshot: Dict[str, float] = dict(getattr(self, "_last_tracking_snapshot", {}))
        tracking_data = getattr(self, "tracking_data", {})
        if isinstance(tracking_data, Mapping):
            fresh_keys: list[str] = []
            for key, values in tracking_data.items():
                if isinstance(values, (list, tuple)) and values:
                    if key.endswith("(min)"):
                        snapshot[key] = float(np.min(values))
                    elif key.endswith("(max)"):
                        snapshot[key] = float(np.max(values))
                    else:
                        snapshot[key] = float(np.mean(values))
                    fresh_keys.append(key)
                else:
                    coerced = _coerce_scalar_metric(values)
                    if np.isfinite(coerced):
                        snapshot[key] = coerced
                        fresh_keys.append(key)
            if fresh_keys and not getattr(self, "_tracking_keys_logged", False):
                print(f"[tracking] available keys at first update: {sorted(fresh_keys)}", flush=True)
                self._tracking_keys_logged = True
        self._last_tracking_snapshot = snapshot
        original_write_tracking_data(timestep=timestep, timesteps=timesteps)

    agent.write_tracking_data = MethodType(_write_tracking_data_with_snapshot, agent)
    return agent, {"policy": shared_policy, "value": shared_value}


def _move_to_device(value: Any, device: str):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_move_to_device(item, device) for item in value)
    return value


def _agent_state_from_env(raw_env: RealSimRobotEnv, agent_id: str, device: str) -> Dict[str, torch.Tensor]:
    state = _tensorize_eval_state(raw_env.state(), device=device)
    return {agent: state for agent in raw_env.possible_agents}


def _flatten_mapping(prefix: str, data: Mapping[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in data.items():
        column = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(_flatten_mapping(column, value))
        else:
            flat[column] = value
    return flat


def _prepare_telemetry_paths(run_cfg: ScriptRunConfig) -> TelemetryPaths:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_name = f"{run_cfg.telemetry_tag}_{run_cfg.mode}_stage{run_cfg.stage_id}_{timestamp}"
    session_dir = RUNS_DIR / run_cfg.experiment_name / "telemetry" / session_name
    csv_dir = session_dir / "csv"
    plots_dir = session_dir / "plots"
    csv_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return TelemetryPaths(
        session_dir=session_dir,
        csv_dir=csv_dir,
        plots_dir=plots_dir,
        run_config_path=session_dir / "run_config.json",
        smoke_steps_csv=csv_dir / "smoke_steps.csv",
        smoke_summary_csv=csv_dir / "smoke_summary.csv",
        eval_summary_csv=csv_dir / "eval_summary.csv",
        eval_episodes_csv=csv_dir / "eval_episodes.csv",
    )


def _write_run_config(
    paths: TelemetryPaths,
    run_cfg: ScriptRunConfig,
    backend_cfg: BackendConfig,
    train_cfg: TrainConfig,
) -> None:
    payload = {
        "run": asdict(run_cfg),
        "backend": {
            "mode": backend_cfg.mode,
            "obs_mode": backend_cfg.obs_mode,
            "player_number": backend_cfg.player_number,
            "frames_per_step": backend_cfg.frames_per_step,
            "obs_timeout_ms": backend_cfg.obs_timeout_ms,
            "simrobot_profile": backend_cfg.simrobot_profile,
            "simrobot_scene_path": None if backend_cfg.simrobot_scene_path is None else str(backend_cfg.simrobot_scene_path),
            "simrobot_robot_name": backend_cfg.simrobot_robot_name,
            "simrobot_theta_offset": backend_cfg.simrobot_theta_offset,
            "simrobot_launch": backend_cfg.simrobot_launch,
            "embedded_qt_platform": backend_cfg.embedded_qt_platform,
            "embedded_3d_preflight": backend_cfg.embedded_3d_preflight,
            "simrobot_bin": str(backend_cfg.simrobot_bin),
        },
        "parallel_player_numbers": list(run_cfg.parallel_player_numbers),
        "wire_audit": asdict(train_cfg.wire_audit),
    }
    paths.run_config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _append_csv_row(path: Path, row: Mapping[str, Any]) -> None:
    pd = _pd()
    new_df = pd.DataFrame([dict(row)])
    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, new_df], ignore_index=True, sort=False)
    else:
        combined = new_df
    combined.to_csv(path, index=False)


def _plot_smoke_steps(paths: TelemetryPaths) -> None:
    if not paths.smoke_steps_csv.exists():
        return
    pd = _pd()
    plt = _plt()
    df = pd.read_csv(paths.smoke_steps_csv)
    if df.empty:
        return
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    axes = axes.ravel()

    axes[0].plot(df["step"], df["reward"], marker="o")
    axes[0].set_title("Smoke Reward by Step")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Reward")

    if "ball_distance_mm" in df.columns:
        axes[1].plot(df["step"], df["ball_distance_mm"], marker="o", color="tab:orange")
        axes[1].set_title("Ball Distance")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("mm")
    else:
        axes[1].axis("off")

    if "ball_angle_deg" in df.columns:
        axes[2].plot(df["step"], df["ball_angle_deg"], marker="o", color="tab:green")
        axes[2].set_title("Ball Angle")
        axes[2].set_xlabel("Step")
        axes[2].set_ylabel("deg")
    else:
        axes[2].axis("off")

    takeover = df["takeover_envelope"] if "takeover_envelope" in df.columns else pd.Series(np.zeros(len(df)))
    axes[3].step(df["step"], takeover, where="post", label="takeover")
    if "projection_applied" in df.columns:
        axes[3].step(df["step"], df["projection_applied"], where="post", label="projection")
    axes[3].set_title("Takeover / Projection")
    axes[3].set_xlabel("Step")
    axes[3].legend()

    if "ball_confidence_score" in df.columns:
        axes[4].plot(df["step"], df["ball_confidence_score"], marker="o", color="tab:red")
        axes[4].set_title("Ball Confidence")
        axes[4].set_xlabel("Step")
        axes[4].set_ylabel("score")
    else:
        axes[4].axis("off")

    if "ball_trusted_mask" in df.columns:
        axes[5].step(df["step"], df["ball_trusted_mask"], where="post", label="trusted")
        if "ball_rescued_mask" in df.columns:
            axes[5].step(df["step"], df["ball_rescued_mask"], where="post", label="rescued")
        axes[5].set_title("Ball Masks")
        axes[5].set_xlabel("Step")
        axes[5].legend()
    else:
        axes[5].axis("off")

    fig.tight_layout()
    fig.savefig(paths.plots_dir / "smoke_overview.png", dpi=150)
    plt.close(fig)


def _plot_eval_metrics(paths: TelemetryPaths) -> None:
    if not paths.eval_summary_csv.exists():
        return
    pd = _pd()
    plt = _plt()
    df = pd.read_csv(paths.eval_summary_csv)
    if df.empty:
        return

    metric_columns = [
        "episode_reward_mean",
        "clean_takeover_rate",
        "clean_finish_rate",
        "thrash_rate",
        "collision_rate",
        "stability_score",
        "ball_trusted_rate",
        "ball_rescued_rate",
        "misleading_seen_synthetic_rate",
    ]
    available = [column for column in metric_columns if column in df.columns]
    if not available:
        return

    fig, axes = plt.subplots(len(available), 1, figsize=(10, 3 * len(available)), sharex=True)
    if len(available) == 1:
        axes = [axes]
    for axis, column in zip(axes, available):
        axis.plot(df["timestep"], df[column], marker="o")
        axis.set_ylabel(column)
        axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Timestep")
    fig.suptitle("Evaluation Metrics")
    fig.tight_layout()
    fig.savefig(paths.plots_dir / "eval_metrics_overview.png", dpi=150)
    plt.close(fig)

    reward_term_columns = [column for column in df.columns if column.startswith("reward_terms_mean.")]
    if reward_term_columns:
        fig, axis = plt.subplots(figsize=(12, 6))
        for column in sorted(reward_term_columns):
            axis.plot(df["timestep"], df[column], marker="o", label=column.removeprefix("reward_terms_mean."))
        axis.set_title("Reward Terms Mean")
        axis.set_xlabel("Timestep")
        axis.grid(True, alpha=0.3)
        axis.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(paths.plots_dir / "eval_reward_terms.png", dpi=150)
        plt.close(fig)


def _tensorize_eval_obs(obs: Mapping[str, np.ndarray], device: str) -> Dict[str, torch.Tensor]:
    return {
        agent: torch.as_tensor(value, device=device, dtype=torch.float32).view(1, -1)
        for agent, value in obs.items()
    }


def _tensorize_eval_state(state: np.ndarray, device: str) -> torch.Tensor:
    return torch.as_tensor(state, device=device, dtype=torch.float32).view(1, -1)


def _extract_tracking_dict(agent: MAPPO) -> Mapping[str, object]:
    # Prefer the snapshot: it's built after each write_tracking_data call and
    # persists loss keys across skrl's internal clear() between rollouts.
    # The live tracking_data only contains reward/episode metrics mid-rollout.
    snapshot = getattr(agent, "_last_tracking_snapshot", None)
    if isinstance(snapshot, Mapping) and snapshot:
        return snapshot
    for attr_name in ("track_data", "tracking_data", "_track_data", "_tracking_data"):
        value = getattr(agent, attr_name, None)
        if callable(value):
            try:
                value = value()
            except TypeError:
                continue
        if isinstance(value, Mapping):
            if value:
                return value
    return {}


def _extract_agent_stat(agent: MAPPO, *candidate_keys: str) -> float:
    tracking = _extract_tracking_dict(agent)
    for key in candidate_keys:
        if key in tracking:
            value = _coerce_scalar_metric(tracking[key])
            if np.isfinite(value):
                return value
    return float("nan")


def _diagnose_nan_source(agent: MAPPO, *candidate_keys: str) -> str:
    """Return a one-line string explaining WHY _extract_agent_stat returned NaN.

    Returns one of:
      "key_not_found"   — none of the candidate keys exist in the tracking dict
      "value_nan"       — key found, stored value is NaN  (genuine numerical NaN)
      "value_inf"       — key found, stored value is ±Inf (numerical overflow)
    """
    tracking = _extract_tracking_dict(agent)
    if not tracking:
        return "key_not_found (tracking dict is empty)"
    for key in candidate_keys:
        if key in tracking:
            value = _coerce_scalar_metric(tracking[key])
            if np.isnan(value):
                return f"value_nan (key='{key}' raw={tracking[key]!r})"
            if np.isinf(value):
                return f"value_inf (key='{key}' raw={tracking[key]!r})"
            return f"value_finite={value:.4f} (key='{key}')"
    return f"key_not_found (tried={candidate_keys!r}, available_sample={list(tracking)[:4]!r})"


def sample_smoke_action(
    agent: MAPPO,
    raw_env: RealSimRobotEnv,
    wrapped_env,
    observations,
    timestep: int,
    timesteps: int,
    device: str,
) -> StructuredSkillAction:
    agent_id = wrapped_env.possible_agents[0]
    with torch.no_grad():
        obs_device = _move_to_device(observations, device)
        actions, _ = agent.act(
            observations=obs_device,
            states=_agent_state_from_env(raw_env, agent_id, device),
            timestep=timestep,
            timesteps=timesteps,
        )
    action_tensor = actions[agent_id].detach().cpu().reshape(-1).numpy()
    proposed_action = StructuredSkillAction(
        skill_type=int(round(float(action_tensor[0]))),
        target_x=float(action_tensor[1]),
        target_y=float(action_tensor[2]),
        target_theta=float(action_tensor[3]),
        pass_target=int(round(float(action_tensor[4]))),
    )
    single_env = next(iter(raw_env._envs.values())) if isinstance(raw_env, ParallelSlotSimRobotEnv) else raw_env
    if single_env._last_backend_obs is None:
        return proposed_action
    projected_action, _ = single_env.project_action(proposed_action, obs=single_env._last_backend_obs)
    return projected_action


def _aggregate_eval_metrics(episode_results: list[Mapping[str, object]]) -> Dict[str, float]:
    if not episode_results:
        return {}

    metric_names = [
        "clean_walk",
        "clean_takeover",
        "clean_finish",
        "clean_episode",
        "collision_episode",
        "takeover_success",
        "goal_rate",
        "thrash_rate",
        "mean_time_to_takeover",
        "walk_compat_rate",
        "dribble_compat_rate",
        "shoot_compat_rate",
        "bad_walk_rate",
        "bad_dribble_rate",
        "bad_shoot_rate",
        "policy_stand_rate",
        "policy_walk_rate",
        "policy_dribble_rate",
        "policy_shoot_rate",
        "executed_stand_rate",
        "executed_walk_rate",
        "executed_dribble_rate",
        "executed_shoot_rate",
        "policy_skill_prob_walk_mean",
        "policy_skill_prob_dribble_mean",
        "policy_skill_prob_shoot_mean",
        "obstacle_clearance_success",
        "dynamic_obstacle_success",
        "corridor_keep_rate",
        "ball_progress_post_takeover_mean",
        "stability_score",
        "skill_projection_rate",
        "handoff_override_rate",
        "target_repair_rate",
        "policy_skill_entropy",
        "policy_param_std_mean",
        "synthetic_pose_rate",
        "synthetic_ball_rate",
        "obs_mode_estimated_rate",
        "obs_mode_corrected_rate",
        "pose_alignment_warning_rate",
        "ball_stale_rate",
        "ball_collapsed_rate",
        "corrected_exported_robot_pose_rate",
        "corrected_exported_ball_rate",
        "ball_export_fresh_rate",
        "ball_export_source_estimated_rate",
        "ball_export_source_reset_rate",
        "ball_export_source_dynamic_rate",
        "natural_ball_seen_rate",
        "natural_ball_consistent_rate",
        "ball_visible_rate",
        "ball_trusted_rate",
        "ball_semi_trusted_rate",
        "ball_degraded_rate",
        "ball_rescued_rate",
        "ball_lost_rate",
        "misleading_seen_synthetic_rate",
        "fov_perception_failure_rate",
        "ball_confidence_score_mean",
        "first_ball_seen_after_reset_ms",
        "first_fresh_export_after_reset_ms",
        "first_non_synthetic_ball_after_reset_ms",
        "first_trusted_ball_after_reset_ms",
        "first_natural_ball_seen_after_reset_ms",
        "episode_reward_mean",
        "episode_reward_total",
    ]
    summary: Dict[str, float] = {}
    for metric_name in metric_names:
        values = [
            float(result.get("episode_metrics", {}).get(metric_name, 0.0))
            for result in episode_results
            if result.get("episode_metrics")
        ]
        if values:
            summary[metric_name] = float(np.mean(values))

    summary["clean_walk_rate"] = summary.get("clean_walk", 0.0)
    summary["clean_takeover_rate"] = summary.get("clean_takeover", 0.0)
    summary["clean_finish_rate"] = summary.get("clean_finish", 0.0)
    summary["clean_episode_rate"] = summary.get("clean_episode", 0.0)
    summary["collision_rate"] = summary.get("collision_episode", 0.0)
    summary["takeover_success_rate"] = summary.get("takeover_success", 0.0)
    summary["win_rate"] = summary.get("goal_rate", 0.0)
    summary["walk_in_walk_zone_rate"] = summary.get("walk_compat_rate", 0.0)
    summary["dribble_in_dribble_zone_rate"] = summary.get("dribble_compat_rate", 0.0)
    summary["shoot_in_shoot_zone_rate"] = summary.get("shoot_compat_rate", 0.0)
    summary["bad_duel_rate"] = 0.0

    reward_terms_sum: Dict[str, float] = {}
    reward_weighted_terms_sum: Dict[str, float] = {}
    for result in episode_results:
        for key, value in result.get("reward_terms_sum", {}).items():
            reward_terms_sum[key] = float(reward_terms_sum.get(key, 0.0) + value)
        for key, value in result.get("reward_weighted_terms_sum", {}).items():
            reward_weighted_terms_sum[key] = float(reward_weighted_terms_sum.get(key, 0.0) + value)
    num_results = max(1, len(episode_results))
    summary["reward_terms_mean"] = {key: value / num_results for key, value in reward_terms_sum.items()}
    summary["reward_weighted_terms_mean"] = {key: value / num_results for key, value in reward_weighted_terms_sum.items()}
    return summary


def evaluate_agent(
    agent: MAPPO,
    eval_env: RealSimRobotEnvWrapper,
    *,
    device: str,
    episodes: int,
    stage_id: int,
    timestep: int,
    timesteps: int,
) -> tuple[Dict[str, float], list[Mapping[str, object]]]:
    previous_mode = getattr(agent, "training", None)
    agent.enable_training_mode(False)

    def policy_fn(obs, infos):
        del infos
        obs_tensor = _tensorize_eval_obs(obs, device=device)
        state_value = _tensorize_eval_state(eval_env._env.state(), device=device)
        state_tensor = {agent_id: state_value for agent_id in eval_env.possible_agents}
        with torch.no_grad():
            actions, outputs = agent.act(
                observations=obs_tensor,
                states=state_tensor,
                timestep=timestep,
                timesteps=timesteps,
            )
        safe_eval_actions = {k: torch.nan_to_num(v, nan=0.0, posinf=1.0, neginf=-1.0) for k, v in actions.items()}
        payloads = {}
        for agent_id, action_tensor in (
            (uid, safe_eval_actions[uid].detach().cpu().reshape(-1).numpy())
            for uid in eval_env.possible_agents
        ):
            payload = {
                "skill_type": int(round(float(action_tensor[0]))),
                "policy_skill_raw": float(action_tensor[0]),
                "target_x": float(action_tensor[1]),
                "target_y": float(action_tensor[2]),
                "target_theta": float(action_tensor[3]),
                "pass_target": int(round(float(action_tensor[4]))),
            }
            agent_outputs = outputs.get(agent_id, {})
            skill_probs = agent_outputs.get("skill_probs")
            if torch.is_tensor(skill_probs) and skill_probs.numel() >= len(SKILL_ORDER):
                flat_probs = skill_probs.detach().cpu().reshape(-1, len(SKILL_ORDER))[0].numpy()
                for skill_index, skill_name in enumerate(SKILL_ORDER):
                    payload[f"policy_skill_prob_{skill_name}"] = float(flat_probs[skill_index])
            skill_entropy = agent_outputs.get("skill_entropy")
            if torch.is_tensor(skill_entropy) and skill_entropy.numel():
                payload["policy_skill_entropy"] = float(skill_entropy.detach().cpu().reshape(-1)[0].item())
            log_std = agent_outputs.get("log_std")
            if torch.is_tensor(log_std) and log_std.numel():
                payload["policy_param_std_mean"] = float(log_std.detach().cpu().exp().mean().item())
            payloads[agent_id] = payload
        return payloads

    episode_results = [
        eval_env.collect_episode(policy_fn, seed=TRAIN.seed + episode_idx, capture_steps=True)
        for episode_idx in range(episodes)
    ]
    summary = _aggregate_eval_metrics(episode_results)
    summary["stage_id"] = float(stage_id)

    if previous_mode is not None:
        agent.enable_training_mode(bool(previous_mode))
    else:
        agent.enable_training_mode(True)
    return summary, episode_results


def save_checkpoint(
    models: Mapping[str, torch.nn.Module],
    *,
    train_cfg: TrainConfig,
    timestep: int,
    stage_id: int,
    eval_summary: Mapping[str, object] | None = None,
) -> Path:
    checkpoint_dir = RUNS_DIR / train_cfg.experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"checkpoint_stage{stage_id}_step{timestep}.pt"
    torch.save(
        {
            "timestep": int(timestep),
            "stage_id": int(stage_id),
            "policy_state_dict": models["policy"].state_dict(),
            "value_state_dict": models["value"].state_dict(),
            "eval_summary": dict(eval_summary or {}),
        },
        checkpoint_path,
    )
    return checkpoint_path


def print_eval_summary(
    stage_id: int,
    timestep: int,
    summary: Mapping[str, object],
    agent: MAPPO,
    *,
    wire_audit: Mapping[str, object],
) -> None:
    agent_id = "player_1"
    reward_mean = float(summary.get("episode_reward_mean", 0.0))
    clean_takeover_rate = float(summary.get("clean_takeover_rate", 0.0))
    takeover_success_rate = float(summary.get("takeover_success_rate", 0.0))
    clean_finish_rate = float(summary.get("clean_finish_rate", 0.0))
    thrash_rate = float(summary.get("thrash_rate", 0.0))
    collision_rate = float(summary.get("collision_rate", 0.0))
    mean_time_to_takeover = float(summary.get("mean_time_to_takeover", 0.0))
    stability_score = float(summary.get("stability_score", 0.0))
    ball_trusted_rate = float(summary.get("ball_trusted_rate", 0.0))
    ball_rescued_rate = float(summary.get("ball_rescued_rate", 0.0))
    ball_confidence_score_mean = float(summary.get("ball_confidence_score_mean", 0.0))
    policy_loss = _extract_agent_stat(agent, f"Loss / Policy loss ({agent_id})", "Loss / Policy loss", "policy_loss")
    value_loss = _extract_agent_stat(agent, f"Loss / Value loss ({agent_id})", "Loss / Value loss", "value_loss")
    entropy = _extract_agent_stat(
        agent,
        f"Loss / Entropy loss ({agent_id})",
        "Loss / Entropy loss",
        "entropy_loss",
        "entropy",
    )
    print(
        "[eval] "
        f"step={timestep} stage={stage_id} "
        f"reward_mean={reward_mean:.3f} "
        f"takeover_clean={clean_takeover_rate:.3f} takeover_success={takeover_success_rate:.3f} "
        f"finish={clean_finish_rate:.3f} "
        f"thrash={thrash_rate:.3f} collision={collision_rate:.3f} "
        f"ball_trusted={ball_trusted_rate:.3f} rescued={ball_rescued_rate:.3f} "
        f"ball_conf={ball_confidence_score_mean:.3f} "
        f"time_to_takeover={mean_time_to_takeover:.1f} stability={stability_score:.3f} "
        f"policy_loss={policy_loss:.4f} value_loss={value_loss:.4f} entropy={entropy:.4f} "
        f"wire={wire_audit.get('wire_audit_status', 'UNKNOWN')}",
        flush=True,
    )


def _build_smoke_step_row(step: int, reward_value: float, step_info: Mapping[str, Any]) -> Dict[str, Any]:
    raw_obs = step_info.get("raw_obs", {})
    ball_rel_x = float(raw_obs.get("ball_rel_x", 0.0))
    ball_rel_y = float(raw_obs.get("ball_rel_y", 0.0))
    return {
        "step": int(step),
        "stage_id": int(step_info.get("stage_id", 0)),
        "configured_obs_mode": step_info.get("configured_obs_mode"),
        "reward": float(reward_value),
        "policy_skill_name": step_info.get("policy_skill_name"),
        "executed_skill_name": step_info.get("executed_skill_name"),
        "preferred_skill_name": step_info.get("preferred_skill_name"),
        "policy_skill_index": int(step_info.get("policy_skill_index", 0)),
        "policy_skill_entropy": float(step_info.get("policy_skill_entropy", 0.0)),
        "policy_param_std_mean": float(step_info.get("policy_param_std_mean", 0.0)),
        "policy_skill_prob_stand": float(step_info.get("policy_skill_prob_stand", 0.0)),
        "policy_skill_prob_walk": float(step_info.get("policy_skill_prob_walk", 0.0)),
        "policy_skill_prob_dribble": float(step_info.get("policy_skill_prob_dribble", 0.0)),
        "policy_skill_prob_shoot": float(step_info.get("policy_skill_prob_shoot", 0.0)),
        "policy_skill_prob_pass": float(step_info.get("policy_skill_prob_pass", 0.0)),
        "policy_skill_prob_block": float(step_info.get("policy_skill_prob_block", 0.0)),
        "policy_skill_prob_mark": float(step_info.get("policy_skill_prob_mark", 0.0)),
        "policy_skill_prob_observe": float(step_info.get("policy_skill_prob_observe", 0.0)),
        "projection_applied": int(bool(step_info.get("projection_applied", False))),
        "synthetic_pose_observation": int(bool(step_info.get("synthetic_pose_observation", False))),
        "synthetic_ball_observation": int(bool(step_info.get("synthetic_ball_observation", False))),
        "synthetic_pose_reason": step_info.get("synthetic_pose_reason"),
        "synthetic_ball_reason": step_info.get("synthetic_ball_reason"),
        "ball_regime": step_info.get("ball_regime"),
        "ball_confidence_score": float(step_info.get("ball_confidence_score", 0.0)),
        "ball_visible_mask": int(bool(step_info.get("ball_visible_mask", False))),
        "ball_trusted_mask": int(bool(step_info.get("ball_trusted_mask", False))),
        "ball_semi_trusted_mask": int(bool(step_info.get("ball_semi_trusted_mask", False))),
        "ball_degraded_mask": int(bool(step_info.get("ball_degraded_mask", False))),
        "ball_rescued_mask": int(bool(step_info.get("ball_rescued_mask", False))),
        "ball_lost_mask": int(bool(step_info.get("ball_lost_mask", False))),
        "obs_export_mode": int(step_info.get("obs_export_mode", 0)),
        "corrected_exported_robot_pose": int(bool(step_info.get("corrected_exported_robot_pose", False))),
        "corrected_exported_ball": int(bool(step_info.get("corrected_exported_ball", False))),
        "ball_export_source": int(step_info.get("ball_export_source", 0)),
        "ball_export_fresh": int(bool(step_info.get("ball_export_fresh", False))),
        "natural_time_since_ball_seen": float(step_info.get("natural_time_since_ball_seen", 0.0)),
        "natural_ball_seen_percentage": float(step_info.get("natural_ball_seen_percentage", 0.0)),
        "natural_ball_consistent_with_game_state": int(bool(step_info.get("natural_ball_consistent_with_game_state", False))),
        "reset_applied_frame": int(step_info.get("reset_applied_frame", 0)),
        "dynamic_applied_frame": int(step_info.get("dynamic_applied_frame", 0)),
        "reset_applied_age_ms": int(step_info.get("reset_applied_age_ms", -1)),
        "dynamic_applied_age_ms": int(step_info.get("dynamic_applied_age_ms", -1)),
        "pose_alignment_error_mm": float(step_info.get("pose_alignment_error_mm", 0.0)),
        "pose_alignment_warning": int(bool(step_info.get("pose_alignment_warning", False))),
        "ball_observation_stale": int(bool(step_info.get("ball_observation_stale", False))),
        "ball_observation_collapsed": int(bool(step_info.get("ball_observation_collapsed", False))),
        "estimated_robot_x": float(step_info.get("estimated_robot_x", 0.0)),
        "estimated_robot_y": float(step_info.get("estimated_robot_y", 0.0)),
        "estimated_ball_x": float(step_info.get("estimated_ball_x", 0.0)),
        "estimated_ball_y": float(step_info.get("estimated_ball_y", 0.0)),
        "takeover_envelope": int(bool(step_info.get("takeover_envelope", False))),
        "walk_zone": int(bool(step_info.get("walk_zone", False))),
        "dribble_zone": int(bool(step_info.get("dribble_zone", False))),
        "shoot_zone": int(bool(step_info.get("shoot_zone", False))),
        "bridge_frame": int(step_info.get("bridge_frame", 0)),
        "ball_distance_mm": float(np.hypot(ball_rel_x, ball_rel_y)),
        "ball_angle_deg": float(np.degrees(np.arctan2(ball_rel_y, ball_rel_x))),
        "time_since_ball_seen": float(raw_obs.get("time_since_ball_seen", 0.0)),
        "ball_seen_percentage": float(raw_obs.get("ball_seen_percentage", 0.0)),
        "shot_opening_with_obstacles": float(raw_obs.get("shot_opening_with_obstacles", 0.0)),
        "nearest_opponent_front_dist": float(raw_obs.get("nearest_opponent_front_dist", 0.0)),
        "nearest_front_obstacle_dist": float(raw_obs.get("nearest_front_obstacle_dist", 0.0)),
        **_flatten_mapping("reward_terms", step_info.get("reward_terms", {})),
        **_flatten_mapping("reward_weighted_terms", step_info.get("reward_weighted_terms", {})),
    }


def _build_eval_summary_row(
    *,
    timestep: int,
    stage_id: int,
    summary: Mapping[str, object],
    agent: MAPPO,
    wire_audit: Mapping[str, object],
) -> Dict[str, Any]:
    agent_id = "player_1"
    policy = agent.policies.get(agent_id) if hasattr(agent, "policies") else None
    policy_distribution = policy.distribution(role="policy") if policy is not None and hasattr(policy, "distribution") else None
    row = {
        "timestep": int(timestep),
        "stage_id": int(stage_id),
        "policy_backend": "skrl_hybrid",
        "policy_continuous_std_mean": (
            float(policy_distribution.stddev.detach().mean().item())
            if policy_distribution is not None and hasattr(policy_distribution, "stddev")
            else float("nan")
        ),
        "policy_loss": _extract_agent_stat(agent, f"Loss / Policy loss ({agent_id})", "Loss / Policy loss", "policy_loss"),
        "value_loss": _extract_agent_stat(agent, f"Loss / Value loss ({agent_id})", "Loss / Value loss", "value_loss"),
        "entropy": _extract_agent_stat(
            agent,
            f"Loss / Entropy loss ({agent_id})",
            "Loss / Entropy loss",
            "entropy_loss",
            "entropy",
        ),
    }
    if policy is not None and hasattr(policy, "active_skill_mask"):
        mask = policy.active_skill_mask.detach().cpu().numpy().reshape(-1)
        for idx, skill_name in enumerate(SKILL_ORDER):
            row[f"policy_mask_{skill_name}"] = float(mask[idx])
    for key, value in summary.items():
        if isinstance(value, Mapping):
            row.update(_flatten_mapping(key, value))
        else:
            row[key] = value
    row.update(dict(wire_audit))
    return row


def _build_smoke_summary_row(
    *,
    steps: int,
    total_reward: float,
    final_info: Mapping[str, object],
    wire_audit: Mapping[str, object],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "steps": int(steps),
        "total_reward": float(total_reward),
        "configured_obs_mode": final_info.get("configured_obs_mode"),
        "backend_mode": final_info.get("backend_mode"),
        "stage_id": int(final_info.get("stage_id", 0)),
    }
    for prefix in ("running_metrics", "episode_metrics"):
        value = final_info.get(prefix, {})
        if isinstance(value, Mapping):
            row.update(_flatten_mapping(prefix, value))
    row.update(dict(wire_audit))
    return row


def _append_eval_episode_rows(
    paths: TelemetryPaths,
    *,
    timestep: int,
    stage_id: int,
    episode_results: list[Mapping[str, object]],
) -> None:
    for episode_index, result in enumerate(episode_results):
        episode_row: Dict[str, Any] = {
            "timestep": int(timestep),
            "stage_id": int(stage_id),
            "episode_index": int(episode_index),
            "steps": int(result.get("steps", 0)),
            "mean_reward": float(result.get("mean_reward", 0.0)),
        }
        episode_row.update(_flatten_mapping("episode_metrics", result.get("episode_metrics", {})))
        episode_row.update(_flatten_mapping("reward_terms_sum", result.get("reward_terms_sum", {})))
        episode_row.update(_flatten_mapping("reward_weighted_terms_sum", result.get("reward_weighted_terms_sum", {})))
        _append_csv_row(paths.eval_episodes_csv, episode_row)

        for step_record in result.get("step_records", []):
            step_row = {
                "timestep": int(timestep),
                "stage_id": int(stage_id),
                "episode_index": int(episode_index),
            }
            step_row.update(step_record)
            _append_csv_row(paths.eval_episodes_csv.with_name("eval_episode_steps.csv"), step_row)


def run_smoke(
    steps: int,
    backend: str,
    device: str,
    obs_mode: str,
    simrobot_profile: str,
    stage_id: int,
    simrobot_launch: bool = False,
    embedded_qt_platform: str | None = None,
    embedded_3d_preflight: bool = True,
    telemetry_paths: TelemetryPaths | None = None,
    save_plots: bool = True,
    abort_on_wire_fail: bool = False,
) -> None:
    backend_cfg = replace(
        ENV.backend,
        mode=backend,
        obs_mode=obs_mode,
        simrobot_profile=simrobot_profile,
        simrobot_launch=simrobot_launch,
        embedded_qt_platform=embedded_qt_platform,
        embedded_3d_preflight=embedded_3d_preflight,
    )
    env_cfg = make_env_config(stage_id, backend_cfg)
    raw_env, wrapped_env, _ = build_env(env_cfg)
    total_reward = 0.0
    completed_steps = 0
    final_step_info: Mapping[str, object] = {}
    try:
        observations, infos = wrapped_env.reset()
        agent, _ = build_mappo(wrapped_env, device, TRAIN, env_cfg.active_skill_names)
        primary_agent = wrapped_env.possible_agents[0]
        all_agents = wrapped_env.possible_agents
        for step in range(steps):
            action = sample_smoke_action(agent, raw_env, wrapped_env, observations, step, steps, device)
            action_tensor = torch.as_tensor(action_to_numpy(action), device="cpu").view(1, -1)
            observations, rewards, terminated, truncated, infos = wrapped_env.step(
                {a: action_tensor for a in all_agents}
            )
            if primary_agent not in rewards or primary_agent not in infos:
                if infos:
                    final_step_info = next(iter(infos.values()))
                break
            reward_tensor = rewards[primary_agent]
            reward_value = float(reward_tensor.item() if hasattr(reward_tensor, "item") else reward_tensor)
            total_reward += reward_value
            completed_steps = step + 1
            step_info = infos[primary_agent]
            final_step_info = step_info
            if telemetry_paths is not None:
                _append_csv_row(telemetry_paths.smoke_steps_csv, _build_smoke_step_row(step + 1, reward_value, step_info))
            print(
                f"[smoke] step={step + 1} stage={stage_id} "
                f"policy_skill={step_info.get('policy_skill_name')} "
                f"exec_skill={step_info.get('executed_skill_name')} "
                f"reward={reward_value:.3f} takeover={int(step_info.get('takeover_envelope', False))} "
                f"frame={step_info['bridge_frame']}",
                flush=True,
            )
            terminated_value = terminated[primary_agent]
            truncated_value = truncated[primary_agent]
            is_terminated = bool(terminated_value.item() if hasattr(terminated_value, "item") else terminated_value)
            is_truncated = bool(truncated_value.item() if hasattr(truncated_value, "item") else truncated_value)
            if is_terminated or is_truncated:
                print(f"[smoke] episode_metrics={step_info.get('episode_metrics', {})}", flush=True)
                break
    finally:
        wrapped_env.close()
        raw_env.close()
    smoke_metrics = {}
    if isinstance(final_step_info.get("episode_metrics"), Mapping):
        smoke_metrics.update(final_step_info["episode_metrics"])
    elif isinstance(final_step_info.get("running_metrics"), Mapping):
        smoke_metrics.update(final_step_info["running_metrics"])
    wire_audit = build_wire_audit(smoke_metrics, backend_cfg=backend_cfg, audit_cfg=TRAIN.wire_audit)
    if telemetry_paths is not None and final_step_info:
        _append_csv_row(
            telemetry_paths.smoke_summary_csv,
            _build_smoke_summary_row(
                steps=completed_steps,
                total_reward=total_reward,
                final_info=final_step_info,
                wire_audit=wire_audit,
            ),
        )
    print(
        f"[wire] mode={backend_cfg.obs_mode} status={wire_audit['wire_audit_status']} "
        f"failures={wire_audit['wire_audit_failure_count']} warnings={wire_audit['wire_audit_warning_count']}",
        flush=True,
    )
    if wire_audit["wire_audit_failures"]:
        print(f"[wire] fail_details={wire_audit['wire_audit_failures']}", flush=True)
    if wire_audit["wire_audit_warnings"]:
        print(f"[wire] warn_details={wire_audit['wire_audit_warnings']}", flush=True)
    if telemetry_paths is not None and save_plots:
        _plot_smoke_steps(telemetry_paths)
    if abort_on_wire_fail and wire_audit["wire_audit_status"] == "FAIL":
        raise RuntimeError(f"Wire audit failed in smoke run: {wire_audit['wire_audit_failures']}")
    print(f"[smoke] total_reward={total_reward:.3f}", flush=True)


def run_training(
    timesteps: int,
    backend: str,
    device: str,
    obs_mode: str,
    simrobot_profile: str,
    stage_id: int,
    max_stage_id: int,
    simrobot_launch: bool = False,
    embedded_qt_platform: str | None = None,
    embedded_3d_preflight: bool = True,
    telemetry_paths: TelemetryPaths | None = None,
    save_plots: bool = True,
    abort_on_wire_fail: bool = False,
) -> None:
    backend_cfg = replace(
        ENV.backend,
        mode=backend,
        obs_mode=obs_mode,
        simrobot_profile=simrobot_profile,
        simrobot_launch=simrobot_launch,
        embedded_qt_platform=embedded_qt_platform,
        embedded_3d_preflight=embedded_3d_preflight,
    )
    current_stage = stage_id
    env_cfg = make_env_config(current_stage, backend_cfg)
    raw_env, wrapped_env, eval_wrapper = build_env(env_cfg)
    agent, checkpoint_models = build_mappo(wrapped_env, device, TRAIN, env_cfg.active_skill_names)
    trainer_cfg = SequentialTrainerCfg(
        timesteps=timesteps,
        headless=True,
        disable_progressbar=True,
        close_environment_at_exit=True,
    )
    agent.init(trainer_cfg=trainer_cfg)
    agent.enable_training_mode(True)

    try:
        observations, infos = wrapped_env.reset()
        stage_history: Dict[int, Dict[str, float]] = {}
        agent_id = wrapped_env.possible_agents[0]
        _nan_reload_count = 0

        # Save initial weights as bootstrap checkpoint so the NaN guard can
        # fire from the very first rollout (before checkpoint_interval elapses).
        _initial_ckpt = save_checkpoint(
            checkpoint_models,
            train_cfg=TRAIN,
            timestep=0,
            stage_id=current_stage,
            eval_summary=None,
        )
        last_good_checkpoint_path: Path = _initial_ckpt
        print(f"[checkpoint] initial saved={_initial_ckpt}", flush=True)

        for timestep in range(timesteps):
            observations_device = _move_to_device(observations, device)
            states_device = _agent_state_from_env(raw_env, agent_id, device)
            agent.pre_interaction(timestep=timestep, timesteps=timesteps)
            with torch.no_grad():
                actions, _ = agent.act(
                    observations=observations_device,
                    states=states_device,
                    timestep=timestep,
                    timesteps=timesteps,
                )
                # Guard against NaN/Inf actions from a corrupted policy (prevents SimRobot hang)
                safe_actions = {k: torch.nan_to_num(v, nan=0.0, posinf=1.0, neginf=-1.0) for k, v in actions.items()}
                next_observations, rewards, terminated, truncated, infos = wrapped_env.step(_move_to_device(safe_actions, "cpu"))
                next_observations_device = _move_to_device(next_observations, device)
                next_states_device = _agent_state_from_env(raw_env, agent_id, device)
                agent.record_transition(
                    observations=observations_device,
                    states=states_device,
                    actions=safe_actions,
                    rewards=_move_to_device(rewards, device),
                    next_observations=next_observations_device,
                    next_states=next_states_device,
                    terminated=_move_to_device(terminated, device),
                    truncated=_move_to_device(truncated, device),
                    infos=infos,
                    timestep=timestep,
                    timesteps=timesteps,
                )
            agent.post_interaction(timestep=timestep, timesteps=timesteps)

            if (timestep + 1) % TRAIN.rollout_steps == 0:
                _ploss = _extract_agent_stat(
                    agent,
                    f"Loss / Policy loss ({agent_id})",
                    "Loss / Policy loss",
                    "policy_loss",
                )
                if np.isnan(_ploss) and last_good_checkpoint_path is not None and _nan_reload_count < 10:
                    _nan_reload_count += 1
                    _nan_diag = _diagnose_nan_source(
                        agent,
                        f"Loss / Policy loss ({agent_id})",
                        "Loss / Policy loss",
                        "policy_loss",
                    )
                    print(
                        f"[nan-guard] step={timestep + 1} reload #{_nan_reload_count}/10"
                        f" diag={_nan_diag} ← {last_good_checkpoint_path.name}",
                        flush=True,
                    )
                    _ckpt = torch.load(last_good_checkpoint_path, map_location=device, weights_only=True)
                    checkpoint_models["policy"].load_state_dict(_ckpt["policy_state_dict"])
                    checkpoint_models["value"].load_state_dict(_ckpt["value_state_dict"])
                    # Reset log_std to initial value to escape degenerate σ regime
                    with torch.no_grad():
                        if hasattr(checkpoint_models["policy"], "log_std_parameter"):
                            checkpoint_models["policy"].log_std_parameter.fill_(-0.5)

            if not wrapped_env.agents:
                observations, infos = wrapped_env.reset()
            else:
                observations = next_observations

            should_eval = (timestep + 1) % TRAIN.eval_interval == 0 or (timestep + 1) == timesteps
            if not should_eval:
                continue

            eval_summary, episode_results = evaluate_agent(
                agent,
                eval_wrapper,
                device=device,
                episodes=TRAIN.eval_episodes,
                stage_id=current_stage,
                timestep=timestep + 1,
                timesteps=timesteps,
            )
            wire_audit = build_wire_audit(eval_summary, backend_cfg=backend_cfg, audit_cfg=TRAIN.wire_audit)
            print_eval_summary(current_stage, timestep + 1, eval_summary, agent, wire_audit=wire_audit)
            if telemetry_paths is not None:
                eval_row = _build_eval_summary_row(
                    timestep=timestep + 1,
                    stage_id=current_stage,
                    summary=eval_summary,
                    agent=agent,
                    wire_audit=wire_audit,
                )
                _append_csv_row(telemetry_paths.eval_summary_csv, eval_row)
                _append_eval_episode_rows(
                    telemetry_paths,
                    timestep=timestep + 1,
                    stage_id=current_stage,
                    episode_results=episode_results,
                )
                if save_plots:
                    _plot_eval_metrics(telemetry_paths)

            if (timestep + 1) % TRAIN.checkpoint_interval == 0 or (timestep + 1) == timesteps:
                checkpoint_path = save_checkpoint(
                    checkpoint_models,
                    train_cfg=TRAIN,
                    timestep=timestep + 1,
                    stage_id=current_stage,
                    eval_summary={**dict(eval_summary), **dict(wire_audit)},
                )
                print(f"[checkpoint] saved={checkpoint_path}", flush=True)
                _ploss_ckpt = _extract_agent_stat(
                    agent,
                    f"Loss / Policy loss ({agent_id})",
                    "Loss / Policy loss",
                    "policy_loss",
                )
                if not np.isnan(_ploss_ckpt):
                    last_good_checkpoint_path = checkpoint_path
                    _nan_reload_count = 0

            if abort_on_wire_fail and wire_audit["wire_audit_status"] == "FAIL":
                raise RuntimeError(f"Wire audit failed at timestep {timestep + 1}: {wire_audit['wire_audit_failures']}")

            if (timestep + 1) < timesteps:
                observations, infos = wrapped_env.reset()

            promotion_metrics = {**dict(eval_summary), **dict(wire_audit)}
            _policy_healthy = not np.isnan(
                _extract_agent_stat(agent, f"Loss / Policy loss ({agent_id})", "Loss / Policy loss", "policy_loss")
            )
            if TRAIN.auto_promote and _policy_healthy and current_stage < max_stage_id and promote_stage(current_stage, promotion_metrics, stage_history):
                stage_history[current_stage] = dict(promotion_metrics)
                current_stage += 1
                print(f"[curriculum] promoted_to_stage={current_stage} at_step={timestep + 1}", flush=True)
                wrapped_env.close()
                raw_env.close()
                env_cfg = make_env_config(current_stage, backend_cfg)
                raw_env, wrapped_env, eval_wrapper = build_env(env_cfg)
                if hasattr(checkpoint_models["policy"], "set_active_skill_names"):
                    checkpoint_models["policy"].set_active_skill_names(env_cfg.active_skill_names)
                observations, infos = wrapped_env.reset()
    finally:
        wrapped_env.close()
        raw_env.close()
    print(f"[train] completed timesteps={timesteps} final_stage={current_stage}", flush=True)


def main(run_cfg: ScriptRunConfig = IDE_RUN) -> None:
    effective_train = replace(
        TRAIN,
        trainer_timesteps=run_cfg.train_steps,
        rollout_steps=run_cfg.rollout_steps,
        learning_epochs=run_cfg.learning_epochs,
        mini_batches=run_cfg.mini_batches,
        eval_interval=run_cfg.eval_interval,
        checkpoint_interval=run_cfg.checkpoint_interval,
        eval_episodes=run_cfg.eval_episodes,
    )
    effective_env = replace(
        ENV,
        max_episode_steps=run_cfg.max_episode_steps,
        parallel_player_numbers=run_cfg.parallel_player_numbers,
    )
    backend_cfg = replace(
        effective_env.backend,
        mode=run_cfg.backend,
        obs_mode=run_cfg.obs_mode,
        simrobot_profile=run_cfg.simrobot_profile,
        simrobot_launch=run_cfg.simrobot_launch,
        embedded_qt_platform=run_cfg.embedded_qt_platform,
        embedded_3d_preflight=run_cfg.embedded_3d_preflight,
    )
    effective_env = replace(effective_env, backend=backend_cfg)
    telemetry_paths = _prepare_telemetry_paths(run_cfg) if run_cfg.save_telemetry else None
    if telemetry_paths is not None:
        _write_run_config(telemetry_paths, run_cfg, backend_cfg, effective_train)
        print(f"[telemetry] session_dir={telemetry_paths.session_dir}", flush=True)
    print(
        f"[run] mode={run_cfg.mode} learner_device={run_cfg.learner_device} "
        f"env_backend={run_cfg.backend} obs_mode={run_cfg.obs_mode} "
        f"simrobot_profile={run_cfg.simrobot_profile} stage={run_cfg.stage_id} "
        f"max_stage={run_cfg.max_stage_id} "
        f"embedded_qt_platform={run_cfg.embedded_qt_platform or 'auto'} "
        f"embedded_3d_preflight={int(bool(run_cfg.embedded_3d_preflight))} "
        f"train_steps={effective_train.trainer_timesteps} rollout={effective_train.rollout_steps} "
        f"eval_interval={effective_train.eval_interval} checkpoint_interval={effective_train.checkpoint_interval}",
        flush=True,
    )

    original_train = TRAIN
    original_env = ENV
    globals()["TRAIN"] = effective_train
    globals()["ENV"] = effective_env
    try:
        runtime = _visible_runtime_for_run(run_cfg, effective_env)
        if runtime is not None:
            runtime.start()
        if run_cfg.mode == "smoke":
            run_smoke(
                run_cfg.smoke_steps,
                backend=run_cfg.backend,
                device=run_cfg.learner_device,
                obs_mode=run_cfg.obs_mode,
                simrobot_profile=run_cfg.simrobot_profile,
                stage_id=run_cfg.stage_id,
                simrobot_launch=run_cfg.simrobot_launch,
                embedded_qt_platform=run_cfg.embedded_qt_platform,
                embedded_3d_preflight=run_cfg.embedded_3d_preflight,
                telemetry_paths=telemetry_paths,
                save_plots=run_cfg.save_plots,
                abort_on_wire_fail=run_cfg.abort_on_wire_fail,
            )
        else:
            run_training(
                effective_train.trainer_timesteps,
                backend=run_cfg.backend,
                device=run_cfg.learner_device,
                obs_mode=run_cfg.obs_mode,
                simrobot_profile=run_cfg.simrobot_profile,
                stage_id=run_cfg.stage_id,
                max_stage_id=run_cfg.max_stage_id,
                simrobot_launch=run_cfg.simrobot_launch,
                embedded_qt_platform=run_cfg.embedded_qt_platform,
                embedded_3d_preflight=run_cfg.embedded_3d_preflight,
                telemetry_paths=telemetry_paths,
                save_plots=run_cfg.save_plots,
                abort_on_wire_fail=run_cfg.abort_on_wire_fail,
            )
    finally:
        if "runtime" in locals() and runtime is not None:
            runtime.close()
        globals()["TRAIN"] = original_train
        globals()["ENV"] = original_env


if __name__ == "__main__":
    main(_run_cfg_from_env())
