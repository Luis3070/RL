"""
train.py — Entrenamiento MAPPO para NAO Soccer Fase 1.

Estructura:
  1. Definición de redes (Policy y Value) — GaussianMixin y DeterministicMixin
  2. Setup de skrl MAPPO con parameter sharing
  3. Loop de entrenamiento con logging CSV
  4. Plots de convergencia y reward al final

Requirements:
    pip install skrl pettingzoo gymnasium torch numpy matplotlib pandas

Uso:
    python train.py
    python train.py --stage 3        # forzar stage del curriculum
    python train.py --eval           # solo evaluar checkpoint
    python train.py --no-fov         # desactivar FOV (baseline)
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")   # sin display — guarda PNGs
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# skrl imports
from skrl.multi_agents.torch.mappo import MAPPO, MAPPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.trainers.torch import SequentialTrainer

from bridge.bridge_protocol import expected_transport_mode, is_stable_transport_mode
from config import (
    FIELD, OBS, ACTION, GLOBAL_STATE,
    NETWORK, PPO_CFG, TRAINING, LOG, CURRICULUM,
    QUICK_DEBUG, build_quick_debug_curriculum,
    build_quick_debug_ppo_cfg, build_quick_debug_training_cfg,
    build_quick_debug_logging_cfg,
    PRACTICAL_TRAINING, build_practical_training_cfg,
    build_practical_logging_cfg,
)
from takeover_trace import TakeoverTraceLogger
from wrapper import NaoSoccerEnvWrapper

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# ESCALA DE REWARDS
# Factor fijo (PPO_CFG.reward_scale) en lugar de RunningMeanStd.
# RunningMeanStd normaliza a media=0, destruyendo la señal RELATIVA del shaping
# (el agente no puede distinguir "cerca del balón" de "lejos del balón" si ambos
# quedan centrados en 0). Un factor fijo solo escala la magnitud absoluta sin
# alterar las diferencias entre pasos — la policy aprende la misma señal, solo
# en un rango más amigable para el critic.
# Calibrado en config.py: reward_scale=0.05 → max shaping ~4.5, sparse ~0.5
# ─────────────────────────────────────────────────────────────────────────────

def scale_reward(r: float) -> float:
    """Aplica el factor de escala fijo definido en PPO_CFG.reward_scale."""
    return float(r) * PPO_CFG.reward_scale



# ─────────────────────────────────────────────────────────────────────────────
# UTILIDAD DE DIAGNÓSTICO NaN
# Actívala poniendo NAN_DEBUG=True. En producción déjala en False para velocidad.
# ─────────────────────────────────────────────────────────────────────────────

NAN_DEBUG: bool = False  # NaN diagnosticado y corregido. Reactivar solo si reaparece.

_MAPPO_ACT_COMPAT_MODE: Optional[int] = None
_MAPPO_RECORD_COMPAT_MODE: Optional[int] = None

def check_finite(name: str,
                 x: Any,
                 episode: Optional[int]   = None,
                 step: Optional[int]      = None,
                 agent_id: Optional[str]  = None) -> bool:
    """
    Verifica que todos los valores en x sean finitos (no NaN, no Inf).

    Úsalo en tres capas de la cadena:
      1. actions_t / actions_dict  → bug en policy o decode_action
      2. reward / next_obs / state → bug en física o reward shaping
      3. durante el update         → torch.autograd.set_detect_anomaly(True) lo cubre

    Retorna True si todo está bien, False si se detectó NaN/Inf.
    No lanza excepción — solo imprime para no interrumpir el run.
    """
    if not NAN_DEBUG:
        return True

    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    elif isinstance(x, (int, float)):
        arr = np.asarray([x], dtype=np.float64)
    else:
        try:
            arr = np.asarray(x, dtype=np.float64)
        except Exception:
            return True   # no convertible → ignorar

    ok = bool(np.all(np.isfinite(arr)))
    if not ok:
        print("\n" + "=" * 70)
        print(f"[NaN/Inf DETECTADO] {name}")
        print(f"  episode={episode}  step={step}  agent={agent_id}")
        print(f"  shape={arr.shape}")
        flat = arr.reshape(-1)
        print(f"  primeros 10 valores : {flat[:10]}")
        nan_idx = np.where(~np.isfinite(flat))[0]
        print(f"  índices no-finitos  : {nan_idx[:10]}  (total={len(nan_idx)})")
        print("=" * 70 + "\n")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# MODELOS DE RED NEURONAL
# ─────────────────────────────────────────────────────────────────────────────

def _build_mlp(input_size: int, hidden_sizes: tuple,
               output_size: int, activation: nn.Module) -> nn.Sequential:
    """Construye un MLP con capas ocultas y activaciones."""
    layers: List[nn.Module] = []
    prev = input_size
    for h in hidden_sizes:
        layers += [nn.Linear(prev, h), activation()]
        prev = h
    layers.append(nn.Linear(prev, output_size))
    return nn.Sequential(*layers)


class PolicyNetwork(GaussianMixin, Model):
    """
    Red del actor (policy). Salida: distribución Gaussiana sobre las acciones.

    Input:  observación local del agente (OBS.total floats)
    Output: media de la distribución (ACTION.size floats),
            log_std aprendido como parámetro.
    """

    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self,
            clip_actions=True,
            clip_log_std=True,
            min_log_std=NETWORK.log_std_min,
            max_log_std=NETWORK.log_std_max,
        )

        act_cls = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}[NETWORK.activation]

        self.net = _build_mlp(
            input_size=OBS.total,
            hidden_sizes=NETWORK.hidden_sizes,
            output_size=ACTION.size,
            activation=act_cls,
        )
        self.log_std_param = nn.Parameter(
            torch.ones(ACTION.size) * NETWORK.log_std_init
        )

    def compute(self, inputs: Dict, role: str = ""):
        # Compatibilidad con skrl: según versión usa "observations" o "obs"
        obs = inputs.get("observations", inputs.get("obs", inputs.get("states")))
        if obs is None:
            raise KeyError("PolicyNetwork.compute esperaba 'observations', 'obs' o 'states' en inputs")
        mean = self.net(obs)
        log_std = self.log_std_param.expand_as(mean)
        return mean, log_std, {}


class ValueNetwork(DeterministicMixin, Model):
    """
    Red del critic centralizado (value function).

    Input:  estado global del entorno (31 floats) — CTDE: visión completa
    Output: estimado escalar del valor del estado.
    """

    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        act_cls = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}[NETWORK.activation]

        self.net = _build_mlp(
            input_size=GLOBAL_STATE.total,
            hidden_sizes=NETWORK.hidden_sizes,
            output_size=1,
            activation=act_cls,
        )

    def compute(self, inputs: Dict, role: str = ""):
        # El critic centralizado usa estado global: skrl puede pasarlo como "states"
        state = inputs.get("states", inputs.get("obs"))
        if state is None:
            raise KeyError("ValueNetwork.compute esperaba 'states' o 'obs' en inputs")
        return self.net(state), {}


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

class MetricsLogger:
    """
    Registra métricas por episodio en CSV y genera plots PNG.

    Columnas CSV:
        episode, stage_id, stage_name, own_score, opp_score,
        mean_reward, steps, handoff_reached_rate, takeover_ready_rate, takeover_success_rate,
        goal_rate_w50, win_rate_w50,
        policy_loss, value_loss, entropy, elapsed_s
    """

    COLUMNS = [
        "episode", "stage_id", "stage_name",
        "own_score", "opp_score",
        "mean_reward", "steps",
        "handoff_reached_rate",
        "takeover_ready_rate", "takeover_success_rate",
        "takeover_success_rate_dribble_duel",
        "takeover_success_rate_shoot", "takeover_success_rate_pass",
        "takeover_coverage_rate_dribble_duel",
        "takeover_coverage_rate_shoot", "takeover_coverage_rate_pass",
        "playball_entry_rate", "ready_but_walk_rate", "ready_touch_but_walk_rate",
        "goal_rate_w50", "win_rate_w50",
        "policy_loss", "value_loss", "entropy",
        "elapsed_s",
        "episode_wall_s", "mean_env_step_ms", "mean_bridge_ms",
        "mean_bridge_team_ms",
        "process_memory_mb", "stability_score",
        "bridge_reconnect_count", "contract_validation_rate",
        "non_pipe_rate", "not_playing_rate",
        "ep_takeover_timeout_rate", "ep_takeover_skill_cleared_rate",
        "ep_takeover_active_change_rate", "ep_takeover_other_end_rate",
    ]

    def __init__(self, log_dir: str):
        self._dir = Path(log_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = self._dir / LOG.csv_file

        self._episode = 0
        self._start_time = time.time()
        self._rows: List[Dict] = []

        # Ventanas deslizantes para goal_rate y win_rate
        self._goal_window:  List[int] = []
        self._win_window:   List[int] = []

        # Crear CSV con encabezado
        with open(self._csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
            writer.writeheader()

    def log_episode(self,
                     own_score: int,
                     opp_score: int,
                     mean_reward: float,
                     steps: int,
                     stage_id: int,
                     stage_name: str,
                     handoff_reached_rate: float = 0.0,
                     takeover_ready_rate: float = 0.0,
                     takeover_success_rate: float = 0.0,
                     takeover_success_rate_dribble_duel: float = 0.0,
                     takeover_success_rate_shoot: float = 0.0,
                     takeover_success_rate_pass: float = 0.0,
                     takeover_coverage_rate_dribble_duel: float = 0.0,
                     takeover_coverage_rate_shoot: float = 0.0,
                     takeover_coverage_rate_pass: float = 0.0,
                     playball_entry_rate: float = 0.0,
                     ready_but_walk_rate: float = 0.0,
                     ready_touch_but_walk_rate: float = 0.0,
                     policy_loss: float  = 0.0,
                     value_loss: float   = 0.0,
                     entropy: float      = 0.0,
                     episode_wall_s: float = 0.0,
                     mean_env_step_ms: float = 0.0,
                     mean_bridge_ms: float = 0.0,
                     mean_bridge_team_ms: float = 0.0,
                     process_memory_mb: float = 0.0,
                     stability_score: float = 0.0,
                     bridge_reconnect_count: int = 0,
                     contract_validation_rate: float = 0.0,
                     non_pipe_rate: float = 0.0,
                     not_playing_rate: float = 0.0,
                     ep_takeover_timeout_rate: float = 0.0,
                     ep_takeover_skill_cleared_rate: float = 0.0,
                     ep_takeover_active_change_rate: float = 0.0,
                     ep_takeover_other_end_rate: float = 0.0) -> None:
        """Registra métricas BHuman-aware."""
        self._episode += 1

        # Actualizar ventanas (tamaño 100)
        self._goal_window.append(int(own_score > 0))
        self._win_window.append(int(own_score > opp_score))
        if len(self._goal_window) > 100:
            self._goal_window.pop(0)
        if len(self._win_window) > 100:
            self._win_window.pop(0)

        row = {
            "episode":              self._episode,
            "stage_id":             stage_id,
            "stage_name":           stage_name,
            "own_score":            own_score,
            "opp_score":            opp_score,
            "mean_reward":          round(mean_reward, 4),
            "steps":                steps,
            "handoff_reached_rate": round(handoff_reached_rate, 4),
            "takeover_ready_rate":  round(takeover_ready_rate, 4),
            "takeover_success_rate": round(takeover_success_rate, 4),
            "takeover_success_rate_dribble_duel": round(takeover_success_rate_dribble_duel, 4),
            "takeover_success_rate_shoot": round(takeover_success_rate_shoot, 4),
            "takeover_success_rate_pass": round(takeover_success_rate_pass, 4),
            "takeover_coverage_rate_dribble_duel": round(takeover_coverage_rate_dribble_duel, 4),
            "takeover_coverage_rate_shoot": round(takeover_coverage_rate_shoot, 4),
            "takeover_coverage_rate_pass": round(takeover_coverage_rate_pass, 4),
            "playball_entry_rate":  round(playball_entry_rate, 4),
            "ready_but_walk_rate":  round(ready_but_walk_rate, 4),
            "ready_touch_but_walk_rate": round(ready_touch_but_walk_rate, 4),
            "goal_rate_w50":        round(float(np.mean(self._goal_window)),  3),
            "win_rate_w50":         round(float(np.mean(self._win_window)),   3),
            "policy_loss":          round(policy_loss, 5),
            "value_loss":           round(value_loss,  5),
            "entropy":              round(entropy,     5),
            "elapsed_s":            round(time.time() - self._start_time, 1),
            "episode_wall_s":       round(episode_wall_s, 3),
            "mean_env_step_ms":     round(mean_env_step_ms, 3),
            "mean_bridge_ms":       round(mean_bridge_ms, 3),
            "mean_bridge_team_ms":  round(mean_bridge_team_ms, 3),
            "process_memory_mb":    round(process_memory_mb, 2),
            "stability_score":      round(stability_score, 4),
            "bridge_reconnect_count": int(bridge_reconnect_count),
            "contract_validation_rate": round(contract_validation_rate, 4),
            "non_pipe_rate":        round(non_pipe_rate, 4),
            "not_playing_rate":     round(not_playing_rate, 4),
            "ep_takeover_timeout_rate": round(ep_takeover_timeout_rate, 4),
            "ep_takeover_skill_cleared_rate": round(ep_takeover_skill_cleared_rate, 4),
            "ep_takeover_active_change_rate": round(ep_takeover_active_change_rate, 4),
            "ep_takeover_other_end_rate": round(ep_takeover_other_end_rate, 4),
        }
        self._rows.append(row)

        # Append al CSV sin reescribir todo
        with open(self._csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
            writer.writerow(row)

    def print_summary(self, episode: int) -> None:
        """Imprime resumen en consola."""
        if not self._rows:
            return
        r = self._rows[-1]
        print(f"[Ep {episode:6d}] Stage={r['stage_id']}({r['stage_name'][:10]}) | "
              f"Score={r['own_score']}-{r['opp_score']} | "
              f"Reward={r['mean_reward']:+.3f} | "
              f"HR={r['handoff_reached_rate']:.2f} | "
              f"TkReady={r['takeover_ready_rate']:.2f} | "
              f"TkSucc={r['takeover_success_rate']:.4f} | "
              f"DD={r['takeover_success_rate_dribble_duel']:.3f} | "
              f"Sh={r['takeover_success_rate_shoot']:.3f} | "
              f"Pa={r['takeover_success_rate_pass']:.3f} | "
              f"PB={r['playball_entry_rate']:.2f} | "
              f"RBW={r['ready_but_walk_rate']:.2f} | "
              f"TO={r['ep_takeover_timeout_rate']:.2f} | "
              f"CLR={r['ep_takeover_skill_cleared_rate']:.2f} | "
              f"Stable={r['stability_score']:.2f} | "
              f"Br={r['mean_bridge_ms']:.1f}ms | "
              f"TeamBr={r.get('mean_bridge_team_ms', r['mean_bridge_ms']):.1f}ms | "
              f"Win={r['win_rate_w50']:.2f} | "
              f"Elapsed={r['elapsed_s']:.0f}s")

    def plot_convergence(self) -> None:
        """Genera y guarda plots de convergencia como PNG."""
        if len(self._rows) < 2:
            return

        episodes     = [r["episode"]       for r in self._rows]
        rewards      = [r["mean_reward"]   for r in self._rows]
        goal_rates   = [r["goal_rate_w50"] for r in self._rows]
        win_rates    = [r["win_rate_w50"]  for r in self._rows]
        policy_loss  = [r["policy_loss"]   for r in self._rows]
        value_loss   = [r["value_loss"]    for r in self._rows]
        entropy      = [r["entropy"]       for r in self._rows]
        stages       = [r["stage_id"]      for r in self._rows]

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle("MAPPO NAO Soccer — Convergencia BHuman", fontsize=14)

        _plot_metric(axes[0, 0], episodes, rewards,      "Mean Reward",    "Episodio", "Reward", stages)
        _plot_metric(axes[0, 1], episodes, goal_rates,   "Goal Rate (w50)","Episodio", "Tasa",   stages)
        _plot_metric(axes[0, 2], episodes, win_rates,    "Win Rate (w50)", "Episodio", "Tasa",   stages)
        _plot_metric(axes[1, 0], episodes, policy_loss,  "Policy Loss",    "Episodio", "Loss",   stages)
        _plot_metric(axes[1, 1], episodes, value_loss,   "Value Loss",     "Episodio", "Loss",   stages)
        _plot_metric(axes[1, 2], episodes, entropy,      "Entropy",        "Episodio", "Nats",   stages)

        plt.tight_layout()
        plot_path = self._dir / "convergence.png"
        plt.savefig(plot_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

    def plot_scores(self) -> None:
        """Plot de scores acumulados (goles a favor y en contra)."""
        if len(self._rows) < 2:
            return

        episodes = [r["episode"]   for r in self._rows]
        own      = [r["own_score"] for r in self._rows]
        opp      = [r["opp_score"] for r in self._rows]

        # Acumulado
        own_cum = np.cumsum(own)
        opp_cum = np.cumsum(opp)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("MAPPO NAO Soccer — Goles", fontsize=13)

        ax1.plot(episodes, own, alpha=0.4, label="Goles propios")
        ax1.plot(episodes, opp, alpha=0.4, label="Goles concedidos")
        ax1.set_title("Goles por episodio")
        ax1.set_xlabel("Episodio")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(episodes, own_cum, label="Acum. propio")
        ax2.plot(episodes, opp_cum, label="Acum. concedido")
        ax2.set_title("Goles acumulados")
        ax2.set_xlabel("Episodio")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self._dir / "scores.png", dpi=120, bbox_inches="tight")
        plt.close(fig)


    def plot_rates(self) -> None:
        """
        PNG dedicado a readiness, takeover success agregado, success por macro y cobertura.

        Métricas BHuman-aware para monitorear el progreso del curriculum.

        takeover_ready (azul):   handoff envelope alcanzado
        takeover_success (naranja): BHuman ejecutó exitosamente
        dribble_duel (verde):    éxito en macros hoy cubiertos por curriculum
        shoot/pass (rojo/morado): éxito en macros avanzados
        coverage (líneas punteadas): porcentaje de episodios donde hubo al menos
        una oportunidad observada para esa familia de macros
        goal_rate / win_rate:    resultado deportivo agregado

        Las líneas verticales grises marcan los cambios de stage.
        """
        if len(self._rows) < 2:
            return

        episodes       = [r["episode"]               for r in self._rows]
        handoff_rates  = [r["handoff_reached_rate"]   for r in self._rows]
        tk_ready_rates = [r["takeover_ready_rate"]    for r in self._rows]
        tk_succ_rates  = [r["takeover_success_rate"]  for r in self._rows]
        tk_dd_rates    = [r["takeover_success_rate_dribble_duel"] for r in self._rows]
        tk_sh_rates    = [r["takeover_success_rate_shoot"] for r in self._rows]
        tk_pa_rates    = [r["takeover_success_rate_pass"] for r in self._rows]
        cov_dd_rates   = [r["takeover_coverage_rate_dribble_duel"] for r in self._rows]
        cov_sh_rates   = [r["takeover_coverage_rate_shoot"] for r in self._rows]
        cov_pa_rates   = [r["takeover_coverage_rate_pass"] for r in self._rows]
        goal_rates     = [r["goal_rate_w50"]          for r in self._rows]
        win_rates      = [r["win_rate_w50"]           for r in self._rows]
        stages         = [r["stage_id"]               for r in self._rows]

        fig, axes = plt.subplots(6, 1, figsize=(14, 18), sharex=True)
        fig.suptitle("MAPPO NAO Soccer — BHuman Metrics por Stage", fontsize=14)

        series = [
            (axes[0], handoff_rates, "Handoff Reached Rate", "Tasa", "steelblue", "Handoff reached"),
            (axes[1], tk_ready_rates, "TkReady Rate", "Tasa", "royalblue", "BHuman accepted takeover"),
            (axes[2], tk_succ_rates,  "Takeover Success Rate","Tasa", "darkorange", "BHuman success"),
            (axes[3], tk_dd_rates,    "Takeover Success D+D", "Tasa", "seagreen",   "Dribble/Duel success"),
            (axes[4], goal_rates,     "Goal / Win Rate (w100)", "Tasa", "teal", "Metió gol"),
        ]

        for ax, data, title, ylabel, color, label in series:
            # Señal cruda (transparente)
            ax.plot(episodes, data, linewidth=0.6, alpha=0.4, color=color)
            # Suavizado
            if len(data) >= 20:
                smooth = np.convolve(data, np.ones(20) / 20, mode="valid")
                ax.plot(episodes[19:], smooth, linewidth=2.0,
                        color=color, alpha=0.95, label=label)
            ax.set_title(title, fontsize=11)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_ylim(-0.05, 1.10)
            ax.legend(loc="upper left", fontsize=8)
            ax.grid(True, alpha=0.3)

            # Líneas verticales de cambio de stage
            prev_s = stages[0] if stages else 0
            for i, s in enumerate(stages):
                if s != prev_s:
                    ax.axvline(x=episodes[i], color="gray",
                               linestyle="--", alpha=0.6, linewidth=1.0)
                    ax.text(episodes[i], 1.05, f"S{s}",
                            fontsize=7, color="gray", ha="center")
                    prev_s = s

        axes[3].plot(episodes, cov_dd_rates, linestyle="--", linewidth=1.5,
                     color="darkgreen", alpha=0.9, label="D+D coverage")
        axes[3].legend(loc="upper left", fontsize=8)

        axes[4].cla()
        axes[4].plot(episodes, tk_sh_rates, linewidth=0.6, alpha=0.4, color="firebrick")
        if len(tk_sh_rates) >= 20:
            smooth_sh = np.convolve(tk_sh_rates, np.ones(20) / 20, mode="valid")
            axes[4].plot(episodes[19:], smooth_sh, linewidth=2.0,
                         color="firebrick", alpha=0.95, label="Shoot success")
        axes[4].plot(episodes, tk_pa_rates, linewidth=0.8, alpha=0.55,
                     color="purple", label="Pass success")
        if len(tk_pa_rates) >= 20:
            smooth_pa = np.convolve(tk_pa_rates, np.ones(20) / 20, mode="valid")
            axes[4].plot(episodes[19:], smooth_pa, linewidth=2.0,
                         color="purple", alpha=0.95)
        axes[4].plot(episodes, cov_sh_rates, linestyle="--", linewidth=1.5,
                     color="firebrick", alpha=0.8, label="Shoot coverage")
        axes[4].plot(episodes, cov_pa_rates, linestyle="--", linewidth=1.5,
                     color="mediumpurple", alpha=0.8, label="Pass coverage")
        axes[4].set_title("Takeover Success Shoot/Pass", fontsize=11)
        axes[4].set_ylabel("Tasa", fontsize=9)
        axes[4].set_ylim(-0.05, 1.10)
        axes[4].legend(loc="upper left", fontsize=8)
        axes[4].grid(True, alpha=0.3)

        axes[5].plot(episodes, goal_rates, linewidth=0.6, alpha=0.4, color="teal")
        if len(goal_rates) >= 20:
            smooth_goal = np.convolve(goal_rates, np.ones(20) / 20, mode="valid")
            axes[5].plot(episodes[19:], smooth_goal, linewidth=2.0,
                         color="teal", alpha=0.95, label="Metio gol")
        axes[5].plot(episodes, win_rates, linewidth=0.8, alpha=0.5,
                     color="goldenrod", label="Ganó el episodio")
        if len(win_rates) >= 20:
            smooth_win = np.convolve(win_rates, np.ones(20) / 20, mode="valid")
            axes[5].plot(episodes[19:], smooth_win, linewidth=2.0,
                         color="goldenrod", alpha=0.95)
        axes[5].set_title("Goal / Win Rate (w100)", fontsize=11)
        axes[5].set_ylabel("Tasa", fontsize=9)
        axes[5].set_ylim(-0.05, 1.10)
        axes[5].legend(loc="upper left", fontsize=8)
        axes[5].grid(True, alpha=0.3)
        prev_s = stages[0] if stages else 0
        for i, s in enumerate(stages):
            if s != prev_s:
                axes[5].axvline(x=episodes[i], color="gray",
                                linestyle="--", alpha=0.6, linewidth=1.0)
                axes[5].text(episodes[i], 1.05, f"S{s}",
                             fontsize=7, color="gray", ha="center")
                prev_s = s

        axes[-1].set_xlabel("Episodio", fontsize=10)
        plt.tight_layout()
        plt.savefig(self._dir / "rates.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    def plot_watchdog(self) -> None:
        """PNG único para tiempos, memoria y estabilidad del entrenamiento."""
        if len(self._rows) < 2:
            return

        episodes = [r["episode"] for r in self._rows]
        stages = [r["stage_id"] for r in self._rows]
        episode_wall_s = [r["episode_wall_s"] for r in self._rows]
        env_step_ms = [r["mean_env_step_ms"] for r in self._rows]
        bridge_ms = [r["mean_bridge_ms"] for r in self._rows]
        bridge_team_ms = [
            r.get("mean_bridge_team_ms", r["mean_bridge_ms"])
            for r in self._rows
        ]
        memory_mb = [r["process_memory_mb"] for r in self._rows]
        stability = [r["stability_score"] for r in self._rows]
        reconnects = [r["bridge_reconnect_count"] for r in self._rows]

        fig, axes = plt.subplots(3, 2, figsize=(15, 11), sharex=True)
        fig.suptitle("MAPPO NAO Soccer - Watchdog de Tiempos, Memoria y Estabilidad", fontsize=14)

        _plot_metric(axes[0, 0], episodes, episode_wall_s, "Tiempo por episodio", "Episodio", "s", stages)
        _plot_metric(axes[0, 1], episodes, env_step_ms, "Latencia media env.step", "Episodio", "ms", stages)
        axes[1, 0].plot(episodes, bridge_ms, linewidth=0.9, alpha=0.8, label="bridge activo")
        axes[1, 0].plot(episodes, bridge_team_ms, linewidth=0.9, alpha=0.8, label="sidecar equipo")
        axes[1, 0].set_title("Latencia bridge / sidecar", fontsize=11)
        axes[1, 0].set_xlabel("Episodio", fontsize=9)
        axes[1, 0].set_ylabel("ms", fontsize=9)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend(loc="upper left", fontsize=8)
        prev_s = stages[0] if stages else 0
        for i, s in enumerate(stages):
            if s != prev_s:
                axes[1, 0].axvline(x=episodes[i], color="gray",
                                   linestyle="--", alpha=0.6, linewidth=1.0)
                prev_s = s
        _plot_metric(axes[1, 1], episodes, memory_mb, "Memoria RSS del proceso", "Episodio", "MB", stages)
        _plot_metric(axes[2, 0], episodes, stability, "Stability Score", "Episodio", "0-1", stages)
        axes[2, 0].set_ylim(-0.05, 1.05)
        _plot_metric(axes[2, 1], episodes, reconnects, "Reconexiones del bridge", "Episodio", "count", stages)

        plt.tight_layout()
        plt.savefig(self._dir / "watchdog.png", dpi=120, bbox_inches="tight")
        plt.close(fig)


def _get_process_memory_mb() -> float:
    """RSS del proceso actual, con fallback stdlib en Windows."""
    try:
        import psutil  # type: ignore
        return float(psutil.Process(os.getpid()).memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        pass

    if sys.platform.startswith("win"):
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        process = kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(
            process,
            ctypes.byref(counters),
            counters.cb,
        )
        if ok:
            return float(counters.WorkingSetSize) / (1024.0 * 1024.0)

    return 0.0


def _plot_metric(ax, x, y, title, xlabel, ylabel, stages):
    """Dibuja una métrica con líneas verticales de cambio de stage."""
    ax.plot(x, y, linewidth=0.8, alpha=0.7)

    # Smoothed
    if len(y) >= 10:
        smooth = np.convolve(y, np.ones(10) / 10, mode="valid")
        ax.plot(x[9:], smooth, linewidth=1.5, color="red", alpha=0.9)

    # Líneas de cambio de stage
    prev_s = stages[0] if stages else 0
    for i, s in enumerate(stages):
        if s != prev_s:
            ax.axvline(x=x[i], color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
            prev_s = s

    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, alpha=0.3)


# ─────────────────────────────────────────────────────────────────────────────
# SETUP DE SKRL MAPPO
# ─────────────────────────────────────────────────────────────────────────────

def build_mappo_agent(env_wrapper: NaoSoccerEnvWrapper,
                      device: str,
                      ppo_cfg=PPO_CFG) -> MAPPO:
    """
    Construye el agente MAPPO de skrl con parameter sharing REAL.

    Issue 5: se crea UNA sola instancia de PolicyNetwork y UNA de ValueNetwork.
    Todos los aid referencian las MISMAS instancias. Si se crearan instancias
    separadas por aid, cada agente tendría sus propios pesos — eso NO es
    parameter sharing; es multi-policy independiente disfrazada de MAPPO.
    """
    possible_agents  = env_wrapper.possible_agents
    obs_space        = env_wrapper.observation_spaces[possible_agents[0]]
    act_space        = env_wrapper.action_spaces[possible_agents[0]]
    shared_obs_space = env_wrapper.shared_observation_spaces[possible_agents[0]]

    # UNA sola instancia compartida — todos los aid apuntan a los mismos objetos
    shared_policy = PolicyNetwork(obs_space,       act_space, device).to(device)
    shared_value  = ValueNetwork(shared_obs_space, act_space, device).to(device)

    models   = {}
    memories = {}
    for aid in possible_agents:
        models[aid] = {
            "policy": shared_policy,   # misma instancia, no copia
            "value":  shared_value,    # misma instancia, no copia
        }
        memories[aid] = RandomMemory(
            memory_size=ppo_cfg.rollout_steps,
            num_envs=2,
            device=device,
        )

    cfg = MAPPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"]           = ppo_cfg.rollout_steps
    cfg["learning_epochs"]    = ppo_cfg.update_epochs
    cfg["mini_batches"]       = ppo_cfg.mini_batches
    cfg["discount_factor"]    = ppo_cfg.gamma
    cfg["lambda"]             = ppo_cfg.lambda_gae
    cfg["learning_rate"]      = ppo_cfg.learning_rate
    cfg["ratio_clip"]         = ppo_cfg.clip_ratio
    cfg["value_loss_scale"]   = ppo_cfg.value_coeff
    cfg["entropy_loss_scale"] = ppo_cfg.entropy_coeff
    cfg["grad_norm_clip"]     = ppo_cfg.max_grad_norm
    cfg["rewards_shaper"]     = None
    # KL threshold: frena el update si la policy se aleja demasiado
    # en un solo epoch. Previene la entropía creciente observada en
    # los runs anteriores. skrl lo soporta como "kl_threshold".
    cfg["kl_threshold"]       = ppo_cfg.kl_threshold

    if "experiment" in cfg and isinstance(cfg["experiment"], dict):
        if isinstance(cfg["experiment"].get("write_interval"), str):
            cfg["experiment"]["write_interval"] = 0
        if isinstance(cfg["experiment"].get("checkpoint_interval"), str):
            cfg["experiment"]["checkpoint_interval"] = 0

    agent = MAPPO(
        possible_agents=possible_agents,
        models=models,
        memories=memories,
        cfg=cfg,
        observation_spaces={aid: obs_space       for aid in possible_agents},
        action_spaces    ={aid: act_space        for aid in possible_agents},
        device=device,
        shared_observation_spaces={aid: shared_obs_space for aid in possible_agents},
    )
    return agent



def reset_policy_weights(agent: MAPPO, device: str) -> None:
    """
    Reinicializa policy Y critic cuando la policy está permanentemente corrompida.

    FIX CRÍTICO: el docstring anterior decía "NO reinicializa el critic", pero
    eso era correcto solo cuando el NaN ocurría dentro del mismo stage. Cuando
    se carga un checkpoint de Stage 0 en Stage 1, el critic aprendió V(s)≈+40,
    pero los rewards de Stage 1 son -14 a -30. El TD error es ~60/paso → value
    loss ~3600/step → gradientes que explotan el optimizer incluso con grad_clip.
    El warmup no ayuda porque el problema no es la calidad de los datos sino
    la magnitud del error del critic cargado.

    Solución: resetear AMBAS redes cuando la policy produce NaN persistente.
    El critic parte de cero junto a la policy — el costo es reaprender V(s),
    que con rollout_steps=1024 ocurre rápido (2-3 episodios).
    """
    first_aid   = next(iter(agent.models))
    policy_net  = agent.models[first_aid]["policy"]
    value_net   = agent.models[first_aid]["value"]

    print("\n[RESET] Reinicializando pesos de PolicyNetwork + ValueNetwork...")
    with torch.no_grad():
        for net in [policy_net, value_net]:
            for module in net.modules():
                if isinstance(module, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
        if hasattr(policy_net, "log_std_param"):
            policy_net.log_std_param.data.fill_(NETWORK.log_std_init)

    # Resetear estados internos del optimizer (momentos Adam).
    # Sin esto, Adam mantiene momentos de la distribución anterior y aplica
    # pasos grandes en dirección equivocada en el primer update post-reset.
    # Defensivo: skrl puede exponer optimizers de diferentes formas según versión.
    try:
        if hasattr(agent, "optimizers"):
            opts = agent.optimizers
            if isinstance(opts, dict):
                for v in opts.values():
                    if isinstance(v, dict):
                        for opt in v.values():
                            if hasattr(opt, "state"):
                                opt.state.clear()
                    elif hasattr(v, "state"):
                        v.state.clear()
        elif hasattr(agent, "optimizer"):
            if hasattr(agent.optimizer, "state"):
                agent.optimizer.state.clear()
        print("[RESET] Optimizer states limpiados.")
    except Exception as e:
        print(f"[RESET] Optimizer state clear falló (no crítico): {e}")

    print(f"[RESET] Pesos + optimizers reinicializados. log_std = {NETWORK.log_std_init}")


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNÓSTICO DEL TRAINING LOOP (Issue 2)
# ─────────────────────────────────────────────────────────────────────────────

class DiagnosticsProbe:
    """
    Responde la pregunta: "¿los parámetros cambian después del update o no?"

    Imprime:
      - Keys de infos después de reset() y step()
      - Norma del global_state que entra al critic
      - Cada cuántos steps ocurre el update
      - Norma de pesos ANTES y DESPUÉS del primer update → delta > 0 confirma que
        los gradientes llegan y el optimizador actualiza.

    Solo activo en el primer episodio para no spam.
    """

    def __init__(self):
        self._reset_logged   = False
        self._step_logged    = False
        self._weight_snap    = None    # norma de pesos antes del update
        self._update_count   = 0
        self._first_update_done = False

    def log_reset(self, infos: Dict, obs: Dict) -> None:
        if self._reset_logged:
            return
        print("\n" + "="*60)
        print("[Diagnostics] Después de reset():")
        if infos:
            aid = next(iter(infos))
            print(f"  infos['{aid}'].keys() = {list(infos[aid].keys())}")
        if obs:
            aid = next(iter(obs))
            print(f"  obs['{aid}'].shape   = {obs[aid].shape}")
            print(f"  obs['{aid}'].norm()  = {float(np.linalg.norm(obs[aid])):.4f}")
        self._reset_logged = True

    def log_step(self, infos: Dict, rewards: Dict, agent_ids: List[str]) -> None:
        if self._step_logged:
            return
        print("\n[Diagnostics] Primer step():")
        if infos:
            aid = next(iter(infos))
            print(f"  infos['{aid}'].keys() = {list(infos[aid].keys())}")
            state = infos[aid].get("state", None)
            if state is not None:
                print(f"  global_state.norm()   = {float(np.linalg.norm(state)):.4f}")
                print(f"  global_state[:5]      = {state[:5].round(4)}")
        if rewards:
            print(f"  rewards               = {dict((k, round(v,4)) for k,v in rewards.items())}")
        print(f"  agentes activos       = {agent_ids}")
        self._step_logged = True

    def snapshot_weights(self, agent: MAPPO) -> None:
        """Captura la norma de los pesos de policy antes del primer update."""
        if self._weight_snap is not None:
            return
        first_aid = next(iter(agent.models))
        policy    = agent.models[first_aid]["policy"]
        total_norm = sum(
            float(p.data.norm()) for p in policy.parameters() if p.requires_grad
        )
        self._weight_snap = total_norm
        print(f"\n[Diagnostics] Norma pesos policy ANTES del update: {total_norm:.6f}")

    def log_update(self, agent: MAPPO, timestep: int) -> None:
        """Registra cada update y compara pesos tras el primero."""
        self._update_count += 1
        print(f"\n[Diagnostics] Update #{self._update_count} en step {timestep}")

        if not self._first_update_done and self._weight_snap is not None:
            first_aid = next(iter(agent.models))
            policy    = agent.models[first_aid]["policy"]
            total_norm_after = sum(
                float(p.data.norm()) for p in policy.parameters() if p.requires_grad
            )
            delta = abs(total_norm_after - self._weight_snap)
            print(f"  Norma pesos ANTES: {self._weight_snap:.6f}")
            print(f"  Norma pesos DESPUÉS: {total_norm_after:.6f}")
            print(f"  Delta: {delta:.6f}  {'✅ pesos actualizados' if delta > 1e-8 else '❌ pesos SIN CAMBIO — revisar plumbing'}")
            self._first_update_done = True


def _extract_mappo_actions(act_output: Any,
                           requested_agents: List[str]) -> Dict[str, torch.Tensor]:
    """Normaliza la salida de agent.act() a un dict[agent_id] -> acción tensor."""
    actions = act_output[0] if isinstance(act_output, tuple) else act_output

    if isinstance(actions, dict):
        return actions

    # Compatibilidad defensiva: algunos wrappers pueden devolver tensor directo
    if len(requested_agents) == 1:
        return {requested_agents[0]: actions}

    raise TypeError("Salida de MAPPO.act no reconocida para múltiples agentes")


def _copy_vector_to_tensor(dst: torch.Tensor, value: Any) -> None:
    """
    Copia un vector CPU/GPU a un tensor prealocado preservando forma y dtype.

    Este helper reduce allocaciones repetidas dentro del loop de entrenamiento.
    """
    if value is None:
        dst.zero_()
        return

    if isinstance(value, torch.Tensor):
        src = value.detach()
        if src.device != dst.device or src.dtype != dst.dtype:
            src = src.to(device=dst.device, dtype=dst.dtype)
        dst.copy_(src.reshape_as(dst), non_blocking=dst.is_cuda)
        return

    src_arr = np.ascontiguousarray(np.asarray(value, dtype=np.float32).reshape(-1))
    if src_arr.shape[0] != dst.numel():
        raise ValueError(
            f"Shape mismatch copiando vector: expected={dst.numel()} got={src_arr.shape[0]}"
        )
    dst.copy_(torch.from_numpy(src_arr), non_blocking=dst.is_cuda)


class TrainingTensorCache:
    """
    Reutiliza buffers por agente para act() y record_transition().

    skrl copia las muestras al buffer de memoria, asÃ­ que es seguro reciclar
    estos tensores entre pasos.
    """

    def __init__(self, agent_ids: List[str], device: str):
        self.agent_ids = list(agent_ids)
        self.device = torch.device(device)

        self.obs = {
            aid: torch.zeros(OBS.total, dtype=torch.float32, device=self.device)
            for aid in self.agent_ids
        }
        self.states = {
            aid: torch.zeros(GLOBAL_STATE.total, dtype=torch.float32, device=self.device)
            for aid in self.agent_ids
        }
        self.actions = {
            aid: torch.zeros(ACTION.size, dtype=torch.float32, device=self.device)
            for aid in self.agent_ids
        }
        self.rewards = {
            aid: torch.zeros(1, dtype=torch.float32, device=self.device)
            for aid in self.agent_ids
        }
        self.next_obs = {
            aid: torch.zeros(OBS.total, dtype=torch.float32, device=self.device)
            for aid in self.agent_ids
        }
        self.next_states = {
            aid: torch.zeros(GLOBAL_STATE.total, dtype=torch.float32, device=self.device)
            for aid in self.agent_ids
        }
        self.terminated = {
            aid: torch.zeros(1, dtype=torch.bool, device=self.device)
            for aid in self.agent_ids
        }
        self.truncated = {
            aid: torch.zeros(1, dtype=torch.bool, device=self.device)
            for aid in self.agent_ids
        }
        self.shared_states = torch.zeros(GLOBAL_STATE.total, dtype=torch.float32, device=self.device)
        self.shared_next_states = torch.zeros(GLOBAL_STATE.total, dtype=torch.float32, device=self.device)
        self.infos = {
            "shared_states": self.shared_states,
            "shared_next_states": self.shared_next_states,
        }

    def load_policy_inputs(self, obs: Dict[str, Any], infos: Dict[str, Any]) -> None:
        for aid in self.agent_ids:
            _copy_vector_to_tensor(self.obs[aid], obs.get(aid))
            _copy_vector_to_tensor(self.states[aid], infos.get(aid, {}).get("state"))

    def load_transition_inputs(
        self,
        curr_obs: Dict[str, Any],
        curr_infos: Dict[str, Any],
        actions_dict: Dict[str, np.ndarray],
        rewards: Dict[str, float],
        next_obs: Dict[str, Any],
        infos: Dict[str, Any],
        terminateds: Dict[str, Any],
        truncateds: Dict[str, Any],
        active_agents: List[str],
    ) -> None:
        active_set = set(active_agents)
        _copy_vector_to_tensor(
            self.shared_states,
            next(iter(curr_infos.values()), {}).get("state"),
        )
        _copy_vector_to_tensor(
            self.shared_next_states,
            next(iter(infos.values()), {}).get("state"),
        )

        for aid in self.agent_ids:
            _copy_vector_to_tensor(self.obs[aid], curr_obs.get(aid))
            _copy_vector_to_tensor(self.states[aid], curr_infos.get(aid, {}).get("state"))
            _copy_vector_to_tensor(self.actions[aid], actions_dict.get(aid))
            self.rewards[aid].fill_(scale_reward(rewards.get(aid, 0.0)))
            _copy_vector_to_tensor(self.next_obs[aid], next_obs.get(aid))
            _copy_vector_to_tensor(self.next_states[aid], infos.get(aid, {}).get("state"))
            self.terminated[aid].fill_(bool(terminateds.get(aid, aid not in active_set)))
            self.truncated[aid].fill_(bool(truncateds.get(aid, False)))


def _mappo_act_compat(agent: MAPPO,
                      observations: Dict[str, torch.Tensor],
                      states: Dict[str, torch.Tensor],
                      timestep: int,
                      timesteps: int) -> Dict[str, torch.Tensor]:
    """Llama MAPPO.act con compatibilidad entre firmas de skrl."""
    global _MAPPO_ACT_COMPAT_MODE

    def _invoke(mode: int) -> Any:
        if mode == 0:
            return agent.act(
                observations=observations,
                states=states,
                timestep=timestep,
                timesteps=timesteps,
            )
        if mode == 1:
            return agent.act(
                observations,
                states,
                timestep=timestep,
                timesteps=timesteps,
            )
        if mode == 2:
            return agent.act(
                states=observations,
                timestep=timestep,
                timesteps=timesteps,
            )
        if mode == 3:
            return agent.act(observations, timestep, timesteps)
        raise RuntimeError(f"Modo act compat desconocido: {mode}")

    if _MAPPO_ACT_COMPAT_MODE is not None:
        output = _invoke(_MAPPO_ACT_COMPAT_MODE)
        return _extract_mappo_actions(output, list(observations.keys()))

    last_err: Optional[Exception] = None
    for mode in range(4):
        try:
            output = _invoke(mode)
            _MAPPO_ACT_COMPAT_MODE = mode
            return _extract_mappo_actions(output, list(observations.keys()))
        except TypeError as err:
            last_err = err

    if last_err is not None:
        raise last_err
    raise RuntimeError("No se pudo ejecutar MAPPO.act")


def _mappo_record_transition_compat(agent: MAPPO,
                                    observations: Dict[str, torch.Tensor],
                                    states: Dict[str, torch.Tensor],
                                    actions: Dict[str, torch.Tensor],
                                    rewards: Dict[str, torch.Tensor],
                                    next_observations: Dict[str, torch.Tensor],
                                    next_states: Dict[str, torch.Tensor],
                                    terminated: Dict[str, torch.Tensor],
                                    truncated: Dict[str, torch.Tensor],
                                    infos: Dict[str, Any],
                                    timestep: int,
                                    timesteps: int) -> None:
    """Llama MAPPO.record_transition con compatibilidad entre firmas de skrl."""
    global _MAPPO_RECORD_COMPAT_MODE

    def _invoke(mode: int) -> None:
        if mode == 0:
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
            return
        if mode == 1:
            agent.record_transition(
                states=observations,
                actions=actions,
                rewards=rewards,
                next_states=next_observations,
                terminated=terminated,
                truncated=truncated,
                infos=infos,
                timestep=timestep,
                timesteps=timesteps,
            )
            return
        if mode == 2:
            agent.record_transition(
                observations,
                actions,
                rewards,
                next_observations,
                terminated,
                truncated,
                infos,
                timestep,
                timesteps,
            )
            return
        raise RuntimeError(f"Modo record compat desconocido: {mode}")

    if _MAPPO_RECORD_COMPAT_MODE is not None:
        _invoke(_MAPPO_RECORD_COMPAT_MODE)
        return

    last_err: Optional[Exception] = None
    for mode in range(3):
        try:
            _invoke(mode)
            _MAPPO_RECORD_COMPAT_MODE = mode
            return
        except TypeError as err:
            last_err = err

    if last_err is not None:
        raise last_err
    raise RuntimeError("No se pudo ejecutar MAPPO.record_transition")


def _mappo_update_compat(agent: MAPPO, timestep: int, timesteps: int) -> None:
    """Llama la actualización MAPPO compatible con distintas versiones de skrl."""
    if hasattr(agent, "update"):
        agent.update(timestep=timestep, timesteps=timesteps)
        return

    if hasattr(agent, "_update"):
        # Algunas versiones exponen solo _update
        agent._update(timestep, timesteps)  # type: ignore[attr-defined]
        return

    raise AttributeError("MAPPO no expone update() ni _update()")


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINTS
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(agent: MAPPO, episode: int, checkpoint_dir: str,
                    stage_name: str = "") -> None:
    """Checkpoint con stage_name opcional en filename."""
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    safe = stage_name.replace(" ", "_") if stage_name else ""
    filename = f"mappo_stage{safe}_ep{episode:06d}.pt" if safe else f"mappo_ep{episode:06d}.pt"
    checkpoint_path = path / filename
    agent.save(str(checkpoint_path))
    print(f"[Checkpoint] Guardado: {checkpoint_path}")


def load_checkpoint(agent: MAPPO, checkpoint_path: str) -> None:
    """Carga pesos desde un checkpoint."""
    agent.load(checkpoint_path)
    print(f"[Checkpoint] Cargado: {checkpoint_path}")


# ─────────────────────────────────────────────────────────────────────────────
# LOOP DE ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

def training_loop(env_wrapper: NaoSoccerEnvWrapper,
                   agent: MAPPO,
                   logger: MetricsLogger,
                   trace_logger: Optional[TakeoverTraceLogger],
                   device: str,
                   use_fov: bool,
                   total_steps: int,
                   log_cfg: Any,
                   ppo_cfg=PPO_CFG,
                   initial_stage: int = 0) -> None:
    """
    Loop de entrenamiento MAPPO.

    Estructura:
      outer while  → episodios
      inner while  → pasos del episodio (ep_rewards acumula aquí)
      update       → cuando steps_since_update >= rollout_steps

    Issues resueltos:
      Issue 2: DiagnosticsProbe — verifica que pesos cambian tras update.
      Issue 4: act y record solo sobre active_agents; sin tensores dummy.
    """
    if initial_stage > 0:
        env_wrapper.set_curriculum_stage(initial_stage)

    total_env_steps    = 0
    episode            = 0
    steps_since_update  = 0
    probe               = DiagnosticsProbe()
    # reward_scale fijo — ver función scale_reward() y PPO_CFG.reward_scale
    # Contador de pasos consecutivos con acción NaN.
    nan_action_streak          = 0
    NAN_ACTION_RESET_THRESHOLD = 50   # pasos NaN consecutivos antes de resetear

    # ── Warmup: saltar los primeros N rollouts antes de actualizar ────────
    # Causa del NaN en ep=2: con rollout_steps=1024, el primer update llega
    # al paso 1024 cuando los pesos son todavía aleatorios y los rewards tienen
    # alta varianza (first_touch_bonus spikes). Un solo update con esa data
    # corrupta quema la red inmediatamente.
    # Solución: esperar WARMUP_ROLLOUTS rollouts completos antes del primer
    # update. Esto da tiempo para que el buffer acumule transitions más
    # representativas con pesos que ya generaron algo de señal.
    # 2 rollouts × 1024 steps = 2048 pasos de warmup ≈ 5 episodios de Stage 0.
    # Warmup siempre activo, incluso con checkpoint.
    # Razón: al cargar un Stage 0 checkpoint en Stage 1, el critic tiene V(s)≈+40
    # pero los rewards son -14 a -30. El primer update sin warmup causa NaN inmediato.
    # El warmup permite que el critic vea algunos episodios reales de Stage 1
    # antes del primer update, reduciendo el shock de distribución.
    WARMUP_ROLLOUTS: int = 1
    warmup_done:     bool = False   # siempre hacer warmup al inicio
    warmup_count:    int  = 0
    WATCHDOG_PRINT_EVERY_STEPS: int = 500

    obs, infos = env_wrapper.reset(seed=TRAINING.seed)
    probe.log_reset(infos, obs)   # Issue 2: keys de infos + norma de obs

    print(f"\n{'='*60}")
    print(f"Entrenamiento MAPPO NAO Soccer — Fase 1")
    print(f"Device: {device} | FOV: {use_fov}")
    print(f"Stage inicial: {env_wrapper.curriculum_manager.current_stage.name}")
    print(f"Total steps objetivo: {total_steps:,}")
    print(f"{'='*60}\n")

    all_agents = list(env_wrapper.possible_agents)
    tensor_cache = TrainingTensorCache(all_agents, device)

    while total_env_steps < total_steps:

        # Variables de episodio — una sola vez por episodio
        ep_rewards      = {aid: 0.0 for aid in env_wrapper.possible_agents}
        ep_steps        = 0
        ep_goal_scored  = 0
        ep_goal_ceded   = 0
        ep_start_time   = time.perf_counter()
        ep_env_step_ms_sum = 0.0
        ep_bridge_us_sum = 0.0
        ep_bridge_team_us_sum = 0.0
        ep_contract_validation_steps = 0
        ep_non_pipe_steps = 0
        ep_not_playing_steps = 0
        ep_stability_good_steps = 0
        ep_bridge_reconnect_max = 0
        current_rollout = getattr(
            env_wrapper.curriculum_manager.current_stage,
            'rollout_steps',
            ppo_cfg.rollout_steps,
        )

        # ── LOOP INTERNO: pasos del episodio ───────────────────────────────
        while True:
            active_agents = list(env_wrapper.agents)
            if not active_agents:
                break  # env ya terminó (post-truncation)

            # ── Act: SOLO active_agents (Issue 4) ─────────────────────────
            actions_dict: Dict[str, np.ndarray] = {}
            tensor_cache.load_policy_inputs(obs, infos)
            with torch.inference_mode():
                actions_t = _mappo_act_compat(
                    agent=agent,
                    observations=tensor_cache.obs,
                    states=tensor_cache.states,
                    timestep=total_env_steps,
                    timesteps=total_steps,
                )
                # ── NaN check: acciones directas de la policy ─────────────
                if NAN_DEBUG:
                    for aid, a in actions_t.items():
                        check_finite("actions_t", a, episode, total_env_steps, aid)

                # Solo enviar acciones para agentes activos al entorno.
                any_nan_this_step = False
                for aid in active_agents:
                    at = actions_t.get(aid)
                    if at is None:
                        actions_dict[aid] = np.zeros(ACTION.size, dtype=np.float32)
                    else:
                        a_np = np.ascontiguousarray(
                            at.detach().to(device="cpu", dtype=torch.float32).numpy().reshape(-1)
                        )
                        if not np.all(np.isfinite(a_np)):
                            any_nan_this_step = True
                        actions_dict[aid] = a_np

                # ── Contador de NaN consecutivos — reset si la red está quemada
                if any_nan_this_step:
                    nan_action_streak += 1
                    if nan_action_streak >= NAN_ACTION_RESET_THRESHOLD:
                        print(f"\n[RESET TRIGGER] {nan_action_streak} pasos consecutivos "
                              f"con acción NaN (ep={episode}, step={total_env_steps}). "
                              f"La policy está corrompida — reiniciando pesos.")
                        reset_policy_weights(agent, device)
                        nan_action_streak = 0
                        # Limpiar buffers y reactivar warmup para que los
                        # pesos recién inicializados no se quemen en el
                        # primer update posterior al reset.
                        for mem in agent.memories.values():
                            if hasattr(mem, "reset"):
                                mem.reset()
                        steps_since_update = 0
                        warmup_done  = False
                        warmup_count = 0
                        print("[WARMUP] Warmup reactivado tras reset de pesos.")
                else:
                    nan_action_streak = 0   # reset del streak si las acciones son finitas

                # ── NaN check: acciones decodificadas (numpy) ─────────────
                if NAN_DEBUG:
                    for aid, a in actions_dict.items():
                        check_finite("actions_dict", a, episode, total_env_steps, aid)

            # ── Step ───────────────────────────────────────────────────────
            curr_obs   = obs
            curr_infos = infos
            step_t0 = time.perf_counter()
            next_obs, rewards, terminateds, truncateds, infos = env_wrapper.step(actions_dict)
            step_ms = (time.perf_counter() - step_t0) * 1000.0
            ep_env_step_ms_sum += step_ms

            # ── NaN check: salidas del entorno ─────────────────────────────
            if NAN_DEBUG:
                for aid in active_agents:
                    check_finite("reward",   rewards.get(aid, 0.0),
                                 episode, total_env_steps, aid)
                    check_finite("next_obs", next_obs.get(aid, np.zeros(OBS.total)),
                                 episode, total_env_steps, aid)
                    check_finite("state",
                                 infos.get(aid, {}).get(
                                     "state", np.zeros(GLOBAL_STATE.total)),
                                 episode, total_env_steps, aid)

            probe.log_step(infos, rewards, active_agents)   # Issue 2

            # ── Record: SOLO active_agents (Issue 4) ───────────────────────
            # Agentes inactivos no tienen obs real, reward real ni acción real.
            # Sus ceros contaminarían el buffer con gradientes falsos.
            tensor_cache.load_transition_inputs(
                curr_obs=curr_obs,
                curr_infos=curr_infos,
                actions_dict=actions_dict,
                rewards=rewards,
                next_obs=next_obs,
                infos=infos,
                terminateds=terminateds,
                truncateds=truncateds,
                active_agents=active_agents,
            )

            with torch.no_grad():
                _mappo_record_transition_compat(
                    agent=agent,
                    observations=tensor_cache.obs,
                    states=tensor_cache.states,
                    actions=tensor_cache.actions,
                    rewards=tensor_cache.rewards,
                    next_observations=tensor_cache.next_obs,
                    next_states=tensor_cache.next_states,
                    terminated=tensor_cache.terminated,
                    truncated=tensor_cache.truncated,
                    infos=tensor_cache.infos,
                    timestep=total_env_steps,
                    timesteps=total_steps,
                )
            if False:
                _mappo_record_transition_compat(
                    agent=agent,
                    observations=tensor_cache.obs,
                    states=tensor_cache.states,
                    actions=tensor_cache.actions,
                    rewards={
                        # Normalizar rewards antes de entrar al buffer de skrl.
                        # Esto mantiene las ventajas GAE en un rango razonable
                        # para el critic, evitando que value_loss escale a ~15
                        # y desestabilice la policy compartida (ver causa raíz).
                        # La normalización es running (online) — no requiere
                        # conocer la distribución de rewards de antemano.
                        aid: torch.FloatTensor([
                            scale_reward(rewards.get(aid, 0.0))
                        ]).to(device)
                        for aid in all_agents
                    },
                    next_observations={
                        aid: torch.FloatTensor(
                            next_obs.get(aid, np.zeros(OBS.total, dtype=np.float32))
                        ).to(device) for aid in all_agents
                    },
                    next_states={
                        aid: torch.FloatTensor(
                            infos.get(aid, {}).get(
                                "state", np.zeros(GLOBAL_STATE.total, dtype=np.float32)
                            )
                        ).to(device) for aid in all_agents
                    },
                    terminated={
                        aid: torch.BoolTensor([terminateds.get(aid, aid not in active_agents)]).to(device)
                        for aid in all_agents
                    },
                    truncated={
                        aid: torch.BoolTensor([truncateds.get(aid, False)]).to(device)
                        for aid in all_agents
                    },
                    infos=infos_t,
                    timestep=total_env_steps,
                    timesteps=total_steps,
                )

            # ── Acumular métricas ──────────────────────────────────────────
            for aid in active_agents:
                ep_rewards[aid] += rewards.get(aid, 0.0)

            ep_steps           += 1
            total_env_steps    += len(active_agents)
            steps_since_update += len(active_agents)

            if infos:
                sample          = next(iter(infos.values()))
                if trace_logger is not None:
                    trace_logger.log_step(
                        episode=episode + 1,
                        stage_id=env_wrapper.curriculum_manager.current_stage.stage_id,
                        stage_name=env_wrapper.curriculum_manager.current_stage.name,
                        info=sample,
                    )
                ep_goal_scored += int(sample.get("goal_scored",   False))
                ep_goal_ceded  += int(sample.get("goal_conceded", False))
                ep_bridge_team_us_sum += float(
                    sample.get(
                        "bridge_debug_team_decision_time_us",
                        sample.get("bridge_decision_time_us", 0.0),
                    )
                )
                ep_bridge_us_sum += float(
                    sample.get(
                        "bridge_debug_active_agent_decision_time_us",
                        sample.get("bridge_decision_time_us", 0.0),
                    )
                )
                ep_contract_validation_steps += int(bool(sample.get("sidecar_contract_validation_mode", False)))
                transport_mode = sample.get("bridge_debug_transport_mode", "")
                ep_non_pipe_steps += int(transport_mode != expected_transport_mode())
                ep_not_playing_steps += int(sample.get("bridge_debug_game_state") != "playing")
                ep_bridge_reconnect_max = max(
                    ep_bridge_reconnect_max,
                    int(sample.get("bridge_debug_transport_reconnect_count", 0)),
                )
                is_stable_step = (
                    sample.get("sidecar_backend") == "real"
                    and not bool(sample.get("sidecar_contract_validation_mode", False))
                    and not bool(sample.get("takeover_simulator_active", False))
                    and is_stable_transport_mode(transport_mode)
                    and sample.get("bridge_debug_game_state") == "playing"
                    and sample.get("bridge_debug_requested_state") == "playing"
                )
                ep_stability_good_steps += int(is_stable_step)

                if ep_steps == 0 or ((ep_steps + 1) % WATCHDOG_PRINT_EVERY_STEPS == 0):
                    elapsed_ep_s = time.perf_counter() - ep_start_time
                    mean_env_step_ms = ep_env_step_ms_sum / max(1, ep_steps + 1)
                    mean_bridge_ms = (ep_bridge_us_sum / max(1, ep_steps + 1)) / 1000.0
                    mean_bridge_team_ms = (ep_bridge_team_us_sum / max(1, ep_steps + 1)) / 1000.0
                    memory_mb = _get_process_memory_mb()
                    stability_score = ep_stability_good_steps / max(1, ep_steps + 1)
                    print(
                        f"[Watchdog] ep={episode + 1} step={ep_steps + 1} "
                        f"elapsed={elapsed_ep_s:.1f}s "
                        f"env_step_mean={mean_env_step_ms:.1f}ms "
                        f"bridge_mean={mean_bridge_ms:.1f}ms "
                        f"bridge_team_mean={mean_bridge_team_ms:.1f}ms "
                        f"mem={memory_mb:.1f}MB "
                        f"stable={stability_score:.2f} "
                        f"game_state={sample.get('bridge_debug_game_state')} "
                        f"req_state={sample.get('bridge_debug_requested_state')}"
                    )

            obs = next_obs

            # ── Update ─────────────────────────────────────────────────────
            if steps_since_update >= current_rollout:
                if not warmup_done:
                    # Durante warmup: contar rollouts completos sin actualizar.
                    # El buffer se llena con data real antes del primer update.
                    warmup_count += 1
                    if warmup_count >= WARMUP_ROLLOUTS:
                        warmup_done = True
                        print(f"[WARMUP] Completados {WARMUP_ROLLOUTS} rollouts de warmup. "
                              f"Iniciando updates en step {total_env_steps}.")
                    else:
                        print(f"[WARMUP] Rollout {warmup_count}/{WARMUP_ROLLOUTS} "
                              f"— acumulando datos antes del primer update.")
                    steps_since_update = 0
                else:
                    probe.snapshot_weights(agent)
                    _mappo_update_compat(agent, timestep=total_env_steps, timesteps=total_steps)
                    probe.log_update(agent, total_env_steps)
                    if NAN_DEBUG and probe._update_count == 1:
                        _debug_tracking_data_keys(agent)
                    steps_since_update = 0

            if truncateds.get("__all__") or terminateds.get("__all__"):
                break

        # ── LOGGING DEL EPISODIO ───────────────────────────────────────────
        episode   += 1
        n_active   = env_wrapper.curriculum_manager.current_stage.num_rl_agents
        active_ids = env_wrapper.possible_agents[:n_active]
        mean_rew   = float(np.mean([ep_rewards[aid] for aid in active_ids]))
        stage      = env_wrapper.curriculum_manager.current_stage
        episode_wall_s = time.perf_counter() - ep_start_time
        mean_env_step_ms = ep_env_step_ms_sum / max(1, ep_steps)
        mean_bridge_ms = (ep_bridge_us_sum / max(1, ep_steps)) / 1000.0
        mean_bridge_team_ms = (ep_bridge_team_us_sum / max(1, ep_steps)) / 1000.0
        process_memory_mb = _get_process_memory_mb()
        stability_score = ep_stability_good_steps / max(1, ep_steps)
        contract_validation_rate = ep_contract_validation_steps / max(1, ep_steps)
        non_pipe_rate = ep_non_pipe_steps / max(1, ep_steps)
        not_playing_rate = ep_not_playing_steps / max(1, ep_steps)

        policy_loss = value_loss = entropy = 0.0
        if hasattr(agent, "tracking_data") and agent.tracking_data:
            td = agent.tracking_data
            # Búsqueda robusta por substring — las claves exactas varían entre
            # versiones de skrl (e.g. "Loss / Policy loss" vs "policy_loss").
            def _find_last(td: dict, *substrings: str) -> float:
                for key, vals in td.items():
                    key_lower = key.lower()
                    if any(s.lower() in key_lower for s in substrings):
                        return float(vals[-1]) if vals else 0.0
                return 0.0

            policy_loss = _find_last(td, "policy")
            value_loss  = _find_last(td, "value")
            entropy     = _find_last(td, "entropy")

        # ── Avance de curriculum (BHuman-aware) ────────────────────────────
        current_stage_id   = env_wrapper.curriculum_manager.stage_id
        current_stage_name = env_wrapper.curriculum_manager.current_stage.name
        advanced = env_wrapper.record_episode_metrics(
            own_score = env_wrapper._env._game.own_score,
            opp_score = env_wrapper._env._game.opp_score,
            infos     = infos,
        )
        if advanced:
            ckpt = f"{current_stage_id}_{current_stage_name}"
            save_checkpoint(agent, episode, log_cfg.checkpoint_dir, stage_name=ckpt)
            env_wrapper._sync_stage()
            stage = env_wrapper.curriculum_manager.current_stage
            print(f"\n[Curriculum] Stage {current_stage_id}({current_stage_name}) → Stage {stage.stage_id}({stage.name})")

        # Get BHuman metrics from aggregator
        bhuman_m = env_wrapper.metrics.get_metrics()
        sample_info = next(iter(infos.values())) if infos else {}

        logger.log_episode(
            own_score             =env_wrapper._env._game.own_score,
            opp_score             =env_wrapper._env._game.opp_score,
            mean_reward           =mean_rew,
            steps                 =ep_steps,
            stage_id              =stage.stage_id,
            stage_name            =stage.name,
            handoff_reached_rate  =bhuman_m["handoff_reached_rate"],
            takeover_ready_rate   =bhuman_m["takeover_ready_rate"],
            takeover_success_rate =bhuman_m["takeover_success_rate"],
            takeover_success_rate_dribble_duel = bhuman_m["takeover_success_rate_dribble_duel"],
            takeover_success_rate_shoot = bhuman_m["takeover_success_rate_shoot"],
            takeover_success_rate_pass = bhuman_m["takeover_success_rate_pass"],
            takeover_coverage_rate_dribble_duel = bhuman_m["takeover_coverage_rate_dribble_duel"],
            takeover_coverage_rate_shoot = bhuman_m["takeover_coverage_rate_shoot"],
            takeover_coverage_rate_pass = bhuman_m["takeover_coverage_rate_pass"],
            playball_entry_rate   =bhuman_m["playball_entry_rate"],
            ready_but_walk_rate   =bhuman_m["ready_but_walk_rate"],
            ready_touch_but_walk_rate=bhuman_m["ready_touch_but_walk_rate"],
            policy_loss           =policy_loss,
            value_loss            =value_loss,
            entropy               =entropy,
            episode_wall_s        =episode_wall_s,
            mean_env_step_ms      =mean_env_step_ms,
            mean_bridge_ms        =mean_bridge_ms,
            mean_bridge_team_ms   =mean_bridge_team_ms,
            process_memory_mb     =process_memory_mb,
            stability_score       =stability_score,
            bridge_reconnect_count=ep_bridge_reconnect_max,
            contract_validation_rate=contract_validation_rate,
            non_pipe_rate         =non_pipe_rate,
            not_playing_rate      =not_playing_rate,
            ep_takeover_timeout_rate=float(
                sample_info.get("episode_takeover_timeout_rate", 0.0)
            ),
            ep_takeover_skill_cleared_rate=float(
                sample_info.get("episode_takeover_skill_cleared_rate", 0.0)
            ),
            ep_takeover_active_change_rate=float(
                sample_info.get("episode_takeover_active_change_rate", 0.0)
            ),
            ep_takeover_other_end_rate=float(
                sample_info.get("episode_takeover_other_end_rate", 0.0)
            ),
        )

        if episode % log_cfg.print_every_eps == 0:
            logger.print_summary(episode)
            prog = env_wrapper.curriculum_manager.get_progress_stats()
            print(f"         Curriculum: rate={prog['current_rate']:.3f} "
                  f"(umbral={prog['advance_threshold']}) "
                  f"ventana={prog['window_fill']}")
            print(f"         Watchdog: episode_wall={episode_wall_s:.1f}s "
                  f"mean_env_step={mean_env_step_ms:.1f}ms "
                  f"mean_bridge={mean_bridge_ms:.1f}ms "
                  f"mean_bridge_team={mean_bridge_team_ms:.1f}ms "
                  f"handoff={bhuman_m['handoff_reached_rate']:.2f} "
                  f"playball={bhuman_m['playball_entry_rate']:.2f} "
                  f"ready_walk={bhuman_m['ready_but_walk_rate']:.2f} "
                  f"to={float(sample_info.get('episode_takeover_timeout_rate', 0.0)):.2f} "
                  f"clr={float(sample_info.get('episode_takeover_skill_cleared_rate', 0.0)):.2f} "
                  f"mem={process_memory_mb:.1f}MB "
                  f"stable={stability_score:.2f} "
                  f"cv_rate={contract_validation_rate:.2f} "
                  f"non_pipe={non_pipe_rate:.2f} "
                  f"reconnects={ep_bridge_reconnect_max} "
                  f"steps={ep_steps}")

        if episode % log_cfg.save_every_eps == 0:
            save_checkpoint(agent, episode, log_cfg.checkpoint_dir)

        if episode % log_cfg.plot_every_eps == 0:
            logger.plot_convergence()
            logger.plot_scores()
            logger.plot_rates()
            logger.plot_watchdog()

        obs, infos = env_wrapper.reset()

    # ── Final ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Entrenamiento completado. {total_env_steps:,} steps, {episode} episodios.")
    print(f"{'='*60}")
    save_checkpoint(agent, episode, log_cfg.checkpoint_dir)
    logger.plot_convergence()
    logger.plot_scores()
    logger.plot_rates()
    logger.plot_watchdog()
    print(f"Resultados en: {logger._dir}/")

def evaluate(env_wrapper: NaoSoccerEnvWrapper,
              agent: MAPPO,
              n_episodes: int,
              device: str) -> Dict[str, float]:
    """
    Evalúa el agente durante n_episodes y retorna métricas promedio.
    Sin actualización de gradientes.
    """
    results = []
    eval_timestep = 0

    for ep in range(n_episodes):
        def policy_fn(obs_dict):
            nonlocal eval_timestep
            actions = {}
            with torch.no_grad():
                observations_t: Dict[str, torch.Tensor] = {}
                states_t: Dict[str, torch.Tensor] = {}

                for aid in env_wrapper.possible_agents:
                    obs_arr = obs_dict.get(aid, np.zeros(OBS.total, dtype=np.float32))
                    obs_t = torch.FloatTensor(obs_arr).to(device)
                    state_t = torch.zeros((GLOBAL_STATE.total,), dtype=torch.float32, device=device)
                    observations_t[aid] = obs_t
                    states_t[aid] = state_t

                if observations_t:
                    actions_t = _mappo_act_compat(
                        agent=agent,
                        observations=observations_t,
                        states=states_t,
                        timestep=eval_timestep,
                        timesteps=max(1, n_episodes),
                    )
                    for aid, action_t in actions_t.items():
                        actions[aid] = action_t.detach().cpu().numpy().reshape(-1)
                eval_timestep += 1
            return actions

        result = env_wrapper.collect_episode(policy_fn=policy_fn, seed=ep)
        results.append(result)

    own_scores = [r["own_score"]   for r in results]
    opp_scores = [r["opp_score"]   for r in results]
    rewards    = [r["mean_reward"] for r in results]

    summary = {
        "mean_own_score": float(np.mean(own_scores)),
        "mean_opp_score": float(np.mean(opp_scores)),
        "goal_rate":      float(np.mean([s > 0 for s in own_scores])),
        "win_rate":       float(np.mean([o > p for o, p in zip(own_scores, opp_scores)])),
        "mean_reward":    float(np.mean(rewards)),
    }

    print(f"\n[Evaluación — {n_episodes} eps]")
    for k, v in summary.items():
        print(f"  {k}: {v:.3f}")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def _debug_tracking_data_keys(agent: MAPPO) -> None:
    """
    Imprime las claves reales de tracking_data de skrl en el primer update.

    Por qué importa: el logger asume que las claves son exactamente
    "Loss / Policy loss", "Loss / Value loss", "Loss / Entropy loss".
    Si skrl usa nombres distintos (cambian entre versiones), el logger
    siempre registra 0.0 aunque el update sí ocurra, produciendo las
    gráficas de pérdida plana que ves en convergence.png.
    """
    if not hasattr(agent, "tracking_data") or not agent.tracking_data:
        print("[DEBUG tracking_data] dict vacío o no existe — el update no pobló tracking_data")
        return
    print("\n[DEBUG tracking_data] Claves disponibles en agent.tracking_data:")
    for k, v in agent.tracking_data.items():
        last_val = v[-1] if v else "LISTA VACÍA"
        print(f"  {repr(k):50s} → último valor: {last_val}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--quick-debug",
        action="store_true",
        help="Run the lightweight quick-debug training profile",
    )
    args, _unknown_args = parser.parse_known_args()

    # ── Configuración centralizada — editar aquí, no en CLI ─────────────────
    # Para cambiar parámetros: modificar TRAINING en config.py o las
    # constantes de abajo directamente.
    INITIAL_STAGE: int        = 0    # stage de inicio (0-6)
    QUICK_DEBUG_PROFILE: bool = args.quick_debug
    PRACTICAL_TRAINING_PROFILE: bool = not QUICK_DEBUG_PROFILE
    # FOV SOLO afecta a la detección de OPONENTES (cono ±27.5°, 5000mm).
    # El BALÓN es siempre omnisciente (TeamBallModel de BHuman) — ver plan .md.
    # True  = oponentes filtrados por FOV realista (modo entrenamiento normal)
    # False = oponentes omniscientes (baseline de diagnóstico)
    USE_OPPONENT_FOV: bool = True
    CHECKPOINT:    Optional[str] = None
    EVAL_ONLY:     bool       = False    # True = solo evaluar (requiere CHECKPOINT)
    DEVICE:        Optional[str] = None  # None = auto-detectar cuda/cpu
    # ────────────────────────────────────────────────────────────────────────

    if NAN_DEBUG:
        torch.autograd.set_detect_anomaly(True)
        print("[Main] torch.autograd.set_detect_anomaly(True) ACTIVADO")

    device = DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Main] Device: {device}")
    if device == "cuda":
        print(f"[Main] GPU: {torch.cuda.get_device_name(0)}")

    active_curriculum = CURRICULUM
    active_ppo_cfg = PPO_CFG
    active_training_cfg = TRAINING
    active_log_cfg = LOG

    if QUICK_DEBUG_PROFILE:
        active_curriculum = build_quick_debug_curriculum(CURRICULUM, QUICK_DEBUG)
        active_ppo_cfg = build_quick_debug_ppo_cfg(PPO_CFG, QUICK_DEBUG)
        active_training_cfg = build_quick_debug_training_cfg(TRAINING, QUICK_DEBUG)
        active_log_cfg = build_quick_debug_logging_cfg(LOG, QUICK_DEBUG)
        print("[Main] QUICK_DEBUG_PROFILE activado")
        print(
            f"[Main] Quick profile: total_steps={active_training_cfg.total_steps}, "
            f"rollout_steps={active_ppo_cfg.rollout_steps}, "
            f"stage0_episode_steps={active_curriculum[0].episode_steps}, "
            f"stage0_window={active_curriculum[0].advance_window}"
        )
    elif PRACTICAL_TRAINING_PROFILE:
        active_training_cfg = build_practical_training_cfg(TRAINING, PRACTICAL_TRAINING)
        active_log_cfg = build_practical_logging_cfg(LOG, PRACTICAL_TRAINING)
        print("[Main] PRACTICAL_TRAINING_PROFILE activado")
        print(
            f"[Main] Practical profile: total_steps={active_training_cfg.total_steps}, "
            f"save_every={active_log_cfg.save_every_eps}, "
            f"plot_every={active_log_cfg.plot_every_eps}, "
            f"print_every={active_log_cfg.print_every_eps}"
        )

    env_wrapper = NaoSoccerEnvWrapper(
        stage_cfg=active_curriculum[INITIAL_STAGE],
        curriculum_stages=active_curriculum,
        use_fov=USE_OPPONENT_FOV,
        seed=active_training_cfg.seed,
    )

    agent = build_mappo_agent(env_wrapper, device, ppo_cfg=active_ppo_cfg)

    if hasattr(agent, "init"):
        agent.init()

    if CHECKPOINT:
        load_checkpoint(agent, CHECKPOINT)

    if EVAL_ONLY:
        evaluate(env_wrapper, agent, n_episodes=100, device=device)
        return

    run_prefix = "quickdebug_" if QUICK_DEBUG_PROFILE else ("practical_" if PRACTICAL_TRAINING_PROFILE else "")
    run_name = f"{run_prefix}stage{INITIAL_STAGE}_fov{USE_OPPONENT_FOV}_{int(time.time())}"
    log_dir  = os.path.join(LOG.log_dir, run_name)
    logger   = MetricsLogger(log_dir)
    trace_logger = TakeoverTraceLogger.maybe_create(log_dir)
    active_log_cfg.checkpoint_dir = os.path.join(log_dir, "checkpoints")

    print(f"[Main] Logs en: {log_dir}")
    if trace_logger is not None:
        print(f"[Main] Takeover trace en: {trace_logger.path}")

    try:
        training_loop(
            env_wrapper   =env_wrapper,
            agent         =agent,
            logger        =logger,
            trace_logger  =trace_logger,
            device        =device,
            use_fov       =USE_OPPONENT_FOV,
            total_steps   =active_training_cfg.total_steps,
            log_cfg       =active_log_cfg,
            ppo_cfg       =active_ppo_cfg,
            initial_stage =INITIAL_STAGE,
        )
    finally:
        if trace_logger is not None:
            trace_logger.close()


if __name__ == "__main__":
    main()
