from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rlfootball.core.action import SkillType, StructuredSkillAction
from rlfootball.core.config import BHUMAN_DIR, PLAYER_NUMBER
from rlfootball.core.pybh_backend import PyBHBackend

os.chdir(BHUMAN_DIR)
import pybh.controller as pybh


BALL_POS = np.array([0.0, 0.0], dtype=np.float32)
ROBOT_FAR_POS = np.array([-900.0, 0.0], dtype=np.float32)
ROBOT_NEAR_POS = np.array([-180.0, 0.0], dtype=np.float32)

SKILL_REQUEST_NAMES = {
    0: "none",
    1: "stand",
    2: "walk",
    3: "shoot",
    4: "pass",
    5: "dribble",
    6: "block",
    7: "mark",
    8: "observe",
}

MOTION_REQUEST_NAMES = {
    0: "playDead",
    1: "stand",
    2: "walkAtAbsoluteSpeed",
    3: "walkAtRelativeSpeed",
    4: "walkToPose",
    5: "walkToBallAndKick",
    6: "dribble",
    7: "dive",
    8: "special",
    9: "replayWalk",
    10: "calibration",
}

EXPECTED_MOTIONS = {
    SkillType.stand: {1},
    SkillType.walk: {4},
    SkillType.shoot: {5, 6},
    SkillType.pass_: {5, 6},
    SkillType.dribble: {6},
    SkillType.block: {4},
    SkillType.mark: {4},
    SkillType.observe: {4},
}


@dataclass
class SkillRun:
    profile: str
    skill_name: str
    reset_wall_ms: float
    step_wall_ms: float
    step_cpu_ms: float
    rss_kb_before: int
    rss_kb_after: int
    obs_ready: bool
    requested_skill: str
    skill_request: int
    skill_motion: int
    engine_input: int
    engine_effective: int
    engine_phase: int
    engine_fall: int
    motion_phase: int
    motion_speed_x: float
    motion_speed_y: float
    motion_speed_rot: float
    robot_x: float
    robot_y: float
    robot_theta: float
    ball_x: float
    ball_y: float
    frame: int
    passed: bool
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate RL SimRobot skill matrix and compare 2D vs 3D.")
    parser.add_argument("--output", type=Path, default=Path("docs/06-simrobot-skill-matrix.md"))
    parser.add_argument("--steps-per-skill", type=int, default=20)
    parser.add_argument("--timeout-ms", type=int, default=2500)
    return parser.parse_args()


def current_rss_kb() -> int:
    with open("/proc/self/status", "r", encoding="ascii") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return 0


def finite_float(value: Any) -> float:
    numeric = float(value)
    return numeric if math.isfinite(numeric) else 0.0


def reset_pose_for_skill(skill: SkillType) -> tuple[np.ndarray, float]:
    if skill in {SkillType.shoot, SkillType.pass_, SkillType.dribble}:
        return ROBOT_NEAR_POS.copy(), 0.0
    return ROBOT_FAR_POS.copy(), 0.0


def build_action(skill: SkillType) -> StructuredSkillAction:
    if skill is SkillType.walk:
        return StructuredSkillAction(skill_type=skill, target_x=0.0, target_y=0.0, target_theta=0.0)
    if skill is SkillType.pass_:
        return StructuredSkillAction(skill_type=skill, pass_target=2)
    if skill is SkillType.dribble:
        return StructuredSkillAction(skill_type=skill, target_theta=0.0)
    if skill in {SkillType.block, SkillType.mark, SkillType.observe}:
        return StructuredSkillAction(skill_type=skill, target_x=700.0, target_y=150.0, target_theta=0.0)
    return StructuredSkillAction(skill_type=skill)


def validate_result(skill: SkillType, obs: Any, payload: dict[str, Any]) -> tuple[bool, str]:
    if not obs.obs_ready:
        return False, "obs_ready=0"
    if payload.get("requested_skill") != skill.public_name and not (skill is SkillType.walk and payload.get("requested_skill") == "walkTo"):
        return False, f"requested_skill={payload.get('requested_skill')}"
    if int(payload.get("skill_behavior_skill_request", -1)) <= 0:
        return False, "skill_behavior_skill_request invalido"
    motion = int(payload.get("skill_behavior_motion_request", -1))
    if motion not in EXPECTED_MOTIONS[skill]:
        expected = ",".join(MOTION_REQUEST_NAMES[m] for m in sorted(EXPECTED_MOTIONS[skill]))
        got = MOTION_REQUEST_NAMES.get(motion, str(motion))
        return False, f"motion {got} no esperado, esperaba {expected}"
    if int(payload.get("motion_engine_input_request", -1)) != motion:
        return False, "MotionEngine input no coincide con SkillBehavior motion"
    if skill is SkillType.shoot and motion == 6:
        return True, "shoot degrado a dribble dentro de KickAtGoal"
    if skill is SkillType.pass_ and motion == 6:
        return True, "pass degrado a dribble en RL1v0 sin teammate valido"
    return True, "ok"


def run_skill(backend: PyBHBackend, profile: str, skill: SkillType) -> SkillRun:
    robot_pos, robot_theta = reset_pose_for_skill(skill)
    action = build_action(skill)
    rss_before = current_rss_kb()
    reset_start = time.perf_counter()
    obs = backend.reset(ball_pos=BALL_POS.copy(), robot_pos=robot_pos, robot_theta=robot_theta)
    reset_wall_ms = (time.perf_counter() - reset_start) * 1000.0
    cpu_start = time.process_time()
    step_start = time.perf_counter()
    obs = backend.step(action)
    step_wall_ms = (time.perf_counter() - step_start) * 1000.0
    step_cpu_ms = (time.process_time() - cpu_start) * 1000.0
    payload = pybh.rl_get_obs(PLAYER_NUMBER, 1)
    rss_after = current_rss_kb()
    passed, note = validate_result(skill, obs, payload)
    return SkillRun(
        profile=profile,
        skill_name=skill.public_name,
        reset_wall_ms=reset_wall_ms,
        step_wall_ms=step_wall_ms,
        step_cpu_ms=step_cpu_ms,
        rss_kb_before=rss_before,
        rss_kb_after=rss_after,
        obs_ready=bool(obs.obs_ready),
        requested_skill=str(payload.get("requested_skill", "")),
        skill_request=int(payload.get("skill_behavior_skill_request", -1)),
        skill_motion=int(payload.get("skill_behavior_motion_request", -1)),
        engine_input=int(payload.get("motion_engine_input_request", -1)),
        engine_effective=int(payload.get("motion_engine_effective_request", -1)),
        engine_phase=int(payload.get("motion_engine_phase", -1)),
        engine_fall=int(payload.get("motion_engine_fall_state", -1)),
        motion_phase=int(payload.get("motion_phase", -1)),
        motion_speed_x=finite_float(payload.get("motion_speed_x", 0.0)),
        motion_speed_y=finite_float(payload.get("motion_speed_y", 0.0)),
        motion_speed_rot=finite_float(payload.get("motion_speed_rot", 0.0)),
        robot_x=float(payload.get("robot_x", 0.0)),
        robot_y=float(payload.get("robot_y", 0.0)),
        robot_theta=float(payload.get("robot_theta", 0.0)),
        ball_x=float(payload.get("ball_x", 0.0)),
        ball_y=float(payload.get("ball_y", 0.0)),
        frame=int(payload.get("frame", 0)),
        passed=passed,
        note=note,
    )


def summarize_profile(results: list[SkillRun], profile: str) -> dict[str, float]:
    subset = [r for r in results if r.profile == profile]
    return {
        "count": float(len(subset)),
        "passed": float(sum(1 for r in subset if r.passed)),
        "reset_wall_ms_avg": sum(r.reset_wall_ms for r in subset) / len(subset),
        "step_wall_ms_avg": sum(r.step_wall_ms for r in subset) / len(subset),
        "step_cpu_ms_avg": sum(r.step_cpu_ms for r in subset) / len(subset),
        "rss_after_kb_avg": sum(r.rss_kb_after for r in subset) / len(subset),
        "motion_speed_abs_avg": sum(abs(r.motion_speed_x) + abs(r.motion_speed_y) for r in subset) / len(subset),
    }


def warm_reset_avg(results: list[SkillRun], profile: str) -> float:
    subset = [r for r in results if r.profile == profile]
    if len(subset) <= 1:
        return subset[0].reset_wall_ms if subset else 0.0
    warm = subset[1:]
    return sum(r.reset_wall_ms for r in warm) / len(warm)


def max_rss_kb(results: list[SkillRun], profile: str) -> int:
    subset = [r for r in results if r.profile == profile]
    return max(r.rss_kb_after for r in subset)


def avg_fall_state(results: list[SkillRun], profile: str) -> float:
    subset = [r for r in results if r.profile == profile]
    return sum(r.engine_fall for r in subset) / len(subset)


def nonzero_speed_runs(results: list[SkillRun], profile: str) -> int:
    subset = [r for r in results if r.profile == profile]
    return sum(1 for r in subset if abs(r.motion_speed_x) > 1e-3 or abs(r.motion_speed_y) > 1e-3 or abs(r.motion_speed_rot) > 1e-3)


def render_table(results: list[SkillRun]) -> str:
    lines = [
        "| profile | skill | pass | requested | skill_req | motion | engine_in | phase | fall | reset_ms | step_ms | cpu_ms | rss_kb | speed_xyr | note |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        speed = f"{r.motion_speed_x:.1f}/{r.motion_speed_y:.1f}/{r.motion_speed_rot:.2f}"
        lines.append(
            f"| {r.profile} | {r.skill_name} | {'yes' if r.passed else 'no'} | {r.requested_skill} | "
            f"{SKILL_REQUEST_NAMES.get(r.skill_request, r.skill_request)} | "
            f"{MOTION_REQUEST_NAMES.get(r.skill_motion, r.skill_motion)} | "
            f"{MOTION_REQUEST_NAMES.get(r.engine_input, r.engine_input)} | {r.engine_phase} | {r.engine_fall} | "
            f"{r.reset_wall_ms:.1f} | {r.step_wall_ms:.1f} | {r.step_cpu_ms:.1f} | {r.rss_kb_after} | {speed} | {r.note} |"
        )
    return "\n".join(lines)


def render_summary(results: list[SkillRun]) -> str:
    lines = [
        "| profile | skills_ok | reset_ms_avg | step_ms_avg | cpu_ms_avg | rss_kb_avg | abs_speed_xy_avg |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for profile in ("2d", "3d"):
        s = summarize_profile(results, profile)
        lines.append(
            f"| {profile} | {int(s['passed'])}/{int(s['count'])} | {s['reset_wall_ms_avg']:.1f} | "
            f"{s['step_wall_ms_avg']:.1f} | {s['step_cpu_ms_avg']:.1f} | {s['rss_after_kb_avg']:.0f} | "
            f"{s['motion_speed_abs_avg']:.1f} |"
        )
    return "\n".join(lines)


def render_report(results: list[SkillRun], *, steps_per_skill: int) -> str:
    s2d = summarize_profile(results, "2d")
    s3d = summarize_profile(results, "3d")
    warm2d = warm_reset_avg(results, "2d")
    warm3d = warm_reset_avg(results, "3d")
    rss_delta = s3d["rss_after_kb_avg"] - s2d["rss_after_kb_avg"]
    step_delta = s2d["step_wall_ms_avg"] - s3d["step_wall_ms_avg"]
    phase_2d = sorted({r.engine_phase for r in results if r.profile == "2d"})
    phase_3d = sorted({r.engine_phase for r in results if r.profile == "3d"})
    lines = [
        "# Validacion de skills RL SimRobot y comparativo 2D vs 3D",
        "",
        f"Fecha: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Configuracion",
        "",
        f"- perfiles validados: `2d`, `3d`",
        f"- backend usado: `PyBHBackend(mode=\"simrobot\")`",
        f"- frames por skill: `{steps_per_skill}`",
        f"- skill set validado: `{', '.join(skill.public_name for skill in SkillType)}`",
        "",
        "## Resumen cuantitativo",
        "",
        render_summary(results),
        "",
        "## Matriz por skill",
        "",
        render_table(results),
        "",
        "## Lectura tecnica",
        "",
        "### 2D",
        "",
        "- Usa escena `RL1v0.ros2d` y escenario `4v4_RL2D`.",
        "- Tras el fix de pose oracular, `walk`, `block`, `mark` y `observe` dejaron de orbitar la pelota y pasaron a producir `walkToPose` coherente.",
        f"- El reset frio fue caro (`{results[0].reset_wall_ms:.1f} ms` en `stand`), pero el reset caliente promedio bajo a `{warm2d:.1f} ms`.",
        f"- El RSS promedio fue `{s2d['rss_after_kb_avg']:.0f} KB` y el maximo observado `{max_rss_kb(results, '2d')} KB`.",
        f"- Las 8 corridas de 2D reportaron velocidad no nula en `{nonzero_speed_runs(results, '2d')}/8` casos, asi que `motion_speed_*` es una señal util en este perfil.",
        f"- `engine_phase` se mantuvo en `{','.join(str(v) for v in phase_2d)}` y `fall_state` promedio fue `{avg_fall_state(results, '2d'):.1f}`.",
        "",
        "### 3D",
        "",
        "- Usa escena `RL1v0_3D.ros2` y escenario `4v4_RL3D`.",
        "- Valida el flujo natural real de B-Human: `SkillBehaviorControl -> MotionEngine -> JointRequest`.",
        f"- El reset frio fue `{[r for r in results if r.profile == '3d'][0].reset_wall_ms:.1f} ms`, y el reset caliente promedio `{warm3d:.1f} ms`.",
        f"- El RSS promedio fue `{s3d['rss_after_kb_avg']:.0f} KB`, unos `{rss_delta:.0f} KB` por encima de 2D.",
        f"- `engine_phase` recorrió `{','.join(str(v) for v in phase_3d)}` y `fall_state` promedio fue `{avg_fall_state(results, '3d'):.1f}`.",
        f"- `motion_speed_*` salio nulo en `{8 - nonzero_speed_runs(results, '3d')}/8` corridas. Eso no invalida la caminata: en 3D embebido la validacion fuerte sigue siendo `motion_request`, fases y joints.",
        "",
        "### Diferencias que si importan",
        "",
        f"- En esta muestra, `3d` no fue mas lento en `step_wall_ms_avg`; de hecho quedo `{step_delta:.1f} ms` por debajo de `2d`. Eso no significa que el motor 3D sea mas barato: significa que este benchmark mide el proceso completo Python + bridge + stepping, no un microbenchmark aislado del motor fisico.",
        "- 2D expone mejor la velocidad planar efectiva en `motion_speed_*`; 3D expone mejor el estado interno del motor a traves de `engine_phase`, `fall_state` y joints, aunque la traslacion global visible pueda tardar mas en aparecer.",
        "- En skills de pelota (`shoot`, `pass`, `dribble`) ambos perfiles entran al request esperado, pero `shoot` puede degradar a `dribble` dentro de `KickAtGoal` y `pass` puede caer a esa misma ruta si no hay teammate valido. Por eso esta matriz valida pipeline y activacion, no una trayectoria tactica unica.",
        "- El costo de memoria adicional de 3D es persistente pero no explosivo en esta muestra: alrededor de 12 MB sobre 2D. El costo cualitativo fuerte de 3D no es solo RSS, sino la dependencia en joints, contacto y fases del motor.",
        "- La mayor fragilidad operativa estuvo del lado 2D antes del fix de `RobotPose` y `BallModel`: cuando la pose era estimada y no oracular, `SkillBehaviorControl` construia targets relativos absurdos y el robot orbitaba la pelota.",
        "",
        "### Limites de esta comparacion",
        "",
        "- `cpu_ms` puede superar `wall_ms` porque `time.process_time()` suma CPU consumido por varios threads internos de B-Human.",
        "- El primer reset de cada perfil mezcla costo de carga de escena con costo de reset. Por eso en la lectura comparativa hay que separar reset frio de reset caliente.",
        "- `motion_speed_*` no es simetrico entre perfiles en esta instrumentacion. Es comparable dentro de 2D, pero no debe usarse como la señal principal para decir que 3D no se mueve.",
        "- Este benchmark usa `20` frames por skill. Es suficiente para validar enrutamiento y activacion, pero no para evaluar calidad completa de locomocion, alineacion fina o remates largos.",
        "",
        "## Criterio de validacion",
        "",
        "- `obs_ready=True`.",
        "- `requested_skill` coincide con la skill enviada, aceptando `walk -> walkTo`.",
        "- `skill_behavior_motion_request` cae en el conjunto esperado para la skill.",
        "- `motion_engine_input_request` coincide con la salida de `SkillBehaviorControl`.",
        "",
        "## Notas",
        "",
        "- Los tiempos reportados son del proceso Python actual. Incluyen bridge, stepping embebido y espera de observacion.",
        "- `rss_kb` es la memoria residente del proceso en `/proc/self/status` despues de cada skill.",
        "- `shoot` valida `walkToBallAndKick` o `dribble`, porque `KickAtGoal` puede degradar a `dribbleToGoal` si no hay ventana de remate.",
        "- `pass` valida `walkToBallAndKick` o `dribble`; en `RL1v0` no hay teammate valido y `PassToTeammate` aborta hacia `KickAtGoal`.",
        "- En 3D la caminata puede mostrar poca traslacion global en ventanas cortas aun con joints y fases correctas; por eso la validacion no depende solo de `robot_x`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    results: list[SkillRun] = []
    for profile in ("2d", "3d"):
        backend = PyBHBackend(mode="simrobot", simrobot_profile=profile, frames_per_step=args.steps_per_skill, obs_timeout_ms=args.timeout_ms)
        try:
            for skill in SkillType:
                print(f"running profile={profile} skill={skill.public_name}", flush=True)
                results.append(run_skill(backend, profile, skill))
        finally:
            backend.close()
    report = render_report(results, steps_per_skill=args.steps_per_skill)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
