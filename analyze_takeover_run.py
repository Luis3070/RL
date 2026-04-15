from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from config import FIELD
from takeover_trace import TRACE_FILE_NAME


def _to_float(row: dict[str, str], key: str) -> float:
    raw = row.get(key, "")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _to_int(row: dict[str, str], key: str) -> int:
    raw = row.get(key, "")
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return 0


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _avg(rows: Iterable[dict[str, str]], key: str) -> float:
    vals = [_to_float(row, key) for row in rows]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def _bucket_xy(x: float | None, y: float | None, bucket_mm: int) -> str:
    if x is None or y is None:
        return "-"
    bx = int(round(x / bucket_mm) * bucket_mm)
    by = int(round(y / bucket_mm) * bucket_mm)
    return f"({bx},{by})"


def _normalize_angle_rad(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _count_oscillation_events(rows: list[dict[str, str]]) -> tuple[int, int]:
    bearing_flips = 0
    heading_jumps = 0
    prev: dict[str, float | int] | None = None
    for row in rows:
        episode = int(_to_float(row, "episode"))
        env_step = int(_to_float(row, "env_step"))
        bearing = _to_float(row, "agent_ball_bearing_deg")
        theta = _to_float(row, "agent_theta_rad")
        if prev is not None and episode == prev["episode"] and env_step == prev["env_step"] + 1:
            prev_bearing = float(prev["bearing"])
            if abs(prev_bearing) >= 12.0 and abs(bearing) >= 12.0:
                if (prev_bearing > 0.0) != (bearing > 0.0):
                    bearing_flips += 1
            theta_delta = abs(_normalize_angle_rad(theta - float(prev["theta"])))
            if theta_delta >= math.radians(35.0):
                heading_jumps += 1
        prev = {
            "episode": episode,
            "env_step": env_step,
            "bearing": bearing,
            "theta": theta,
        }
    return bearing_flips, heading_jumps


# Columnas de flags por step (sumar por episodio) — instrumentación extendida
_TRACE_DIAG_FLAG_SUM_KEYS: tuple[str, ...] = (
    "diag_ball_teleport_flag",
    "diag_agent_teleport_flag",
    "diag_shoot_backward_flag",
    "diag_theta_snap_prep_flag",
    "diag_control_owner_changed",
    "diag_active_agent_changed_step",
    "diag_ball_large_jump_flag",
    "diag_ball_agent_overlap_flag",
    "diag_ball_agent_penetration_flag",
    "diag_opp_pressure_flag",
    "diag_theta_snap_motion_prep_flag",
    "diag_rapid_kick_proxy_flag",
    "diag_sidecar_macro_neq_takeover_macro",
    "diag_handoff_ready_rl_control_flag",
    "diag_macro_active_without_takeover_flag",
    "diag_shoot_vel_retrograde_x_flag",
    "diag_post_kick_inversion_flag",
)

# Tasas / contadores episódicos (tomar última fila del episodio en la traza)
_TRACE_DIAG_LAST_ROW_KEYS: tuple[str, ...] = (
    "episode_diag_control_owner_changes",
    "episode_diag_ball_teleport_frames",
    "episode_diag_agent_teleport_frames",
    "episode_diag_shoot_backward_frames",
    "episode_diag_theta_snap_prep_frames",
    "episode_diag_ball_overlap_frames",
    "episode_diag_ball_penetration_frames",
    "episode_diag_soft_ball_jump_frames",
    "episode_diag_motion_vs_body_mismatch_frames",
    "episode_diag_theta_snap_motion_prep_frames",
    "episode_diag_rapid_kick_proxy_frames",
    "episode_diag_post_kick_inversion_frames",
    "episode_diag_shoot_vel_retrograde_frames",
    "episode_diag_opp_pressure_frames",
    "episode_diag_ball_overlap_rate",
    "episode_diag_ball_penetration_rate",
    "episode_diag_soft_ball_jump_rate",
    "episode_diag_motion_vs_body_mismatch_rate",
    "episode_diag_theta_snap_motion_prep_rate",
    "episode_diag_rapid_kick_proxy_rate",
    "episode_diag_post_kick_inversion_rate",
    "episode_diag_shoot_vel_retrograde_rate",
    "episode_diag_opp_pressure_rate",
    "episode_diag_control_owner_change_rate",
)


def _trace_has_extended_diagnostics(rows: list[dict[str, str]]) -> bool:
    if not rows:
        return False
    keys = rows[0].keys()
    return "diag_control_transition" in keys and "diag_ball_agent_sep_mm" in keys


def _trace_has_reward_physics_columns(rows: list[dict[str, str]]) -> bool:
    if not rows:
        return False
    keys = rows[0].keys()
    return "rw_total" in keys and "diag_ball_collision_tags" in keys


def summarize_trace_reward_physics(rows: list[dict[str, str]]) -> None:
    """Reward por componente + resolución de colisión balón (nuevo env)."""
    if not _trace_has_reward_physics_columns(rows):
        return
    print("\n--- trace_reward_physics ---")
    n = len(rows)
    coll = Counter()
    for row in rows:
        tag = (row.get("diag_ball_collision_tags") or "none").strip()
        if tag and tag != "none":
            for part in tag.split("|"):
                coll[part] += 1
    if coll:
        print(f"ball_collision_tag_steps={sum(coll.values())} unique_tags={len(coll)}")
        print("  top=" + ", ".join(f"{k}:{v}" for k, v in coll.most_common(10)))
    else:
        print("ball_collision_tags=(none in trace sample)")

    zweik = sum(1 for row in rows if _to_int(row, "diag_activation_zweikampf_flag"))
    print(f"activation_zweikampf_steps={zweik} ({zweik / max(1, n):.4%} of rows)")

    rw_keys = (
        "rw_ready_but_walk_penalty",
        "rw_takeover_success",
        "rw_takeover_failure",
        "rw_handoff_alignment",
        "rw_approach_ball",
    )
    for key in rw_keys:
        vals = [abs(_to_float(row, key)) for row in rows]
        nz = sum(1 for v in vals if v > 1e-9)
        if nz:
            print(
                f"{key} non_zero={nz} mean_abs={sum(vals) / len(vals):.5f} "
                f"max_abs={max(vals):.5f}"
            )


def summarize_trace_extended_diagnostics(
    rows: list[dict[str, str]],
    *,
    top_episodes: int,
) -> None:
    """Resumen de la telemetría extendida (CSV post-instrumentación)."""
    if not _trace_has_extended_diagnostics(rows):
        print(
            "\ntrace_diagnostics=skipped "
            "(no hay columnas extendidas; re-ejecuta entrenamiento con el env actual)"
        )
        return

    print("\n--- trace_diagnostics (instrumentación extendida) ---")
    n = len(rows)
    print(f"diag_rows={n}")

    # Totales globales (conteo de steps con flag)
    print("global_flag_steps:")
    for key in _TRACE_DIAG_FLAG_SUM_KEYS:
        c = sum(1 for row in rows if _to_int(row, key) != 0)
        if c:
            print(f"  {key}={c} ({c / max(1, n):.4%} of rows)")

    # Distribución de transiciones de control (muestra)
    trans = Counter(
        (row.get("diag_control_transition") or "").strip()
        for row in rows
        if (row.get("diag_control_transition") or "").strip()
        and (row.get("diag_control_transition") or "")[:4] != "INIT"
    )
    if trans:
        print("control_transition_top=" + ", ".join(f"{k}:{v}" for k, v in trans.most_common(12)))

    # Agregados continuos
    seps = [_to_float(r, "diag_ball_agent_sep_mm") for r in rows if r.get("diag_ball_agent_sep_mm", "") != ""]
    if seps:
        ss = sorted(seps)
        p05_i = max(0, min(len(ss) - 1, int(0.05 * max(len(ss) - 1, 1))))
        med_i = len(ss) // 2
        print(
            "ball_agent_sep_mm="
            f"min={min(ss):.1f} p05={ss[p05_i]:.1f} "
            f"median={ss[med_i]:.1f} "
            f"mean={sum(ss) / len(ss):.1f}"
        )
    mvb = []
    for r in rows:
        raw = r.get("diag_motion_walk_theta_vs_body_deg", "")
        if raw not in ("", None):
            try:
                mvb.append(abs(float(raw)))
            except (TypeError, ValueError):
                pass
    if mvb:
        sm = sorted(mvb)
        p95_i = max(0, min(len(sm) - 1, int(0.95 * max(len(sm) - 1, 1))))
        print(
            "motion_walk_theta_vs_body_abs_deg="
            f"max={max(sm):.1f} p95={sm[p95_i]:.1f} "
            f"mean={sum(sm) / len(sm):.1f}"
        )
    post_dots = []
    for r in rows:
        raw = r.get("diag_post_kick_vel_align_dot", "")
        if raw not in ("", None):
            try:
                post_dots.append(float(raw))
            except (TypeError, ValueError):
                pass
    if post_dots:
        bad = sum(1 for d in post_dots if d < 0.2)
        print(
            f"post_kick_vel_align_dot n={len(post_dots)} "
            f"mean={sum(post_dots) / len(post_dots):.3f} "
            f"lt0.2={bad} ({bad / len(post_dots):.2%})"
        )

    # Por episodio
    by_ep: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_ep[str(row.get("episode", ""))].append(row)

    ep_scores: list[tuple[float, str, dict[str, float | int]]] = []
    for ep, ep_rows in by_ep.items():
        if not ep or ep_rows is None:
            continue
        ep_rows = sorted(ep_rows, key=lambda r: _to_int(r, "env_step"))
        last = ep_rows[-1]
        sums: dict[str, int] = {k: sum(_to_int(r, k) for r in ep_rows) for k in _TRACE_DIAG_FLAG_SUM_KEYS}
        last_snap = {k: _to_float(last, k) for k in _TRACE_DIAG_LAST_ROW_KEYS if k in last}
        steps = len(ep_rows)
        # Puntuación heurística para ordenar episodios “dolorosos”
        score = float(
            sums["diag_ball_agent_penetration_flag"] * 5.0
            + sums["diag_post_kick_inversion_flag"] * 4.0
            + sums["diag_rapid_kick_proxy_flag"] * 3.0
            + sums["diag_ball_teleport_flag"] * 3.0
            + sums["diag_control_owner_changed"] * 1.5
            + sums["diag_theta_snap_motion_prep_flag"] * 1.2
            + sums["diag_shoot_vel_retrograde_x_flag"] * 2.0
            + sums["diag_ball_agent_overlap_flag"] * 0.5
            + _to_float(last, "episode_diag_control_owner_change_rate") * 50.0
        )
        ep_scores.append(
            (
                score,
                ep,
                {
                    "steps": steps,
                    "sums": sums,
                    "last": last_snap,
                },
            )
        )

    ep_scores.sort(key=lambda t: -t[0])
    print(f"\nepisode_diag_ranking (top {top_episodes} by heuristic pain_score):")
    print(
        "  columns: ep | score | steps | ctlChg | penetr | overlap | rapidKick | "
        "postInv | retroX | motSnap | ctlChgRate(last)"
    )
    for score, ep, pack in ep_scores[: max(1, top_episodes)]:
        s: dict[str, int] = pack["sums"]
        lr: dict[str, float] = pack["last"]
        print(
            f"  ep{ep} score={score:.1f} steps={pack['steps']} "
            f"ctlChg={s['diag_control_owner_changed']} "
            f"penetr={s['diag_ball_agent_penetration_flag']} "
            f"overlap={s['diag_ball_agent_overlap_flag']} "
            f"rapidKick={s['diag_rapid_kick_proxy_flag']} "
            f"postInv={s['diag_post_kick_inversion_flag']} "
            f"retroX={s['diag_shoot_vel_retrograde_x_flag']} "
            f"motSnap={s['diag_theta_snap_motion_prep_flag']} "
            f"ctlChgRate={lr.get('episode_diag_control_owner_change_rate', 0.0):.4f}"
        )


def _best_window(rows: list[dict[str, str]], window: int, key: str) -> tuple[int, int, float]:
    if not rows:
        return 0, 0, 0.0
    if len(rows) <= window:
        return 1, len(rows), _avg(rows, key)
    best = (-1.0, 0, 0)
    for start in range(0, len(rows) - window + 1):
        chunk = rows[start:start + window]
        score = _avg(chunk, key)
        if score > best[0]:
            best = (score, start + 1, start + window)
    return best[1], best[2], best[0]


def summarize_metrics(run_dir: Path, window: int) -> None:
    metrics_path = run_dir / "metrics.csv"
    rows = _load_csv(metrics_path)
    print(f"run_dir={run_dir}")
    print(f"episodes={len(rows)}")
    if not rows:
        return

    for row in rows:
        if _to_float(row, "playball_entry_rate") > 0.0:
            print(
                "first_playball="
                f"ep{row['episode']} stage{row['stage_id']} "
                f"pb={_to_float(row, 'playball_entry_rate'):.4f} "
                f"tk={_to_float(row, 'takeover_success_rate'):.4f}"
            )
            break

    for row in rows:
        if _to_float(row, "takeover_success_rate") > 0.0:
            print(
                "first_tksuccess="
                f"ep{row['episode']} stage{row['stage_id']} "
                f"pb={_to_float(row, 'playball_entry_rate'):.4f} "
                f"tk={_to_float(row, 'takeover_success_rate'):.4f}"
            )
            break

    print("\nby_stage")
    stage_ids = sorted({row["stage_id"] for row in rows}, key=lambda s: int(s))
    for stage_id in stage_ids:
        chunk = [row for row in rows if row["stage_id"] == stage_id]
        stage_name = chunk[0].get("stage_name", "")
        print(
            f"  stage={stage_id} name={stage_name} n={len(chunk)} "
            f"pb={_avg(chunk, 'playball_entry_rate'):.4f} "
            f"tk={_avg(chunk, 'takeover_success_rate'):.4f} "
            f"rbw={_avg(chunk, 'ready_but_walk_rate'):.4f} "
            f"rtbw={_avg(chunk, 'ready_touch_but_walk_rate'):.4f} "
            f"to={_avg(chunk, 'ep_takeover_timeout_rate'):.4f} "
            f"clr={_avg(chunk, 'ep_takeover_skill_cleared_rate'):.4f}"
        )

    def print_window(label: str, chunk: list[dict[str, str]]) -> None:
        print(
            f"  {label}: "
            f"pb={_avg(chunk, 'playball_entry_rate'):.4f} "
            f"tk={_avg(chunk, 'takeover_success_rate'):.4f} "
            f"rbw={_avg(chunk, 'ready_but_walk_rate'):.4f} "
            f"rtbw={_avg(chunk, 'ready_touch_but_walk_rate'):.4f} "
            f"to={_avg(chunk, 'ep_takeover_timeout_rate'):.4f} "
            f"clr={_avg(chunk, 'ep_takeover_skill_cleared_rate'):.4f}"
        )

    print(f"\nwindows(window={window})")
    print_window("first", rows[:window])
    if len(rows) > window:
        mid_start = max(0, len(rows) // 2 - window // 2)
        print_window("middle", rows[mid_start:mid_start + window])
        print_window("last", rows[-window:])

    start_ep, end_ep, best_score = _best_window(rows, window, "takeover_success_rate")
    print(
        f"\nbest_tksuccess_window="
        f"ep{start_ep}-ep{end_ep} "
        f"tk={best_score:.4f}"
    )


def summarize_trace(run_dir: Path, bucket_mm: int, top_episodes: int) -> None:
    trace_path = run_dir / TRACE_FILE_NAME
    if not trace_path.exists():
        print(f"\ntrace={trace_path} (missing)")
        print("note=enable TRACE_TAKEOVER_STEPS=1 before the next run to capture per-step WalkTo/ShootTo")
        return

    rows = _load_csv(trace_path)
    if not rows:
        print(f"\ntrace={trace_path} (empty)")
        return

    play_rows = [
        row for row in rows
        if str(row.get("macro_type", "none")).startswith("playBall_")
        or row.get("takeover_active") == "1"
        or row.get("playball_entry") == "1"
    ]
    print(f"\ntrace={trace_path}")
    print(f"trace_rows={len(rows)}")
    print(f"playball_rows={len(play_rows)}")
    if not play_rows:
        return

    macro_counts = Counter(row.get("macro_type", "none") for row in play_rows)
    end_reason_counts = Counter(
        row.get("macro_end_reason", "none")
        for row in play_rows
        if row.get("macro_end_reason", "none") != "none"
    )
    walk_present = [
        row for row in play_rows
        if row.get("behavior_walk_x") not in ("", None)
        and row.get("behavior_walk_y") not in ("", None)
    ]
    shoot_present = [
        row for row in play_rows
        if row.get("shoot_target_x") not in ("", None)
        and row.get("shoot_target_y") not in ("", None)
    ]

    print(
        f"walkto_present={len(walk_present)}/{len(play_rows)} "
        f"({len(walk_present) / max(1, len(play_rows)):.2%})"
    )
    print(
        f"shootto_present={len(shoot_present)}/{len(play_rows)} "
        f"({len(shoot_present) / max(1, len(play_rows)):.2%})"
    )
    print("macro_freq=" + ", ".join(f"{k}:{v}" for k, v in macro_counts.most_common()))
    if end_reason_counts:
        print("end_reason_freq=" + ", ".join(f"{k}:{v}" for k, v in end_reason_counts.most_common()))

    walk_buckets = Counter(
        _bucket_xy(_to_float(row, "behavior_walk_x"), _to_float(row, "behavior_walk_y"), bucket_mm)
        for row in walk_present
    )
    shoot_buckets = Counter(
        _bucket_xy(_to_float(row, "shoot_target_x"), _to_float(row, "shoot_target_y"), bucket_mm)
        for row in shoot_present
    )
    exec_buckets = Counter(
        _bucket_xy(_to_float(row, "effective_target_x"), _to_float(row, "effective_target_y"), bucket_mm)
        for row in play_rows
        if row.get("effective_target_x") not in ("", None)
        and row.get("effective_target_y") not in ("", None)
    )
    if walk_buckets:
        print("top_walkto=" + ", ".join(f"{k}:{v}" for k, v in walk_buckets.most_common(8)))
    if shoot_buckets:
        print("top_shootto=" + ", ".join(f"{k}:{v}" for k, v in shoot_buckets.most_common(8)))
    if exec_buckets:
        print("top_execto=" + ", ".join(f"{k}:{v}" for k, v in exec_buckets.most_common(8)))

    gap_vals = []
    for row in walk_present:
        if row.get("ball_x") in ("", None) or row.get("ball_y") in ("", None):
            continue
        dx = _to_float(row, "behavior_walk_x") - _to_float(row, "ball_x")
        dy = _to_float(row, "behavior_walk_y") - _to_float(row, "ball_y")
        gap_vals.append((dx * dx + dy * dy) ** 0.5)
    if gap_vals:
        print(
            "walkto_ball_gap_mm="
            f"mean={sum(gap_vals) / len(gap_vals):.1f} "
            f"max={max(gap_vals):.1f}"
        )

    if any(row.get("agent_x") not in ("", None) for row in play_rows):
        near_goal_rows = [
            row for row in play_rows
            if max(_to_float(row, "agent_x"), _to_float(row, "ball_x")) >= FIELD.x_opponent_ground - 450.0
        ]
        mismatch_rows = [
            row for row in play_rows
            if row.get("macro_type") == "playBall_dribble"
            and row.get("sidecar_skill_request") in {"shoot", "pass"}
        ]
        effective_mismatch_rows = [
            row for row in play_rows
            if row.get("sidecar_macro_type_effective") == "playBall_dribble"
            and row.get("sidecar_skill_request") in {"shoot", "pass"}
        ]
        bearing_flips, heading_jumps = _count_oscillation_events(near_goal_rows)
        execto_ball_gaps = []
        for row in play_rows:
            if row.get("effective_target_x") in ("", None) or row.get("effective_target_y") in ("", None):
                continue
            if row.get("ball_x") in ("", None) or row.get("ball_y") in ("", None):
                continue
            dx = _to_float(row, "effective_target_x") - _to_float(row, "ball_x")
            dy = _to_float(row, "effective_target_y") - _to_float(row, "ball_y")
            execto_ball_gaps.append((dx * dx + dy * dy) ** 0.5)
        print(
            "kinematics="
            f"agent_speed_mean={_avg(play_rows, 'agent_speed'):.1f} "
            f"ball_speed_mean={_avg(play_rows, 'ball_speed'):.1f} "
            f"agent_ball_dist_mean={_avg(play_rows, 'agent_ball_distance_mm'):.1f} "
            f"agent_ball_bearing_abs_mean={_avg(play_rows, 'agent_ball_bearing_deg'):.1f}"
        )
        print(
            "near_goal="
            f"rows={len(near_goal_rows)} "
            f"bearing_flips={bearing_flips} "
            f"heading_jumps={heading_jumps}"
        )
        print(
            "macro_skill_mismatch="
            f"dribble_with_shoot_or_pass={len(mismatch_rows)} "
            f"effective_dribble_with_shoot_or_pass={len(effective_mismatch_rows)}"
        )
        if execto_ball_gaps:
            print(
                "execto_ball_gap_mm="
                f"mean={sum(execto_ball_gaps) / len(execto_ball_gaps):.1f} "
                f"max={max(execto_ball_gaps):.1f}"
            )

    summarize_trace_extended_diagnostics(rows, top_episodes=top_episodes)
    summarize_trace_reward_physics(rows)


def resolve_run_dir(path_str: str) -> Path:
    path = Path(path_str).resolve()
    if path.is_file():
        return path.parent
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize takeover metrics and optional trace for a training run.")
    parser.add_argument("run_path", help="Run directory or metrics.csv path")
    parser.add_argument("--window", type=int, default=50, help="Window size for first/middle/last summary")
    parser.add_argument("--bucket-mm", type=int, default=100, help="Bucket size for WalkTo/ShootTo target clustering")
    parser.add_argument(
        "--top-episodes",
        type=int,
        default=20,
        help="Cuántos episodios listar en el ranking de diagnósticos extendidos",
    )
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run_path)
    summarize_metrics(run_dir, max(1, args.window))
    summarize_trace(run_dir, max(1, args.bucket_mm), max(1, args.top_episodes))


if __name__ == "__main__":
    main()
