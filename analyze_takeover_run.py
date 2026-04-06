from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Iterable

from takeover_trace import TRACE_FILE_NAME


def _to_float(row: dict[str, str], key: str) -> float:
    raw = row.get(key, "")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


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


def summarize_trace(run_dir: Path, bucket_mm: int) -> None:
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
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run_path)
    summarize_metrics(run_dir, max(1, args.window))
    summarize_trace(run_dir, max(1, args.bucket_mm))


if __name__ == "__main__":
    main()
