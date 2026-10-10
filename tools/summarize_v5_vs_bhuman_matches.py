"""Aggregate raw step logs from run_v5_vs_bhuman_match.py into a match-metrics table.

Usage:
  python tools/summarize_v5_vs_bhuman_matches.py runs/paper_match_v5_vs_bhuman_live/batch_<stamp>

Reads every match_*.json in the given directory (each written by
run_v5_vs_bhuman_match.py: one live match, v5 merged RL policy on players 2/3 vs real
B-Human). Computes, per match and aggregated:
  - score (goals_for/goals_against, ball-crossing-line proxy — no score field exists in
    the Python<->C++ bridge) and W/D/L result for v5.
  - possession share: which side controls the ball, from BALL_CONTROL_RADIUS_MM proximity
    on the ball_authority RL player's own (freshness-gated) ball estimate vs its
    nearest_opponent_dist. Not ground truth — same caveat as the rest of this harness.
  - recoveries: opponent-possession -> RL-possession transitions.
  - pass attempts/success: rising edges of skill=="pass" on the striker; "success" if
    within PASS_WINDOW_STEPS the ball is RL-possessed by the OTHER RL player without an
    intervening opponent-possession step.
  - stall flag: a match is flagged "stalled" if RL robot positions barely move (both RL
    players' position stays within STALL_RADIUS_MM of a fixed point) for the final
    STALL_WINDOW_S of the match — the "stand-collapse" failure mode documented in project
    history, not an infra bug. Stalled matches are reported separately, not silently
    averaged in with healthy ones.

Writes .docs/FIGURES/table_match_metrics_v5_vs_real_bhuman.md and .csv.

Goal counting does NOT use the goals_for/goals_against fields written live by
run_v5_vs_bhuman_match.py — those were found to be inflated by a confirmed bug: the
live detector confirms a goal off ball_authority's (whichever RL player is nearest its
OWN ball estimate) reading each step, and when authority switches to a DIFFERENT robot
mid-play, that robot's independent (and sometimes stale) estimate can jump thousands of
mm in one step, satisfying the "ball back near center" confirmation without the real
ball moving. See tools/recompute_goals_corrected.py for the concrete example that found
this. This script re-derives goals from the raw step log with the fix: a pending goal
only confirms if the SAME ball_authority agent that saw the crossing also reports the
return to center.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

BALL_CONTROL_RADIUS_MM = 350.0
PASS_WINDOW_STEPS = 40  # ~5s at 8Hz
STALL_WINDOW_S = 90.0
STALL_RADIUS_MM = 250.0

GOAL_X = 4500.0
OWN_GOAL_X_MM = -4500.0
OWN_GOAL_POST_Y_MM = 800.0
GOAL_LINE_MARGIN_MM = 100.0
GOAL_COOLDOWN_S = 3.0
KICKOFF_RETURN_RADIUS_MM = 700.0


def corrected_goal_count(steps: list[dict]) -> tuple[int, int]:
    """Re-derive goals_for/goals_against with the authority-continuity fix. See module
    docstring; mirrors tools/recompute_goals_corrected.py's recount()."""
    pending_side = None
    pending_since = None
    pending_authority = None
    cooldown_until = 0.0
    scored_for = scored_against = 0
    for s in steps:
        t = s["t"]
        bx, by = s["ball_x"], s["ball_y"]
        auth = s.get("ball_authority")
        if not s.get("ball_fresh", True) or t < cooldown_until:
            continue
        if pending_side is None:
            if bx >= GOAL_X - GOAL_LINE_MARGIN_MM and abs(by) <= OWN_GOAL_POST_Y_MM:
                pending_side, pending_since, pending_authority = "for", t, auth
            elif bx <= OWN_GOAL_X_MM + GOAL_LINE_MARGIN_MM and abs(by) <= OWN_GOAL_POST_Y_MM:
                pending_side, pending_since, pending_authority = "against", t, auth
            continue
        if auth != pending_authority:
            pending_side, pending_since, pending_authority = None, None, None
            continue
        if math.hypot(bx, by) <= KICKOFF_RETURN_RADIUS_MM:
            if pending_side == "for":
                scored_for += 1
            else:
                scored_against += 1
            cooldown_until = t + GOAL_COOLDOWN_S
            pending_side, pending_since, pending_authority = None, None, None
        elif pending_since is not None and (t - pending_since) > 15.0:
            pending_side, pending_since, pending_authority = None, None, None
    return scored_for, scored_against


def possession_side(agents: dict, ball_fresh: bool) -> str:
    """'rl' | 'opp' | 'contested' for one step, or 'unknown' if the ball isn't fresh."""
    if not ball_fresh:
        return "unknown"
    rl_dists = [a["ball_dist"] for a in agents.values()]
    opp_dists = [a["nearest_opponent_dist"] for a in agents.values()]
    rl_close = min(rl_dists) < BALL_CONTROL_RADIUS_MM if rl_dists else False
    opp_close = min(opp_dists) < BALL_CONTROL_RADIUS_MM if opp_dists else False
    if rl_close and not opp_close:
        return "rl"
    if opp_close and not rl_close:
        return "opp"
    if rl_close and opp_close:
        return "contested"
    return "loose"


def true_possession_side(step: dict) -> str:
    """Possession from ABSOLUTE field positions: whichever side's nearest player is
    closest to the ball, requiring a control-radius contact and a clear margin.

    Supersedes `possession_side`, which compared our robots' distance-to-ball against
    each robot's own egocentric `nearest_opponent_dist` -- i.e. "is an opponent near
    ME", not "does the opponent have the ball" -- and which sat at the not-seen
    sentinel a third to half of the time. Returns "unknown" for logs written before the
    2026-08-11 GlobalOpponentsModel export existed, so old batches degrade rather than
    silently reporting a wrong number.
    """
    if "opp_nearest_ball_dist" not in step:
        return "unknown"
    if not step.get("ball_fresh", False):
        return "unknown"
    opp_d = step.get("opp_nearest_ball_dist")
    rl_d = step.get("rl_nearest_ball_dist")
    if opp_d is None or rl_d is None:
        return "unknown"
    rl_close = rl_d < BALL_CONTROL_RADIUS_MM
    opp_close = opp_d < BALL_CONTROL_RADIUS_MM
    if rl_close and not opp_close:
        return "rl"
    if opp_close and not rl_close:
        return "opp"
    if rl_close and opp_close:
        return "contested"
    return "loose"


def analyze_match(path: Path) -> dict:
    d = json.loads(path.read_text())
    steps = d["steps"]
    active_players = d["active_players"]
    # Ground-truth GameController score when present (2026-08-11 onward); older logs
    # fall back to the ball-crossing proxy recount.
    has_gc = bool(steps) and "gc_goals_for" in steps[0]
    has_true_poss = bool(steps) and "opp_nearest_ball_dist" in steps[0]

    poss_counts = {"rl": 0, "opp": 0, "contested": 0, "loose": 0, "unknown": 0}
    recoveries = 0
    opp_recoveries = 0
    prev_side: str | None = None

    pass_attempts = 0
    pass_success = 0
    pending_pass: dict | None = None  # {"t_step": i, "passer": aid}
    prev_skill: dict[str, str | None] = {str(p): None for p in active_players}

    skill_counts: dict[str, dict[str, int]] = {str(p): {} for p in active_players}

    for i, s in enumerate(steps):
        agents = s["agents"]
        side = true_possession_side(s) if has_true_poss else possession_side(agents, s.get("ball_fresh", False))
        poss_counts[side] += 1
        if prev_side == "opp" and side == "rl":
            recoveries += 1
        elif prev_side == "rl" and side == "opp":
            opp_recoveries += 1
        if side in ("rl", "opp"):
            prev_side = side

        for aid_str, a in agents.items():
            skill_counts.setdefault(aid_str, {})
            skill_counts[aid_str][a["skill"]] = skill_counts[aid_str].get(a["skill"], 0) + 1
            if a["role"] == "striker" and a["skill"] == "pass" and prev_skill.get(aid_str) != "pass":
                pass_attempts += 1
                pending_pass = {"i": i, "passer": aid_str}
            prev_skill[aid_str] = a["skill"]

        if pending_pass is not None:
            if i - pending_pass["i"] > PASS_WINDOW_STEPS:
                pending_pass = None
            elif side == "opp":
                pending_pass = None  # intercepted / lost — not a success
            elif side == "rl":
                other_holds = any(
                    aid_str != pending_pass["passer"] and a["ball_dist"] < BALL_CONTROL_RADIUS_MM
                    for aid_str, a in agents.items()
                )
                if other_holds:
                    pass_success += 1
                    pending_pass = None

    total = max(1, len(steps))
    duration_s = steps[-1]["t"] if steps else 0.0

    # Stall detection: RL robot positions barely move in the final STALL_WINDOW_S.
    tail = [s for s in steps if duration_s - s["t"] <= STALL_WINDOW_S]
    stalled = False
    if len(tail) >= 10:
        per_agent_spread = []
        for p in active_players:
            aid_str = str(p)
            xs = [s["agents"][aid_str]["robot_x"] for s in tail if aid_str in s["agents"]]
            ys = [s["agents"][aid_str]["robot_y"] for s in tail if aid_str in s["agents"]]
            if xs and ys:
                spread = max(max(xs) - min(xs), max(ys) - min(ys))
                per_agent_spread.append(spread)
        stalled = bool(per_agent_spread) and max(per_agent_spread) < STALL_RADIUS_MM

    if has_gc:
        goals_for = int(steps[-1]["gc_goals_for"])
        goals_against = int(steps[-1]["gc_goals_against"])
    else:
        goals_for, goals_against = corrected_goal_count(steps)
    result = "W" if goals_for > goals_against else ("L" if goals_for < goals_against else "D")

    return {
        "match": path.stem,
        "duration_s": duration_s,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "result": result,
        "stalled": stalled,
        "possession_rl_pct": 100.0 * poss_counts["rl"] / total,
        "possession_opp_pct": 100.0 * poss_counts["opp"] / total,
        "possession_contested_pct": 100.0 * poss_counts["contested"] / total,
        "possession_loose_pct": 100.0 * poss_counts["loose"] / total,
        "possession_unknown_pct": 100.0 * poss_counts["unknown"] / total,
        "recoveries": recoveries,
        "opp_recoveries": opp_recoveries,
        "pass_attempts": pass_attempts,
        "pass_success": pass_success,
        "pass_success_rate": (100.0 * pass_success / pass_attempts) if pass_attempts else 0.0,
        "skill_counts": skill_counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_dir", type=Path)
    parser.add_argument("--tag", type=str, default="", help="Output filename suffix, e.g. '3v3' or '4v4'.")
    args = parser.parse_args()
    tag_suffix = f"_{args.tag}" if args.tag else ""

    match_files = sorted(args.batch_dir.glob("match_*.json"))
    if not match_files:
        raise SystemExit(f"no match_*.json files found under {args.batch_dir}")

    per_match = [analyze_match(p) for p in match_files]

    healthy = [m for m in per_match if not m["stalled"]]
    stalled = [m for m in per_match if m["stalled"]]

    def avg(key: str, rows: list[dict]) -> float:
        vals = [r[key] for r in rows]
        return statistics.mean(vals) if vals else 0.0

    wins = sum(1 for m in per_match if m["result"] == "W")
    draws = sum(1 for m in per_match if m["result"] == "D")
    losses = sum(1 for m in per_match if m["result"] == "L")
    total_gf = sum(m["goals_for"] for m in per_match)
    total_ga = sum(m["goals_against"] for m in per_match)

    print(f"[summary] {len(per_match)} matches | record (v5) W-D-L = {wins}-{draws}-{losses} "
          f"| goals {total_gf}-{total_ga} | stalled matches: {len(stalled)}/{len(per_match)}")

    fig_dir = Path("/home/luis/GitHub/RL/.docs/FIGURES")
    fig_dir.mkdir(parents=True, exist_ok=True)

    sample_match = json.loads(match_files[0].read_text())
    scene_name = Path(sample_match.get("scene", "?")).stem
    n_field = len(sample_match.get("active_players", []))

    md_lines = [
        "# v5 merged RL vs real B-Human — live match results",
        "",
        f"{len(per_match)} live matches, `{scene_name}` scene (1 B-Human GK + {n_field} RL field "
        "vs real B-Human opponents, real B-Human team behavior — not scripted dummies). "
        "v5 = `runs/merge_v5_attack_defense/checkpoint_stage8_step10000.pt`. "
        "All metrics below (possession/recoveries/pass) are proxies computed from ball/robot "
        "telemetry — there is no score or possession field in the Python<->C++ bridge; goals "
        "are a ball-crossing-goal-line proxy confirmed by the post-goal kickoff-return, and "
        "additionally require the confirming reading to come from the same ball_authority "
        "robot that saw the crossing (an earlier version without this check double-counted "
        "goals off ball_authority handoffs between robots' independent ball estimates).",
        "",
        f"**Record (v5 vs B-Human): {wins}W-{draws}D-{losses}L, goals {total_gf}-{total_ga}.**",
        f"**{len(stalled)}/{len(per_match)} matches flagged STALLED** (RL robots barely moved "
        f"in the final {STALL_WINDOW_S:.0f}s) — a real, occasionally-occurring policy "
        "behavior (\"stand-collapse\": masked skill logits leave stand/walk near-tied), not "
        "an infrastructure fault; see notes below.",
        "",
        "## Per-match results",
        "",
        "| Match | Result | Score (v5-BH) | Duration (s) | Stalled | Possession RL/Opp/Contested/Loose/Unk (%) | Recoveries RL/BH | Pass att/succ |",
        "|---|:-:|:-:|---:|:-:|---|---:|---:|",
    ]
    for m in per_match:
        md_lines.append(
            f"| {m['match']} | {m['result']} | {m['goals_for']}-{m['goals_against']} | "
            f"{m['duration_s']:.0f} | {'YES' if m['stalled'] else 'no'} | "
            f"{m['possession_rl_pct']:.1f}/{m['possession_opp_pct']:.1f}/"
            f"{m['possession_contested_pct']:.1f}/{m['possession_loose_pct']:.1f}/"
            f"{m['possession_unknown_pct']:.1f} | "
            f"{m['recoveries']}/{m['opp_recoveries']} | {m['pass_attempts']}/{m['pass_success']} |"
        )

    md_lines += [
        "",
        "## Aggregate (healthy matches only, excludes stalled)",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Matches | {len(healthy)} / {len(per_match)} |",
        f"| Possession RL (%) | {avg('possession_rl_pct', healthy):.1f} |",
        f"| Possession B-Human (%) | {avg('possession_opp_pct', healthy):.1f} |",
        f"| Possession loose/unclear (%) | {avg('possession_loose_pct', healthy) + avg('possession_unknown_pct', healthy):.1f} |",
        f"| Recoveries RL / match | {avg('recoveries', healthy):.2f} |",
        f"| Recoveries B-Human / match | {avg('opp_recoveries', healthy):.2f} |",
        f"| Pass attempts / match | {avg('pass_attempts', healthy):.2f} |",
        f"| Pass success rate (%) | {avg('pass_success_rate', healthy):.1f} |",
        "",
    ]

    md_path = fig_dir / f"table_match_metrics_v5_vs_real_bhuman{tag_suffix}.md"
    md_path.write_text("\n".join(md_lines) + "\n")

    csv_path = fig_dir / f"table_match_metrics_v5_vs_real_bhuman{tag_suffix}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[k for k in per_match[0].keys() if k != "skill_counts"])
        writer.writeheader()
        for m in per_match:
            writer.writerow({k: v for k, v in m.items() if k != "skill_counts"})

    print(f"wrote {md_path}")
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
