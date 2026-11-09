#!/usr/bin/env python3
"""In-sim diagnostic: is the off_ball_support TRIANGLE target None / unreachable / just-not-reached?

Two reward interventions (v4.3 gate-off+bonus, v4.4 dense nav shaping) left support_triangle_hold_rate
at EXACTLY 0.000 while open-hold worked. The hold reward, the shaping, AND the metric all sit behind
`if target is not None`. This probe hooks TeamSlotSimRobotEnv._refresh_team_state and records, per step:
  - is the off_ball_support role assigned this step?
  - are all three roles (striker/open/off_ball) simultaneously present? (triangle_ema only updates then)
  - is team_state.triangle_ema None?
  - off_ball distance to triangle_ema (mm) when both exist
  - same for open_support vs open_lane_ema (control: open-hold works, so this should look healthy)
Then prints the distributions so we can classify: (1) target None, (2) set-but-unreachable, (3) set-but-
not-reached. Minimal-train (256 steps) + BC off so it's ~pure v4.4 behaviour over a 12-ep eval.
"""
from __future__ import annotations
import atexit, json, math, os, sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "runs/offline_team_eval"
OUT.mkdir(parents=True, exist_ok=True)
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
recs: list[dict] = []


def _dump() -> None:
    path = OUT / f"triangle_diag_{STAMP}.json"
    if not recs:
        path.write_text(json.dumps({"steps": 0}))
        print(f"[tri-diag] NO steps captured -> {path}", flush=True)
        return
    n = len(recs)
    off_assigned = [r for r in recs if r["off_present"]]
    open_assigned = [r for r in recs if r["open_present"]]
    tri_none = [r for r in off_assigned if r["tri_none"]]
    tri_val = [r for r in off_assigned if not r["tri_none"] and r["off_dist"] is not None]
    open_val = [r for r in open_assigned if not r["open_none"] and r["open_dist"] is not None]

    def pct(a, b):
        return 100.0 * len(a) / max(1, len(b))

    def quant(vals, q):
        if not vals:
            return float("nan")
        s = sorted(vals)
        return s[min(len(s) - 1, int(q * len(s)))]

    off_d = [r["off_dist"] for r in tri_val]
    open_d = [r["open_dist"] for r in open_val]
    summary = {
        "steps": n,
        "off_role_assigned_pct": pct(off_assigned, recs),
        "all_three_roles_pct": pct([r for r in recs if r["all_three"]], recs),
        "off_steps_triangle_None_pct": pct(tri_none, off_assigned),
        "off_dist_median": quant(off_d, 0.5),
        "off_dist_p10": quant(off_d, 0.10),
        "off_dist_p90": quant(off_d, 0.90),
        "off_within_600_pct": pct([d for d in off_d if d < 600], off_d) if off_d else float("nan"),
        "off_within_900_pct": pct([d for d in off_d if d < 900], off_d) if off_d else float("nan"),
        "CONTROL_open_role_assigned_pct": pct(open_assigned, recs),
        "CONTROL_open_None_pct": pct([r for r in open_assigned if r["open_none"]], open_assigned),
        "CONTROL_open_dist_median": quant(open_d, 0.5),
        "CONTROL_open_within_600_pct": pct([d for d in open_d if d < 600], open_d) if open_d else float("nan"),
        "phase_attack_pct": pct([r for r in recs if r["phase"] != "defense"], recs),
    }
    path.write_text(json.dumps(summary, indent=2))
    print("\n[tri-diag] ==== TRIANGLE TARGET DIAGNOSIS ====", flush=True)
    for k, v in summary.items():
        print(f"  {k:34s} {v:.1f}" if isinstance(v, float) else f"  {k:34s} {v}", flush=True)
    print(f"[tri-diag] -> {path}", flush=True)


atexit.register(_dump)


def _install() -> None:
    import team_parallel_environment as tpe

    orig = tpe.TeamSlotSimRobotEnv._refresh_team_state

    def patched(self):
        orig(self)
        try:
            roles = dict(self._roles)
            ts = self._team_state
            raw = dict(self._raw_obs)
            off_id = next((pn for pn, r in roles.items() if r == "off_ball_support"), None)
            open_id = next((pn for pn, r in roles.items() if r == "open_support"), None)
            strk_id = next((pn for pn, r in roles.items() if r == "striker"), None)
            tri = ts.triangle_ema
            opn = ts.open_lane_ema
            off_dist = None
            if off_id is not None and tri is not None and off_id in raw:
                o = raw[off_id]
                off_dist = math.hypot(float(o.robot_x) - tri[0], float(o.robot_y) - tri[1])
            open_dist = None
            if open_id is not None and opn is not None and open_id in raw:
                o = raw[open_id]
                open_dist = math.hypot(float(o.robot_x) - opn[0], float(o.robot_y) - opn[1])
            recs.append({
                "off_present": off_id is not None,
                "open_present": open_id is not None,
                "all_three": None not in (off_id, open_id, strk_id),
                "tri_none": tri is None,
                "open_none": opn is None,
                "off_dist": off_dist,
                "open_dist": open_dist,
                "phase": getattr(ts, "phase", "?"),
            })
        except Exception as e:  # never break the run
            recs.append({"off_present": False, "open_present": False, "all_three": False,
                         "tri_none": True, "open_none": True, "off_dist": None, "open_dist": None,
                         "phase": "err:%s" % type(e).__name__})

    tpe.TeamSlotSimRobotEnv._refresh_team_state = patched
    print("[tri-diag] _refresh_team_state accumulator installed", flush=True)


def main() -> None:
    os.environ.setdefault("RL_GATE_SHOOT_OPEN_ENTER", "0.30")
    os.environ.setdefault("RL_GATE_SHOOT_OPEN_EXIT", "0.20")
    os.environ.setdefault("RL_VISIBLE_RESET_MAX_ATTEMPTS", "8")
    os.environ.setdefault("RL_MA_PROD_TRAIN_STEPS", "256")
    os.environ.setdefault("RL_MA_PROD_EVAL_INTERVAL", "256")
    os.environ.setdefault("RL_MA_PROD_EVAL_EPISODES", "12")
    os.environ.setdefault("RL_MA_PROD_MAX_EPISODE_STEPS", "800")
    os.environ.setdefault("RL_MA_PROD_BC_BETA_MAX", "0")
    os.environ.setdefault("RL_MA_PROD_BC_BETA_MIN", "0")
    os.environ.setdefault("RL_MA_PROD_CHECKPOINT_INTERVAL", "100000")
    os.environ.setdefault("RL_MA_PROD_EXPERIMENT_NAME", "triangle_diag")
    # Triangle-forcing pool (same as v4.3/v4.4) so off_ball gets real triangles.
    os.environ.setdefault("RL_PRODUCTION_SCENARIO_SET", "team")
    os.environ.setdefault(
        "RL_PRODUCTION_SCENARIO_WEIGHTS",
        "support_striker_live:4,team_3v3_full_dynamic:3,support_striker_far:1,"
        "team_2v0_clean_band:0,team_2v0_static_blocker:0,team_3v0_clean:0,team_3v0_static_pinch:0,"
        "team_3v0_dynamic:0,giveandgo_outlet_present:0,giveandgo_outlet_absent:0,pass_forcing:0",
    )
    _install()
    import train_multiagent_production as tmp
    sys.argv = [
        "triangle_target_diag", "--stage-id", "7",
        "--checkpoint", str(REPO / "runs/multiagent_v4_4/checkpoint_stage7_step12000.pt"),
        "--dataset", str(REPO / "runs/teacher_multiagent_dataset/dataset_v4_backbone_plus_overlay/samples.npz"),
    ]
    tmp.run(tmp.parse_args())


if __name__ == "__main__":
    main()
