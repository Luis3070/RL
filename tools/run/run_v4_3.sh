#!/usr/bin/env bash
# v4.3 = v4.2 + TRIANGLE-HOLD (off-ball support positioning). Path-2 mini-iteration before stage 3.
# ROOT CAUSE of support_triangle_hold_rate ~0.003 (persisted across v3/v3.1/v4/v4.2): NOT a data gap
# (corpus off_ball_support = stand 0.55) and NOT a metric artifact (open & triangle use the same 600mm
# detector; open reaches 0.16-0.5). The triangle HOLD REWARD almost never fired because it was gated on
# the off-ball player being a LIVE PASS OPTION (support_is_live_pass_option) — rarely true for the
# deep/third player — so PPO got no signal to navigate to + hold the triangle. That contingent gate
# (RL_MA_CONTINGENT_HOLD=1) was ON in EVERY prior run.
# FIX (env-only, warm-start v4.2 so pass 99.91% + 5/5 shot-discipline are PRESERVED via weights+KL anchor):
#   (1) RL_MA_CONTINGENT_HOLD=0  -> triangle hold reward is purely geometric (reward holding the triangle
#       whenever in-radius + standing, not only when a live outlet). THE key untried lever.
#   (2) raise the triangle (0.06->0.20) and open (0.07->0.10) hold bonuses so PPO has a strong gradient.
#   (3) triangle-FORCING scenario mix (support_striker_live heaviest -> off_ball gets a real triangle),
#       + a little support_striker_far for hold-vs-reposition discrimination, + pass/3v3/preshoot to
#       hold the line on pass + shot discipline.
# Metric stays honest: support_*_hold_rate detector is hard-coded 600mm; we did NOT widen it.
set -u
REPO=/home/luis/GitHub/RL
export PYTHONPATH="$REPO:/home/luis/GitHub/SabanaHerons2026/Make/Python"
export DISPLAY="${DISPLAY:-:1}"
export RL_GATE_SHOOT_OPEN_ENTER=0.30 RL_GATE_SHOOT_OPEN_EXIT=0.20
export RL_MA_TEACHER_VISIBLE_SCENE_NAME=RL1v0_3D_visible
export RL_VISIBLE_RESET_MAX_ATTEMPTS=8
export RL_MA_SHOOT_THRESHOLD=0.30
# --- triangle-hold reward fix (env-only) ---
export RL_MA_CONTINGENT_HOLD=0
export RL_MA_SUPPORT_TRIANGLE_HOLD_BONUS=0.20
export RL_MA_SUPPORT_OPEN_HOLD_BONUS=0.10
PY="${PY:-$HOME/anaconda3/envs/RLNAO/bin/python}"
TM=runs/teacher_multiagent_dataset
FINAL="$TM/dataset_v4_backbone_plus_overlay"
cd "$REPO"

# Triangle-forcing pool: support_striker_live 31% (off_ball holds triangle), 3v3 23% (triangle geometry
# under pressure), pass_forcing 15% + giveandgo 8% (keep pass), preshoot 15% (keep shot discipline),
# support_striker_far 8% (hold-vs-reposition discrimination). All other team/eval scenarios zeroed.
export RL_PRODUCTION_SCENARIO_SET="team,eval"
export RL_PRODUCTION_SCENARIO_WEIGHTS="support_striker_live:4,team_3v3_full_dynamic:3,pass_forcing:2,preshoot_bad_angle_boundary:2,giveandgo_outlet_present:1,support_striker_far:1,team_2v0_clean_band:0,team_2v0_static_blocker:0,team_3v0_clean:0,team_3v0_static_pinch:0,team_3v0_dynamic:0,giveandgo_outlet_absent:0,finish_band_2p5_3m_clean:0,finish_band_2p5_3m_static_lane:0,dynamic_full_field_pressure:0,blocked_corridor_repair:0"

RL_MA_PROD_TRAIN_STEPS=12000 RL_MA_PROD_MAX_EPISODE_STEPS=800 \
RL_MA_PROD_EVAL_INTERVAL=3000 RL_MA_PROD_CHECKPOINT_INTERVAL=3000 \
RL_MA_PROD_EXPERIMENT_NAME=multiagent_v4_3 \
RL_MA_PROD_BC_BETA_MAX=0.30 RL_MA_PROD_BC_BETA_MIN=0.05 \
RL_MA_PROD_BC_HALF_LIFE_EVALS=3 RL_MA_PROD_BC_REPLAY_UPDATES=4 \
RL_MA_BC_DECISION_WINDOW_ONLY=1 RL_MA_BC_STRIKER_PASS_FRAC=0.30 \
"$PY" -m rlfootball.training.train_multiagent_production --stage-id 7 \
  --checkpoint runs/multiagent_v4_2/checkpoint_stage7_step12000.pt \
  --dataset "$FINAL/samples.npz"
