#!/usr/bin/env bash
# Phase-A verification: confirm the NEW pass-completion metric (pass_attempts /
# pass_completions / pass_completion_rate) populates eval_summary.csv with sane values.
# Pass-heavy pool so attempts>0; pure-policy eval on v4.2@12000 (BC off). Coordinator off
# (irrelevant to the metric). 6 episodes to keep it fast — this is a plumbing check, not a verdict.
set -u
REPO=/home/luis/GitHub/RL
export PYTHONPATH="$REPO:/home/luis/GitHub/SabanaHerons2026/Make/Python"
export DISPLAY="${DISPLAY:-:1}"
export RL_GATE_SHOOT_OPEN_ENTER=0.30 RL_GATE_SHOOT_OPEN_EXIT=0.20
export RL_MA_TEACHER_VISIBLE_SCENE_NAME=RL1v0_3D_visible
export RL_VISIBLE_RESET_MAX_ATTEMPTS=8
export RL_MA_SHOOT_THRESHOLD=0.30
unset RL_MA_OFFBALL_COORD 2>/dev/null || true
PY="${PY:-$HOME/anaconda3/envs/RLNAO/bin/python}"
TM=runs/teacher_multiagent_dataset
FINAL="$TM/dataset_v4_backbone_plus_overlay"
cd "$REPO"

# Pass-generating pool: pass_forcing + giveandgo + support_striker_live dominate.
export RL_PRODUCTION_SCENARIO_SET="team,eval"
export RL_PRODUCTION_SCENARIO_WEIGHTS="pass_forcing:4,giveandgo_outlet_present:3,support_striker_live:2,team_3v3_full_dynamic:1,giveandgo_outlet_absent:0,support_striker_far:0,team_2v0_clean_band:0,team_2v0_static_blocker:0,team_3v0_clean:0,team_3v0_static_pinch:0,team_3v0_dynamic:0,finish_band_2p5_3m_clean:0,finish_band_2p5_3m_static_lane:0,dynamic_full_field_pressure:0,blocked_corridor_repair:0,preshoot_bad_angle_boundary:0"

RL_MA_PROD_TRAIN_STEPS=256 RL_MA_PROD_MAX_EPISODE_STEPS=800 \
RL_MA_PROD_EVAL_INTERVAL=256 RL_MA_PROD_CHECKPOINT_INTERVAL=100000 \
RL_MA_PROD_EVAL_EPISODES=6 \
RL_MA_PROD_EXPERIMENT_NAME=pass_metric_verify \
RL_MA_PROD_BC_BETA_MAX=0.0 RL_MA_PROD_BC_BETA_MIN=0.0 \
"$PY" train_multiagent_production.py --stage-id 7 \
  --checkpoint runs/multiagent_v4_2/checkpoint_stage7_step12000.pt \
  --dataset "$FINAL/samples.npz"
