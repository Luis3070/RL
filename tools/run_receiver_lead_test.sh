#!/usr/bin/env bash
# Stage-3 item-2 test: RECEIVER-LEAD inference override (RL_MA_RECEIVER_LEAD=1), isolated
# (coordinator OFF). When a pass is in flight, the support nearest the ball's predicted end
# leads toward that arrival point instead of holding its static slot. Layered on v4.2@12000,
# NO retrain. Baseline to beat (coordinator/lead OFF): pass_completion_rate 0.000, min_dist 1238mm.
# Pass-heavy pool, pure-policy (BC off), 8 episodes.
set -u
REPO=/home/luis/GitHub/RL
export PYTHONPATH="$REPO:/home/luis/GitHub/SabanaHerons2026/Make/Python"
export DISPLAY="${DISPLAY:-:1}"
export RL_GATE_SHOOT_OPEN_ENTER=0.30 RL_GATE_SHOOT_OPEN_EXIT=0.20
export RL_MA_TEACHER_VISIBLE_SCENE_NAME=RL1v0_3D_visible
export RL_VISIBLE_RESET_MAX_ATTEMPTS=8
export RL_MA_SHOOT_THRESHOLD=0.30
export RL_MA_RECEIVER_LEAD=1
unset RL_MA_OFFBALL_COORD 2>/dev/null || true
PY="${PY:-$HOME/anaconda3/envs/RLNAO/bin/python}"
TM=runs/teacher_multiagent_dataset
FINAL="$TM/dataset_v4_backbone_plus_overlay"
cd "$REPO"

export RL_PRODUCTION_SCENARIO_SET="team,eval"
export RL_PRODUCTION_SCENARIO_WEIGHTS="pass_forcing:4,giveandgo_outlet_present:3,support_striker_live:2,team_3v3_full_dynamic:1,giveandgo_outlet_absent:0,support_striker_far:0,team_2v0_clean_band:0,team_2v0_static_blocker:0,team_3v0_clean:0,team_3v0_static_pinch:0,team_3v0_dynamic:0,finish_band_2p5_3m_clean:0,finish_band_2p5_3m_static_lane:0,dynamic_full_field_pressure:0,blocked_corridor_repair:0,preshoot_bad_angle_boundary:0"

RL_MA_PROD_TRAIN_STEPS=256 RL_MA_PROD_MAX_EPISODE_STEPS=800 \
RL_MA_PROD_EVAL_INTERVAL=256 RL_MA_PROD_CHECKPOINT_INTERVAL=100000 \
RL_MA_PROD_EVAL_EPISODES=8 \
RL_MA_PROD_EXPERIMENT_NAME=receiver_lead_test \
RL_MA_PROD_BC_BETA_MAX=0.0 RL_MA_PROD_BC_BETA_MIN=0.0 \
"$PY" train_multiagent_production.py --stage-id 7 \
  --checkpoint runs/multiagent_v4_2/checkpoint_stage7_step12000.pt \
  --dataset "$FINAL/samples.npz"
