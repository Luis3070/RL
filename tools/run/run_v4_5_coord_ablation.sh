#!/usr/bin/env bash
# v4.5 ABLATION: repair-fix ALONE (off-ball coordinator OFF).
# The broad coop eval (runs/v4_2_coord_broad, triangle_hold 0.464) had BOTH the always-on role-aware
# _repair_execution_targets fix AND RL_MA_OFFBALL_COORD=1. The very first coordinator test (0.001) was
# taken BEFORE the repair-fix was found, so it does NOT isolate the coordinator. This run flips ONLY the
# coordinator OFF (default) while keeping the always-on repair-fix, on the SAME v4.2@12000, SAME broad
# coop scenario pool, SAME minimal-train pure-policy eval (256 steps, BC beta 0, 16 eps). It answers:
# is the coordinator load-bearing, or does the repair-fix alone let the learned policy reach the slot?
# -> decides whether the coordinator must be ported to the C++ bridge for embedded ONNX deploy.
set -u
REPO=/home/luis/GitHub/RL
export PYTHONPATH="$REPO:/home/luis/GitHub/SabanaHerons2026/Make/Python"
export DISPLAY="${DISPLAY:-:1}"
export RL_GATE_SHOOT_OPEN_ENTER=0.30 RL_GATE_SHOOT_OPEN_EXIT=0.20
export RL_MA_TEACHER_VISIBLE_SCENE_NAME=RL1v0_3D_visible
export RL_VISIBLE_RESET_MAX_ATTEMPTS=8
export RL_MA_SHOOT_THRESHOLD=0.30
# COORDINATOR OFF (default). Repair-fix is always-on code (no flag). Reward knobs at v4.2 defaults
# (NOT v4.3/v4.4 knobs) to match the coordinator-ON broad eval exactly.
unset RL_MA_OFFBALL_COORD 2>/dev/null || true
PY="${PY:-$HOME/anaconda3/envs/RLNAO/bin/python}"
TM=runs/teacher_multiagent_dataset
FINAL="$TM/dataset_v4_backbone_plus_overlay"
cd "$REPO"

# SAME broad coop pool as runs/v4_2_coord_broad (incl support_far + giveandgo_absent contrasts).
export RL_PRODUCTION_SCENARIO_SET="team,eval"
export RL_PRODUCTION_SCENARIO_WEIGHTS="pass_forcing:3,giveandgo_outlet_present:2,support_striker_live:2,support_striker_far:2,team_3v3_full_dynamic:2,giveandgo_outlet_absent:1,team_2v0_clean_band:0,team_2v0_static_blocker:0,team_3v0_clean:0,team_3v0_static_pinch:0,team_3v0_dynamic:0,finish_band_2p5_3m_clean:0,finish_band_2p5_3m_static_lane:0,dynamic_full_field_pressure:0,blocked_corridor_repair:0,preshoot_bad_angle_boundary:0"

# Minimal-train pure-policy eval: 256 steps, BC off (beta 0), 16 eval episodes.
RL_MA_PROD_TRAIN_STEPS=256 RL_MA_PROD_MAX_EPISODE_STEPS=800 \
RL_MA_PROD_EVAL_INTERVAL=256 RL_MA_PROD_CHECKPOINT_INTERVAL=100000 \
RL_MA_PROD_EVAL_EPISODES=16 \
RL_MA_PROD_EXPERIMENT_NAME=v4_2_coord_off_ablation \
RL_MA_PROD_BC_BETA_MAX=0.0 RL_MA_PROD_BC_BETA_MIN=0.0 \
"$PY" -m rlfootball.training.train_multiagent_production --stage-id 7 \
  --checkpoint runs/multiagent_v4_2/checkpoint_stage7_step12000.pt \
  --dataset "$FINAL/samples.npz"
