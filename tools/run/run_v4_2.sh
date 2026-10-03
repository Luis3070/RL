#!/usr/bin/env bash
# v4.2 = v4.1 + SHOT DISCIPLINE.
# Root cause of the persistent preshoot_bad_angle_boundary FAIL (bad_shoot_rate>0.25 in
# v4 AND v4.1): the `team` PPO scenario pool (team_production_scenarios) contains ZERO
# bad-angle states, so PPO never experiences a bad-angle shoot decision and can't unlearn
# the v2b-inherited "shoot anyway" reflex. behavioral_evaluator's preshoot scenario spawns
# the striker 24-38 deg off the goal heading at 2.5-3.0m -> shooting there = not shoot_zone
# = bad_shoot.
# FIX (surgical, no gate-threshold change, no episode-step change, no corpus change):
#   inject production_curriculum's preshoot_bad_angle_boundary into the PPO reset pool at
#   ~25% weight. Its stage_id=4 -> reward_weights bad_shoot=-2.0, so PPO is penalized for
#   the bad-angle shot and learns to realign/dribble first. The eval-set's other 4 scenarios
#   are zeroed so only the bad-angle state is added; the team cooperation pool is untouched.
# Everything else IDENTICAL to v4.1 (warm-start v2b@7500, same corpus, beta-max 0.30
# BC-dominant to protect the 99% pass, gate fix already in train.py).
set -u
REPO=/home/luis/GitHub/RL
export PYTHONPATH="$REPO:/home/luis/GitHub/SabanaHerons2026/Make/Python"
export DISPLAY="${DISPLAY:-:1}"
export RL_GATE_SHOOT_OPEN_ENTER=0.30 RL_GATE_SHOOT_OPEN_EXIT=0.20
export RL_MA_TEACHER_VISIBLE_SCENE_NAME=RL1v0_3D_visible
export RL_VISIBLE_RESET_MAX_ATTEMPTS=8   # training resilience (fail-fast=2 is for dataset gen)
export RL_MA_SHOOT_THRESHOLD=0.30
PY="${PY:-$HOME/anaconda3/envs/RLNAO/bin/python}"
TM=runs/teacher_multiagent_dataset
FINAL="$TM/dataset_v4_backbone_plus_overlay"
cd "$REPO"

# Add the eval pool (for preshoot_bad_angle_boundary) ON TOP of the team pool, then weight so
# ONLY the bad-angle scenario is sampled from eval (~25% of episodes); team weights preserved.
export RL_PRODUCTION_SCENARIO_SET="team,eval"
export RL_PRODUCTION_SCENARIO_WEIGHTS="team_3v0_clean:1,team_3v0_static_pinch:2,team_3v0_dynamic:2,team_3v3_full_dynamic:3,preshoot_bad_angle_boundary:5,finish_band_2p5_3m_clean:0,finish_band_2p5_3m_static_lane:0,dynamic_full_field_pressure:0,blocked_corridor_repair:0"

RL_MA_PROD_TRAIN_STEPS=12000 RL_MA_PROD_MAX_EPISODE_STEPS=800 \
RL_MA_PROD_EVAL_INTERVAL=3000 RL_MA_PROD_CHECKPOINT_INTERVAL=3000 \
RL_MA_PROD_SCENARIO_SET=team RL_MA_PROD_EXPERIMENT_NAME=multiagent_v4_2 \
RL_MA_PROD_BC_BETA_MAX=0.30 RL_MA_PROD_BC_BETA_MIN=0.05 \
RL_MA_PROD_BC_HALF_LIFE_EVALS=3 RL_MA_PROD_BC_REPLAY_UPDATES=4 \
RL_MA_BC_DECISION_WINDOW_ONLY=1 RL_MA_BC_STRIKER_PASS_FRAC=0.30 \
"$PY" -m rlfootball.training.train_multiagent_production --stage-id 7 \
  --checkpoint runs/multiagent_stage7_v2b_teamrewards/checkpoint_stage7_step7500.pt \
  --dataset "$FINAL/samples.npz"
