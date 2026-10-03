#!/usr/bin/env python3
"""Parity harness for the obs47 defender transplant (SabanaHerons2026 C++).

Validates the two pieces most likely to diverge between the Python training pipeline
and the C++ embedded inference (Src/Libs/RL + StrategyBehaviorControl):

  1. The role-conditioned skill mask. Compares the C++ port
     (StrategyBehaviorControl.cpp:applyRoleConditionedMask) against the *real*
     train.gate_skill_mask_from_obs ANDed with the production stage mask, over every
     combination of armed gate bits, for the off_ball_support role.
  2. The ONNX contract: loads ppo_defender_hsl2026.onnx, runs a few obs47 fixtures and
     dumps (skill_logits, param_mean, masked argmax skill) as JSON so the same vectors
     can be cross-checked against the C++ "[RL] Embedded PPO action" logs in SimRobot.

Run from the RL repo root:
    python tools/eval/parity_defender_dump.py
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from rlfootball.core.observation import ObsIndex, OBS_SIZE  # noqa: E402
from rlfootball.core.action import SKILL_ORDER  # noqa: E402
from rlfootball.training.train import gate_skill_mask_from_obs  # noqa: E402

MODEL_PATH = REPO_ROOT / "models" / "defender" / "ppo_defender_hsl2026.onnx"

# Production stage-8 active skills (from the manifest): mark and observe are never active.
STAGE_ACTIVE = {"stand", "walk", "shoot", "pass", "dribble", "block"}
STAGE_MASK = np.array([1.0 if s in STAGE_ACTIVE else 0.0 for s in SKILL_ORDER], dtype=np.float32)

SKILL_IDX = {name: i for i, name in enumerate(SKILL_ORDER)}


def cpp_role_mask_offball(shoot_armed: bool, dribble_armed: bool,
                          pass_armed: bool, observe_armed: bool) -> np.ndarray:
    """Re-implementation of the C++ applyRoleConditionedMask for role=offBallSupport.

    Mirrors StrategyBehaviorControl.cpp exactly: start all-allowed, forbid mark/observe
    (stage), then forbid per role. stand/walk always allowed.
    """
    allowed = np.ones(len(SKILL_ORDER), dtype=np.float32)
    striker = False
    off_ball = True

    allowed[SKILL_IDX["mark"]] = 0.0
    allowed[SKILL_IDX["observe"]] = 0.0

    if not (shoot_armed and striker):
        allowed[SKILL_IDX["shoot"]] = 0.0
    if not (pass_armed and striker):
        allowed[SKILL_IDX["pass"]] = 0.0
    if not (dribble_armed and (striker or off_ball)):
        allowed[SKILL_IDX["dribble"]] = 0.0
    if not (dribble_armed and not striker):
        allowed[SKILL_IDX["block"]] = 0.0
    # stand and walk always allowed.
    return allowed


def real_combined_mask_offball(shoot_armed: bool, dribble_armed: bool,
                               pass_armed: bool, observe_armed: bool) -> np.ndarray:
    """Ground truth: STAGE_MASK AND train.gate_skill_mask_from_obs, role=off_ball_support."""
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    obs[int(ObsIndex.shoot_armed)] = 1.0 if shoot_armed else 0.0
    obs[int(ObsIndex.dribble_armed)] = 1.0 if dribble_armed else 0.0
    obs[int(ObsIndex.pass_armed)] = 1.0 if pass_armed else 0.0
    obs[int(ObsIndex.observe_armed)] = 1.0 if observe_armed else 0.0
    obs[int(ObsIndex.is_off_ball_support)] = 1.0
    role_mask = gate_skill_mask_from_obs(torch.from_numpy(obs[None, :])).numpy()[0]
    return STAGE_MASK * role_mask


def test_mask_parity() -> bool:
    ok = True
    print("== Mask parity (role=off_ball_support): C++ port vs real gate_skill_mask_from_obs ==")
    for shoot_armed, dribble_armed, pass_armed, observe_armed in itertools.product([False, True], repeat=4):
        cpp = (cpp_role_mask_offball(shoot_armed, dribble_armed, pass_armed, observe_armed) > 0.5)
        real = (real_combined_mask_offball(shoot_armed, dribble_armed, pass_armed, observe_armed) > 0.5)
        if not np.array_equal(cpp, real):
            ok = False
            cpp_allowed = [SKILL_ORDER[i] for i in np.where(cpp)[0]]
            real_allowed = [SKILL_ORDER[i] for i in np.where(real)[0]]
            print(f"  MISMATCH shoot={int(shoot_armed)} dribble={int(dribble_armed)} "
                  f"pass={int(pass_armed)} observe={int(observe_armed)}")
            print(f"    C++ allows : {cpp_allowed}")
            print(f"    real allows: {real_allowed}")
    if ok:
        print("  OK — all 16 armed-bit combinations match.")
    return ok


def test_onnx_fixtures() -> bool:
    try:
        import onnxruntime as ort
    except ImportError:
        print("== ONNX fixtures: SKIPPED (onnxruntime not installed) ==")
        return True
    if not MODEL_PATH.exists():
        print(f"== ONNX fixtures: SKIPPED (model not found at {MODEL_PATH}) ==")
        return True

    print(f"== ONNX fixtures: {MODEL_PATH.name} ==")
    sess = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    fixtures = {}
    rng = np.random.default_rng(0)
    cases = {
        "zeros_offball": _obs_offball(),
        "dribble_armed": _obs_offball(dribble_armed=True, ball_rel_x=0.05),
        "random_offball": _obs_offball(extra=rng.uniform(-0.3, 0.3, OBS_SIZE).astype(np.float32)),
    }
    for name, obs in cases.items():
        logits, params = sess.run(None, {input_name: obs[None, :]})
        logits = logits[0].astype(float)
        params = params[0].astype(float)
        # Apply the same combined mask the C++ uses for this fixture's armed bits.
        sa = obs[int(ObsIndex.shoot_armed)] >= 0.5
        da = obs[int(ObsIndex.dribble_armed)] >= 0.5
        pa = obs[int(ObsIndex.pass_armed)] >= 0.5
        oa = obs[int(ObsIndex.observe_armed)] >= 0.5
        mask = cpp_role_mask_offball(sa, da, pa, oa)
        masked = np.where(mask > 0.5, logits, -1e9)
        skill_idx = int(np.argmax(masked))
        fixtures[name] = {
            "obs47": [round(float(v), 6) for v in obs],
            "skill_logits": [round(v, 6) for v in logits],
            "param_mean": [round(v, 6) for v in params],
            "masked_skill_idx": skill_idx,
            "masked_skill": SKILL_ORDER[skill_idx],
        }
        print(f"  {name:16s} -> skill={SKILL_ORDER[skill_idx]:8s} "
              f"params={[round(v, 3) for v in params]}")

    out = REPO_ROOT / "tools" / "parity_defender_fixtures.json"
    out.write_text(json.dumps(fixtures, indent=2))
    print(f"  wrote {out}")
    return True


def _obs_offball(*, dribble_armed: bool = False, shoot_armed: bool = False,
                 ball_rel_x: float = 0.0, extra: np.ndarray | None = None) -> np.ndarray:
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    if extra is not None:
        obs = extra.copy()
    obs[int(ObsIndex.is_striker)] = 0.0
    obs[int(ObsIndex.is_open_support)] = 0.0
    obs[int(ObsIndex.is_off_ball_support)] = 1.0
    obs[int(ObsIndex.shoot_armed)] = 1.0 if shoot_armed else 0.0
    obs[int(ObsIndex.dribble_armed)] = 1.0 if dribble_armed else 0.0
    obs[int(ObsIndex.pass_armed)] = 0.0
    obs[int(ObsIndex.observe_armed)] = 0.0
    obs[int(ObsIndex.ball_rel_x)] = ball_rel_x
    return obs


def main() -> int:
    mask_ok = test_mask_parity()
    onnx_ok = test_onnx_fixtures()
    print()
    if mask_ok and onnx_ok:
        print("PARITY OK")
        return 0
    print("PARITY FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
