"""
diagnose_stage0_takeover_heuristic.py

Short diagnostic for Stage 0 takeover learnability under a simple positioning
heuristic. This is not a training script. It answers a narrower question:

  "If the active RL agent reliably reaches the handoff envelope, does the
   current real BHuman-in-loop integration convert those handoffs into
   successful takeovers?"

The heuristic only controls positioning:
  - place the agent slightly behind the ball on the x axis
  - let theta be handled by the environment's face-ball logic

This matches the intended Stage 0 action semantics: RL does NOT kick/dribble
directly; it only places the robot so BHuman can take over.
"""

from __future__ import annotations

import os
from collections import Counter

import numpy as np

os.environ["RL_EXECUTION_MODE"] = "bhuman_sidecar"

from config import CURRICULUM
from environment import NaoSoccerEnv


EPISODES = 10
SEED_BASE = 100
STAGE = CURRICULUM[0]
DESIRED_BALL_OFFSET_MM = np.array([-180.0, 0.0], dtype=np.float32)


def _heuristic_action(env: NaoSoccerEnv, aid: str) -> np.ndarray:
    agent = env._agents_state[aid]
    desired = env._ball.pos + DESIRED_BALL_OFFSET_MM
    delta = desired - agent.pos
    horizon = max(float(STAGE.local_horizon), 1.0)

    action = np.zeros(3, dtype=np.float32)
    action[0] = float(np.clip(delta[0] / horizon, -1.0, 1.0))
    action[1] = float(np.clip(delta[1] / horizon, -1.0, 1.0))
    return action


def _run_episode(seed: int) -> dict:
    env = NaoSoccerEnv(stage_cfg=STAGE, use_fov=True, seed=seed)
    try:
        _, infos = env.reset(seed=seed)
        last_info = next(iter(infos.values())) if infos else {}

        while env.agents:
            actions = {aid: _heuristic_action(env, aid) for aid in env.agents}
            _, _, terminateds, truncateds, infos = env.step(actions)
            if infos:
                last_info = next(iter(infos.values()))
            if terminateds.get("__all__") or truncateds.get("__all__"):
                break

        return {
            "handoff_achieved": bool(last_info.get("episode_handoff_achieved", False)),
            "takeover_count": int(last_info.get("episode_takeover_count", 0)),
            "takeover_success": int(last_info.get("episode_takeover_success", 0)),
            "events": list(env._takeover_events),
        }
    finally:
        env.close()


def main() -> None:
    print("[DIAG] Stage 0 heuristic takeover probe")
    print(f"  episodes={EPISODES}")
    print(f"  stage={STAGE.stage_id}:{STAGE.name}")

    handoff_eps = 0
    takeover_total = 0
    takeover_success_total = 0
    reasons: Counter[str] = Counter()
    macro_types: Counter[str] = Counter()

    for ep in range(EPISODES):
        result = _run_episode(SEED_BASE + ep)
        handoff_eps += int(result["handoff_achieved"])
        takeover_total += result["takeover_count"]
        takeover_success_total += result["takeover_success"]

        for event in result["events"]:
            reasons[str(event.get("end_reason", "unknown"))] += 1
            macro_types[str(event.get("type", "unknown"))] += 1

        print(
            f"  ep={ep + 1:02d} "
            f"handoff={int(result['handoff_achieved'])} "
            f"takeovers={result['takeover_count']} "
            f"successes={result['takeover_success']}"
        )

    success_rate = (
        float(takeover_success_total) / float(takeover_total)
        if takeover_total > 0 else 0.0
    )

    print("\nRESULT")
    print(f"  handoff_episodes={handoff_eps}/{EPISODES}")
    print(f"  takeover_success_total={takeover_success_total}")
    print(f"  takeover_count_total={takeover_total}")
    print(f"  takeover_success_rate={success_rate:.4f}")
    print(f"  end_reasons={dict(reasons)}")
    print(f"  macro_types={dict(macro_types)}")


if __name__ == "__main__":
    main()
