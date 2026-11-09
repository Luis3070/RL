from __future__ import annotations

import gymnasium as gym
import torch

from rlfootball.training.train import PolicyNetwork, ValueNetwork, _load_state_dict_obs_padded, gate_skill_mask_from_obs


def _spaces(obs_size: int):
    observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(obs_size,), dtype=float)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=float)
    return observation_space, action_space


def test_gate_skill_mask_depends_only_on_gate_columns():
    """Mask at 41-dim obs depends only on cols 23/24 (single-agent path)."""
    base_obs = torch.zeros((1, 41), dtype=torch.float32)
    base_obs[0, 23] = 1.0
    base_mask = gate_skill_mask_from_obs(base_obs).clone()

    for perturb_idx in list(range(0, 23)) + list(range(26, 41)):
        obs = base_obs.clone()
        obs[0, perturb_idx] = 0.5
        mask = gate_skill_mask_from_obs(obs)
        assert torch.equal(mask, base_mask), f"mask depends on idx {perturb_idx}"


def test_gate_skill_mask_team_depends_only_on_specified_columns():
    """Mask at 47-dim obs depends only on cols 23/24/41/42/44/45/46.

    Perturbing any other column must not change the mask. Validates Design A
    end-to-end including the multi-agent role-conditioned path.
    """
    base_obs = torch.zeros((1, 47), dtype=torch.float32)
    base_obs[0, 23] = 1.0   # shoot_armed
    base_obs[0, 44] = 1.0   # is_striker
    base_mask = gate_skill_mask_from_obs(base_obs).clone()

    gate_cols = {23, 24, 41, 42, 44, 45, 46}
    for perturb_idx in [i for i in range(47) if i not in gate_cols]:
        obs = base_obs.clone()
        obs[0, perturb_idx] = 0.5
        mask = gate_skill_mask_from_obs(obs)
        assert torch.equal(mask, base_mask), (
            f"47-dim mask depends on non-gate idx {perturb_idx}"
        )


def test_gate_skill_mask_role_conditioned_forbid():
    """Role bits enforce the §11.A.4 forbid contract.

    - open_support: shoot/dribble/pass all forbidden regardless of armed flags.
    - off_ball_support: shoot/pass forbidden; dribble allowed if armed (escape hatch).
    - striker: standard single-agent gate, plus pass when pass_armed.
    """
    from rlfootball.core.action import SKILL_ORDER

    shoot_idx = SKILL_ORDER.index("shoot")
    pass_idx = SKILL_ORDER.index("pass")
    dribble_idx = SKILL_ORDER.index("dribble")

    # Open-support: even with everything armed, shoot/dribble/pass must stay 0.
    obs_open = torch.zeros((1, 47), dtype=torch.float32)
    obs_open[0, 23] = 1.0           # shoot_armed
    obs_open[0, 24] = 1.0           # dribble_armed
    obs_open[0, 41] = 1.0           # pass_armed
    obs_open[0, 45] = 1.0           # is_open_support
    mask = gate_skill_mask_from_obs(obs_open)
    assert mask[0, shoot_idx] == 0.0
    assert mask[0, dribble_idx] == 0.0
    assert mask[0, pass_idx] == 0.0

    # Off-ball-support: shoot/pass forbidden; dribble allowed (escape hatch).
    obs_off = torch.zeros((1, 47), dtype=torch.float32)
    obs_off[0, 23] = 1.0
    obs_off[0, 24] = 1.0
    obs_off[0, 41] = 1.0
    obs_off[0, 46] = 1.0            # is_off_ball_support
    mask = gate_skill_mask_from_obs(obs_off)
    assert mask[0, shoot_idx] == 0.0
    assert mask[0, pass_idx] == 0.0
    assert mask[0, dribble_idx] == 1.0   # escape hatch open

    # Striker: pass requires pass_armed; shoot follows shoot_armed.
    obs_str = torch.zeros((1, 47), dtype=torch.float32)
    obs_str[0, 23] = 1.0
    obs_str[0, 41] = 1.0
    obs_str[0, 44] = 1.0            # is_striker
    mask = gate_skill_mask_from_obs(obs_str)
    assert mask[0, shoot_idx] == 1.0
    assert mask[0, pass_idx] == 1.0


def test_obs_padding_preserves_policy_and_value_outputs():
    """26 → 41 zero-pad: legacy contract preserved bit-identical."""
    torch.manual_seed(7)

    obs26_space, action_space = _spaces(26)
    obs41_space, _ = _spaces(41)

    policy26 = PolicyNetwork(obs26_space, action_space, "cpu", 32, param_log_prob_weight=0.25)
    policy41 = PolicyNetwork(obs41_space, action_space, "cpu", 32, param_log_prob_weight=0.25)
    _load_state_dict_obs_padded(policy41, policy26.state_dict(), label="test-policy")

    value26 = ValueNetwork(obs26_space, obs26_space, action_space, "cpu", 32)
    value41 = ValueNetwork(obs41_space, obs41_space, action_space, "cpu", 32)
    _load_state_dict_obs_padded(value41, value26.state_dict(), label="test-value")

    obs26 = torch.randn((4, 26), dtype=torch.float32)
    obs41 = torch.zeros((4, 41), dtype=torch.float32)
    obs41[:, :26] = obs26

    with torch.no_grad():
        skill_logits26, outputs26 = policy26.compute({"observations": obs26})
        skill_logits41, outputs41 = policy41.compute({"observations": obs41})
        values26, _ = value26.compute({"states": obs26, "observations": obs26})
        values41, _ = value41.compute({"states": obs41, "observations": obs41})

    assert torch.allclose(skill_logits41, skill_logits26, atol=1e-6)
    assert torch.allclose(outputs41["param_mean"], outputs26["param_mean"], atol=1e-6)
    assert torch.allclose(outputs41["log_std"], outputs26["log_std"], atol=1e-6)
    assert torch.allclose(values41, values26, atol=1e-6)


def test_obs_padding_preserves_policy_value_outputs_26_to_47():
    """26 → 47 zero-pad: the validated 4/5 striker behaviour holds at 47-dim.

    Critical safety net for the Tue/Wed sprint: warm-starting the production
    single-agent checkpoint into the new 47-dim head must produce bit-identical
    policy and value outputs when the 21 new columns (26..46) are zero.
    """
    torch.manual_seed(11)

    obs26_space, action_space = _spaces(26)
    obs47_space, _ = _spaces(47)

    policy26 = PolicyNetwork(obs26_space, action_space, "cpu", 32, param_log_prob_weight=0.25)
    policy47 = PolicyNetwork(obs47_space, action_space, "cpu", 32, param_log_prob_weight=0.25)
    _load_state_dict_obs_padded(policy47, policy26.state_dict(), label="test-policy-47")

    value26 = ValueNetwork(obs26_space, obs26_space, action_space, "cpu", 32)
    value47 = ValueNetwork(obs47_space, obs47_space, action_space, "cpu", 32)
    _load_state_dict_obs_padded(value47, value26.state_dict(), label="test-value-47")

    obs26 = torch.randn((4, 26), dtype=torch.float32)
    obs47 = torch.zeros((4, 47), dtype=torch.float32)
    obs47[:, :26] = obs26

    with torch.no_grad():
        skill26, out26 = policy26.compute({"observations": obs26})
        skill47, out47 = policy47.compute({"observations": obs47})
        vals26, _ = value26.compute({"states": obs26, "observations": obs26})
        vals47, _ = value47.compute({"states": obs47, "observations": obs47})

    assert torch.allclose(skill47, skill26, atol=1e-6), "skill logits drift at 47-dim"
    assert torch.allclose(out47["param_mean"], out26["param_mean"], atol=1e-6)
    assert torch.allclose(out47["log_std"], out26["log_std"], atol=1e-6)
    assert torch.allclose(vals47, vals26, atol=1e-6), "value head drifts at 47-dim"
