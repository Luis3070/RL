from __future__ import annotations

import gymnasium as gym
import torch

from train import PolicyNetwork, ValueNetwork, _load_state_dict_obs_padded, gate_skill_mask_from_obs


def _spaces(obs_size: int):
    observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(obs_size,), dtype=float)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=float)
    return observation_space, action_space


def test_gate_skill_mask_depends_only_on_gate_columns():
    base_obs = torch.zeros((1, 41), dtype=torch.float32)
    base_obs[0, 23] = 1.0
    base_mask = gate_skill_mask_from_obs(base_obs).clone()

    for perturb_idx in list(range(0, 23)) + list(range(26, 41)):
        obs = base_obs.clone()
        obs[0, perturb_idx] = 0.5
        mask = gate_skill_mask_from_obs(obs)
        assert torch.equal(mask, base_mask), f"mask depends on idx {perturb_idx}"


def test_obs_padding_preserves_policy_and_value_outputs():
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
