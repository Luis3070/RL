#!/usr/bin/env bash
# Shared path setup. Meant to be SOURCED by launcher scripts:
#   source "<path-to>/scripts/env.sh"
# Loads <repo>/.env if present, then exports RL_REPO_DIR and SABANAHERONS_DIR.
# Precedence: variables already set in the environment > .env > derived defaults
# (repo root from this file's location; ../SabanaHerons2026 next to the repo).
_rl_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_rl_pre_repo="${RL_REPO_DIR-}"
_rl_pre_bh="${SABANAHERONS_DIR-}"
if [[ -f "${_rl_root}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  . "${_rl_root}/.env"
  set +a
fi
[[ -n "${_rl_pre_repo}" ]] && RL_REPO_DIR="${_rl_pre_repo}"
[[ -n "${_rl_pre_bh}" ]] && SABANAHERONS_DIR="${_rl_pre_bh}"
export RL_REPO_DIR="${RL_REPO_DIR:-${_rl_root}}"
export SABANAHERONS_DIR="${SABANAHERONS_DIR:-$(dirname "${RL_REPO_DIR}")/SabanaHerons2026}"
unset _rl_root _rl_pre_repo _rl_pre_bh
