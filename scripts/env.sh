#!/usr/bin/env bash
# Shared path setup. Meant to be SOURCED by launcher scripts:
#   source "<path-to>/scripts/env.sh"
# Loads <repo>/.env if present, then exports RL_REPO_DIR and SABANAHERONS_DIR.
# Precedence: environment > .env > derived defaults
# (repo root from this file's location; ../SabanaHerons2026 next to the repo).
_rl_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Parse only simple KEY=VALUE lines for the two documented keys; never execute
# the file. Mirrors _load_dotenv in rlfootball/core/config.py.
if [[ -f "${_rl_root}/.env" ]]; then
  while IFS= read -r _rl_line || [[ -n "${_rl_line}" ]]; do
    _rl_line="${_rl_line#"${_rl_line%%[![:space:]]*}"}"
    _rl_line="${_rl_line%"${_rl_line##*[![:space:]]}"}"
    [[ "${_rl_line}" == export[[:space:]]* ]] && _rl_line="${_rl_line#export}" \
      && _rl_line="${_rl_line#"${_rl_line%%[![:space:]]*}"}"
    [[ "${_rl_line}" == *=* ]] || continue
    _rl_key="${_rl_line%%=*}"
    _rl_key="${_rl_key%"${_rl_key##*[![:space:]]}"}"
    [[ "${_rl_key}" == RL_REPO_DIR || "${_rl_key}" == SABANAHERONS_DIR ]] || continue
    _rl_val="${_rl_line#*=}"
    _rl_val="${_rl_val#"${_rl_val%%[![:space:]]*}"}"
    if [[ "${_rl_val}" =~ ^\"([^\"]*)\" || "${_rl_val}" =~ ^\'([^\']*)\' ]]; then
      _rl_val="${BASH_REMATCH[1]}"
    else
      [[ "${_rl_val}" =~ ^(.*[^[:space:]])?[[:space:]]+#.*$ ]] && _rl_val="${BASH_REMATCH[1]}"
      [[ "${_rl_val}" =~ ^[[:space:]]*#.*$ ]] && _rl_val=""
    fi
    [[ -n "${!_rl_key-}" ]] || export "${_rl_key}=${_rl_val}"
  done < "${_rl_root}/.env"
fi
export RL_REPO_DIR="${RL_REPO_DIR:-${_rl_root}}"
export SABANAHERONS_DIR="${SABANAHERONS_DIR:-$(dirname "${RL_REPO_DIR}")/SabanaHerons2026}"
unset _rl_root _rl_line _rl_key _rl_val
