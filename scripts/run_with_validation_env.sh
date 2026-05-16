#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BHUMAN_DIR="$(cd "${ROOT_DIR}/../SabanaHerons2026" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
LOCAL_CACHE_DIR="${ROOT_DIR}/.runtime-cache"

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <command> [args...]" >&2
  exit 1
fi

if [[ -x "${VENV_DIR}/bin/python" ]]; then
  export PATH="${VENV_DIR}/bin:${PATH}"
  export PYTHONPATH="${BHUMAN_DIR}/Make/Python${PYTHONPATH:+:${PYTHONPATH}}"
elif [[ -d "${ROOT_DIR}/.deps" ]]; then
  export PYTHONPATH="${ROOT_DIR}/.deps:${BHUMAN_DIR}/Make/Python${PYTHONPATH:+:${PYTHONPATH}}"
else
  cat >&2 <<EOF
No Python environment found.

Expected one of:
  ${VENV_DIR}
  ${ROOT_DIR}/.deps

Run:
  ${ROOT_DIR}/scripts/setup_validation_env.sh
EOF
  exit 1
fi

mkdir -p "${LOCAL_CACHE_DIR}/matplotlib"
export MPLCONFIGDIR="${LOCAL_CACHE_DIR}/matplotlib"

if command -v Xvfb >/dev/null 2>&1; then
  :
else
  cat >&2 <<'EOF'
Xvfb not found.

Install it with:
  sudo apt install xvfb
EOF
  exit 1
fi

exec "$@"
