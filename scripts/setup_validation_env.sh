#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
ROOT_DIR="$RL_REPO_DIR"
BHUMAN_DIR="$SABANAHERONS_DIR"
VENV_DIR="${ROOT_DIR}/.venv"

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

need_cmd python3

if ! python3 -m venv --help >/dev/null 2>&1; then
  cat >&2 <<'EOF'
python3 can not create virtual environments on this machine.

Install this first:
  sudo apt install python3.12-venv
EOF
  exit 1
fi

if ! command -v Xvfb >/dev/null 2>&1; then
  cat >&2 <<'EOF'
Xvfb is not available in PATH.

Install this first:
  sudo apt install xvfb
EOF
  exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV_DIR}/bin/pip" install \
  torch \
  skrl \
  pettingzoo \
  gymnasium \
  pandas \
  matplotlib \
  scipy \
  psutil \
  tqdm

cat <<EOF
Validation environment ready.

Use it through:
  ${ROOT_DIR}/scripts/run_with_validation_env.sh python3 -m rlfootball.diagnostics.diag_n6_slots

Current assumptions:
  Venv:      ${VENV_DIR}
  PYBH path: ${BHUMAN_DIR}/Make/Python
  Xvfb:      $(command -v Xvfb)
EOF
