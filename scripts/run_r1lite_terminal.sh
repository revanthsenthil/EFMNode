#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EFMNODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_CONFIG="${EFMNODE_DIR}/config.r1lite_terminal.toml"

usage() {
  cat <<'EOF'
Usage: ./scripts/run_r1lite_terminal.sh --model-path /abs/path/to/checkpoint_dir [--config-path /abs/path/to/config.toml]

This starts EFMNode in terminal mode for R1Lite. Instructions are read from:
  EFMNode/scheduler/instruction/instruction.txt

Send a command with:
  ./scripts/send_terminal_instruction.sh "pick up the red bottle"
EOF
}

MODEL_PATH=""
CONFIG_PATH="${DEFAULT_CONFIG}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path)
      MODEL_PATH="${2:-}"
      shift 2
      ;;
    --config-path)
      CONFIG_PATH="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${MODEL_PATH}" ]]; then
  echo "--model-path is required." >&2
  usage
  exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Model path does not exist: ${MODEL_PATH}" >&2
  exit 1
fi

if [[ ! -f "${MODEL_PATH}/config.yaml" ]]; then
  echo "Missing ${MODEL_PATH}/config.yaml" >&2
  exit 1
fi

if [[ ! -f "${MODEL_PATH}/dataset_stats.json" ]]; then
  echo "Missing ${MODEL_PATH}/dataset_stats.json" >&2
  exit 1
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config file does not exist: ${CONFIG_PATH}" >&2
  exit 1
fi

if [[ -f /opt/ros/humble/setup.bash ]]; then
  # ROS setup scripts can reference unset variables internally, which conflicts
  # with `set -u`. Temporarily relax nounset while sourcing.
  set +u
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
  set -u
fi

"${SCRIPT_DIR}/send_terminal_instruction.sh" nothing >/dev/null

exec "${SCRIPT_DIR}/run.sh" \
  --config-path "${CONFIG_PATH}" \
  --model-path "${MODEL_PATH}"
