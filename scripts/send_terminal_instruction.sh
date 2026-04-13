#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTRUCTION_FILE="${SCRIPT_DIR}/../scheduler/instruction/instruction.txt"

usage() {
  cat <<'EOF'
Usage: ./scripts/send_terminal_instruction.sh "pick up the red bottle"

Special commands:
  reset
  nothing
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

RAW_INSTRUCTION="$*"

case "${RAW_INSTRUCTION}" in
  reset|nothing)
    INSTRUCTION="${RAW_INSTRUCTION}"
    ;;
  "[Low]"*|"[High]"*)
    INSTRUCTION="${RAW_INSTRUCTION}"
    ;;
  "[low]"*)
    INSTRUCTION="${RAW_INSTRUCTION/\[low\]/[Low]}"
    ;;
  *)
    INSTRUCTION="[Low]: ${RAW_INSTRUCTION}"
    ;;
esac

printf '%s\n' "${INSTRUCTION}" > "${INSTRUCTION_FILE}"
echo "Wrote instruction to ${INSTRUCTION_FILE}: ${INSTRUCTION}"
