#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EFMNODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_CONFIG="${EFMNODE_DIR}/config.r1lite_zero_shot_terminal.toml"

usage() {
  cat <<'EOF'
Usage: ./scripts/run_r1lite_zero_shot_terminal.sh --model-path /abs/path/to/checkpoint_dir [--config-path /abs/path/to/config.toml]

This starts zero-shot R1Lite terminal inference with TensorRT engines and
Gemini-based bbox selection. Set one of these before launch:
  export GEMINI_API_KEY=...
  export GOOGLE_API_KEY=...
  export VLM_API_KEY=...

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

for required_file in config.yaml dataset_stats.json gemma_rmsnorm.so; do
  if [[ ! -f "${MODEL_PATH}/${required_file}" ]]; then
    echo "Missing ${MODEL_PATH}/${required_file}" >&2
    exit 1
  fi
done

ENGINE_FAMILY=""

if [[ -f "${MODEL_PATH}/prefill.fp16.engine" && -f "${MODEL_PATH}/decode.fp16.engine" ]]; then
  ENGINE_FAMILY="vendor_prefill_decode"
elif [[ -f "${MODEL_PATH}/galaxea_zero_encoder_opt.fp16.engine" && -f "${MODEL_PATH}/galaxea_zero_predictor_opt.fp16.engine" ]]; then
  ENGINE_FAMILY="galaxea_zero_pair"
else
  echo "Missing a supported TensorRT engine pair in ${MODEL_PATH}" >&2
  echo "Expected either prefill/decode or galaxea_zero_encoder_opt/galaxea_zero_predictor_opt." >&2
  exit 1
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config file does not exist: ${CONFIG_PATH}" >&2
  exit 1
fi

if [[ -z "${GEMINI_API_KEY:-}" && -z "${GOOGLE_API_KEY:-}" && -z "${VLM_API_KEY:-}" ]]; then
  echo "Missing Gemini API key." >&2
  echo "Set GEMINI_API_KEY, GOOGLE_API_KEY, or VLM_API_KEY before launch." >&2
  exit 1
fi

for trt_lib_dir in \
  /usr/TensorRT-10.13.0.35/targets/x86_64-linux-gnu/lib \
  /usr/TensorRT-10.13.0.35/lib \
  /data/TensorRT-10.13.0.35/targets/x86_64-linux-gnu/lib \
  /data/TensorRT-10.13.0.35/lib
do
  if [[ -d "${trt_lib_dir}" ]]; then
    export LD_LIBRARY_PATH="${trt_lib_dir}:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
    break
  fi
done

ZERO_SHOT_ENGINE_FAMILY="${ENGINE_FAMILY}" python3 - <<'PY'
import importlib.util
import os
import sys

try:
    import torch
except Exception as exc:
    print(f"Failed to import torch from {sys.executable}: {exc}", file=sys.stderr)
    raise SystemExit(1)

if importlib.util.find_spec("tensorrt") is None:
    print(
        "TensorRT is not installed in the active Python environment.\n"
        "Use:\n"
        "  uv pip install --python /home/revanths/GalaxeaVLA/.venv/bin/python "
        "tensorrt==10.13.0.35 tensorrt-cu12==10.13.0.35 "
        "tensorrt-cu12-bindings==10.13.0.35 tensorrt-cu12-libs==10.13.0.35",
        file=sys.stderr,
    )
    raise SystemExit(1)

import tensorrt as trt

if (
    os.environ.get("ZERO_SHOT_ENGINE_FAMILY") == "vendor_prefill_decode"
    and trt.__version__ != "10.13.0.35"
):
    print(
        "Vendor prefill/decode engines expect TensorRT 10.13.0.35.\n"
        f"Active TensorRT version is {trt.__version__}.\n"
        "Use the vendor-matching CUDA-12 / TRT-10.13 environment before launch.",
        file=sys.stderr,
    )
    raise SystemExit(1)

if not torch.cuda.is_available():
    print(
        "torch.cuda.is_available() is false in the active Python environment.\n"
        "Verify the NVIDIA driver is loaded and that this shell can access the GPU.",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY

exec "${SCRIPT_DIR}/run_r1lite_terminal.sh" \
  --config-path "${CONFIG_PATH}" \
  --model-path "${MODEL_PATH}"
