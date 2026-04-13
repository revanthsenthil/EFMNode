#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HF_ENDPOINT="https://hf-mirror.com"
export PYTORCH_JIT_LOG_LEVEL='profiling_graph_executor_impl'
export HYDRA_FULL_ERROR=1
export PYTHONPATH="${SCRIPT_DIR}/../:${PYTHONPATH:-}"
for trt_lib_dir in \
  /usr/TensorRT-10.13.0.35/targets/x86_64-linux-gnu/lib \
  /usr/TensorRT-10.13.0.35/lib \
  /data/TensorRT-10.13.0.35/targets/x86_64-linux-gnu/lib \
  /data/TensorRT-10.13.0.35/lib
do
  if [[ -d "${trt_lib_dir}" ]]; then
    export LD_LIBRARY_PATH="${trt_lib_dir}:/usr/lib/x86_64-linux-gnu/:${LD_LIBRARY_PATH:-}"
    break
  fi
done

python3 "${SCRIPT_DIR}/../serving/policy_server.py" "$@"
