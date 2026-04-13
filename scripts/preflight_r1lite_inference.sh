#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/preflight_r1lite_inference.sh --model-path /abs/path/to/checkpoint_dir

Checks:
  - ros2 is available
  - required R1Lite topics are present
  - action topics and their current publishers
  - checkpoint directory contains the expected files

Note:
  EFMNode now accepts either:
    model_state_dict.pt
  or:
    model.pt
  or TensorRT engine pairs:
    galaxea_zero_encoder_opt.fp16.engine + galaxea_zero_predictor_opt.fp16.engine
  or:
    prefill.fp16.engine + decode.fp16.engine
EOF
}

MODEL_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path)
      MODEL_PATH="${2:-}"
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

if ! command -v ros2 >/dev/null 2>&1; then
  echo "[FAIL] ros2 is not on PATH."
  exit 1
fi

echo "[INFO] ros2 found: $(command -v ros2)"
echo "[INFO] model path: ${MODEL_PATH}"

required_topics=(
  "/hdas/camera_head/left_raw/image_raw_color/compressed"
  "/hdas/camera_wrist_left/color/image_raw/compressed"
  "/hdas/camera_wrist_right/color/image_raw/compressed"
  "/hdas/feedback_arm_left"
  "/hdas/feedback_arm_right"
  "/hdas/feedback_torso"
  "/hdas/feedback_gripper_left"
  "/hdas/feedback_gripper_right"
  "/motion_control/pose_ee_arm_left"
  "/motion_control/pose_ee_arm_right"
)

action_topics=(
  "/motion_target/target_joint_state_arm_left"
  "/motion_target/target_joint_state_arm_right"
  "/motion_target/target_joint_state_torso"
  "/motion_target/target_pose_arm_left"
  "/motion_target/target_pose_arm_right"
  "/motion_target/target_position_gripper_left"
  "/motion_target/target_position_gripper_right"
)

topic_list="$(ros2 topic list || true)"

echo
echo "[CHECK] required observation topics"
missing_topics=0
for topic in "${required_topics[@]}"; do
  if grep -Fxq "${topic}" <<<"${topic_list}"; then
    echo "[OK]   ${topic}"
  else
    echo "[MISS] ${topic}"
    missing_topics=1
  fi
done

echo
echo "[CHECK] action-topic publishers"
for topic in "${action_topics[@]}"; do
  echo "[INFO] ${topic}"
  ros2 topic info -v "${topic}" 2>/dev/null || echo "  no info available"
done

echo
echo "[CHECK] checkpoint files"
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[FAIL] checkpoint directory does not exist"
  exit 1
fi

for file in config.yaml dataset_stats.json; do
  if [[ -f "${MODEL_PATH}/${file}" ]]; then
    echo "[OK]   ${MODEL_PATH}/${file}"
  else
    echo "[MISS] ${MODEL_PATH}/${file}"
  fi
done

if [[ -f "${MODEL_PATH}/model_state_dict.pt" ]]; then
  echo "[OK]   ${MODEL_PATH}/model_state_dict.pt"
elif [[ -f "${MODEL_PATH}/model.pt" ]]; then
  echo "[OK]   ${MODEL_PATH}/model.pt"
elif [[ -f "${MODEL_PATH}/galaxea_zero_encoder_opt.fp16.engine" && -f "${MODEL_PATH}/galaxea_zero_predictor_opt.fp16.engine" ]]; then
  echo "[OK]   ${MODEL_PATH}/galaxea_zero_encoder_opt.fp16.engine"
  echo "[OK]   ${MODEL_PATH}/galaxea_zero_predictor_opt.fp16.engine"
elif [[ -f "${MODEL_PATH}/prefill.fp16.engine" && -f "${MODEL_PATH}/decode.fp16.engine" ]]; then
  echo "[OK]   ${MODEL_PATH}/prefill.fp16.engine"
  echo "[OK]   ${MODEL_PATH}/decode.fp16.engine"
else
  echo "[MISS] no supported model weights or TensorRT engine pair found in ${MODEL_PATH}"
fi

if [[ -f "${MODEL_PATH}/gemma_rmsnorm.so" ]]; then
  echo "[OK]   ${MODEL_PATH}/gemma_rmsnorm.so"
fi

echo
if [[ "${missing_topics}" -ne 0 ]]; then
  echo "[SUMMARY] Some required topics are missing."
  echo "          EFMNode will not run live inference until those topics exist or the topic map is adjusted."
  exit 2
fi

echo "[SUMMARY] Topic and checkpoint preflight completed."
