#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-qwenvl}"
export DISPLAY=""

IMAGE_PATH="${1:-/data1/user/ycliu/VLM-planning/resource/real_607/pickup/goose/pickup_white_goose.jpg}"
TASK_INSTRUCTION="${2:-Pick the target object and place it into the target container safely.}"
OUT_DIR="${3:-${ROOT_DIR}/results/virtual_simpleenv_$(date +%Y%m%d_%H%M%S)}"

echo "[RUN] env=${CONDA_ENV}"
echo "[RUN] image=${IMAGE_PATH}"
echo "[RUN] task=${TASK_INSTRUCTION}"
echo "[RUN] out=${OUT_DIR}"

conda run --no-capture-output -n "${CONDA_ENV}" \
  python "${ROOT_DIR}/virtual_simpleenv_pipeline.py" \
  --image "${IMAGE_PATH}" \
  --task-instruction "${TASK_INSTRUCTION}" \
  --out-dir "${OUT_DIR}" \
  --pipeline-option ours_vlm \
  --vlm-backend local_qwen_vl \
  --planner-backend local_qwen3_instruct \
  --qwen-model /data1/user/ycliu/VLM-5d/models/qwen3-vl-8b \
  --planner-model /data1/user/ycliu/VLM-5d/models/qwen3-8b-instruct \
  --vlm5d-root /data1/user/ycliu/VLM-5d \
  --sam2-model-dir /data1/user/ycliu/VLM-5d/models/sam2 \
  --moge-model /data1/user/ycliu/VLM-5d/models/moge-2 \
  --device cuda \
  --skip-validation
