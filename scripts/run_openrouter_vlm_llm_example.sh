#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-qwenvl}"
API_KEY="${OPENROUTER_API_KEY:-}"

IMAGE_PATH="${1:-/data1/user/ycliu/VLM-planning/resource/real_607/pickup/goose/pickup_white_goose.jpg}"
TASK_INSTRUCTION="${2:-Pick the target object and place it into the target container safely.}"
KEYPOINTS_3D_JSON="${3:-/data1/user/ycliu/VLM-planning/results/virtual_simpleenv_local_ours_20260328/keypoints_3d_moge.json}"
OUT_DIR="${4:-${ROOT_DIR}/results/openrouter_api_example_$(date +%Y%m%d_%H%M%S)}"

echo "[RUN] env=${CONDA_ENV}"
echo "[RUN] image=${IMAGE_PATH}"
echo "[RUN] task=${TASK_INSTRUCTION}"
echo "[RUN] keypoints3d=${KEYPOINTS_3D_JSON}"
echo "[RUN] out=${OUT_DIR}"

conda run --no-capture-output -n "${CONDA_ENV}" \
  python "${ROOT_DIR}/scripts/openrouter_vlm_llm_api_example.py" \
  --image "${IMAGE_PATH}" \
  --task "${TASK_INSTRUCTION}" \
  --keypoints-3d-json "${KEYPOINTS_3D_JSON}" \
  --out-dir "${OUT_DIR}" \
  --api-key "${API_KEY}"

