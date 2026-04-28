#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROBOTWIN_DIR="${ROOT_DIR}/environment/RoboTwin"
CONDA_ENV="${CONDA_ENV:-robotwin}"

TASK="${1:-place_a2b_right}"
MAX_TRIES="${2:-1}"
SEED_START="${3:-0}"
OUTPUT_JSON="${4:-${ROBOTWIN_DIR}/test_results/${TASK}_vlm_policy_one_round_$(date +%Y%m%d_%H%M%S).json}"

TASK_CONFIG="${TASK_CONFIG:-task_config/demo_clean_smoke1_nowrist.yml}"
POLICY_CONFIG="${POLICY_CONFIG:-policy/Your_Policy/deploy_policy.yml}"

cd "${ROBOTWIN_DIR}"

echo "[RUN] conda_env=${CONDA_ENV}"
echo "[RUN] task=${TASK}"
echo "[RUN] task_config=${TASK_CONFIG}"
echo "[RUN] policy_config=${POLICY_CONFIG}"
echo "[RUN] max_tries=${MAX_TRIES}, seed_start=${SEED_START}"
echo "[RUN] output=${OUTPUT_JSON}"

# instruction留空时，run_vlm_policy_one_round_test.py 会自动读取
# description/task_instruction/<task>.json 里的 full_description。
conda run --no-capture-output -n "${CONDA_ENV}" \
  python script/run_vlm_policy_one_round_test.py \
  --task "${TASK}" \
  --config "${TASK_CONFIG}" \
  --policy-config "${POLICY_CONFIG}" \
  --instruction "" \
  --max-tries "${MAX_TRIES}" \
  --seed-start "${SEED_START}" \
  --output "${OUTPUT_JSON}"

echo "[OK] result_json=${OUTPUT_JSON}"
