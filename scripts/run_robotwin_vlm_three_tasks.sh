#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ONE="${ROOT_DIR}/scripts/run_robotwin_vlm_task.sh"
STAMP="$(date +%Y%m%d_%H%M%S)"

MAX_TRIES="${1:-1}"
SEED_START="${2:-0}"

TASKS=(
  "move_pillbottle_pad"
  "place_a2b_right"
  "place_object_stand"
)

for task in "${TASKS[@]}"; do
  out_json="${ROOT_DIR}/environment/RoboTwin/test_results/${task}_vlm_policy_one_round_${STAMP}.json"
  "${RUN_ONE}" "${task}" "${MAX_TRIES}" "${SEED_START}" "${out_json}"
done

echo "[OK] all tasks finished."
