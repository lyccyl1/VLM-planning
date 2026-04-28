#!/bin/bash

# Optional distributed-style placeholders (kept for consistency with your templates)
num_nodes=1
num_gpu_per_node=1
master_addr=gpu02
master_port=6266
node_rank=0

# HTML generation does not require GPU, keep this only if your env expects it
export CUDA_VISIBLE_DEVICES=6

trace_json=/data1/user/ycliu/VLM-planning/results/pick_up_goose_gt_intrinsics/trace_qwen3.5_plus.json
keypoints_3d_json=/data1/user/ycliu/VLM-planning/results/pick_up_goose_gt_intrinsics/keypoints_3d_gt_npy_intrinsics.json

out_html=/data1/user/ycliu/VLM-planning/results/pick_up_goose_gt_intrinsics/trajectory_validation_3d.html
out_report_json=/data1/user/ycliu/VLM-planning/results/pick_up_goose_gt_intrinsics/trajectory_validation_report_from_module.json

log_file=/data1/user/ycliu/VLM-planning/results/pick_up_goose_gt_intrinsics/generate_trajectory_html.log

nohup python /data1/user/ycliu/VLM-planning/generate_trajectory_html.py \
  --trace-json "${trace_json}" \
  --keypoints-3d-json "${keypoints_3d_json}" \
  --out-html "${out_html}" \
  --out-report-json "${out_report_json}" \
  > "${log_file}" 2>&1 &

pid=$!
echo "[LAUNCHED] PID=${pid}"
echo "[LOG] ${log_file}"
echo "[OUT_HTML] ${out_html}"
echo "[OUT_REPORT] ${out_report_json}"
