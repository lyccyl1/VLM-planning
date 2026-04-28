#!/usr/bin/env bash
set -euo pipefail

cd /data1/user/ycliu/VLM-planning
python pipeline_2d_to_3d_sam2_moge.py \
  --image /data1/user/ycliu/VLM-planning/resource/real_607/pickup/lid/frame_0001_20260316_142614_814_color.jpg\
  --keypoints /data1/user/ycliu/1.py \
  --depth-source gt_npy_intrinsics \
  --gt-depth-npy /data1/user/ycliu/VLM-planning/resource/real_607/pickup/lid/frame_0001_20260316_142614_814_color.jpg \
  --camera-fx 893.822509765625 \
  --camera-fy 893.6419067382812 \
  --camera-cx 649.1566162109375 \
  --camera-cy 372.6593933105469 \
  --disable-sam2 \
  --out-dir /data1/user/ycliu/VLM-planning/results/pick_up_goose_gt_intrinsics
