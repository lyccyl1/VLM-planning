#!bin/bash
num_nodes=1
num_gpu_per_node=1
master_addr=gpu02
master_port=6222
node_rank=0
export CUDA_VISIBLE_DEVICES=2
torchrun --nnodes ${num_nodes} \
  --nproc_per_node ${num_gpu_per_node}\
  --node_rank ${node_rank} \
  --master_addr ${master_addr} \
  --master_port ${master_port} /data1/user/ycliu/VLM-Planner/pipeline_2d_to_3d_sam2_moge.py \
    --image /data1/user/ycliu/closebox.png \
    --keypoints /data1/user/ycliu/VLM-Planner/vlm_2d_points/closebox_chatgpt.json \
    --point-order xy \
    --point-space norm1000 \
    --device cuda \
    --out-dir /data1/user/ycliu/VLM-Planner/results/user_2d_to_3d
      # > /data1/user/ycliu/MTLoRA/eval-base-new3.log 2>&1 &