# VLM-planning

面向 RoboTwin 的 VLM/LLM 闭环操作测试仓库。
当前主流程已切到 OpenRouter + Qwen-VL-Max，核心目标是把 2D 关键点与原图严格对齐后，再映射到 3D 并生成可执行 6D 轨迹。

## 当前闭环 Pipeline（执行链路）

1. VLM 先在整图上给目标物体框（bbox）。
2. 对 bbox 区域做局部裁剪，再次让 VLM 预测角点/关键点。
3. 把局部坐标严格映射回原图坐标，保证 2D 点和原图一一对应。
4. 结合深度信息与分割结果（配置里 `sam3_model_dir`，当前路径使用 SAM 系列模型目录）将 2D 点映射到 3D 点。
5. LLM 基于 3D 关键点规划 6D 末端轨迹。
6. RoboTwin 执行轨迹并输出视频、调试目录与 HTML 可视化。

## 你当前使用的默认配置

RoboTwin policy 配置文件：
- `environment/RoboTwin/policy/Your_Policy/deploy_policy.openrouter_qwenvlmax_autorun_tmp.yml`

关键项（已在该配置中）：
- `method: vlm_openrouter`
- `llm_backend: openrouter`
- `openrouter_model: qwen/qwen-vl-max`
- `openrouter_vlm_model: qwen/qwen-vl-max`
- `vlm_preprocess_strategy: bbox_crop_corners`
- `vlm_crop_box_source: vlm_then_sam`
- `llm_prompt_file: /data1/user/ycliu/WORKSPACE/prompt_dsr1.log`
- `qwen_api_key_file: /data1/user/ycliu/WORKSPACE/key.md`

## 快速开始（按任务名 + 轮次重复测试）

> 这是当前推荐入口：只改任务名和轮次即可复用整套流程。

```bash
cd /data1/user/ycliu/VLM-planning/environment/RoboTwin

bash script/run_vlm_algo_loop.sh \
  --task stack_blocks_two \
  --rounds 3 \
  --seed-start 0 \
  --seed-step 1 \
  --config task_config/demo_clean_smoke1_nowrist.yml \
  --policy-config policy/Your_Policy/deploy_policy.openrouter_qwenvlmax_autorun_tmp.yml \
  --conda-env robotwin
```

切任务示例：

```bash
bash script/run_vlm_algo_loop.sh --task move_can_pot --rounds 5 --seed-start 0 --seed-step 1
```

## 产物路径（每轮保留视频 + HTML）

`run_vlm_algo_loop.sh` 每次会生成：
- 汇总目录：`environment/RoboTwin/test_results/loop_<task>_<timestamp>/`
- 汇总表：`summary.tsv`
- 每轮结果 JSON：`round_<i>_seed_<s>.json`

每轮详细调试目录在结果 JSON 的 `attempts[0].debug_dir` 字段里，典型文件包括：
- `episode0.mp4`（机械臂执行视频）
- `plan_exec_visualization.html`（轨迹可视化）
- `trajectory_6d_executable.json`
- `trajectory_6d_llm_raw.json`
- `vlm_raw_response.txt`
- `llm_raw_response.txt`
- `keypoint_quality_attempts.json`

另外，one-round 视频会存到：
- `environment/RoboTwin/eval_result/one_round_videos/<task>/seed_<seed>_<timestamp>/episode0.mp4`

## 单轮测试脚本（底层入口）

`run_vlm_algo_loop.sh` 内部调用：
- `environment/RoboTwin/script/run_vlm_policy_one_round_test.py`

常用参数：

```bash
cd /data1/user/ycliu/VLM-planning/environment/RoboTwin

conda run -n robotwin python script/run_vlm_policy_one_round_test.py \
  --task stack_blocks_two \
  --config task_config/demo_clean_smoke1_nowrist.yml \
  --policy-config policy/Your_Policy/deploy_policy.openrouter_qwenvlmax_autorun_tmp.yml \
  --max-tries 1 \
  --seed-start 0 \
  --output test_results/stack_blocks_two_vlm_policy_result.json
```

说明：
- 若不传 `--instruction`，会自动读取 `description/task_instruction/<task>.json` 的 `full_description`。
- 脚本会自动尝试从环境变量、policy config、`/data1/user/ycliu/WORKSPACE/key.md` 读取 OpenRouter key。

## 常见任务名

可用任务定义位于：
- `environment/RoboTwin/description/task_instruction/`

例如：
- `stack_blocks_two`
- `stack_blocks_three`
- `move_can_pot`
- `place_phone_stand`
- `place_object_stand`

## 结果判读建议（执行与规划是否对齐）

建议按以下顺序核对：
1. 看 `trajectory_6d_executable.json` 中末端姿态（四元数/欧拉）是否符合预期抓取方向。
2. 对照 `episode0.mp4` 判断实际执行姿态是否与规划一致。
3. 若不一致，优先看：
   - `plan_exec_visualization.html`
   - `llm_raw_response.txt`
   - `keypoint_quality_attempts.json`
4. 对 `move_can_pot` 这类抓取方向敏感任务，优先通过 prompt 约束抓取姿态（如要求 top-down grasp / yaw 限制），再复测多 seed。

## 备注

- 本仓库含较多环境与结果数据目录，日常代码提交建议排除 `results/`、`test_results/`、`logs/` 等结果文件。
- 当前根仓库与 `environment/RoboTwin` 是独立 git 仓库结构；更新时请分别确认各自分支与远端。
