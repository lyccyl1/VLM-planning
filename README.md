# VLM-planning

本仓库提供一个面向机器人操作场景的实用型 2D 到 3D 关键点规划流水线。
核心流程包含：

- 从 VLM 生成的 JSON 中读取 2D 关键点。
- 基于 SAM2 的局部关键点掩码匹配与修正。
- 基于 MoGe 的单目 3D 点重建与关键点抬升。
- 通过后处理进行 3D 几何约束对齐（leveling）。

主要用途是把语义化的 2D 关键点转换为更稳定的 3D 规划锚点，供下游机器人规划与控制模块直接使用。

## Current Repo Layout

```text
VLM-planning/
  pipeline_2d_to_3d_sam2_moge.py   # Main 2D -> SAM2 -> 3D pipeline
  rotated_3d_leveling.py           # 3D leveling and constraint projection
  vlm_2d_points/                   # Example VLM keypoint JSON
  results/                         # Example outputs
  tests/                           # Unit tests for core math/utils
  scripts/eval_auto.sh             # Example launcher script
  environment/
    RoboTwin/                      # External environment repo (embedded git repo)
    SimplerEnv/                    # External environment repo (embedded git repo)
```

## Pipeline Overview

### Step 1: 2D keypoint refinement and 3D lifting

`pipeline_2d_to_3d_sam2_moge.py`:

1. Load RGB image and labeled 2D keypoints.
2. Refine each keypoint onto SAM2 mask support.
3. Infer dense 3D points with MoGe.
4. Lift refined 2D keypoints into robust 3D points (median in local window).
5. Save intermediate visualizations and JSON outputs.

### Step 2: 3D leveling constraints

`rotated_3d_leveling.py`:

1. Load 3D keypoints from Step 1.
2. Resolve required semantic groups.
3. Compute rotation that aligns the primary group plane with the horizontal plane.
4. Optionally enforce strict equal-height projection per target group.
5. Save rotated 3D keypoints, report JSON, and visualization.

## Dependencies

### Python packages

Required by scripts:

- `numpy`
- `matplotlib`
- `Pillow`
- `torch`
- `moge` (for `MoGeModel`)

Optional (for tests only):

- standard library `unittest` (already included in Python)

### External code/model dependencies

The main pipeline imports:

- `perception.sam2_keypoint_matching` from a `VLM-5d` checkout (`--vlm5d-root`).
- SAM2 model directory (`--sam2-model-dir`).
- MoGe checkpoint file (`--moge-model`).

Default script paths are absolute and environment-specific; you should override them in your own setup.

## Quick Start

### 1) Run 2D -> 3D reconstruction

```bash
cd /data1/user/ycliu/VLM-planning

python pipeline_2d_to_3d_sam2_moge.py \
  --image /path/to/input_rgb.png \
  --keypoints /path/to/vlm_keypoints.json \
  --vlm5d-root /path/to/VLM-5d \
  --sam2-model-dir /path/to/VLM-5d/models/sam2 \
  --moge-model /path/to/VLM-5d/models/moge-2/model.pt \
  --out-dir /path/to/output_dir \
  --device cuda \
  --point-order xy \
  --point-space norm1000 \
  --output-point-order xy \
  --output-point-space pixel
```

### 2) Run 3D leveling

```bash
cd /data1/user/ycliu/VLM-planning

python rotated_3d_leveling.py \
  --input-json /path/to/output_dir/keypoints_3d_moge.json \
  --out-png /path/to/output_dir/keypoints_3d_rotated.png \
  --out-json /path/to/output_dir/keypoints_3d_rotated.json \
  --report-json /path/to/output_dir/keypoints_3d_rotation_report.json
```

Disable strict equal-height projection:

```bash
python rotated_3d_leveling.py \
  --input-json /path/to/output_dir/keypoints_3d_moge.json \
  --out-png /path/to/output_dir/keypoints_3d_rotated.png \
  --out-json /path/to/output_dir/keypoints_3d_rotated.json \
  --report-json /path/to/output_dir/keypoints_3d_rotation_report.json \
  --no-strict-equal-height
```

## Outputs

After Step 1, the output directory typically contains:

- `sam2_matched_keypoints_full.json`
- `sam2_refined_keypoints.json`
- `keypoints_3d_moge.json`
- `input_keypoints_2d.png`
- `keypoints_2d_correspondence.png`
- `sam2_overlay.png`
- `moge_depth_with_keypoints.png`
- `keypoints_3d_reconstruction.png`

After Step 2:

- `keypoints_3d_rotated.json`
- `keypoints_3d_rotated.png`
- `keypoints_3d_rotation_report.json`

## Key Arguments

### `pipeline_2d_to_3d_sam2_moge.py`

- `--image`: Input RGB image.
- `--keypoints`: Labeled 2D keypoint JSON.
- `--vlm5d-root`: Path to VLM-5d repository (for SAM2 matching module).
- `--sam2-model-dir`: SAM2 checkpoint/config root.
- `--moge-model`: MoGe model checkpoint path.
- `--out-dir`: Output folder.
- `--device`: `cuda` or `cpu`.
- `--point-order`: Input order (`xy` or `yx`).
- `--point-space`: Input coordinate space (`pixel` or `norm1000`).
- `--single-mask`: Disable SAM2 multi-mask selection.
- `--window-size`: Local robust lifting window radius.

### `rotated_3d_leveling.py`

- `--input-json`: 3D keypoint JSON from pipeline step 1.
- `--out-png`: Rotated model visualization.
- `--out-json`: Rotated 3D keypoints.
- `--report-json`: Rotation and leveling report.
- `--no-strict-equal-height`: Keep only rotational alignment without final projection-to-same-z per group.

## Expected 3D label groups

The leveling script requires semantic labels (or configured aliases) for two groups:

- Group 1 (lid plane):
  `lid_top_edge_midpoint`, `lid_surface_center`, `lid_bottom_edge_midpoint`
- Group 2 (box bottom line):
  `box_outer_bottom_front_corner`, `box_outer_bottom_right_corner`

Alias fallbacks are implemented for the two box corner labels.

## Testing

Run unit tests:

```bash
cd /data1/user/ycliu/VLM-planning
python -m unittest tests/test_pipeline_2d_to_3d_utils.py tests/test_rotated_3d_leveling.py
```

Covered checks include:

- nearest point selection inside mask.
- robust local 3D lifting behavior.
- leveling rotation correctness.
- strict equal-height projection behavior.
- rotated figure generation.

## Notes and Known Caveats

- Several defaults in scripts are absolute local paths and may not exist on another machine.
- `scripts/eval_auto.sh` currently references `/data1/user/ycliu/VLM-Planner/...`; update to your current repo path if needed.
- `environment/RoboTwin` and `environment/SimplerEnv` are embedded external repositories, not regular source folders.
- This repo currently tracks generated artifacts in `results/` and bytecode caches in `__pycache__/`; clean them if you want a source-only layout.

## Suggested Next Improvements

- Add a minimal `requirements.txt` for reproducible setup.
- Add one end-to-end smoke test with small input assets.
- Replace absolute defaults with repo-relative defaults and environment variables.
- Add schema validation for keypoint JSON input.
