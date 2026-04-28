#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_PATH="${1:-/data1/user/ycliu/VLM-planning/resource/real_607/pickup/goose/pickup_white_goose.jpg}"
KEYPOINTS_PATH="${2:-/data1/user/ycliu/1.py}"
OUT_DIR="${3:-${ROOT_DIR}/results/test_pipeline_nosam2_$(date +%Y%m%d_%H%M%S)}"
RESIZE_WIDTH="${4:-690}"
RESIZE_HEIGHT="${5:-479}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
DEPTH_SOURCE="${DEPTH_SOURCE:-moge}"
GT_DEPTH_NPY="${GT_DEPTH_NPY:-}"
CAMERA_FX="${CAMERA_FX:-}"
CAMERA_FY="${CAMERA_FY:-}"
CAMERA_CX="${CAMERA_CX:-}"
CAMERA_CY="${CAMERA_CY:-}"

mkdir -p "${OUT_DIR}"

KEYPOINTS_ARG="${KEYPOINTS_PATH}"

# Support /data1/user/ycliu/1.py styles:
# 1) points = [(x, y, "label"), ...]
# 2) points = [(x, y), ...] + labels = ["...", ...]
if [[ "${KEYPOINTS_PATH}" == *.py ]]; then
  CONVERTED_JSON="${OUT_DIR}/keypoints_from_points_format.json"
  if "${PYTHON_BIN}" - "${KEYPOINTS_PATH}" "${CONVERTED_JSON}" <<'PY'
import importlib.util
import json
import pathlib
import sys

py_path = pathlib.Path(sys.argv[1])
out_path = pathlib.Path(sys.argv[2])

spec = importlib.util.spec_from_file_location("_points_module", str(py_path))
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import: {py_path}")

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)  # type: ignore[union-attr]

pts = getattr(module, "points", None)
if not isinstance(pts, list) or len(pts) == 0:
    raise ValueError("Expected non-empty list variable `points` in keypoint .py")

label_candidates = [
    "labels",
    "label_names",
    "point_labels",
    "semantic_labels",
    "names",
]
labels = None
for name in label_candidates:
    value = getattr(module, name, None)
    if isinstance(value, list) and len(value) >= len(pts):
        labels = value
        break

entries = []
for i, item in enumerate(pts):
    if not isinstance(item, (list, tuple)) or len(item) < 2:
        raise ValueError(f"points[{i}] must have at least (x, y), got: {item!r}")

    x = float(item[0])
    y = float(item[1])

    if len(item) >= 3:
        label = str(item[2])
    elif labels is not None:
        label = str(labels[i])
    else:
        label = f"point_{i + 1}"

    entries.append({"label": label, "point": [x, y]})

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[POINTS-CONVERTED] {out_path}")
PY
  then
    KEYPOINTS_ARG="${CONVERTED_JSON}"
  fi
fi

EXTRA_ARGS=(--depth-source "${DEPTH_SOURCE}")
if [[ -n "${GT_DEPTH_NPY}" ]]; then
  EXTRA_ARGS+=(--gt-depth-npy "${GT_DEPTH_NPY}")
fi
if [[ "${DEPTH_SOURCE}" == "gt_npy_intrinsics" ]]; then
  if [[ -z "${CAMERA_FX}" || -z "${CAMERA_FY}" || -z "${CAMERA_CX}" || -z "${CAMERA_CY}" ]]; then
    echo "[ERR] DEPTH_SOURCE=gt_npy_intrinsics requires CAMERA_FX/CAMERA_FY/CAMERA_CX/CAMERA_CY" >&2
    exit 1
  fi
  EXTRA_ARGS+=(
    --camera-fx "${CAMERA_FX}"
    --camera-fy "${CAMERA_FY}"
    --camera-cx "${CAMERA_CX}"
    --camera-cy "${CAMERA_CY}"
  )
fi

"${PYTHON_BIN}" "${ROOT_DIR}/pipeline_2d_to_3d_sam2_moge.py" \
  --image "${IMAGE_PATH}" \
  --keypoints "${KEYPOINTS_ARG}" \
  --vlm5d-root /data1/user/ycliu/VLM-5d \
  --sam2-model-dir /data1/user/ycliu/VLM-5d/models/sam2 \
  --moge-model /data1/user/ycliu/VLM-5d/models/moge-2/model.pt \
  --point-order xy \
  --point-space pixel \
  --output-point-order xy \
  --output-point-space pixel \
  --resize-width "${RESIZE_WIDTH}" \
  --resize-height "${RESIZE_HEIGHT}" \
  --device "${DEVICE}" \
  "${EXTRA_ARGS[@]}" \
  --out-dir "${OUT_DIR}"

echo "[KEYPOINTS-INPUT] ${KEYPOINTS_PATH}"
echo "[KEYPOINTS-USED] ${KEYPOINTS_ARG}"
echo "[DEPTH-SOURCE] ${DEPTH_SOURCE}"
if [[ -n "${GT_DEPTH_NPY}" ]]; then
  echo "[GT-DEPTH-NPY] ${GT_DEPTH_NPY}"
fi
if [[ "${DEPTH_SOURCE}" == "gt_npy_intrinsics" ]]; then
  echo "[CAMERA] fx=${CAMERA_FX}, fy=${CAMERA_FY}, cx=${CAMERA_CX}, cy=${CAMERA_CY}"
fi
echo "[TEST-PIPELINE-OUT] ${OUT_DIR}"
echo "[2D-OVERLAY-ORIG] ${OUT_DIR}/keypoints_on_original_image.png"
echo "[3D-RECON] ${OUT_DIR}/keypoints_3d_reconstruction.png"
