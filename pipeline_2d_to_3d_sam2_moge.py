#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _ensure_vlm5d_on_path(vlm5d_root: Path) -> None:
    if not vlm5d_root.exists():
        raise FileNotFoundError(f"VLM-5d root not found: {vlm5d_root}")
    root_str = str(vlm5d_root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _read_rgb_image(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _read_depth_npy(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Depth .npy not found: {path}")
    depth = np.load(path)
    if depth.ndim != 2:
        raise ValueError(f"Depth map must be 2D [H,W], got {depth.shape}")
    return depth.astype(np.float32, copy=False)


def _resize_depth_to_resolution(
    depth: np.ndarray,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    if depth.ndim != 2:
        raise ValueError(f"Expected depth map [H,W], got {depth.shape}")
    h, w = depth.shape
    if target_width <= 0 or target_height <= 0:
        return depth.astype(np.float32, copy=False)
    tw = int(target_width)
    th = int(target_height)
    if (w, h) == (tw, th):
        return depth.astype(np.float32, copy=False)
    resized = np.asarray(
        Image.fromarray(depth.astype(np.float32), mode="F").resize((tw, th), resample=Image.BILINEAR),
        dtype=np.float32,
    )
    return resized


def _infer_default_gt_depth_path(image_path: Path) -> Path | None:
    name = image_path.name
    candidates: List[Path] = []
    if "_color." in name:
        candidates.append(image_path.with_name(name.replace("_color.", "_depth.npy")))
    candidates.append(image_path.with_suffix(".npy"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_gt_depth_for_processing(
    image_path: Path,
    gt_depth_npy: Path | None,
    target_width: int,
    target_height: int,
) -> Tuple[np.ndarray, Path]:
    gt_depth_path = gt_depth_npy
    if gt_depth_path is None:
        gt_depth_path = _infer_default_gt_depth_path(image_path)
        if gt_depth_path is None:
            raise ValueError(
                "Using GT depth requires --gt-depth-npy, or an image path that can auto-resolve "
                "to '*_depth.npy' (for example '*_color.jpg' -> '*_depth.npy')."
            )
    gt_depth_raw = _read_depth_npy(gt_depth_path)
    gt_depth_resized = _resize_depth_to_resolution(
        depth=gt_depth_raw,
        target_width=target_width,
        target_height=target_height,
    )
    return gt_depth_resized, gt_depth_path


def _resize_image_to_resolution(
    image_rgb: np.ndarray,
    target_width: int,
    target_height: int,
) -> Tuple[np.ndarray, float, float]:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image [H,W,3], got {image_rgb.shape}")

    h, w = image_rgb.shape[:2]
    if target_width <= 0 or target_height <= 0:
        return image_rgb, 1.0, 1.0

    tw = int(target_width)
    th = int(target_height)
    if (w, h) == (tw, th):
        return image_rgb, 1.0, 1.0

    resized = np.asarray(
        Image.fromarray(image_rgb).resize((tw, th), resample=Image.BILINEAR),
        dtype=np.uint8,
    )
    scale_x = float(tw) / float(max(1, w))
    scale_y = float(th) / float(max(1, h))
    return resized, scale_x, scale_y


def _scale_keypoints(keypoints: Sequence[Any], scale_x: float, scale_y: float, keypoint_cls: Any) -> List[Any]:
    if abs(scale_x - 1.0) < 1e-9 and abs(scale_y - 1.0) < 1e-9:
        return list(keypoints)

    out: List[Any] = []
    for kp in keypoints:
        out.append(
            keypoint_cls(
                label=kp.label,
                x=float(kp.x) * float(scale_x),
                y=float(kp.y) * float(scale_y),
            )
        )
    return out


def _coerce_xy(value: Any) -> Tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    raise ValueError(f"Invalid point value: {value!r}")


def _normalize_keypoint_entries(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, dict):
        if "keypoints" in raw:
            return _normalize_keypoint_entries(raw["keypoints"])
        out = []
        for label, point in raw.items():
            x, y = _coerce_xy(point)
            out.append({"label": str(label), "point": [x, y]})
        if not out:
            raise ValueError("Keypoint mapping is empty.")
        return out

    if not isinstance(raw, list):
        raise ValueError("Keypoints must be a list or dict.")

    out: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw):
        if isinstance(item, dict):
            if "label" not in item:
                raise ValueError(f"Keypoint[{idx}] missing 'label'.")

            if "point" in item:
                x, y = _coerce_xy(item["point"])
            elif "xy" in item:
                x, y = _coerce_xy(item["xy"])
            elif "x" in item and "y" in item:
                x, y = float(item["x"]), float(item["y"])
            else:
                raise ValueError(
                    f"Keypoint[{idx}] missing coordinates: expected 'point', 'xy', or 'x'+'y'."
                )
            out.append({"label": str(item["label"]), "point": [x, y]})
            continue

        if isinstance(item, (list, tuple)) and len(item) >= 2:
            # Support tuple/list formats:
            # 1) (x, y)
            # 2) (x, y, label)
            # 3) (label, x, y)
            if isinstance(item[0], (int, float)) and isinstance(item[1], (int, float)):
                x, y = float(item[0]), float(item[1])
                label = str(item[2]) if len(item) >= 3 else f"point_{idx + 1}"
                out.append({"label": label, "point": [x, y]})
                continue
            if len(item) >= 3 and isinstance(item[0], str) and isinstance(item[1], (int, float)) and isinstance(item[2], (int, float)):
                label = str(item[0])
                x, y = float(item[1]), float(item[2])
                out.append({"label": label, "point": [x, y]})
                continue

        raise ValueError(
            f"Keypoint[{idx}] invalid entry. Expected dict or tuple/list in (x,y,label)/(label,x,y), got: {item!r}"
        )

    if not out:
        raise ValueError("Keypoint list is empty.")
    return out


def _load_keypoint_entries_from_python_module(py_path: Path) -> List[Dict[str, Any]]:
    if not py_path.exists():
        raise FileNotFoundError(f"Keypoint code file not found: {py_path}")

    module_name = f"_kp_module_{abs(hash(str(py_path.resolve())))}"
    spec = importlib.util.spec_from_file_location(module_name, str(py_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import keypoint code file: {py_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    for attr in ("KEYPOINTS", "keypoints", "points"):
        if hasattr(module, attr):
            return _normalize_keypoint_entries(getattr(module, attr))

    for fn_name in ("get_keypoints", "build_keypoints"):
        fn = getattr(module, fn_name, None)
        if callable(fn):
            return _normalize_keypoint_entries(fn())

    raise ValueError(
        f"Cannot find keypoints in {py_path}. Expected KEYPOINTS/keypoints/points or get_keypoints()/build_keypoints()."
    )

def nearest_point_in_mask(mask: np.ndarray, point_xy: Tuple[float, float]) -> Tuple[float, float]:
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D, got shape={mask.shape}")
    x, y = float(point_xy[0]), float(point_xy[1])
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return x, y
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    d2 = (pts[:, 0] - x) ** 2 + (pts[:, 1] - y) ** 2
    best = pts[int(np.argmin(d2))]
    return float(best[0]), float(best[1])


def _match_keypoints_into_mask(
    image_rgb: np.ndarray,
    keypoints: Sequence[Any],
    sam2_model_dir: Path,
    device: str,
    multimask_output: bool,
    load_predictor_fn: Any,
    matched_cls: Any,
) -> List[Any]:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required for SAM-2 matching.") from exc

    predictor = load_predictor_fn(model_dir=sam2_model_dir, device=device)
    predictor.set_image(image_rgb)
    h, w = image_rgb.shape[:2]

    use_amp = device.startswith("cuda") and torch.cuda.is_available()
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_amp
        else contextlib.nullcontext()
    )

    out: List[Any] = []
    with torch.inference_mode(), amp_ctx:
        for kp in keypoints:
            point_coords = np.array([[kp.x, kp.y]], dtype=np.float32)
            point_labels = np.array([1], dtype=np.int32)
            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=multimask_output,
            )
            if masks is None or len(masks) == 0:
                matched_x, matched_y = float(kp.x), float(kp.y)
                area = 0
                score = 0.0
                inside = False
            else:
                best_idx = int(np.argmax(scores))
                score = float(scores[best_idx])
                mask = np.asarray(masks[best_idx], dtype=np.uint8)
                area = int(mask.sum())

                ix = int(round(_clamp(kp.x, 0.0, float(max(0, w - 1)))))
                iy = int(round(_clamp(kp.y, 0.0, float(max(0, h - 1)))))
                inside = bool(mask[iy, ix] > 0)
                if inside:
                    matched_x, matched_y = float(kp.x), float(kp.y)
                else:
                    matched_x, matched_y = nearest_point_in_mask(mask, point_xy=(kp.x, kp.y))

            dist = math.hypot(matched_x - kp.x, matched_y - kp.y)
            out.append(
                matched_cls(
                    label=kp.label,
                    input_x=float(kp.x),
                    input_y=float(kp.y),
                    matched_x=float(matched_x),
                    matched_y=float(matched_y),
                    distance_px=float(dist),
                    mask_area_px=int(area),
                    sam_score=float(score),
                    input_inside_mask=inside,
                )
            )
    return out


def _new_pixel_exact_canvas(image: np.ndarray) -> Tuple[Any, Any, int]:
    h, w = image.shape[:2]
    dpi = 100
    fig = plt.figure(figsize=(float(w) / dpi, float(h) / dpi), dpi=dpi)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.axis("off")
    return fig, ax, dpi


def _build_identity_matches(keypoints: Sequence[Any], matched_cls: Any) -> List[Any]:
    out: List[Any] = []
    for kp in keypoints:
        out.append(
            matched_cls(
                label=kp.label,
                input_x=float(kp.x),
                input_y=float(kp.y),
                matched_x=float(kp.x),
                matched_y=float(kp.y),
                distance_px=0.0,
                mask_area_px=0,
                sam_score=0.0,
                input_inside_mask=True,
            )
        )
    return out


def _draw_input_points(image_rgb: np.ndarray, keypoints: Sequence[Any], save_path: Path, title: str) -> None:
    fig, ax, dpi = _new_pixel_exact_canvas(image_rgb)
    ax.imshow(image_rgb)
    xs = [float(kp.x) for kp in keypoints]
    ys = [float(kp.y) for kp in keypoints]
    ax.scatter(xs, ys, c="#ff2d55", s=28, edgecolors="white", linewidths=0.7)
    for idx, kp in enumerate(keypoints, start=1):
        ax.text(
            float(kp.x) + 6,
            float(kp.y) - 6,
            f"{idx}. {kp.label}",
            fontsize=7,
            color="#ffe082",
            bbox={"facecolor": "black", "alpha": 0.45, "pad": 1.0},
        )
    ax.text(
        0.01,
        0.99,
        title,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.5, "pad": 2.0},
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def _draw_2d_correspondence(image_rgb: np.ndarray, matches: Sequence[Any], save_path: Path, title: str) -> None:
    fig, ax, dpi = _new_pixel_exact_canvas(image_rgb)
    ax.imshow(image_rgb)
    for idx, m in enumerate(matches, start=1):
        x0, y0 = float(m.input_x), float(m.input_y)
        x1, y1 = float(m.matched_x), float(m.matched_y)
        ax.plot([x0, x1], [y0, y1], color="#ffd54f", linewidth=1.0)
        ax.scatter([x0], [y0], c="#ff1744", s=22)
        ax.scatter([x1], [y1], c="#00e676", s=22)
        ax.text(
            x1 + 5,
            y1 - 5,
            f"{idx}. {m.label}",
            fontsize=7,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.4, "pad": 1.0},
        )
    ax.text(
        0.01,
        0.99,
        title,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.5, "pad": 2.0},
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def _lift_keypoints_to_3d(
    points_3d_map: np.ndarray,
    matches: Sequence[Any],
    window_size: int,
) -> Dict[str, List[float]]:
    h, w = points_3d_map.shape[:2]
    out: Dict[str, List[float]] = {}
    for m in matches:
        px = int(round(_clamp(float(m.matched_x), 0.0, float(max(0, w - 1)))))
        py = int(round(_clamp(float(m.matched_y), 0.0, float(max(0, h - 1)))))
        x0 = max(0, px - window_size)
        x1 = min(w, px + window_size + 1)
        y0 = max(0, py - window_size)
        y1 = min(h, py + window_size + 1)
        patch = points_3d_map[y0:y1, x0:x1].reshape(-1, 3)
        valid = patch[np.isfinite(patch).all(axis=1)]
        if len(valid) == 0:
            center = points_3d_map[py, px]
            if not np.isfinite(center).all():
                continue
            xyz = center.astype(np.float32)
        else:
            xyz = np.median(valid, axis=0).astype(np.float32)
        out[str(m.label)] = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
    return out


def _merge_moge_xy_with_depth(moge_points_3d_map: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
    if moge_points_3d_map.ndim != 3 or moge_points_3d_map.shape[2] != 3:
        raise ValueError(f"MoGe points map must be [H,W,3], got {moge_points_3d_map.shape}")
    if depth_map.ndim != 2:
        raise ValueError(f"Depth map must be [H,W], got {depth_map.shape}")
    if tuple(moge_points_3d_map.shape[:2]) != tuple(depth_map.shape):
        raise ValueError(
            f"Shape mismatch: MoGe={moge_points_3d_map.shape[:2]}, depth={depth_map.shape}"
        )

    merged = moge_points_3d_map.astype(np.float32, copy=True)
    z_moge = merged[..., 2]
    z_gt = depth_map.astype(np.float32, copy=False)
    valid_gt = np.isfinite(z_gt) & (z_gt > 0.0)
    merged[..., 2] = np.where(valid_gt, z_gt, z_moge)
    return merged


def _project_depth_to_3d_with_intrinsics(
    depth_map: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    if depth_map.ndim != 2:
        raise ValueError(f"Depth map must be [H,W], got {depth_map.shape}")
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError(f"Invalid intrinsics: fx={fx}, fy={fy}. Both must be > 0.")

    z = depth_map.astype(np.float32, copy=False)
    h, w = z.shape
    uu, vv = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
        indexing="xy",
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        x = (uu - float(cx)) * z / float(fx)
        y = (vv - float(cy)) * z / float(fy)

    points = np.stack([x, y, z], axis=-1).astype(np.float32, copy=False)
    valid = np.isfinite(z) & (z > 0.0)
    points[~valid] = np.nan
    return points


def _set_equal_axes(ax: Any, xyz: np.ndarray) -> None:
    if xyz.size == 0:
        return
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = (mins + maxs) / 2.0
    span = float((maxs - mins).max())
    half = max(span * 0.5, 1e-6)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)


def _draw_3d_points(keypoints_3d: Dict[str, List[float]], save_path: Path) -> None:
    if not keypoints_3d:
        raise RuntimeError("No valid 3D keypoints were reconstructed.")
    labels = list(keypoints_3d.keys())
    xyz = np.array([keypoints_3d[k] for k in labels], dtype=np.float32)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=xyz[:, 2], cmap="viridis", s=38)
    for idx, label in enumerate(labels, start=1):
        ax.text(
            float(xyz[idx - 1, 0]),
            float(xyz[idx - 1, 1]),
            float(xyz[idx - 1, 2]),
            f"{idx}.{label}",
            fontsize=7,
        )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Reconstructed Keypoints")
    _set_equal_axes(ax, xyz)
    fig.colorbar(sc, ax=ax, shrink=0.65, label="Z")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=240)
    plt.close(fig)


def _draw_depth_preview(points_3d_map: np.ndarray, matches: Sequence[Any], save_path: Path, title: str) -> None:
    depth = points_3d_map[..., 2].astype(np.float32)
    valid = np.isfinite(depth)
    if not valid.any():
        raise RuntimeError("Depth map has no valid values.")
    fill = float(np.median(depth[valid]))
    depth = depth.copy()
    depth[~valid] = fill

    fig, ax, dpi = _new_pixel_exact_canvas(depth)
    ax.imshow(depth, cmap="magma")
    for m in matches:
        ax.scatter([m.matched_x], [m.matched_y], c="#00e676", s=18)
    ax.text(
        0.01,
        0.99,
        title,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.5, "pad": 2.0},
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def _save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _run_moge_points(image_rgb: np.ndarray, model_path: Path, device: str) -> np.ndarray:
    try:
        import torch
        from moge.model.v2 import MoGeModel
    except Exception as exc:
        raise RuntimeError("MoGe dependency missing. Ensure `moge` and `torch` are installed.") from exc

    model = MoGeModel.from_pretrained(str(model_path)).to(device)
    model.eval()
    with torch.no_grad():
        x = torch.tensor(image_rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
        out = model.infer(x)
        points_3d_map = out["points"].detach().cpu().numpy()
    return points_3d_map


def _build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="2D keypoints -> (optional SAM2) -> MoGe 3D reconstruction."
    )
    parser.add_argument("--image", type=Path, default=Path("/data1/user/ycliu/closebox.png"))
    parser.add_argument(
        "--keypoints",
        type=Path,
        default=repo_root / "vlm_2d_points/closebox_gemini.py",
        help="2D keypoints file (.json) or Python file (.py with KEYPOINTS/keypoints/get_keypoints/build_keypoints).",
    )
    parser.add_argument("--vlm5d-root", type=Path, default=Path("/data1/user/ycliu/VLM-5d"))
    parser.add_argument("--sam2-model-dir", type=Path, default=Path("/data1/user/ycliu/VLM-5d/models/sam2"))
    parser.add_argument("--moge-model", type=Path, default=Path("/data1/user/ycliu/VLM-5d/models/moge-2/model.pt"))
    parser.add_argument("--out-dir", type=Path, default=repo_root / "results/user_2d_to_3d")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--point-order", choices=["xy", "yx"], default="xy")
    parser.add_argument("--point-space", choices=["pixel", "norm1000"], default="pixel")
    parser.add_argument("--output-point-order", choices=["xy", "yx"], default="xy")
    parser.add_argument("--output-point-space", choices=["pixel", "norm1000"], default="pixel")
    parser.add_argument("--single-mask", action="store_true", help="Use only one SAM2 mask candidate.")
    parser.add_argument("--window-size", type=int, default=5, help="Window radius for robust 3D sampling.")
    parser.add_argument("--disable-sam2", action="store_true", help="Skip SAM2 and use input keypoints directly.")
    parser.add_argument(
        "--depth-source",
        choices=["moge", "gt_npy", "gt_npy_moge_xy", "gt_npy_intrinsics"],
        default="moge",
        help=(
            "Depth/3D source: 'moge'=MoGe XYZ; 'gt_npy' or 'gt_npy_moge_xy'=GT depth as Z + MoGe X,Y; "
            "'gt_npy_intrinsics'=project GT depth to XYZ using camera intrinsics."
        ),
    )
    parser.add_argument(
        "--gt-depth-npy",
        type=Path,
        default=None,
        help="Ground-truth depth .npy path used in GT depth modes. "
        "If omitted, auto-tries replacing '_color.*' with '_depth.npy'.",
    )
    parser.add_argument("--camera-fx", type=float, default=None, help="Camera fx for --depth-source=gt_npy_intrinsics.")
    parser.add_argument("--camera-fy", type=float, default=None, help="Camera fy for --depth-source=gt_npy_intrinsics.")
    parser.add_argument("--camera-cx", type=float, default=None, help="Camera cx for --depth-source=gt_npy_intrinsics.")
    parser.add_argument("--camera-cy", type=float, default=None, help="Camera cy for --depth-source=gt_npy_intrinsics.")
    parser.add_argument("--resize-width", type=int, default=1280, help="Target width for processing image.")
    parser.add_argument("--resize-height", type=int, default=720, help="Target height for processing image.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    _ensure_vlm5d_on_path(args.vlm5d_root)
    out_dir = args.out_dir

    from perception.sam2_keypoint_matching import (  # pylint: disable=import-error
        Keypoint2D,
        MatchedKeypoint,
        _load_sam2_predictor,
        load_labeled_keypoints,
        save_match_overlay,
        serialize_matched_keypoints,
    )

    image_rgb_orig = _read_rgb_image(args.image)
    orig_h, orig_w = image_rgb_orig.shape[:2]

    keypoint_json_path = args.keypoints
    converted_keypoint_json: Path | None = None
    if args.keypoints.suffix.lower() == ".py":
        entries = _load_keypoint_entries_from_python_module(args.keypoints)
        converted_keypoint_json = out_dir / "keypoints_from_code.json"
        _save_json(entries, converted_keypoint_json)
        keypoint_json_path = converted_keypoint_json

    keypoints_orig = load_labeled_keypoints(
        json_path=keypoint_json_path,
        point_order=args.point_order,
        point_space=args.point_space,
        image_shape=(orig_h, orig_w),
    )
    if not keypoints_orig:
        raise RuntimeError(f"No valid keypoints parsed from: {keypoint_json_path}")

    image_rgb, scale_x, scale_y = _resize_image_to_resolution(
        image_rgb=image_rgb_orig,
        target_width=int(args.resize_width),
        target_height=int(args.resize_height),
    )
    h, w = image_rgb.shape[:2]
    keypoints_proc = _scale_keypoints(
        keypoints=keypoints_orig,
        scale_x=scale_x,
        scale_y=scale_y,
        keypoint_cls=Keypoint2D,
    )

    if args.disable_sam2:
        matches = _build_identity_matches(keypoints=keypoints_proc, matched_cls=MatchedKeypoint)
    else:
        matches = _match_keypoints_into_mask(
            image_rgb=image_rgb,
            keypoints=keypoints_proc,
            sam2_model_dir=args.sam2_model_dir,
            device=args.device,
            multimask_output=not args.single_mask,
            load_predictor_fn=_load_sam2_predictor,
            matched_cls=MatchedKeypoint,
        )

    payload = serialize_matched_keypoints(
        matches=matches,
        output_point_order=args.output_point_order,
        output_point_space=args.output_point_space,
        image_shape=(h, w),
    )

    depth_source = str(args.depth_source)
    if depth_source == "gt_npy":
        depth_source = "gt_npy_moge_xy"

    points_3d_map: np.ndarray
    depth_preview_title: str
    used_gt_depth_path: Path | None = None
    intrinsic_info: str | None = None

    if depth_source == "moge":
        points_3d_map = _run_moge_points(image_rgb=image_rgb, model_path=args.moge_model, device=args.device)
        depth_preview_title = "MoGe Depth (Z) with Keypoints"
    elif depth_source == "gt_npy_moge_xy":
        gt_depth_resized, gt_depth_path = _resolve_gt_depth_for_processing(
            image_path=args.image,
            gt_depth_npy=args.gt_depth_npy,
            target_width=w,
            target_height=h,
        )
        moge_points_3d_map = _run_moge_points(image_rgb=image_rgb, model_path=args.moge_model, device=args.device)
        points_3d_map = _merge_moge_xy_with_depth(
            moge_points_3d_map=moge_points_3d_map,
            depth_map=gt_depth_resized,
        )
        depth_preview_title = "GT Depth (Z) + MoGe X,Y with Keypoints"
        used_gt_depth_path = gt_depth_path
    elif depth_source == "gt_npy_intrinsics":
        missing_intrinsics = [
            name
            for name, value in (
                ("camera_fx", args.camera_fx),
                ("camera_fy", args.camera_fy),
                ("camera_cx", args.camera_cx),
                ("camera_cy", args.camera_cy),
            )
            if value is None
        ]
        if missing_intrinsics:
            raise ValueError(
                "depth-source=gt_npy_intrinsics requires --camera-fx/--camera-fy/--camera-cx/--camera-cy; "
                f"missing: {', '.join(missing_intrinsics)}"
            )

        gt_depth_resized, gt_depth_path = _resolve_gt_depth_for_processing(
            image_path=args.image,
            gt_depth_npy=args.gt_depth_npy,
            target_width=w,
            target_height=h,
        )

        fx = float(args.camera_fx) * float(scale_x)
        fy = float(args.camera_fy) * float(scale_y)
        cx = float(args.camera_cx) * float(scale_x)
        cy = float(args.camera_cy) * float(scale_y)

        points_3d_map = _project_depth_to_3d_with_intrinsics(
            depth_map=gt_depth_resized,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
        )
        depth_preview_title = "GT Depth + Camera Intrinsics -> 3D with Keypoints"
        used_gt_depth_path = gt_depth_path
        intrinsic_info = f"fx={fx:.6f}, fy={fy:.6f}, cx={cx:.6f}, cy={cy:.6f}"
    else:
        raise ValueError(f"Unsupported depth source: {depth_source}")

    keypoints_3d = _lift_keypoints_to_3d(
        points_3d_map=points_3d_map,
        matches=matches,
        window_size=max(0, int(args.window_size)),
    )

    full_json = out_dir / "sam2_matched_keypoints_full.json"
    refined_json = out_dir / "sam2_refined_keypoints.json"
    keypoints_on_original_png = out_dir / "keypoints_on_original_image.png"
    keypoints_on_image_png = out_dir / "keypoints_on_image.png"
    corr_png = out_dir / "keypoints_2d_correspondence.png"
    overlay_png = out_dir / "sam2_overlay.png"
    if depth_source == "moge":
        depth_png = out_dir / "moge_depth_with_keypoints.png"
        keypoints_3d_json = out_dir / "keypoints_3d_moge.json"
    else:
        depth_png = out_dir / f"{depth_source}_depth_with_keypoints.png"
        keypoints_3d_json = out_dir / f"keypoints_3d_{depth_source}.json"
    keypoints_3d_png = out_dir / "keypoints_3d_reconstruction.png"

    _save_json(payload, full_json)
    _save_json(payload["refined_keypoints"], refined_json)
    _save_json(keypoints_3d, keypoints_3d_json)

    _draw_input_points(
        image_rgb=image_rgb_orig,
        keypoints=keypoints_orig,
        save_path=keypoints_on_original_png,
        title="Input Keypoints Over Original Image",
    )
    _draw_input_points(
        image_rgb=image_rgb,
        keypoints=keypoints_proc,
        save_path=keypoints_on_image_png,
        title=f"Processing Image ({w}x{h}) with Keypoints",
    )
    _draw_2d_correspondence(
        image_rgb=image_rgb,
        matches=matches,
        save_path=corr_png,
        title="2D Keypoint Correspondence (Input -> Used)",
    )

    if not args.disable_sam2:
        save_match_overlay(image_rgb=image_rgb, matches=matches, save_path=overlay_png)

    _draw_depth_preview(
        points_3d_map=points_3d_map,
        matches=matches,
        save_path=depth_png,
        title=depth_preview_title,
    )
    _draw_3d_points(keypoints_3d=keypoints_3d, save_path=keypoints_3d_png)

    mean_shift = float(np.mean([m.distance_px for m in matches])) if matches else 0.0
    max_shift = float(np.max([m.distance_px for m in matches])) if matches else 0.0
    mode = "direct-keypoints" if args.disable_sam2 else "sam2-refined"
    print(
        f"[OK] mode={mode}, depth_source={depth_source}, resolution={w}x{h}, "
        f"keypoints={len(matches)}, mean_shift_px={mean_shift:.4f}, max_shift_px={max_shift:.4f}"
    )
    if used_gt_depth_path is not None:
        print(f"[INFO] gt_depth_npy={used_gt_depth_path}")
    if intrinsic_info is not None:
        print(f"[INFO] scaled_intrinsics={intrinsic_info}")
    print(f"[OUT] {full_json}")
    print(f"[OUT] {refined_json}")
    print(f"[OUT] {keypoints_on_original_png}")
    print(f"[OUT] {keypoints_on_image_png}")
    print(f"[OUT] {corr_png}")
    if not args.disable_sam2:
        print(f"[OUT] {overlay_png}")
    print(f"[OUT] {depth_png}")
    print(f"[OUT] {keypoints_3d_json}")
    print(f"[OUT] {keypoints_3d_png}")
    if converted_keypoint_json is not None:
        print(f"[OUT] {converted_keypoint_json}")


if __name__ == "__main__":
    main()
