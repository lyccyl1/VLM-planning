#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
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


def _draw_input_points(image_rgb: np.ndarray, keypoints: Sequence[Any], save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
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
    ax.set_title("Input 2D Keypoints")
    ax.axis("off")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close(fig)


def _draw_2d_correspondence(image_rgb: np.ndarray, matches: Sequence[Any], save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
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
    ax.set_title("2D Keypoint Correspondence (Input -> SAM2 Refined)")
    ax.axis("off")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
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
    ax.set_title("3D Reconstructed Keypoints (MoGe)")
    _set_equal_axes(ax, xyz)
    fig.colorbar(sc, ax=ax, shrink=0.65, label="Z")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=240)
    plt.close(fig)


def _draw_depth_preview(points_3d_map: np.ndarray, matches: Sequence[Any], save_path: Path) -> None:
    depth = points_3d_map[..., 2].astype(np.float32)
    valid = np.isfinite(depth)
    if not valid.any():
        raise RuntimeError("MoGe depth has no valid values.")
    fill = float(np.median(depth[valid]))
    depth = depth.copy()
    depth[~valid] = fill

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(depth, cmap="magma")
    for m in matches:
        ax.scatter([m.matched_x], [m.matched_y], c="#00e676", s=18)
    ax.set_title("MoGe Depth (Z) with Refined Keypoints")
    ax.axis("off")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
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
    parser = argparse.ArgumentParser(
        description="2D keypoints -> SAM2 refinement -> MoGe 3D reconstruction."
    )
    parser.add_argument("--image", type=Path, default=Path("/data1/user/ycliu/closebox.png"))
    parser.add_argument(
        "--keypoints",
        type=Path,
        default=Path("/data1/user/ycliu/VLM-5d/examples/vlm_user_2d_points.json"),
    )
    parser.add_argument("--vlm5d-root", type=Path, default=Path("/data1/user/ycliu/VLM-5d"))
    parser.add_argument("--sam2-model-dir", type=Path, default=Path("/data1/user/ycliu/VLM-5d/models/sam2"))
    parser.add_argument("--moge-model", type=Path, default=Path("/data1/user/ycliu/VLM-5d/models/moge-2/model.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("/data1/user/ycliu/VLM-Planner/results/user_2d_to_3d"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--point-order", choices=["xy", "yx"], default="xy")
    parser.add_argument("--point-space", choices=["pixel", "norm1000"], default="norm1000")
    parser.add_argument("--output-point-order", choices=["xy", "yx"], default="xy")
    parser.add_argument("--output-point-space", choices=["pixel", "norm1000"], default="pixel")
    parser.add_argument("--single-mask", action="store_true", help="Use only one SAM2 mask candidate.")
    parser.add_argument("--window-size", type=int, default=5, help="Window radius for robust 3D sampling.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    _ensure_vlm5d_on_path(args.vlm5d_root)

    from perception.sam2_keypoint_matching import (  # pylint: disable=import-error
        MatchedKeypoint,
        _load_sam2_predictor,
        load_labeled_keypoints,
        save_match_overlay,
        serialize_matched_keypoints,
    )

    image_rgb = _read_rgb_image(args.image)
    h, w = image_rgb.shape[:2]
    keypoints = load_labeled_keypoints(
        json_path=args.keypoints,
        point_order=args.point_order,
        point_space=args.point_space,
        image_shape=(h, w),
    )
    if not keypoints:
        raise RuntimeError(f"No valid keypoints parsed from: {args.keypoints}")

    matches = _match_keypoints_into_mask(
        image_rgb=image_rgb,
        keypoints=keypoints,
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

    points_3d_map = _run_moge_points(image_rgb=image_rgb, model_path=args.moge_model, device=args.device)
    keypoints_3d = _lift_keypoints_to_3d(
        points_3d_map=points_3d_map,
        matches=matches,
        window_size=max(0, int(args.window_size)),
    )

    out_dir = args.out_dir
    full_json = out_dir / "sam2_matched_keypoints_full.json"
    refined_json = out_dir / "sam2_refined_keypoints.json"
    input_png = out_dir / "input_keypoints_2d.png"
    corr_png = out_dir / "keypoints_2d_correspondence.png"
    overlay_png = out_dir / "sam2_overlay.png"
    depth_png = out_dir / "moge_depth_with_keypoints.png"
    keypoints_3d_json = out_dir / "keypoints_3d_moge.json"
    keypoints_3d_png = out_dir / "keypoints_3d_reconstruction.png"

    _save_json(payload, full_json)
    _save_json(payload["refined_keypoints"], refined_json)
    _save_json(keypoints_3d, keypoints_3d_json)
    _draw_input_points(image_rgb=image_rgb, keypoints=keypoints, save_path=input_png)
    _draw_2d_correspondence(image_rgb=image_rgb, matches=matches, save_path=corr_png)
    save_match_overlay(image_rgb=image_rgb, matches=matches, save_path=overlay_png)
    _draw_depth_preview(points_3d_map=points_3d_map, matches=matches, save_path=depth_png)
    _draw_3d_points(keypoints_3d=keypoints_3d, save_path=keypoints_3d_png)

    mean_shift = float(np.mean([m.distance_px for m in matches])) if matches else 0.0
    max_shift = float(np.max([m.distance_px for m in matches])) if matches else 0.0
    print(f"[OK] keypoints={len(matches)}, mean_shift_px={mean_shift:.4f}, max_shift_px={max_shift:.4f}")
    print(f"[OUT] {full_json}")
    print(f"[OUT] {refined_json}")
    print(f"[OUT] {input_png}")
    print(f"[OUT] {corr_png}")
    print(f"[OUT] {overlay_png}")
    print(f"[OUT] {depth_png}")
    print(f"[OUT] {keypoints_3d_json}")
    print(f"[OUT] {keypoints_3d_png}")


if __name__ == "__main__":
    main()
