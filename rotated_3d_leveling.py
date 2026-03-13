#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_GROUPS: Tuple[Tuple[str, ...], ...] = (
    ("lid_top_edge_midpoint", "lid_surface_center", "lid_bottom_edge_midpoint"),
    ("box_outer_bottom_front_corner", "box_outer_bottom_right_corner"),
)

ALIASES: Dict[str, Tuple[str, ...]] = {
    "box_outer_bottom_front_corner": (
        "box_bottom_front_left_corner",
        "box_floor_front_left_corner",
        "box_outer_front_left_corner",
    ),
    "box_outer_bottom_right_corner": (
        "box_bottom_front_right_corner",
        "box_floor_front_right_corner",
        "box_outer_front_right_corner",
    ),
}


def load_keypoints_3d(json_path: Path) -> Dict[str, List[float]]:
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    out: Dict[str, List[float]] = {}

    if isinstance(raw, dict):
        for label, coords in raw.items():
            if isinstance(coords, (list, tuple)) and len(coords) >= 3:
                out[str(label)] = [float(coords[0]), float(coords[1]), float(coords[2])]
        if out:
            return out

    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            label = item.get("label")
            point = item.get("point", item.get("xyz"))
            if label is None or not isinstance(point, (list, tuple)) or len(point) < 3:
                continue
            out[str(label)] = [float(point[0]), float(point[1]), float(point[2])]
        if out:
            return out

    raise ValueError("Unsupported 3D keypoint JSON format.")


def _resolve_label(keypoints: Dict[str, List[float]], label: str) -> str:
    if label in keypoints:
        return label
    for alias in ALIASES.get(label, ()):
        if alias in keypoints:
            return alias
    raise KeyError(f"Missing required label: {label}")


def resolve_groups(
    keypoints: Dict[str, List[float]],
    groups: Sequence[Sequence[str]] = DEFAULT_GROUPS,
) -> Tuple[Tuple[str, ...], ...]:
    resolved: List[Tuple[str, ...]] = []
    for group in groups:
        labels = tuple(_resolve_label(keypoints, label) for label in group)
        if len(labels) >= 2:
            resolved.append(labels)
    if not resolved:
        raise ValueError("No valid groups resolved for leveling.")
    return tuple(resolved)


def rotation_from_to(src_vec: np.ndarray, dst_vec: np.ndarray) -> np.ndarray:
    src = np.asarray(src_vec, dtype=np.float64)
    dst = np.asarray(dst_vec, dtype=np.float64)
    src /= np.linalg.norm(src)
    dst /= np.linalg.norm(dst)

    v = np.cross(src, dst)
    c = float(np.dot(src, dst))
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        if c > 0:
            return np.eye(3, dtype=np.float32)
        helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(src[0]) > 0.9:
            helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = np.cross(src, helper)
        axis /= np.linalg.norm(axis)
        k = np.array(
            [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
            dtype=np.float64,
        )
        return (np.eye(3, dtype=np.float64) + 2.0 * (k @ k)).astype(np.float32)

    k = np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float64,
    )
    rot = np.eye(3, dtype=np.float64) + k + (k @ k) * ((1.0 - c) / (s * s))
    return rot.astype(np.float32)


def compute_leveling_rotation(
    keypoints: Dict[str, List[float]],
    resolved_groups: Sequence[Sequence[str]],
) -> Tuple[np.ndarray, np.ndarray]:
    # Use group_1 as primary plane so the lid triplet is exactly flattened by rotation.
    primary = resolved_groups[0]
    primary_pts = np.array([keypoints[label] for label in primary], dtype=np.float64)
    constrained_pts: List[np.ndarray] = list(primary_pts)

    if len(primary_pts) >= 3:
        a, b, c = primary_pts[0], primary_pts[1], primary_pts[2]
        normal = np.cross(b - a, c - a)
        n_norm = float(np.linalg.norm(normal))
        if n_norm < 1e-12:
            normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            normal = normal / n_norm
    else:
        cov = np.zeros((3, 3), dtype=np.float64)
        ref = primary_pts[0]
        diffs = primary_pts[1:] - ref[None, :]
        if len(diffs) > 0:
            cov += diffs.T @ diffs
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, int(np.argmin(eigvals))]
        normal /= np.linalg.norm(normal)

    if normal[2] < 0:
        normal = -normal
    rot = rotation_from_to(normal, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    pivot = np.array(constrained_pts, dtype=np.float64).mean(axis=0)
    return rot.astype(np.float32), pivot.astype(np.float32)


def apply_rotation(
    keypoints: Dict[str, List[float]],
    rotation: np.ndarray,
    pivot: np.ndarray,
) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for label, coords in keypoints.items():
        p = np.array(coords, dtype=np.float32)
        pr = rotation @ (p - pivot) + pivot
        out[label] = [float(pr[0]), float(pr[1]), float(pr[2])]
    return out


def group_height_stats(
    keypoints: Dict[str, List[float]],
    resolved_groups: Sequence[Sequence[str]],
) -> Dict[str, Dict[str, float]]:
    report: Dict[str, Dict[str, float]] = {}
    for idx, group in enumerate(resolved_groups, start=1):
        zs = np.array([keypoints[label][2] for label in group], dtype=np.float64)
        report[f"group_{idx}"] = {
            "z_min": float(zs.min()),
            "z_max": float(zs.max()),
            "z_std": float(zs.std()),
            "z_span": float(zs.max() - zs.min()),
        }
    return report


def enforce_group_equal_height(
    keypoints: Dict[str, List[float]],
    resolved_groups: Sequence[Sequence[str]],
    mode: str = "mean",
) -> Dict[str, List[float]]:
    out = {label: [float(v) for v in coords] for label, coords in keypoints.items()}
    for group in resolved_groups:
        zs = np.array([out[label][2] for label in group], dtype=np.float64)
        if mode == "first":
            target_z = float(zs[0])
        else:
            target_z = float(zs.mean())
        for label in group:
            out[label][2] = target_z
    return out


def _set_equal_axes(ax: plt.Axes, xyz: np.ndarray) -> None:
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = (mins + maxs) / 2.0
    span = float((maxs - mins).max())
    half = max(span * 0.5, 1e-6)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)


def draw_rotated_model(
    keypoints: Dict[str, List[float]],
    resolved_groups: Sequence[Sequence[str]],
    save_path: Path,
    title: str = "Rotated 3D Model With Leveling Constraints",
) -> None:
    labels = list(keypoints.keys())
    xyz = np.array([keypoints[label] for label in labels], dtype=np.float32)
    constrained = {label for group in resolved_groups for label in group}

    colors = []
    for label in labels:
        if label in resolved_groups[0]:
            colors.append("#ff7043")
        elif len(resolved_groups) > 1 and label in resolved_groups[1]:
            colors.append("#29b6f6")
        elif label in constrained:
            colors.append("#9ccc65")
        else:
            colors.append("#bdbdbd")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, s=36)
    for idx, label in enumerate(labels, start=1):
        ax.text(
            float(xyz[idx - 1, 0]),
            float(xyz[idx - 1, 1]),
            float(xyz[idx - 1, 2]),
            f"{idx}.{label}",
            fontsize=7,
        )

    ax.scatter([], [], [], c="#ff7043", s=36, label="lid leveling group")
    ax.scatter([], [], [], c="#29b6f6", s=36, label="box bottom group")
    ax.scatter([], [], [], c="#bdbdbd", s=36, label="other points")

    # Draw horizontal reference planes for the two target groups.
    for idx, group in enumerate(resolved_groups[:2], start=1):
        grp = np.array([keypoints[label] for label in group], dtype=np.float32)
        z0 = float(np.mean(grp[:, 2]))
        x0, x1 = float(grp[:, 0].min()), float(grp[:, 0].max())
        y0, y1 = float(grp[:, 1].min()), float(grp[:, 1].max())
        pad = 0.04
        xx, yy = np.meshgrid([x0 - pad, x1 + pad], [y0 - pad, y1 + pad])
        zz = np.full_like(xx, z0, dtype=np.float32)
        face = "#ff7043" if idx == 1 else "#29b6f6"
        ax.plot_surface(xx, yy, zz, color=face, alpha=0.18, linewidth=0, shade=False)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend(loc="best")
    _set_equal_axes(ax, xyz)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=240)
    plt.close(fig)


def run_leveling_and_draw(
    input_json: Path,
    out_png: Path,
    out_json: Path,
    report_json: Path,
    groups: Sequence[Sequence[str]] = DEFAULT_GROUPS,
    strict_equal_height: bool = True,
) -> Dict[str, Dict[str, float]]:
    keypoints = load_keypoints_3d(input_json)
    resolved = resolve_groups(keypoints=keypoints, groups=groups)

    rot, pivot = compute_leveling_rotation(keypoints=keypoints, resolved_groups=resolved)
    rotated = apply_rotation(keypoints=keypoints, rotation=rot, pivot=pivot)
    stats_before = group_height_stats(keypoints=rotated, resolved_groups=resolved)
    if strict_equal_height:
        rotated = enforce_group_equal_height(keypoints=rotated, resolved_groups=resolved, mode="mean")
    stats = group_height_stats(keypoints=rotated, resolved_groups=resolved)

    draw_rotated_model(keypoints=rotated, resolved_groups=resolved, save_path=out_png)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rotated, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "resolved_groups": [list(group) for group in resolved],
        "rotation_matrix": np.asarray(rot, dtype=np.float32).tolist(),
        "pivot": np.asarray(pivot, dtype=np.float32).tolist(),
        "strict_equal_height": bool(strict_equal_height),
        "height_stats_before_projection": stats_before,
        "height_stats": stats,
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rotate 3D keypoints to satisfy leveling constraints.")
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path("/data1/user/ycliu/VLM-Planner/results/user_2d_to_3d/keypoints_3d_moge.json"),
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=Path("/data1/user/ycliu/VLM-Planner/results/user_2d_to_3d/keypoints_3d_rotated.png"),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("/data1/user/ycliu/VLM-Planner/results/user_2d_to_3d/keypoints_3d_rotated.json"),
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("/data1/user/ycliu/VLM-Planner/results/user_2d_to_3d/keypoints_3d_rotation_report.json"),
    )
    parser.add_argument(
        "--no-strict-equal-height",
        action="store_true",
        help="Disable post-rotation equal-height projection for target groups.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    stats = run_leveling_and_draw(
        input_json=args.input_json,
        out_png=args.out_png,
        out_json=args.out_json,
        report_json=args.report_json,
        strict_equal_height=not args.no_strict_equal_height,
    )
    print("[OK] rotated model saved")
    print(f"[OUT] {args.out_png}")
    print(f"[OUT] {args.out_json}")
    print(f"[OUT] {args.report_json}")
    for group, item in stats.items():
        print(
            f"[STAT] {group}: z_span={item['z_span']:.8f}, z_std={item['z_std']:.8f}, "
            f"z_min={item['z_min']:.6f}, z_max={item['z_max']:.6f}"
        )


if __name__ == "__main__":
    main()
