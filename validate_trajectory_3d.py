#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


def _as_finite_float(value: Any, name: str) -> float:
    try:
        out = float(value)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"{name} must be numeric, got: {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got: {value!r}")
    return out


def load_trajectory(trace_json: Path) -> List[Dict[str, Any]]:
    raw = json.loads(trace_json.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        for key in ("trajectory", "trace", "steps"):
            if key in raw:
                raw = raw[key]
                break
    if not isinstance(raw, list) or len(raw) == 0:
        raise ValueError("Trajectory JSON must be a non-empty list, or dict with key trajectory/trace/steps.")

    out: List[Dict[str, Any]] = []
    for i, step in enumerate(raw):
        if not isinstance(step, dict):
            raise ValueError(f"Trajectory step[{i}] must be a dict.")
        x = _as_finite_float(step.get("x"), f"step[{i}].x")
        y = _as_finite_float(step.get("y"), f"step[{i}].y")
        z = _as_finite_float(step.get("z"), f"step[{i}].z")
        rx = _as_finite_float(step.get("rx", 0.0), f"step[{i}].rx")
        ry = _as_finite_float(step.get("ry", 0.0), f"step[{i}].ry")
        rz = _as_finite_float(step.get("rz", 0.0), f"step[{i}].rz")
        grip = _as_finite_float(step.get("grip"), f"step[{i}].grip")

        out.append(
            {
                "index": i,
                "x": x,
                "y": y,
                "z": z,
                "rx": rx,
                "ry": ry,
                "rz": rz,
                "grip": grip,
                "description": str(step.get("description", step.get("comment", ""))),
            }
        )
    return out


def load_keypoints_3d(json_path: Path) -> Dict[str, List[float]]:
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise ValueError("3D keypoints JSON must be a non-empty dict: {label: [x,y,z]}.")

    out: Dict[str, List[float]] = {}
    for label, xyz in raw.items():
        if not isinstance(label, str):
            raise ValueError("Keypoint label must be string.")
        if not isinstance(xyz, (list, tuple)) or len(xyz) < 3:
            raise ValueError(f"Keypoint[{label}] must be [x,y,z].")
        out[label] = [
            _as_finite_float(xyz[0], f"keypoints[{label}].x"),
            _as_finite_float(xyz[1], f"keypoints[{label}].y"),
            _as_finite_float(xyz[2], f"keypoints[{label}].z"),
        ]
    return out


def find_grip_events(trajectory: Sequence[Dict[str, Any]]) -> Tuple[int | None, int | None]:
    if len(trajectory) < 2:
        return None, None

    states = [float(s["grip"]) > 0.5 for s in trajectory]
    initial_state = states[0]

    close_idx: int | None = None
    open_idx: int | None = None

    for i in range(1, len(states)):
        if states[i] != initial_state:
            close_idx = i
            break

    if close_idx is not None:
        for i in range(close_idx + 1, len(states)):
            if states[i] == initial_state:
                open_idx = i
                break

    return close_idx, open_idx


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _safe_unit(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if n <= 1e-12:
        return fallback.copy()
    return vec / n


def _euler_xyz_to_rotation(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    rx_m = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    ry_m = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rz_m = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz_m @ ry_m @ rx_m


def _euler_xyz_to_tool_z(rx: float, ry: float, rz: float) -> np.ndarray:
    rot = _euler_xyz_to_rotation(rx, ry, rz)
    return _safe_unit(rot[:, 2], np.array([0.0, 0.0, 1.0], dtype=np.float64))


def _set_equal_axes(ax: Any, xyz: np.ndarray) -> None:
    if xyz.size == 0:
        return
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = (mins + maxs) / 2.0
    span = float((maxs - mins).max())
    half = max(span * 0.55, 1e-6)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)


def _extract_plot_data(
    trajectory: Sequence[Dict[str, Any]],
    keypoints_3d: Dict[str, List[float]],
    report: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray, int | None, int | None, set[int]]:
    traj_xyz = np.array([[s["x"], s["y"], s["z"]] for s in trajectory], dtype=np.float64)
    grips = np.array([float(s["grip"]) for s in trajectory], dtype=np.float64)
    rpy = np.array(
        [[float(s.get("rx", 0.0)), float(s.get("ry", 0.0)), float(s.get("rz", 0.0))] for s in trajectory],
        dtype=np.float64,
    )
    descs = np.array([str(s.get("description", "")) for s in trajectory], dtype=object)

    kp_labels = list(keypoints_3d.keys())
    kp_xyz = np.array([keypoints_3d[k] for k in kp_labels], dtype=np.float64)

    close_idx = report["checks"]["grip_sequence_valid"]["close_step_index"]
    open_idx = report["checks"]["grip_sequence_valid"]["open_step_index"]
    outside_indices = set(report["checks"]["within_bbox"]["outside_step_indices"])
    return traj_xyz, grips, rpy, descs, kp_labels, kp_xyz, close_idx, open_idx, outside_indices


def validate_trajectory(
    trajectory: Sequence[Dict[str, Any]],
    keypoints_3d: Dict[str, List[float]],
    grasp_label: str,
    place_label: str,
    bbox_margin: float,
    max_step_distance: float,
    grasp_distance_threshold: float,
    place_distance_threshold: float,
) -> Dict[str, Any]:
    if len(trajectory) == 0:
        raise ValueError("Trajectory is empty.")
    if grasp_label not in keypoints_3d:
        raise ValueError(f"Grasp label not found in 3D keypoints: {grasp_label}")
    if place_label not in keypoints_3d:
        raise ValueError(f"Place label not found in 3D keypoints: {place_label}")

    traj_xyz = np.array([[s["x"], s["y"], s["z"]] for s in trajectory], dtype=np.float64)
    kp_labels = list(keypoints_3d.keys())
    kp_xyz = np.array([keypoints_3d[k] for k in kp_labels], dtype=np.float64)

    mins = kp_xyz.min(axis=0)
    maxs = kp_xyz.max(axis=0)
    low = mins - float(bbox_margin)
    high = maxs + float(bbox_margin)
    inside_mask = np.logical_and(traj_xyz >= low, traj_xyz <= high).all(axis=1)
    outside_indices = [int(i) for i in np.where(~inside_mask)[0].tolist()]

    step_dists = np.linalg.norm(np.diff(traj_xyz, axis=0), axis=1) if len(traj_xyz) > 1 else np.array([], dtype=np.float64)
    max_observed_step_dist = float(step_dists.max()) if step_dists.size > 0 else 0.0
    violating_edges = [int(i + 1) for i, d in enumerate(step_dists.tolist()) if d > float(max_step_distance)]

    close_idx, open_idx = find_grip_events(trajectory)
    close_xyz = traj_xyz[int(close_idx)] if close_idx is not None else None
    open_xyz = traj_xyz[int(open_idx)] if open_idx is not None else None

    grasp_xyz = np.array(keypoints_3d[grasp_label], dtype=np.float64)
    place_xyz = np.array(keypoints_3d[place_label], dtype=np.float64)

    grasp_dist = _distance(close_xyz, grasp_xyz) if close_xyz is not None else None
    place_dist = _distance(open_xyz, place_xyz) if open_xyz is not None else None

    checks: Dict[str, Dict[str, Any]] = {
        "within_bbox": {
            "passed": len(outside_indices) == 0,
            "bbox_low": [float(x) for x in low.tolist()],
            "bbox_high": [float(x) for x in high.tolist()],
            "outside_step_indices": outside_indices,
        },
        "step_distance": {
            "passed": len(violating_edges) == 0,
            "max_allowed": float(max_step_distance),
            "max_observed": float(max_observed_step_dist),
            "violating_edge_indices": violating_edges,
        },
        "grip_sequence_valid": {
            "passed": close_idx is not None and open_idx is not None and int(open_idx) > int(close_idx),
            "close_step_index": int(close_idx) if close_idx is not None else None,
            "open_step_index": int(open_idx) if open_idx is not None else None,
        },
        "grasp_target_distance": {
            "passed": (grasp_dist is not None) and (float(grasp_dist) <= float(grasp_distance_threshold)),
            "threshold": float(grasp_distance_threshold),
            "distance": float(grasp_dist) if grasp_dist is not None else None,
            "target_label": grasp_label,
            "close_step_index": int(close_idx) if close_idx is not None else None,
        },
        "place_target_distance": {
            "passed": (place_dist is not None) and (float(place_dist) <= float(place_distance_threshold)),
            "threshold": float(place_distance_threshold),
            "distance": float(place_dist) if place_dist is not None else None,
            "target_label": place_label,
            "open_step_index": int(open_idx) if open_idx is not None else None,
        },
    }

    overall_valid = bool(all(bool(v["passed"]) for v in checks.values()))

    return {
        "overall_valid": overall_valid,
        "summary": {
            "num_steps": int(len(trajectory)),
            "num_keypoints": int(len(keypoints_3d)),
            "grasp_label": grasp_label,
            "place_label": place_label,
        },
        "checks": checks,
    }


def save_trajectory_validation_plot(
    trajectory: Sequence[Dict[str, Any]],
    keypoints_3d: Dict[str, List[float]],
    report: Dict[str, Any],
    out_png: Path,
) -> None:
    traj_xyz, grips, _, _, kp_labels, kp_xyz, close_idx, open_idx, outside_indices = _extract_plot_data(
        trajectory=trajectory,
        keypoints_3d=keypoints_3d,
        report=report,
    )

    fig = plt.figure(figsize=(12.8, 7.2))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(kp_xyz[:, 0], kp_xyz[:, 1], kp_xyz[:, 2], c="#1e88e5", s=28, label="3D keypoints")
    for i, label in enumerate(kp_labels):
        ax.text(float(kp_xyz[i, 0]), float(kp_xyz[i, 1]), float(kp_xyz[i, 2]), label, fontsize=6)

    ax.plot(traj_xyz[:, 0], traj_xyz[:, 1], traj_xyz[:, 2], color="#fdd835", linewidth=1.8, label="trajectory")

    open_mask = grips > 0.5
    close_mask = ~open_mask
    if np.any(open_mask):
        ax.scatter(traj_xyz[open_mask, 0], traj_xyz[open_mask, 1], traj_xyz[open_mask, 2], c="#2e7d32", s=24, label="grip=open")
    if np.any(close_mask):
        ax.scatter(traj_xyz[close_mask, 0], traj_xyz[close_mask, 1], traj_xyz[close_mask, 2], c="#c62828", s=24, label="grip=close")

    for i in range(len(traj_xyz)):
        ax.text(float(traj_xyz[i, 0]), float(traj_xyz[i, 1]), float(traj_xyz[i, 2]), str(i + 1), fontsize=7)

    if close_idx is not None:
        ax.scatter([traj_xyz[int(close_idx), 0]], [traj_xyz[int(close_idx), 1]], [traj_xyz[int(close_idx), 2]], marker="^", c="#d32f2f", s=90, label="close event")
    if open_idx is not None:
        ax.scatter([traj_xyz[int(open_idx), 0]], [traj_xyz[int(open_idx), 1]], [traj_xyz[int(open_idx), 2]], marker="^", c="#2e7d32", s=90, label="open event")

    if outside_indices:
        outside = np.array([traj_xyz[i] for i in sorted(outside_indices)], dtype=np.float64)
        ax.scatter(outside[:, 0], outside[:, 1], outside[:, 2], marker="x", c="#8e24aa", s=80, label="outside bbox")

    all_xyz = np.concatenate([kp_xyz, traj_xyz], axis=0)
    _set_equal_axes(ax, all_xyz)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Trajectory Validation in 3D Space | valid={report['overall_valid']}")
    ax.legend(loc="best", fontsize=8)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=240)
    plt.close(fig)


def save_trajectory_validation_html(
    trajectory: Sequence[Dict[str, Any]],
    keypoints_3d: Dict[str, List[float]],
    report: Dict[str, Any],
    out_html: Path,
) -> None:
    traj_xyz, grips, rpy, descs, kp_labels, kp_xyz, close_idx, open_idx, outside_indices = _extract_plot_data(
        trajectory=trajectory,
        keypoints_3d=keypoints_3d,
        report=report,
    )

    step_text = [str(i + 1) for i in range(len(traj_xyz))]
    step_hover = [
        f"step={i + 1}<br>grip={float(grips[i]):.0f}<br>rx={rpy[i,0]:.4f}<br>ry={rpy[i,1]:.4f}<br>rz={rpy[i,2]:.4f}<br>{descs[i]}"
        for i in range(len(traj_xyz))
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=kp_xyz[:, 0],
            y=kp_xyz[:, 1],
            z=kp_xyz[:, 2],
            mode="markers+text",
            text=kp_labels,
            textposition="top center",
            marker={"size": 4, "color": "#1e88e5"},
            name="3D keypoints",
            hovertemplate="%{text}<br>x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=traj_xyz[:, 0],
            y=traj_xyz[:, 1],
            z=traj_xyz[:, 2],
            mode="lines+markers+text",
            text=step_text,
            textposition="top center",
            line={"width": 4, "color": "#fdd835"},
            marker={"size": 4, "color": np.where(grips > 0.5, "#2e7d32", "#c62828")},
            customdata=np.array(step_hover, dtype=object),
            name="trajectory",
            hovertemplate="%{customdata}<br>x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>",
        )
    )

    span = float((kp_xyz.max(axis=0) - kp_xyz.min(axis=0)).max()) if len(kp_xyz) > 0 else 1.0
    axis_len = max(0.03, span * 0.08)
    plane_half_x = max(0.015, span * 0.035)
    plane_half_z = max(0.02, span * 0.05)

    ozx: List[float | None] = []
    ozy: List[float | None] = []
    ozz: List[float | None] = []

    oxx: List[float | None] = []
    oxy: List[float | None] = []
    oxz: List[float | None] = []

    edge_x: List[float | None] = []
    edge_y: List[float | None] = []
    edge_z: List[float | None] = []

    plane_x: List[float] = []
    plane_y: List[float] = []
    plane_z: List[float] = []
    tri_i: List[int] = []
    tri_j: List[int] = []
    tri_k: List[int] = []

    for i in range(len(traj_xyz)):
        rot = _euler_xyz_to_rotation(float(rpy[i, 0]), float(rpy[i, 1]), float(rpy[i, 2]))

        z_axis = _safe_unit(rot[:, 2], np.array([0.0, 0.0, 1.0], dtype=np.float64))
        x_axis = _safe_unit(rot[:, 0], np.array([1.0, 0.0, 0.0], dtype=np.float64))
        y_axis = _safe_unit(np.cross(z_axis, x_axis), np.array([0.0, 1.0, 0.0], dtype=np.float64))
        x_axis = _safe_unit(np.cross(y_axis, z_axis), np.array([1.0, 0.0, 0.0], dtype=np.float64))

        center = traj_xyz[i]

        z_end = center + z_axis * axis_len
        ozx.extend([float(center[0]), float(z_end[0]), None])
        ozy.extend([float(center[1]), float(z_end[1]), None])
        ozz.extend([float(center[2]), float(z_end[2]), None])

        x_end = center + x_axis * axis_len
        oxx.extend([float(center[0]), float(x_end[0]), None])
        oxy.extend([float(center[1]), float(x_end[1]), None])
        oxz.extend([float(center[2]), float(x_end[2]), None])

        # XZ plane: Z is approach direction, X is jaw opening direction.
        c1 = center + x_axis * plane_half_x + z_axis * plane_half_z
        c2 = center - x_axis * plane_half_x + z_axis * plane_half_z
        c3 = center - x_axis * plane_half_x - z_axis * plane_half_z
        c4 = center + x_axis * plane_half_x - z_axis * plane_half_z
        corners = [c1, c2, c3, c4]

        for c in corners + [c1]:
            edge_x.append(float(c[0]))
            edge_y.append(float(c[1]))
            edge_z.append(float(c[2]))
        edge_x.append(None)
        edge_y.append(None)
        edge_z.append(None)

        base = len(plane_x)
        for c in corners:
            plane_x.append(float(c[0]))
            plane_y.append(float(c[1]))
            plane_z.append(float(c[2]))
        tri_i.extend([base, base])
        tri_j.extend([base + 1, base + 2])
        tri_k.extend([base + 2, base + 3])

    fig.add_trace(
        go.Scatter3d(
            x=ozx,
            y=ozy,
            z=ozz,
            mode="lines",
            line={"width": 3, "color": "#26c6da"},
            name="orientation (rx,ry,rz)",
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=oxx,
            y=oxy,
            z=oxz,
            mode="lines",
            line={"width": 2, "color": "#ff7043"},
            name="local X (jaw opening)",
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Mesh3d(
            x=plane_x,
            y=plane_y,
            z=plane_z,
            i=tri_i,
            j=tri_j,
            k=tri_k,
            color="#ffb74d",
            opacity=0.22,
            name="gripper XZ plane",
            hoverinfo="skip",
            showscale=False,
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode="lines",
            line={"width": 2, "color": "#ef6c00"},
            name="gripper XZ boundary",
            hoverinfo="skip",
        )
    )

    if close_idx is not None:
        fig.add_trace(
            go.Scatter3d(
                x=[traj_xyz[int(close_idx), 0]],
                y=[traj_xyz[int(close_idx), 1]],
                z=[traj_xyz[int(close_idx), 2]],
                mode="markers",
                marker={"size": 8, "color": "#d32f2f", "symbol": "diamond"},
                name="close event",
            )
        )
    if open_idx is not None:
        fig.add_trace(
            go.Scatter3d(
                x=[traj_xyz[int(open_idx), 0]],
                y=[traj_xyz[int(open_idx), 1]],
                z=[traj_xyz[int(open_idx), 2]],
                mode="markers",
                marker={"size": 8, "color": "#2e7d32", "symbol": "diamond"},
                name="open event",
            )
        )

    if outside_indices:
        outside = np.array([traj_xyz[i] for i in sorted(outside_indices)], dtype=np.float64)
        fig.add_trace(
            go.Scatter3d(
                x=outside[:, 0],
                y=outside[:, 1],
                z=outside[:, 2],
                mode="markers",
                marker={"size": 7, "color": "#8e24aa", "symbol": "x"},
                name="outside bbox",
            )
        )

    fig.update_layout(
        title=f"Trajectory Validation 3D (interactive) | valid={report['overall_valid']}",
        scene={
            "xaxis_title": "X",
            "yaxis_title": "Y",
            "zaxis_title": "Z",
            "aspectmode": "data",
        },
        legend={"orientation": "h", "y": 1.02, "x": 0.0},
        margin={"l": 0, "r": 0, "t": 45, "b": 0},
        width=1280,
        height=720,
    )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs=True, full_html=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate trajectory JSON against 3D keypoints and export interactive 3D visualization.")
    parser.add_argument("--trace-json", type=Path, required=True)
    parser.add_argument("--keypoints-3d-json", type=Path, required=True)
    parser.add_argument("--out-report-json", type=Path, default=None)
    parser.add_argument("--out-plot-png", type=Path, default=None)
    parser.add_argument("--out-interactive-html", type=Path, default=None)
    parser.add_argument("--no-static-png", action="store_true", help="Skip static PNG export.")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--grasp-label", type=str, default="Carrot Middle")
    parser.add_argument("--place-label", type=str, default="Basket Center Inner")
    parser.add_argument("--bbox-margin", type=float, default=0.2)
    parser.add_argument("--max-step-distance", type=float, default=0.35)
    parser.add_argument("--grasp-distance-threshold", type=float, default=0.10)
    parser.add_argument("--place-distance-threshold", type=float, default=0.10)
    parser.add_argument("--fail-on-invalid", action="store_true", help="Return non-zero code if trajectory is invalid.")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    trajectory = load_trajectory(args.trace_json)
    keypoints_3d = load_keypoints_3d(args.keypoints_3d_json)

    out_dir = args.out_dir if args.out_dir is not None else args.keypoints_3d_json.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    out_report_json = args.out_report_json or (out_dir / "trajectory_validation_report.json")
    out_plot_png = args.out_plot_png or (out_dir / "trajectory_validation_3d.png")
    out_interactive_html = args.out_interactive_html or (out_dir / "trajectory_validation_3d.html")

    report = validate_trajectory(
        trajectory=trajectory,
        keypoints_3d=keypoints_3d,
        grasp_label=args.grasp_label,
        place_label=args.place_label,
        bbox_margin=float(args.bbox_margin),
        max_step_distance=float(args.max_step_distance),
        grasp_distance_threshold=float(args.grasp_distance_threshold),
        place_distance_threshold=float(args.place_distance_threshold),
    )

    out_report_json.parent.mkdir(parents=True, exist_ok=True)
    out_report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.no_static_png:
        save_trajectory_validation_plot(
            trajectory=trajectory,
            keypoints_3d=keypoints_3d,
            report=report,
            out_png=out_plot_png,
        )

    save_trajectory_validation_html(
        trajectory=trajectory,
        keypoints_3d=keypoints_3d,
        report=report,
        out_html=out_interactive_html,
    )

    print(f"[OK] overall_valid={report['overall_valid']}")
    print(f"[OUT] {out_report_json}")
    if not args.no_static_png:
        print(f"[OUT] {out_plot_png}")
    print(f"[OUT] {out_interactive_html}")

    if args.fail_on_invalid and not report["overall_valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
