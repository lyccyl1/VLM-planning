#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence

from validate_trajectory_3d import (
    find_grip_events,
    load_keypoints_3d,
    load_trajectory,
    save_trajectory_validation_html,
    validate_trajectory,
)


def _nearest_label(point_xyz: Sequence[float], keypoints_3d: Dict[str, Sequence[float]]) -> str:
    px, py, pz = float(point_xyz[0]), float(point_xyz[1]), float(point_xyz[2])
    best_label = ""
    best_d2 = float("inf")
    for label, xyz in keypoints_3d.items():
        dx = float(xyz[0]) - px
        dy = float(xyz[1]) - py
        dz = float(xyz[2]) - pz
        d2 = dx * dx + dy * dy + dz * dz
        if d2 < best_d2:
            best_d2 = d2
            best_label = str(label)
    if not best_label:
        raise ValueError("Cannot resolve nearest label from empty keypoints.")
    return best_label


def _resolve_target_labels(
    trajectory: Sequence[Dict[str, Any]],
    keypoints_3d: Dict[str, Sequence[float]],
    grasp_label: str,
    place_label: str,
) -> tuple[str, str]:
    labels = list(keypoints_3d.keys())
    if len(labels) == 0:
        raise ValueError("3D keypoints is empty.")

    close_idx, open_idx = find_grip_events(trajectory)

    resolved_grasp = grasp_label
    if resolved_grasp not in keypoints_3d:
        if close_idx is not None:
            step = trajectory[int(close_idx)]
            resolved_grasp = _nearest_label([step["x"], step["y"], step["z"]], keypoints_3d)
        else:
            resolved_grasp = str(labels[0])

    resolved_place = place_label
    if resolved_place not in keypoints_3d:
        if open_idx is not None:
            step = trajectory[int(open_idx)]
            resolved_place = _nearest_label([step["x"], step["y"], step["z"]], keypoints_3d)
        else:
            fallback = [x for x in labels if x != resolved_grasp]
            resolved_place = str(fallback[0] if fallback else labels[0])

    return resolved_grasp, resolved_place


def generate_trajectory_html(
    trace_json: str | Path,
    keypoints_3d_json: str | Path,
    out_html: str | Path | None = None,
    out_report_json: str | Path | None = None,
    grasp_label: str = "Carrot Middle",
    place_label: str = "Basket Center Inner",
    bbox_margin: float = 0.2,
    max_step_distance: float = 0.35,
    grasp_distance_threshold: float = 0.10,
    place_distance_threshold: float = 0.10,
) -> Dict[str, Any]:
    trace_json = Path(trace_json)
    keypoints_3d_json = Path(keypoints_3d_json)

    trajectory = load_trajectory(trace_json)
    keypoints_3d = load_keypoints_3d(keypoints_3d_json)

    resolved_grasp, resolved_place = _resolve_target_labels(
        trajectory=trajectory,
        keypoints_3d=keypoints_3d,
        grasp_label=grasp_label,
        place_label=place_label,
    )

    out_html_path = Path(out_html) if out_html is not None else (keypoints_3d_json.parent / "trajectory_validation_3d.html")

    report = validate_trajectory(
        trajectory=trajectory,
        keypoints_3d=keypoints_3d,
        grasp_label=resolved_grasp,
        place_label=resolved_place,
        bbox_margin=float(bbox_margin),
        max_step_distance=float(max_step_distance),
        grasp_distance_threshold=float(grasp_distance_threshold),
        place_distance_threshold=float(place_distance_threshold),
    )

    save_trajectory_validation_html(
        trajectory=trajectory,
        keypoints_3d=keypoints_3d,
        report=report,
        out_html=out_html_path,
    )

    out_report_path: Path | None = None
    if out_report_json is not None:
        out_report_path = Path(out_report_json)
        out_report_path.parent.mkdir(parents=True, exist_ok=True)
        out_report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "overall_valid": bool(report["overall_valid"]),
        "resolved_grasp_label": resolved_grasp,
        "resolved_place_label": resolved_place,
        "out_html": str(out_html_path),
        "out_report_json": str(out_report_path) if out_report_path is not None else None,
        "report": report,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate interactive trajectory HTML from trajectory JSON + 3D keypoints JSON.")
    parser.add_argument("--trace-json", type=Path, required=True)
    parser.add_argument("--keypoints-3d-json", type=Path, required=True)
    parser.add_argument("--out-html", type=Path, default=None)
    parser.add_argument("--out-report-json", type=Path, default=None)
    parser.add_argument("--grasp-label", type=str, default="Carrot Middle")
    parser.add_argument("--place-label", type=str, default="Basket Center Inner")
    parser.add_argument("--bbox-margin", type=float, default=0.2)
    parser.add_argument("--max-step-distance", type=float, default=0.35)
    parser.add_argument("--grasp-distance-threshold", type=float, default=0.10)
    parser.add_argument("--place-distance-threshold", type=float, default=0.10)
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    result = generate_trajectory_html(
        trace_json=args.trace_json,
        keypoints_3d_json=args.keypoints_3d_json,
        out_html=args.out_html,
        out_report_json=args.out_report_json,
        grasp_label=args.grasp_label,
        place_label=args.place_label,
        bbox_margin=float(args.bbox_margin),
        max_step_distance=float(args.max_step_distance),
        grasp_distance_threshold=float(args.grasp_distance_threshold),
        place_distance_threshold=float(args.place_distance_threshold),
    )

    print(f"[OK] overall_valid={result['overall_valid']}")
    print(f"[LABEL] grasp={result['resolved_grasp_label']} place={result['resolved_place_label']}")
    print(f"[OUT] {result['out_html']}")
    if result["out_report_json"] is not None:
        print(f"[OUT] {result['out_report_json']}")


if __name__ == "__main__":
    main()
