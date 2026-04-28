from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from validate_trajectory_3d import (
    _build_parser,
    find_grip_events,
    load_trajectory,
    save_trajectory_validation_html,
    validate_trajectory,
)


class TestValidateTrajectory3D(unittest.TestCase):
    def test_find_grip_events_detects_close_then_open(self) -> None:
        traj = [
            {"grip": 1, "x": 0.0, "y": 0.0, "z": 0.0},
            {"grip": 1, "x": 0.0, "y": 0.0, "z": 0.0},
            {"grip": 0, "x": 0.0, "y": 0.0, "z": 0.0},
            {"grip": 0, "x": 0.0, "y": 0.0, "z": 0.0},
            {"grip": 1, "x": 0.0, "y": 0.0, "z": 0.0},
        ]
        close_idx, open_idx = find_grip_events(traj)
        self.assertEqual(close_idx, 2)
        self.assertEqual(open_idx, 4)

    def test_validate_trajectory_passes_on_reasonable_case(self) -> None:
        keypoints = {
            "Carrot Middle": [0.1, 0.0, 0.5],
            "Basket Center Inner": [0.4, -0.2, 0.6],
            "aux": [0.0, 0.2, 0.4],
        }
        traj = [
            {"x": 0.1, "y": 0.0, "z": 0.4, "grip": 1},
            {"x": 0.1, "y": 0.0, "z": 0.5, "grip": 1},
            {"x": 0.1, "y": 0.0, "z": 0.5, "grip": 0},
            {"x": 0.1, "y": 0.0, "z": 0.4, "grip": 0},
            {"x": 0.4, "y": -0.2, "z": 0.4, "grip": 0},
            {"x": 0.4, "y": -0.2, "z": 0.6, "grip": 0},
            {"x": 0.4, "y": -0.2, "z": 0.6, "grip": 1},
        ]

        report = validate_trajectory(
            trajectory=traj,
            keypoints_3d=keypoints,
            grasp_label="Carrot Middle",
            place_label="Basket Center Inner",
            bbox_margin=0.2,
            max_step_distance=0.5,
            grasp_distance_threshold=0.1,
            place_distance_threshold=0.1,
        )
        self.assertTrue(report["overall_valid"])
        self.assertTrue(report["checks"]["grip_sequence_valid"]["passed"])
        self.assertTrue(report["checks"]["grasp_target_distance"]["passed"])
        self.assertTrue(report["checks"]["place_target_distance"]["passed"])

    def test_validate_trajectory_fails_when_outside_bbox(self) -> None:
        keypoints = {
            "Carrot Middle": [0.0, 0.0, 0.0],
            "Basket Center Inner": [1.0, 1.0, 1.0],
        }
        traj = [
            {"x": 10.0, "y": 0.0, "z": 0.0, "grip": 1},
            {"x": 10.0, "y": 0.0, "z": 0.0, "grip": 0},
            {"x": 10.0, "y": 0.0, "z": 0.0, "grip": 1},
        ]
        report = validate_trajectory(
            trajectory=traj,
            keypoints_3d=keypoints,
            grasp_label="Carrot Middle",
            place_label="Basket Center Inner",
            bbox_margin=0.1,
            max_step_distance=20.0,
            grasp_distance_threshold=20.0,
            place_distance_threshold=20.0,
        )
        self.assertFalse(report["checks"]["within_bbox"]["passed"])

    def test_parser_includes_interactive_html_args(self) -> None:
        parser = _build_parser()
        args = parser.parse_args([
            "--trace-json",
            "trace.json",
            "--keypoints-3d-json",
            "kps.json",
        ])
        self.assertIsNone(args.out_interactive_html)
        self.assertIsNone(args.out_plot_png)
        self.assertFalse(args.no_static_png)

    def test_load_trajectory_reads_rpy_fields(self) -> None:
        data = [
            {"x": 0.1, "y": 0.2, "z": 0.3, "grip": 1, "rx": 0.4, "ry": 0.5, "rz": 0.6, "description": "a"},
            {"x": 0.2, "y": 0.3, "z": 0.4, "grip": 0},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "trace.json"
            trace_path.write_text(json.dumps(data), encoding="utf-8")
            traj = load_trajectory(trace_path)

        self.assertEqual(len(traj), 2)
        self.assertAlmostEqual(float(traj[0]["rx"]), 0.4)
        self.assertAlmostEqual(float(traj[0]["ry"]), 0.5)
        self.assertAlmostEqual(float(traj[0]["rz"]), 0.6)
        self.assertAlmostEqual(float(traj[1]["rx"]), 0.0)
        self.assertAlmostEqual(float(traj[1]["ry"]), 0.0)
        self.assertAlmostEqual(float(traj[1]["rz"]), 0.0)

    def test_load_trajectory_accepts_wrapped_trajectory_key(self) -> None:
        data = {
            "trajectory": [
                {"x": 0.1, "y": 0.2, "z": 0.3, "grip": 1, "rx": 0.4, "ry": 0.5, "rz": 0.6, "comment": "wrapped"},
                {"x": 0.2, "y": 0.3, "z": 0.4, "grip": 0},
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "trace_wrapped.json"
            trace_path.write_text(json.dumps(data), encoding="utf-8")
            traj = load_trajectory(trace_path)

        self.assertEqual(len(traj), 2)
        self.assertEqual(str(traj[0]["description"]), "wrapped")

    def test_html_contains_rpy_and_orientation_trace(self) -> None:
        keypoints = {
            "Carrot Middle": [0.1, 0.0, 0.5],
            "Basket Center Inner": [0.4, -0.2, 0.6],
        }
        traj = [
            {"x": 0.1, "y": 0.0, "z": 0.5, "grip": 1, "rx": 0.1, "ry": 0.2, "rz": 0.3, "description": "p1"},
            {"x": 0.1, "y": 0.0, "z": 0.5, "grip": 0, "rx": 0.1, "ry": 0.2, "rz": 0.3, "description": "p2"},
            {"x": 0.4, "y": -0.2, "z": 0.6, "grip": 1, "rx": -0.1, "ry": 0.0, "rz": 0.2, "description": "p3"},
        ]
        report = validate_trajectory(
            trajectory=traj,
            keypoints_3d=keypoints,
            grasp_label="Carrot Middle",
            place_label="Basket Center Inner",
            bbox_margin=0.2,
            max_step_distance=1.0,
            grasp_distance_threshold=0.2,
            place_distance_threshold=0.2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_html = Path(tmpdir) / "vis.html"
            save_trajectory_validation_html(
                trajectory=traj,
                keypoints_3d=keypoints,
                report=report,
                out_html=out_html,
            )
            html = out_html.read_text(encoding="utf-8")

        self.assertIn("orientation (rx,ry,rz)", html)
        self.assertIn("gripper XZ plane", html)
        self.assertIn("rx=", html)
        self.assertIn("ry=", html)
        self.assertIn("rz=", html)


if __name__ == "__main__":
    unittest.main()
