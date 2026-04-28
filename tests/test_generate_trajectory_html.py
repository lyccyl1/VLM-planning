from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from generate_trajectory_html import generate_trajectory_html


class TestGenerateTrajectoryHTML(unittest.TestCase):
    def test_generate_html_from_two_json_inputs(self) -> None:
        keypoints = {
            "Carrot Middle": [0.1, 0.0, 0.5],
            "Basket Center Inner": [0.4, -0.2, 0.6],
        }
        trajectory = [
            {"x": 0.1, "y": 0.0, "z": 0.4, "grip": 1, "rx": 0.1, "ry": 0.2, "rz": 0.3, "description": "p1"},
            {"x": 0.1, "y": 0.0, "z": 0.5, "grip": 1, "rx": 0.1, "ry": 0.2, "rz": 0.3, "description": "p2"},
            {"x": 0.1, "y": 0.0, "z": 0.5, "grip": 0, "rx": 0.1, "ry": 0.2, "rz": 0.3, "description": "p3"},
            {"x": 0.4, "y": -0.2, "z": 0.6, "grip": 0, "rx": -0.1, "ry": 0.0, "rz": 0.2, "description": "p4"},
            {"x": 0.4, "y": -0.2, "z": 0.6, "grip": 1, "rx": -0.1, "ry": 0.0, "rz": 0.2, "description": "p5"},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            trace_json = tmp / "trace.json"
            keypoints_json = tmp / "keypoints_3d.json"
            out_html = tmp / "trajectory.html"
            out_report = tmp / "report.json"

            trace_json.write_text(json.dumps(trajectory), encoding="utf-8")
            keypoints_json.write_text(json.dumps(keypoints), encoding="utf-8")

            result = generate_trajectory_html(
                trace_json=trace_json,
                keypoints_3d_json=keypoints_json,
                out_html=out_html,
                out_report_json=out_report,
            )

            self.assertTrue(out_html.exists())
            self.assertTrue(out_report.exists())
            self.assertEqual(result["out_html"], str(out_html))
            self.assertEqual(result["out_report_json"], str(out_report))

            html = out_html.read_text(encoding="utf-8")
            self.assertIn("gripper XZ plane", html)
            self.assertIn("orientation (rx,ry,rz)", html)
            self.assertIn("rx=", html)


if __name__ == "__main__":
    unittest.main()
