from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import json

from rotated_3d_leveling import (
    DEFAULT_GROUPS,
    apply_rotation,
    compute_leveling_rotation,
    draw_rotated_model,
    group_height_stats,
    run_leveling_and_draw,
    resolve_groups,
)


def _axis_angle_to_rot(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    v = 1.0 - c
    return np.array(
        [
            [x * x * v + c, x * y * v - z * s, x * z * v + y * s],
            [y * x * v + z * s, y * y * v + c, y * z * v - x * s],
            [z * x * v - y * s, z * y * v + x * s, z * z * v + c],
        ],
        dtype=np.float32,
    )


class TestRotated3DLeveling(unittest.TestCase):
    def _build_synthetic_points(self) -> dict[str, list[float]]:
        canonical = {
            "lid_top_edge_midpoint": [0.0, 0.0, 1.0],
            "lid_surface_center": [0.2, 0.1, 1.0],
            "lid_bottom_edge_midpoint": [0.4, -0.1, 1.0],
            "box_outer_bottom_front_corner": [1.0, 0.0, -0.4],
            "box_outer_bottom_right_corner": [1.4, 0.2, -0.4],
            "other": [0.3, 0.5, 0.2],
        }
        rot = _axis_angle_to_rot(axis=np.array([1.0, 1.0, 0.5]), angle_rad=0.68)
        out: dict[str, list[float]] = {}
        for label, p in canonical.items():
            pr = rot @ np.array(p, dtype=np.float32)
            out[label] = [float(pr[0]), float(pr[1]), float(pr[2])]
        return out

    def test_leveling_makes_target_groups_equal_height(self) -> None:
        keypoints = self._build_synthetic_points()
        resolved = resolve_groups(keypoints=keypoints, groups=DEFAULT_GROUPS)
        rot, pivot = compute_leveling_rotation(keypoints=keypoints, resolved_groups=resolved)
        rotated = apply_rotation(keypoints=keypoints, rotation=rot, pivot=pivot)
        stats = group_height_stats(keypoints=rotated, resolved_groups=resolved)

        self.assertLess(stats["group_1"]["z_span"], 1e-5)
        self.assertLess(stats["group_2"]["z_span"], 1e-5)

    def test_draw_rotated_model_creates_png(self) -> None:
        keypoints = self._build_synthetic_points()
        resolved = resolve_groups(keypoints=keypoints, groups=DEFAULT_GROUPS)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "rotated.png"
            draw_rotated_model(keypoints=keypoints, resolved_groups=resolved, save_path=out)
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 0)

    def test_run_leveling_with_strict_projection_forces_zero_span(self) -> None:
        keypoints = self._build_synthetic_points()
        # Inject inconsistency: same rotation cannot satisfy both groups perfectly.
        keypoints["box_outer_bottom_right_corner"][2] += 0.2
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            in_json = td_path / "in.json"
            out_png = td_path / "out.png"
            out_json = td_path / "out.json"
            report_json = td_path / "report.json"
            in_json.write_text(json.dumps(keypoints, ensure_ascii=False, indent=2), encoding="utf-8")

            stats = run_leveling_and_draw(
                input_json=in_json,
                out_png=out_png,
                out_json=out_json,
                report_json=report_json,
                strict_equal_height=True,
            )
            self.assertAlmostEqual(stats["group_1"]["z_span"], 0.0, places=8)
            self.assertAlmostEqual(stats["group_2"]["z_span"], 0.0, places=8)


if __name__ == "__main__":
    unittest.main()
