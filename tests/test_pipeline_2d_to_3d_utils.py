from __future__ import annotations

from dataclasses import dataclass
import unittest

import numpy as np

from pipeline_2d_to_3d_sam2_moge import _lift_keypoints_to_3d, nearest_point_in_mask


@dataclass
class _DummyMatch:
    label: str
    matched_x: float
    matched_y: float


class TestPipeline2DTo3DUtils(unittest.TestCase):
    def test_nearest_point_in_mask_returns_nearest_inside_pixel(self) -> None:
        mask = np.zeros((9, 9), dtype=np.uint8)
        mask[2:5, 2:5] = 1
        x, y = nearest_point_in_mask(mask, point_xy=(7.2, 7.1))
        self.assertEqual((x, y), (4.0, 4.0))

    def test_lift_keypoints_to_3d_uses_patch_median(self) -> None:
        points = np.zeros((7, 7, 3), dtype=np.float32)
        points[..., 0] = 1.0
        points[..., 1] = 2.0
        points[..., 2] = 3.0
        points[3, 3] = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        matches = [_DummyMatch(label="p", matched_x=3.0, matched_y=3.0)]

        out = _lift_keypoints_to_3d(points_3d_map=points, matches=matches, window_size=1)
        self.assertIn("p", out)
        self.assertTrue(np.allclose(out["p"], [1.0, 2.0, 3.0]))

    def test_lift_keypoints_to_3d_skips_all_nan_point(self) -> None:
        points = np.full((5, 5, 3), np.nan, dtype=np.float32)
        matches = [_DummyMatch(label="missing", matched_x=2.0, matched_y=2.0)]
        out = _lift_keypoints_to_3d(points_3d_map=points, matches=matches, window_size=1)
        self.assertEqual(out, {})


if __name__ == "__main__":
    unittest.main()
