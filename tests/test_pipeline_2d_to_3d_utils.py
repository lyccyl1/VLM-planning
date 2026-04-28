from __future__ import annotations

from dataclasses import dataclass
import unittest

import numpy as np

from pipeline_2d_to_3d_sam2_moge import (
    _build_identity_matches,
    _lift_keypoints_to_3d,
    _project_depth_to_3d_with_intrinsics,
    _resize_image_to_resolution,
    nearest_point_in_mask,
)


@dataclass
class _DummyMatch:
    label: str
    matched_x: float
    matched_y: float


@dataclass
class _DummyKeypoint:
    label: str
    x: float
    y: float


@dataclass
class _DummyMatched:
    label: str
    input_x: float
    input_y: float
    matched_x: float
    matched_y: float
    distance_px: float
    mask_area_px: int
    sam_score: float
    input_inside_mask: bool


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

    def test_resize_image_to_resolution_returns_scaled_shape(self) -> None:
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        resized, sx, sy = _resize_image_to_resolution(image, target_width=1280, target_height=720)
        self.assertEqual(resized.shape, (720, 1280, 3))
        self.assertAlmostEqual(sx, 6.4)
        self.assertAlmostEqual(sy, 7.2)

    def test_build_identity_matches_zero_shift(self) -> None:
        keypoints = [
            _DummyKeypoint(label="a", x=11.0, y=22.0),
            _DummyKeypoint(label="b", x=33.0, y=44.0),
        ]
        matches = _build_identity_matches(keypoints=keypoints, matched_cls=_DummyMatched)
        self.assertEqual(len(matches), 2)
        self.assertEqual(matches[0].matched_x, 11.0)
        self.assertEqual(matches[0].matched_y, 22.0)
        self.assertEqual(matches[0].distance_px, 0.0)

    def test_project_depth_to_3d_with_intrinsics(self) -> None:
        depth = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        pts = _project_depth_to_3d_with_intrinsics(
            depth_map=depth,
            fx=2.0,
            fy=4.0,
            cx=0.0,
            cy=0.0,
        )
        self.assertEqual(pts.shape, (2, 2, 3))
        self.assertTrue(np.allclose(pts[0, 0], [0.0, 0.0, 1.0]))
        self.assertTrue(np.allclose(pts[0, 1], [1.0, 0.0, 2.0]))
        self.assertTrue(np.allclose(pts[1, 0], [0.0, 0.75, 3.0]))

    def test_project_depth_to_3d_marks_invalid_depth_as_nan(self) -> None:
        depth = np.array([[1.0, 0.0], [np.inf, 2.0]], dtype=np.float32)
        pts = _project_depth_to_3d_with_intrinsics(
            depth_map=depth,
            fx=1.0,
            fy=1.0,
            cx=0.0,
            cy=0.0,
        )
        self.assertTrue(np.isnan(pts[0, 1]).all())
        self.assertTrue(np.isnan(pts[1, 0]).all())
        self.assertTrue(np.isfinite(pts[1, 1]).all())

    def test_project_depth_to_3d_rejects_nonpositive_focal_length(self) -> None:
        depth = np.ones((2, 2), dtype=np.float32)
        with self.assertRaises(ValueError):
            _project_depth_to_3d_with_intrinsics(depth_map=depth, fx=0.0, fy=1.0, cx=0.0, cy=0.0)


if __name__ == "__main__":
    unittest.main()
