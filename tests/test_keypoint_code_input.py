from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from pipeline_2d_to_3d_sam2_moge import (
    _load_keypoint_entries_from_python_module,
    _normalize_keypoint_entries,
)


class TestKeypointCodeInput(unittest.TestCase):
    def test_normalize_dict_mapping(self) -> None:
        raw = {
            "p1": [10, 20],
            "p2": [30.5, 40.25],
        }
        out = _normalize_keypoint_entries(raw)
        self.assertEqual(
            out,
            [
                {"label": "p1", "point": [10.0, 20.0]},
                {"label": "p2", "point": [30.5, 40.25]},
            ],
        )

    def test_normalize_list_xy_fields(self) -> None:
        raw = [
            {"label": "a", "xy": [1, 2]},
            {"label": "b", "x": 3, "y": 4},
        ]
        out = _normalize_keypoint_entries(raw)
        self.assertEqual(
            out,
            [
                {"label": "a", "point": [1.0, 2.0]},
                {"label": "b", "point": [3.0, 4.0]},
            ],
        )

    def test_load_from_python_module_keypoints_variable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            py_file = Path(tmpdir) / "kps.py"
            py_file.write_text("KEYPOINTS = [{'label': 'p', 'point': [11, 22]}]\n", encoding="utf-8")
            out = _load_keypoint_entries_from_python_module(py_file)
            self.assertEqual(out, [{"label": "p", "point": [11.0, 22.0]}])

    def test_load_from_python_module_function(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            py_file = Path(tmpdir) / "kps.py"
            py_file.write_text(
                "def get_keypoints():\n"
                "    return {'q': [5, 6]}\n",
                encoding="utf-8",
            )
            out = _load_keypoint_entries_from_python_module(py_file)
            self.assertEqual(out, [{"label": "q", "point": [5.0, 6.0]}])


if __name__ == "__main__":
    unittest.main()
