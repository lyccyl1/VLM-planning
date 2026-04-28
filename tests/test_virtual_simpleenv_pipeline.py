import unittest

from virtual_simpleenv_pipeline import (
    _euler_delta_to_rot_axangle,
    _keypoints_as_semantic_list,
    _sanitize_english_label,
    _strip_think_tags,
    build_simpleenv_actions_from_trace,
    extract_json_block,
    heuristic_plan_trace,
    normalize_keypoints_payload,
)


class TestVirtualSimpleEnvPipeline(unittest.TestCase):
    def test_extract_json_block_from_wrapped_text(self) -> None:
        raw = "Some prefix\n```json\n{\"a\": 1}\n```\nSuffix"
        block = extract_json_block(raw)
        self.assertEqual(block, "{\"a\": 1}")

    def test_normalize_keypoints_payload(self) -> None:
        raw = {
            "keypoints": [
                {"label": "grasp_point", "point": [120, 220]},
                {"label": "target_center", "xy": [320, 420]},
            ]
        }
        out = normalize_keypoints_payload(raw, image_w=640, image_h=480)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["label"], "grasp_point")
        self.assertEqual(out[0]["point"], [120.0, 220.0])
        self.assertEqual(out[1]["point"], [320.0, 420.0])

    def test_build_simpleenv_actions_from_trace(self) -> None:
        trace = [
            {"x": 0.1, "y": 0.2, "z": 0.3, "rx": 3.14, "ry": 0.0, "rz": 0.0, "grip": 0},
            {"x": 0.15, "y": 0.25, "z": 0.35, "rx": 3.14, "ry": 0.0, "rz": 0.1, "grip": 1},
        ]
        out = build_simpleenv_actions_from_trace(trace)
        self.assertIn("actions", out)
        self.assertEqual(len(out["actions"]), 2)
        self.assertEqual(len(out["actions"][0]["action_7d"]), 7)
        self.assertEqual(out["actions"][0]["action_7d"][-1], -1.0)
        self.assertEqual(out["actions"][1]["action_7d"][-1], 1.0)
        self.assertIn("env_step_actions_7d", out)
        self.assertEqual(out["env_step_actions_7d"][1], out["actions"][1]["action_7d"])
        self.assertEqual(out["actions"][1]["rotation_delta"][2], 0.1)
        self.assertAlmostEqual(out["actions"][1]["rot_axangle"][2], 0.1, places=6)
        self.assertEqual(out["actions"][1]["gripper"], [1.0])
        self.assertEqual(out["actions"][1]["terminate_episode"], [0.0])
        self.assertEqual(out["actions"][1]["open_gripper"], [0.0])

    def test_euler_delta_to_rot_axangle(self) -> None:
        x, y, z = _euler_delta_to_rot_axangle(0.3, 0.0, 0.0)
        self.assertAlmostEqual(x, 0.3, places=6)
        self.assertAlmostEqual(y, 0.0, places=6)
        self.assertAlmostEqual(z, 0.0, places=6)

    def test_heuristic_plan_trace_has_grip_toggle(self) -> None:
        keypoints_3d = {
            "grasp_point": [0.1, 0.2, 0.3],
            "place_point": [0.4, 0.5, 0.35],
        }
        plan = heuristic_plan_trace(
            task_instruction="pick and place",
            keypoints_3d=keypoints_3d,
        )
        self.assertIn("trajectory", plan)
        traj = plan["trajectory"]
        self.assertGreaterEqual(len(traj), 6)
        grips = [int(float(s["grip"]) > 0.5) for s in traj]
        self.assertIn(1, grips)
        self.assertEqual(grips[-1], 0)

    def test_sanitize_english_label(self) -> None:
        self.assertEqual(_sanitize_english_label("盒子 corner#1", idx=1), "corner 1")

    def test_keypoints_as_semantic_list(self) -> None:
        out = _keypoints_as_semantic_list(
            [
                {"label": "Object Center", "point": [10, 20]},
                {"label": "目标点", "xy": [30, 40]},
            ]
        )
        self.assertEqual(out[0], {"point": [10.0, 20.0], "label": "object center"})
        self.assertEqual(out[1], {"point": [30.0, 40.0], "label": "point_2"})

    def test_strip_think_tags(self) -> None:
        text = "<think>internal reasoning</think>{\"trajectory\":[]}"
        self.assertEqual(_strip_think_tags(text), "{\"trajectory\":[]}")


if __name__ == "__main__":
    unittest.main()
