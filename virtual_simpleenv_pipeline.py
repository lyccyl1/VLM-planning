#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


def extract_json_block(text: str) -> str:
    s = str(text).strip()
    if not s:
        raise ValueError("Empty model output.")

    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, flags=re.IGNORECASE)
    if fence_match:
        candidate = fence_match.group(1).strip()
        if candidate:
            s = candidate

    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        return s

    decoder = json.JSONDecoder()
    for i, ch in enumerate(s):
        if ch not in "{[":
            continue
        candidate = s[i:]
        try:
            _, end = decoder.raw_decode(candidate)
            return candidate[:end]
        except Exception:
            continue
    raise ValueError("No JSON object/array detected in model output.")


def _loads_json_loose(text: str) -> Any:
    block = extract_json_block(text)
    try:
        return json.loads(block)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(block.strip())
        return obj


def _to_float(v: Any) -> float:
    return float(v)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _parse_point_xy(item: Mapping[str, Any]) -> Tuple[float, float]:
    if "point" in item and isinstance(item["point"], (list, tuple)) and len(item["point"]) >= 2:
        return _to_float(item["point"][0]), _to_float(item["point"][1])
    if "xy" in item and isinstance(item["xy"], (list, tuple)) and len(item["xy"]) >= 2:
        return _to_float(item["xy"][0]), _to_float(item["xy"][1])
    if "x" in item and "y" in item:
        return _to_float(item["x"]), _to_float(item["y"])
    raise ValueError(f"Invalid keypoint entry, missing coordinates: {item}")


def normalize_keypoints_payload(raw: Any, image_w: int, image_h: int) -> List[Dict[str, Any]]:
    if image_w <= 0 or image_h <= 0:
        raise ValueError(f"Invalid image size: {image_w}x{image_h}")

    if isinstance(raw, dict):
        if "keypoints" in raw:
            items = raw["keypoints"]
        elif "points" in raw:
            items = raw["points"]
        else:
            items = []
            for k, v in raw.items():
                if isinstance(v, (list, tuple)) and len(v) >= 2:
                    items.append({"label": str(k), "point": [v[0], v[1]]})
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError(f"Unsupported keypoint payload type: {type(raw).__name__}")

    if not isinstance(items, list) or len(items) == 0:
        raise ValueError("No keypoints found in payload.")

    out: List[Dict[str, Any]] = []
    for i, obj in enumerate(items):
        if isinstance(obj, Mapping):
            label = str(obj.get("label", f"point_{i+1}"))
            x, y = _parse_point_xy(obj)
            conf = obj.get("confidence", None)
        elif isinstance(obj, (list, tuple)) and len(obj) >= 2:
            if isinstance(obj[0], str) and len(obj) >= 3:
                label = str(obj[0])
                x, y = _to_float(obj[1]), _to_float(obj[2])
                conf = None
            else:
                label = f"point_{i+1}"
                x, y = _to_float(obj[0]), _to_float(obj[1])
                conf = None
        else:
            continue

        x = _clamp(float(x), 0.0, float(max(0, image_w - 1)))
        y = _clamp(float(y), 0.0, float(max(0, image_h - 1)))
        item: Dict[str, Any] = {"label": label, "point": [x, y]}
        if conf is not None:
            try:
                item["confidence"] = _clamp(float(conf), 0.0, 1.0)
            except Exception:
                pass
        out.append(item)

    if len(out) == 0:
        raise ValueError("Failed to normalize any valid keypoints.")
    return out


def _sanitize_english_label(label: str, idx: int) -> str:
    text = re.sub(r"[^A-Za-z0-9 _-]+", " ", str(label))
    text = re.sub(r"\s+", " ", text).strip().lower()
    if not text:
        return f"point_{idx}"
    return text


def _keypoints_as_semantic_list(keypoints: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, item in enumerate(keypoints, start=1):
        x, y = _parse_point_xy(item)
        label = _sanitize_english_label(str(item.get("label", f"point_{idx}")), idx=idx)
        out.append({"point": [float(x), float(y)], "label": label})
    return out


def _strip_think_tags(text: str) -> str:
    s = str(text).strip()
    if not s:
        return s
    stripped = re.sub(r"<think>[\s\S]*?</think>", "", s, flags=re.IGNORECASE).strip()
    return stripped if stripped else s


def _draw_keypoints_coordinate_plot(
    semantic_keypoints: Sequence[Mapping[str, Any]],
    image_w: int,
    image_h: int,
    save_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 7))
    xs = [float(item["point"][0]) for item in semantic_keypoints]
    ys = [float(item["point"][1]) for item in semantic_keypoints]
    ax.scatter(xs, ys, c="#d04f2b", s=46)

    for i, item in enumerate(semantic_keypoints, start=1):
        x = float(item["point"][0])
        y = float(item["point"][1])
        label = str(item["label"])
        ax.text(x + 6.0, y - 6.0, f"{i}:{label}", fontsize=8, color="#153a5b")

    ax.set_title("2D Keypoints in Pixel Coordinate System")
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")
    ax.set_xlim(0.0, float(max(1, image_w)))
    ax.set_ylim(float(max(1, image_h)), 0.0)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(save_path, dpi=170)
    plt.close(fig)


def _resolve_moge_model_path(path: Path) -> Path:
    p = Path(path)
    if p.is_file():
        return p
    if p.is_dir():
        cand = p / "model.pt"
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Cannot resolve MoGe model file from: {path}")


def _keyword_pick(labels: Sequence[str], keywords: Sequence[str], exclude: Iterable[str] | None = None) -> str | None:
    exclude_set = set(exclude or [])
    for kw in keywords:
        for lb in labels:
            if lb in exclude_set:
                continue
            if kw in lb.lower():
                return lb
    return None


def _infer_grasp_place_labels(
    keypoints_3d: Mapping[str, Sequence[float]],
    task_instruction: str,
) -> Tuple[str, str]:
    labels = list(keypoints_3d.keys())
    if len(labels) < 2:
        raise ValueError("Need at least 2 keypoints for grasp/place inference.")

    text = task_instruction.lower()
    grasp = _keyword_pick(
        labels,
        [
            "grasp_point",
            "pick",
            "grasp",
            "object_center",
            "object",
            "carrot",
            "goose",
            "can",
            "cup",
            "bottle",
            "lid",
            "handle",
        ],
    )
    if grasp is None:
        grasp = labels[0]

    place = _keyword_pick(
        labels,
        [
            "place_point",
            "target_center",
            "target",
            "goal",
            "basket",
            "box",
            "container",
            "inside",
            "tray",
            "plate",
            "table",
        ],
        exclude=[grasp],
    )
    if place is None:
        place = labels[1] if len(labels) > 1 else labels[0]

    if "put into" in text or "inside" in text:
        alt = _keyword_pick(labels, ["inside", "center", "target", "box", "basket"], exclude=[grasp])
        if alt is not None:
            place = alt

    return grasp, place


def heuristic_plan_trace(
    task_instruction: str,
    keypoints_3d: Mapping[str, Sequence[float]],
    hover_offset: float = 0.12,
    place_drop_offset: float = 0.08,
    default_rpy: Tuple[float, float, float] = (3.14159, 0.0, 0.0),
) -> Dict[str, Any]:
    grasp_label, place_label = _infer_grasp_place_labels(keypoints_3d, task_instruction)

    gx, gy, gz = [float(v) for v in keypoints_3d[grasp_label][:3]]
    px, py, pz = [float(v) for v in keypoints_3d[place_label][:3]]
    rx, ry, rz = [float(v) for v in default_rpy]

    pre_grasp = [gx, gy, gz + hover_offset]
    lift = [gx, gy, gz + hover_offset]
    pre_place = [px, py, pz + hover_offset]
    place = [px, py, pz + place_drop_offset]
    retreat = [px, py, pz + hover_offset]

    traj = [
        {
            "description": "move to pre-grasp hover",
            "x": pre_grasp[0],
            "y": pre_grasp[1],
            "z": pre_grasp[2],
            "rx": rx,
            "ry": ry,
            "rz": rz,
            "grip": 0,
        },
        {
            "description": "descend to grasp point",
            "x": gx,
            "y": gy,
            "z": gz,
            "rx": rx,
            "ry": ry,
            "rz": rz,
            "grip": 0,
        },
        {
            "description": "close gripper",
            "x": gx,
            "y": gy,
            "z": gz,
            "rx": rx,
            "ry": ry,
            "rz": rz,
            "grip": 1,
        },
        {
            "description": "lift object",
            "x": lift[0],
            "y": lift[1],
            "z": lift[2],
            "rx": rx,
            "ry": ry,
            "rz": rz,
            "grip": 1,
        },
        {
            "description": "move to pre-place hover",
            "x": pre_place[0],
            "y": pre_place[1],
            "z": pre_place[2],
            "rx": rx,
            "ry": ry,
            "rz": rz,
            "grip": 1,
        },
        {
            "description": "descend to place point",
            "x": place[0],
            "y": place[1],
            "z": place[2],
            "rx": rx,
            "ry": ry,
            "rz": rz,
            "grip": 1,
        },
        {
            "description": "open gripper",
            "x": place[0],
            "y": place[1],
            "z": place[2],
            "rx": rx,
            "ry": ry,
            "rz": rz,
            "grip": 0,
        },
        {
            "description": "retreat to safe height",
            "x": retreat[0],
            "y": retreat[1],
            "z": retreat[2],
            "rx": rx,
            "ry": ry,
            "rz": rz,
            "grip": 0,
        },
    ]
    return {"grasp_label": grasp_label, "place_label": place_label, "trajectory": traj, "source": "heuristic"}


def _normalize_trace_steps(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, dict):
        for key in ("trajectory", "trace", "steps"):
            if key in raw:
                raw = raw[key]
                break
    if not isinstance(raw, list) or len(raw) == 0:
        raise ValueError("Trajectory payload must be a non-empty list.")

    out: List[Dict[str, Any]] = []
    for i, step in enumerate(raw):
        if isinstance(step, Mapping):
            x = float(step["x"])
            y = float(step["y"])
            z = float(step["z"])
            rx = float(step.get("rx", 3.14159))
            ry = float(step.get("ry", 0.0))
            rz = float(step.get("rz", 0.0))
            grip_raw = step.get("grip", 0)
            description = str(step.get("description", f"step_{i+1}"))
        elif isinstance(step, (list, tuple)) and len(step) >= 7:
            x = float(step[0])
            y = float(step[1])
            z = float(step[2])
            rx = float(step[3])
            ry = float(step[4])
            rz = float(step[5])
            grip_raw = step[6]
            description = f"step_{i+1}"
        else:
            raise ValueError(f"trajectory[{i}] must be an object or [x,y,z,rx,ry,rz,grip].")
        if isinstance(grip_raw, bool):
            grip = 1 if grip_raw else 0
        else:
            grip = 1 if float(grip_raw) > 0.5 else 0
        out.append(
            {
                "description": description,
                "x": x,
                "y": y,
                "z": z,
                "rx": rx,
                "ry": ry,
                "rz": rz,
                "grip": grip,
            }
        )
    return out


def _has_valid_grip_toggle(trace: Sequence[Mapping[str, Any]]) -> bool:
    if len(trace) < 3:
        return False
    states = [1 if float(s.get("grip", 0)) > 0.5 else 0 for s in trace]
    saw_close = False
    for prev, cur in zip(states[:-1], states[1:]):
        if prev == 0 and cur == 1:
            saw_close = True
        if saw_close and prev == 1 and cur == 0:
            return True
    return False


def _euler_delta_to_rot_axangle(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float]:
    """Convert delta Euler (roll, pitch, yaw) to axis-angle vector."""

    cr = math.cos(float(roll))
    sr = math.sin(float(roll))
    cp = math.cos(float(pitch))
    sp = math.sin(float(pitch))
    cy = math.cos(float(yaw))
    sy = math.sin(float(yaw))

    # R = Rz(yaw) * Ry(pitch) * Rx(roll)
    r00 = cy * cp
    r01 = cy * sp * sr - sy * cr
    r02 = cy * sp * cr + sy * sr

    r10 = sy * cp
    r11 = sy * sp * sr + cy * cr
    r12 = sy * sp * cr - cy * sr

    r20 = -sp
    r21 = cp * sr
    r22 = cp * cr

    trace = r00 + r11 + r22
    cos_theta = _clamp((trace - 1.0) * 0.5, -1.0, 1.0)
    theta = float(math.acos(cos_theta))

    if theta < 1e-8:
        return (0.0, 0.0, 0.0)

    sin_theta = math.sin(theta)
    if abs(sin_theta) > 1e-8:
        ax = (r21 - r12) / (2.0 * sin_theta)
        ay = (r02 - r20) / (2.0 * sin_theta)
        az = (r10 - r01) / (2.0 * sin_theta)
    else:
        # Near pi: robust axis extraction from diagonal.
        ax = math.sqrt(max(0.0, (r00 + 1.0) * 0.5))
        ay = math.sqrt(max(0.0, (r11 + 1.0) * 0.5))
        az = math.sqrt(max(0.0, (r22 + 1.0) * 0.5))
        ax = math.copysign(ax, (r21 - r12) if abs(r21 - r12) > 1e-12 else 1.0)
        ay = math.copysign(ay, (r02 - r20) if abs(r02 - r20) > 1e-12 else 1.0)
        az = math.copysign(az, (r10 - r01) if abs(r10 - r01) > 1e-12 else 1.0)

    norm = math.sqrt(ax * ax + ay * ay + az * az)
    if norm < 1e-12:
        return (0.0, 0.0, 0.0)

    ax /= norm
    ay /= norm
    az /= norm
    return (ax * theta, ay * theta, az * theta)


def build_simpleenv_actions_from_trace(trace: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if len(trace) == 0:
        raise ValueError("Trace is empty.")

    steps = _normalize_trace_steps(list(trace))
    actions: List[Dict[str, Any]] = []
    prev = steps[0]
    for i, s in enumerate(steps):
        if i == 0:
            dx = dy = dz = 0.0
            drx_euler = dry_euler = drz_euler = 0.0
        else:
            dx = float(s["x"] - prev["x"])
            dy = float(s["y"] - prev["y"])
            dz = float(s["z"] - prev["z"])
            drx_euler = float(s["rx"] - prev["rx"])
            dry_euler = float(s["ry"] - prev["ry"])
            drz_euler = float(s["rz"] - prev["rz"])

        drx_ax, dry_ax, drz_ax = _euler_delta_to_rot_axangle(drx_euler, dry_euler, drz_euler)

        grip_01 = 1.0 if float(s["grip"]) > 0.5 else 0.0
        open_gripper = 0.0 if grip_01 > 0.5 else 1.0
        terminate_episode = 0.0
        grip_simpleenv = 1.0 if grip_01 > 0.5 else -1.0
        action_7d = [dx, dy, dz, drx_ax, dry_ax, drz_ax, grip_simpleenv]
        actions.append(
            {
                "t": i,
                "description": s.get("description", f"step_{i+1}"),
                "pose_6d_abs": [float(s["x"]), float(s["y"]), float(s["z"]), float(s["rx"]), float(s["ry"]), float(s["rz"])],
                "world_vector": [dx, dy, dz],
                "rotation_delta": [drx_euler, dry_euler, drz_euler],
                "rot_axangle": [drx_ax, dry_ax, drz_ax],
                "open_gripper": [open_gripper],
                "gripper_01": grip_01,
                "gripper_simpleenv": grip_simpleenv,
                "gripper_closedness_action": [grip_simpleenv],
                "gripper": [grip_simpleenv],
                "terminate_episode": [terminate_episode],
                "action_model_raw": {
                    "world_vector": [dx, dy, dz],
                    "rotation_delta": [drx_euler, dry_euler, drz_euler],
                    "open_gripper": [open_gripper],
                },
                "action_model_processed": {
                    "world_vector": [dx, dy, dz],
                    "rot_axangle": [drx_ax, dry_ax, drz_ax],
                    "gripper": [grip_simpleenv],
                    "terminate_episode": [terminate_episode],
                },
                "action_7d": action_7d,
                "action_7d_euler_delta": [dx, dy, dz, drx_euler, dry_euler, drz_euler, grip_simpleenv],
            }
        )
        prev = s

    return {
        "action_space": "[dx,dy,dz,d_rx,d_ry,d_rz,gripper]",
        "rotation_repr": "rot_axangle (SimpleEnv evaluator expects axis-angle delta).",
        "gripper_convention": "-1(open), +1(close)",
        "simpleenv_expected_processed_action_keys": ["world_vector", "rot_axangle", "gripper", "terminate_episode"],
        "simpleenv_expected_raw_action_keys": ["world_vector", "rotation_delta", "open_gripper"],
        "env_step_actions_7d": [a["action_7d"] for a in actions],
        "actions": actions,
    }


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


@dataclass
class QwenRuntime:
    model: Any
    processor: Any
    model_device: str


@dataclass
class TextLLMRuntime:
    model: Any
    tokenizer: Any
    model_device: str


def _load_qwen_runtime(model_path: Path) -> QwenRuntime:
    import torch
    import transformers

    model = None
    last_exc: Exception | None = None
    loaders: List[Any] = []
    if hasattr(transformers, "AutoModelForImageTextToText"):
        loaders.append(transformers.AutoModelForImageTextToText)
    if hasattr(transformers, "Qwen2_5_VLForConditionalGeneration"):
        loaders.append(transformers.Qwen2_5_VLForConditionalGeneration)
    if hasattr(transformers, "Qwen2VLForConditionalGeneration"):
        loaders.append(transformers.Qwen2VLForConditionalGeneration)

    for loader in loaders:
        try:
            model = loader.from_pretrained(
                str(model_path),
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
            )
            break
        except Exception as exc:  # pragma: no cover - runtime fallback
            last_exc = exc
            model = None
    if model is None:
        raise RuntimeError(f"Failed to load Qwen model from {model_path}. Last error: {last_exc}")

    processor = transformers.AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    try:
        first_param = next(model.parameters())
        model_device = str(first_param.device)
    except Exception:
        model_device = "cuda" if torch.cuda.is_available() else "cpu"
    return QwenRuntime(model=model, processor=processor, model_device=model_device)


def _ensure_model_dir(model_path: Path, hf_repo: str) -> Path:
    if model_path.exists():
        return model_path
    from huggingface_hub import snapshot_download

    model_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=hf_repo,
        local_dir=str(model_path),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return model_path


def _load_text_llm_runtime(model_path: Path, hf_repo: str) -> TextLLMRuntime:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    local_dir = _ensure_model_dir(model_path=model_path, hf_repo=hf_repo)
    model = AutoModelForCausalLM.from_pretrained(
        str(local_dir),
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(local_dir), trust_remote_code=True)
    try:
        first_param = next(model.parameters())
        model_device = str(first_param.device)
    except Exception:
        model_device = "cuda" if torch.cuda.is_available() else "cpu"
    return TextLLMRuntime(model=model, tokenizer=tokenizer, model_device=model_device)


def _qwen_chat(runtime: QwenRuntime, messages: List[Dict[str, Any]], max_new_tokens: int, temperature: float) -> str:
    import torch
    from qwen_vl_utils import process_vision_info

    text = runtime.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = runtime.processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    if runtime.model_device:
        try:
            inputs = inputs.to(runtime.model_device)
        except Exception:
            pass

    gen_kwargs: Dict[str, Any] = {"max_new_tokens": int(max_new_tokens)}
    if temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = float(temperature)
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        generated_ids = runtime.model.generate(**inputs, **gen_kwargs)
    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    out = runtime.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return out[0] if out else ""


def _text_llm_chat(runtime: TextLLMRuntime, messages: List[Dict[str, Any]], max_new_tokens: int, temperature: float) -> str:
    import torch

    template_kwargs: Dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    try:
        sig = inspect.signature(runtime.tokenizer.apply_chat_template)
        if "enable_thinking" in sig.parameters:
            template_kwargs["enable_thinking"] = False
    except Exception:
        pass
    try:
        prompt = runtime.tokenizer.apply_chat_template(messages, **template_kwargs)
    except TypeError:
        template_kwargs.pop("enable_thinking", None)
        prompt = runtime.tokenizer.apply_chat_template(messages, **template_kwargs)
    inputs = runtime.tokenizer([prompt], return_tensors="pt")
    if runtime.model_device:
        for k in list(inputs.keys()):
            try:
                inputs[k] = inputs[k].to(runtime.model_device)
            except Exception:
                pass

    gen_kwargs: Dict[str, Any] = {"max_new_tokens": int(max_new_tokens)}
    if temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = float(temperature)
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        output_ids = runtime.model.generate(**inputs, **gen_kwargs)
    trimmed = output_ids[:, inputs["input_ids"].shape[1]:]
    text = runtime.tokenizer.batch_decode(trimmed, skip_special_tokens=True)
    return _strip_think_tags(text[0] if text else "")


def _generate_2d_keypoints_with_qwen(
    runtime: QwenRuntime,
    image_path: Path,
    task_instruction: str,
    max_new_tokens: int,
    temperature: float,
    min_keypoints: int,
    prompt_style: str,
) -> Tuple[List[Dict[str, Any]], str, Tuple[int, int]]:
    from PIL import Image

    with Image.open(image_path) as img:
        w, h = img.size

    if prompt_style == "ours_vlm":
        prompt = (
            "你是机器人视觉关键点标注器。\n"
            f"任务：{task_instruction}\n"
            f"图像尺寸：width={w}, height={h}\n"
            "请严格输出 JSON 数组，不要 markdown，不要额外解释。\n"
            "输出格式必须是：["
            "{\"point\": [x, y], \"label\": \"english semantic label\"}"
            "]。\n"
            "要求：\n"
            "1) 用2D坐标组描述出你看到的东西；\n"
            f"2) 尽量详细，至少 {int(min_keypoints)} 个关键点；\n"
            "3) label 必须为英文语义短语；\n"
            "4) 坐标为像素系 XY，并且在图像边界内。"
        )
    else:
        prompt = (
            "You are a robotic manipulation keypoint detector.\n"
            f"Task: {task_instruction}\n"
            f"Image size: width={w}, height={h}\n"
            "Output JSON only. No markdown.\n"
            "Return object with key 'keypoints' as list of points for manipulation planning.\n"
            "Each item must be: {\"label\": string, \"point\": [x, y], \"confidence\": number}.\n"
            "Coordinates must be pixel-space XY (not normalized), within image bounds.\n"
            "Must include labels close to: grasp_point, place_point, object_center, target_center.\n"
            "Keep 6-20 keypoints total."
        )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    raw_text = _qwen_chat(runtime, messages=messages, max_new_tokens=max_new_tokens, temperature=temperature)
    raw_obj = _loads_json_loose(raw_text)
    keypoints = normalize_keypoints_payload(raw_obj, image_w=w, image_h=h)
    semantic = _keypoints_as_semantic_list(keypoints)
    if len(semantic) < int(min_keypoints):
        raise ValueError(f"VLM returned only {len(semantic)} keypoints, below minimum {int(min_keypoints)}.")
    return semantic, raw_text, (w, h)


def _image_to_data_url(image_path: Path) -> str:
    import base64
    import mimetypes

    mime, _ = mimetypes.guess_type(str(image_path))
    if not mime:
        mime = "image/png"
    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _openrouter_chat(
    *,
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    max_new_tokens: int,
    temperature: float,
    site_url: str,
    site_name: str,
    timeout_s: int,
) -> str:
    import requests

    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is empty. Set env OPENROUTER_API_KEY or pass --openrouter-api-key.")
    headers = {"Authorization": f"Bearer {api_key}"}
    if site_url:
        headers["HTTP-Referer"] = site_url
    if site_name:
        headers["X-OpenRouter-Title"] = site_name
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": int(max_new_tokens),
        "temperature": float(temperature),
    }
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=int(timeout_s),
    )
    if resp.status_code >= 300:
        raise RuntimeError(f"OpenRouter request failed ({resp.status_code}): {resp.text}")
    obj = resp.json()
    choices = obj.get("choices", [])
    if not choices:
        raise RuntimeError(f"OpenRouter response has no choices: {obj}")
    msg = choices[0].get("message", {})
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, Mapping) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join([p for p in parts if p])
    return str(content)


def _generate_2d_keypoints_with_openrouter(
    *,
    image_path: Path,
    task_instruction: str,
    model: str,
    api_key: str,
    max_new_tokens: int,
    temperature: float,
    min_keypoints: int,
    site_url: str,
    site_name: str,
    timeout_s: int,
) -> Tuple[List[Dict[str, Any]], str, Tuple[int, int]]:
    from PIL import Image

    with Image.open(image_path) as img:
        w, h = img.size
    data_url = _image_to_data_url(image_path)
    prompt = (
        "你是机器人视觉关键点标注器。\n"
        f"任务：{task_instruction}\n"
        f"图像尺寸：width={w}, height={h}\n"
        "请严格输出 JSON 数组，不要 markdown，不要额外解释。\n"
        "输出格式：[{\"point\": [x, y], \"label\": \"english semantic label\"}]。\n"
        f"尽量详细，至少 {int(min_keypoints)} 个关键点，label 必须为英文。"
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]
    raw_text = _openrouter_chat(
        api_key=api_key,
        model=model,
        messages=messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        site_url=site_url,
        site_name=site_name,
        timeout_s=timeout_s,
    )
    raw_obj = _loads_json_loose(raw_text)
    keypoints = normalize_keypoints_payload(raw_obj, image_w=w, image_h=h)
    semantic = _keypoints_as_semantic_list(keypoints)
    if len(semantic) < int(min_keypoints):
        raise ValueError(f"OpenRouter VLM returned only {len(semantic)} keypoints, below minimum {int(min_keypoints)}.")
    return semantic, raw_text, (w, h)


def _generate_6d_trace_with_qwen(
    runtime: QwenRuntime,
    task_instruction: str,
    keypoints_3d: Mapping[str, Sequence[float]],
    max_new_tokens: int,
    temperature: float,
) -> Tuple[Dict[str, Any], str]:
    keypoints_json = json.dumps(keypoints_3d, ensure_ascii=False, indent=2)
    prompt = (
        "You are a robot motion planner.\n"
        f"Task instruction: {task_instruction}\n"
        "Given 3D keypoints, generate an absolute 6D pose trajectory for a gripper.\n"
        "Output JSON only.\n"
        "Required format:\n"
        "{\n"
        "  \"grasp_label\": \"...\",\n"
        "  \"place_label\": \"...\",\n"
        "  \"trajectory\": [\n"
        "    {\"description\":\"...\", \"x\":..., \"y\":..., \"z\":..., \"rx\":..., \"ry\":..., \"rz\":..., \"grip\":0 or 1}\n"
        "  ]\n"
        "}\n"
        "Rules:\n"
        "- 6~12 steps.\n"
        "- Include at least one close event (grip 0->1) and one open event (grip 1->0).\n"
        "- Prefer rx=3.14159, ry=0, rz=0 unless task demands otherwise.\n"
        "- Keep waypoint motion smooth.\n"
        f"3D keypoints:\n{keypoints_json}\n"
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    raw_text = _qwen_chat(runtime, messages=messages, max_new_tokens=max_new_tokens, temperature=temperature)
    obj = _loads_json_loose(raw_text)
    traj = _normalize_trace_steps(obj)
    grasp = str(obj.get("grasp_label", "")).strip()
    place = str(obj.get("place_label", "")).strip()
    if not grasp or grasp not in keypoints_3d or not place or place not in keypoints_3d:
        g2, p2 = _infer_grasp_place_labels(keypoints_3d, task_instruction=task_instruction)
        if not grasp or grasp not in keypoints_3d:
            grasp = g2
        if not place or place not in keypoints_3d:
            place = p2
    return {"grasp_label": grasp, "place_label": place, "trajectory": traj, "source": "qwen"}, raw_text


def _generate_6d_trace_with_text_llm(
    runtime: TextLLMRuntime,
    task_instruction: str,
    keypoints_3d: Mapping[str, Sequence[float]],
    max_new_tokens: int,
    temperature: float,
) -> Tuple[Dict[str, Any], str]:
    keypoints_json = json.dumps(keypoints_3d, ensure_ascii=False, indent=2)
    prompt = (
        "/no_think\n"
        "你是机械臂轨迹规划器。根据我提供的物体关键点3D坐标，生成机械臂末端执行器的完整任务轨迹。\n"
        f"任务：{task_instruction}。\n"
        "不要输出<think>或任何推理过程。\n"
        "返回的结果是json格式，相机坐标系下给出(x, y, z, rx, ry, rz, grip)序列表示机械臂夹爪位姿。\n"
        "其中x, y, z是空间坐标，rx, ry, rz为欧拉角，grip为夹爪开合状态(0=open, 1=close)。\n"
        "对于机械臂末端执行器，定义其局部坐标系如下：Z轴为抓取接近方向，X轴为夹爪开合方向，Y轴由右手定则确定。\n"
        "XZ平面表示夹爪主平面，机械臂夹爪宽度为0.10m，参考宽度决定主夹爪平面的位置。\n"
        "要求：\n"
        "1) 仅输出 JSON 对象，不要 markdown，不要解释。\n"
        "2) JSON 结构必须是：{\"grasp_label\":\"...\",\"place_label\":\"...\",\"trajectory\":[[x,y,z,rx,ry,rz,grip], ...]}\n"
        "3) trajectory 固定 8 步，数值尽量保留4位小数。\n"
        "4) 必须包含一次抓取闭合(0->1)和一次放置张开(1->0)。\n"
        "5) 轨迹应平滑且非随机，围绕关键点组织。\n"
        f"其中3D坐标为：\n{keypoints_json}\n"
    )
    raw_text = _text_llm_chat(
        runtime=runtime,
        messages=[{"role": "user", "content": prompt}],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    obj = _loads_json_loose(raw_text)
    traj = _normalize_trace_steps(obj)
    grasp = str(obj.get("grasp_label", "")).strip()
    place = str(obj.get("place_label", "")).strip()
    if not grasp or grasp not in keypoints_3d or not place or place not in keypoints_3d:
        g2, p2 = _infer_grasp_place_labels(keypoints_3d, task_instruction=task_instruction)
        if not grasp or grasp not in keypoints_3d:
            grasp = g2
        if not place or place not in keypoints_3d:
            place = p2
    return {"grasp_label": grasp, "place_label": place, "trajectory": traj, "source": "qwen3-8b-instruct"}, raw_text


def _generate_6d_trace_with_openrouter(
    *,
    task_instruction: str,
    keypoints_3d: Mapping[str, Sequence[float]],
    model: str,
    api_key: str,
    max_new_tokens: int,
    temperature: float,
    site_url: str,
    site_name: str,
    timeout_s: int,
) -> Tuple[Dict[str, Any], str]:
    keypoints_json = json.dumps(keypoints_3d, ensure_ascii=False, indent=2)
    prompt = (
        "/no_think\n"
        "你是机械臂轨迹规划器。根据我提供的物体关键点3D坐标，生成机械臂末端执行器的完整任务轨迹。\n"
        f"任务：{task_instruction}。\n"
        "返回JSON对象，包含grasp_label、place_label、trajectory。\n"
        "trajectory使用紧凑数组格式: [[x,y,z,rx,ry,rz,grip], ...]，grip为0/1。\n"
        "末端坐标系定义：Z为approach方向，X为夹爪开合方向，Y按右手定则；夹爪宽度0.10m。\n"
        "轨迹固定8步，应平滑且非随机，且必须包含一次0->1和一次1->0。\n"
        f"其中3D坐标为：\n{keypoints_json}\n"
    )
    raw_text = _openrouter_chat(
        api_key=api_key,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        site_url=site_url,
        site_name=site_name,
        timeout_s=timeout_s,
    )
    obj = _loads_json_loose(raw_text)
    traj = _normalize_trace_steps(obj)
    grasp = str(obj.get("grasp_label", "")).strip()
    place = str(obj.get("place_label", "")).strip()
    if not grasp or grasp not in keypoints_3d or not place or place not in keypoints_3d:
        g2, p2 = _infer_grasp_place_labels(keypoints_3d, task_instruction=task_instruction)
        if not grasp or grasp not in keypoints_3d:
            grasp = g2
        if not place or place not in keypoints_3d:
            place = p2
    return {"grasp_label": grasp, "place_label": place, "trajectory": traj, "source": "openrouter"}, raw_text


def _run_2d_to_3d_pipeline(
    repo_root: Path,
    image_path: Path,
    keypoints_json_path: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> Path:
    depth_source = str(args.depth_source)
    moge_model = _resolve_moge_model_path(Path(args.moge_model))
    cmd: List[str] = [
        sys.executable,
        str(repo_root / "pipeline_2d_to_3d_sam2_moge.py"),
        "--image",
        str(image_path),
        "--keypoints",
        str(keypoints_json_path),
        "--vlm5d-root",
        str(args.vlm5d_root),
        "--sam2-model-dir",
        str(args.sam2_model_dir),
        "--moge-model",
        str(moge_model),
        "--out-dir",
        str(out_dir),
        "--device",
        str(args.device),
        "--point-order",
        "xy",
        "--point-space",
        "pixel",
        "--output-point-order",
        "xy",
        "--output-point-space",
        "pixel",
        "--resize-width",
        str(int(args.resize_width)),
        "--resize-height",
        str(int(args.resize_height)),
        "--depth-source",
        depth_source,
    ]
    if args.disable_sam2:
        cmd.append("--disable-sam2")
    if args.gt_depth_npy is not None:
        cmd.extend(["--gt-depth-npy", str(args.gt_depth_npy)])
    if depth_source == "gt_npy_intrinsics":
        for name in ("camera_fx", "camera_fy", "camera_cx", "camera_cy"):
            if getattr(args, name) is None:
                raise ValueError(f"--depth-source=gt_npy_intrinsics requires --{name.replace('_', '-')}")
        cmd.extend(
            [
                "--camera-fx",
                str(float(args.camera_fx)),
                "--camera-fy",
                str(float(args.camera_fy)),
                "--camera-cx",
                str(float(args.camera_cx)),
                "--camera-cy",
                str(float(args.camera_cy)),
            ]
        )
    subprocess.run(cmd, check=True)
    if depth_source == "moge":
        out_json = out_dir / "keypoints_3d_moge.json"
    else:
        out_json = out_dir / f"keypoints_3d_{depth_source}.json"
    if not out_json.exists():
        raise FileNotFoundError(f"Expected 3D keypoints JSON not found: {out_json}")
    return out_json


def _maybe_run_validation(
    repo_root: Path,
    trace_json: Path,
    keypoints_3d_json: Path,
    out_dir: Path,
    grasp_label: str,
    place_label: str,
) -> Tuple[bool, str]:
    cmd = [
        sys.executable,
        str(repo_root / "validate_trajectory_3d.py"),
        "--trace-json",
        str(trace_json),
        "--keypoints-3d-json",
        str(keypoints_3d_json),
        "--grasp-label",
        grasp_label,
        "--place-label",
        place_label,
        "--out-report-json",
        str(out_dir / "trajectory_validation_report.json"),
        "--out-interactive-html",
        str(out_dir / "trajectory_validation_3d.html"),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    ok = proc.returncode == 0
    msg = (proc.stdout or "") + (proc.stderr or "")
    return ok, msg.strip()


def _maybe_run_environment_execution_test(
    repo_root: Path,
    out_dir: Path,
    simpleenv_actions_json: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"invoked": bool(args.run_env_test)}
    if not args.run_env_test:
        return result

    simpleenv_root = (repo_root / "environment" / "SimplerEnv").resolve()
    maniskill_root = (simpleenv_root / "ManiSkill2_real2sim").resolve()
    runner = (repo_root / "scripts" / "simpleenv_rollout_from_actions.py").resolve()
    report_json = out_dir / "simpleenv_execution_report.json"
    cmd = [
        "conda",
        "run",
        "-n",
        str(args.env_test_conda_env),
        "env",
        f"PYTHONPATH={simpleenv_root}:{maniskill_root}",
        "python",
        str(runner),
        "--actions-json",
        str(simpleenv_actions_json),
        "--task",
        str(args.env_test_task),
        "--seed",
        str(int(args.env_test_seed)),
        "--attempts",
        str(int(args.env_test_attempts)),
        "--out",
        str(report_json),
    ]
    if int(args.env_test_max_steps) > 0:
        cmd.extend(["--max-steps", str(int(args.env_test_max_steps))])

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=int(args.env_test_timeout_s))
    except Exception as exc:
        result.update({"ok": False, "error": str(exc), "command": cmd})
        return result

    env_log = (proc.stdout or "") + (proc.stderr or "")
    (out_dir / "simpleenv_execution_stdout_stderr.txt").write_text(env_log, encoding="utf-8")
    result.update({"ok": proc.returncode == 0, "return_code": int(proc.returncode), "command": cmd})
    if report_json.exists():
        try:
            report_obj = json.loads(report_json.read_text(encoding="utf-8"))
            result["report_json"] = str(report_json)
            result["overall_success"] = bool(report_obj.get("overall_success", False))
            result["episodes"] = report_obj.get("episodes", [])
        except Exception as exc:
            result["report_parse_error"] = str(exc)
    return result


def _build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Virtual pipeline: local Qwen3-VL 2D keypoints -> 2D2Depth/3D -> LLM 6D trace -> SimpleEnv-aligned actions."
    )
    parser.add_argument("--image", type=Path, required=True, help="Input RGB image.")
    parser.add_argument("--task-instruction", type=str, required=True, help="Natural language task instruction.")
    parser.add_argument("--pipeline-option", choices=["legacy", "ours_vlm"], default="ours_vlm")
    parser.add_argument("--qwen-model", "--vlm-model", dest="qwen_model", type=Path, default=Path("/data1/user/ycliu/VLM-5d/models/qwen3-vl-8b"))
    parser.add_argument("--planner-model", type=Path, default=Path("/data1/user/ycliu/VLM-5d/models/qwen3-8b-instruct"))
    parser.add_argument("--planner-model-hf-repo", type=str, default="Qwen/Qwen3-8B-Instruct")
    parser.add_argument("--vlm-backend", choices=["local_qwen_vl", "openrouter"], default="local_qwen_vl")
    parser.add_argument("--planner-backend", choices=["local_qwen3_instruct", "local_qwen_vl", "openrouter"], default="local_qwen3_instruct")
    parser.add_argument("--openrouter-api-key", type=str, default=os.environ.get("OPENROUTER_API_KEY", ""))
    parser.add_argument("--openrouter-site-url", type=str, default="http://localhost")
    parser.add_argument("--openrouter-site-name", type=str, default="VLM-planning-local")
    parser.add_argument("--openrouter-vlm-model", type=str, default="qwen/qwen2.5-vl-7b-instruct")
    parser.add_argument("--openrouter-llm-model", type=str, default="qwen/qwen3-8b")
    parser.add_argument("--openrouter-timeout-s", type=int, default=180)
    parser.add_argument("--min-keypoints", type=int, default=20)
    parser.add_argument("--keypoint-attempts", type=int, default=3)
    parser.add_argument("--planner-attempts", type=int, default=3)
    parser.add_argument("--vlm5d-root", type=Path, default=Path("/data1/user/ycliu/VLM-5d"))
    parser.add_argument("--sam2-model-dir", type=Path, default=Path("/data1/user/ycliu/VLM-5d/models/sam2"))
    parser.add_argument("--moge-model", type=Path, default=Path("/data1/user/ycliu/VLM-5d/models/moge-2"))
    parser.add_argument("--out-dir", type=Path, default=repo_root / "results" / f"virtual_simpleenv_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resize-width", type=int, default=1280)
    parser.add_argument("--resize-height", type=int, default=720)
    parser.add_argument("--disable-sam2", action="store_true")
    parser.add_argument("--depth-source", choices=["moge", "gt_npy", "gt_npy_moge_xy", "gt_npy_intrinsics"], default="moge")
    parser.add_argument("--gt-depth-npy", type=Path, default=None)
    parser.add_argument("--camera-fx", type=float, default=None)
    parser.add_argument("--camera-fy", type=float, default=None)
    parser.add_argument("--camera-cx", type=float, default=None)
    parser.add_argument("--camera-cy", type=float, default=None)
    parser.add_argument("--qwen-keypoint-max-new-tokens", type=int, default=768)
    parser.add_argument("--qwen-plan-max-new-tokens", type=int, default=1024)
    parser.add_argument("--qwen-temperature", type=float, default=0.0)
    parser.add_argument("--run-env-test", action="store_true")
    parser.add_argument("--env-test-conda-env", type=str, default="robotwin")
    parser.add_argument("--env-test-task", type=str, default="google_robot_pick_coke_can")
    parser.add_argument("--env-test-seed", type=int, default=0)
    parser.add_argument("--env-test-attempts", type=int, default=3)
    parser.add_argument("--env-test-max-steps", type=int, default=0)
    parser.add_argument("--env-test-timeout-s", type=int, default=300)
    parser.add_argument("--skip-validation", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    repo_root = Path(__file__).resolve().parent
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    qwen_runtime: QwenRuntime | None = None
    if args.vlm_backend == "local_qwen_vl" or args.planner_backend == "local_qwen_vl":
        qwen_runtime = _load_qwen_runtime(args.qwen_model)

    prompt_style = "ours_vlm" if args.pipeline_option == "ours_vlm" else "legacy"
    min_keypoints = max(int(args.min_keypoints), 20 if args.pipeline_option == "ours_vlm" else 6)
    keypoint_attempt_logs: List[str] = []
    keypoints_2d: List[Dict[str, Any]] | None = None
    raw_keypoint_text = ""
    image_size_xy = (0, 0)
    keypoint_last_exc: Exception | None = None
    for attempt in range(1, int(args.keypoint_attempts) + 1):
        try:
            if args.vlm_backend == "openrouter":
                keypoints_2d, raw_keypoint_text, image_size_xy = _generate_2d_keypoints_with_openrouter(
                    image_path=args.image,
                    task_instruction=args.task_instruction,
                    model=str(args.openrouter_vlm_model),
                    api_key=str(args.openrouter_api_key),
                    max_new_tokens=int(args.qwen_keypoint_max_new_tokens),
                    temperature=float(args.qwen_temperature),
                    min_keypoints=min_keypoints,
                    site_url=str(args.openrouter_site_url),
                    site_name=str(args.openrouter_site_name),
                    timeout_s=int(args.openrouter_timeout_s),
                )
            else:
                if qwen_runtime is None:
                    raise RuntimeError("Local Qwen-VL runtime is not initialized.")
                keypoints_2d, raw_keypoint_text, image_size_xy = _generate_2d_keypoints_with_qwen(
                    runtime=qwen_runtime,
                    image_path=args.image,
                    task_instruction=args.task_instruction,
                    max_new_tokens=int(args.qwen_keypoint_max_new_tokens),
                    temperature=float(args.qwen_temperature),
                    min_keypoints=min_keypoints,
                    prompt_style=prompt_style,
                )
            break
        except Exception as exc:
            keypoint_last_exc = exc
            keypoint_attempt_logs.append(f"[attempt {attempt}] {exc}")
    if keypoints_2d is None:
        raise RuntimeError(f"2D keypoint generation failed after {int(args.keypoint_attempts)} attempts: {keypoint_last_exc}")

    keypoints_2d_json = out_dir / "vlm_keypoints_qwen3vl.json"
    _save_json(keypoints_2d_json, keypoints_2d)
    keypoints_2d_semantic_json = out_dir / "vlm_keypoints_semantic_list.json"
    _save_json(keypoints_2d_semantic_json, keypoints_2d)
    w, h = image_size_xy
    keypoints_plot = out_dir / "vlm_keypoints_coordinate_plot.png"
    _draw_keypoints_coordinate_plot(keypoints_2d, image_w=int(w), image_h=int(h), save_path=keypoints_plot)
    (out_dir / "vlm_keypoints_qwen3vl_raw.txt").write_text(raw_keypoint_text, encoding="utf-8")
    (out_dir / "vlm_keypoints_attempts.log").write_text("\n".join(keypoint_attempt_logs), encoding="utf-8")

    keypoints_3d_json = _run_2d_to_3d_pipeline(
        repo_root=repo_root,
        image_path=args.image,
        keypoints_json_path=keypoints_2d_json,
        out_dir=out_dir,
        args=args,
    )
    keypoints_3d = json.loads(keypoints_3d_json.read_text(encoding="utf-8"))
    if not isinstance(keypoints_3d, dict) or len(keypoints_3d) < 2:
        raise ValueError(f"Invalid 3D keypoints output: {keypoints_3d_json}")

    used_fallback = False
    planner_raw = ""
    planner_attempt_logs: List[str] = []
    trace: List[Dict[str, Any]] | None = None
    grasp_label = ""
    place_label = ""
    text_runtime: TextLLMRuntime | None = None
    if args.planner_backend == "local_qwen3_instruct":
        text_runtime = _load_text_llm_runtime(
            model_path=Path(args.planner_model),
            hf_repo=str(args.planner_model_hf_repo),
        )

    plan_last_exc: Exception | None = None
    for attempt in range(1, int(args.planner_attempts) + 1):
        try:
            if args.planner_backend == "local_qwen3_instruct":
                if text_runtime is None:
                    raise RuntimeError("Local text LLM runtime is not initialized.")
                plan_obj, planner_raw = _generate_6d_trace_with_text_llm(
                    runtime=text_runtime,
                    task_instruction=args.task_instruction,
                    keypoints_3d=keypoints_3d,
                    max_new_tokens=int(args.qwen_plan_max_new_tokens),
                    temperature=float(args.qwen_temperature),
                )
            elif args.planner_backend == "openrouter":
                plan_obj, planner_raw = _generate_6d_trace_with_openrouter(
                    task_instruction=args.task_instruction,
                    keypoints_3d=keypoints_3d,
                    model=str(args.openrouter_llm_model),
                    api_key=str(args.openrouter_api_key),
                    max_new_tokens=int(args.qwen_plan_max_new_tokens),
                    temperature=float(args.qwen_temperature),
                    site_url=str(args.openrouter_site_url),
                    site_name=str(args.openrouter_site_name),
                    timeout_s=int(args.openrouter_timeout_s),
                )
            else:
                if qwen_runtime is None:
                    raise RuntimeError("Local Qwen-VL runtime is not initialized.")
                plan_obj, planner_raw = _generate_6d_trace_with_qwen(
                    runtime=qwen_runtime,
                    task_instruction=args.task_instruction,
                    keypoints_3d=keypoints_3d,
                    max_new_tokens=int(args.qwen_plan_max_new_tokens),
                    temperature=float(args.qwen_temperature),
                )
            trace = plan_obj["trajectory"]
            if not _has_valid_grip_toggle(trace):
                raise ValueError("LLM trajectory lacks close/open grip toggle.")
            grasp_label = str(plan_obj["grasp_label"])
            place_label = str(plan_obj["place_label"])
            break
        except Exception as exc:
            plan_last_exc = exc
            planner_attempt_logs.append(f"[attempt {attempt}] {exc}")

    if trace is None:
        used_fallback = True
        fallback = heuristic_plan_trace(task_instruction=args.task_instruction, keypoints_3d=keypoints_3d)
        trace = fallback["trajectory"]
        grasp_label = str(fallback["grasp_label"])
        place_label = str(fallback["place_label"])
        planner_raw = f"[FALLBACK] {plan_last_exc}"

    trace_json_path = out_dir / "trace_6d_pose.json"
    _save_json(trace_json_path, trace)
    _save_json(out_dir / "trace_6d_pose_qwen.json", trace)
    (out_dir / "trajectory_planner_raw.txt").write_text(planner_raw, encoding="utf-8")
    (out_dir / "trajectory_planner_attempts.log").write_text("\n".join(planner_attempt_logs), encoding="utf-8")

    simpleenv_actions = build_simpleenv_actions_from_trace(trace)
    simpleenv_actions_json = out_dir / "simpleenv_aligned_actions.json"
    _save_json(simpleenv_actions_json, simpleenv_actions)
    simpleenv_rollout_input = {
        "task_description": args.task_instruction,
        "observation": {
            "image_path": str(args.image),
            "note": "Single-image virtual observation; no simulator state is used.",
        },
        "action_sequence_processed": [a["action_model_processed"] for a in simpleenv_actions["actions"]],
        "action_sequence_raw": [a["action_model_raw"] for a in simpleenv_actions["actions"]],
        "env_step_actions_7d": simpleenv_actions["env_step_actions_7d"],
    }
    simpleenv_rollout_input_json = out_dir / "simpleenv_rollout_input.json"
    _save_json(simpleenv_rollout_input_json, simpleenv_rollout_input)

    env_test_result = _maybe_run_environment_execution_test(
        repo_root=repo_root,
        out_dir=out_dir,
        simpleenv_actions_json=simpleenv_actions_json,
        args=args,
    )

    validation_ok = None
    validation_log = ""
    if not args.skip_validation:
        validation_ok, validation_log = _maybe_run_validation(
            repo_root=repo_root,
            trace_json=trace_json_path,
            keypoints_3d_json=keypoints_3d_json,
            out_dir=out_dir,
            grasp_label=grasp_label,
            place_label=place_label,
        )
        (out_dir / "trajectory_validation_stdout_stderr.txt").write_text(validation_log, encoding="utf-8")

    bundle = {
        "task_instruction": args.task_instruction,
        "image": str(args.image),
        "pipeline_option": str(args.pipeline_option),
        "vlm_backend": str(args.vlm_backend),
        "planner_backend": str(args.planner_backend),
        "qwen_model": str(args.qwen_model),
        "planner_model": str(args.planner_model),
        "keypoints_2d_json": str(keypoints_2d_json),
        "keypoints_2d_semantic_json": str(keypoints_2d_semantic_json),
        "keypoints_2d_coordinate_plot": str(keypoints_plot),
        "keypoints_3d_json": str(keypoints_3d_json),
        "trace_6d_pose_json": str(trace_json_path),
        "simpleenv_aligned_actions_json": str(simpleenv_actions_json),
        "simpleenv_rollout_input_json": str(simpleenv_rollout_input_json),
        "grasp_label": grasp_label,
        "place_label": place_label,
        "planner_fallback_used": used_fallback,
        "validation_ok": validation_ok,
        "environment_execution": env_test_result,
        "notes": "Trajectory is deterministic (model-planned + parsed), converted to non-random SimpleEnv-aligned action sequence.",
    }
    _save_json(out_dir / "simpleenv_virtual_input_bundle.json", bundle)

    print(f"[OK] virtual pipeline finished: {out_dir}")
    print(f"[OUT] 2D keypoints: {keypoints_2d_json}")
    print(f"[OUT] 2D coordinate plot: {keypoints_plot}")
    print(f"[OUT] 3D keypoints: {keypoints_3d_json}")
    print(f"[OUT] 6D trace: {trace_json_path}")
    print(f"[OUT] simpleenv actions: {simpleenv_actions_json}")
    print(f"[OUT] bundle: {out_dir / 'simpleenv_virtual_input_bundle.json'}")
    if args.run_env_test:
        print(f"[ENV TEST] {env_test_result}")
    if validation_ok is not None:
        print(f"[VALIDATION] ok={validation_ok}")


if __name__ == "__main__":
    main()
