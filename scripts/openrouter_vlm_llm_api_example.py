#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
from pathlib import Path
from typing import Any, Dict, List

import requests


def image_to_data_url(image_path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(image_path))
    if not mime:
        mime = "image/png"
    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def openrouter_chat(
    *,
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int = 1024,
    temperature: float = 0.0,
    referer: str = "http://localhost",
    title: str = "VLM-planning-openrouter-example",
) -> str:
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is empty. Fill it when you want to run this script.")
    headers = {"Authorization": f"Bearer {api_key}"}
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-OpenRouter-Title"] = title
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
    }
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=180,
    )
    resp.raise_for_status()
    obj = resp.json()
    content = obj["choices"][0]["message"]["content"]
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(x.get("text", "")) for x in content if isinstance(x, dict))
    return str(content)


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenRouter API example for VLM keypoints + LLM 6D trajectory.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--keypoints-3d-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--vlm-model", type=str, default="qwen/qwen2.5-vl-7b-instruct")
    parser.add_argument("--llm-model", type=str, default="qwen/qwen3-8b")
    args = parser.parse_args()

    key = args.api_key.strip()
    if not key:
        print("[INFO] API key is empty by design. Fill --api-key or OPENROUTER_API_KEY to run real requests.")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    keypoints_3d = json.loads(args.keypoints_3d_json.read_text(encoding="utf-8"))
    img_url = image_to_data_url(args.image)

    vlm_prompt = (
        "你是机器人视觉关键点标注器。"
        "用2D坐标组描述出你看到的东西。尽量详细，至少20个关键点，并配有语义信息。"
        "请严格输出JSON数组: [{\"point\":[x,y],\"label\":\"english semantic label\"}]。"
    )
    vlm_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": vlm_prompt},
                {"type": "image_url", "image_url": {"url": img_url}},
            ],
        }
    ]
    vlm_raw = openrouter_chat(api_key=key, model=args.vlm_model, messages=vlm_messages)
    (args.out_dir / "openrouter_vlm_raw.txt").write_text(vlm_raw, encoding="utf-8")

    llm_prompt = (
        "你是机械臂轨迹规划器。根据我提供的物体关键点3D坐标，生成机械臂末端执行器的完整任务轨迹。\n"
        f"任务：{args.task}\n"
        "返回JSON格式，给出(x,y,z,rx,ry,rz,grip)序列，grip为0或1。\n"
        "输出格式: {\"grasp_label\":\"...\",\"place_label\":\"...\",\"trajectory\":[{\"x\":..,\"y\":..,\"z\":..,\"rx\":..,\"ry\":..,\"rz\":..,\"grip\":0}]}\n"
        "不要输出解释。\n"
        f"其中3D坐标为：\n{json.dumps(keypoints_3d, ensure_ascii=False, indent=2)}"
    )
    llm_messages = [{"role": "user", "content": llm_prompt}]
    llm_raw = openrouter_chat(api_key=key, model=args.llm_model, messages=llm_messages)
    (args.out_dir / "openrouter_llm_raw.txt").write_text(llm_raw, encoding="utf-8")

    print(f"[OK] wrote: {args.out_dir / 'openrouter_vlm_raw.txt'}")
    print(f"[OK] wrote: {args.out_dir / 'openrouter_llm_raw.txt'}")


if __name__ == "__main__":
    main()

