#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, List, Mapping, Sequence

import numpy as np

# Follow SimplerEnv convention in headless servers.
os.environ.setdefault("DISPLAY", "")


def _load_action_sequence(actions_json: Path) -> List[List[float]]:
    raw = json.loads(actions_json.read_text(encoding="utf-8"))
    seq: Any = None
    if isinstance(raw, Mapping):
        if "env_step_actions_7d" in raw:
            seq = raw["env_step_actions_7d"]
        elif "actions" in raw and isinstance(raw["actions"], list):
            seq = [obj.get("action_7d", []) for obj in raw["actions"]]
    elif isinstance(raw, list):
        seq = raw

    if not isinstance(seq, list) or len(seq) == 0:
        raise ValueError(f"Cannot parse 7D action sequence from: {actions_json}")

    out: List[List[float]] = []
    for i, item in enumerate(seq):
        if not isinstance(item, Sequence):
            raise ValueError(f"Action #{i} is not a sequence: {item}")
        arr = [float(x) for x in list(item)[:7]]
        if len(arr) < 7:
            arr = arr + [0.0] * (7 - len(arr))
        out.append(arr)
    return out


def _ensure_import_path(repo_root: Path) -> None:
    candidates = [
        repo_root / "environment" / "SimplerEnv",
        repo_root / "environment" / "SimplerEnv" / "ManiSkill2_real2sim",
    ]
    for p in candidates:
        s = str(p.resolve())
        if s not in sys.path:
            sys.path.insert(0, s)


def run_rollout(
    *,
    actions: Sequence[Sequence[float]],
    task: str,
    seed: int,
    attempts: int,
    max_steps: int,
) -> dict[str, Any]:
    import simpler_env

    episodes: List[dict[str, Any]] = []
    overall_success = False
    for k in range(max(1, int(attempts))):
        env = simpler_env.make(
            task,
            obs_mode="rgbd",
            robot="google_robot_static",
            sim_freq=513,
            control_freq=3,
            renderer_kwargs={"offscreen_only": True},
            max_episode_steps=max(max_steps, len(actions), 1),
            prepackaged_config=True,
            rgb_overlay_path=None,
        )
        obs, info = env.reset(seed=int(seed) + k)
        ep_success = False
        reward_sum = 0.0
        steps = 0
        done = False
        truncated = False
        for t, act in enumerate(actions):
            action = np.asarray(act, dtype=np.float32)
            obs, reward, done, truncated, info = env.step(action)
            steps = t + 1
            reward_sum += float(reward)
            if bool(info.get("success", False)):
                ep_success = True
            if done or truncated:
                break

        env.close()
        episodes.append(
            {
                "attempt": k + 1,
                "seed": int(seed) + k,
                "steps_executed": int(steps),
                "reward_sum": float(reward_sum),
                "done": bool(done),
                "truncated": bool(truncated),
                "success": bool(ep_success),
                "episode_stats": info.get("episode_stats", {}) if isinstance(info, Mapping) else {},
            }
        )
        if ep_success:
            overall_success = True

    return {
        "overall_success": bool(overall_success),
        "attempts": int(attempts),
        "task": task,
        "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay generated 7D action sequence in SimplerEnv.")
    parser.add_argument("--actions-json", type=Path, required=True)
    parser.add_argument("--task", type=str, default="google_robot_pick_coke_can")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    _ensure_import_path(repo_root)
    actions = _load_action_sequence(args.actions_json)
    rollout = run_rollout(
        actions=actions,
        task=str(args.task),
        seed=int(args.seed),
        attempts=int(args.attempts),
        max_steps=int(args.max_steps),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rollout, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"overall_success={rollout['overall_success']}")
    print(f"report={args.out.resolve()}")


if __name__ == "__main__":
    main()
