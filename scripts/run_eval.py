#!/usr/bin/env python3
import os
import sys
import subprocess
import shlex
from typing import Any, Dict, List

import yaml

CONFIG_PATH = "configs/default.yaml"


def _read_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _ensure_list_task(task_field) -> List[str]:
    if task_field is None:
        raise ValueError("eval.task must be set (as a list), e.g., task: [\"tau_bench\"]")
    if isinstance(task_field, list):
        return [str(t) for t in task_field]
    # if provided as a single string, coerce to list
    return [str(task_field)]


def _compose_model_args(model_cfg: Dict[str, Any], eval_cfg: Dict[str, Any]) -> str:
    # Required: pretrained/tokenizer come from model config
    pretrained = model_cfg.get("pretrained") or model_cfg.get("name")
    tokenizer = model_cfg.get("tokenizer") or pretrained
    if not pretrained:
        raise ValueError("model.pretrained or model.name must be set in the config")

    # Additional args come from eval.model_args mapping
    args_map = eval_cfg.get("model_args") or {}

    # Build ordered list of key=value pairs: pretrained, tokenizer, then the rest in YAML order
    parts: List[str] = [
        f"pretrained={pretrained}",
        f"tokenizer={tokenizer}",
    ]
    for k, v in args_map.items():
        # Booleans/None/numbers/strings -> string repr expected by evalchemy
        if isinstance(v, bool):
            sval = "True" if v else "False"
        elif v is None:
            sval = "None"
        else:
            sval = str(v)
        parts.append(f"{k}={sval}")

    return ",".join(parts)


def run() -> None:
    cfg = _read_config(CONFIG_PATH)

    eval_cfg: Dict[str, Any] = cfg.get("eval", {}) or {}
    model_cfg: Dict[str, Any] = cfg.get("model", {}) or {}

    tasks = _ensure_list_task(eval_cfg.get("task"))
    backend = eval_cfg.get("model_backend", "vllm")
    batch_size = str(eval_cfg.get("batch_size", "auto"))
    output_dir = eval_cfg.get("output_dir", "results/eval")
    os.makedirs(output_dir, exist_ok=True)

    model_args = _compose_model_args(model_cfg, eval_cfg)

    cmd = [
        sys.executable,
        "-m",
        "eval.eval",
        "--model",
        backend,
        "--tasks",
        ",".join(tasks),
        "--batch_size",
        batch_size,
        "--output_path",
        output_dir,
        "--model_args",
        model_args,
    ]

    print("[run_eval] Running:")
    print(" ", " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    # No CLI args: everything is driven by configs/default.yaml
    run()
