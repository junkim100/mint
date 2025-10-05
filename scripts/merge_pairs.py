#!/usr/bin/env python
"""Merge counterfactual pairs.

Two modes are supported:
1) Scenario-sharded mode: merge pairs from per-GPU directories: data/pairs_gpu{N}
2) Layer-sharded mode: merge per-layer outputs from data/pairs/layer_{L}/pair_{i}.pt into full pairs
"""

from pathlib import Path
import shutil
from fire import Fire
import torch

from mint.config import load_config
from mint.logging_utils import setup_logger

logger = setup_logger(__name__)


def merge_scenario_shards(base_dir: Path, num_gpus: int) -> int:
    total_pairs = 0
    for gpu_id in range(num_gpus):
        gpu_dir = Path(f"{base_dir}_gpu{gpu_id}")
        if not gpu_dir.exists():
            logger.warning(f"GPU {gpu_id} directory not found: {gpu_dir}")
            continue
        pair_files = sorted(gpu_dir.glob("pair_*.pt"))
        logger.info(f"GPU {gpu_id}: Found {len(pair_files)} pairs")
        for pair_file in pair_files:
            new_name = f"pair_{total_pairs:06d}.pt"
            new_path = base_dir / new_name
            shutil.copy(pair_file, new_path)
            total_pairs += 1
    return total_pairs


def merge_layer_shards(base_dir: Path, layers: list[int], max_pairs: int) -> int:
    # Expect directories: base_dir/layer_{L}
    layer_dirs = [base_dir / f"layer_{L}" for L in layers]
    for d in layer_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Missing layer output directory: {d}")
    total_pairs = 0
    for idx in range(max_pairs):
        # Load all layers for this index
        phi_no_tool = {}
        phi_with_tool = {}
        h_no_tool = {}
        h_with_tool = {}
        tool_label = None
        metadata = None
        for L in layers:
            p = base_dir / f"layer_{L}" / f"pair_{idx:06d}.pt"
            if not p.exists():
                logger.warning(f"Missing file for idx={idx}, layer={L}: {p}")
                break
            data = torch.load(p)
            # Merge dicts (each file contains only its own layer)
            for k in ["phi_no_tool", "phi_with_tool", "h_no_tool", "h_with_tool"]:
                dct = locals()[k]
                # Each incoming dict likely has a single key {L: tensor}
                for layer_id, tensor in data[k].items():
                    dct[int(layer_id)] = tensor
            # metadata/tool_label consistent across layers
            tool_label = data.get("tool_label", tool_label)
            metadata = data.get("metadata", metadata)
        else:
            # Only if loop didn't break: save merged
            out_path = base_dir / f"pair_{total_pairs:06d}.pt"
            torch.save(
                {
                    "phi_no_tool": phi_no_tool,
                    "phi_with_tool": phi_with_tool,
                    "h_no_tool": h_no_tool,
                    "h_with_tool": h_with_tool,
                    "tool_label": tool_label,
                    "metadata": metadata,
                },
                out_path,
            )
            total_pairs += 1
    return total_pairs


def main(config_path: str = "configs/default.yaml", num_gpus: int = 8, mode: str = "auto"):
    """Merge pairs into a single directory.

    Args:
        config_path: Path to MINT config file
        num_gpus: Number of GPUs used (for scenario-sharded mode)
        mode: "auto" | "scenario" | "layer"
    """
    logger.info("=" * 80)
    logger.info("Merging pairs")
    logger.info("=" * 80)

    # Load config
    config = load_config(config_path)
    base_dir = Path(config.data.pairs_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # Detect mode
    if mode == "auto":
        if any((base_dir.parent / f"{base_dir.name}_gpu{i}").exists() for i in range(num_gpus)):
            mode = "scenario"
        elif any((base_dir / f"layer_{L}").exists() for L in config.mte.layers):
            mode = "layer"
        else:
            mode = "scenario"
    logger.info(f"Merge mode: {mode}")

    total_pairs = 0
    if mode == "scenario":
        total_pairs = merge_scenario_shards(base_dir, num_gpus)
        # Clean up GPU-specific directories
        logger.info("Cleaning up GPU-specific directories...")
        for gpu_id in range(num_gpus):
            gpu_dir = Path(f"{base_dir}_gpu{gpu_id}")
            if gpu_dir.exists():
                shutil.rmtree(gpu_dir)
                logger.info(f"  Removed {gpu_dir}")
    else:
        total_pairs = merge_layer_shards(base_dir, config.mte.layers, config.data.max_pairs)
        # Clean up layer-specific directories
        logger.info("Cleaning up layer-specific directories...")
        for L in config.mte.layers:
            d = base_dir / f"layer_{L}"
            if d.exists():
                shutil.rmtree(d)
                logger.info(f"  Removed {d}")

    logger.info("=" * 80)
    logger.info(f"Merged {total_pairs} pairs into {base_dir}")
    logger.info("=" * 80)
    logger.info("Merge complete!")


if __name__ == "__main__":
    Fire(main)
