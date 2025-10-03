#!/usr/bin/env python3
"""
Conformal calibration (Phase C) - PRODUCTION VERSION.

This script:
1. Loads trained value heads
2. Loads calibration pairs
3. Computes conformal prediction quantiles
4. Saves calibrators

Usage:
    python scripts/calibrate.py --alpha=0.1 --output_dir=checkpoints/calibrators
"""

import torch
import fire
import logging
from pathlib import Path
import json

from mint.training.phase_b import ValueHead
from mint.training.phase_c import ConformalCalibrationTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calibrate(
    pairs_path: str = "data/counterfactual_pairs/pairs.pt",
    value_head_dir: str = "checkpoints/value_heads",
    output_dir: str = "checkpoints/calibrators",
    alpha: float = 0.1,
    batch_size: int = 32,
    device: str = "cuda",
):
    """
    Conformal calibration.

    Args:
        pairs_path: Path to calibration pairs
        value_head_dir: Directory containing value heads
        output_dir: Directory to save calibrators
        alpha: Miscoverage rate (e.g., 0.1 for 90% coverage)
        batch_size: Batch size
        device: Device to use
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Conformal Calibration (Phase C) - PRODUCTION VERSION")
    logger.info("=" * 80)

    # Load pairs
    logger.info("\n[1/3] Loading calibration pairs...")
    pairs = torch.load(pairs_path, weights_only=False)
    logger.info(f"Loaded {len(pairs)} pairs")

    # Get list of tools
    tools = list(set(p.tool_name for p in pairs))
    logger.info(f"Tools: {tools}")

    # Calibrate for each tool
    logger.info(f"\n[2/3] Calibrating for {len(tools)} tools...")

    for tool_name in tools:
        logger.info(f"\n--- Calibrating for: {tool_name} ---")

        # Load value head
        value_head_path = Path(value_head_dir) / f"{tool_name}_value_head.pt"
        if not value_head_path.exists():
            logger.warning(f"Value head not found for {tool_name}, skipping")
            continue

        checkpoint = torch.load(value_head_path, weights_only=False)

        # Initialize value head
        value_head = ValueHead(
            hidden_size=4096,
            num_r_layers=8,
            hidden_dims=[2048, 256],
            dropout=0.1,
        )
        value_head.load_state_dict(checkpoint["model_state_dict"])
        value_head.to(device=device, dtype=torch.bfloat16)

        # Filter pairs for this tool
        tool_pairs = [p for p in pairs if p.tool_name == tool_name]
        logger.info(f"  {len(tool_pairs)} pairs for {tool_name}")

        if len(tool_pairs) == 0:
            logger.warning(f"No pairs for {tool_name}, skipping")
            continue

        # Initialize calibration trainer
        calibration_trainer = ConformalCalibrationTrainer(
            value_head=value_head,
            device=device,
        )

        # Calibrate
        logger.info(f"  Calibrating with α={alpha}...")
        calibrator = calibration_trainer.calibrate(
            pairs=tool_pairs,
            alpha=alpha,
            batch_size=batch_size,
        )

        # Save calibrator
        save_path = output_path / f"{tool_name}_calibrator.pt"
        calibrator.save(str(save_path))
        logger.info(f"  Saved calibrator to {save_path}")

    # Save metadata
    metadata = {
        "tools": tools,
        "alpha": alpha,
        "num_pairs": len(pairs),
    }

    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("\n" + "=" * 80)
    logger.info("Conformal calibration completed!")
    logger.info(f"Calibrators saved to: {output_dir}")
    logger.info("=" * 80)

    logger.info("\nNext steps:")
    logger.info("1. Run MINT inference with: python scripts/evaluate_mint.py")


if __name__ == "__main__":
    fire.Fire(calibrate)

