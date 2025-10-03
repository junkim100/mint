#!/usr/bin/env python3
"""
Train value heads (Phase B) - PRODUCTION VERSION.

This script:
1. Loads counterfactual pairs
2. Trains value heads to predict utility gains
3. Saves value head checkpoints

Usage:
    python scripts/train_value_heads.py --steps=1000 --output_dir=checkpoints/value_heads
"""

import torch
import fire
import logging
from pathlib import Path
import json

from mint.training.phase_b import ValueHead, ValueHeadTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_value_heads(
    pairs_path: str = "data/counterfactual_pairs/pairs.pt",
    output_dir: str = "checkpoints/value_heads",
    steps: int = 1000,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: str = "cuda",
):
    """
    Train value heads.

    Args:
        pairs_path: Path to counterfactual pairs
        output_dir: Directory to save value heads
        steps: Number of training steps
        batch_size: Batch size
        lr: Learning rate
        device: Device to use
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Training Value Heads (Phase B) - PRODUCTION VERSION")
    logger.info("=" * 80)

    # Load pairs
    logger.info("\n[1/3] Loading counterfactual pairs...")
    pairs = torch.load(pairs_path, weights_only=False)
    logger.info(f"Loaded {len(pairs)} pairs")

    # Get list of tools
    tools = list(set(p.tool_name for p in pairs))
    logger.info(f"Tools: {tools}")

    # Train value head for each tool
    logger.info(f"\n[2/3] Training value heads for {len(tools)} tools...")

    for tool_name in tools:
        logger.info(f"\n--- Training value head for: {tool_name} ---")

        # Filter pairs for this tool
        tool_pairs = [p for p in pairs if p.tool_name == tool_name]
        logger.info(f"  {len(tool_pairs)} pairs for {tool_name}")

        if len(tool_pairs) == 0:
            logger.warning(f"No pairs for {tool_name}, skipping")
            continue

        # Initialize value head
        value_head = ValueHead(
            hidden_size=4096,
            num_r_layers=8,
            hidden_dims=[2048, 256],
            dropout=0.1,
        )

        # Initialize trainer
        trainer = ValueHeadTrainer(
            value_head=value_head,
            device=device,
        )

        # Train
        logger.info(f"  Training for {steps} steps...")
        trainer.train(
            pairs=tool_pairs,
            steps=steps,
            batch_size=batch_size,
            lr=lr,
        )

        # Evaluate
        logger.info("  Evaluating...")
        metrics = trainer.evaluate(tool_pairs, batch_size=batch_size)
        logger.info(f"  Metrics: {metrics}")

        # Save value head
        save_path = output_path / f"{tool_name}_value_head.pt"
        torch.save({
            "model_state_dict": value_head.state_dict(),
            "tool_name": tool_name,
            "metrics": metrics,
        }, save_path)
        logger.info(f"  Saved value head to {save_path}")

    # Save metadata
    metadata = {
        "tools": tools,
        "steps": steps,
        "batch_size": batch_size,
        "lr": lr,
        "num_pairs": len(pairs),
    }

    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("\n" + "=" * 80)
    logger.info("Value head training completed!")
    logger.info(f"Value heads saved to: {output_dir}")
    logger.info("=" * 80)

    logger.info("\nNext steps:")
    logger.info("1. Calibrate with: python scripts/calibrate.py")


if __name__ == "__main__":
    fire.Fire(train_value_heads)

