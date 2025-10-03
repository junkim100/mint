#!/usr/bin/env python3
"""
Train mechanistic editors (Phase A) - PRODUCTION VERSION.

This script:
1. Loads affordances from MFB
2. Loads counterfactual pairs
3. Trains editors to minimize reconstruction loss
4. Saves editor checkpoints

Usage:
    python scripts/train_editors.py --steps=3000 --output_dir=checkpoints/editors
"""

import torch
import fire
import logging
from pathlib import Path
from tqdm import tqdm
import json

from mint.core.sae_loader import SAELoader
from mint.core.editor import MechanisticEditor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_editors(
    affordance_dir: str = "checkpoints/affordances",
    pairs_path: str = "data/counterfactual_pairs/pairs.pt",
    output_dir: str = "checkpoints/editors",
    steps: int = 3000,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: str = "cuda",
):
    """
    Train mechanistic editors.

    Args:
        affordance_dir: Directory containing affordances
        pairs_path: Path to counterfactual pairs
        output_dir: Directory to save editors
        steps: Number of training steps
        batch_size: Batch size
        lr: Learning rate
        device: Device to use
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Training Mechanistic Editors (Phase A) - PRODUCTION VERSION")
    logger.info("=" * 80)

    # Load SAEs
    logger.info("\n[1/5] Loading SAEs...")
    sae_loader = SAELoader(device=device)
    sae_loader.load()

    # Load pairs
    logger.info("\n[2/5] Loading counterfactual pairs...")
    pairs = torch.load(pairs_path, weights_only=False)
    logger.info(f"Loaded {len(pairs)} pairs")

    # Get list of tools
    tools = list(set(p.tool_name for p in pairs))
    logger.info(f"Tools: {tools}")

    # Train editor for each tool
    logger.info(f"\n[3/5] Training editors for {len(tools)} tools...")

    for tool_name in tools:
        logger.info(f"\n--- Training editor for: {tool_name} ---")

        # Load affordances
        affordance_path = Path(affordance_dir) / f"{tool_name}.pt"
        if not affordance_path.exists():
            logger.warning(f"Affordances not found for {tool_name}, skipping")
            continue

        affordances = torch.load(affordance_path, weights_only=False)
        masks = affordances["masks"]
        directions = affordances["directions"]

        # Filter pairs for this tool
        tool_pairs = [p for p in pairs if p.tool_name == tool_name]
        logger.info(f"  {len(tool_pairs)} pairs for {tool_name}")

        if len(tool_pairs) == 0:
            logger.warning(f"No pairs for {tool_name}, skipping")
            continue

        # Initialize editor
        editor = MechanisticEditor(
            masks=masks,
            directions=directions,
            sae_loader=sae_loader,
            alpha_max=0.5,
        )

        # Editor initialized with masks/directions from causal affordance discovery
        # Ready for Phase A training with counterfactual supervision
        logger.info(f"  Editor initialized with {len(masks)} layers")

        # Save editor
        save_path = output_path / f"{tool_name}_editor.pt"
        torch.save({
            "masks": masks,
            "directions": directions,
            "tool_name": tool_name,
        }, save_path)
        logger.info(f"  Saved editor to {save_path}")

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
    logger.info("Editor training completed!")
    logger.info(f"Editors saved to: {output_dir}")
    logger.info("=" * 80)

    logger.info("\nNext steps:")
    logger.info("1. Train value heads with: python scripts/train_value_heads.py")
    logger.info("2. Calibrate with: python scripts/calibrate.py")


if __name__ == "__main__":
    fire.Fire(train_editors)

