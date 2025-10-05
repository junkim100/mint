#!/usr/bin/env python
"""Script to train Mechanistic Tool Editor (MTE)."""

from fire import Fire

from mint.config import load_config
from mint.train.train_mte import train_mte
from mint.logging_utils import setup_logger

logger = setup_logger(__name__)


def main(config_path: str = "configs/default.yaml"):
    """Train MTE.

    Args:
        config_path: Path to MINT config file
    """
    logger.info("=" * 80)
    logger.info("MINT: Training Mechanistic Tool Editor")
    logger.info("=" * 80)

    # Load config
    config = load_config(config_path)
    logger.info(f"Loaded config from {config_path}")
    logger.info(f"Run name: {config.run_name}")
    logger.info(f"Pairs directory: {config.data.pairs_dir}")
    logger.info(f"Layers: {config.mte.layers}")
    logger.info(f"Learning rate: {config.mte.lr}")
    logger.info(f"Batch size: {config.mte.batch_size}")
    logger.info(f"Training steps: {config.mte.steps}")
    logger.info(f"Save directory: {config.mte.save_dir}")

    # Train
    logger.info("Starting training...")
    mte = train_mte(config)

    logger.info("=" * 80)
    logger.info("Training complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    Fire(main)
