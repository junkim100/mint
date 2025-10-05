"""Training loop for Mechanistic Tool Editor (MTE)."""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from mint.config import MintConfig
from mint.data.counterfactual_pairs import load_pairs
from mint.models.mte import MechanisticToolEditor, compute_mte_loss
from mint.models.sae_loader import decode_features
from mint.train.datasets import create_dataloader
from mint.utils.seeds import set_seed
from mint.logging_utils import setup_logger

logger = setup_logger(__name__)


def train_mte(
    config: MintConfig,
    pairs_dir: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
) -> MechanisticToolEditor:
    """Train the Mechanistic Tool Editor.

    Args:
        config: MINT configuration
        pairs_dir: Directory containing counterfactual pairs (overrides config)
        checkpoint_path: Optional checkpoint to resume from

    Returns:
        Trained MTE model
    """
    # Set seed
    set_seed(config.seed)

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training MTE on device: {device}")

    # Load pairs
    pairs_dir = pairs_dir or config.data.pairs_dir
    logger.info(f"Loading pairs from {pairs_dir}")
    pairs = load_pairs(pairs_dir)

    if not pairs:
        raise ValueError(f"No pairs found in {pairs_dir}")

    logger.info(f"Loaded {len(pairs)} pairs")

    # Create dataloader
    dataloader = create_dataloader(
        pairs=pairs,
        layers=config.mte.layers,
        batch_size=config.mte.batch_size,
        shuffle=True,
    )

    # Initialize MTE
    feature_dim = config.mte.hidden_dim * config.mte.hidden_expansion

    if checkpoint_path and Path(checkpoint_path).exists():
        logger.info(f"Loading MTE from checkpoint: {checkpoint_path}")
        mte = MechanisticToolEditor.load(checkpoint_path, device=device)
    else:
        logger.info("Initializing new MTE")
        mte = MechanisticToolEditor(
            layers=config.mte.layers,
            feature_dim=feature_dim,
            edit_norm_cap=config.mte.edit_norm_cap,
        ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(mte.parameters(), lr=config.mte.lr)

    # Training loop
    logger.info(f"Starting training for {config.mte.steps} steps")

    mte.train()
    global_step = 0
    epoch = 0

    # Create save directory
    save_dir = Path(config.mte.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(total=config.mte.steps, desc="Training MTE")

    while global_step < config.mte.steps:
        epoch += 1

        for batch in dataloader:
            if global_step >= config.mte.steps:
                break

            # Move to device
            phi_no = {k: v.to(device) for k, v in batch["phi_no_tool"].items()}
            h_with = {k: v.to(device) for k, v in batch["h_with_tool"].items()}

            # Forward pass
            phi_edited = mte(phi_no)

            # Compute loss
            def decode_fn(layer_id, phi):
                return decode_features(
                    layer_id,
                    phi,
                    config.mte.hidden_dim,
                    config.mte.hidden_expansion,
                )

            loss, metrics = compute_mte_loss(
                phi_edited=phi_edited,
                h_with_tool=h_with,
                decode_fn=decode_fn,
                l2_penalty=config.mte.l2_penalty,
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            global_step += 1
            pbar.update(1)
            pbar.set_postfix(metrics)

            # Save checkpoint
            if global_step % 500 == 0 or global_step == config.mte.steps:
                checkpoint_file = save_dir / f"mte_step_{global_step}.pt"
                mte.save(str(checkpoint_file))
                logger.info(f"Saved checkpoint: {checkpoint_file}")

    pbar.close()

    # Save final model
    final_checkpoint = save_dir / "mte_final.pt"
    mte.save(str(final_checkpoint))
    logger.info(f"Training complete. Final model saved to {final_checkpoint}")

    return mte
