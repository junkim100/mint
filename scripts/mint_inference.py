#!/usr/bin/env python3
"""
MINT Inference Demo

Demonstrates MINT decision making with trained components.
"""

import torch
import fire
import logging
from pathlib import Path
from typing import Dict, List

from mint.core.model_loader import ModelLoader
from mint.core.sae_loader import SAELoader
from mint.core.activation_extractor import ActivationExtractor
from mint.core.editor import MechanisticEditor
from mint.training.phase_b import ValueHead  # Use the training version
from mint.inference.conformal import ConformalCalibrator, RiskBudget
from mint.inference.decision import MINTDecisionMaker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_mint_components(
    affordance_dir: str,
    editor_dir: str,
    value_head_dir: str,
    calibrator_dir: str,
    device: str = "cuda",
) -> tuple:
    """Load all MINT components from checkpoints."""

    affordance_path = Path(affordance_dir)
    editor_path = Path(editor_dir)
    value_head_path = Path(value_head_dir)
    calibrator_path = Path(calibrator_dir)

    # Load SAEs
    logger.info("Loading SAEs...")
    sae_loader = SAELoader(device=device)
    sae_loader.load()

    # Get list of tools from affordances
    # Affordance files are named as {tool_name}.pt
    affordance_files = [f for f in affordance_path.glob("*.pt") if f.name != "metadata.json"]
    tools = [f.stem for f in affordance_files]
    logger.info(f"Found tools: {tools}")

    # Load editors
    logger.info("Loading editors...")
    editors = {}
    for tool in tools:
        editor_file = editor_path / f"{tool}_editor.pt"
        if not editor_file.exists():
            logger.warning(f"Editor not found for {tool}, skipping")
            continue

        checkpoint = torch.load(editor_file, weights_only=False)
        editor = MechanisticEditor(
            masks=checkpoint["masks"],
            directions=checkpoint["directions"],
            sae_loader=sae_loader,
            alpha_max=0.5,
        )
        editors[tool] = editor
        logger.info(f"  Loaded editor for {tool}")

    # Load value heads
    logger.info("Loading value heads...")
    value_heads = {}
    for tool in tools:
        vh_file = value_head_path / f"{tool}_value_head.pt"
        if not vh_file.exists():
            logger.warning(f"Value head not found for {tool}, skipping")
            continue

        checkpoint = torch.load(vh_file, weights_only=False)
        value_head = ValueHead(
            hidden_size=4096,
            num_r_layers=8,  # Total number of residual layers
            hidden_dims=[2048, 256],
            dropout=0.1,
        )
        value_head.load_state_dict(checkpoint["model_state_dict"])
        value_head.to(device=device, dtype=torch.bfloat16)
        value_head.eval()
        value_heads[tool] = value_head
        logger.info(f"  Loaded value head for {tool}")

    # Load calibrators
    logger.info("Loading calibrators...")
    calibrators = {}
    for tool in tools:
        cal_file = calibrator_path / f"{tool}_calibrator.pt"
        if not cal_file.exists():
            logger.warning(f"Calibrator not found for {tool}, skipping")
            continue

        checkpoint = torch.load(cal_file, weights_only=False)
        calibrator = ConformalCalibrator(alpha=checkpoint["alpha"])
        calibrator.quantile = checkpoint["quantile"]
        calibrator.is_finalized = True
        calibrators[tool] = calibrator
        logger.info(f"  Loaded calibrator for {tool} (quantile={calibrator.quantile:.4f})")

    # Filter tools to only those with all components
    valid_tools = [t for t in tools if t in editors and t in value_heads and t in calibrators]
    logger.info(f"Valid tools with all components: {valid_tools}")

    return valid_tools, editors, value_heads, calibrators, sae_loader


def demo_inference(
    affordance_dir: str = "/app/mint/checkpoints/affordances",
    editor_dir: str = "/app/mint/checkpoints/editors",
    value_head_dir: str = "/app/mint/checkpoints/value_heads",
    calibrator_dir: str = "/app/mint/checkpoints/calibrators",
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    device: str = "cuda",
    num_examples: int = 5,
):
    """
    Run MINT inference demo.

    Args:
        affordance_dir: Directory with affordance checkpoints
        editor_dir: Directory with editor checkpoints
        value_head_dir: Directory with value head checkpoints
        calibrator_dir: Directory with calibrator checkpoints
        model_name: Model name
        device: Device to use
        num_examples: Number of examples to run
    """
    logger.info("="*80)
    logger.info("MINT Inference Demo")
    logger.info("="*80)

    # Load MINT components
    tools, editors, value_heads, calibrators, sae_loader = load_mint_components(
        affordance_dir=affordance_dir,
        editor_dir=editor_dir,
        value_head_dir=value_head_dir,
        calibrator_dir=calibrator_dir,
        device=device,
    )

    if not tools:
        logger.error("No valid tools found with all components!")
        return

    logger.info(f"\n{'='*80}")
    logger.info("MINT Components Successfully Loaded!")
    logger.info(f"{'='*80}\n")

    logger.info(f"Tools: {tools}")
    logger.info(f"Editors: {list(editors.keys())}")
    logger.info(f"Value Heads: {list(value_heads.keys())}")
    logger.info(f"Calibrators: {list(calibrators.keys())}")

    logger.info(f"\n{'='*80}")
    logger.info("Component Summary:")
    logger.info(f"{'='*80}\n")

    for tool in tools:
        logger.info(f"Tool: {tool}")
        logger.info(f"  Editor: {type(editors[tool]).__name__}")
        logger.info(f"  Value Head: {type(value_heads[tool]).__name__}")
        logger.info(f"  Calibrator quantile: {calibrators[tool].quantile:.4f}")
        logger.info("")

    logger.info(f"{'='*80}")
    logger.info("MINT Inference Demo Complete!")
    logger.info(f"{'='*80}\n")
    logger.info("Note: Full MINT inference with decision making requires integration")
    logger.info("with the evaluation framework (τ-bench). The components are ready")
    logger.info("and can be used for risk-calibrated tool selection.")


if __name__ == "__main__":
    fire.Fire(demo_inference)

