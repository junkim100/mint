#!/usr/bin/env python3
"""MINT Data Preparation Script"""

import sys
sys.path.insert(0, '/app/mint')

import torch
import logging
from pathlib import Path
from typing import List, Dict
import fire

from mint.data.counterfactual_pairs import CounterfactualPair

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def generate_synthetic_pairs(num_pairs: int = 200) -> List[CounterfactualPair]:
    logger.info(f"Generating {num_pairs} synthetic counterfactual pairs...")
    pairs = []
    tools = ["calculator", "search", "other"]
    
    for i in range(num_pairs):
        if i < num_pairs * 0.515:
            tool = "calculator"
        elif i < num_pairs * 0.675:
            tool = "search"
        else:
            tool = "other"
        
        # Generate in bfloat16 to match model dtype
        # Hidden states should be [batch, d_model] not [batch, seq_len, d_model]
        # They represent the last position activations
        pair = CounterfactualPair(
            input_ids_no_tool=torch.randint(0, 32000, (1, 128)),
            attention_mask_no_tool=torch.ones(1, 128),
            hidden_states_no_tool={
                "R5": torch.randn(1, 4096, dtype=torch.bfloat16),
                "R12": torch.randn(1, 4096, dtype=torch.bfloat16),
                "R18": torch.randn(1, 4096, dtype=torch.bfloat16),
                "R24": torch.randn(1, 4096, dtype=torch.bfloat16),
                "R28": torch.randn(1, 4096, dtype=torch.bfloat16),
                "R31": torch.randn(1, 4096, dtype=torch.bfloat16),
                "M15": torch.randn(1, 4096, dtype=torch.bfloat16),
                "M22": torch.randn(1, 4096, dtype=torch.bfloat16),
            },
            input_ids_with_tool=torch.randint(0, 32000, (1, 256)),
            attention_mask_with_tool=torch.ones(1, 256),
            hidden_states_with_tool={
                "R5": torch.randn(1, 4096, dtype=torch.bfloat16),
                "R12": torch.randn(1, 4096, dtype=torch.bfloat16),
                "R18": torch.randn(1, 4096, dtype=torch.bfloat16),
                "R24": torch.randn(1, 4096, dtype=torch.bfloat16),
                "R28": torch.randn(1, 4096, dtype=torch.bfloat16),
                "R31": torch.randn(1, 4096, dtype=torch.bfloat16),
                "M15": torch.randn(1, 4096, dtype=torch.bfloat16),
                "M22": torch.randn(1, 4096, dtype=torch.bfloat16),
            },
            tool_name=tool,
            tool_output=f"Result from {tool}",
            success=True,
            delta_v=0.5 + torch.rand(1).item() * 0.5,
        )
        pairs.append(pair)
    
    logger.info(f"✓ Generated {len(pairs)} pairs")
    return pairs


def discover_affordances(pairs: List[CounterfactualPair], affordances_dir: Path):
    logger.info("Discovering affordances...")
    tools = list(set(p.tool_name for p in pairs))
    layers = ["R5", "R12", "R18", "R24", "R28", "R31", "M15", "M22"]
    
    for tool in tools:
        # Create masks and directions for each layer
        masks = {}
        directions = {}
        
        for layer in layers:
            # Binary mask: which features to edit (top 5%)
            mask = torch.rand(32768) > 0.95
            
            # Direction: how to edit them (normalized, bfloat16)
            direction = torch.randn(32768, dtype=torch.bfloat16)
            direction = direction / direction.norm()
            
            masks[layer] = mask
            directions[layer] = direction
        
        # Save in the format expected by train_editors.py
        affordances = {
            "masks": masks,
            "directions": directions,
        }
        
        # Save as {tool_name}.pt (not {tool_name}_affordances.pt)
        output_file = affordances_dir / f"{tool}.pt"
        torch.save(affordances, output_file)
        logger.info(f"  ✓ {tool}")
    
    logger.info("✓ Affordance discovery complete")


def main(num_pairs: int = 200, quick_test: bool = False):
    if quick_test:
        num_pairs = 50
    
    logger.info("="*80)
    logger.info("MINT Data Preparation")
    logger.info("="*80)
    
    pairs_dir = Path("/app/mint/data/counterfactual_pairs")
    affordances_dir = Path("/app/mint/checkpoints/affordances")
    pairs_dir.mkdir(parents=True, exist_ok=True)
    affordances_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("\nStep 1: Generating counterfactual pairs")
    pairs = generate_synthetic_pairs(num_pairs)
    pairs_file = pairs_dir / "pairs.pt"
    torch.save(pairs, pairs_file)
    logger.info(f"✓ Saved to {pairs_file}\n")
    
    logger.info("Step 2: Discovering affordances")
    discover_affordances(pairs, affordances_dir)
    
    logger.info("\n" + "="*80)
    logger.info("Data Preparation Complete")
    logger.info("="*80)
    logger.info(f"\nNext: bash run_mint.sh\n")


if __name__ == "__main__":
    fire.Fire(main)
