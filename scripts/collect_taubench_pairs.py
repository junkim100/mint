#!/usr/bin/env python3
"""
Collect real counterfactual pairs from τ-bench.

This script replaces the synthetic data generation with real data collection
from τ-bench tool executions.

Implements Proposal Section 4, Phase A:
"Use real tool outputs from τ-bench to construct counterfactual pairs"

Usage:
    python scripts/collect_taubench_pairs.py --num_pairs 500 --output data/counterfactual_pairs/real_pairs.pt

Reference: MINT Proposal Section 4, Phase A
"""

import fire
import torch
import logging
from pathlib import Path
from typing import List, Dict, Optional
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mint.data.counterfactual_pairs import CounterfactualPair, CounterfactualPairBuilder
from mint.core.activation_extractor import ActivationExtractor
from mint.core.sae_loader import SAELoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_taubench_data(
    dataset_path: str = "data/taubench/trajectories.json",
    num_samples: Optional[int] = None,
) -> List[Dict]:
    """
    Load τ-bench trajectories.
    
    Args:
        dataset_path: Path to τ-bench dataset
        num_samples: Optional number of samples to load
        
    Returns:
        List of trajectory dictionaries
    """
    import json
    
    logger.info(f"Loading τ-bench data from {dataset_path}")
    
    if not Path(dataset_path).exists():
        logger.error(f"τ-bench dataset not found at {dataset_path}")
        logger.info(
            "Please download τ-bench data first:\n"
            "  1. Clone τ-bench: git clone https://github.com/sierra-research/tau-bench\n"
            "  2. Extract trajectories to data/taubench/trajectories.json\n"
            "  3. Run this script again"
        )
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    if num_samples:
        data = data[:num_samples]
    
    logger.info(f"Loaded {len(data)} trajectories")
    return data


def extract_tool_calls(trajectory: Dict) -> List[Dict]:
    """
    Extract tool calls from a τ-bench trajectory.
    
    Args:
        trajectory: Trajectory dictionary
        
    Returns:
        List of tool call dictionaries
    """
    tool_calls = []
    
    # τ-bench format: trajectory contains steps with tool calls
    for step in trajectory.get('steps', []):
        if 'tool_call' in step:
            tool_call = {
                'context': step.get('context', ''),
                'tool_name': step['tool_call'].get('name', 'unknown'),
                'tool_input': step['tool_call'].get('input', ''),
                'tool_output': step['tool_call'].get('output', ''),
                'success': step.get('success', False),
                'reward': step.get('reward', 0.0),
            }
            tool_calls.append(tool_call)
    
    return tool_calls


def main(
    num_pairs: int = 500,
    output: str = "data/counterfactual_pairs/real_pairs.pt",
    taubench_path: str = "data/taubench/trajectories.json",
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    sae_release: str = "llama-scope-8b",
    device: str = "cuda",
):
    """
    Collect real counterfactual pairs from τ-bench.
    
    Args:
        num_pairs: Number of pairs to collect
        output: Output path for pairs
        taubench_path: Path to τ-bench dataset
        model_name: Base model name
        sae_release: SAE release name
        device: Device to use
    """
    logger.info("=" * 80)
    logger.info("MINT Real Data Collection")
    logger.info("=" * 80)
    
    # Create output directory
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load τ-bench data
    try:
        trajectories = load_taubench_data(
            dataset_path=taubench_path,
            num_samples=num_pairs * 2,  # Load extra in case some are invalid
        )
    except FileNotFoundError:
        logger.error("Cannot proceed without τ-bench data")
        logger.info(
            "\n" + "=" * 80 + "\n"
            "FALLBACK: Using synthetic data generation\n"
            "To use real data, please:\n"
            "  1. Download τ-bench dataset\n"
            "  2. Place at data/taubench/trajectories.json\n"
            "  3. Run this script again\n"
            + "=" * 80
        )
        return
    
    # Initialize components
    logger.info(f"Loading model: {model_name}")
    activation_extractor = ActivationExtractor(
        model_name=model_name,
        device=device,
    )
    
    logger.info(f"Loading SAEs: {sae_release}")
    sae_loader = SAELoader(
        release=sae_release,
        device=device,
    )
    
    # Initialize pair builder
    pair_builder = CounterfactualPairBuilder(
        activation_extractor=activation_extractor,
        sae_loader=sae_loader,
        device=device,
    )
    
    # Collect pairs
    logger.info(f"Collecting {num_pairs} counterfactual pairs...")
    pairs = []
    
    for i, trajectory in enumerate(trajectories):
        if len(pairs) >= num_pairs:
            break
        
        # Extract tool calls
        tool_calls = extract_tool_calls(trajectory)
        
        if not tool_calls:
            continue
        
        # Build pairs from tool calls
        for tool_call in tool_calls:
            if len(pairs) >= num_pairs:
                break
            
            try:
                # Build counterfactual pair
                pair = pair_builder.build_pair(
                    context=tool_call['context'],
                    tool_name=tool_call['tool_name'],
                    tool_output=tool_call['tool_output'],
                    success=tool_call['success'],
                    delta_v=tool_call['reward'],
                )
                
                pairs.append(pair)
                
                if len(pairs) % 50 == 0:
                    logger.info(f"Collected {len(pairs)}/{num_pairs} pairs")
            
            except Exception as e:
                logger.warning(f"Failed to build pair: {e}")
                continue
    
    logger.info(f"Collected {len(pairs)} pairs")
    
    # Save pairs
    logger.info(f"Saving pairs to {output}")
    torch.save(pairs, output)
    
    # Print statistics
    logger.info("\n" + "=" * 80)
    logger.info("Collection Statistics")
    logger.info("=" * 80)
    logger.info(f"Total pairs: {len(pairs)}")
    
    # Tool distribution
    tool_counts = {}
    for pair in pairs:
        tool_counts[pair.tool_name] = tool_counts.get(pair.tool_name, 0) + 1
    
    logger.info("\nTool distribution:")
    for tool, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {tool}: {count} ({count/len(pairs)*100:.1f}%)")
    
    # Success rate
    success_rate = sum(1 for p in pairs if p.success) / len(pairs)
    logger.info(f"\nSuccess rate: {success_rate:.1%}")
    
    # Average ΔV
    avg_delta_v = sum(p.delta_v for p in pairs) / len(pairs)
    logger.info(f"Average ΔV: {avg_delta_v:.4f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ Real data collection complete!")
    logger.info("=" * 80)
    logger.info(f"\nSaved to: {output}")
    logger.info(f"Use this file for training instead of synthetic data")


if __name__ == "__main__":
    fire.Fire(main)

