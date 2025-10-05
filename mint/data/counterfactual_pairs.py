"""Counterfactual pair creation and management.

This module handles the creation and storage of counterfactual pairs:
no-tool vs with-tool SAE features at selected transformer layers.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import numpy as np
from tqdm import tqdm
import gc

from mint.models.sae_loader import clear_sae_cache


@dataclass
class CounterfactualPair:
    """A counterfactual pair of no-tool vs with-tool representations.

    Attributes:
        phi_no_tool: Dictionary mapping layer_id -> no-tool SAE features
        phi_with_tool: Dictionary mapping layer_id -> with-tool SAE features
        h_no_tool: Dictionary mapping layer_id -> no-tool hidden states
        h_with_tool: Dictionary mapping layer_id -> with-tool hidden states
        tool_label: Tool category (e.g., "search", "weather")
        metadata: Additional metadata (query, response, etc.)
    """

    phi_no_tool: Dict[int, torch.Tensor]
    phi_with_tool: Dict[int, torch.Tensor]
    h_no_tool: Dict[int, torch.Tensor]
    h_with_tool: Dict[int, torch.Tensor]
    tool_label: str
    metadata: Dict[str, Any]


def create_pair(
    model,
    tokenizer,
    query: str,
    tool_output: Optional[str],
    tool_label: str,
    layers: List[int],
    encode_fn,
    device: str = "cuda",
) -> CounterfactualPair:
    """Create a counterfactual pair by running model with/without tool.

    Args:
        model: Base language model
        tokenizer: Tokenizer
        query: User query
        tool_output: Tool output to inject (None for no-tool run)
        tool_label: Tool category label
        layers: Layers to extract features from
        encode_fn: Function to encode hidden states to SAE features
        device: Device to run on

    Returns:
        CounterfactualPair instance
    """
    # Prepare inputs
    if tool_output is None:
        # No-tool prompt
        prompt = query
    else:
        # With-tool prompt (inject tool output)
        prompt = f"{query}\n\nTool output: {tool_output}"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Run model and capture hidden states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # Tuple of (num_layers + 1) tensors

    # Extract features at selected layers
    # Process one layer at a time to avoid OOM
    phi_dict = {}
    h_dict = {}

    for layer_id in layers:
        # Get hidden states (layer_id + 1 because hidden_states[0] is embeddings)
        h = hidden_states[layer_id + 1]  # [batch, seq_len, hidden_dim]

        # Encode to SAE features
        phi = encode_fn(layer_id, h)

        # Move to CPU immediately and drop GPU tensors
        phi_cpu = phi.detach().to("cpu")
        h_cpu = h.detach().to("cpu")
        phi_dict[layer_id] = phi_cpu
        h_dict[layer_id] = h_cpu

        # Explicitly free temporaries
        del phi, h
        clear_sae_cache()
        torch.cuda.empty_cache()

    # Free hidden states and outputs to release GPU memory
    try:
        del hidden_states
    except Exception:
        pass
    torch.cuda.empty_cache()

    return phi_dict, h_dict


def save_pairs(
    pairs: List[CounterfactualPair],
    output_dir: str,
) -> None:
    """Save counterfactual pairs to disk.

    Args:
        pairs: List of counterfactual pairs
        output_dir: Output directory
    """
    import json

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect metadata for JSON summary
    json_summary = []

    # Save as PyTorch tensors
    for i, pair in enumerate(tqdm(pairs, desc="Saving pairs")):
        pair_path = output_path / f"pair_{i:06d}.pt"
        torch.save(
            {
                "phi_no_tool": pair.phi_no_tool,
                "phi_with_tool": pair.phi_with_tool,
                "h_no_tool": pair.h_no_tool,
                "h_with_tool": pair.h_with_tool,
                "tool_label": pair.tool_label,
                "metadata": pair.metadata,
            },
            pair_path,
        )

        # Add to JSON summary
        pair_summary = {
            "pair_id": i,
            "file": f"pair_{i:06d}.pt",
            "tool_label": pair.tool_label,
            "metadata": pair.metadata,
            "shapes": {
                "phi_no_tool": {layer: list(tensor.shape) for layer, tensor in pair.phi_no_tool.items()},
                "phi_with_tool": {layer: list(tensor.shape) for layer, tensor in pair.phi_with_tool.items()},
                "h_no_tool": {layer: list(tensor.shape) for layer, tensor in pair.h_no_tool.items()},
                "h_with_tool": {layer: list(tensor.shape) for layer, tensor in pair.h_with_tool.items()},
            }
        }
        json_summary.append(pair_summary)

    # Save JSON summary
    json_path = output_path / "pairs_summary.json"
    with open(json_path, "w") as f:
        json.dump({
            "total_pairs": len(pairs),
            "pairs": json_summary
        }, f, indent=2)

    print(f"Saved {len(pairs)} pairs to {output_dir}")
    print(f"JSON summary saved to {json_path}")


def load_pairs(
    input_dir: str,
    max_pairs: Optional[int] = None,
) -> List[CounterfactualPair]:
    """Load counterfactual pairs from disk.

    Args:
        input_dir: Input directory
        max_pairs: Maximum number of pairs to load

    Returns:
        List of counterfactual pairs
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Pairs directory not found: {input_dir}")

    pair_files = sorted(input_path.glob("pair_*.pt"))

    if max_pairs is not None:
        pair_files = pair_files[:max_pairs]

    pairs = []
    for pair_file in tqdm(pair_files, desc="Loading pairs"):
        data = torch.load(pair_file)
        pair = CounterfactualPair(
            phi_no_tool=data["phi_no_tool"],
            phi_with_tool=data["phi_with_tool"],
            h_no_tool=data["h_no_tool"],
            h_with_tool=data["h_with_tool"],
            tool_label=data["tool_label"],
            metadata=data["metadata"],
        )
        pairs.append(pair)

    print(f"Loaded {len(pairs)} pairs from {input_dir}")
    return pairs


def filter_pairs_by_tool(
    pairs: List[CounterfactualPair],
    tool_labels: List[str],
) -> List[CounterfactualPair]:
    """Filter pairs by tool labels.

    Args:
        pairs: List of pairs
        tool_labels: Tool labels to keep

    Returns:
        Filtered list of pairs
    """
    return [p for p in pairs if p.tool_label in tool_labels]


def get_pair_statistics(pairs: List[CounterfactualPair]) -> Dict[str, Any]:
    """Compute statistics about pairs.

    Args:
        pairs: List of pairs

    Returns:
        Dictionary of statistics
    """
    tool_counts = {}
    for pair in pairs:
        tool_counts[pair.tool_label] = tool_counts.get(pair.tool_label, 0) + 1

    return {
        "total_pairs": len(pairs),
        "tool_counts": tool_counts,
        "tools": list(tool_counts.keys()),
    }
