"""Dataset classes for MTE training."""

from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader

from mint.data.counterfactual_pairs import CounterfactualPair


class PairDataset(Dataset):
    """Dataset of counterfactual pairs for MTE training."""

    def __init__(
        self,
        pairs: List[CounterfactualPair],
        layers: List[int],
    ):
        """Initialize dataset.

        Args:
            pairs: List of counterfactual pairs
            layers: Layers to use for training
        """
        self.pairs = pairs
        self.layers = layers

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single pair.

        Returns:
            Dictionary with:
                - phi_no_tool: Dict[layer_id -> features]
                - phi_with_tool: Dict[layer_id -> features]
                - h_no_tool: Dict[layer_id -> hidden states]
                - h_with_tool: Dict[layer_id -> hidden states]
                - tool_label: str
        """
        pair = self.pairs[idx]

        # Filter to selected layers
        phi_no = {k: v for k, v in pair.phi_no_tool.items() if k in self.layers}
        phi_with = {k: v for k, v in pair.phi_with_tool.items() if k in self.layers}
        h_no = {k: v for k, v in pair.h_no_tool.items() if k in self.layers}
        h_with = {k: v for k, v in pair.h_with_tool.items() if k in self.layers}

        return {
            "phi_no_tool": phi_no,
            "phi_with_tool": phi_with,
            "h_no_tool": h_no,
            "h_with_tool": h_with,
            "tool_label": pair.tool_label,
        }


def collate_pairs(batch: List[Dict]) -> Dict:
    """Collate function for pair batches.

    Args:
        batch: List of pair dictionaries

    Returns:
        Batched dictionary
    """
    # Get all layers
    layers = list(batch[0]["phi_no_tool"].keys())

    # Collate by layer
    phi_no_batch = {layer: [] for layer in layers}
    phi_with_batch = {layer: [] for layer in layers}
    h_no_batch = {layer: [] for layer in layers}
    h_with_batch = {layer: [] for layer in layers}
    tool_labels = []

    for item in batch:
        for layer in layers:
            phi_no_batch[layer].append(item["phi_no_tool"][layer])
            phi_with_batch[layer].append(item["phi_with_tool"][layer])
            h_no_batch[layer].append(item["h_no_tool"][layer])
            h_with_batch[layer].append(item["h_with_tool"][layer])
        tool_labels.append(item["tool_label"])

    # Stack tensors (assuming same sequence length within batch)
    # In production, you may need padding
    for layer in layers:
        phi_no_batch[layer] = torch.cat(phi_no_batch[layer], dim=0)
        phi_with_batch[layer] = torch.cat(phi_with_batch[layer], dim=0)
        h_no_batch[layer] = torch.cat(h_no_batch[layer], dim=0)
        h_with_batch[layer] = torch.cat(h_with_batch[layer], dim=0)

    return {
        "phi_no_tool": phi_no_batch,
        "phi_with_tool": phi_with_batch,
        "h_no_tool": h_no_batch,
        "h_with_tool": h_with_batch,
        "tool_labels": tool_labels,
    }


def create_dataloader(
    pairs: List[CounterfactualPair],
    layers: List[int],
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader for pairs.

    Args:
        pairs: List of counterfactual pairs
        layers: Layers to use
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes

    Returns:
        DataLoader instance
    """
    dataset = PairDataset(pairs, layers)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_pairs,
    )
