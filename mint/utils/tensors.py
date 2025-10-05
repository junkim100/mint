"""Tensor utility functions."""

from typing import Dict, List, Optional

import torch


def normalize_tensor(
    tensor: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalize a tensor along a dimension.

    Args:
        tensor: Input tensor
        dim: Dimension to normalize along
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor
    """
    norm = tensor.norm(dim=dim, keepdim=True)
    return tensor / (norm + eps)


def safe_divide(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Safely divide two tensors with numerical stability.

    Args:
        numerator: Numerator tensor
        denominator: Denominator tensor
        eps: Small constant to add to denominator

    Returns:
        Division result
    """
    return numerator / (denominator + eps)


def batch_tensors(
    tensors: List[torch.Tensor],
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Batch a list of tensors with padding.

    Args:
        tensors: List of tensors to batch
        pad_value: Value to use for padding

    Returns:
        Batched tensor
    """
    if not tensors:
        raise ValueError("Cannot batch empty list of tensors")

    # Get max shape for each dimension
    max_shape = [max(t.shape[i] for t in tensors) for i in range(tensors[0].ndim)]

    # Pad and stack
    padded = []
    for tensor in tensors:
        pad_sizes = []
        for i in range(tensor.ndim - 1, -1, -1):
            pad_sizes.extend([0, max_shape[i] - tensor.shape[i]])
        padded_tensor = torch.nn.functional.pad(tensor, pad_sizes, value=pad_value)
        padded.append(padded_tensor)

    return torch.stack(padded)


def dict_to_device(
    tensor_dict: Dict[int, torch.Tensor],
    device: torch.device,
) -> Dict[int, torch.Tensor]:
    """Move all tensors in a dictionary to a device.

    Args:
        tensor_dict: Dictionary of tensors
        device: Target device

    Returns:
        Dictionary with tensors on target device
    """
    return {k: v.to(device) for k, v in tensor_dict.items()}
