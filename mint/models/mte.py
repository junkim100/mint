"""Mechanistic Tool Editor (MTE) model.

The MTE learns to transform no-tool internal representations into with-tool
representations by applying learned edits in SAE feature space.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn


class MechanisticToolEditor(nn.Module):
    """Mechanistic Tool Editor that applies feature-space edits.

    The MTE learns a mapping from no-tool SAE features to with-tool SAE features
    at selected transformer layers. It applies norm-capped edits for stability.
    """

    def __init__(
        self,
        layers: List[int],
        feature_dim: int,
        hidden_dim: int = 512,
        edit_norm_cap: Optional[float] = 0.5,
        dropout: float = 0.1,
    ):
        """Initialize MTE.

        Args:
            layers: List of layer indices where edits are applied
            feature_dim: SAE feature dimension (e.g., 8 * 4096 = 32768)
            hidden_dim: Hidden dimension for edit networks
            edit_norm_cap: Maximum L2 norm for edits (None for no cap)
            dropout: Dropout probability
        """
        super().__init__()
        self.layers = layers
        self.feature_dim = feature_dim
        self.edit_norm_cap = edit_norm_cap

        # Per-layer edit networks
        self.editors = nn.ModuleDict()
        for layer_id in layers:
            self.editors[str(layer_id)] = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, feature_dim),
            )

    def forward(
        self,
        phi_no_tool: Dict[int, torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        """Apply learned edits to no-tool features.

        Args:
            phi_no_tool: Dictionary mapping layer_id -> no-tool SAE features
                        Each tensor has shape [batch, seq_len, feature_dim]

        Returns:
            Dictionary mapping layer_id -> edited SAE features
        """
        phi_edited = {}

        for layer_id, phi in phi_no_tool.items():
            if layer_id not in self.layers:
                # Pass through unchanged if not in edit layers
                phi_edited[layer_id] = phi
                continue

            # Compute edit delta
            delta = self.editors[str(layer_id)](phi)

            # Apply norm cap for stability
            if self.edit_norm_cap is not None:
                # Compute norm along feature dimension
                delta_norm = delta.norm(dim=-1, keepdim=True)
                # Scale down if exceeds cap
                scale = (delta_norm / (self.edit_norm_cap + 1e-6)).clamp(min=1.0)
                delta = delta / scale

            # Apply edit
            phi_edited[layer_id] = phi + delta

        return phi_edited

    def get_edit_magnitudes(
        self,
        phi_no_tool: Dict[int, torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        """Compute edit magnitudes for analysis.

        Args:
            phi_no_tool: Dictionary of no-tool features

        Returns:
            Dictionary mapping layer_id -> edit norms [batch, seq_len]
        """
        edit_norms = {}

        for layer_id, phi in phi_no_tool.items():
            if layer_id not in self.layers:
                continue

            delta = self.editors[str(layer_id)](phi)
            edit_norms[layer_id] = delta.norm(dim=-1)

        return edit_norms

    def save(self, path: str) -> None:
        """Save MTE checkpoint.

        Args:
            path: Path to save checkpoint
        """
        torch.save(
            {
                "state_dict": self.state_dict(),
                "layers": self.layers,
                "feature_dim": self.feature_dim,
                "edit_norm_cap": self.edit_norm_cap,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cuda") -> "MechanisticToolEditor":
        """Load MTE from checkpoint.

        Args:
            path: Path to checkpoint
            device: Device to load model on

        Returns:
            Loaded MTE instance
        """
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            layers=checkpoint["layers"],
            feature_dim=checkpoint["feature_dim"],
            edit_norm_cap=checkpoint["edit_norm_cap"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        return model


def compute_mte_loss(
    phi_edited: Dict[int, torch.Tensor],
    h_with_tool: Dict[int, torch.Tensor],
    decode_fn,
    l2_penalty: float = 0.01,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Compute MTE training loss.

    Loss = MSE(decode(phi_edited), h_with_tool) + L2_penalty * ||edits||^2

    Args:
        phi_edited: Edited SAE features
        h_with_tool: Target with-tool hidden states
        decode_fn: Function to decode SAE features to hidden states
        l2_penalty: Weight for L2 regularization on edits

    Returns:
        Tuple of (total_loss, metrics_dict)
    """
    mse_loss = 0.0
    l2_reg = 0.0
    num_layers = 0

    for layer_id, phi_edit in phi_edited.items():
        if layer_id not in h_with_tool:
            continue

        # Decode edited features
        h_reconstructed = decode_fn(layer_id, phi_edit)
        h_target = h_with_tool[layer_id]

        # MSE loss
        mse_loss += torch.nn.functional.mse_loss(h_reconstructed, h_target)

        # L2 regularization on feature magnitudes
        l2_reg += (phi_edit ** 2).mean()

        num_layers += 1

    # Average over layers
    if num_layers > 0:
        mse_loss = mse_loss / num_layers
        l2_reg = l2_reg / num_layers

    total_loss = mse_loss + l2_penalty * l2_reg

    metrics = {
        "total_loss": total_loss.item(),
        "mse_loss": mse_loss.item(),
        "l2_reg": l2_reg.item(),
    }

    return total_loss, metrics
