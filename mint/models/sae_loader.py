"""Sparse Autoencoder (SAE) loader and interface.

This module provides a minimal interface for loading and using SAEs
to encode/decode hidden states at selected transformer layers.

SAEs are loaded from safetensors checkpoints with the following expected structure:
- encoder.weight: [feature_dim, hidden_dim]
- encoder.bias: [feature_dim]
- decoder.weight: [hidden_dim, feature_dim]
- decoder.bias: [hidden_dim]
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from safetensors.torch import load_file as load_safetensors


@dataclass
class SAEHandle:
    """Handle for a Sparse Autoencoder at a specific layer.

    Attributes:
        layer_id: Transformer layer index
        encoder: SAE module with encode_only method
        decoder: SAE module with decode_only method (same as encoder)
        hidden_dim: Input hidden dimension
        feature_dim: SAE feature dimension (typically 8x hidden_dim)
    """

    layer_id: int
    encoder: nn.Module  # RealSAE instance
    decoder: nn.Module  # RealSAE instance (same as encoder)
    hidden_dim: int
    feature_dim: int

    def encode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Encode hidden states to SAE features.

        Args:
            hidden_states: Hidden states [batch, seq_len, hidden_dim]

        Returns:
            SAE features [batch, seq_len, feature_dim] with ReLU activation
        """
        return self.encoder.encode_only(hidden_states)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode SAE features back to hidden states.

        Args:
            features: SAE features [batch, seq_len, feature_dim]

        Returns:
            Reconstructed hidden states [batch, seq_len, hidden_dim]
        """
        return self.decoder.decode_only(features)


class RealSAE(nn.Module):
    """Real SAE loaded from safetensors checkpoint.

    Expected checkpoint structure:
    - encoder.weight: [feature_dim, hidden_dim]
    - encoder.bias: [feature_dim]
    - decoder.weight: [hidden_dim, feature_dim]
    - decoder.bias: [hidden_dim]
    """

    def __init__(self, hidden_dim: int, feature_dim: int, checkpoint_path: Optional[str] = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim

        # Initialize encoder and decoder
        self.encoder = nn.Linear(hidden_dim, feature_dim, bias=True)
        self.decoder = nn.Linear(feature_dim, hidden_dim, bias=True)

        # Load from checkpoint if provided
        if checkpoint_path and Path(checkpoint_path).exists():
            self._load_from_checkpoint(checkpoint_path)

    def _load_from_checkpoint(self, checkpoint_path: str):
        """Load SAE weights from safetensors checkpoint."""
        try:
            state_dict = load_safetensors(checkpoint_path)

            # Downcast to bfloat16 to reduce GPU memory footprint
            target_dtype = torch.bfloat16

            # Map checkpoint keys to module keys
            # Expected keys: encoder.weight, encoder.bias, decoder.weight, decoder.bias
            if "encoder.weight" in state_dict:
                self.encoder.weight.data = state_dict["encoder.weight"].to(target_dtype)
            if "encoder.bias" in state_dict:
                self.encoder.bias.data = state_dict["encoder.bias"].to(target_dtype)
            if "decoder.weight" in state_dict:
                self.decoder.weight.data = state_dict["decoder.weight"].to(target_dtype)
            if "decoder.bias" in state_dict:
                self.decoder.bias.data = state_dict["decoder.bias"].to(target_dtype)

        except Exception as e:
            raise RuntimeError(f"Failed to load SAE from {checkpoint_path}: {e}")

    def encode_only(self, x: torch.Tensor) -> torch.Tensor:
        """Encode hidden states to SAE features (with ReLU activation)."""
        # Convert to SAE dtype if needed
        x_converted = x.to(self.encoder.weight.dtype)
        return torch.relu(self.encoder(x_converted))

    def decode_only(self, features: torch.Tensor) -> torch.Tensor:
        """Decode SAE features back to hidden states."""
        # Convert to SAE dtype if needed
        features_converted = features.to(self.decoder.weight.dtype)
        return self.decoder(features_converted)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (encode then decode)."""
        features = self.encode_only(x)
        reconstruction = self.decode_only(features)
        return reconstruction


# Global cache for loaded SAEs (keyed by (layer_id, checkpoint_path))
_SAE_CACHE: Dict[tuple, SAEHandle] = {}


def load_sae(
    layer_id: int,
    hidden_dim: int = 4096,
    expansion_factor: int = 8,
    checkpoint_path: Optional[str] = None,
    device: str = "cuda",
) -> SAEHandle:
    """Load or create an SAE for a specific layer.

    Args:
        layer_id: Transformer layer index
        hidden_dim: Hidden dimension (4096 for Llama-3.1-8B)
        expansion_factor: SAE expansion factor (typically 8)
        checkpoint_path: Path to SAE safetensors checkpoint (required for real SAEs)
        device: Device to load SAE on

    Returns:
        SAEHandle instance

    Raises:
        FileNotFoundError: If checkpoint_path is provided but doesn't exist
        RuntimeError: If SAE loading fails
    """
    # Check cache
    cache_key = (layer_id, checkpoint_path)
    if cache_key in _SAE_CACHE:
        return _SAE_CACHE[cache_key]

    feature_dim = hidden_dim * expansion_factor

    # Load real SAE from checkpoint
    if checkpoint_path:
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(
                f"SAE checkpoint not found: {checkpoint_path}\n"
                f"Please ensure SAE checkpoints are available for layer {layer_id}.\n"
                f"Expected format: safetensors with encoder/decoder weights."
            )
        sae = RealSAE(hidden_dim, feature_dim, checkpoint_path).to(device)
    else:
        # Create placeholder SAE (for testing only)
        import warnings
        warnings.warn(
            f"No checkpoint provided for layer {layer_id}. Using placeholder SAE. "
            "This should only be used for testing. For real experiments, provide SAE checkpoints.",
            UserWarning
        )
        sae = RealSAE(hidden_dim, feature_dim, checkpoint_path=None).to(device)

    # Create handle with encode/decode methods
    handle = SAEHandle(
        layer_id=layer_id,
        encoder=sae,  # Pass full SAE for encode_only method
        decoder=sae,  # Pass full SAE for decode_only method
        hidden_dim=hidden_dim,
        feature_dim=feature_dim,
    )

    # Cache
    _SAE_CACHE[cache_key] = handle

    return handle


def encode_features(
    layer_id: int,
    hidden_states: torch.Tensor,
    hidden_dim: int = 4096,
    expansion_factor: int = 8,
    checkpoint_path: Optional[str] = None,
) -> torch.Tensor:
    """Encode hidden states to SAE features.

    Args:
        layer_id: Layer index
        hidden_states: Hidden states to encode
        hidden_dim: Hidden dimension
        expansion_factor: SAE expansion factor
        checkpoint_path: Optional path to SAE checkpoint

    Returns:
        SAE features with ReLU activation
    """
    device = hidden_states.device if isinstance(hidden_states, torch.Tensor) else "cuda"
    sae = load_sae(layer_id, hidden_dim, expansion_factor, checkpoint_path, device=device)
    return sae.encode(hidden_states)


def decode_features(
    layer_id: int,
    features: torch.Tensor,
    hidden_dim: int = 4096,
    expansion_factor: int = 8,
    checkpoint_path: Optional[str] = None,
) -> torch.Tensor:
    """Decode SAE features back to hidden states.

    Args:
        layer_id: Layer index
        features: SAE features to decode
        hidden_dim: Hidden dimension
        expansion_factor: SAE expansion factor
        checkpoint_path: Optional path to SAE checkpoint

    Returns:
        Reconstructed hidden states
    """
    device = features.device if isinstance(features, torch.Tensor) else "cuda"
    sae = load_sae(layer_id, hidden_dim, expansion_factor, checkpoint_path, device=device)
    return sae.decode(features)


def clear_sae_cache() -> None:
    """Clear the SAE cache and free GPU memory."""
    global _SAE_CACHE
    try:
        # Move any loaded modules to CPU to release GPU memory before clearing
        for handle in _SAE_CACHE.values():
            try:
                handle.encoder.to("cpu")
                handle.decoder.to("cpu")
            except Exception:
                pass
    finally:
        _SAE_CACHE.clear()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
