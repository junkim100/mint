"""SAE loading utilities for MINT using Llama-Scope SAEs.

Llama-Scope naming convention:
- l[layer]r_8x for residual stream SAEs (e.g., l5r_8x)
- l[layer]m_8x for MLP SAEs (e.g., l15m_8x)
"""

import torch
from sae_lens import SAE
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class SAELoader:
    """Loads and manages Llama-Scope SAEs for specified layers."""

    def __init__(
        self,
        residual_layers: List[int] = [5, 12, 18, 24, 28, 31],
        mlp_layers: List[int] = [15, 22],
        expansion: str = "8x",  # "8x" for 32K features, "32x" for 128K features
        device: str = "cuda",
    ):
        """
        Initialize SAE loader.

        Args:
            residual_layers: List of layer indices for residual stream SAEs
            mlp_layers: List of layer indices for MLP SAEs
            expansion: Expansion factor ("8x" or "32x")
            device: Device to load SAEs on
        """
        self.residual_layers = residual_layers
        self.mlp_layers = mlp_layers
        self.expansion = expansion
        self.device = device

        # Llama-Scope release names (correct format from SAELens)
        self.residual_release = f"llama_scope_lxr_{expansion}"
        self.mlp_release = f"llama_scope_lxm_{expansion}"

        self.sae_r: Dict[int, SAE] = {}
        self.sae_m: Dict[int, SAE] = {}

    def load(self) -> tuple[Dict[int, SAE], Dict[int, SAE]]:
        """
        Load all SAEs.

        Returns:
            Tuple of (residual_saes, mlp_saes) dictionaries
        """
        logger.info(f"Loading Llama-Scope SAEs with {self.expansion} expansion")
        logger.info(f"Residual layers: {self.residual_layers}")
        logger.info(f"MLP layers: {self.mlp_layers}")

        # Load residual SAEs
        logger.info(f"Loading residual SAEs from release: {self.residual_release}")
        for layer in self.residual_layers:
            # Llama-Scope naming: l[layer]r_[expansion] (lowercase, underscore)
            sae_id = f"l{layer}r_{self.expansion}"
            logger.info(f"  Loading R-SAE for layer {layer}: {sae_id}")
            try:
                self.sae_r[layer] = SAE.from_pretrained(
                    release=self.residual_release,
                    sae_id=sae_id,
                    device=self.device
                )
                logger.info(f"    ✓ Loaded successfully")
            except Exception as e:
                logger.error(f"    ✗ Failed to load: {e}")
                raise

        # Load MLP SAEs
        logger.info(f"Loading MLP SAEs from release: {self.mlp_release}")
        for layer in self.mlp_layers:
            # Llama-Scope naming: l[layer]m_[expansion] (lowercase, underscore)
            sae_id = f"l{layer}m_{self.expansion}"
            logger.info(f"  Loading M-SAE for layer {layer}: {sae_id}")
            try:
                self.sae_m[layer] = SAE.from_pretrained(
                    release=self.mlp_release,
                    sae_id=sae_id,
                    device=self.device
                )
                logger.info(f"    ✓ Loaded successfully")
            except Exception as e:
                logger.error(f"    ✗ Failed to load: {e}")
                raise

        logger.info(f"✓ Loaded {len(self.sae_r)} residual SAEs and {len(self.sae_m)} MLP SAEs")
        return self.sae_r, self.sae_m

    def get_sae_info(self) -> Dict[str, any]:
        """Get information about loaded SAEs."""
        info = {
            "residual": {},
            "mlp": {},
            "expansion": self.expansion,
        }

        for layer, sae in self.sae_r.items():
            info["residual"][layer] = {
                "d_sae": sae.cfg.d_sae,
                "d_in": sae.cfg.d_in,
                "expansion_factor": sae.cfg.d_sae / sae.cfg.d_in,
                "sae_id": f"l{layer}r_{self.expansion}",
            }

        for layer, sae in self.sae_m.items():
            info["mlp"][layer] = {
                "d_sae": sae.cfg.d_sae,
                "d_in": sae.cfg.d_in,
                "expansion_factor": sae.cfg.d_sae / sae.cfg.d_in,
                "sae_id": f"l{layer}m_{self.expansion}",
            }

        return info

    def get_all_layer_keys(self) -> List[str]:
        """Get all layer keys in format 'R{layer}' or 'M{layer}'."""
        keys = []
        keys.extend([f"R{layer}" for layer in sorted(self.residual_layers)])
        keys.extend([f"M{layer}" for layer in sorted(self.mlp_layers)])
        return keys

    def get_sae(self, layer_key: str) -> SAE:
        """
        Get SAE by layer key.

        Args:
            layer_key: Layer key in format 'R{layer}' or 'M{layer}'

        Returns:
            SAE object
        """
        if layer_key.startswith("R"):
            layer = int(layer_key[1:])
            if layer not in self.sae_r:
                raise ValueError(f"Residual SAE for layer {layer} not loaded")
            return self.sae_r[layer]
        elif layer_key.startswith("M"):
            layer = int(layer_key[1:])
            if layer not in self.sae_m:
                raise ValueError(f"MLP SAE for layer {layer} not loaded")
            return self.sae_m[layer]
        else:
            raise ValueError(f"Invalid layer key: {layer_key}. Must start with 'R' or 'M'")

    def encode_layer(self, layer_key: str, activations: torch.Tensor) -> torch.Tensor:
        """
        Encode activations with SAE.

        Args:
            layer_key: Layer key in format 'R{layer}' or 'M{layer}'
            activations: Activations tensor [batch, hidden_size]

        Returns:
            SAE latents [batch, d_sae]
        """
        sae = self.get_sae(layer_key)
        return sae.encode(activations)

    def decode_layer(self, layer_key: str, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode SAE latents back to activations.

        Args:
            layer_key: Layer key in format 'R{layer}' or 'M{layer}'
            latents: SAE latents [batch, d_sae]

        Returns:
            Reconstructed activations [batch, hidden_size]
        """
        sae = self.get_sae(layer_key)
        return sae.decode(latents)

    def get_feature_dims(self) -> Dict[str, int]:
        """Get feature dimensions for each layer."""
        dims = {}
        for layer in self.residual_layers:
            key = f"R{layer}"
            dims[key] = self.sae_r[layer].cfg.d_sae
        for layer in self.mlp_layers:
            key = f"M{layer}"
            dims[key] = self.sae_m[layer].cfg.d_sae
        return dims

    @staticmethod
    def list_available_saes(expansion: str = "8x") -> Dict[str, List[str]]:
        """
        List available SAE IDs for Llama-Scope.

        Args:
            expansion: Expansion factor ("8x" or "32x")

        Returns:
            Dictionary with 'residual' and 'mlp' lists of SAE IDs
        """
        # Llama-3.1-8B has 32 layers (0-31)
        available = {
            "residual": [f"l{i}r_{expansion}" for i in range(32)],
            "mlp": [f"l{i}m_{expansion}" for i in range(32)],
            "attention": [f"l{i}a_{expansion}" for i in range(32)],  # Not recommended
            "transcoder": [f"l{i}tc_{expansion}" for i in range(32)],
        }
        return available


# Convenience function for quick loading
def load_llama_scope_saes(
    residual_layers: List[int] = [5, 12, 18, 24, 28, 31],
    mlp_layers: List[int] = [15, 22],
    expansion: str = "8x",
    device: str = "cuda",
) -> tuple[Dict[int, SAE], Dict[int, SAE]]:
    """
    Quick function to load Llama-Scope SAEs.

    Args:
        residual_layers: List of layer indices for residual stream SAEs
        mlp_layers: List of layer indices for MLP SAEs
        expansion: Expansion factor ("8x" or "32x")
        device: Device to load SAEs on

    Returns:
        Tuple of (residual_saes, mlp_saes) dictionaries
    """
    loader = SAELoader(
        residual_layers=residual_layers,
        mlp_layers=mlp_layers,
        expansion=expansion,
        device=device
    )
    return loader.load()

