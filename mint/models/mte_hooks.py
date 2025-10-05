"""MTE activation hooks for generation.

This module provides forward hooks that apply MTE edits during model generation.
The hooks intercept activations at selected layers, encode them to SAE features,
apply MTE edits, and decode back to hidden states.
"""

from typing import Dict, List, Optional, Callable

import torch
import torch.nn as nn

from mint.models.mte import MechanisticToolEditor
from mint.models.sae_loader import load_sae
from mint.logging_utils import setup_logger

logger = setup_logger(__name__)


class MTEHookManager:
    """Manager for MTE forward hooks during generation.

    This class registers forward hooks on transformer layers that:
    1. Encode hidden states to SAE features
    2. Apply MTE edits
    3. Decode back to hidden states
    4. Replace the original activations
    """

    def __init__(
        self,
        mte: MechanisticToolEditor,
        layers: List[int],
        sae_checkpoints: Dict[str, str],
        hidden_dim: int = 4096,
        expansion_factor: int = 8,
        edit_strength: float = 1.0,
        device: str = "cuda",
    ):
        """Initialize hook manager.

        Args:
            mte: Trained MTE model
            layers: Layers to apply hooks to
            sae_checkpoints: Dict mapping layer_id (as string) to checkpoint path
            hidden_dim: Hidden dimension
            expansion_factor: SAE expansion factor
            edit_strength: Scalar multiplier for edit magnitude
            device: Device
        """
        self.mte = mte
        self.layers = layers
        self.sae_checkpoints = sae_checkpoints
        self.hidden_dim = hidden_dim
        self.expansion_factor = expansion_factor
        self.edit_strength = edit_strength
        self.device = device

        # Load SAEs for all layers
        self.saes = {}
        for layer_id in layers:
            checkpoint_path = sae_checkpoints.get(str(layer_id))
            self.saes[layer_id] = load_sae(
                layer_id=layer_id,
                hidden_dim=hidden_dim,
                expansion_factor=expansion_factor,
                checkpoint_path=checkpoint_path,
                device=device,
            )

        # Store hook handles
        self.hook_handles = []

    def _create_hook(self, layer_id: int) -> Callable:
        """Create a forward hook for a specific layer.

        Args:
            layer_id: Layer index

        Returns:
            Hook function
        """
        def hook_fn(module, input, output):
            """Forward hook that applies MTE edit.

            Args:
                module: The layer module
                input: Input tuple (usually contains hidden states)
                output: Output tuple (hidden_states, ...)

            Returns:
                Modified output with MTE edits applied
            """
            # Extract hidden states from output
            # For transformer layers, output is typically (hidden_states, ...)
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # Encode to SAE features
            sae = self.saes[layer_id]
            phi_no_tool = sae.encode(hidden_states)

            # Apply MTE edit
            with torch.no_grad():
                phi_dict = {layer_id: phi_no_tool}
                phi_edited_dict = self.mte(phi_dict)
                phi_edited = phi_edited_dict[layer_id]

            # Apply edit strength
            if self.edit_strength != 1.0:
                delta = phi_edited - phi_no_tool
                phi_edited = phi_no_tool + self.edit_strength * delta

            # Decode back to hidden states
            hidden_states_edited = sae.decode(phi_edited)

            # Return modified output
            if isinstance(output, tuple):
                return (hidden_states_edited,) + output[1:]
            else:
                return hidden_states_edited

        return hook_fn

    def register_hooks(self, model: nn.Module) -> None:
        """Register forward hooks on model layers.

        Args:
            model: The transformer model
        """
        # For Llama models, layers are in model.model.layers
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            transformer_layers = model.model.layers
        elif hasattr(model, "layers"):
            transformer_layers = model.layers
        else:
            raise ValueError("Could not find transformer layers in model")

        logger.info(f"Registering MTE hooks on layers: {self.layers}")

        for layer_id in self.layers:
            if layer_id >= len(transformer_layers):
                logger.warning(f"Layer {layer_id} out of range, skipping")
                continue

            layer = transformer_layers[layer_id]
            hook = self._create_hook(layer_id)
            handle = layer.register_forward_hook(hook)
            self.hook_handles.append(handle)

        logger.info(f"Registered {len(self.hook_handles)} MTE hooks")

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
        logger.info("Removed all MTE hooks")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - remove hooks."""
        self.remove_hooks()


def apply_mte_during_generation(
    model: nn.Module,
    mte: MechanisticToolEditor,
    layers: List[int],
    sae_checkpoints: Dict[str, str],
    hidden_dim: int = 4096,
    expansion_factor: int = 8,
    edit_strength: float = 1.0,
) -> MTEHookManager:
    """Apply MTE edits during generation using forward hooks.

    This is a convenience function that creates and registers an MTEHookManager.
    Use as a context manager to automatically remove hooks after generation.

    Args:
        model: Transformer model
        mte: Trained MTE model
        layers: Layers to apply edits to
        sae_checkpoints: Dict mapping layer_id (as string) to checkpoint path
        hidden_dim: Hidden dimension
        expansion_factor: SAE expansion factor
        edit_strength: Scalar multiplier for edit magnitude

    Returns:
        MTEHookManager instance (use as context manager)

    Example:
        >>> with apply_mte_during_generation(model, mte, layers, sae_ckpts) as hook_mgr:
        >>>     outputs = model.generate(**inputs)
    """
    device = next(model.parameters()).device

    hook_manager = MTEHookManager(
        mte=mte,
        layers=layers,
        sae_checkpoints=sae_checkpoints,
        hidden_dim=hidden_dim,
        expansion_factor=expansion_factor,
        edit_strength=edit_strength,
        device=device,
    )

    hook_manager.register_hooks(model)

    return hook_manager
