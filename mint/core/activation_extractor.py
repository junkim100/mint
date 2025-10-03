"""Activation extraction utilities for MINT."""

import torch
from typing import Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logger = logging.getLogger(__name__)


class ActivationExtractor:
    """Extracts hidden states and encodes them with SAEs at decision points."""
    
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        sae_loader,
        residual_layers: List[int] = [5, 12, 18, 24, 28, 31],
        mlp_layers: List[int] = [15, 22],
    ):
        """
        Initialize activation extractor.
        
        Args:
            model: Loaded language model
            tokenizer: Tokenizer
            sae_loader: SAELoader instance with loaded SAEs
            residual_layers: List of residual layer indices
            mlp_layers: List of MLP layer indices
        """
        self.model = model
        self.tokenizer = tokenizer
        self.sae_loader = sae_loader
        self.residual_layers = residual_layers
        self.mlp_layers = mlp_layers
        
    def extract_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position: str = "last",
    ) -> Dict[str, torch.Tensor]:
        """
        Extract hidden states at specified position.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            position: Position to extract ('last', 'all', or int index)
            
        Returns:
            Dictionary mapping layer keys to hidden states [batch, hidden_size]
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=True
            )
        
        # HF returns tuple: hidden_states[k] is after block k-1 (k=0 is embeddings)
        hidden_states = outputs.hidden_states  # len = n_layers+1, each [B, T, H]
        
        # Determine position indices
        if position == "last":
            # Last non-padding token per sequence
            last_idx = attention_mask.sum(dim=1) - 1  # [batch]
            batch_indices = torch.arange(hidden_states[0].size(0), device=hidden_states[0].device)
        elif position == "all":
            # Return all positions (for sequence-level analysis)
            pass
        elif isinstance(position, int):
            # Specific position
            last_idx = torch.full((hidden_states[0].size(0),), position, device=hidden_states[0].device)
            batch_indices = torch.arange(hidden_states[0].size(0), device=hidden_states[0].device)
        else:
            raise ValueError(f"Invalid position: {position}")
        
        # Extract activations per layer
        activations = {}
        
        # Residual layers
        for layer in self.residual_layers:
            # hidden_states[layer+1] is the output after block `layer`
            h_layer = hidden_states[layer + 1]  # [batch, seq_len, hidden_size]
            
            if position == "all":
                activations[f"R{layer}"] = h_layer  # [batch, seq_len, hidden_size]
            else:
                # Extract at specific position
                h_pos = h_layer[batch_indices, last_idx]  # [batch, hidden_size]
                activations[f"R{layer}"] = h_pos
        
        # MLP layers (approximate as block output for now)
        for layer in self.mlp_layers:
            h_layer = hidden_states[layer + 1]  # [batch, seq_len, hidden_size]
            
            if position == "all":
                activations[f"M{layer}"] = h_layer
            else:
                h_pos = h_layer[batch_indices, last_idx]
                activations[f"M{layer}"] = h_pos
        
        return activations
    
    def encode_with_saes(
        self,
        activations: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Encode activations with SAEs.
        
        Args:
            activations: Dictionary mapping layer keys to activations
            
        Returns:
            Dictionary mapping layer keys to SAE latents
        """
        latents = {}
        
        for layer_key, acts in activations.items():
            # Handle both [batch, hidden] and [batch, seq_len, hidden]
            original_shape = acts.shape
            if len(original_shape) == 3:
                # Flatten sequence dimension
                batch_size, seq_len, hidden_size = original_shape
                acts_flat = acts.reshape(-1, hidden_size)
                latents_flat = self.sae_loader.encode_layer(layer_key, acts_flat)
                # Reshape back
                latents[layer_key] = latents_flat.reshape(batch_size, seq_len, -1)
            else:
                latents[layer_key] = self.sae_loader.encode_layer(layer_key, acts)
        
        return latents
    
    def extract_and_encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position: str = "last",
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Extract hidden states and encode with SAEs in one call.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            position: Position to extract
            
        Returns:
            Tuple of (activations, latents) dictionaries
        """
        activations = self.extract_hidden_states(input_ids, attention_mask, position)
        latents = self.encode_with_saes(activations)
        return activations, latents
    
    def concat_latents(
        self,
        latents: Dict[str, torch.Tensor],
        layer_order: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, List[str], List[int]]:
        """
        Concatenate latents from all layers into a single vector.
        
        Args:
            latents: Dictionary mapping layer keys to latents
            layer_order: Optional order of layers (default: sorted R then M)
            
        Returns:
            Tuple of (concatenated_latents, layer_keys, dims_per_layer)
        """
        if layer_order is None:
            # Default order: R layers sorted, then M layers sorted
            r_keys = sorted([k for k in latents.keys() if k.startswith("R")],
                          key=lambda x: int(x[1:]))
            m_keys = sorted([k for k in latents.keys() if k.startswith("M")],
                          key=lambda x: int(x[1:]))
            layer_order = r_keys + m_keys
        
        # Concatenate
        latent_list = [latents[k] for k in layer_order]
        concat_latents = torch.cat(latent_list, dim=-1)
        
        # Track dimensions
        dims_per_layer = [latents[k].shape[-1] for k in layer_order]
        
        return concat_latents, layer_order, dims_per_layer

