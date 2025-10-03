"""
Mechanistic Tool Editors for MINT.

Implements Section 3.2 of the MINT proposal:
    φ̃^(ℓ) = φ^(ℓ) + α_u^(ℓ) ⊙ m_u^(ℓ) ⊙ w_u^(ℓ)

And Section 3.3 (Hidden-State Counterfactuals):
    "Run a short forward pass with ã^(ℓ) substituted for a^(ℓ)"

Reference: MINT Proposal Sections 3.2 and 3.3
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class GateMLP(nn.Module):
    """
    Hypernetwork that predicts context-dependent gates α_u^(ℓ).

    Implements the "tiny hypernetwork that reads a pooled state and tool descriptor"
    from Proposal Section 3.2.

    The gates control the strength of feature-space interventions:
        φ̃^(ℓ) = φ^(ℓ) + α_u^(ℓ) ⊙ m_u^(ℓ) ⊙ w_u^(ℓ)

    Reference: MINT Proposal Section 3.2 (Mechanistic Tool Editors)
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        hidden_dim: int = 512,
        alpha_max: float = 0.5,
    ):
        """
        Initialize gate hypernetwork.

        Args:
            d_in: Input dimension (pooled context vector size)
            d_out: Output dimension (number of gates = number of intervention layers)
            hidden_dim: Hidden layer dimension
            alpha_max: Maximum gate value (controls edit magnitude)
        """
        super().__init__()
        self.alpha_max = alpha_max

        # Small MLP: context → gates
        # This is the "tiny hypernetwork" from the proposal
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, d_out),
            nn.Sigmoid()  # Output in [0, 1]
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Predict gates α_u^(ℓ) from pooled context.

        Args:
            context: Pooled context vector [batch, d_in]
                    Typically: mean-pooled R-layer activations

        Returns:
            Gates α_u^(ℓ) [batch, d_out] in range [0, alpha_max]
        """
        # Predict gates in [0, 1], then scale to [0, alpha_max]
        return self.alpha_max * self.net(context)


class MechanisticEditor(nn.Module):
    """
    Mechanistic Tool Editor that applies feature-space interventions.

    Implements Proposal Section 3.2:
        φ̃^(ℓ) = φ^(ℓ) + α_u^(ℓ) ⊙ m_u^(ℓ) ⊙ w_u^(ℓ)

    where:
    - φ^(ℓ): SAE latents at layer ℓ
    - m_u^(ℓ): Binary mask indicating tool-relevant features
    - w_u^(ℓ): Signed direction vector for edits
    - α_u^(ℓ): Context-dependent gate (learned via hypernetwork)

    Then decode back to get edited activations ã^(ℓ).

    Reference: MINT Proposal Section 3.2 (Mechanistic Tool Editors)
    """

    def __init__(
        self,
        masks: Dict[str, torch.Tensor],
        directions: Dict[str, torch.Tensor],
        sae_loader,
        alpha_max: float = 0.5,
        use_learned_gates: bool = True,
        gate_hidden_dim: int = 512,
    ):
        """
        Initialize mechanistic editor.

        Args:
            masks: Dict mapping layer keys to binary masks m_u^(ℓ) [d_sae]
                  Indicates which features are tool-relevant
            directions: Dict mapping layer keys to signed direction vectors w_u^(ℓ) [d_sae]
                       Indicates how to edit each feature
            sae_loader: SAELoader instance for encoding/decoding
            alpha_max: Maximum gate value (controls edit magnitude)
            use_learned_gates: Whether to use context-dependent gates α_u^(ℓ)
                              (True = learned via hypernetwork, False = fixed scalar)
            gate_hidden_dim: Hidden dimension for gate hypernetwork
        """
        super().__init__()

        self.layer_keys = sorted(masks.keys())
        self.sae_loader = sae_loader
        self.alpha_max = alpha_max
        self.use_learned_gates = use_learned_gates

        # Register masks m_u^(ℓ) and directions w_u^(ℓ) as buffers (not parameters)
        # These are learned during affordance discovery (Phase A)
        for key in self.layer_keys:
            self.register_buffer(f"mask_{key}", masks[key].float())
            self.register_buffer(f"dir_{key}", directions[key])

        # Context-dependent gates α_u^(ℓ)
        # This is the "tiny hypernetwork" from Proposal Section 3.2
        if use_learned_gates:
            # Context dimension: concatenated R-layer hidden states
            r_keys = [k for k in self.layer_keys if k.startswith("R")]
            d_ctx = len(r_keys) * sae_loader.sae_r[int(r_keys[0][1:])].cfg.d_in

            # Hypernetwork: pooled context → per-layer gates
            self.gates = GateMLP(
                d_in=d_ctx,
                d_out=len(self.layer_keys),
                hidden_dim=gate_hidden_dim,
                alpha_max=alpha_max
            )
        else:
            self.gates = None

    def _get_mask(self, layer_key: str) -> torch.Tensor:
        """Get mask for layer."""
        return getattr(self, f"mask_{layer_key}")

    def _get_direction(self, layer_key: str) -> torch.Tensor:
        """Get direction for layer."""
        return getattr(self, f"dir_{layer_key}")

    def forward(
        self,
        hidden_by_layer: Dict[str, torch.Tensor],
        ctx_vec: Optional[torch.Tensor] = None,
        gate_override: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply feature-space edits to hidden states.

        Args:
            hidden_by_layer: Dict mapping layer keys to activations [batch, hidden_size]
            ctx_vec: Optional context vector for learned gates [batch, d_ctx]
            gate_override: Optional fixed gate value (overrides learned gates)

        Returns:
            Dict mapping layer keys to edited activations [batch, hidden_size]
        """
        edited = {}

        # Determine gates
        if gate_override is not None:
            # Use fixed gate for all layers
            gates_per_layer = {k: gate_override for k in self.layer_keys}
        elif self.use_learned_gates and ctx_vec is not None:
            # Use learned gates
            gate_values = self.gates(ctx_vec)  # [batch, num_layers]
            gates_per_layer = {
                k: gate_values[:, i].unsqueeze(-1)  # [batch, 1]
                for i, k in enumerate(self.layer_keys)
            }
        else:
            # Default: fixed scalar gate
            gates_per_layer = {k: self.alpha_max * 0.5 for k in self.layer_keys}

        # Apply edits per layer
        for layer_key in self.layer_keys:
            if layer_key not in hidden_by_layer:
                logger.warning(f"Layer {layer_key} not in hidden_by_layer, skipping")
                continue

            # Get SAE
            sae = self.sae_loader.get_sae(layer_key)

            # Encode
            h = hidden_by_layer[layer_key]  # [batch, hidden_size]
            phi = sae.encode(h)  # [batch, d_sae]

            # Get mask and direction
            mask = self._get_mask(layer_key)  # [d_sae]
            direction = self._get_direction(layer_key)  # [d_sae]

            # Compute delta
            gate = gates_per_layer[layer_key]
            if isinstance(gate, float):
                delta_latent = gate * (mask * direction)  # [d_sae]
                delta_latent = delta_latent.unsqueeze(0).expand_as(phi)  # [batch, d_sae]
            else:
                # gate is [batch, 1]
                delta_latent = gate * (mask * direction).unsqueeze(0)  # [batch, d_sae]

            # Apply edit
            phi_tilde = phi + delta_latent

            # Decode
            h_tilde = sae.decode(phi_tilde)  # [batch, hidden_size]

            edited[layer_key] = h_tilde

        return edited

    def get_edit_norm(
        self,
        hidden_by_layer: Dict[str, torch.Tensor],
        ctx_vec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute L2 norm of edits (for regularization).

        Args:
            hidden_by_layer: Dict mapping layer keys to activations
            ctx_vec: Optional context vector

        Returns:
            Scalar tensor with total edit norm
        """
        edited = self.forward(hidden_by_layer, ctx_vec)

        total_norm = 0.0
        for key in self.layer_keys:
            if key in hidden_by_layer and key in edited:
                delta = edited[key] - hidden_by_layer[key]
                total_norm += torch.norm(delta, p=2) ** 2

        return total_norm

    def generate_counterfactual_state(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        baseline_hidden_states: Dict[str, torch.Tensor],
        ctx_vec: Optional[torch.Tensor] = None,
        gate_override: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate counterfactual hidden state H̃_t^(u) via forward pass with edited activations.

        This implements the KEY missing piece from Proposal Section 3.3:
        "Run a short forward pass with ã^(ℓ) substituted for a^(ℓ)"

        The procedure:
        1. Apply feature-space edits to get h̃^(ℓ) for intervention layers
        2. Run model forward pass with intervention hooks that substitute h̃^(ℓ) for h^(ℓ)
        3. Extract final counterfactual hidden state H̃_t^(u)

        This enables true counterfactual reasoning: "what would the model's internal
        state look like if the tool knowledge were present?"

        Args:
            model: Language model (transformers model)
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            baseline_hidden_states: Baseline hidden states H_t (before editing)
            ctx_vec: Optional context vector for learned gates
            gate_override: Optional fixed gate value

        Returns:
            Counterfactual hidden states H̃_t^(u) as Dict[layer_key, tensor]
        """
        # Step 1: Apply feature-space edits to get h̃^(ℓ)
        edited_states = self.forward(
            baseline_hidden_states,
            ctx_vec=ctx_vec,
            gate_override=gate_override,
        )

        # Step 2: Map layer keys to layer indices
        # Layer keys are like "R5", "R12", "M15", etc.
        intervention_layers = {}
        for layer_key in self.layer_keys:
            if layer_key.startswith("R"):
                layer_idx = int(layer_key[1:])
                intervention_layers[layer_idx] = edited_states[layer_key]
            elif layer_key.startswith("M"):
                # MLP layers - for now we skip these in forward pass interventions
                # (they would require intervening in MLP sublayer specifically)
                pass

        # Step 3: Run forward pass with intervention hooks
        with torch.no_grad():
            # Define intervention hook
            def make_intervention_hook(layer_idx: int, edited_activation: torch.Tensor):
                """Create hook that replaces layer output with edited activation."""
                def hook(module, input, output):
                    # output is typically a tuple (hidden_states, ...)
                    if isinstance(output, tuple):
                        # Replace hidden states (first element)
                        # Use edited activation at last token position
                        original_hidden = output[0]
                        modified_hidden = original_hidden.clone()
                        modified_hidden[:, -1, :] = edited_activation
                        return (modified_hidden,) + output[1:]
                    else:
                        # Direct hidden state tensor
                        modified = output.clone()
                        modified[:, -1, :] = edited_activation
                        return modified
                return hook

            # Register hooks for intervention layers
            hooks = []
            for layer_idx, edited_act in intervention_layers.items():
                if layer_idx < len(model.model.layers):
                    hook = model.model.layers[layer_idx].register_forward_hook(
                        make_intervention_hook(layer_idx, edited_act)
                    )
                    hooks.append(hook)

            # Forward pass with interventions
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

            # Extract counterfactual hidden states
            # Get hidden states from all layers
            all_hidden_states = outputs.hidden_states  # Tuple of [batch, seq, hidden]

            counterfactual_states = {}
            for layer_key in self.layer_keys:
                if layer_key.startswith("R"):
                    layer_idx = int(layer_key[1:])
                    if layer_idx < len(all_hidden_states):
                        # Extract last token position
                        h_cf = all_hidden_states[layer_idx][:, -1, :]  # [batch, hidden]
                        counterfactual_states[layer_key] = h_cf

            # Remove hooks
            for hook in hooks:
                hook.remove()

        return counterfactual_states

