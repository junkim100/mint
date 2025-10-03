"""
Value head for predicting utility gains from tool use.

Implements Section 3.3 of the MINT proposal:
    ΔV̂_u = g(H̃_t^(u)) - g(H_t)

where g: H → V maps hidden states to value predictions, and we compute
the counterfactual utility gain by taking the difference between edited
and baseline state values.

Reference: MINT Proposal Section 3.3 (Hidden-State Counterfactuals)
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class ValueHead(nn.Module):
    """
    Value head that predicts utility V(H) from hidden states.

    Implements the value function g(H) from Proposal Section 3.3.
    To compute counterfactual utility gains:
        ΔV̂_u = g(H̃_t^(u)) - g(H_t)

    This is the CORRECT architecture per the proposal - we compute
    g(H̃) and g(H) separately, then take their difference.
    """

    def __init__(
        self,
        hidden_size: int,
        num_r_layers: int,
        hidden_dims: List[int] = [2048, 256],
        dropout: float = 0.1,
    ):
        """
        Initialize value head.

        Args:
            hidden_size: Hidden size of each layer (d_model)
            num_r_layers: Number of R layers to concatenate
            hidden_dims: List of hidden layer dimensions for MLP
            dropout: Dropout probability
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_r_layers = num_r_layers

        # Input dimension: concatenated R-layer activations
        # This is the dimension of H in the proposal
        d_in = hidden_size * num_r_layers

        # Build MLP: g(H) → V
        # This is the function g(·) from Proposal Section 3.3
        layers = []
        prev_dim = d_in

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        # Output layer: maps to scalar value V
        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, hidden_states: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Predict value V(H) from hidden states.

        This implements g(H) from Proposal Section 3.3.

        Args:
            hidden_states: Dictionary mapping layer keys to hidden states
                          e.g., {"R5": [batch, d_model], "R12": [batch, d_model], ...}

        Returns:
            Predicted value V(H) [batch]
        """
        # Concatenate all layers in sorted order
        h_list = [hidden_states[k] for k in sorted(hidden_states.keys())]
        h_concat = torch.cat(h_list, dim=-1)  # [batch, num_r_layers * hidden_size]

        # Predict value: V = g(H)
        value = self.net(h_concat)  # [batch, 1]
        return value.squeeze(-1)  # [batch]

    def predict_delta_v(
        self,
        baseline_states: Dict[str, torch.Tensor],
        edited_states: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Predict counterfactual utility gain ΔV̂_u = g(H̃_t^(u)) - g(H_t).

        This is the KEY method that implements the proposal's counterfactual
        value estimation (Proposal Section 3.3).

        Args:
            baseline_states: Baseline hidden states H_t
            edited_states: Edited (counterfactual) hidden states H̃_t^(u)

        Returns:
            Predicted utility gain ΔV̂_u [batch]
        """
        # Compute V(H̃_t^(u))
        v_edited = self.forward(edited_states)

        # Compute V(H_t)
        v_baseline = self.forward(baseline_states)

        # Compute difference: ΔV̂_u = g(H̃_t^(u)) - g(H_t)
        delta_v = v_edited - v_baseline

        return delta_v

    def forward_from_list(self, acts_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Predict value from list of activation tensors (backward compatibility).

        Args:
            acts_list: List of activation tensors [batch, hidden_size]
                      Should be in order of R layers

        Returns:
            Predicted V [batch]
        """
        # Concatenate activations
        x = torch.cat(acts_list, dim=-1)  # [batch, hidden_size * num_r_layers]

        # Forward through network
        out = self.net(x).squeeze(-1)  # [batch]

        return out


class MultiToolValueHead(nn.Module):
    """
    Wrapper for multiple value heads (one per tool).
    """

    def __init__(
        self,
        tool_names: List[str],
        hidden_size: int,
        num_r_layers: int,
        hidden_dims: List[int] = [2048, 256],
        dropout: float = 0.1,
        shared_backbone: bool = False,
    ):
        """
        Initialize multi-tool value head.

        Args:
            tool_names: List of tool names
            hidden_size: Hidden size of each layer
            num_r_layers: Number of R layers
            hidden_dims: Hidden layer dimensions
            dropout: Dropout probability
            shared_backbone: Whether to share backbone across tools
        """
        super().__init__()

        self.tool_names = tool_names
        self.shared_backbone = shared_backbone

        if shared_backbone:
            # Shared backbone with tool-specific heads
            d_in = hidden_size * num_r_layers

            # Shared layers
            shared_layers = []
            prev_dim = d_in
            for hidden_dim in hidden_dims[:-1]:
                shared_layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ])
                prev_dim = hidden_dim

            self.backbone = nn.Sequential(*shared_layers)

            # Tool-specific heads
            self.heads = nn.ModuleDict({
                tool: nn.Sequential(
                    nn.Linear(prev_dim, hidden_dims[-1]),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dims[-1], 1)
                )
                for tool in tool_names
            })
        else:
            # Separate value head per tool
            self.heads = nn.ModuleDict({
                tool: ValueHead(hidden_size, num_r_layers, hidden_dims, dropout)
                for tool in tool_names
            })

    def forward(
        self,
        tool_name: str,
        acts_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Predict ΔV for a specific tool.

        Args:
            tool_name: Name of the tool
            acts_list: List of activation tensors

        Returns:
            Predicted ΔV [batch]
        """
        if tool_name not in self.heads:
            raise ValueError(f"Unknown tool: {tool_name}")

        if self.shared_backbone:
            x = torch.cat(acts_list, dim=-1)
            features = self.backbone(x)
            return self.heads[tool_name](features).squeeze(-1)
        else:
            return self.heads[tool_name](acts_list)

