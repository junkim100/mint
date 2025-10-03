"""
Faithfulness regularization for MINT editors.

Implements Proposal Section 4, Phase C (Faithfulness & Causal Regularization):
- Ablation faithfulness: Penalize decision invariance when features ablated
- Contrastive causal InfoNCE: Encourage ΔV̂_u* > ΔV̂_u≠u*

Reference: MINT Proposal Section 4, Phase C
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class AblationFaithfulnessLoss(nn.Module):
    """
    Ablation faithfulness loss.
    
    Implements the proposal's ablation faithfulness:
    "Randomly zero top-influence features in m_u^(ℓ); 
    penalize decision invariance"
    
    L_faith = E[𝟙{decision unchanged}]
    
    Reference: MINT Proposal Section 4, Phase C
    """
    
    def __init__(
        self,
        ablation_ratio: float = 0.2,
        top_k_features: int = 100,
    ):
        """
        Initialize ablation faithfulness loss.
        
        Args:
            ablation_ratio: Fraction of top features to ablate
            top_k_features: Number of top features to consider
        """
        super().__init__()
        self.ablation_ratio = ablation_ratio
        self.top_k_features = top_k_features
    
    def forward(
        self,
        editor: nn.Module,
        value_head: nn.Module,
        hidden_states: Dict[str, torch.Tensor],
        ctx_vec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute ablation faithfulness loss.
        
        Args:
            editor: MechanisticEditor instance
            value_head: ValueHead instance
            hidden_states: Baseline hidden states
            ctx_vec: Optional context vector
            
        Returns:
            Faithfulness loss (lower is better)
        """
        with torch.no_grad():
            # Get original decision (ΔV prediction)
            edited_original = editor(hidden_states, ctx_vec)
            delta_v_original = value_head.predict_delta_v(
                hidden_states, edited_original
            )
        
        # Ablate top-k features in the editor's masks
        # This simulates "what if these features weren't important?"
        ablated_editor = self._ablate_top_features(editor)
        
        # Get decision with ablated features
        edited_ablated = ablated_editor(hidden_states, ctx_vec)
        delta_v_ablated = value_head.predict_delta_v(
            hidden_states, edited_ablated
        )
        
        # Compute decision change
        # We want decisions to CHANGE when important features are ablated
        # So we penalize SMALL changes (decision invariance)
        delta_change = torch.abs(delta_v_ablated - delta_v_original)
        
        # Loss: penalize small changes (want large changes when ablating)
        # Use negative log to encourage larger changes
        faithfulness_loss = -torch.log(delta_change + 1e-8).mean()
        
        return faithfulness_loss
    
    def _ablate_top_features(self, editor: nn.Module) -> nn.Module:
        """
        Create a copy of editor with top features ablated.
        
        Args:
            editor: Original editor
            
        Returns:
            Editor with ablated features
        """
        # Create a temporary copy (we'll modify masks)
        import copy
        ablated_editor = copy.deepcopy(editor)
        
        # For each layer, ablate top-k features
        for layer_key in editor.layer_keys:
            mask = editor._get_mask(layer_key)
            direction = editor._get_direction(layer_key)
            
            # Find top-k features by absolute direction magnitude
            abs_dir = torch.abs(direction)
            topk_indices = torch.topk(
                abs_dir, 
                k=min(self.top_k_features, len(abs_dir))
            ).indices
            
            # Ablate a fraction of top features
            num_ablate = int(len(topk_indices) * self.ablation_ratio)
            ablate_indices = topk_indices[:num_ablate]
            
            # Zero out these features in the mask
            ablated_mask = mask.clone()
            ablated_mask[ablate_indices] = 0.0
            
            # Update the ablated editor's mask
            ablated_editor.register_buffer(
                f"mask_{layer_key}", 
                ablated_mask
            )
        
        return ablated_editor


class ContrastiveCausalLoss(nn.Module):
    """
    Contrastive causal InfoNCE loss.
    
    Implements the proposal's contrastive loss:
    "Encourage ΔV̂_u* > ΔV̂_u≠u* on the same state"
    
    For a state where tool u* is optimal, we want:
    ΔV̂_u* > ΔV̂_u for all u ≠ u*
    
    Reference: MINT Proposal Section 4, Phase C
    """
    
    def __init__(
        self,
        temperature: float = 0.1,
        margin: float = 0.5,
    ):
        """
        Initialize contrastive causal loss.
        
        Args:
            temperature: Temperature for InfoNCE
            margin: Margin for ranking loss
        """
        super().__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(
        self,
        editors: Dict[str, nn.Module],
        value_heads: Dict[str, nn.Module],
        hidden_states: Dict[str, torch.Tensor],
        optimal_tool: str,
        ctx_vec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute contrastive causal loss.
        
        Args:
            editors: Dict of tool name -> MechanisticEditor
            value_heads: Dict of tool name -> ValueHead
            hidden_states: Baseline hidden states
            optimal_tool: Name of the optimal tool for this state
            ctx_vec: Optional context vector
            
        Returns:
            Contrastive loss
        """
        # Compute ΔV for all tools
        delta_vs = {}
        
        for tool_name, editor in editors.items():
            # Apply editor
            edited = editor(hidden_states, ctx_vec)
            
            # Predict ΔV
            delta_v = value_heads[tool_name].predict_delta_v(
                hidden_states, edited
            )
            delta_vs[tool_name] = delta_v
        
        # Get optimal tool's ΔV
        delta_v_optimal = delta_vs[optimal_tool]
        
        # InfoNCE loss: maximize delta_v_optimal relative to others
        # exp(ΔV_u* / τ) / Σ_u exp(ΔV_u / τ)
        logits = torch.stack([
            delta_vs[tool] / self.temperature 
            for tool in sorted(delta_vs.keys())
        ], dim=0)  # [num_tools, batch]
        
        # Get index of optimal tool
        tool_names = sorted(delta_vs.keys())
        optimal_idx = tool_names.index(optimal_tool)
        
        # Cross-entropy loss (want optimal tool to have highest logit)
        targets = torch.full(
            (logits.size(1),), 
            optimal_idx, 
            dtype=torch.long,
            device=logits.device
        )
        
        ce_loss = F.cross_entropy(logits.T, targets)
        
        # Also add margin ranking loss for robustness
        # Want: ΔV_u* > ΔV_u + margin for all u ≠ u*
        ranking_losses = []
        for tool_name, delta_v in delta_vs.items():
            if tool_name != optimal_tool:
                # Margin ranking: want delta_v_optimal > delta_v + margin
                ranking_loss = F.relu(
                    self.margin - (delta_v_optimal - delta_v)
                )
                ranking_losses.append(ranking_loss)
        
        if ranking_losses:
            ranking_loss = torch.stack(ranking_losses).mean()
        else:
            ranking_loss = torch.tensor(0.0, device=logits.device)
        
        # Combined loss
        total_loss = ce_loss + ranking_loss
        
        return total_loss


class FaithfulnessRegularizer:
    """
    Combined faithfulness regularizer for Phase C.
    
    Combines ablation faithfulness and contrastive causal losses.
    """
    
    def __init__(
        self,
        lambda_ablation: float = 0.1,
        lambda_contrastive: float = 0.5,
        ablation_ratio: float = 0.2,
        temperature: float = 0.1,
        margin: float = 0.5,
    ):
        """
        Initialize faithfulness regularizer.
        
        Args:
            lambda_ablation: Weight for ablation faithfulness loss
            lambda_contrastive: Weight for contrastive causal loss
            ablation_ratio: Fraction of features to ablate
            temperature: Temperature for InfoNCE
            margin: Margin for ranking loss
        """
        self.lambda_ablation = lambda_ablation
        self.lambda_contrastive = lambda_contrastive
        
        self.ablation_loss = AblationFaithfulnessLoss(
            ablation_ratio=ablation_ratio
        )
        self.contrastive_loss = ContrastiveCausalLoss(
            temperature=temperature,
            margin=margin,
        )
    
    def compute_loss(
        self,
        editor: nn.Module,
        value_head: nn.Module,
        hidden_states: Dict[str, torch.Tensor],
        ctx_vec: Optional[torch.Tensor] = None,
        editors_all: Optional[Dict[str, nn.Module]] = None,
        value_heads_all: Optional[Dict[str, nn.Module]] = None,
        optimal_tool: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined faithfulness loss.
        
        Args:
            editor: Current tool's editor
            value_head: Current tool's value head
            hidden_states: Baseline hidden states
            ctx_vec: Optional context vector
            editors_all: Dict of all editors (for contrastive loss)
            value_heads_all: Dict of all value heads (for contrastive loss)
            optimal_tool: Name of optimal tool (for contrastive loss)
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        losses = {}
        total_loss = 0.0
        
        # Ablation faithfulness loss
        if self.lambda_ablation > 0:
            ablation_loss = self.ablation_loss(
                editor, value_head, hidden_states, ctx_vec
            )
            losses['ablation_faith'] = ablation_loss.item()
            total_loss += self.lambda_ablation * ablation_loss
        
        # Contrastive causal loss (if multi-tool context available)
        if (self.lambda_contrastive > 0 and 
            editors_all is not None and 
            value_heads_all is not None and
            optimal_tool is not None):
            contrastive_loss = self.contrastive_loss(
                editors_all, value_heads_all, hidden_states, 
                optimal_tool, ctx_vec
            )
            losses['contrastive'] = contrastive_loss.item()
            total_loss += self.lambda_contrastive * contrastive_loss
        
        return total_loss, losses

