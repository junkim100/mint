"""
Phase A: Counterfactual supervision for editors.

Implements Proposal Section 4, Phase A (Counterfactual Supervision):
    Minimize ||Π_edit(E_u*(H_t) - H_t^(+u*))||²

where:
- E_u*: Editor for tool u*
- H_t: No-tool hidden state
- H_t^(+u*): With-tool hidden state (from real τ-bench tool outputs)
- Π_edit: Projection onto features in m_u*^(ℓ) (masked features)

This teaches editors to recreate the mechanistic effect of the tool.

Reference: MINT Proposal Section 4, Phase A
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class EditorTrainer:
    """
    Trainer for mechanistic editors (Phase A).

    Trains editors to reconstruct with-tool states from no-tool states
    using real counterfactual pairs from τ-bench (not synthetic).

    Implements the counterfactual supervision loss from Proposal Section 4, Phase A.
    """

    def __init__(
        self,
        editor: nn.Module,
        sae_loader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr: float = 1e-4,
        lambda_small_edit: float = 5e-4,
        epsilon_lipschitz: float = 0.5,
        enforce_lipschitz: bool = True,
        device: str = "cuda",
    ):
        """
        Initialize editor trainer.

        Args:
            editor: MechanisticEditor instance
            sae_loader: SAELoader instance
            optimizer: Optional optimizer (created if None)
            lr: Learning rate
            lambda_small_edit: Weight for small-edit penalty (η in proposal)
            epsilon_lipschitz: E-Lipschitz constraint (max edit norm per layer)
            enforce_lipschitz: Whether to enforce E-Lipschitz constraints
            device: Device to train on
        """
        self.editor = editor.to(device)
        self.sae_loader = sae_loader
        self.lambda_small_edit = lambda_small_edit
        self.epsilon_lipschitz = epsilon_lipschitz
        self.enforce_lipschitz = enforce_lipschitz
        self.device = device

        # Create optimizer if not provided
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.editor.parameters(),
                lr=lr,
                weight_decay=0.01,
            )
        else:
            self.optimizer = optimizer

    def train_step(
        self,
        hidden_no_tool: Dict[str, torch.Tensor],
        hidden_with_tool: Dict[str, torch.Tensor],
        ctx_vec: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Single training step with counterfactual supervision.

        Implements Proposal Section 4, Phase A:
        Minimize ||Π_edit(E_u*(H_t) - H_t^(+u*))||²

        where:
        - H_t = hidden_no_tool (from no-tool trajectory)
        - H_t^(+u*) = hidden_with_tool (from with-tool trajectory with real tool output)
        - Π_edit = projection onto masked features m_u*^(ℓ)

        Args:
            hidden_no_tool: No-tool hidden states H_t from τ-bench
            hidden_with_tool: With-tool hidden states H_t^(+u*) from τ-bench
            ctx_vec: Optional context vector for learned gates

        Returns:
            Dictionary of losses
        """
        self.optimizer.zero_grad(set_to_none=True)

        # Apply editor: E_u*(H_t) → edited states
        edited = self.editor(hidden_no_tool, ctx_vec)

        # Compute counterfactual supervision loss on masked features
        # This is the L_counterfactual from Proposal Section 4, Phase A
        total_loss = 0.0
        recon_loss = 0.0

        for layer_key in edited.keys():
            if layer_key not in hidden_with_tool:
                continue

            # Get SAE for this layer
            sae = self.sae_loader.get_sae(layer_key)

            # Encode both edited and target (with-tool) states
            phi_edited = sae.encode(edited[layer_key])  # φ̃ = encode(E_u*(H_t))
            phi_target = sae.encode(hidden_with_tool[layer_key])  # φ_target = encode(H_t^(+u*))

            # Get mask m_u*^(ℓ) for this layer
            mask = self.editor._get_mask(layer_key)

            # Compute loss only on masked features: Π_edit(φ̃ - φ_target)
            # This ensures we only supervise on tool-relevant features
            delta = phi_edited - phi_target
            masked_delta = delta * mask.float()  # Apply mask (projection)

            # L2 loss on masked features
            layer_loss = (masked_delta ** 2).sum(dim=-1).mean()
            recon_loss += layer_loss

        # Small-edit penalty (η·L_small-edit from proposal)
        # Encourages editors to make minimal changes
        edit_norm = self.editor.get_edit_norm(hidden_no_tool, ctx_vec)
        small_edit_loss = self.lambda_small_edit * edit_norm

        # E-Lipschitz constraint penalty (from Proposal Section 3.2)
        # Penalize edits that exceed epsilon_lipschitz per layer
        lipschitz_loss = 0.0
        if self.enforce_lipschitz:
            for layer_key in edited.keys():
                if layer_key not in hidden_no_tool:
                    continue

                # Compute edit magnitude: ||edited - original||
                edit_delta = edited[layer_key] - hidden_no_tool[layer_key]
                edit_magnitude = torch.norm(edit_delta, p=2, dim=-1).mean()

                # Penalize if exceeds epsilon
                lipschitz_violation = torch.relu(edit_magnitude - self.epsilon_lipschitz)
                lipschitz_loss += lipschitz_violation

        # Total loss: L_counterfactual + η·L_small-edit + L_lipschitz
        total_loss = recon_loss + small_edit_loss + lipschitz_loss

        # Backward pass
        total_loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.editor.parameters(), max_norm=1.0)

        # Update parameters
        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "small_edit_loss": small_edit_loss.item(),
            "lipschitz_loss": lipschitz_loss.item() if isinstance(lipschitz_loss, torch.Tensor) else lipschitz_loss,
        }

    def train_epoch(
        self,
        dataloader,
        epoch: int,
    ) -> Dict[str, float]:
        """
        Train for one epoch on real τ-bench counterfactual pairs.

        The dataloader should yield batches of CounterfactualPair objects
        collected from τ-bench (not synthetic data).

        Args:
            dataloader: DataLoader yielding real τ-bench counterfactual pairs
                       Each batch contains (hidden_no_tool, hidden_with_tool, ctx_vec)
            epoch: Current epoch number

        Returns:
            Dictionary of average losses
        """
        self.editor.train()

        total_losses = {
            "total_loss": 0.0,
            "recon_loss": 0.0,
            "small_edit_loss": 0.0,
        }
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            hidden_no_tool = batch["hidden_no_tool"]
            hidden_with_tool = batch["hidden_with_tool"]
            ctx_vec = batch.get("ctx_vec", None)

            # Move to device
            hidden_no_tool = {k: v.to(self.device) for k, v in hidden_no_tool.items()}
            hidden_with_tool = {k: v.to(self.device) for k, v in hidden_with_tool.items()}
            if ctx_vec is not None:
                ctx_vec = ctx_vec.to(self.device)

            # Train step with counterfactual supervision
            losses = self.train_step(hidden_no_tool, hidden_with_tool, ctx_vec)

            # Accumulate
            for key, value in losses.items():
                total_losses[key] += value
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({k: f"{v:.4f}" for k, v in losses.items()})

        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}

        logger.info(f"Epoch {epoch} completed. Avg losses: {avg_losses}")

        return avg_losses

    def save_checkpoint(self, path: str):
        """Save checkpoint."""
        torch.save({
            "editor_state_dict": self.editor.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.editor.load_state_dict(checkpoint["editor_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(f"Loaded checkpoint from {path}")

