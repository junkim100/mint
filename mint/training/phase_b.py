"""
Phase B: Value head training for utility prediction.

Implements Proposal Section 4, Phase B (Utility Modeling):
    Train g to predict observed return gains ΔV from edited states
    Loss: (ΔV̂_u* - ΔV)² + λ·Cox/CE for success

Reference: MINT Proposal Section 4, Phase B
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List
import logging

# Import the CORRECT value head from mint.core.value_head
# This uses the architecture: ΔV̂_u = g(H̃_t^(u)) - g(H_t)
from mint.core.value_head import ValueHead

logger = logging.getLogger(__name__)


class ValueHeadTrainer:
    """
    Trainer for value heads (Phase B).

    Implements Proposal Section 4, Phase B:
    Train g to predict observed return gains ΔV from edited states.
    Loss: (ΔV̂_u* - ΔV)² where ΔV is measured from real task outcomes.
    """

    def __init__(
        self,
        value_head: ValueHead,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize value head trainer.

        Args:
            value_head: ValueHead model (from mint.core.value_head)
            device: Device to train on
            dtype: Data type for the model (default: bfloat16)
        """
        self.value_head = value_head.to(device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype

    def train(
        self,
        pairs: List[Dict],
        steps: int = 1000,
        batch_size: int = 32,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        lambda_success: float = 0.1,
        use_success_loss: bool = True,
    ):
        """
        Train value head on observed return gains from real counterfactual pairs.

        Implements Proposal Section 4, Phase B:
        - Uses real ΔV from τ-bench task outcomes (not synthetic)
        - Computes ΔV̂_u = g(H̃_t^(u)) - g(H_t) via predict_delta_v()
        - Minimizes (ΔV̂_u - ΔV)² + λ·Cox/CE for success

        Args:
            pairs: List of CounterfactualPair objects from τ-bench
            steps: Number of training steps
            batch_size: Batch size
            lr: Learning rate
            weight_decay: Weight decay
            lambda_success: Weight for success prediction loss
            use_success_loss: Whether to use Cox/CE success prediction
        """
        optimizer = optim.AdamW(
            self.value_head.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        self.value_head.train()

        for step in range(steps):
            # Sample batch
            batch_indices = torch.randint(0, len(pairs), (batch_size,))
            batch = [pairs[i] for i in batch_indices]

            # Get states
            baseline_states = {}
            edited_states = {}

            # Stack states for each layer
            for key in batch[0].hidden_states_no_tool.keys():
                baseline_states[key] = torch.stack([
                    p.hidden_states_no_tool[key] for p in batch
                ]).to(self.device)

                edited_states[key] = torch.stack([
                    p.hidden_states_with_tool[key] for p in batch
                ]).to(self.device)

            # Get target ΔV (observed return gains from real task outcomes)
            target_delta_v = torch.tensor(
                [p.delta_v for p in batch],
                device=self.device,
                dtype=self.dtype,
            )

            # Forward pass: ΔV̂_u = g(H̃_t^(u)) - g(H_t)
            # This uses the CORRECT architecture from Proposal Section 3.3
            pred_delta_v = self.value_head.predict_delta_v(baseline_states, edited_states)

            # Loss: (ΔV̂_u - ΔV)²
            # Ensure shapes match
            pred_delta_v_flat = pred_delta_v.squeeze(-1) if pred_delta_v.dim() > 1 else pred_delta_v
            target_delta_v_flat = target_delta_v.squeeze(-1) if target_delta_v.dim() > 1 else target_delta_v
            mse_loss = nn.functional.mse_loss(pred_delta_v_flat, target_delta_v_flat)

            # Optional: Cox/CE for success prediction (Proposal Section 4, Phase B)
            success_loss = 0.0
            if use_success_loss:
                # Get success labels
                success_labels = torch.tensor(
                    [p.success for p in batch],
                    device=self.device,
                    dtype=torch.float32,
                )

                # Predict success from ΔV (higher ΔV → higher success probability)
                # Use sigmoid to convert ΔV to probability
                success_probs = torch.sigmoid(pred_delta_v).squeeze(-1) if pred_delta_v.dim() > 1 else torch.sigmoid(pred_delta_v)

                # Binary cross-entropy loss (ensure same dtype)
                success_loss = nn.functional.binary_cross_entropy(
                    success_probs, success_labels.to(success_probs.dtype)
                )
                success_loss = lambda_success * success_loss

            # Total loss
            loss = mse_loss + success_loss

            # Backward
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # Log
            if step % 100 == 0 or step == steps - 1:
                log_msg = (
                    f"Step {step}/{steps}: "
                    f"loss={loss.item():.4f}, "
                    f"mse={mse_loss.item():.4f}, "
                    f"mean_pred={pred_delta_v.mean().item():.4f}, "
                    f"mean_target={target_delta_v.mean().item():.4f}"
                )
                if use_success_loss:
                    log_msg += f", success_loss={success_loss.item():.4f}"
                logger.info(log_msg)

    def evaluate(
        self,
        pairs: List[Dict],
        batch_size: int = 32,
    ) -> Dict[str, float]:
        """
        Evaluate value head on observed return gains.

        Args:
            pairs: List of CounterfactualPair objects
            batch_size: Batch size

        Returns:
            Dictionary of metrics (MSE, MAE, etc.)
        """
        self.value_head.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i:i+batch_size]

                # Get states
                baseline_states = {}
                edited_states = {}

                for key in batch[0].hidden_states_no_tool.keys():
                    baseline_states[key] = torch.stack([
                        p.hidden_states_no_tool[key] for p in batch
                    ]).to(self.device)

                    edited_states[key] = torch.stack([
                        p.hidden_states_with_tool[key] for p in batch
                    ]).to(self.device)

                # Get target ΔV (observed return gains)
                target_delta_v = torch.tensor(
                    [p.delta_v for p in batch],
                    device=self.device,
                    dtype=self.dtype,
                )

                # Forward pass: ΔV̂_u = g(H̃_t^(u)) - g(H_t)
                pred_delta_v = self.value_head.predict_delta_v(baseline_states, edited_states)

                all_preds.append(pred_delta_v.cpu())
                all_targets.append(target_delta_v.cpu())

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        # Compute metrics
        mse = nn.functional.mse_loss(all_preds, all_targets).item()
        mae = (all_preds - all_targets).abs().mean().item()

        return {
            "mse": mse,
            "mae": mae,
            "mean_pred": all_preds.mean().item(),
            "mean_target": all_targets.mean().item(),
        }

