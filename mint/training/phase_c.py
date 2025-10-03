"""
Phase C: Conformal calibration and faithfulness regularization.

Implements Proposal Section 4, Phase C:
- Conformal calibration for risk-calibrated predictions
- Ablation faithfulness regularization
- Contrastive causal InfoNCE

Reference: MINT Proposal Section 4, Phase C
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ConformalCalibrator:
    """
    Conformal prediction calibrator for distribution-free uncertainty quantification.

    Computes lower confidence bounds (LCB) on predicted utility gains
    to enable risk-calibrated tool selection.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        device: str = "cuda",
    ):
        """
        Initialize conformal calibrator.

        Args:
            alpha: Miscoverage rate (e.g., 0.1 for 90% coverage)
            device: Device to use
        """
        self.alpha = alpha
        self.device = device
        self.quantile = None

    def calibrate(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ):
        """
        Calibrate on a calibration set.

        Computes the (1-alpha) quantile of absolute residuals.

        Args:
            predictions: Predicted values [N]
            targets: True values [N]
        """
        # Compute residuals
        residuals = torch.abs(predictions - targets)

        # Compute quantile (convert to float32 for quantile calculation)
        n = len(residuals)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)

        self.quantile = torch.quantile(residuals.float(), q_level).item()

        logger.info(
            f"Calibrated with α={self.alpha}: "
            f"quantile={self.quantile:.4f} "
            f"(n={n})"
        )

    def get_lower_bound(
        self,
        predictions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get lower confidence bound on predictions.

        Args:
            predictions: Predicted values [N]

        Returns:
            Lower confidence bounds [N]
        """
        if self.quantile is None:
            raise ValueError("Calibrator not calibrated. Call calibrate() first.")

        return predictions - self.quantile

    def get_upper_bound(
        self,
        predictions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get upper confidence bound on predictions.

        Args:
            predictions: Predicted values [N]

        Returns:
            Upper confidence bounds [N]
        """
        if self.quantile is None:
            raise ValueError("Calibrator not calibrated. Call calibrate() first.")

        return predictions + self.quantile

    def save(self, path: str):
        """Save calibrator."""
        torch.save({
            "alpha": self.alpha,
            "quantile": self.quantile,
        }, path)
        logger.info(f"Saved calibrator to {path}")

    @staticmethod
    def load(path: str, device: str = "cuda") -> "ConformalCalibrator":
        """Load calibrator."""
        data = torch.load(path, map_location=device, weights_only=False)
        calibrator = ConformalCalibrator(alpha=data["alpha"], device=device)
        calibrator.quantile = data["quantile"]
        logger.info(f"Loaded calibrator from {path}")
        return calibrator


class ConformalCalibrationTrainer:
    """
    Trainer for conformal calibration (Phase C).

    Now includes faithfulness regularization from Proposal Section 4, Phase C.
    """

    def __init__(
        self,
        value_head,
        editor: Optional[any] = None,
        device: str = "cuda",
        use_faithfulness: bool = False,
        lambda_ablation: float = 0.1,
        lambda_contrastive: float = 0.5,
    ):
        """
        Initialize conformal calibration trainer.

        Args:
            value_head: Trained value head
            editor: Optional editor for faithfulness regularization
            device: Device to use
            use_faithfulness: Whether to use faithfulness regularization
            lambda_ablation: Weight for ablation faithfulness loss
            lambda_contrastive: Weight for contrastive causal loss
        """
        self.value_head = value_head.to(device)
        self.editor = editor.to(device) if editor is not None else None
        self.device = device
        self.use_faithfulness = use_faithfulness

        # Faithfulness regularizer (if enabled)
        if use_faithfulness and editor is not None:
            from mint.training.faithfulness import FaithfulnessRegularizer
            self.faithfulness_regularizer = FaithfulnessRegularizer(
                lambda_ablation=lambda_ablation,
                lambda_contrastive=lambda_contrastive,
            )
        else:
            self.faithfulness_regularizer = None

    def calibrate(
        self,
        pairs: List[Dict],
        alpha: float = 0.1,
        batch_size: int = 32,
    ) -> ConformalCalibrator:
        """
        Calibrate value head predictions.

        Args:
            pairs: Calibration pairs
            alpha: Miscoverage rate
            batch_size: Batch size

        Returns:
            Calibrated ConformalCalibrator
        """
        logger.info(f"Calibrating with {len(pairs)} pairs (α={alpha})")

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

                # Get target ΔV
                target_delta_v = torch.tensor(
                    [p.delta_v for p in batch],
                    device=self.device,
                    dtype=torch.bfloat16,
                )

                # Forward pass
                pred_delta_v = self.value_head.predict_delta_v(baseline_states, edited_states)

                all_preds.append(pred_delta_v.cpu())
                all_targets.append(target_delta_v.cpu())

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        # Create and calibrate
        calibrator = ConformalCalibrator(alpha=alpha, device=self.device)
        calibrator.calibrate(all_preds, all_targets)

        # Log statistics
        lcb = calibrator.get_lower_bound(all_preds)
        coverage = (lcb <= all_targets).float().mean().item()

        logger.info(
            f"Calibration complete: "
            f"coverage={coverage:.3f} "
            f"(target={1-alpha:.3f})"
        )

        return calibrator

    def train_with_faithfulness(
        self,
        pairs: List[Dict],
        steps: int = 500,
        batch_size: int = 16,
        lr: float = 1e-5,
    ) -> Dict[str, float]:
        """
        Fine-tune editor and value head with faithfulness regularization.

        Implements Proposal Section 4, Phase C:
        - Ablation faithfulness: Penalize decision invariance
        - Contrastive causal InfoNCE: Encourage ΔV̂_u* > ΔV̂_u≠u*

        Args:
            pairs: Training pairs
            steps: Number of training steps
            batch_size: Batch size
            lr: Learning rate (small for fine-tuning)

        Returns:
            Dictionary of training statistics
        """
        if not self.use_faithfulness or self.faithfulness_regularizer is None:
            logger.warning("Faithfulness regularization not enabled")
            return {}

        if self.editor is None:
            logger.warning("No editor provided for faithfulness training")
            return {}

        logger.info(f"Training with faithfulness regularization for {steps} steps")

        # Create optimizer for fine-tuning
        params = list(self.editor.parameters()) + list(self.value_head.parameters())
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)

        self.editor.train()
        self.value_head.train()

        total_stats = {
            'ablation_faith': 0.0,
            'contrastive': 0.0,
            'total_loss': 0.0,
        }

        for step in range(steps):
            # Sample batch
            batch_indices = torch.randint(0, len(pairs), (batch_size,))
            batch = [pairs[i] for i in batch_indices]

            # Get states
            hidden_states = {}
            for key in batch[0].hidden_states_no_tool.keys():
                hidden_states[key] = torch.stack([
                    p.hidden_states_no_tool[key] for p in batch
                ]).to(self.device)

            # Compute faithfulness loss
            faith_loss, faith_stats = self.faithfulness_regularizer.compute_loss(
                editor=self.editor,
                value_head=self.value_head,
                hidden_states=hidden_states,
                ctx_vec=None,  # Could compute from hidden_states if needed
            )

            # Backward
            optimizer.zero_grad(set_to_none=True)
            faith_loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            # Accumulate stats
            for key, value in faith_stats.items():
                total_stats[key] += value
            total_stats['total_loss'] += faith_loss.item()

            # Log
            if step % 100 == 0 or step == steps - 1:
                logger.info(
                    f"Faithfulness step {step}/{steps}: "
                    f"loss={faith_loss.item():.4f}, "
                    + ", ".join(f"{k}={v:.4f}" for k, v in faith_stats.items())
                )

        # Average stats
        avg_stats = {k: v / steps for k, v in total_stats.items()}

        logger.info(f"Faithfulness training complete. Avg stats: {avg_stats}")

        return avg_stats

