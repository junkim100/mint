"""Phase C: Conformal calibration for risk-calibrated predictions."""

import torch
import numpy as np
from typing import Dict, List, Tuple
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
    """Trainer for conformal calibration (Phase C)."""

    def __init__(
        self,
        value_head,
        device: str = "cuda",
    ):
        """
        Initialize conformal calibration trainer.

        Args:
            value_head: Trained value head
            device: Device to use
        """
        self.value_head = value_head.to(device)
        self.device = device

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

