"""Conformal prediction for risk-calibrated decisions."""

import torch
import numpy as np
from typing import List, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)


class ConformalCalibrator:
    """
    Conformal calibrator for computing distribution-free lower confidence bounds.
    
    Uses split conformal prediction to provide valid LCBs on predicted utility gains.
    """
    
    def __init__(
        self,
        alpha: float = 0.1,
        method: str = "split",
    ):
        """
        Initialize conformal calibrator.
        
        Args:
            alpha: Miscoverage level (1-alpha confidence)
            method: Calibration method ('split', 'online', 'adaptive')
        """
        self.alpha = alpha
        self.method = method
        self.residuals: List[float] = []
        self.quantile: Optional[float] = None
        self.is_finalized = False
    
    def add_residuals(
        self,
        predictions: Union[torch.Tensor, np.ndarray, List[float]],
        targets: Union[torch.Tensor, np.ndarray, List[float]],
    ):
        """
        Add calibration residuals.
        
        Args:
            predictions: Predicted ΔV values
            targets: True ΔV values
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        predictions = np.asarray(predictions).flatten()
        targets = np.asarray(targets).flatten()
        
        # Compute residuals: r = target - prediction
        residuals = targets - predictions
        self.residuals.extend(residuals.tolist())
    
    def finalize(self):
        """
        Finalize calibration by computing quantile.
        """
        if len(self.residuals) == 0:
            logger.warning("No residuals added, using default quantile of 0")
            self.quantile = 0.0
        else:
            # Compute (1-alpha) quantile of residuals
            self.quantile = float(np.quantile(self.residuals, 1 - self.alpha))
            logger.info(
                f"Finalized calibrator with {len(self.residuals)} residuals. "
                f"Quantile (1-{self.alpha}): {self.quantile:.4f}"
            )
        
        self.is_finalized = True
    
    def lcb(
        self,
        predictions: Union[torch.Tensor, np.ndarray, float],
    ) -> Union[torch.Tensor, np.ndarray, float]:
        """
        Compute lower confidence bound.
        
        Args:
            predictions: Predicted ΔV values
            
        Returns:
            Lower confidence bounds: LCB = prediction - quantile
        """
        if not self.is_finalized:
            raise ValueError("Calibrator not finalized. Call finalize() first.")
        
        if isinstance(predictions, torch.Tensor):
            return predictions - self.quantile
        elif isinstance(predictions, np.ndarray):
            return predictions - self.quantile
        else:
            return predictions - self.quantile
    
    def ucb(
        self,
        predictions: Union[torch.Tensor, np.ndarray, float],
    ) -> Union[torch.Tensor, np.ndarray, float]:
        """
        Compute upper confidence bound (for completeness).
        
        Args:
            predictions: Predicted ΔV values
            
        Returns:
            Upper confidence bounds: UCB = prediction + quantile
        """
        if not self.is_finalized:
            raise ValueError("Calibrator not finalized. Call finalize() first.")
        
        if isinstance(predictions, torch.Tensor):
            return predictions + self.quantile
        elif isinstance(predictions, np.ndarray):
            return predictions + self.quantile
        else:
            return predictions + self.quantile
    
    def get_stats(self) -> Dict[str, float]:
        """Get calibration statistics."""
        if not self.is_finalized:
            return {"status": "not_finalized"}
        
        residuals = np.array(self.residuals)
        return {
            "num_samples": len(residuals),
            "quantile": self.quantile,
            "alpha": self.alpha,
            "mean_residual": float(np.mean(residuals)),
            "std_residual": float(np.std(residuals)),
            "min_residual": float(np.min(residuals)),
            "max_residual": float(np.max(residuals)),
        }


class RiskBudget:
    """
    Risk budget tracker for trajectory-level risk control.
    
    Tracks cumulative risk across multiple tool calls in a trajectory.
    """
    
    def __init__(
        self,
        total_alpha: float = 0.1,
        per_tool_alpha: Optional[Dict[str, float]] = None,
        trajectory_length: int = 10,
        allocation: str = "uniform",
    ):
        """
        Initialize risk budget.
        
        Args:
            total_alpha: Total risk budget for trajectory
            per_tool_alpha: Optional per-tool risk budgets
            trajectory_length: Expected trajectory length
            allocation: Budget allocation strategy ('uniform', 'adaptive')
        """
        self.total_alpha = total_alpha
        self.per_tool_alpha = per_tool_alpha or {}
        self.trajectory_length = trajectory_length
        self.allocation = allocation
        
        # Track consumed budget
        self.consumed_alpha = 0.0
        self.tool_calls: List[str] = []
        self.step = 0
    
    def allows(self, tool_name: str) -> bool:
        """
        Check if calling a tool is allowed under the budget.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            True if tool call is allowed
        """
        if tool_name == "NoTool":
            return True
        
        # Check total budget
        if self.allocation == "uniform":
            step_budget = self.total_alpha / self.trajectory_length
        else:
            # Adaptive: more budget early on
            remaining_steps = max(1, self.trajectory_length - self.step)
            step_budget = (self.total_alpha - self.consumed_alpha) / remaining_steps
        
        # Check per-tool budget if specified
        if tool_name in self.per_tool_alpha:
            tool_budget = self.per_tool_alpha[tool_name]
            tool_consumed = sum(1 for t in self.tool_calls if t == tool_name) * step_budget
            if tool_consumed >= tool_budget:
                return False
        
        # Check total budget
        return self.consumed_alpha + step_budget <= self.total_alpha
    
    def consume(self, tool_name: str):
        """
        Consume budget for a tool call.
        
        Args:
            tool_name: Name of the tool
        """
        if tool_name == "NoTool":
            self.step += 1
            return
        
        if self.allocation == "uniform":
            step_budget = self.total_alpha / self.trajectory_length
        else:
            remaining_steps = max(1, self.trajectory_length - self.step)
            step_budget = (self.total_alpha - self.consumed_alpha) / remaining_steps
        
        self.consumed_alpha += step_budget
        self.tool_calls.append(tool_name)
        self.step += 1
    
    def reset(self):
        """Reset budget for new trajectory."""
        self.consumed_alpha = 0.0
        self.tool_calls = []
        self.step = 0
    
    def get_stats(self) -> Dict[str, any]:
        """Get budget statistics."""
        return {
            "total_alpha": self.total_alpha,
            "consumed_alpha": self.consumed_alpha,
            "remaining_alpha": self.total_alpha - self.consumed_alpha,
            "num_tool_calls": len(self.tool_calls),
            "step": self.step,
            "tool_calls": self.tool_calls,
        }

