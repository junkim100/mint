"""
Online/Sequential Conformal Prediction for MINT.

Implements Proposal Appendix B (Online / Sequential Variant):
- Prequential residuals for non-exchangeable data
- Nonconformity martingales for trajectory-level guarantees
- Anytime validity under distribution shift

Reference: MINT Proposal Section 5, Appendix B
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class OnlineConformalPredictor:
    """
    Online conformal predictor with martingale-based guarantees.
    
    Implements the proposal's online/sequential conformal:
    "Use prequential residuals r_t = ΔV_t - ΔV̂_t with nonconformity 
    martingales M_t and stopping-time bounds to maintain a 
    trajectory-level error budget"
    
    Reference: MINT Proposal Appendix B
    """
    
    def __init__(
        self,
        alpha: float = 0.1,
        window_size: int = 100,
        lambda_param: float = 0.5,
    ):
        """
        Initialize online conformal predictor.
        
        Args:
            alpha: Miscoverage rate (e.g., 0.1 for 90% coverage)
            window_size: Size of sliding window for quantile estimation
            lambda_param: Parameter for martingale construction
        """
        self.alpha = alpha
        self.window_size = window_size
        self.lambda_param = lambda_param
        
        # Online statistics
        self.residuals: List[float] = []
        self.predictions: List[float] = []
        self.targets: List[float] = []
        self.timestep = 0
        
        # Martingale tracking
        self.martingale_value = 1.0
        self.wealth_history: List[float] = []
        
        # Current quantile estimate
        self.quantile = 0.0
    
    def update(
        self,
        prediction: float,
        target: float,
    ):
        """
        Update online conformal predictor with new observation.
        
        This implements the prequential (predictive sequential) approach:
        1. Make prediction with current quantile
        2. Observe true value
        3. Compute residual
        4. Update quantile and martingale
        
        Args:
            prediction: Predicted ΔV value
            target: True ΔV value
        """
        # Compute residual
        residual = abs(target - prediction)
        
        # Store
        self.residuals.append(residual)
        self.predictions.append(prediction)
        self.targets.append(target)
        self.timestep += 1
        
        # Update quantile using sliding window
        window_residuals = self.residuals[-self.window_size:]
        if len(window_residuals) > 0:
            # Compute (1-alpha) quantile
            n = len(window_residuals)
            q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
            q_level = min(q_level, 1.0)
            self.quantile = float(np.quantile(window_residuals, q_level))
        
        # Update martingale for trajectory-level guarantees
        self._update_martingale(residual)
        
        logger.debug(
            f"Online conformal update t={self.timestep}: "
            f"residual={residual:.4f}, quantile={self.quantile:.4f}, "
            f"martingale={self.martingale_value:.4f}"
        )
    
    def _update_martingale(self, residual: float):
        """
        Update nonconformity martingale.
        
        The martingale M_t tracks whether we're violating coverage.
        If M_t grows large, we're making too many errors.
        
        Args:
            residual: Current residual
        """
        # Betting function: bet on whether residual > quantile
        # This is a simplified martingale construction
        if residual > self.quantile:
            # Violation: multiply by (1 + λ)
            self.martingale_value *= (1 + self.lambda_param)
        else:
            # No violation: multiply by (1 - λ·α)
            self.martingale_value *= (1 - self.lambda_param * self.alpha)
        
        self.wealth_history.append(self.martingale_value)
    
    def get_lcb(
        self,
        prediction: float,
    ) -> float:
        """
        Get lower confidence bound for a new prediction.
        
        Args:
            prediction: Predicted ΔV value
            
        Returns:
            Lower confidence bound
        """
        return prediction - self.quantile
    
    def get_ucb(
        self,
        prediction: float,
    ) -> float:
        """
        Get upper confidence bound for a new prediction.
        
        Args:
            prediction: Predicted ΔV value
            
        Returns:
            Upper confidence bound
        """
        return prediction + self.quantile
    
    def check_violation(
        self,
        threshold: float = 10.0,
    ) -> bool:
        """
        Check if martingale indicates coverage violation.
        
        If martingale value exceeds threshold, we're likely violating
        the coverage guarantee and should recalibrate.
        
        Args:
            threshold: Martingale threshold for violation
            
        Returns:
            True if violation detected
        """
        return self.martingale_value > threshold
    
    def reset_martingale(self):
        """Reset martingale to 1.0 (e.g., after recalibration)."""
        self.martingale_value = 1.0
        logger.info("Reset martingale to 1.0")
    
    def get_coverage(self) -> float:
        """
        Compute empirical coverage on observed data.
        
        Returns:
            Coverage rate (fraction of times true value in prediction interval)
        """
        if len(self.residuals) == 0:
            return 0.0
        
        # Count how many times |target - prediction| <= quantile
        covered = sum(1 for r in self.residuals if r <= self.quantile)
        return covered / len(self.residuals)
    
    def save(self, path: str):
        """Save online conformal state."""
        state = {
            'alpha': self.alpha,
            'window_size': self.window_size,
            'lambda_param': self.lambda_param,
            'residuals': self.residuals,
            'predictions': self.predictions,
            'targets': self.targets,
            'timestep': self.timestep,
            'martingale_value': self.martingale_value,
            'wealth_history': self.wealth_history,
            'quantile': self.quantile,
        }
        torch.save(state, path)
        logger.info(f"Saved online conformal state to {path}")
    
    @staticmethod
    def load(path: str) -> "OnlineConformalPredictor":
        """Load online conformal state."""
        state = torch.load(path, weights_only=False)
        
        predictor = OnlineConformalPredictor(
            alpha=state['alpha'],
            window_size=state['window_size'],
            lambda_param=state['lambda_param'],
        )
        
        predictor.residuals = state['residuals']
        predictor.predictions = state['predictions']
        predictor.targets = state['targets']
        predictor.timestep = state['timestep']
        predictor.martingale_value = state['martingale_value']
        predictor.wealth_history = state['wealth_history']
        predictor.quantile = state['quantile']
        
        logger.info(f"Loaded online conformal state from {path}")
        return predictor


class AdaptiveConformalPredictor:
    """
    Adaptive conformal predictor that switches between split and online.
    
    Uses split conformal when data is exchangeable, switches to online
    when distribution shift is detected.
    """
    
    def __init__(
        self,
        alpha: float = 0.1,
        shift_threshold: float = 10.0,
        window_size: int = 100,
    ):
        """
        Initialize adaptive conformal predictor.
        
        Args:
            alpha: Miscoverage rate
            shift_threshold: Martingale threshold for detecting shift
            window_size: Window size for online mode
        """
        self.alpha = alpha
        self.shift_threshold = shift_threshold
        
        # Split conformal (initial mode)
        self.split_quantile: Optional[float] = None
        self.is_calibrated = False
        
        # Online conformal (fallback mode)
        self.online_predictor = OnlineConformalPredictor(
            alpha=alpha,
            window_size=window_size,
        )
        
        # Mode tracking
        self.mode = "split"  # "split" or "online"
    
    def calibrate(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ):
        """
        Calibrate using split conformal.
        
        Args:
            predictions: Predicted values [N]
            targets: True values [N]
        """
        residuals = np.abs(predictions - targets)
        
        n = len(residuals)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)
        
        self.split_quantile = float(np.quantile(residuals, q_level))
        self.is_calibrated = True
        self.mode = "split"
        
        logger.info(
            f"Calibrated with split conformal: "
            f"quantile={self.split_quantile:.4f} (n={n})"
        )
    
    def update(
        self,
        prediction: float,
        target: float,
    ):
        """
        Update with new observation (online mode).
        
        Args:
            prediction: Predicted value
            target: True value
        """
        # Always update online predictor
        self.online_predictor.update(prediction, target)
        
        # Check for distribution shift
        if self.mode == "split":
            if self.online_predictor.check_violation(self.shift_threshold):
                logger.warning(
                    f"Distribution shift detected! "
                    f"Switching to online conformal mode. "
                    f"Martingale: {self.online_predictor.martingale_value:.2f}"
                )
                self.mode = "online"
    
    def get_lcb(self, prediction: float) -> float:
        """Get lower confidence bound."""
        if self.mode == "split" and self.is_calibrated:
            return prediction - self.split_quantile
        else:
            return self.online_predictor.get_lcb(prediction)
    
    def get_ucb(self, prediction: float) -> float:
        """Get upper confidence bound."""
        if self.mode == "split" and self.is_calibrated:
            return prediction + self.split_quantile
        else:
            return self.online_predictor.get_ucb(prediction)
    
    def get_mode(self) -> str:
        """Get current mode (split or online)."""
        return self.mode

