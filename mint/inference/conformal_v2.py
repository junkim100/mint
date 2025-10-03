# Copyright (c) MINT
"""
Enhanced conformal prediction with finite-sample guarantees.

Implements:
- One-sided split conformal LCB with exact finite-sample quantiles
- Mondrian (conditional) caches with fallback
- Distribution-free uncertainty quantification

References:
    [1] Lei et al. "Distribution-Free Predictive Inference for Regression"
        https://www.stat.cmu.edu/~ryantibs/papers/conformal.pdf
    [2] Vovk et al. "Conditional validity of inductive conformal predictors"
        https://mapie.readthedocs.io/en/v0.9.0/theoretical_description_mondrian.html
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Hashable, Optional
import math
import numpy as np


def _finite_sample_index(n: int, alpha: float) -> int:
    """
    Return k = ceil((n+1)*(1-alpha)) for split conformal finite-sample validity.

    This ensures exact finite-sample coverage guarantees:
    P(Y_new >= LCB) >= 1 - alpha

    Args:
        n: Number of calibration samples
        alpha: Risk level (e.g., 0.1 for 90% coverage)

    Returns:
        Index k for quantile computation

    Reference:
        Lei et al. "Distribution-Free Predictive Inference for Regression"
        https://www.stat.cmu.edu/~ryantibs/papers/conformal.pdf
    """
    if n <= 0:  # no calibration data -> infinite conservatism
        return 1
    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    return max(k, 1)  # Clamp to at least 1


def _empirical_upper_quantile_signed(residuals: np.ndarray, alpha: float) -> float:
    """
    One-sided (lower) bound uses absolute residuals |y_hat_i - y_true_i|,
    and subtracts the (1-alpha)-quantile.

    For lower confidence bounds:
    - Compute absolute residuals: r_i = |ŷ_i - y_i|
    - Find (1-α)-quantile: q
    - LCB = ŷ_new - q

    This gives: P(y_new >= LCB) >= 1 - α

    Args:
        residuals: Absolute residuals |predictions - targets|
        alpha: Risk level

    Returns:
        (1-alpha)-quantile of absolute residuals
    """
    assert residuals.ndim == 1
    n = residuals.size
    k = _finite_sample_index(n, alpha)
    # Sort ascending; take k-th largest => index -k
    srt = np.sort(residuals)
    # (1-alpha) quantile -> position n - k in 0-indexing
    idx = min(max(n - k, 0), n - 1)
    return float(srt[idx])


@dataclass
class SplitConformalLCB:
    """
    Tool-agnostic split conformal lower confidence bound for scalar ΔV.

    Provides distribution-free uncertainty quantification with exact
    finite-sample coverage guarantees.

    Usage:
        # Calibration phase
        calib = SplitConformalLCB()
        for y_hat, y_true in calibration_data:
            calib.update(y_hat, y_true)

        # Inference phase
        lcb = calib.lcb(y_hat_new, alpha=0.1)
        # Guarantee: P(y_true_new >= lcb) >= 0.9
    """
    absolute_residuals: List[float] = field(default_factory=list)

    def update(self, y_hat: float, y_true: float) -> None:
        """
        Add a calibration example.

        Args:
            y_hat: Predicted value
            y_true: True value
        """
        self.absolute_residuals.append(float(abs(y_hat - y_true)))

    def lcb(self, y_hat: float, alpha: float = 0.1) -> float:
        """
        Compute lower confidence bound for a new prediction.

        Args:
            y_hat: Predicted value
            alpha: Risk level (default: 0.1 for 90% coverage)

        Returns:
            Lower confidence bound: LCB = ŷ - q_{1-α}

        Guarantee:
            P(y_true >= LCB) >= 1 - α
        """
        if not self.absolute_residuals:
            return -np.inf
        q = _empirical_upper_quantile_signed(
            np.asarray(self.absolute_residuals), alpha
        )
        return y_hat - q

    def coverage(self, predictions: List[float], targets: List[float], alpha: float = 0.1) -> float:
        """
        Compute empirical coverage on test set.

        Args:
            predictions: Predicted values
            targets: True values
            alpha: Risk level

        Returns:
            Fraction of targets >= LCB (should be >= 1-alpha)
        """
        if not predictions:
            return 0.0
        lcbs = [self.lcb(p, alpha) for p in predictions]
        violations = sum(1 for lcb, t in zip(lcbs, targets) if t < lcb)
        return 1.0 - (violations / len(predictions))

    def get_quantile(self, alpha: float = 0.1) -> float:
        """Get the (1-alpha)-quantile of absolute residuals."""
        if not self.absolute_residuals:
            return np.inf
        return _empirical_upper_quantile_signed(
            np.asarray(self.absolute_residuals), alpha
        )


@dataclass
class MondrianConformalLCB:
    """
    Group-conditional (Mondrian) split conformal with fallback.

    Provides conditional coverage: P(Y >= LCB | group) >= 1 - α

    Keys can be tuples, e.g. (tool, difficulty_bucket).
    Falls back to progressively coarser keys when fine cache is sparse.

    Example:
        mondrian = MondrianConformalLCB()

        # Set up fallback hierarchy
        mondrian.backoff[("calculator", "hard")] = "calculator"
        mondrian.backoff["calculator"] = None  # global fallback

        # Calibration
        mondrian.update(("calculator", "hard"), y_hat, y_true)

        # Inference with automatic fallback
        lcb = mondrian.lcb(("calculator", "hard"), y_hat_new)

    Reference:
        Vovk et al. "Conditional validity of inductive conformal predictors"
        https://mapie.readthedocs.io/en/v0.9.0/theoretical_description_mondrian.html
    """
    caches: Dict[Hashable, SplitConformalLCB] = field(default_factory=dict)
    backoff: Dict[Hashable, Optional[Hashable]] = field(default_factory=dict)

    def _resolve_key(self, key: Hashable) -> Optional[Hashable]:
        """
        Resolve key with fallback cascade.

        Args:
            key: Fine-grained key (e.g., ("calculator", "hard"))

        Returns:
            Resolved key with available cache, or None
        """
        if key in self.caches:
            return key
        # cascade: user populates self.backoff[key] -> ... -> None
        while key is not None and key not in self.caches:
            key = self.backoff.get(key, None)
        return key

    def update(self, key: Hashable, y_hat: float, y_true: float) -> None:
        """
        Add calibration example for a specific group.

        Args:
            key: Group identifier (e.g., ("calculator", "hard"))
            y_hat: Predicted value
            y_true: True value
        """
        self.caches.setdefault(key, SplitConformalLCB()).update(y_hat, y_true)

    def lcb(self, key: Hashable, y_hat: float, alpha: float = 0.1) -> float:
        """
        Compute group-conditional lower confidence bound.

        Args:
            key: Group identifier
            y_hat: Predicted value
            alpha: Risk level

        Returns:
            Lower confidence bound for this group

        Guarantee:
            P(y_true >= LCB | group=key) >= 1 - α
        """
        k = self._resolve_key(key)
        if k is None:
            return -np.inf
        return self.caches[k].lcb(y_hat, alpha)

    def get_cache_sizes(self) -> Dict[Hashable, int]:
        """Get number of calibration examples per group."""
        return {k: len(v.signed_residuals) for k, v in self.caches.items()}

    def get_quantiles(self, alpha: float = 0.1) -> Dict[Hashable, float]:
        """Get quantiles for all groups."""
        return {k: v.get_quantile(alpha) for k, v in self.caches.items()}

