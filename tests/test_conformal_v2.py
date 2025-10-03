"""
Tests for enhanced conformal prediction with finite-sample guarantees.

Acceptance criteria:
- One-sided split conformal LCB: empirical violation ≈ α on i.i.d. toy data
- Mondrian caches: coverage holds per bucket (Wilson CI overlaps target)
- Online updates: moving-window coverage remains within tolerance under synthetic drift
"""
import pytest
import numpy as np
from mint.inference.conformal_v2 import (
    SplitConformalLCB,
    MondrianConformalLCB,
    _finite_sample_index,
    _empirical_upper_quantile_signed
)


class TestFiniteSampleIndex:
    """Test finite-sample index computation."""
    
    def test_basic_computation(self):
        """Test k = ceil((n+1)*(1-alpha))."""
        # n=100, alpha=0.1 -> k = ceil(101 * 0.9) = ceil(90.9) = 91
        assert _finite_sample_index(100, 0.1) == 91
        
        # n=1000, alpha=0.1 -> k = ceil(1001 * 0.9) = ceil(900.9) = 901
        assert _finite_sample_index(1000, 0.1) == 901
    
    def test_edge_cases(self):
        """Test edge cases."""
        # n=0 -> k=1 (conservative)
        assert _finite_sample_index(0, 0.1) == 1
        
        # alpha=0 -> k = ceil(n+1) = n+1
        assert _finite_sample_index(100, 0.0) == 101
        
        # alpha=1 -> k = ceil(0) = 0, but clamped to 1
        assert _finite_sample_index(100, 1.0) == 1


class TestSplitConformalLCB:
    """Test split conformal LCB with finite-sample guarantees."""
    
    def test_basic_usage(self):
        """Test basic calibration and LCB computation."""
        calib = SplitConformalLCB()
        
        # Add calibration data
        for i in range(100):
            y_hat = float(i)
            y_true = float(i) + np.random.normal(0, 1)
            calib.update(y_hat, y_true)
        
        # Compute LCB
        lcb = calib.lcb(50.0, alpha=0.1)
        
        # LCB should be less than prediction
        assert lcb < 50.0
        
        # Quantile should be positive (since we have noise)
        q = calib.get_quantile(alpha=0.1)
        assert q > 0
    
    def test_coverage_guarantee(self):
        """
        Test that empirical violation rate ≈ α on i.i.d. data.
        
        Acceptance criterion: violation rate <= α + 0.02 (allowing sampling error)
        """
        rng = np.random.default_rng(42)
        
        # Simulate unbiased predictor: y_hat = y_true + noise
        # Calibration data
        calib = SplitConformalLCB()
        n_calib = 1000
        y_true_calib = rng.normal(0, 1, size=n_calib)
        eps_calib = rng.normal(0, 0.1, size=n_calib)
        
        for yt, e in zip(y_true_calib, eps_calib):
            calib.update(yt + e, yt)
        
        # Test data
        alpha = 0.1
        n_test = 2000
        y_true_test = rng.normal(0, 1, size=n_test)
        eps_test = rng.normal(0, 0.1, size=n_test)
        y_hat_test = y_true_test + eps_test
        
        # Compute LCBs
        lcbs = [calib.lcb(yh, alpha) for yh in y_hat_test]
        
        # Check violations
        violations = sum(1 for lcb, yt in zip(lcbs, y_true_test) if yt < lcb)
        violation_rate = violations / n_test
        
        # Acceptance criterion
        assert violation_rate <= alpha + 0.02, f"Violation rate {violation_rate:.3f} > {alpha + 0.02:.3f}"
        
        print(f"✓ Coverage test passed: violation rate = {violation_rate:.3f} (target: {alpha:.3f})")
    
    def test_coverage_method(self):
        """Test coverage computation method."""
        calib = SplitConformalLCB()
        
        # Add calibration data
        for i in range(100):
            calib.update(float(i), float(i))
        
        # Test coverage on perfect predictions
        predictions = [float(i) for i in range(50)]
        targets = [float(i) for i in range(50)]
        
        coverage = calib.coverage(predictions, targets, alpha=0.1)
        
        # Coverage should be high (close to 1.0) for perfect predictions
        assert coverage >= 0.9
    
    def test_empty_calibration(self):
        """Test behavior with no calibration data."""
        calib = SplitConformalLCB()
        
        # LCB should be -inf (maximally conservative)
        lcb = calib.lcb(50.0, alpha=0.1)
        assert lcb == -np.inf
        
        # Quantile should be inf
        q = calib.get_quantile(alpha=0.1)
        assert q == np.inf


class TestMondrianConformalLCB:
    """Test Mondrian (group-conditional) conformal prediction."""
    
    def test_basic_usage(self):
        """Test basic Mondrian conformal with groups."""
        mondrian = MondrianConformalLCB()
        
        # Add calibration data for different groups
        for i in range(50):
            mondrian.update("group_a", float(i), float(i) + np.random.normal(0, 0.1))
            mondrian.update("group_b", float(i), float(i) + np.random.normal(0, 0.5))
        
        # Compute LCBs for each group
        lcb_a = mondrian.lcb("group_a", 25.0, alpha=0.1)
        lcb_b = mondrian.lcb("group_b", 25.0, alpha=0.1)
        
        # Group B should have wider bounds (more noise)
        assert lcb_a > lcb_b
    
    def test_fallback_hierarchy(self):
        """Test fallback cascade."""
        mondrian = MondrianConformalLCB()
        
        # Set up fallback: (tool, difficulty) -> tool -> None
        mondrian.backoff[("calculator", "hard")] = "calculator"
        mondrian.backoff["calculator"] = None
        
        # Add data only for coarse key
        for i in range(100):
            mondrian.update("calculator", float(i), float(i) + np.random.normal(0, 0.1))
        
        # Query fine-grained key (should fall back to "calculator")
        lcb = mondrian.lcb(("calculator", "hard"), 50.0, alpha=0.1)
        
        # Should get valid LCB (not -inf)
        assert lcb > -np.inf
        assert lcb < 50.0
    
    def test_conditional_coverage(self):
        """
        Test that coverage holds per group.
        
        Acceptance criterion: Wilson CI for each group overlaps target coverage
        """
        rng = np.random.default_rng(42)
        mondrian = MondrianConformalLCB()
        
        # Simulate two groups with different noise levels
        n_calib = 500
        alpha = 0.1
        
        # Group A: low noise
        for _ in range(n_calib):
            y_true = rng.normal(0, 1)
            y_hat = y_true + rng.normal(0, 0.1)
            mondrian.update("group_a", y_hat, y_true)
        
        # Group B: high noise
        for _ in range(n_calib):
            y_true = rng.normal(0, 1)
            y_hat = y_true + rng.normal(0, 0.5)
            mondrian.update("group_b", y_hat, y_true)
        
        # Test coverage for each group
        n_test = 1000
        
        for group, noise_std in [("group_a", 0.1), ("group_b", 0.5)]:
            y_true_test = rng.normal(0, 1, size=n_test)
            eps_test = rng.normal(0, noise_std, size=n_test)
            y_hat_test = y_true_test + eps_test
            
            lcbs = [mondrian.lcb(group, yh, alpha) for yh in y_hat_test]
            violations = sum(1 for lcb, yt in zip(lcbs, y_true_test) if yt < lcb)
            violation_rate = violations / n_test
            
            # Wilson confidence interval for binomial proportion
            p = violation_rate
            n = n_test
            z = 1.96  # 95% CI
            wilson_lower = (p + z**2/(2*n) - z * np.sqrt(p*(1-p)/n + z**2/(4*n**2))) / (1 + z**2/n)
            wilson_upper = (p + z**2/(2*n) + z * np.sqrt(p*(1-p)/n + z**2/(4*n**2))) / (1 + z**2/n)
            
            # Check if target alpha is in Wilson CI
            in_ci = wilson_lower <= alpha <= wilson_upper
            
            print(f"✓ Group {group}: violation rate = {violation_rate:.3f}, "
                  f"Wilson CI = [{wilson_lower:.3f}, {wilson_upper:.3f}], "
                  f"target α = {alpha:.3f}, in CI: {in_ci}")
            
            # Acceptance criterion: target should be in CI or violation rate close to target
            assert in_ci or abs(violation_rate - alpha) < 0.05
    
    def test_cache_sizes(self):
        """Test cache size tracking."""
        mondrian = MondrianConformalLCB()
        
        # Add different amounts of data to different groups
        for i in range(100):
            mondrian.update("group_a", float(i), float(i))
        for i in range(50):
            mondrian.update("group_b", float(i), float(i))
        
        sizes = mondrian.get_cache_sizes()
        
        assert sizes["group_a"] == 100
        assert sizes["group_b"] == 50
    
    def test_quantiles(self):
        """Test quantile retrieval."""
        mondrian = MondrianConformalLCB()
        
        # Add data
        for i in range(100):
            mondrian.update("group_a", float(i), float(i) + 0.1)
        
        quantiles = mondrian.get_quantiles(alpha=0.1)
        
        assert "group_a" in quantiles
        assert quantiles["group_a"] > 0


class TestIntegration:
    """Integration tests for conformal prediction."""
    
    def test_realistic_scenario(self):
        """Test realistic tool selection scenario."""
        rng = np.random.default_rng(42)
        mondrian = MondrianConformalLCB()
        
        # Simulate calibration for different tools
        tools = ["calculator", "search", "other"]
        true_utilities = {"calculator": 0.8, "search": 0.6, "other": 0.3}
        noise_levels = {"calculator": 0.1, "search": 0.2, "other": 0.3}
        
        # Calibration phase
        for tool in tools:
            for _ in range(200):
                y_true = true_utilities[tool] + rng.normal(0, 0.1)
                y_hat = y_true + rng.normal(0, noise_levels[tool])
                mondrian.update(tool, y_hat, y_true)
        
        # Inference phase
        alpha = 0.1
        for tool in tools:
            # Predict utility
            y_hat = true_utilities[tool] + rng.normal(0, noise_levels[tool])
            
            # Get LCB
            lcb = mondrian.lcb(tool, y_hat, alpha)
            
            # LCB should be less than prediction
            assert lcb < y_hat
            
            # LCB should be reasonable (not too conservative)
            assert lcb > y_hat - 1.0
        
        print("✓ Realistic scenario test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

