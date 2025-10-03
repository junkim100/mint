#!/usr/bin/env python3
"""
Evaluate decision-quality metrics for MINT.

Metrics:
- Wasted Tool Rate (WTR): Fraction of tool calls where true utility <= 0
- Selective Success@τ: Success rate when LCB >= τ (with coverage)
- Net Utility: Mean utility gain minus costs
- Reliability curves: Calibration of LCB predictions

Usage:
    python scripts/eval_decision_metrics.py \
        --pred_json results/predictions.json \
        --out_json results/metrics.json \
        --out_plots results/plots/
"""
from __future__ import annotations
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def wasted_tool_rate(y_true_delta: np.ndarray, used_tool: np.ndarray) -> float:
    """
    Compute Wasted Tool Rate (WTR).
    
    WTR = fraction of tool calls where true utility <= 0
    
    Args:
        y_true_delta: True utility gains [N]
        used_tool: Binary indicator of tool use [N]
    
    Returns:
        WTR in [0, 1]
    """
    tool_calls = used_tool == 1
    if not tool_calls.any():
        return 0.0
    
    wasted = (y_true_delta[tool_calls] <= 0.0).mean()
    return float(wasted)


def selective_success_at_tau(
    success: np.ndarray,
    lcb: np.ndarray,
    tau: float
) -> Tuple[float, float]:
    """
    Compute Selective Success@τ.
    
    Returns success rate when LCB >= τ, along with coverage (fraction selected).
    
    Args:
        success: Binary success indicators [N]
        lcb: Lower confidence bounds [N]
        tau: Threshold
    
    Returns:
        (selective_success, coverage)
    """
    mask = lcb >= tau
    coverage = float(mask.mean())
    
    if not mask.any():
        return 0.0, 0.0
    
    sel_success = float(success[mask].mean())
    return sel_success, coverage


def net_utility(
    y_true_delta: np.ndarray,
    used_tool: np.ndarray,
    cost: np.ndarray
) -> float:
    """
    Compute Net Utility.
    
    Net Utility = mean(true_utility * used_tool - cost * used_tool)
    
    Args:
        y_true_delta: True utility gains [N]
        used_tool: Binary indicator of tool use [N]
        cost: Tool costs [N]
    
    Returns:
        Mean net utility
    """
    net = y_true_delta * used_tool - cost * used_tool
    return float(net.mean())


def reliability_curve(
    y_true_delta: np.ndarray,
    lcb: np.ndarray,
    num_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute reliability curve for LCB calibration.
    
    Bins predictions by LCB value and computes empirical coverage in each bin.
    
    Args:
        y_true_delta: True utility gains [N]
        lcb: Lower confidence bounds [N]
        num_bins: Number of bins
    
    Returns:
        (bin_centers, empirical_coverage, bin_counts)
    """
    # Create bins
    lcb_min, lcb_max = lcb.min(), lcb.max()
    bins = np.linspace(lcb_min, lcb_max, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Compute coverage in each bin
    empirical_coverage = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)
    
    for i in range(num_bins):
        mask = (lcb >= bins[i]) & (lcb < bins[i+1])
        if i == num_bins - 1:  # Include right edge in last bin
            mask = (lcb >= bins[i]) & (lcb <= bins[i+1])
        
        if mask.any():
            # Coverage = fraction where true value >= LCB
            empirical_coverage[i] = (y_true_delta[mask] >= lcb[mask]).mean()
            bin_counts[i] = mask.sum()
    
    return bin_centers, empirical_coverage, bin_counts


def plot_reliability_curve(
    bin_centers: np.ndarray,
    empirical_coverage: np.ndarray,
    bin_counts: np.ndarray,
    target_coverage: float,
    out_path: str
) -> None:
    """Plot reliability curve."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Coverage plot
    ax1.plot(bin_centers, empirical_coverage, 'o-', label='Empirical Coverage')
    ax1.axhline(target_coverage, color='r', linestyle='--', label=f'Target ({target_coverage:.0%})')
    ax1.set_xlabel('LCB Value')
    ax1.set_ylabel('Empirical Coverage')
    ax1.set_title('Reliability Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Histogram
    ax2.bar(bin_centers, bin_counts, width=(bin_centers[1] - bin_centers[0]) * 0.8)
    ax2.set_xlabel('LCB Value')
    ax2.set_ylabel('Count')
    ax2.set_title('LCB Distribution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved reliability curve to {out_path}")


def plot_selective_success(
    success: np.ndarray,
    lcb: np.ndarray,
    thresholds: np.ndarray,
    out_path: str
) -> None:
    """Plot selective success vs coverage trade-off."""
    selective_success_vals = []
    coverage_vals = []
    
    for tau in thresholds:
        sel_succ, cov = selective_success_at_tau(success, lcb, tau)
        selective_success_vals.append(sel_succ)
        coverage_vals.append(cov)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(coverage_vals, selective_success_vals, 'o-', linewidth=2)
    ax.set_xlabel('Coverage (Fraction Selected)')
    ax.set_ylabel('Selective Success Rate')
    ax.set_title('Selective Success vs Coverage Trade-off')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Add diagonal (random baseline)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved selective success plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate decision-quality metrics")
    parser.add_argument("--pred_json", required=True, help="Per-decision log JSON file")
    parser.add_argument("--out_json", required=True, help="Output metrics JSON file")
    parser.add_argument("--out_plots", default=None, help="Output directory for plots")
    parser.add_argument("--target_coverage", type=float, default=0.9, help="Target coverage (1-alpha)")
    args = parser.parse_args()
    
    # Load predictions
    logger.info(f"Loading predictions from {args.pred_json}")
    with open(args.pred_json, 'r') as f:
        data = json.load(f)
    
    # Extract arrays
    y_true = np.array([d["y_true_delta"] for d in data], dtype=float)
    lcb = np.array([d["lcb"] for d in data], dtype=float)
    used = np.array([int(d["used_tool"]) for d in data], dtype=int)
    cost = np.array([float(d.get("cost", 0.0)) for d in data], dtype=float)
    success = np.array([int(d["success"]) for d in data], dtype=int)
    
    logger.info(f"Loaded {len(data)} decisions")
    
    # Compute metrics
    logger.info("Computing metrics...")
    
    wtr = wasted_tool_rate(y_true, used)
    sel_succ_0, cov_0 = selective_success_at_tau(success, lcb, 0.0)
    sel_succ_02, cov_02 = selective_success_at_tau(success, lcb, 0.2)
    sel_succ_05, cov_05 = selective_success_at_tau(success, lcb, 0.5)
    net_util = net_utility(y_true, used, cost)
    
    # Reliability curve
    bin_centers, emp_cov, bin_counts = reliability_curve(y_true, lcb, num_bins=10)
    
    # Compile metrics
    metrics = {
        "wasted_tool_rate": float(wtr),
        "selective_success": {
            "tau_0.0": {"success": float(sel_succ_0), "coverage": float(cov_0)},
            "tau_0.2": {"success": float(sel_succ_02), "coverage": float(cov_02)},
            "tau_0.5": {"success": float(sel_succ_05), "coverage": float(cov_05)},
        },
        "net_utility": float(net_util),
        "reliability_curve": {
            "bin_centers": bin_centers.tolist(),
            "empirical_coverage": emp_cov.tolist(),
            "bin_counts": bin_counts.tolist(),
        },
        "summary": {
            "num_decisions": len(data),
            "num_tool_calls": int(used.sum()),
            "tool_usage_rate": float(used.mean()),
            "mean_lcb": float(lcb.mean()),
            "mean_true_delta_v": float(y_true.mean()),
        }
    }
    
    # Save metrics
    logger.info(f"Saving metrics to {args.out_json}")
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("DECISION METRICS SUMMARY")
    print("="*60)
    print(f"Wasted Tool Rate:        {wtr:.2%}")
    print(f"Net Utility:             {net_util:.4f}")
    print(f"\nSelective Success@τ:")
    print(f"  τ=0.0: {sel_succ_0:.2%} (coverage: {cov_0:.2%})")
    print(f"  τ=0.2: {sel_succ_02:.2%} (coverage: {cov_02:.2%})")
    print(f"  τ=0.5: {sel_succ_05:.2%} (coverage: {cov_05:.2%})")
    print(f"\nTool Usage:              {used.mean():.2%}")
    print(f"Mean LCB:                {lcb.mean():.4f}")
    print(f"Mean True ΔV:            {y_true.mean():.4f}")
    print("="*60 + "\n")
    
    # Generate plots
    if args.out_plots:
        logger.info(f"Generating plots in {args.out_plots}")
        Path(args.out_plots).mkdir(parents=True, exist_ok=True)
        
        # Reliability curve
        plot_reliability_curve(
            bin_centers, emp_cov, bin_counts,
            args.target_coverage,
            f"{args.out_plots}/reliability_curve.png"
        )
        
        # Selective success
        thresholds = np.linspace(lcb.min(), lcb.max(), 20)
        plot_selective_success(
            success, lcb, thresholds,
            f"{args.out_plots}/selective_success.png"
        )
    
    logger.info("Done!")


if __name__ == "__main__":
    main()

