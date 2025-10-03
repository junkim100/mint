#!/usr/bin/env python3
"""
Evaluate faithfulness of mechanistic edits.

Metrics:
- Faithfulness@k: Fraction of decisions that flip when top-k features are ablated
- Placebo flip rate: Fraction of random edits that spuriously flip decisions
- Attribution consistency: Correlation between feature importance and edit magnitude

Usage:
    python scripts/eval_faithfulness.py \
        --explanations_json results/explanations.jsonl \
        --placebo_json results/placebo.jsonl \
        --out_json results/faithfulness_metrics.json
"""
from __future__ import annotations
import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def faithfulness_at_k(decisions: List[Dict], k: int) -> float:
    """
    Compute Faithfulness@k.
    
    Faithfulness@k = fraction of decisions where ablating top-k features flips the decision
    
    Args:
        decisions: List of explanation bundles with ablation results
        k: Number of top features to consider
    
    Returns:
        Faithfulness@k in [0, 1]
    """
    valid_decisions = [
        d for d in decisions
        if len(d.get("ablation_ranked_features", [])) >= k
        and d.get("ablation_flipped") is not None
    ]
    
    if not valid_decisions:
        return 0.0
    
    flipped = [int(d.get("ablation_flipped", False)) for d in valid_decisions]
    return float(np.mean(flipped))


def placebo_rate(placebo_decisions: List[Dict]) -> float:
    """
    Compute placebo flip rate.
    
    Placebo rate = fraction of random edits that spuriously flip decisions
    
    Args:
        placebo_decisions: List of decisions with random (placebo) edits
    
    Returns:
        Placebo flip rate in [0, 1]
    """
    if not placebo_decisions:
        return 0.0
    
    flipped = [int(d.get("flipped", False)) for d in placebo_decisions]
    return float(np.mean(flipped))


def attribution_consistency(decisions: List[Dict]) -> float:
    """
    Compute attribution consistency.
    
    Measures correlation between feature importance (from ablation) and
    edit magnitude (from mechanistic editor).
    
    Args:
        decisions: List of explanation bundles
    
    Returns:
        Pearson correlation coefficient
    """
    importances = []
    magnitudes = []
    
    for d in decisions:
        features = d.get("layer_feature_contrib", [])
        for f in features:
            # Importance: how much ablating this feature changes LCB
            importance = abs(f.get("delta", 0.0))
            # Magnitude: how much the editor changed this feature
            magnitude = abs(f.get("activation_after", 0.0) - f.get("activation_before", 0.0))
            
            importances.append(importance)
            magnitudes.append(magnitude)
    
    if len(importances) < 2:
        return 0.0
    
    # Compute Pearson correlation
    corr = np.corrcoef(importances, magnitudes)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


def feature_sparsity(decisions: List[Dict]) -> Dict[str, float]:
    """
    Compute feature sparsity statistics.
    
    Args:
        decisions: List of explanation bundles
    
    Returns:
        Dictionary with sparsity statistics
    """
    num_active_features = []
    total_features = []
    
    for d in decisions:
        features = d.get("layer_feature_contrib", [])
        active = sum(1 for f in features if abs(f.get("delta", 0.0)) > 1e-6)
        total = len(features)
        
        num_active_features.append(active)
        total_features.append(total)
    
    if not num_active_features:
        return {}
    
    return {
        "mean_active_features": float(np.mean(num_active_features)),
        "std_active_features": float(np.std(num_active_features)),
        "mean_total_features": float(np.mean(total_features)),
        "sparsity_ratio": float(np.mean(num_active_features) / np.mean(total_features)) if np.mean(total_features) > 0 else 0.0,
    }


def edit_magnitude_stats(decisions: List[Dict]) -> Dict[str, float]:
    """
    Compute edit magnitude statistics.
    
    Args:
        decisions: List of explanation bundles
    
    Returns:
        Dictionary with edit magnitude statistics
    """
    magnitudes = []
    
    for d in decisions:
        edit_mags = d.get("edit_magnitudes", {})
        total_mag = sum(edit_mags.values())
        magnitudes.append(total_mag)
    
    if not magnitudes:
        return {}
    
    return {
        "mean_edit_magnitude": float(np.mean(magnitudes)),
        "std_edit_magnitude": float(np.std(magnitudes)),
        "min_edit_magnitude": float(np.min(magnitudes)),
        "max_edit_magnitude": float(np.max(magnitudes)),
    }


def load_explanations(path: str) -> List[Dict]:
    """Load explanations from JSONL file."""
    explanations = []
    with open(path, 'r') as f:
        for line in f:
            explanations.append(json.loads(line))
    return explanations


def main():
    parser = argparse.ArgumentParser(description="Evaluate faithfulness metrics")
    parser.add_argument("--explanations_json", required=True, help="Explanations JSONL file")
    parser.add_argument("--placebo_json", default=None, help="Placebo results JSONL file")
    parser.add_argument("--out_json", required=True, help="Output metrics JSON file")
    args = parser.parse_args()
    
    # Load explanations
    logger.info(f"Loading explanations from {args.explanations_json}")
    explanations = load_explanations(args.explanations_json)
    logger.info(f"Loaded {len(explanations)} explanations")
    
    # Load placebo results if provided
    placebo_results = []
    if args.placebo_json:
        logger.info(f"Loading placebo results from {args.placebo_json}")
        placebo_results = load_explanations(args.placebo_json)
        logger.info(f"Loaded {len(placebo_results)} placebo results")
    
    # Compute metrics
    logger.info("Computing faithfulness metrics...")
    
    metrics = {
        "faithfulness": {
            "faithfulness@5": faithfulness_at_k(explanations, 5),
            "faithfulness@10": faithfulness_at_k(explanations, 10),
            "faithfulness@20": faithfulness_at_k(explanations, 20),
        },
        "placebo_flip_rate": placebo_rate(placebo_results) if placebo_results else None,
        "attribution_consistency": attribution_consistency(explanations),
        "feature_sparsity": feature_sparsity(explanations),
        "edit_magnitude": edit_magnitude_stats(explanations),
        "summary": {
            "num_explanations": len(explanations),
            "num_placebo": len(placebo_results),
        }
    }
    
    # Save metrics
    logger.info(f"Saving metrics to {args.out_json}")
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("FAITHFULNESS METRICS SUMMARY")
    print("="*60)
    print(f"Faithfulness@5:          {metrics['faithfulness']['faithfulness@5']:.2%}")
    print(f"Faithfulness@10:         {metrics['faithfulness']['faithfulness@10']:.2%}")
    print(f"Faithfulness@20:         {metrics['faithfulness']['faithfulness@20']:.2%}")
    
    if metrics['placebo_flip_rate'] is not None:
        print(f"Placebo Flip Rate:       {metrics['placebo_flip_rate']:.2%}")
    
    print(f"Attribution Consistency: {metrics['attribution_consistency']:.3f}")
    
    if metrics['feature_sparsity']:
        print(f"\nFeature Sparsity:")
        print(f"  Mean active features:  {metrics['feature_sparsity']['mean_active_features']:.1f}")
        print(f"  Sparsity ratio:        {metrics['feature_sparsity']['sparsity_ratio']:.2%}")
    
    if metrics['edit_magnitude']:
        print(f"\nEdit Magnitude:")
        print(f"  Mean:                  {metrics['edit_magnitude']['mean_edit_magnitude']:.4f}")
        print(f"  Std:                   {metrics['edit_magnitude']['std_edit_magnitude']:.4f}")
    
    print("="*60 + "\n")
    
    logger.info("Done!")


if __name__ == "__main__":
    main()

