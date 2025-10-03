# Copyright (c) MINT
"""
Explanation bundles for MINT decisions.

Provides detailed explanations of tool selection decisions including:
- Feature contributions
- Edit magnitudes
- Ablation results
- Conformal bounds
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Any, Optional
import json
import torch
import numpy as np


@dataclass
class FeatureContribution:
    """
    Single feature contribution to a decision.
    
    Attributes:
        layer_name: Name of the layer (e.g., "R24", "M15")
        feature_id: Feature index in SAE
        activation_before: Feature activation before edit
        activation_after: Feature activation after edit
        delta: Change in activation (after - before)
        weight: Edit direction weight
        gate: Gating value (how much to apply edit)
    """
    layer_name: str
    feature_id: int
    activation_before: float
    activation_after: float
    delta: float
    weight: float
    gate: float
    
    @property
    def contribution(self) -> float:
        """Total contribution: delta * weight * gate."""
        return self.delta * self.weight * self.gate


@dataclass
class ExplanationBundle:
    """
    Complete explanation for a single tool selection decision.
    
    Provides full transparency into:
    - Which features were edited
    - How much they were edited
    - What the predicted utility was
    - What the conformal bounds were
    - Whether ablation would flip the decision
    
    Usage:
        bundle = ExplanationBundle(
            decision_id="query_123",
            chosen_tool="calculator",
            layer_feature_contrib=[
                FeatureContribution("R24", 5892, 1.23, 2.12, 0.89, 0.89, 1.0)
            ],
            edit_magnitudes={"R24": 0.5},
            predicted_delta_v=0.77,
            lcb=0.38,
            threshold=0.0,
            decision="use_tool",
            notes={"query": "What is 15% of 240?"}
        )
    """
    decision_id: str
    chosen_tool: str
    
    # Feature-level explanations
    layer_feature_contrib: List[FeatureContribution] = field(default_factory=list)
    edit_magnitudes: Dict[str, float] = field(default_factory=dict)  # per-layer L2 norms
    
    # Utility predictions
    predicted_delta_v: float = 0.0
    lcb: float = 0.0
    ucb: Optional[float] = None
    threshold: float = 0.0
    
    # Decision
    decision: str = "no_tool"  # "use_tool" or "no_tool"
    margin: float = 0.0  # LCB - threshold
    
    # Ablation results (for faithfulness)
    lcb_after_ablation: Optional[float] = None
    ablation_flipped: Optional[bool] = None
    ablated_features: List[Tuple[str, int]] = field(default_factory=list)  # (layer, feature_id)
    
    # Alternative tools considered
    alternative_tools: Dict[str, Dict[str, float]] = field(default_factory=dict)  # tool -> {lcb, delta_v}
    
    # Metadata
    notes: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[str] = None
    
    def get_top_features(self, k: int = 10) -> List[FeatureContribution]:
        """
        Get top-k features by contribution magnitude.
        
        Args:
            k: Number of top features to return
        
        Returns:
            List of top-k feature contributions
        """
        sorted_features = sorted(
            self.layer_feature_contrib,
            key=lambda f: abs(f.contribution),
            reverse=True
        )
        return sorted_features[:k]
    
    def get_features_by_layer(self, layer_name: str) -> List[FeatureContribution]:
        """Get all feature contributions for a specific layer."""
        return [f for f in self.layer_feature_contrib if f.layer_name == layer_name]
    
    def get_total_edit_magnitude(self) -> float:
        """Get total edit magnitude across all layers."""
        return sum(self.edit_magnitudes.values())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        # Convert FeatureContribution objects to dicts
        d['layer_feature_contrib'] = [asdict(f) for f in self.layer_feature_contrib]
        return d
    
    def to_json(self, path: str) -> None:
        """Save to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ExplanationBundle:
        """Load from dictionary."""
        # Convert feature dicts back to FeatureContribution objects
        if 'layer_feature_contrib' in d:
            d['layer_feature_contrib'] = [
                FeatureContribution(**f) for f in d['layer_feature_contrib']
            ]
        return cls(**d)
    
    @classmethod
    def from_json(cls, path: str) -> ExplanationBundle:
        """Load from JSON file."""
        with open(path, 'r') as f:
            d = json.load(f)
        return cls.from_dict(d)


@dataclass
class ExplanationCollector:
    """
    Collector for explanation bundles across multiple decisions.
    
    Provides utilities for:
    - Collecting explanations during inference
    - Analyzing patterns across decisions
    - Computing faithfulness metrics
    
    Usage:
        collector = ExplanationCollector()
        
        # During inference
        for query in queries:
            bundle = make_decision_with_explanation(query)
            collector.add(bundle)
        
        # Analysis
        stats = collector.get_statistics()
        collector.save_all("explanations.jsonl")
    """
    explanations: List[ExplanationBundle] = field(default_factory=list)
    
    def add(self, bundle: ExplanationBundle) -> None:
        """Add an explanation bundle."""
        self.explanations.append(bundle)
    
    def get_by_tool(self, tool: str) -> List[ExplanationBundle]:
        """Get all explanations for a specific tool."""
        return [e for e in self.explanations if e.chosen_tool == tool]
    
    def get_by_decision(self, decision: str) -> List[ExplanationBundle]:
        """Get all explanations with a specific decision."""
        return [e for e in self.explanations if e.decision == decision]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Compute statistics across all explanations.
        
        Returns:
            Dictionary with statistics
        """
        if not self.explanations:
            return {}
        
        # Tool usage
        tool_counts = {}
        for e in self.explanations:
            tool_counts[e.chosen_tool] = tool_counts.get(e.chosen_tool, 0) + 1
        
        # Decision counts
        decision_counts = {}
        for e in self.explanations:
            decision_counts[e.decision] = decision_counts.get(e.decision, 0) + 1
        
        # Average metrics
        avg_delta_v = np.mean([e.predicted_delta_v for e in self.explanations])
        avg_lcb = np.mean([e.lcb for e in self.explanations])
        avg_margin = np.mean([e.margin for e in self.explanations])
        avg_edit_mag = np.mean([e.get_total_edit_magnitude() for e in self.explanations])
        
        # Faithfulness (if ablation was performed)
        ablation_results = [e for e in self.explanations if e.ablation_flipped is not None]
        if ablation_results:
            flip_rate = sum(e.ablation_flipped for e in ablation_results) / len(ablation_results)
        else:
            flip_rate = None
        
        return {
            'num_decisions': len(self.explanations),
            'tool_counts': tool_counts,
            'decision_counts': decision_counts,
            'avg_predicted_delta_v': float(avg_delta_v),
            'avg_lcb': float(avg_lcb),
            'avg_margin': float(avg_margin),
            'avg_edit_magnitude': float(avg_edit_mag),
            'ablation_flip_rate': flip_rate,
        }
    
    def save_all(self, path: str, format: str = "jsonl") -> None:
        """
        Save all explanations to file.
        
        Args:
            path: Output file path
            format: "jsonl" (one JSON per line) or "json" (single array)
        """
        if format == "jsonl":
            with open(path, 'w') as f:
                for e in self.explanations:
                    f.write(json.dumps(e.to_dict()) + '\n')
        elif format == "json":
            with open(path, 'w') as f:
                json.dump([e.to_dict() for e in self.explanations], f, indent=2)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    @classmethod
    def load_all(cls, path: str, format: str = "jsonl") -> ExplanationCollector:
        """
        Load explanations from file.
        
        Args:
            path: Input file path
            format: "jsonl" or "json"
        
        Returns:
            ExplanationCollector with loaded explanations
        """
        collector = cls()
        
        if format == "jsonl":
            with open(path, 'r') as f:
                for line in f:
                    d = json.loads(line)
                    collector.add(ExplanationBundle.from_dict(d))
        elif format == "json":
            with open(path, 'r') as f:
                data = json.load(f)
                for d in data:
                    collector.add(ExplanationBundle.from_dict(d))
        else:
            raise ValueError(f"Unknown format: {format}")
        
        return collector
    
    def clear(self) -> None:
        """Clear all explanations."""
        self.explanations.clear()

