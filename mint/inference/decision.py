"""
MINT decision maker for risk-calibrated tool selection.

Implements Proposal Section 3.4 and Section 6:
- Risk-calibrated tool selection with conformal LCBs
- Optional online conformal for distribution shift
- Optional risk budget tracking for multi-step trajectories

Reference: MINT Proposal Section 3.4, Section 6
"""

import torch
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class MINTDecisionMaker:
    """
    MINT decision maker that selects tools based on LCB-adjusted utility.

    For each candidate tool u:
        1. Apply editor to get counterfactual state
        2. Predict ΔV with value head
        3. Compute LCB with conformal calibrator
        4. Subtract cost
        5. Select argmax if positive and within risk budget
    """

    def __init__(
        self,
        tools: List[str],
        editors: Dict[str, any],
        value_heads: Dict[str, any],
        calibrators: Dict[str, any],
        sae_loader: any,
        costs: Dict[str, float],
        risk_budget: any,
        residual_layers: List[int] = [5, 12, 18, 24, 28, 31],
    ):
        """
        Initialize MINT decision maker.

        Args:
            tools: List of tool names
            editors: Dict mapping tool names to MechanisticEditor instances
            value_heads: Dict mapping tool names to ValueHead instances
            calibrators: Dict mapping tool names to ConformalCalibrator or
                        OnlineConformalPredictor or AdaptiveConformalPredictor instances
            sae_loader: SAELoader instance
            costs: Dict mapping tool names to costs
            risk_budget: RiskBudget or HierarchicalRiskBudget instance
            residual_layers: List of residual layer indices

        Note:
            - Supports both split and online conformal prediction
            - Supports risk budget tracking for multi-step trajectories
            - See mint.inference.online_conformal and mint.inference.risk_budget
        """
        self.tools = tools
        self.editors = editors
        self.value_heads = value_heads
        self.calibrators = calibrators
        self.sae_loader = sae_loader
        self.costs = costs
        self.risk_budget = risk_budget
        self.residual_layers = residual_layers

        # Validate
        for tool in tools:
            if tool not in editors:
                raise ValueError(f"Editor for tool '{tool}' not found")
            if tool not in value_heads:
                raise ValueError(f"Value head for tool '{tool}' not found")
            if tool not in calibrators:
                raise ValueError(f"Calibrator for tool '{tool}' not found")
            if tool not in costs:
                logger.warning(f"Cost for tool '{tool}' not found, using 0.0")
                costs[tool] = 0.0

    @torch.no_grad()
    def decide(
        self,
        hidden_by_layer: Dict[str, torch.Tensor],
        ctx_vec: Optional[torch.Tensor] = None,
        return_scores: bool = False,
    ) -> Union[str, Tuple[str, Dict[str, float]]]:
        """
        Make a decision on which tool to call.

        Args:
            hidden_by_layer: Dict mapping layer keys to activations [batch, hidden_size]
            ctx_vec: Optional context vector for learned gates
            return_scores: Whether to return scores for all tools

        Returns:
            Selected tool name, or (tool_name, scores_dict) if return_scores=True
        """
        batch_size = next(iter(hidden_by_layer.values())).size(0)

        if batch_size > 1:
            # Batch processing
            return self._decide_batch(hidden_by_layer, ctx_vec, return_scores)

        # Single item
        scores = {}

        # Baseline: NoTool
        scores["NoTool"] = 0.0

        # Evaluate each tool
        for tool in self.tools:
            # Apply editor
            edited = self.editors[tool](hidden_by_layer, ctx_vec)

            # Get R-layer activations in order
            r_keys = sorted([k for k in edited.keys() if k.startswith("R")],
                          key=lambda x: int(x[1:]))
            edited_acts = [edited[k] for k in r_keys]

            # Predict ΔV
            pred_delta_v = self.value_heads[tool](edited_acts)

            # Apply conformal LCB
            lcb = self.calibrators[tool].lcb(pred_delta_v)

            # Subtract cost
            net_value = lcb - self.costs[tool]

            scores[tool] = float(net_value.item())

        # Select best tool
        best_tool = max(scores.items(), key=lambda x: x[1])
        tool_name, best_score = best_tool

        # Check risk budget
        if best_score > 0 and self.risk_budget.allows(tool_name):
            selected = tool_name
        else:
            selected = "NoTool"

        # Consume budget
        self.risk_budget.consume(selected)

        if return_scores:
            return selected, scores
        else:
            return selected

    @torch.no_grad()
    def _decide_batch(
        self,
        hidden_by_layer: Dict[str, torch.Tensor],
        ctx_vec: Optional[torch.Tensor],
        return_scores: bool,
    ) -> Union[List[str], Tuple[List[str], List[Dict[str, float]]]]:
        """
        Make decisions for a batch of items.

        Args:
            hidden_by_layer: Dict mapping layer keys to activations [batch, hidden_size]
            ctx_vec: Optional context vector
            return_scores: Whether to return scores

        Returns:
            List of selected tool names, or (tool_names, scores_list) if return_scores=True
        """
        batch_size = next(iter(hidden_by_layer.values())).size(0)

        # Compute scores for all tools
        all_scores = {tool: [] for tool in self.tools + ["NoTool"]}

        # NoTool baseline
        all_scores["NoTool"] = [0.0] * batch_size

        # Evaluate each tool
        for tool in self.tools:
            # Apply editor
            edited = self.editors[tool](hidden_by_layer, ctx_vec)

            # Get R-layer activations
            r_keys = sorted([k for k in edited.keys() if k.startswith("R")],
                          key=lambda x: int(x[1:]))
            edited_acts = [edited[k] for k in r_keys]

            # Predict ΔV
            pred_delta_v = self.value_heads[tool](edited_acts)  # [batch]

            # Apply conformal LCB
            lcb = self.calibrators[tool].lcb(pred_delta_v)  # [batch]

            # Subtract cost
            net_value = lcb - self.costs[tool]  # [batch]

            all_scores[tool] = net_value.cpu().tolist()

        # Select best tool per item
        selections = []
        scores_list = []

        for i in range(batch_size):
            item_scores = {tool: all_scores[tool][i] for tool in all_scores}

            best_tool = max(item_scores.items(), key=lambda x: x[1])
            tool_name, best_score = best_tool

            # Check risk budget
            if best_score > 0 and self.risk_budget.allows(tool_name):
                selected = tool_name
            else:
                selected = "NoTool"

            # Consume budget
            self.risk_budget.consume(selected)

            selections.append(selected)
            scores_list.append(item_scores)

        if return_scores:
            return selections, scores_list
        else:
            return selections

    def reset_budget(self):
        """Reset risk budget for new trajectory."""
        self.risk_budget.reset()

