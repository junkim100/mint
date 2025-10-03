"""
Risk budget tracking for multi-step trajectories.

Implements Proposal Section 3.4 and Appendix B:
- Trajectory-level risk budget allocation
- Per-tool family budget tracking
- Anytime stopping guarantees

Reference: MINT Proposal Section 3.4, Appendix B
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class RiskBudget:
    """
    Risk budget tracker for multi-step trajectories.
    
    Implements the proposal's trajectory-level budget:
    "Track a risk budget ρ and use sequential conformal to ensure
    P(any harmful call over T steps) ≤ α given the budget"
    
    Reference: MINT Proposal Section 3.4
    """
    
    def __init__(
        self,
        total_budget: float = 0.1,
        horizon: int = 100,
        allocation_strategy: str = "uniform",
        per_tool_budgets: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize risk budget.
        
        Args:
            total_budget: Total risk budget α (e.g., 0.1 for 10% error rate)
            horizon: Expected trajectory length T
            allocation_strategy: How to allocate budget across steps
                - "uniform": α/T per step
                - "adaptive": Adjust based on observed risk
                - "tool_family": Different budgets per tool type
            per_tool_budgets: Optional dict of tool -> budget fraction
        """
        self.total_budget = total_budget
        self.horizon = horizon
        self.allocation_strategy = allocation_strategy
        self.per_tool_budgets = per_tool_budgets or {}
        
        # Budget tracking
        self.remaining_budget = total_budget
        self.spent_budget = 0.0
        self.timestep = 0
        
        # Per-tool tracking
        self.tool_budgets: Dict[str, float] = {}
        self.tool_spent: Dict[str, float] = {}
        
        # History
        self.budget_history: List[float] = [total_budget]
        self.tool_calls: List[Tuple[int, str, float]] = []  # (timestep, tool, risk)
        
        # Initialize per-tool budgets
        if per_tool_budgets:
            for tool, fraction in per_tool_budgets.items():
                self.tool_budgets[tool] = total_budget * fraction
                self.tool_spent[tool] = 0.0
    
    def get_step_budget(self, timestep: Optional[int] = None) -> float:
        """
        Get risk budget for current step.
        
        Args:
            timestep: Optional timestep (uses self.timestep if None)
            
        Returns:
            Risk budget for this step
        """
        if timestep is None:
            timestep = self.timestep
        
        if self.allocation_strategy == "uniform":
            # Uniform allocation: α/T per step
            return self.total_budget / self.horizon
        
        elif self.allocation_strategy == "adaptive":
            # Adaptive: allocate remaining budget over remaining steps
            remaining_steps = max(1, self.horizon - timestep)
            return self.remaining_budget / remaining_steps
        
        else:
            raise ValueError(f"Unknown allocation strategy: {self.allocation_strategy}")
    
    def can_afford(
        self,
        risk: float,
        tool: Optional[str] = None,
    ) -> bool:
        """
        Check if we can afford a tool call with given risk.
        
        Args:
            risk: Risk of this tool call (e.g., 1 - confidence)
            tool: Optional tool name (for per-tool budgets)
            
        Returns:
            True if within budget
        """
        # Check global budget
        if risk > self.remaining_budget:
            return False
        
        # Check per-tool budget if applicable
        if tool and tool in self.tool_budgets:
            if risk > (self.tool_budgets[tool] - self.tool_spent.get(tool, 0.0)):
                return False
        
        return True
    
    def spend(
        self,
        risk: float,
        tool: Optional[str] = None,
    ):
        """
        Spend risk budget on a tool call.
        
        Args:
            risk: Risk of this tool call
            tool: Optional tool name
        """
        # Spend from global budget
        self.spent_budget += risk
        self.remaining_budget -= risk
        
        # Spend from per-tool budget
        if tool:
            if tool not in self.tool_spent:
                self.tool_spent[tool] = 0.0
            self.tool_spent[tool] += risk
        
        # Record
        self.tool_calls.append((self.timestep, tool or "unknown", risk))
        self.budget_history.append(self.remaining_budget)
        
        logger.debug(
            f"Spent risk budget: {risk:.4f} on {tool or 'unknown'} "
            f"(remaining: {self.remaining_budget:.4f})"
        )
    
    def step(self):
        """Advance to next timestep."""
        self.timestep += 1
    
    def reset(self):
        """Reset budget for new trajectory."""
        self.remaining_budget = self.total_budget
        self.spent_budget = 0.0
        self.timestep = 0
        self.budget_history = [self.total_budget]
        self.tool_calls = []
        
        # Reset per-tool budgets
        for tool in self.tool_budgets:
            self.tool_spent[tool] = 0.0
        
        logger.info("Reset risk budget")
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get budget statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            'total_budget': self.total_budget,
            'spent_budget': self.spent_budget,
            'remaining_budget': self.remaining_budget,
            'utilization': self.spent_budget / self.total_budget if self.total_budget > 0 else 0.0,
            'num_calls': len(self.tool_calls),
            'timestep': self.timestep,
        }
        
        # Per-tool statistics
        for tool, spent in self.tool_spent.items():
            budget = self.tool_budgets.get(tool, 0.0)
            stats[f'{tool}_spent'] = spent
            stats[f'{tool}_remaining'] = budget - spent
            stats[f'{tool}_utilization'] = spent / budget if budget > 0 else 0.0
        
        return stats
    
    def is_depleted(self, threshold: float = 0.01) -> bool:
        """
        Check if budget is depleted.
        
        Args:
            threshold: Minimum remaining budget
            
        Returns:
            True if budget is below threshold
        """
        return self.remaining_budget < threshold
    
    def get_tool_budget(self, tool: str) -> Tuple[float, float]:
        """
        Get budget info for a specific tool.
        
        Args:
            tool: Tool name
            
        Returns:
            Tuple of (allocated_budget, spent_budget)
        """
        allocated = self.tool_budgets.get(tool, 0.0)
        spent = self.tool_spent.get(tool, 0.0)
        return allocated, spent


class HierarchicalRiskBudget(RiskBudget):
    """
    Hierarchical risk budget with tool families.
    
    Implements the proposal's per-tool-family budgets:
    "Track risk budgets per tool family (e.g., external web vs. local calculator)"
    
    Reference: MINT Proposal Section 5
    """
    
    def __init__(
        self,
        total_budget: float = 0.1,
        horizon: int = 100,
        tool_families: Optional[Dict[str, List[str]]] = None,
        family_budgets: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize hierarchical risk budget.
        
        Args:
            total_budget: Total risk budget
            horizon: Expected trajectory length
            tool_families: Dict of family_name -> [tool_names]
            family_budgets: Dict of family_name -> budget_fraction
        """
        super().__init__(
            total_budget=total_budget,
            horizon=horizon,
            allocation_strategy="adaptive",
        )
        
        self.tool_families = tool_families or {}
        self.family_budgets: Dict[str, float] = {}
        self.family_spent: Dict[str, float] = {}
        
        # Initialize family budgets
        if family_budgets:
            for family, fraction in family_budgets.items():
                self.family_budgets[family] = total_budget * fraction
                self.family_spent[family] = 0.0
        else:
            # Equal allocation across families
            num_families = len(self.tool_families)
            if num_families > 0:
                for family in self.tool_families:
                    self.family_budgets[family] = total_budget / num_families
                    self.family_spent[family] = 0.0
    
    def get_tool_family(self, tool: str) -> Optional[str]:
        """Get family for a tool."""
        for family, tools in self.tool_families.items():
            if tool in tools:
                return family
        return None
    
    def can_afford(
        self,
        risk: float,
        tool: Optional[str] = None,
    ) -> bool:
        """Check if we can afford a tool call (with family budget)."""
        # Check global budget
        if not super().can_afford(risk, tool):
            return False
        
        # Check family budget
        if tool:
            family = self.get_tool_family(tool)
            if family and family in self.family_budgets:
                family_remaining = (
                    self.family_budgets[family] - 
                    self.family_spent.get(family, 0.0)
                )
                if risk > family_remaining:
                    return False
        
        return True
    
    def spend(
        self,
        risk: float,
        tool: Optional[str] = None,
    ):
        """Spend risk budget (with family tracking)."""
        # Spend from global and per-tool budgets
        super().spend(risk, tool)
        
        # Spend from family budget
        if tool:
            family = self.get_tool_family(tool)
            if family:
                if family not in self.family_spent:
                    self.family_spent[family] = 0.0
                self.family_spent[family] += risk
    
    def get_statistics(self) -> Dict[str, float]:
        """Get statistics including family budgets."""
        stats = super().get_statistics()
        
        # Add family statistics
        for family, spent in self.family_spent.items():
            budget = self.family_budgets.get(family, 0.0)
            stats[f'family_{family}_spent'] = spent
            stats[f'family_{family}_remaining'] = budget - spent
            stats[f'family_{family}_utilization'] = spent / budget if budget > 0 else 0.0
        
        return stats

