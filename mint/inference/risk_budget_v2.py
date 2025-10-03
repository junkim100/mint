# Copyright (c) MINT
"""
Hierarchical risk budgets for trajectory-level risk control.

Implements:
- Per-tool-family risk budgets
- Adaptive budget allocation
- Risk budget schedulers
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class HierarchicalRiskBudget:
    """
    Hierarchical risk budgets per tool family (e.g., 'web', 'local').
    
    Provides fine-grained control over risk allocation across different
    tool categories, preventing over-reliance on any single tool type.
    
    Usage:
        budget = HierarchicalRiskBudget(
            family_budgets={'web': 0.05, 'local': 0.05},
            tool_families={'search': 'web', 'calculator': 'local'}
        )
        
        # Check if tool call is allowed
        if budget.allows('search', margin=0.3):
            # Use tool
            budget.charge('search', alpha_spend=0.01)
    """
    family_budgets: Dict[str, float]  # e.g., {'web': 0.05, 'local': 0.05}
    tool_families: Dict[str, str] = field(default_factory=dict)  # tool -> family mapping
    family_costs: Dict[str, float] = field(default_factory=dict)  # consumed budget per family
    
    # Tracking
    tool_call_history: List[str] = field(default_factory=list)
    alpha_spend_history: List[float] = field(default_factory=list)

    def allows(self, tool: str, margin: float) -> bool:
        """
        Return True if we have budget AND the candidate margin (LCB - cost) is positive.
        
        Args:
            tool: Tool name (e.g., 'search', 'calculator')
            margin: LCB - cost (should be positive to use tool)
        
        Returns:
            True if tool call is allowed
        """
        # Check margin first (must be positive)
        if margin <= 0.0:
            return False
        
        # Get tool family
        family = self.tool_families.get(tool, 'default')
        
        # Check family budget
        budget = self.family_budgets.get(family, 0.0)
        consumed = self.family_costs.get(family, 0.0)
        
        return consumed < budget
    
    def charge(self, tool: str, alpha_spend: float) -> None:
        """
        Charge budget for a tool call.
        
        Args:
            tool: Tool name
            alpha_spend: Amount of risk budget to consume
        """
        family = self.tool_families.get(tool, 'default')
        
        # Update consumed budget
        self.family_costs[family] = self.family_costs.get(family, 0.0) + alpha_spend
        
        # Track history
        self.tool_call_history.append(tool)
        self.alpha_spend_history.append(alpha_spend)
        
        logger.debug(
            f"Charged {alpha_spend:.4f} to family '{family}' for tool '{tool}'. "
            f"Consumed: {self.family_costs[family]:.4f}/{self.family_budgets.get(family, 0.0):.4f}"
        )
    
    def reset(self, family: Optional[str] = None, value: Optional[float] = None) -> None:
        """
        Reset budget for a family (or all families).
        
        Args:
            family: Family name (None = reset all)
            value: New budget value (None = reset to original)
        """
        if family is None:
            # Reset all families
            self.family_costs.clear()
            self.tool_call_history.clear()
            self.alpha_spend_history.clear()
            logger.info("Reset all family budgets")
        else:
            # Reset specific family
            if value is not None:
                self.family_budgets[family] = value
            self.family_costs[family] = 0.0
            logger.info(f"Reset family '{family}' budget to {self.family_budgets.get(family, 0.0):.4f}")
    
    def get_remaining(self, family: str) -> float:
        """Get remaining budget for a family."""
        budget = self.family_budgets.get(family, 0.0)
        consumed = self.family_costs.get(family, 0.0)
        return max(0.0, budget - consumed)
    
    def get_stats(self) -> Dict[str, any]:
        """Get budget statistics."""
        stats = {
            'total_tool_calls': len(self.tool_call_history),
            'total_alpha_spent': sum(self.alpha_spend_history),
            'families': {}
        }
        
        for family, budget in self.family_budgets.items():
            consumed = self.family_costs.get(family, 0.0)
            stats['families'][family] = {
                'budget': budget,
                'consumed': consumed,
                'remaining': budget - consumed,
                'utilization': consumed / budget if budget > 0 else 0.0
            }
        
        return stats


@dataclass
class AdaptiveRiskScheduler:
    """
    Adaptive risk budget scheduler for trajectories.
    
    Dynamically allocates risk budget across trajectory steps based on:
    - Remaining trajectory length
    - Historical tool usage patterns
    - Current budget consumption
    
    Usage:
        scheduler = AdaptiveRiskScheduler(
            total_alpha=0.1,
            trajectory_length=10
        )
        
        for step in range(10):
            step_budget = scheduler.get_step_budget(step)
            # Use step_budget for this step
            scheduler.record_spend(actual_spend)
    """
    total_alpha: float  # Total risk budget for trajectory
    trajectory_length: int  # Expected trajectory length
    strategy: str = "adaptive"  # "uniform", "adaptive", "conservative"
    
    # State
    consumed_alpha: float = 0.0
    current_step: int = 0
    spend_history: List[float] = field(default_factory=list)
    
    def get_step_budget(self, step: Optional[int] = None) -> float:
        """
        Get risk budget for current step.
        
        Args:
            step: Step number (None = use current_step)
        
        Returns:
            Risk budget for this step
        """
        if step is None:
            step = self.current_step
        
        remaining_steps = max(1, self.trajectory_length - step)
        remaining_budget = max(0.0, self.total_alpha - self.consumed_alpha)
        
        if self.strategy == "uniform":
            # Equal allocation across all steps
            return self.total_alpha / self.trajectory_length
        
        elif self.strategy == "adaptive":
            # Allocate remaining budget evenly across remaining steps
            return remaining_budget / remaining_steps
        
        elif self.strategy == "conservative":
            # Front-load budget (more budget early on)
            # Use exponential decay: budget_t = remaining * (1 - decay)^t
            decay = 0.1
            return remaining_budget * (1 - decay) ** step
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def record_spend(self, alpha_spend: float) -> None:
        """
        Record actual risk budget spent.
        
        Args:
            alpha_spend: Amount of risk budget consumed
        """
        self.consumed_alpha += alpha_spend
        self.spend_history.append(alpha_spend)
        self.current_step += 1
        
        logger.debug(
            f"Step {self.current_step}: spent {alpha_spend:.4f}, "
            f"total consumed: {self.consumed_alpha:.4f}/{self.total_alpha:.4f}"
        )
    
    def reset(self) -> None:
        """Reset scheduler for new trajectory."""
        self.consumed_alpha = 0.0
        self.current_step = 0
        self.spend_history.clear()
        logger.info("Reset risk scheduler")
    
    def get_remaining(self) -> float:
        """Get remaining risk budget."""
        return max(0.0, self.total_alpha - self.consumed_alpha)
    
    def is_exhausted(self) -> bool:
        """Check if risk budget is exhausted."""
        return self.consumed_alpha >= self.total_alpha
    
    def get_stats(self) -> Dict[str, any]:
        """Get scheduler statistics."""
        return {
            'total_alpha': self.total_alpha,
            'consumed_alpha': self.consumed_alpha,
            'remaining_alpha': self.get_remaining(),
            'current_step': self.current_step,
            'trajectory_length': self.trajectory_length,
            'average_spend': sum(self.spend_history) / len(self.spend_history) if self.spend_history else 0.0,
            'strategy': self.strategy,
        }


@dataclass
class CombinedRiskManager:
    """
    Combined risk manager with hierarchical budgets and adaptive scheduling.
    
    Integrates:
    - Hierarchical family budgets
    - Adaptive trajectory scheduling
    - Per-tool tracking
    
    Usage:
        manager = CombinedRiskManager(
            family_budgets={'web': 0.05, 'local': 0.05},
            tool_families={'search': 'web', 'calculator': 'local'},
            total_alpha=0.1,
            trajectory_length=10
        )
        
        for step in range(10):
            if manager.allows('search', margin=0.3):
                # Use tool
                manager.charge('search')
    """
    hierarchical_budget: HierarchicalRiskBudget
    scheduler: AdaptiveRiskScheduler
    
    @classmethod
    def create(
        cls,
        family_budgets: Dict[str, float],
        tool_families: Dict[str, str],
        total_alpha: float = 0.1,
        trajectory_length: int = 10,
        strategy: str = "adaptive"
    ) -> CombinedRiskManager:
        """Create a combined risk manager."""
        hierarchical = HierarchicalRiskBudget(
            family_budgets=family_budgets,
            tool_families=tool_families
        )
        scheduler = AdaptiveRiskScheduler(
            total_alpha=total_alpha,
            trajectory_length=trajectory_length,
            strategy=strategy
        )
        return cls(hierarchical_budget=hierarchical, scheduler=scheduler)
    
    def allows(self, tool: str, margin: float) -> bool:
        """Check if tool call is allowed."""
        # Check hierarchical budget
        if not self.hierarchical_budget.allows(tool, margin):
            return False
        
        # Check scheduler budget
        if self.scheduler.is_exhausted():
            return False
        
        return True
    
    def charge(self, tool: str, alpha_spend: Optional[float] = None) -> None:
        """
        Charge both hierarchical and scheduler budgets.
        
        Args:
            tool: Tool name
            alpha_spend: Amount to charge (None = use step budget)
        """
        if alpha_spend is None:
            alpha_spend = self.scheduler.get_step_budget()
        
        self.hierarchical_budget.charge(tool, alpha_spend)
        self.scheduler.record_spend(alpha_spend)
    
    def reset(self) -> None:
        """Reset both budgets."""
        self.hierarchical_budget.reset()
        self.scheduler.reset()
    
    def get_stats(self) -> Dict[str, any]:
        """Get combined statistics."""
        return {
            'hierarchical': self.hierarchical_budget.get_stats(),
            'scheduler': self.scheduler.get_stats()
        }

