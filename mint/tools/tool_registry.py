"""Tool registry and specifications."""

from dataclasses import dataclass
from typing import List, Dict, Optional, Callable
import logging

logger = logging.getLogger(__name__)


@dataclass
class ToolSpec:
    """Specification for a tool."""
    
    name: str
    description: str
    category: str  # web_search, calculator, rag_retrieval, domain_api, file_system
    cost: float
    positive_indicators: List[str]
    execute_fn: Optional[Callable] = None
    
    def __post_init__(self):
        """Validate tool spec."""
        valid_categories = [
            "web_search",
            "calculator",
            "rag_retrieval",
            "domain_api",
            "file_system",
        ]
        if self.category not in valid_categories:
            raise ValueError(
                f"Invalid category '{self.category}'. "
                f"Must be one of {valid_categories}"
            )


class ToolRegistry:
    """Registry for managing tools."""
    
    def __init__(self):
        """Initialize tool registry."""
        self.tools: Dict[str, ToolSpec] = {}
        self._initialize_default_tools()
    
    def _initialize_default_tools(self):
        """Initialize default tool specifications."""
        
        # Web search / browser
        self.register(ToolSpec(
            name="web_search",
            description="Search the web for current information, facts, and events",
            category="web_search",
            cost=0.10,
            positive_indicators=[
                "open-world facts",
                "date-sensitive",
                "according to",
                "current events",
                "latest",
                "recent",
                "news",
            ],
        ))
        
        # Calculator / code execution
        self.register(ToolSpec(
            name="calculator",
            description="Perform arithmetic calculations and mathematical operations",
            category="calculator",
            cost=0.02,
            positive_indicators=[
                "arithmetic",
                "numbers",
                "calculate",
                "compute",
                "sum",
                "multiply",
                "divide",
                "percentage",
            ],
        ))
        
        # Knowledge retrieval / RAG
        self.register(ToolSpec(
            name="rag_retrieval",
            description="Retrieve relevant knowledge from a knowledge base",
            category="rag_retrieval",
            cost=0.05,
            positive_indicators=[
                "multi-hop",
                "factual query",
                "entity overlap",
                "knowledge base",
                "documentation",
            ],
        ))
        
        # Domain APIs (airline, retail, telecom, etc.)
        self.register(ToolSpec(
            name="domain_api",
            description="Call domain-specific APIs for external state",
            category="domain_api",
            cost=0.08,
            positive_indicators=[
                "schema tokens",
                "external state",
                "API call",
                "database",
                "query",
                "fetch",
            ],
        ))
        
        # File system operations
        self.register(ToolSpec(
            name="file_system",
            description="Perform file system operations (open, save, move, rename)",
            category="file_system",
            cost=0.03,
            positive_indicators=[
                "file path",
                "open",
                "save",
                "move",
                "rename",
                "delete",
                "directory",
            ],
        ))
    
    def register(self, tool_spec: ToolSpec):
        """
        Register a tool.
        
        Args:
            tool_spec: Tool specification
        """
        if tool_spec.name in self.tools:
            logger.warning(f"Tool '{tool_spec.name}' already registered, overwriting")
        
        self.tools[tool_spec.name] = tool_spec
        logger.info(f"Registered tool: {tool_spec.name} ({tool_spec.category})")
    
    def get(self, name: str) -> ToolSpec:
        """
        Get tool specification by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool specification
        """
        if name not in self.tools:
            raise ValueError(f"Tool '{name}' not found in registry")
        return self.tools[name]
    
    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self.tools.keys())
    
    def get_by_category(self, category: str) -> List[ToolSpec]:
        """
        Get all tools in a category.
        
        Args:
            category: Tool category
            
        Returns:
            List of tool specifications
        """
        return [tool for tool in self.tools.values() if tool.category == category]
    
    def get_costs(self) -> Dict[str, float]:
        """Get costs for all tools."""
        costs = {name: tool.cost for name, tool in self.tools.items()}
        costs["NoTool"] = 0.0
        return costs

