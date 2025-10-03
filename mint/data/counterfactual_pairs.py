"""Counterfactual pair construction for Phase A training."""

import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CounterfactualPair:
    """A counterfactual pair for editor training."""
    
    # No-tool state
    input_ids_no_tool: torch.Tensor
    attention_mask_no_tool: torch.Tensor
    hidden_states_no_tool: Dict[str, torch.Tensor]
    
    # With-tool state
    input_ids_with_tool: torch.Tensor
    attention_mask_with_tool: torch.Tensor
    hidden_states_with_tool: Dict[str, torch.Tensor]
    
    # Metadata
    tool_name: str
    tool_output: str
    success: bool
    delta_v: float


class CounterfactualPairBuilder:
    """
    Builds counterfactual pairs for editor training.
    
    For each decision state:
    1. No-tool pass: collect hidden states
    2. With-tool pass: append gold tool output, collect hidden states
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        activation_extractor,
        max_tool_output_length: int = 256,
    ):
        """
        Initialize counterfactual pair builder.
        
        Args:
            model: Language model
            tokenizer: Tokenizer
            activation_extractor: ActivationExtractor instance
            max_tool_output_length: Maximum length for tool output
        """
        self.model = model
        self.tokenizer = tokenizer
        self.activation_extractor = activation_extractor
        self.max_tool_output_length = max_tool_output_length
    
    def build_pair(
        self,
        context: str,
        tool_name: str,
        tool_output: str,
        success: bool,
        delta_v: float,
    ) -> CounterfactualPair:
        """
        Build a counterfactual pair.
        
        Args:
            context: Input context (before tool call)
            tool_name: Name of the tool
            tool_output: Output from the tool
            success: Whether the tool call was successful
            delta_v: Utility gain from using the tool
            
        Returns:
            CounterfactualPair
        """
        # No-tool pass
        inputs_no_tool = self.tokenizer(
            context,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.model.device)
        
        hidden_no_tool, _ = self.activation_extractor.extract_and_encode(
            input_ids=inputs_no_tool["input_ids"],
            attention_mask=inputs_no_tool["attention_mask"],
            position="last",
        )
        
        # With-tool pass: append tool output
        context_with_tool = self._format_with_tool(context, tool_name, tool_output)
        inputs_with_tool = self.tokenizer(
            context_with_tool,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.model.device)
        
        hidden_with_tool, _ = self.activation_extractor.extract_and_encode(
            input_ids=inputs_with_tool["input_ids"],
            attention_mask=inputs_with_tool["attention_mask"],
            position="last",
        )
        
        return CounterfactualPair(
            input_ids_no_tool=inputs_no_tool["input_ids"],
            attention_mask_no_tool=inputs_no_tool["attention_mask"],
            hidden_states_no_tool=hidden_no_tool,
            input_ids_with_tool=inputs_with_tool["input_ids"],
            attention_mask_with_tool=inputs_with_tool["attention_mask"],
            hidden_states_with_tool=hidden_with_tool,
            tool_name=tool_name,
            tool_output=tool_output,
            success=success,
            delta_v=delta_v,
        )
    
    def _format_with_tool(
        self,
        context: str,
        tool_name: str,
        tool_output: str,
    ) -> str:
        """
        Format context with tool output appended.
        
        Args:
            context: Original context
            tool_name: Tool name
            tool_output: Tool output
            
        Returns:
            Formatted context with tool output
        """
        # Truncate tool output if needed
        tool_output_tokens = self.tokenizer.encode(
            tool_output,
            add_special_tokens=False,
        )
        if len(tool_output_tokens) > self.max_tool_output_length:
            tool_output_tokens = tool_output_tokens[:self.max_tool_output_length]
            tool_output = self.tokenizer.decode(tool_output_tokens)
        
        # Format: context + [Tool: tool_name] output
        formatted = f"{context}\n\n[Tool: {tool_name}]\n{tool_output}\n"
        return formatted
    
    def build_batch(
        self,
        examples: List[Dict],
    ) -> List[CounterfactualPair]:
        """
        Build a batch of counterfactual pairs.
        
        Args:
            examples: List of example dictionaries with keys:
                - context: str
                - tool_name: str
                - tool_output: str
                - success: bool
                - delta_v: float
                
        Returns:
            List of CounterfactualPair objects
        """
        pairs = []
        
        for example in examples:
            try:
                pair = self.build_pair(
                    context=example["context"],
                    tool_name=example["tool_name"],
                    tool_output=example["tool_output"],
                    success=example["success"],
                    delta_v=example["delta_v"],
                )
                pairs.append(pair)
            except Exception as e:
                logger.warning(f"Failed to build pair: {e}")
                continue
        
        return pairs

