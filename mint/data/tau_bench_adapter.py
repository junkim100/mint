"""ToolBench adapter for MINT.

This module loads real tool-use data from the ToolBench dataset to create
counterfactual pairs for training the Mechanistic Tool Editor (MTE).

ToolBench contains 187k+ real tool-use examples with actual API calls and responses.

Reference: https://github.com/OpenBMB/ToolBench
"""

from pathlib import Path
from typing import Dict, Iterator, List, Optional, Any
import json
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mint.config import MintConfig
from mint.data.counterfactual_pairs import CounterfactualPair, create_pair
from mint.models.sae_loader import encode_features
from mint.logging_utils import setup_logger

logger = setup_logger(__name__)


class TauBenchAdapter:
    """ToolBench adapter for creating counterfactual pairs from real tool-use data.

    Loads data from the ToolBench dataset which contains real user queries,
    tool calls, and API responses across 3451 tools and 16464 APIs.
    """

    def __init__(self, config: MintConfig, device: str = "cuda"):
        """Initialize adapter.

        Args:
            config: MINT configuration
            device: Device to use (e.g., "cuda:0", "cuda:1", etc.)
        """
        self.config = config
        self.device = device
        self.model = None
        self.tokenizer = None
        self.toolbench_data = None
        self.toolbench_path = Path("data/toolbench/data/toolllama_G123_dfs_train.json")

    def _load_model(self):
        """Load base model and tokenizer."""
        if self.model is None:
            model_id = self.config.model.pretrained or self.config.model.name
            tokenizer_id = self.config.model.tokenizer or model_id
            logger.info(f"Loading model: {model_id} on {self.device}")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=getattr(torch, self.config.model.dtype),
            )
            # Move model to the specified device explicitly to avoid invalid device_map issues
            device = torch.device(self.device)
            try:
                if device.type == "cuda":
                    # Set current CUDA device for this process
                    torch.cuda.set_device(device)
            except Exception:
                pass
            self.model.to(device)
            self.model.eval()

    def _load_toolbench_data(self):
        """Load ToolBench dataset."""
        if self.toolbench_data is None:
            logger.info(f"Loading ToolBench data from {self.toolbench_path}")
            if not self.toolbench_path.exists():
                raise FileNotFoundError(
                    f"ToolBench data not found at {self.toolbench_path}. "
                    f"Please download and extract the ToolBench dataset to data/toolbench/"
                )

            with open(self.toolbench_path) as f:
                self.toolbench_data = json.load(f)

            logger.info(f"Loaded {len(self.toolbench_data)} ToolBench samples")

    def _extract_tool_info(self, conversation: List[Dict]) -> Optional[Dict[str, str]]:
        """Extract tool name and category from a ToolBench conversation.

        Args:
            conversation: List of conversation turns

        Returns:
            Dict with tool info or None if extraction fails
        """
        system_msg = conversation[0]['value']

        # Extract tool name from system message
        # Format: "1.tool_name: description"
        tool_name = None
        if 'You have access of the following tools:' in system_msg:
            lines = system_msg.split('\n')
            for i, line in enumerate(lines):
                if 'You have access of the following tools:' in line and i+1 < len(lines):
                    tool_line = lines[i+1]
                    if tool_line.startswith('1.'):
                        # Extract tool name before colon
                        tool_name = tool_line.split(':')[0].replace('1.', '').strip()
                    break

        return {"tool": tool_name, "category": tool_name} if tool_name else None

    def _extract_query_and_response(self, conversation: List[Dict]) -> Optional[Dict[str, str]]:
        """Extract user query and tool response from a ToolBench conversation.

        Args:
            conversation: List of conversation turns

        Returns:
            Dict with query and tool_output, or None if extraction fails
        """
        # Find user query (turn 1)
        query = None
        if len(conversation) > 1 and conversation[1]['from'] == 'user':
            query = conversation[1]['value'].strip()

        # Find first function response (turn 3 typically)
        tool_output = None
        for turn in conversation:
            if turn['from'] == 'function':
                try:
                    # Parse the function response JSON
                    response_data = json.loads(turn['value'])
                    if 'response' in response_data:
                        tool_output = str(response_data['response'])
                    elif 'error' in response_data and response_data['error']:
                        # Skip error responses
                        return None
                    break
                except json.JSONDecodeError:
                    # If not valid JSON, use raw value
                    tool_output = turn['value']
                    break

        if query and tool_output:
            return {"query": query, "tool_output": tool_output}
        return None

    def _load_toolbench_scenarios(
        self,
        tools: Optional[List[str]],
        max_pairs: int,
        gpu_id: int = 0,
        num_gpus: int = 1,
    ) -> List[Dict[str, Any]]:
        """Load real scenarios from ToolBench dataset.

        Args:
            tools: Tool categories to filter (None = all tools)
            max_pairs: Maximum number of scenarios to load (total across all GPUs)
            gpu_id: GPU ID for this process
            num_gpus: Total number of GPUs

        Returns:
            List of scenario dictionaries for this GPU
        """
        self._load_toolbench_data()

        scenarios = []
        tool_counts = {}

        # Shuffle data for diversity (with fixed seed for reproducibility)
        indices = list(range(len(self.toolbench_data)))
        random.seed(42)
        random.shuffle(indices)

        # Calculate pairs per GPU
        pairs_per_gpu = max_pairs // num_gpus

        # Process scenarios with interleaved distribution across GPUs
        scenario_count = 0
        for idx in indices:
            # Skip scenarios not assigned to this GPU
            if scenario_count % num_gpus != gpu_id:
                scenario_count += 1
                continue

            if len(scenarios) >= pairs_per_gpu:
                break

            sample = self.toolbench_data[idx]
            conversation = sample['conversations']

            # Extract tool info
            tool_info = self._extract_tool_info(conversation)
            if not tool_info:
                scenario_count += 1
                continue

            tool_name = tool_info['tool']

            # Filter by tool categories if specified
            if tools is not None and tool_name not in tools:
                scenario_count += 1
                continue

            # Extract query and response
            query_response = self._extract_query_and_response(conversation)
            if not query_response:
                scenario_count += 1
                continue

            # Create scenario
            scenario = {
                "query": query_response["query"],
                "tool": tool_name,
                "tool_output": query_response["tool_output"],
                "domain": tool_name,  # Use tool name as domain
            }
            scenarios.append(scenario)

            # Track tool distribution
            tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
            scenario_count += 1

        logger.info(f"Loaded {len(scenarios)} real ToolBench scenarios for GPU {gpu_id}/{num_gpus}")
        logger.info(f"Tool distribution: {dict(sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)[:10])}")

        return scenarios

    def iter_pairs(
        self,
        max_pairs: int,
        tools: Optional[List[str]] = None,
        gpu_id: int = 0,
        num_gpus: int = 1,
        layers_override: Optional[List[int]] = None,
    ) -> Iterator[CounterfactualPair]:
        """Iterate over counterfactual pairs from ToolBench scenarios.

        Args:
            max_pairs: Maximum number of pairs to generate (total across all GPUs)
            tools: Tool categories to include (None = all tools)
            gpu_id: GPU ID for this process
            num_gpus: Total number of GPUs
            layers_override: If provided, use these layers instead of config.mte.layers

        Yields:
            CounterfactualPair instances
        """
        self._load_model()

        # Load real ToolBench scenarios for this GPU
        scenarios = self._load_toolbench_scenarios(tools, max_pairs, gpu_id, num_gpus)

        logger.info(f"Creating counterfactual pairs for {len(scenarios)} scenarios")

        for scenario in scenarios:
            try:
                # Determine which layers to process (allow override)
                layers_to_use = layers_override if layers_override is not None else self.config.mte.layers

                # Create no-tool features
                phi_no, h_no = create_pair(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    query=scenario["query"],
                    tool_output=None,  # No tool
                    tool_label=scenario["tool"],
                    layers=layers_to_use,
                    encode_fn=lambda layer_id, h: encode_features(
                        layer_id,
                        h,
                        self.config.mte.hidden_dim,
                        self.config.mte.hidden_expansion,
                        checkpoint_path=self.config.mte.sae_checkpoints.get(str(layer_id)),
                    ),
                    device=self.model.device,
                )

                # Clear CUDA cache to prevent OOM
                torch.cuda.empty_cache()

                # Create with-tool features
                phi_with, h_with = create_pair(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    query=scenario["query"],
                    tool_output=scenario["tool_output"],  # With tool
                    tool_label=scenario["tool"],
                    layers=layers_to_use,
                    encode_fn=lambda layer_id, h: encode_features(
                        layer_id,
                        h,
                        self.config.mte.hidden_dim,
                        self.config.mte.hidden_expansion,
                        checkpoint_path=self.config.mte.sae_checkpoints.get(str(layer_id)),
                    ),
                    device=self.model.device,
                )

                # Clear CUDA cache again
                torch.cuda.empty_cache()

                # Create pair
                pair = CounterfactualPair(
                    phi_no_tool=phi_no,
                    phi_with_tool=phi_with,
                    h_no_tool=h_no,
                    h_with_tool=h_with,
                    tool_label=scenario["tool"],
                    metadata={
                        "query": scenario["query"],
                        "tool_output": scenario["tool_output"][:500],  # Truncate long outputs
                        "domain": scenario["domain"],
                    },
                )

                yield pair

            except Exception as e:
                logger.error(f"Error creating pair for scenario: {e}")
                # Clear cache on error
                torch.cuda.empty_cache()
                continue

    def get_tau_bench_data(self) -> Dict[str, Any]:
        """Get ToolBench dataset metadata.

        Returns:
            Dictionary with dataset information
        """
        self._load_toolbench_data()

        # Extract unique tools
        tools = set()
        for sample in self.toolbench_data[:10000]:  # Sample for speed
            tool_info = self._extract_tool_info(sample['conversations'])
            if tool_info:
                tools.add(tool_info['tool'])

        return {
            "tools": sorted(list(tools)),
            "num_tools": len(tools),
            "num_samples": len(self.toolbench_data),
            "description": "ToolBench: Real tool-use dataset with 187k+ examples",
        }
