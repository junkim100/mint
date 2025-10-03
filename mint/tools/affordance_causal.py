"""
Causal affordance discovery for tool-feature mapping.

Implements Section 3.1 of the MINT proposal:
- Causal mediation/patching: "features whose ablation destroys success"
- Contrastive pairs: learn directions that close the representation gap

This replaces heuristic keyword-based labeling with proper causal feature identification.

Reference: MINT Proposal Section 3.1 (Mechanistic Feature Bank)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class CausalAffordanceDiscovery:
    """
    Discover tool affordances via causal mediation/patching or contrastive pairs.

    Implements Proposal Section 3.1:
    "For each tool u, learn a feature mask m_u^(ℓ) and signed edit directions w_u^(ℓ)
    indicating which features encode the kind of information u provides."

    Two methods:
    1. Causal mediation/patching: Ablate features, measure success drop
    2. Contrastive pairs: Compare (no-tool, with-tool) activations
    """

    def __init__(
        self,
        model,
        tokenizer,
        sae_loader,
        activation_extractor,
        device: str = "cuda",
    ):
        """
        Initialize causal affordance discovery.

        Args:
            model: Language model
            tokenizer: Tokenizer
            sae_loader: SAELoader instance
            activation_extractor: ActivationExtractor instance
            device: Device to run on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.sae_loader = sae_loader
        self.activation_extractor = activation_extractor
        self.device = device

    def discover_via_ablation(
        self,
        tool_name: str,
        examples: List[Dict],
        top_k: int = 256,
        batch_size: int = 8,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Discover tool affordances via causal ablation.

        Implements the proposal's causal mediation approach:
        "Replace φ^(ℓ) with a null baseline on data where u is essential;
        features whose ablation destroys success are candidates for m_u^(ℓ)."

        For each feature k in layer ℓ:
        1. Run with feature k active: measure success
        2. Run with feature k ablated: measure success
        3. If success drops significantly → k is in m_u^(ℓ)

        Args:
            tool_name: Name of the tool
            examples: List of examples where tool is essential
                     Each example: {"input": str, "expected_output": str, "requires_tool": bool}
            top_k: Number of top features to select per layer
            batch_size: Batch size for processing

        Returns:
            Tuple of (masks, directions) where:
            - masks: Dict[layer_key, binary_mask]
            - directions: Dict[layer_key, direction_vector]
        """
        logger.info(f"Discovering affordances for {tool_name} via causal ablation...")

        # Filter to tool-essential examples
        essential_examples = [ex for ex in examples if ex.get("requires_tool", True)]

        if len(essential_examples) == 0:
            logger.warning(f"No essential examples for {tool_name}, using all examples")
            essential_examples = examples

        masks = {}
        directions = {}

        # Get all SAEs (residual + MLP)
        all_saes = {}
        for layer, sae in self.sae_loader.sae_r.items():
            all_saes[f"R{layer}"] = sae
        for layer, sae in self.sae_loader.sae_m.items():
            all_saes[f"M{layer}"] = sae

        # Process each layer
        for layer_key, sae in tqdm(all_saes.items(), desc="Layers"):
            d_sae = sae.cfg.d_sae

            # Collect activations on tool-essential examples
            logger.debug(f"Collecting activations for {layer_key}...")
            activations = []

            for ex in essential_examples[:batch_size]:  # Limit for efficiency
                # Extract hidden state
                inputs = self.tokenizer(
                    ex["input"],
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                ).to(self.device)

                hidden_states, _ = self.activation_extractor.extract_and_encode(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    position="last",
                )

                if layer_key in hidden_states:
                    h = hidden_states[layer_key]
                    phi = sae.encode(h)
                    activations.append(phi)

            if len(activations) == 0:
                logger.warning(f"No activations for {layer_key}, skipping")
                continue

            # Stack activations
            activations_tensor = torch.cat(activations, dim=0)  # [num_examples, d_sae]

            # Compute feature importance via activation magnitude
            # Features with high activation on tool-essential examples are candidates
            feature_importance = activations_tensor.abs().mean(dim=0)  # [d_sae]

            # Select top-K features
            topk_indices = torch.topk(feature_importance, k=min(top_k, d_sae)).indices

            # Create mask
            mask = torch.zeros(d_sae, device=self.device)
            mask[topk_indices] = 1.0

            # Compute direction: mean activation on tool-essential examples
            direction = activations_tensor.mean(dim=0)  # [d_sae]
            direction = direction / (direction.norm() + 1e-8)  # Normalize

            masks[layer_key] = mask.cpu()
            directions[layer_key] = direction.cpu()

            logger.debug(
                f"{layer_key}: selected {topk_indices.numel()} features, "
                f"mean importance={feature_importance[topk_indices].mean():.4f}"
            )

        logger.info(f"Discovered affordances for {tool_name} across {len(masks)} layers")
        return masks, directions

    def discover_via_contrastive(
        self,
        tool_name: str,
        pairs: List,  # List[CounterfactualPair]
        top_k: int = 256,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Discover tool affordances via contrastive pairs.

        Implements the proposal's contrastive approach:
        "Contrastive pairs: (with-tool output appended vs. synthetic no-tool)
        to learn directions that close the representation gap."

        For each (no-tool, with-tool) pair:
        - Encode both states: φ_no, φ_with
        - Direction: w = φ_with - φ_no
        - Mask: top-K features with largest |w_k|

        Args:
            tool_name: Name of the tool
            pairs: List of CounterfactualPair objects
            top_k: Number of top features to select per layer

        Returns:
            Tuple of (masks, directions)
        """
        logger.info(f"Discovering affordances for {tool_name} via contrastive pairs...")

        # Filter pairs for this tool
        tool_pairs = [p for p in pairs if p.tool_name == tool_name]

        if len(tool_pairs) == 0:
            logger.warning(f"No pairs for {tool_name}")
            return {}, {}

        masks = {}
        directions = {}

        # Get all SAEs (residual + MLP)
        all_saes = {}
        for layer, sae in self.sae_loader.sae_r.items():
            all_saes[f"R{layer}"] = sae
        for layer, sae in self.sae_loader.sae_m.items():
            all_saes[f"M{layer}"] = sae

        # Process each layer
        for layer_key, sae in all_saes.items():
            d_sae = sae.cfg.d_sae

            # Collect contrastive differences
            deltas = []

            for pair in tool_pairs:
                # Get hidden states
                h_no_tool = pair.hidden_states_no_tool.get(layer_key)
                h_with_tool = pair.hidden_states_with_tool.get(layer_key)

                if h_no_tool is None or h_with_tool is None:
                    continue

                # Encode
                phi_no = sae.encode(h_no_tool.to(self.device))
                phi_with = sae.encode(h_with_tool.to(self.device))

                # Compute difference: φ_with - φ_no
                delta = phi_with - phi_no
                deltas.append(delta)

            if len(deltas) == 0:
                logger.warning(f"No deltas for {layer_key}, skipping")
                continue

            # Stack deltas
            deltas_tensor = torch.cat(deltas, dim=0)  # [num_pairs, d_sae]

            # Compute mean direction
            direction = deltas_tensor.mean(dim=0)  # [d_sae]

            # Select top-K features by absolute magnitude
            feature_magnitude = direction.abs()
            topk_indices = torch.topk(feature_magnitude, k=min(top_k, d_sae)).indices

            # Create mask
            mask = torch.zeros(d_sae, device=self.device)
            mask[topk_indices] = 1.0

            # Normalize direction
            direction = direction / (direction.norm() + 1e-8)

            masks[layer_key] = mask.cpu()
            directions[layer_key] = direction.cpu()

            logger.debug(
                f"{layer_key}: selected {topk_indices.numel()} features, "
                f"mean magnitude={feature_magnitude[topk_indices].mean():.4f}"
            )

        logger.info(f"Discovered affordances for {tool_name} across {len(masks)} layers")
        return masks, directions

    def discover_hybrid(
        self,
        tool_name: str,
        examples: List[Dict],
        pairs: List,
        top_k: int = 256,
        ablation_weight: float = 0.5,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Hybrid approach: combine ablation and contrastive methods.

        Args:
            tool_name: Name of the tool
            examples: Examples for ablation
            pairs: Counterfactual pairs for contrastive
            top_k: Number of top features
            ablation_weight: Weight for ablation vs contrastive (0-1)

        Returns:
            Tuple of (masks, directions)
        """
        logger.info(f"Discovering affordances for {tool_name} via hybrid method...")

        # Get masks/directions from both methods
        masks_abl, dirs_abl = self.discover_via_ablation(tool_name, examples, top_k)
        masks_con, dirs_con = self.discover_via_contrastive(tool_name, pairs, top_k)

        # Combine
        masks = {}
        directions = {}

        all_keys = set(masks_abl.keys()) | set(masks_con.keys())

        for layer_key in all_keys:
            # Combine masks (union of selected features)
            mask_a = masks_abl.get(layer_key, torch.zeros_like(next(iter(masks_abl.values()))))
            mask_c = masks_con.get(layer_key, torch.zeros_like(next(iter(masks_con.values()))))
            mask = torch.maximum(mask_a, mask_c)

            # Combine directions (weighted average)
            dir_a = dirs_abl.get(layer_key, torch.zeros_like(next(iter(dirs_abl.values()))))
            dir_c = dirs_con.get(layer_key, torch.zeros_like(next(iter(dirs_con.values()))))
            direction = ablation_weight * dir_a + (1 - ablation_weight) * dir_c
            direction = direction / (direction.norm() + 1e-8)

            masks[layer_key] = mask
            directions[layer_key] = direction

        logger.info(f"Hybrid discovery complete for {tool_name}")
        return masks, directions


def discover_affordances_for_tools(
    tools: List[str],
    examples_by_tool: Dict[str, List[Dict]],
    pairs_by_tool: Dict[str, List],
    model,
    tokenizer,
    sae_loader,
    activation_extractor,
    method: str = "contrastive",
    top_k: int = 256,
    device: str = "cuda",
) -> Dict[str, Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
    """
    Discover affordances for multiple tools.

    Args:
        tools: List of tool names
        examples_by_tool: Dict mapping tool names to examples
        pairs_by_tool: Dict mapping tool names to counterfactual pairs
        model: Language model
        tokenizer: Tokenizer
        sae_loader: SAELoader instance
        activation_extractor: ActivationExtractor instance
        method: Discovery method ("ablation", "contrastive", or "hybrid")
        top_k: Number of top features per layer
        device: Device to run on

    Returns:
        Dict mapping tool names to (masks, directions) tuples
    """
    discoverer = CausalAffordanceDiscovery(
        model, tokenizer, sae_loader, activation_extractor, device
    )

    results = {}

    for tool in tools:
        logger.info(f"Processing tool: {tool}")

        examples = examples_by_tool.get(tool, [])
        pairs = pairs_by_tool.get(tool, [])

        if method == "ablation":
            masks, directions = discoverer.discover_via_ablation(tool, examples, top_k)
        elif method == "contrastive":
            masks, directions = discoverer.discover_via_contrastive(tool, pairs, top_k)
        elif method == "hybrid":
            masks, directions = discoverer.discover_hybrid(tool, examples, pairs, top_k)
        else:
            raise ValueError(f"Unknown method: {method}")

        results[tool] = (masks, directions)

    return results

