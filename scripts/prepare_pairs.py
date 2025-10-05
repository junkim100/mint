#!/usr/bin/env python
"""Script to prepare counterfactual pairs from ToolBench."""

from fire import Fire
from tqdm import tqdm
import torch
from pathlib import Path

from mint.config import load_config
from mint.data.tau_bench_adapter import TauBenchAdapter
from mint.data.counterfactual_pairs import create_pair
from mint.models.sae_loader import encode_features
from mint.logging_utils import setup_logger

logger = setup_logger(__name__)


def main(
    config_path: str = "configs/default.yaml",
    gpu_id: int = 0,             # physical CUDA device id
    worker_rank: int = 0,        # index within the worker set (0..num_gpus-1)
    num_gpus: int = 1,
    layer_shard: bool = True,
):
    """Prepare counterfactual pairs from ToolBench dataset.

    Args:
        config_path: Path to MINT config file
        gpu_id: Physical CUDA device id to run on (e.g., 0,4,7)
        worker_rank: Index within the worker set (0..num_gpus-1) used for sharding
        num_gpus: Total number of workers being used (length of GPU list from config)
        layer_shard: If True, assign one layer per worker and process all scenarios
    """
    logger.info("=" * 80)
    mode_str = "Layer-sharded" if layer_shard else f"Scenario-sharded (GPU {gpu_id}/{num_gpus})"
    logger.info(f"MINT: Preparing Counterfactual Pairs from ToolBench [{mode_str}]")
    logger.info("=" * 80)

    # Load config
    config = load_config(config_path)
    logger.info(f"Loaded config from {config_path}")
    logger.info(f"Run name: {config.run_name}")
    # Resolve model id for logging (prefer pretrained when set)
    model_id = config.model.pretrained or config.model.name
    logger.info(f"Model: {model_id}")
    logger.info(f"Layers: {config.mte.layers}")
    logger.info(f"Tools: {config.data.tools if config.data.tools else 'All tools'}")
    logger.info(f"Max pairs: {config.data.max_pairs}")

    # Create adapter with specific GPU
    adapter = TauBenchAdapter(config, device=f"cuda:{gpu_id}")

    # Determine output directory
    output_dir = Path(config.data.pairs_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if layer_shard:
        layers = config.mte.layers
        if worker_rank >= len(layers):
            logger.info(f"Worker {worker_rank} has no assigned layer (only {len(layers)} layers). Exiting.")
            return
        assigned_layer = layers[worker_rank]
        logger.info(f"GPU {gpu_id} (rank {worker_rank}) assigned to layer {assigned_layer}")

        # Load all scenarios (no scenario sharding in layer-shard mode)
        scenarios = adapter._load_toolbench_scenarios(
            tools=config.data.tools,
            max_pairs=config.data.max_pairs,
            gpu_id=0,
            num_gpus=1,
        )
        logger.info(f"Processing {len(scenarios)} scenarios on GPU {gpu_id} for layer {assigned_layer}")

        # Output directory per layer
        layer_dir = Path(config.data.pairs_dir) / f"layer_{assigned_layer}"
        layer_dir.mkdir(parents=True, exist_ok=True)

        # Ensure model is loaded once
        adapter._load_model()

        # Iterate and save per scenario with progress bar
        pbar = tqdm(total=len(scenarios), desc=f"GPU {gpu_id} L{assigned_layer}")
        for idx, scenario in enumerate(scenarios):
            try:
                # Build encode fn for this layer
                def _enc_fn(layer_id, h):
                    return encode_features(
                        layer_id,
                        h,
                        config.mte.hidden_dim,
                        config.mte.hidden_expansion,
                        checkpoint_path=config.mte.sae_checkpoints.get(str(layer_id)),
                    )

                # Create pair for the single assigned layer
                phi_no, h_no = create_pair(
                    model=adapter.model,
                    tokenizer=adapter.tokenizer,
                    query=scenario["query"],
                    tool_output=None,
                    tool_label=scenario["tool"],
                    layers=[assigned_layer],
                    encode_fn=_enc_fn,
                    device=adapter.model.device,
                )
                torch.cuda.empty_cache()
                phi_with, h_with = create_pair(
                    model=adapter.model,
                    tokenizer=adapter.tokenizer,
                    query=scenario["query"],
                    tool_output=scenario["tool_output"],
                    tool_label=scenario["tool"],
                    layers=[assigned_layer],
                    encode_fn=_enc_fn,
                    device=adapter.model.device,
                )
                torch.cuda.empty_cache()

                # Ensure target directory still exists (robust to external cleanup)
                layer_dir.mkdir(parents=True, exist_ok=True)
                # Save immediately as a per-layer file using global index
                out_path = layer_dir / f"pair_{idx:06d}.pt"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "phi_no_tool": phi_no,
                        "phi_with_tool": phi_with,
                        "h_no_tool": h_no,
                        "h_with_tool": h_with,
                        "tool_label": scenario["tool"],
                        "metadata": {
                            "query": scenario["query"],
                            "tool_output": scenario["tool_output"][:500],
                            "domain": scenario["domain"],
                        },
                    },
                    out_path,
                )
                pbar.update(1)
            except Exception as e:
                logger.error(f"Failed on idx {idx}: {e}")
                torch.cuda.empty_cache()
                continue
        pbar.close()
        logger.info(f"Layer {assigned_layer} complete on GPU {gpu_id}")
    else:
        # Scenario-sharded mode (previous behavior) with streaming save
        logger.info(f"Worker rank: {worker_rank} (processing every {num_gpus}th scenario starting from {worker_rank})")
        logger.info("Generating counterfactual pairs from ToolBench...")
        pair_counter = 0
        out_dir = Path(f"{config.data.pairs_dir}_rank{worker_rank}")
        out_dir.mkdir(parents=True, exist_ok=True)

        # We don't know total upfront easily; load scenarios to count
        scenarios = adapter._load_toolbench_scenarios(config.data.tools, config.data.max_pairs, worker_rank, num_gpus)
        pbar = tqdm(total=len(scenarios), desc=f"rank {worker_rank}")

        for pair in adapter.iter_pairs(
            max_pairs=config.data.max_pairs,
            tools=config.data.tools,
            gpu_id=worker_rank,
            num_gpus=num_gpus,
        ):
            out_path = out_dir / f"pair_{pair_counter:06d}.pt"
            torch.save(
                {
                    "phi_no_tool": pair.phi_no_tool,
                    "phi_with_tool": pair.phi_with_tool,
                    "h_no_tool": pair.h_no_tool,
                    "h_with_tool": pair.h_with_tool,
                    "tool_label": pair.tool_label,
                    "metadata": pair.metadata,
                },
                out_path,
            )
            pair_counter += 1
            pbar.update(1)
        pbar.close()

    logger.info("=" * 80)
    logger.info(f"Pair preparation complete for GPU {gpu_id}!")
    logger.info("=" * 80)


if __name__ == "__main__":
    Fire(main)
