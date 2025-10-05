# Pair Generation

This document explains how counterfactual pairs are generated from ToolBench using scripts/prepare_pairs.py and the referenced mint/ modules.

## 1. Main flow of scripts/prepare_pairs.py


CLI (via Fire):
- config_path: YAML config path (default: configs/default.yaml)
- gpu_id: physical CUDA device ID this process uses
- worker_rank: logical rank in [0..num_gpus-1]
- num_gpus: total workers being launched
- layer_shard: True = each worker handles one layer; False = each worker handles a subset of scenarios

High-level steps:
1. Load YAML config with load_config.
2. Initialize TauBenchAdapter(config, device=f"cuda:{gpu_id}").
3. Decide sharding mode:
   - Layer-shard (default): this worker owns exactly one layer and processes ALL scenarios for that layer.
   - Scenario-shard: this worker owns an interleaved subset of scenarios and processes ALL layers for them.
4. For each scenario: build SAE encode_fn, run create_pair twice (no-tool, with-tool), save tensors and metadata into .pt files.

Key code (abridged):
```
config = load_config(config_path)
adapter = TauBenchAdapter(config, device=f"cuda:{gpu_id}")
if layer_shard:
    assigned_layer = config.mte.layers[worker_rank]
    scenarios = adapter._load_toolbench_scenarios(..., gpu_id=0, num_gpus=1)
    adapter._load_model()
    for idx, sc in enumerate(scenarios):
        def _enc_fn(layer_id, h):
            return encode_features(layer_id, h, config.mte.hidden_dim,
                                   config.mte.hidden_expansion,
                                   checkpoint_path=config.mte.sae_checkpoints.get(str(layer_id)))
        phi_no, h_no = create_pair(..., tool_output=None, layers=[assigned_layer], encode_fn=_enc_fn, ...)
        phi_w,  h_w  = create_pair(..., tool_output=sc["tool_output"], layers=[assigned_layer], encode_fn=_enc_fn, ...)
        torch.save({...}, layer_dir / f"pair_{idx:06d}.pt")
else:
    for pair in adapter.iter_pairs(..., gpu_id=worker_rank, num_gpus=num_gpus):
        torch.save({...}, out_dir / f"pair_{pair_counter:06d}.pt")
```

## 2. mint/config.py — configuration dataclasses and loader

Purpose
- Encapsulate run configuration for model, MTE, data, eval, GPU, logging.
- Provide load_config(path) to parse YAML into MintConfig.

Key dataclasses/fields
- ModelConfig: provider ("hf"), name/pretrained/tokenizer, dtype (bfloat16), etc.
- MTEConfig: layers, hidden_dim=4096, hidden_expansion=8, sae_checkpoints (str layer_id → path).
- DataConfig: pairs_dir, max_pairs, tools.
- EvalConfig, GPUConfig, LoggingConfig.

Implementation details
- MTEConfig.__post_init__ validates layer range [0..31] and feature_dim = 4096*8 for Llama‑3.1‑8B.
- load_config reads YAML and builds MintConfig.from_dict.

Data flow
- prepare_pairs.py uses: model.pretrained/name, model.tokenizer, model.dtype; mte.layers; mte.sae_checkpoints; data.max_pairs; data.tools; data.pairs_dir.

## 3. mint/data/tau_bench_adapter.py — ToolBench adapter

Purpose
- Load ToolBench dataset and the base model.
- Extract scenarios from conversations: user query, tool label, tool_output.
- Provide two consumption patterns:
  a) _load_toolbench_scenarios(...) → list of scenarios (for layer-shard mode)
  b) iter_pairs(...) → yields CounterfactualPair objects (for scenario-shard mode)

Key methods
- __init__(config, device): stores MintConfig, device string; defers model load.
- _load_model():
  - Loads AutoTokenizer and AutoModelForCausalLM with dtype from config.model.
  - Moves model to specified cuda device and sets eval().
- _load_toolbench_data(): reads JSON at data/toolbench/data/toolllama_G123_dfs_train.json and caches it.
- _extract_tool_info(conversation): parses first tool name from the system message line following "You have access of the following tools:"; returns {tool, category}.
- _extract_query_and_response(conversation):
  - Query: conversation[1] if from == 'user'.
  - First function turn: if JSON parses and contains 'response', uses it; if JSON['error'] is truthy, skips; else uses raw value.
- _load_toolbench_scenarios(tools, max_pairs, gpu_id, num_gpus):
  - Shuffles indices (seed 42), interleaves by scenario_count % num_gpus == gpu_id, caps at max_pairs // num_gpus.
  - Applies tool filter if provided.
  - Builds scenario dicts: {query, tool, tool_output, domain=tool}.
- iter_pairs(max_pairs, tools, gpu_id, num_gpus, layers_override):
  - Calls _load_model(), fetches scenarios via _load_toolbench_scenarios().
  - For each scenario: create_pair for no-tool and with-tool across layers; yields CounterfactualPair.

Data flow
- Inputs: MintConfig; conversations from ToolBench file.
- Outputs: list of scenarios or yielded CounterfactualPair instances.

Side effects
- I/O: reads ToolBench JSON once.
- GPU: loads the HF model on a cuda device; runs inference to get hidden states.

Key excerpts
```
# Query/response extraction
if len(conversation) > 1 and conversation[1]['from'] == 'user':
    query = conversation[1]['value'].strip()
for turn in conversation:
    if turn['from'] == 'function':
        try:
            j = json.loads(turn['value'])
            if 'response' in j: tool_output = str(j['response'])
            elif 'error' in j and j['error']: return None
        except json.JSONDecodeError:
            tool_output = turn['value']
        break
```

## 4. mint/data/counterfactual_pairs.py — create_pair and helpers

Purpose
- Generate no-tool vs with-tool features for requested layers using the HF model and SAE encoder.

Key function: create_pair(model, tokenizer, query, tool_output, tool_label, layers, encode_fn, device)
- Prompt:
  - No-tool: prompt = query
  - With-tool: prompt = f"{query}\n\nTool output: {tool_output}"
- Inference: model(..., output_hidden_states=True) → hidden_states tuple (len = num_layers + 1; index 0 is embeddings).
- For each layer_id in layers: h = hidden_states[layer_id + 1]; phi = encode_fn(layer_id, h).
- Move phi and h to CPU immediately; clear_sae_cache(); torch.cuda.empty_cache() per layer.
- Returns two dicts on CPU: phi_dict, h_dict.

Other helpers: save_pairs, load_pairs, filter_pairs_by_tool, get_pair_statistics.

Data flow
- Input: model, tokenizer, text prompt; encode_fn.
- Output: dicts mapping layer_id → CPU tensors.

Side effects
- GPU inference for base model; SAE calls.
- Frequent CUDA cache clears to minimize OOM.

Key excerpt
```
prompt = query if tool_output is None else f"{query}\n\nTool output: {tool_output}"
outs = model(**tokenizer(prompt, return_tensors="pt").to(device), output_hidden_states=True)
h = outs.hidden_states[layer_id + 1]
phi = encode_fn(layer_id, h)
```

## 5. mint/models/sae_loader.py — SAE loading and encoding

Purpose
- Load Sparse Autoencoders from safetensors per layer; cache handles; provide encode/decode and cache clearing.

Key APIs
- load_sae(layer_id, hidden_dim=4096, expansion_factor=8, checkpoint_path, device): returns SAEHandle.
- encode_features(layer_id, hidden_states, hidden_dim, expansion_factor, checkpoint_path): loads/uses SAE and returns phi.
- clear_sae_cache(): moves loaded modules to CPU and clears CUDA cache.

Implementation details
- RealSAE: nn.Linear encoder/decoder; encode_only applies ReLU; weights loaded via safetensors and downcast to bfloat16.
- _SAE_CACHE keyed by (layer_id, checkpoint_path) avoids repeated loads.

Data flow
- Input: hidden_states [B, T, 4096]. Output: features [B, T, 32768] for Llama‑3.1‑8B (8× expansion).

Key excerpt
```
def encode_features(layer_id, hidden_states, hidden_dim=4096, expansion_factor=8, checkpoint_path=None):
    sae = load_sae(layer_id, hidden_dim, expansion_factor, checkpoint_path, device=hidden_states.device)
    return sae.encode(hidden_states)
```

## 6. mint/logging_utils.py — logging

Purpose
- Set up module loggers with optional rich console formatting and file logging.

Side effects
- Console/file logging handlers only.

## 7. Multi-GPU and sharding logic

Parameters
- gpu_id: physical CUDA device ID for this process; model and SAEs are placed here.
- worker_rank: logical rank in [0..num_gpus-1].
- num_gpus: number of workers.
- layer_shard:
  - True: process handles exactly one layer (layers[worker_rank]); processes ALL scenarios; writes per-layer files under data/pairs/layer_{L}.
  - False: process handles every num_gpus-th scenario (scenario_count % num_gpus == worker_rank); processes ALL layers per scenario; writes per-scenario files under data/pairs_rank{worker_rank}.

## 8. Output format (.pt files)

Each saved .pt is a dict with:
- phi_no_tool: {layer_id: Tensor} SAE features [1, seq_len, 32768]
- phi_with_tool: {layer_id: Tensor} SAE features [1, seq_len, 32768]
- h_no_tool: {layer_id: Tensor} hidden states [1, seq_len, 4096]
- h_with_tool: {layer_id: Tensor} hidden states [1, seq_len, 4096]
- tool_label: str (tool name)
- metadata: {
    query: str,
    tool_output: str (truncated to 500 chars at write time),
    domain: str (tool name again)
  }

## 9. ToolBench data usage and filtering note

- We do not execute live tools; we use recorded tool outputs from ToolBench conversations.
- Current adapter skips only explicit JSON cases with truthy 'error' fields; HTML/text error pages can pass.
- In a 2,000-file sample of your generated pairs, ~6.9% of tool outputs looked like error responses. Training signal is still dominated by successful outputs (~93%).
- Optional improvement: add keyword-based filtering and/or choose the first successful function turn; expose as a config toggle if desired.

