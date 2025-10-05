# MTE Training

This document explains how the Mechanistic Tool Editor (MTE) is trained using scripts/train_mte.py and the referenced mint/ modules.

## 1. Main flow of scripts/train_mte.py

CLI (via Fire):
- config_path: YAML config path (default: configs/default.yaml)

High-level steps:
1. Load YAML config with load_config.
2. Log basic run params (pairs_dir, layers, lr, batch_size, steps, save_dir).
3. Call train_mte(config) which runs the full training loop and returns the trained model.

Key code (abridged):
```
config = load_config(config_path)
logger.info(f"Pairs directory: {config.data.pairs_dir}")
logger.info(f"Layers: {config.mte.layers}")
logger.info(f"Learning rate: {config.mte.lr}")
...
mte = train_mte(config)
```

## 2. mint/train/train_mte.py — the training loop

Purpose
- Implement the full MTE training loop: load pairs, create DataLoader, initialize the MTE, run forward/backward passes, compute loss, and save checkpoints.

Key function: train_mte(config, pairs_dir=None, checkpoint_path=None) -> MechanisticToolEditor
- Seed: set_seed(config.seed)
- Device: "cuda" if available else "cpu" (single-device training)
- Load pairs: load_pairs(pairs_dir or config.data.pairs_dir)
- Build DataLoader: create_dataloader(pairs, layers=config.mte.layers, batch_size=config.mte.batch_size, shuffle=True)
- Initialize MTE:
  - feature_dim = config.mte.hidden_dim * config.mte.hidden_expansion (e.g., 4096*8=32768)
  - From checkpoint if provided; else new MechanisticToolEditor(layers, feature_dim, edit_norm_cap)
- Optimizer: AdamW over all MTE params with lr=config.mte.lr
- Steps: loop until global_step reaches config.mte.steps
  - For each batch:
    - Move tensors to device: phi_no_tool, h_with_tool (dicts keyed by layer_id)
    - Forward: phi_edited = mte(phi_no)
    - Loss: compute_mte_loss(phi_edited, h_with_tool, decode_fn, l2_penalty=config.mte.l2_penalty)
      * decode_fn(layer_id, phi) -> decode_features(...) using SAEs to reconstruct hidden states
    - Backprop: optimizer.zero_grad(); loss.backward(); optimizer.step()
    - Logging: tqdm postfix with metrics; increment global_step
    - Checkpoint: every 500 steps and at the end
- Final: save mte_final.pt and return model

Key code (abridged):
```
feature_dim = config.mte.hidden_dim * config.mte.hidden_expansion
mte = MechanisticToolEditor(layers=config.mte.layers,
                            feature_dim=feature_dim,
                            edit_norm_cap=config.mte.edit_norm_cap).to(device)
optimizer = torch.optim.AdamW(mte.parameters(), lr=config.mte.lr)
...
phi_edited = mte(phi_no)
loss, metrics = compute_mte_loss(phi_edited, h_with_tool, decode_fn, l2_penalty=config.mte.l2_penalty)
loss.backward(); optimizer.step()
...
mte.save(str(save_dir / f"mte_step_{global_step}.pt"))
```

Data flow
- Input: pairs_dir → load_pairs → List[CounterfactualPair]
- DataLoader: batches contain dicts of per-layer tensors for phi_no_tool, phi_with_tool, h_no_tool, h_with_tool
- Forward: MTE maps phi_no_tool → phi_edited per layer
- Decode: decode_features(layer_id, phi_edited[layer]) → reconstructed hidden states
- Loss: MSE(reconstructed, h_with_tool[layer]) + l2_penalty * mean(phi_edited^2)
- Output: saved checkpoints and final trained model

Side effects
- File I/O: reads pairs (.pt files); writes checkpoints to config.mte.save_dir
- GPU: runs all training on single CUDA device if available; uses SAE decode during loss

## 3. mint/models/mte.py — the MTE model and loss

Purpose
- MechanisticToolEditor applies learned edits to SAE feature representations at selected layers, with optional norm capping for stability.
- compute_mte_loss reconstructs hidden states from edited features and measures MSE to the with-tool targets, plus L2 regularization.

MechanisticToolEditor(layers, feature_dim, hidden_dim=512, edit_norm_cap=0.5, dropout=0.1)
- Per-layer editors: nn.Sequential(Linear→LayerNorm→ReLU→Dropout)×2 → Linear(feature_dim)
- forward(phi_no_tool: Dict[int, Tensor]) -> Dict[int, Tensor]
  - For each layer in input:
    - delta = editor[layer](phi)
    - If edit_norm_cap is set: scale delta to respect max L2 norm along feature dim
    - phi_edited[layer] = phi + delta
- get_edit_magnitudes(...) returns per-token edit norms per layer (for analysis)
- save(path)/load(path): checkpointing utilities (state_dict + meta)

compute_mte_loss(phi_edited, h_with_tool, decode_fn, l2_penalty=0.01)
- For each layer in phi_edited ∩ h_with_tool:
  - h_reconstructed = decode_fn(layer_id, phi_edit)
  - mse_loss += MSE(h_reconstructed, h_target)
  - l2_reg += mean(phi_edit^2)
- Average losses over layers; total_loss = mse_loss + l2_penalty * l2_reg
- Returns (total_loss, metrics: total_loss, mse_loss, l2_reg)

Key code (abridged):
```
delta = self.editors[str(layer_id)](phi)
if self.edit_norm_cap is not None:
    delta_norm = delta.norm(dim=-1, keepdim=True)
    scale = (delta_norm / (self.edit_norm_cap + 1e-6)).clamp(min=1.0)
    delta = delta / scale
phi_edited[layer_id] = phi + delta
...
h_reconstructed = decode_fn(layer_id, phi_edit)
mse_loss += F.mse_loss(h_reconstructed, h_target)
l2_reg += (phi_edit ** 2).mean()
```

## 4. mint/train/datasets.py — dataset and DataLoader

Purpose
- Adapt List[CounterfactualPair] into batched tensors for training.

PairDataset(pairs, layers)
- __getitem__(idx): returns a dict filtering each pair to the selected layers:
  - phi_no_tool, phi_with_tool, h_no_tool, h_with_tool (each: Dict[layer_id → Tensor])
  - tool_label (unused by the core loss, but available)

collate_pairs(batch)
- Gathers all items in the batch and stacks tensors per layer along batch dimension
- Returns a dict of dicts per layer: {phi_no_tool, phi_with_tool, h_no_tool, h_with_tool}
- Assumes same sequence length within batch (no padding logic here)

create_dataloader(pairs, layers, batch_size, shuffle=True, num_workers=0)
- Wraps PairDataset with a DataLoader and custom collate function

Data flow
- Input: List[CounterfactualPair]
- Output: batches with per-layer tensors suitable for MTE forward and loss

## 5. SAE usage during training — decode path

- During loss computation, the edited features are decoded back to hidden-state space using mint.models.sae_loader.decode_features:
  - decode_features(layer_id, features, hidden_dim=4096, expansion_factor=8, checkpoint_path=None)
  - Internally loads/uses the appropriate SAE (cached per (layer, checkpoint)).
- In train_mte, decode_fn is defined inline to capture config.mte.hidden_dim/hidden_expansion and passed into compute_mte_loss.

Key code (abridged):
```
def decode_fn(layer_id, phi):
    return decode_features(layer_id, phi, config.mte.hidden_dim, config.mte.hidden_expansion)
loss, metrics = compute_mte_loss(phi_edited, h_with_tool, decode_fn, l2_penalty=config.mte.l2_penalty)
```

## 6. Configuration knobs used by training

From config.mte:
- layers: which transformer layers to train on
- hidden_dim: base hidden size (4096 for Llama-3.1-8B)
- hidden_expansion: SAE expansion factor (8 for 32768 features)
- edit_norm_cap: cap on edit L2 norm (stability)
- lr: learning rate for AdamW
- batch_size: batch size for DataLoader
- steps: total training steps (global steps)
- save_dir: where to save checkpoints
- l2_penalty: weight on feature magnitude regularization

From config.data:
- pairs_dir: path to the generated pairs

From config.seed:
- seed: deterministic setup for training

## 7. Output artifacts and how to load

- Checkpoints saved to config.mte.save_dir:
  - mte_step_{global_step}.pt every 500 steps and at final step
  - mte_final.pt at training end
- Checkpoint contents include:
  - state_dict, layers, feature_dim, edit_norm_cap
- Loading:
  - MechanisticToolEditor.load(path, device="cuda") returns a ready-to-use model on device

## 8. End-to-end data flow recap

pairs_dir → load_pairs → DataLoader(batch of per-layer tensors)
→ MTE forward: phi_no_tool → phi_edited
→ SAE decode: decode(phi_edited[layer]) → reconstructed hidden states
→ Loss: MSE(reconstructed, h_with_tool) + l2_penalty * mean(phi_edited^2)
→ Optimizer step → periodic checkpoint save → final model save

## 9. Practical notes

- Training device: single GPU if available ("cuda"), else CPU.
- Memory: batches assume uniform sequence length; if sequence lengths vary widely, consider adding padding logic in collate or trimming strategy.
- SAEs: decode path depends on available SAE checkpoints (loaded and cached by sae_loader). Ensure checkpoint paths are configured for all trained layers to reconstruct accurately.
- Edit norm cap: stabilizes training by preventing very large edits; tune via config.mte.edit_norm_cap.

