# MINT: Mechanistic Integration for Tool-Use

A compact, production-ready pipeline to:
- Generate counterfactual pairs
- Train the Mechanistic Tool Editor (MTE)
- Evaluate with evalchemy (installed from this repository)

## Requirements
- CUDA-enabled GPUs and compatible drivers
- Conda environment created from `environment.yml`
- Use `python` from your active environment (no hardcoded paths)

## Installation
```
conda env create -f environment.yml
conda activate mint
pip install -U git+https://github.com/junkim100/evalchemy.git
```

## Pipeline
1) Prepare counterfactual pairs (multi-GPU, layer-sharded)
```
bash scripts/prepare_pairs_multigpu.sh configs/default.yaml
```
- GPUs are read from `gpu.CUDA_VISIBLE_DEVICES` (e.g., "0,4,7"). One worker is launched per listed index.
- Logs: `logs/prepare_pairs_gpu{ID}.log`

2) Merge pairs (if needed)
```
python scripts/merge_pairs.py --config_path configs/default.yaml --mode layer
```

3) Train MTE
```
python scripts/train_mte.py --config_path configs/default.yaml
```

4) Evaluate with evalchemy
```
python scripts/run_eval.py
```
- No CLI args; everything is driven by `configs/default.yaml`.

Example config is provided at configs/default.yaml — edit it as needed and use it for all commands above.



## License
MIT — see LICENSE

