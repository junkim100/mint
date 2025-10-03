# MINT: Mechanism-Informed, Risk-Calibrated Tool Policies

A framework for risk-calibrated tool selection in LLMs using mechanistic interpretability and conformal prediction.

## Overview

MINT estimates the counterfactual treatment effect of tools on model hidden states using mechanistically grounded feature editors, then makes decisions under conformal risk control.

**Key Components:**
- **Sparse Autoencoders (SAEs):** Extract 262K interpretable features from Llama-3.1-8B
- **Causal Affordance Discovery:** Identify tool-relevant features via contrastive analysis
- **Mechanistic Tool Editors:** Learn feature-space interventions simulating tool use with E-Lipschitz constraints
- **Value Heads:** Predict utility gains from counterfactual states with Cox/CE success prediction
- **Conformal Calibration:** Provide distribution-free uncertainty bounds (split + online modes)
- **Faithfulness Regularization:** Ablation faithfulness and contrastive causal InfoNCE losses
- **Risk Budget Tracking:** Trajectory-level error control with hierarchical budgets
- **Risk-Calibrated Decisions:** Select tools via argmax(LCB - cost)

## Quick Start

### 1. Setup

```bash
# Clone repository
git clone https://github.com/junkim100/mint.git
cd mint

# Set environment variables
cp .env.example .env
# Edit .env and add: HF_TOKEN=your_huggingface_token

# Start Docker container
docker run -d --gpus all --name mint_container \
  --env-file .env --shm-size=16g \
  junkim100/mint:latest
```

### 2. Run Complete Pipeline (Training + Evaluation)

```bash
# Enter the Docker container
docker exec -it mint_container bash

# Inside container: Set OpenAI API key (required for TauBench evaluation)
export OPENAI_API_KEY=your_key_here

# Inside container: Run complete pipeline (training + evaluation)
cd /app/mint
bash run_mint.sh
```

**Default behavior (no flags):**
- Runs **both** training AND evaluation
- Training: Phase A → B → C → D
- Evaluation: TauBench by default

**What it does:**
1. **Data Preparation** (if needed):
   - Generates counterfactual pairs
   - Discovers affordances
   - (~1 minute)

2. **MINT Training Pipeline:**
   - Phase A: Train Mechanistic Tool Editors (~6 minutes)
   - Phase B: Train Value Heads (~9 minutes)
   - Phase C: Conformal Calibration (<1 minute)
   - Phase D: MINT Inference (<1 minute)

3. **Benchmark Evaluation:**
   - TauBench: Multi-turn tool interactions (~10 minutes, ~$0.60)

**Total time:** ~30 minutes
**Total cost:** ~$0.60 (for TauBench user simulator)

**Note:** On first run, the script will automatically prepare data if not found.

### 3. Alternative: Prepare Data Manually

```bash
# Inside container: Prepare data only
cd /app/mint
python scripts/prepare_data.py

# Quick test with minimal data (50 pairs)
python scripts/prepare_data.py --quick_test

# Custom number of pairs
python scripts/prepare_data.py --num_pairs 500
```

### 4. Alternative: Run Training or Evaluation Separately

```bash
# Enter container first
docker exec -it mint_container bash
cd /app/mint

# Option 1: Training only (skip evaluation)
bash run_mint.sh --training-only

# Option 2: Evaluation only (skip training, use existing checkpoints)
export OPENAI_API_KEY=your_key_here
bash run_mint.sh --evaluation-only

# Option 3: Custom model and benchmarks
bash run_mint.sh --model gpt-4o-mini --benchmarks TauBench,AIME24

# Option 4: Complete pipeline (default - runs both training + evaluation)
bash run_mint.sh
```

**Flag Summary:**
- No flags → Runs training + evaluation (default)
- `--training-only` → Runs only training, skips evaluation
- `--evaluation-only` → Skips training, runs only evaluation
- `--model MODEL` → Specify model to evaluate (default: gpt-4o)
- `--benchmarks LIST` → Comma-separated benchmarks (default: TauBench)

## Project Structure

```
mint/
├── run_mint.sh                  # ⭐ MAIN ENTRY POINT - Run inside container
├── scripts/
│   ├── run_pipeline.sh          # MINT training pipeline (Phase A, B, C, D)
│   ├── run_evaluation.sh        # Bash evaluation wrapper
│   ├── evaluate.py              # Python evaluation script
│   ├── train_editors.py         # Phase A: Editor training
│   ├── train_value_heads.py     # Phase B: Value head training
│   ├── calibrate.py             # Phase C: Conformal calibration
│   └── mint_inference.py        # Phase D: MINT inference
├── configs/
│   ├── benchmarks.yaml          # Benchmark configuration
│   ├── model_config.yaml        # Model configuration
│   └── data_config.yaml         # Data configuration
├── mint/
│   ├── core/                    # Core MINT components
│   ├── data/                    # Data types and loaders
│   ├── inference/               # Inference components
│   ├── tools/                   # Tool-specific implementations
│   └── training/                # Training phases
└── README.md                    # This file
```

## Advanced Features

### Faithfulness Regularization (Phase C)

MINT now includes faithfulness regularization to improve interpretability and robustness:

- **Ablation Faithfulness:** Penalizes decision invariance when important features are ablated
- **Contrastive Causal InfoNCE:** Encourages ΔV̂_u* > ΔV̂_u for optimal tool u*

Enable during training:
```python
from mint.training.phase_c import ConformalCalibrationTrainer

trainer = ConformalCalibrationTrainer(
    value_head=value_head,
    editor=editor,
    use_faithfulness=True,
    lambda_ablation=0.1,
    lambda_contrastive=0.5,
)

# Fine-tune with faithfulness regularization
trainer.train_with_faithfulness(pairs, steps=500)
```

### Online Conformal Prediction

Adaptive calibration for distribution shift:

```python
from mint.inference.online_conformal import AdaptiveConformalPredictor

calibrator = AdaptiveConformalPredictor(alpha=0.1)

# Calibrate with initial data (split conformal)
calibrator.calibrate(predictions, targets)

# Update online as new data arrives
for pred, target in new_data:
    calibrator.update(pred, target)
    lcb = calibrator.get_lcb(pred)

    # Automatically switches to online mode if shift detected
    if calibrator.get_mode() == "online":
        print("Distribution shift detected!")
```

### Risk Budget Tracking

Trajectory-level error control:

```python
from mint.inference.risk_budget import HierarchicalRiskBudget

# Define tool families and budgets
risk_budget = HierarchicalRiskBudget(
    total_budget=0.1,  # 10% error rate
    horizon=100,  # Expected trajectory length
    tool_families={
        "external": ["search", "api"],
        "local": ["calculator", "parser"],
    },
    family_budgets={
        "external": 0.07,  # 70% of budget
        "local": 0.03,     # 30% of budget
    },
)

# Check before tool call
if risk_budget.can_afford(risk=0.05, tool="search"):
    # Make tool call
    risk_budget.spend(risk=0.05, tool="search")
```

### Real Data Collection

Replace synthetic data with real τ-bench pairs:

```bash
# Download τ-bench dataset first
git clone https://github.com/sierra-research/tau-bench
# Extract trajectories to data/taubench/trajectories.json

# Collect real counterfactual pairs
python scripts/collect_taubench_pairs.py \
    --num_pairs 500 \
    --output data/counterfactual_pairs/real_pairs.pt

# Use in training
python scripts/train_editors.py --data_path data/counterfactual_pairs/real_pairs.pt
```

## Evaluation

MINT uses [evalchemy](https://github.com/evalstate/evalchemy)'s built-in benchmark system for evaluation.

### Supported Benchmarks

**Primary:**
- **TauBench:** Multi-turn tool-agent-user interactions (retail/airline domains)

**Additional:**
- **AIME24:** Advanced math problems
- **IFEval:** Instruction following
- **HumanEval:** Code generation
- **MATH500:** Math problem solving

### Adding New Benchmarks

Edit `configs/benchmarks.yaml` to add any evalchemy-supported benchmark:

```yaml
benchmarks:
  YourBenchmark:
    description: "Your benchmark description"
    requires_api_key: false
    default_args:
      max_tokens: 4096
```

Then run:

```bash
python scripts/evaluate.py --model gpt-4o --benchmarks YourBenchmark
```

### Available evalchemy Benchmarks

- AIME24, AIME25, AMC23
- CodeForces, LiveCodeBench, BigCodeBench
- IFEval, MMLUPro, GPQADiamond
- WMT24, ArenaHard, CreativeWriting
- SWEbench, LogicKor, and more

See `/app/evalchemy/eval/chat_benchmarks/` for full list.

## Configuration

### Benchmark Configuration (`configs/benchmarks.yaml`)

```yaml
benchmarks:
  TauBench:
    default_args:
      env: "retail"              # retail or airline
      start_index: 0
      end_index: 20              # Number of tasks
      max_concurrency: 5

presets:
  quick_test:
    benchmarks: ["TauBench"]
    model: "gpt-4o-mini"
    override_args:
      TauBench:
        end_index: 10
```

### Training Configuration

Edit scripts directly or pass arguments:

```bash
# Custom training steps
docker exec mint_container python /app/mint/scripts/train_editors.py \
  --train_steps 1000 \
  --learning_rate 5e-5
```

## Results

### MINT Training Pipeline

After running `scripts/run_pipeline.sh`:

```
/app/mint/checkpoints/
├── editors/              # Mechanistic Tool Editors
├── value_heads/          # Value prediction networks
└── calibrators/          # Conformal calibration quantiles

/app/mint/results/
└── mint/
    └── mint_results.json # MINT inference results
```

### Benchmark Evaluation

After running `scripts/evaluate.py`:

```
/app/mint/results/benchmarks/
├── TauBench_gpt-4o.json
├── AIME24_gpt-4o.json
└── ...
```

View results:

```bash
docker exec mint_container cat /app/mint/results/benchmarks/TauBench_gpt-4o.json | jq '.results'
```

## Key Files Explained

### Training Pipeline

**`scripts/train_editors.py` (Phase A):**
- Trains Mechanistic Tool Editors that learn feature-space interventions
- Formula: `φ̃^(ℓ) = φ^(ℓ) + α_u^(ℓ) ⊙ m_u^(ℓ) ⊙ w_u^(ℓ)`
- Output: Editor checkpoints

**`scripts/train_value_heads.py` (Phase B):**
- Trains value heads that predict utility gains from counterfactual states
- Formula: `ΔV̂_u = g(H̃_t^(u)) - g(H_t)`
- Output: Value head checkpoints

**`scripts/calibrate.py` (Phase C):**
- Conformal calibration for distribution-free uncertainty bounds
- Formula: `LCB_α(ΔV_u) = ΔV̂_u - Q_α({ΔV̂_i - ΔV_i})`
- Output: Calibration quantiles

**`scripts/mint_inference.py` (Phase D):**
- Applies MINT's risk-calibrated tool selection
- Formula: `argmax_u {LCB_α(ΔV_u) - c(u)}`
- Output: MINT's tool selection decisions

### Evaluation

**`scripts/evaluate.py`:**
- Evaluates models on external benchmarks (TauBench, AIME24, etc.)
- Uses evalchemy's built-in benchmark system
- Output: Benchmark metrics (Pass@1, accuracy, etc.)

**Key Difference:**
- `mint_inference.py`: Internal MINT pipeline step (applies MINT decision-making to training data)
- `evaluate.py`: External evaluation on standard benchmarks (measures performance)

## Citation

```bibtex
@article{mint2025,
  title={MINT: Mechanism-Informed, Risk-Calibrated Tool Policies},
  author={Your Name},
  journal={ICML},
  year={2025}
}
```

## License

MIT License

## Acknowledgments

- **Llama-Scope SAEs:** Pretrained sparse autoencoders
- **evalchemy:** Benchmark evaluation framework
- **Glaive:** Function calling dataset
- **TauBench:** Tool-agent-user interaction benchmark

