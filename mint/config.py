"""Configuration management for MINT."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ModelConfig:
    """Base model configuration.

    Supports either `name` or `pretrained` (plus optional `tokenizer`).
    `pretrained`/`tokenizer` are preferred when provided.
    """

    provider: str = "hf"  # "hf" or "vllm"
    name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    pretrained: Optional[str] = None
    tokenizer: Optional[str] = None
    dtype: str = "bfloat16"
    max_new_tokens: int = 512
    device_map: str = "auto"


@dataclass
class MTEConfig:
    """Mechanistic Tool Editor configuration."""

    layers: List[int] = field(default_factory=lambda: [5, 12, 18, 22, 24, 28, 31])
    feature_space: str = "sae"  # "sae" or "residual"
    edit_norm_cap: float = 0.5
    hidden_expansion: int = 8  # SAE expansion factor
    hidden_dim: int = 4096  # Llama-3.1-8B hidden size
    loss: str = "mse"
    lr: float = 1e-4
    batch_size: int = 32
    steps: int = 2000
    save_dir: str = "checkpoints/mte"
    l2_penalty: float = 0.01
    edit_strength: float = 1.0  # Scalar multiplier for edit magnitude
    sae_checkpoints: Dict[str, str] = field(default_factory=dict)  # layer_id -> checkpoint path

    def __post_init__(self):
        """Validate MTE configuration."""
        # Validate layers are in valid range for Llama-3.1-8B (32 layers)
        for layer in self.layers:
            if not (0 <= layer < 32):
                raise ValueError(f"Layer {layer} out of range [0, 32) for Llama-3.1-8B")

        # Validate feature dimension matches SAE expansion
        expected_feature_dim = self.hidden_dim * self.hidden_expansion
        if expected_feature_dim != 4096 * 8:
            raise ValueError(
                f"Feature dim should be {4096 * 8} (4096 * 8) for Llama-3.1-8B, "
                f"got {expected_feature_dim}"
            )


@dataclass
class DataConfig:
    """Data pipeline configuration."""

    pairs_dir: str = "data/pairs"
    max_pairs: int = 10000  # Total number of pairs to generate
    tools: Optional[List[str]] = None  # None = use all tools from ToolBench


@dataclass
class EvalConfig:
    """Evaluation configuration."""

    # Tasks to run with evalchemy; list of task names (e.g., ["tau_bench"])
    task: List[str] = field(default_factory=lambda: ["tau_bench"])
    output_dir: str = "results/eval"
    model_backend: str = "vllm"
    batch_size: str = "auto"
    # Additional model args forwarded to evalchemy's --model_args
    # Example: {tensor_parallel_size: 8, max_model_len: 131072, dtype: bfloat16, ...}
    model_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GPUConfig:
    """GPU configuration."""

    CUDA_VISIBLE_DEVICES: str = "0,1,2,3,4,5,6,7"


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"


@dataclass
class MintConfig:
    """Main MINT configuration."""

    run_name: str = "mint_tau_local"
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    mte: MTEConfig = field(default_factory=MTEConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def __post_init__(self):
        """Validate MINT configuration."""
        # Ensure eval.task is a non-empty list (e.g., ["tau_bench"])
        if not isinstance(self.eval.task, list) or not self.eval.task:
            raise ValueError("eval.task must be a non-empty list of task names (e.g., ['tau_bench']).")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MintConfig":
        """Create config from dictionary."""
        model_cfg = ModelConfig(**config_dict.get("model", {}))
        mte_cfg = MTEConfig(**config_dict.get("mte", {}))
        data_cfg = DataConfig(**config_dict.get("data", {}))
        eval_cfg = EvalConfig(**config_dict.get("eval", {}))
        gpu_cfg = GPUConfig(**config_dict.get("gpu", {}))
        logging_cfg = LoggingConfig(**config_dict.get("logging", {}))

        return cls(
            run_name=config_dict.get("run_name", "mint_tau_local"),
            seed=config_dict.get("seed", 42),
            model=model_cfg,
            mte=mte_cfg,
            data=data_cfg,
            eval=eval_cfg,
            gpu=gpu_cfg,
            logging=logging_cfg,
        )


def load_config(path: str) -> MintConfig:
    """Load configuration from YAML file.

    Args:
        path: Path to YAML config file

    Returns:
        MintConfig instance
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return MintConfig.from_dict(config_dict)
