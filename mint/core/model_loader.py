"""Model loading utilities for MINT."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ModelLoader:
    """Loads and manages the base LLM (Llama-3.1-8B-Instruct)."""
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        dtype: str = "bfloat16",
        device_map: str = "auto",
        use_cache: bool = True,
        output_hidden_states: bool = True,
        **kwargs
    ):
        """
        Initialize model loader.
        
        Args:
            model_name: HuggingFace model identifier
            dtype: Data type for model weights (bfloat16, float16, float32)
            device_map: Device mapping strategy
            use_cache: Whether to use KV cache
            output_hidden_states: Whether to output hidden states
            **kwargs: Additional arguments for model loading
        """
        self.model_name = model_name
        self.dtype = self._parse_dtype(dtype)
        self.device_map = device_map
        self.use_cache = use_cache
        self.output_hidden_states = output_hidden_states
        self.kwargs = kwargs
        
        self.tokenizer = None
        self.model = None
        
    def _parse_dtype(self, dtype: str) -> torch.dtype:
        """Parse dtype string to torch.dtype."""
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "fp16": torch.float16,
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
        }
        return dtype_map.get(dtype.lower(), torch.bfloat16)
    
    def load(self) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
        """
        Load tokenizer and model.
        
        Returns:
            Tuple of (tokenizer, model)
        """
        logger.info(f"Loading tokenizer from {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True,
            trust_remote_code=True
        )
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Loading model from {self.model_name} with dtype={self.dtype}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map=self.device_map,
            trust_remote_code=True,
            **self.kwargs
        )
        
        # Configure model for hidden state extraction
        self.model.config.use_cache = self.use_cache
        self.model.config.output_hidden_states = self.output_hidden_states
        
        logger.info(f"Model loaded successfully. Device map: {self.model.hf_device_map}")
        
        return self.tokenizer, self.model
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")
        
        return {
            "model_name": self.model_name,
            "num_layers": self.model.config.num_hidden_layers,
            "hidden_size": self.model.config.hidden_size,
            "num_attention_heads": self.model.config.num_attention_heads,
            "vocab_size": self.model.config.vocab_size,
            "dtype": str(self.dtype),
            "device_map": self.model.hf_device_map,
        }
    
    @property
    def num_layers(self) -> int:
        """Get number of transformer layers."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")
        return self.model.config.num_hidden_layers
    
    @property
    def hidden_size(self) -> int:
        """Get hidden size."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")
        return self.model.config.hidden_size

