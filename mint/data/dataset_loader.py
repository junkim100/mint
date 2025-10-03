"""Dataset loading utilities."""

from datasets import load_dataset
from typing import Optional, Dict, Any, Iterator
import logging

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Loads datasets for MINT training and evaluation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize dataset loader.
        
        Args:
            config: Dataset configuration dictionary
        """
        self.config = config or {}
    
    def load_slimpajama(
        self,
        split: str = "train",
        num_samples: int = 10000,
        streaming: bool = True,
    ) -> Iterator:
        """
        Load SlimPajama dataset for activation extraction.
        
        Args:
            split: Dataset split
            num_samples: Number of samples to load
            streaming: Whether to use streaming mode
            
        Returns:
            Dataset iterator
        """
        logger.info(f"Loading SlimPajama dataset (split={split}, n={num_samples})")
        
        if streaming:
            dataset = load_dataset(
                "cerebras/SlimPajama-627B",
                split=split,
                streaming=True,
            )
            # Take first num_samples
            dataset = dataset.take(num_samples)
        else:
            dataset = load_dataset(
                "cerebras/SlimPajama-627B",
                split=f"{split}[:{num_samples}]",
            )
        
        return dataset
    
    def load_tau_bench(
        self,
        split: str = "train",
        domains: Optional[list] = None,
    ):
        """
        Load τ-bench dataset.
        
        Args:
            split: Dataset split
            domains: Optional list of domains to filter
            
        Returns:
            Dataset
        """
        logger.info(f"Loading τ-bench dataset (split={split})")
        
        # Note: Update with actual dataset name when available
        try:
            dataset = load_dataset("sierra-research/tau-bench", split=split)
            
            if domains:
                dataset = dataset.filter(lambda x: x.get("domain") in domains)
                logger.info(f"Filtered to domains: {domains}")
            
            return dataset
        except Exception as e:
            logger.warning(f"Failed to load τ-bench: {e}")
            logger.warning("Using placeholder dataset")
            return []
    
    def load_bfcl(
        self,
        version: str = "v4",
        split: str = "train",
        categories: Optional[list] = None,
    ):
        """
        Load BFCL (Berkeley Function Calling Leaderboard) dataset.
        
        Args:
            version: Dataset version
            split: Dataset split
            categories: Optional list of categories to filter
            
        Returns:
            Dataset
        """
        logger.info(f"Loading BFCL {version} dataset (split={split})")
        
        try:
            dataset = load_dataset(
                "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
                split=split,
            )
            
            if categories:
                dataset = dataset.filter(lambda x: x.get("category") in categories)
                logger.info(f"Filtered to categories: {categories}")
            
            return dataset
        except Exception as e:
            logger.warning(f"Failed to load BFCL: {e}")
            logger.warning("Using placeholder dataset")
            return []
    
    def load_gaia(
        self,
        split: str = "validation",
        levels: Optional[list] = None,
    ):
        """
        Load GAIA dataset.
        
        Args:
            split: Dataset split
            levels: Optional list of difficulty levels to filter
            
        Returns:
            Dataset
        """
        logger.info(f"Loading GAIA dataset (split={split})")
        
        try:
            dataset = load_dataset("gaia-benchmark/GAIA", split=split)
            
            if levels:
                dataset = dataset.filter(lambda x: x.get("level") in levels)
                logger.info(f"Filtered to levels: {levels}")
            
            return dataset
        except Exception as e:
            logger.warning(f"Failed to load GAIA: {e}")
            logger.warning("Using placeholder dataset")
            return []

