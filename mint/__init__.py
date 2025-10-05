"""
MINT: Mechanistic Integration for Tool-Use

A framework for learning mechanistic interventions that enable tool use
in language models through SAE feature-space editing.
"""

__version__ = "0.1.0"
__author__ = "Jun Kim"

from mint.config import MintConfig, load_config

__all__ = ["MintConfig", "load_config", "__version__"]
