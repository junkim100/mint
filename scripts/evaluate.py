#!/usr/bin/env python3
"""
MINT Evaluation Script

Evaluates models on benchmarks using evalchemy's built-in evaluation system.
Reads configuration from benchmarks.yaml.

Usage:
    # Run with preset
    python scripts/evaluate.py --preset quick_test

    # Run specific benchmarks
    python scripts/evaluate.py --model gpt-4o --benchmarks TauBench,AIME24

    # Run single benchmark
    python scripts/evaluate.py --model gpt-4o --benchmarks TauBench
"""

import os
import sys
import yaml
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional
import fire

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BenchmarkEvaluator:
    """Evaluator for running benchmarks using evalchemy."""

    def __init__(self, config_path: str = "configs/benchmarks.yaml"):
        """Initialize evaluator with config file."""
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)

        self.benchmarks = self.config.get('benchmarks', {})
        self.models = self.config.get('models', {})
        self.presets = self.config.get('presets', {})

    def check_prerequisites(self, benchmark: str, model: str):
        """Check if prerequisites are met for benchmark and model."""
        # Check benchmark API key requirements
        if benchmark in self.benchmarks:
            bench_config = self.benchmarks[benchmark]
            if bench_config.get('requires_api_key') and benchmark == 'TauBench':
                if not os.getenv('OPENAI_API_KEY'):
                    raise ValueError(
                        f"{benchmark} requires OPENAI_API_KEY environment variable. "
                        "Set it with: export OPENAI_API_KEY=your_key_here"
                    )

        # Check model API key requirements
        if model in self.models:
            model_config = self.models[model]
            provider = model_config.get('provider')

            if model_config.get('requires_api_key'):
                if provider == 'openai' and not os.getenv('OPENAI_API_KEY'):
                    raise ValueError("OPENAI_API_KEY not set")
                elif provider == 'anthropic' and not os.getenv('ANTHROPIC_API_KEY'):
                    raise ValueError("ANTHROPIC_API_KEY not set")

    def build_model_args(self, model: str, benchmark: str, override_args: Optional[Dict] = None) -> str:
        """Build model_args string for evalchemy."""
        args = {}

        # Get model provider
        if model in self.models:
            provider = self.models[model].get('provider', 'openai')
        else:
            # Infer provider from model name
            if model.startswith('gpt-') or model.startswith('o1-'):
                provider = 'openai'
            elif model.startswith('claude-'):
                provider = 'anthropic'
            elif model.startswith('gemini-'):
                provider = 'google'
            else:
                provider = 'hf'

        # Add model name
        if provider in ['openai', 'anthropic', 'google']:
            args['model'] = model
        else:
            args['pretrained'] = model

        # Add benchmark-specific args
        if benchmark in self.benchmarks:
            bench_config = self.benchmarks[benchmark]
            default_args = bench_config.get('default_args', {})
            args.update(default_args)

        # Apply overrides
        if override_args:
            args.update(override_args)

        # Convert to string
        return ','.join(f"{k}={v}" for k, v in args.items())

    def get_model_type(self, model: str) -> str:
        """Get model type for evalchemy."""
        if model in self.models:
            provider = self.models[model].get('provider')
            if provider == 'openai':
                return 'openai-chat-completions'
            elif provider == 'anthropic':
                return 'anthropic-chat-completions'
            elif provider == 'google':
                return 'google'
            else:
                return 'hf'

        # Infer from model name
        if model.startswith('gpt-') or model.startswith('o1-'):
            return 'openai-chat-completions'
        elif model.startswith('claude-'):
            return 'anthropic-chat-completions'
        elif model.startswith('gemini-'):
            return 'google'
        else:
            return 'hf'

    def run_benchmark(
        self,
        model: str,
        benchmark: str,
        output_dir: str = "/app/mint/results/benchmarks",
        override_args: Optional[Dict] = None,
    ):
        """Run a single benchmark evaluation."""
        logger.info(f"Running {benchmark} on {model}")

        # Check prerequisites
        self.check_prerequisites(benchmark, model)

        # Build command
        model_type = self.get_model_type(model)
        model_args = self.build_model_args(model, benchmark, override_args)
        output_file = Path(output_dir) / f"{benchmark}_{model.replace('/', '_')}.json"

        cmd = [
            'python', '-m', 'eval.eval',
            '--model', model_type,
            '--tasks', benchmark,
            '--model_args', model_args,
            '--output_path', str(output_file),
            '--log_samples',
            '--verbosity', 'INFO',
        ]

        logger.info(f"Command: {' '.join(cmd)}")

        # Run evaluation
        result = subprocess.run(cmd, capture_output=False, text=True)

        if result.returncode != 0:
            logger.error(f"Evaluation failed with return code {result.returncode}")
            return False

        logger.info(f"✓ {benchmark} complete. Results: {output_file}")
        return True

    def run_preset(self, preset_name: str, output_dir: str = "/app/mint/results/benchmarks"):
        """Run a preset evaluation configuration."""
        if preset_name not in self.presets:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(self.presets.keys())}")

        preset = self.presets[preset_name]
        logger.info(f"Running preset: {preset_name}")
        logger.info(f"Description: {preset.get('description', 'N/A')}")

        model = preset['model']
        benchmarks = preset['benchmarks']
        override_args = preset.get('override_args', {})

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Run each benchmark
        for benchmark in benchmarks:
            bench_overrides = override_args.get(benchmark, {})
            self.run_benchmark(model, benchmark, output_dir, bench_overrides)

    def run(
        self,
        model: Optional[str] = None,
        benchmarks: Optional[str] = None,
        preset: Optional[str] = None,
        output_dir: str = "/app/mint/results/benchmarks",
    ):
        """
        Run benchmark evaluation.

        Args:
            model: Model name (e.g., 'gpt-4o', 'claude-3-5-sonnet-20240620')
            benchmarks: Comma-separated list of benchmarks (e.g., 'TauBench,AIME24')
            preset: Preset name (e.g., 'quick_test', 'standard', 'comprehensive')
            output_dir: Directory to save results
        """
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        if preset:
            # Run preset
            self.run_preset(preset, output_dir)
        elif model and benchmarks:
            # Run specific benchmarks
            benchmark_list = benchmarks.split(',')
            for benchmark in benchmark_list:
                self.run_benchmark(model, benchmark.strip(), output_dir)
        else:
            logger.error("Must specify either --preset or both --model and --benchmarks")
            logger.info("\nAvailable presets:")
            for name, config in self.presets.items():
                logger.info(f"  {name}: {config.get('description', 'N/A')}")
            logger.info("\nAvailable benchmarks:")
            for name, config in self.benchmarks.items():
                logger.info(f"  {name}: {config.get('description', 'N/A')}")
            return


def main(
    model: Optional[str] = None,
    benchmarks: Optional[str] = None,
    preset: Optional[str] = None,
    output_dir: str = "/app/mint/results/benchmarks",
    config: str = "configs/benchmarks.yaml",
):
    """
    Run MINT benchmark evaluation.

    Examples:
        # Run quick test preset
        python scripts/evaluate.py --preset quick_test

        # Run specific benchmarks
        python scripts/evaluate.py --model gpt-4o --benchmarks TauBench,AIME24

        # Run single benchmark
        python scripts/evaluate.py --model gpt-4o --benchmarks TauBench
    """
    evaluator = BenchmarkEvaluator(config)
    evaluator.run(model, benchmarks, preset, output_dir)


if __name__ == '__main__':
    fire.Fire(main)

