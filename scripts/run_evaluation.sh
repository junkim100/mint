#!/bin/bash
################################################################################
# MINT Evaluation Script
################################################################################
# Evaluates models on benchmarks using evalchemy's built-in evaluation system.
#
# Supported benchmarks:
#   - TauBench: Multi-turn tool-agent-user interactions
#   - AIME24: Advanced math problems
#   - IFEval: Instruction following
#   - HumanEval: Code generation
#
# Usage:
#   bash scripts/run_evaluation.sh [MODEL] [BENCHMARKS]
#
# Examples:
#   bash scripts/run_evaluation.sh gpt-4o TauBench
#   bash scripts/run_evaluation.sh gpt-4o "TauBench,AIME24"
################################################################################

set -e

# Configuration
MODEL="${1:-gpt-4o}"
BENCHMARKS="${2:-TauBench}"
OUTPUT_DIR="/app/mint/results/benchmarks"

echo "================================================================================"
echo "MINT Benchmark Evaluation"
echo "================================================================================"
echo "Model: $MODEL"
echo "Benchmarks: $BENCHMARKS"
echo ""

mkdir -p "$OUTPUT_DIR"

# Check API key for TauBench
if [[ "$BENCHMARKS" == *"TauBench"* ]] && [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set (required for TauBench)"
    exit 1
fi

# Parse benchmarks
IFS=',' read -ra BENCHMARK_ARRAY <<< "$BENCHMARKS"

for BENCHMARK in "${BENCHMARK_ARRAY[@]}"; do
    echo "================================================================================"
    echo "Evaluating: $BENCHMARK"
    echo "================================================================================"
    
    # Determine model type
    if [[ "$MODEL" == "gpt-"* ]]; then
        MODEL_TYPE="openai"
        MODEL_ARGS="model=$MODEL"
    elif [[ "$MODEL" == "claude-"* ]]; then
        MODEL_TYPE="anthropic"
        MODEL_ARGS="model=$MODEL"
    else
        MODEL_TYPE="hf"
        MODEL_ARGS="pretrained=$MODEL"
    fi
    
    # Add benchmark-specific args
    case "$BENCHMARK" in
        TauBench)
            MODEL_ARGS="$MODEL_ARGS,env=retail,start_index=0,end_index=20"
            ;;
        *)
            MODEL_ARGS="$MODEL_ARGS,max_tokens=4096"
            ;;
    esac
    
    # Run evaluation
    python -m eval.eval \
        --model "$MODEL_TYPE" \
        --tasks "$BENCHMARK" \
        --model_args "$MODEL_ARGS" \
        --output_path "$OUTPUT_DIR/${BENCHMARK}_${MODEL//\//_}.json" \
        --log_samples
    
    echo "✓ $BENCHMARK complete"
    echo ""
done

echo "================================================================================"
echo "Results saved to: $OUTPUT_DIR/"
echo "================================================================================"
ls -lh "$OUTPUT_DIR/"

