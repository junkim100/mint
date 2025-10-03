#!/bin/bash
################################################################################
# MINT Complete Pipeline
################################################################################
# Runs the complete MINT pipeline from training to evaluation:
#   1. MINT Training Pipeline (Phase A, B, C, D)
#   2. Benchmark Evaluation (TauBench, AIME24, etc.)
#
# Prerequisites:
#   - Running inside Docker container
#   - Counterfactual pairs data available
#   - Affordances discovered
#   - OPENAI_API_KEY set (for TauBench evaluation)
#
# Usage (from inside container):
#   # Enter container first
#   docker exec -it mint_container bash
#
#   # Then run this script
#   cd /app/mint
#   bash run_mint.sh
#
#   # Or with options
#   bash run_mint.sh --training-only
#   bash run_mint.sh --evaluation-only
#   bash run_mint.sh --benchmarks TauBench,AIME24
#   bash run_mint.sh --model gpt-4o-mini
################################################################################

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Default configuration
RUN_TRAINING=true
RUN_EVALUATION=true
MODEL="gpt-4o"
BENCHMARKS="TauBench"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --training-only)
            RUN_EVALUATION=false
            shift
            ;;
        --evaluation-only)
            RUN_TRAINING=false
            shift
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --benchmarks)
            BENCHMARKS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: bash run_mint.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --training-only       Run only MINT training pipeline"
            echo "  --evaluation-only     Run only benchmark evaluation"
            echo "  --model MODEL         Model to evaluate (default: gpt-4o)"
            echo "  --benchmarks LIST     Comma-separated benchmarks (default: TauBench)"
            echo "  --help                Show this help message"
            echo ""
            echo "Examples:"
            echo "  bash run_mint.sh                                    # Run complete pipeline"
            echo "  bash run_mint.sh --training-only                    # Training only"
            echo "  bash run_mint.sh --evaluation-only                  # Evaluation only"
            echo "  bash run_mint.sh --model gpt-4o-mini                # Use gpt-4o-mini"
            echo "  bash run_mint.sh --benchmarks TauBench,AIME24       # Multiple benchmarks"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}MINT Complete Pipeline${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""

################################################################################
# Check Prerequisites
################################################################################

# Check if data exists
if [ ! -f "/app/mint/data/counterfactual_pairs/pairs.pt" ]; then
    echo -e "${YELLOW}⚠ Data not found${NC}"
    echo ""
    echo "Counterfactual pairs not found at /app/mint/data/counterfactual_pairs/pairs.pt"
    echo ""
    echo "Would you like to prepare the data now? (y/n)"
    read -p "> " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo -e "${GREEN}Preparing data...${NC}"
        echo ""
        python3 /app/mint/scripts/prepare_data.py
        echo ""
    else
        echo ""
        echo "Please prepare data first:"
        echo "  python scripts/prepare_data.py"
        echo ""
        exit 1
    fi
fi

################################################################################
# Phase 1: MINT Training Pipeline
################################################################################

if [ "$RUN_TRAINING" = true ]; then
    echo -e "${GREEN}[1/2] Running MINT Training Pipeline${NC}"
    echo ""

    bash /app/mint/scripts/run_pipeline.sh

    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✓ MINT Training Pipeline Complete${NC}"
        echo ""
    else
        echo ""
        echo -e "${YELLOW}⚠ MINT Training Pipeline Failed${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}[SKIP] MINT Training Pipeline${NC}"
    echo ""
fi

################################################################################
# Phase 2: Benchmark Evaluation
################################################################################

if [ "$RUN_EVALUATION" = true ]; then
    echo -e "${GREEN}[2/2] Running Benchmark Evaluation${NC}"
    echo ""

    # Check for API key if running TauBench
    if [[ "$BENCHMARKS" == *"TauBench"* ]]; then
        if [ -z "$OPENAI_API_KEY" ]; then
            echo -e "${YELLOW}WARNING: OPENAI_API_KEY not set${NC}"
            echo "TauBench requires OpenAI API key for user simulator"
            echo "Set it with: export OPENAI_API_KEY=your_key_here"
            echo ""
            read -p "Continue without TauBench? (y/n) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                # Remove TauBench from benchmarks
                BENCHMARKS=$(echo "$BENCHMARKS" | sed 's/TauBench,//g' | sed 's/,TauBench//g' | sed 's/TauBench//g')
                if [ -z "$BENCHMARKS" ]; then
                    echo "No benchmarks to run. Exiting."
                    exit 0
                fi
            else
                exit 1
            fi
        fi
    fi

    # Run evaluation
    python /app/mint/scripts/evaluate.py \
        --model "$MODEL" \
        --benchmarks "$BENCHMARKS"

    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✓ Benchmark Evaluation Complete${NC}"
        echo ""
    else
        echo ""
        echo -e "${YELLOW}⚠ Benchmark Evaluation Failed${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}[SKIP] Benchmark Evaluation${NC}"
    echo ""
fi

################################################################################
# Summary
################################################################################

echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}Pipeline Complete${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""

if [ "$RUN_TRAINING" = true ]; then
    echo "MINT Training Results:"
    echo "  - Checkpoints: /app/mint/checkpoints/"
    echo "  - Results: /app/mint/results/mint/"
    echo ""
fi

if [ "$RUN_EVALUATION" = true ]; then
    echo "Benchmark Evaluation Results:"
    echo "  - Results: /app/mint/results/benchmarks/"
    echo ""
    echo "View results:"
    echo "  docker exec mint_container ls -lh /app/mint/results/benchmarks/"
    echo ""
fi

echo "Next steps:"
if [ "$RUN_TRAINING" = true ] && [ "$RUN_EVALUATION" = false ]; then
    echo "  - Run evaluation: bash run_mint.sh --evaluation-only"
elif [ "$RUN_TRAINING" = false ] && [ "$RUN_EVALUATION" = true ]; then
    echo "  - View results: cat /app/mint/results/benchmarks/*.json"
else
    echo "  - View training results: cat /app/mint/results/mint/mint_results.json"
    echo "  - View evaluation results: cat /app/mint/results/benchmarks/*.json"
fi
echo ""

