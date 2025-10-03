#!/bin/bash
################################################################################
# MINT Training Pipeline
################################################################################
# Runs the complete MINT training pipeline:
#   Phase A: Train Mechanistic Tool Editors
#   Phase B: Train Value Heads
#   Phase C: Conformal Calibration
#   Phase D: MINT Inference
#
# Usage: bash scripts/run_pipeline.sh
################################################################################

set -e

echo "================================================================================"
echo "MINT Training Pipeline"
echo "================================================================================"
echo ""

# Configuration
DATA_DIR="/app/mint/data/counterfactual_pairs"
CHECKPOINT_DIR="/app/mint/checkpoints"
OUTPUT_DIR="/app/mint/output"
RESULTS_DIR="/app/mint/results"

# Check prerequisites
echo "Checking prerequisites..."

if [ ! -f "$DATA_DIR/pairs.pt" ]; then
    echo "ERROR: Counterfactual pairs not found at $DATA_DIR/pairs.pt"
    exit 1
fi

if [ ! -d "$CHECKPOINT_DIR/affordances" ]; then
    echo "ERROR: Affordances not found at $CHECKPOINT_DIR/affordances/"
    exit 1
fi

echo "✓ Prerequisites met"
echo ""

# Create output directories
mkdir -p "$OUTPUT_DIR/phase_a" "$OUTPUT_DIR/phase_b" "$OUTPUT_DIR/phase_c" "$OUTPUT_DIR/phase_d"
mkdir -p "$RESULTS_DIR/mint"

################################################################################
# Phase A: Train Mechanistic Tool Editors
################################################################################
echo "================================================================================"
echo "Phase A: Training Mechanistic Tool Editors"
echo "================================================================================"

python3 /app/mint/scripts/train_editors.py \
    --pairs_path "$DATA_DIR/pairs.pt" \
    --affordance_dir "$CHECKPOINT_DIR/affordances" \
    --output_dir "$CHECKPOINT_DIR/editors"

echo "✓ Phase A complete"
echo ""

################################################################################
# Phase B: Train Value Heads
################################################################################
echo "================================================================================"
echo "Phase B: Training Value Heads"
echo "================================================================================"

python3 /app/mint/scripts/train_value_heads.py \
    --pairs_path "$DATA_DIR/pairs.pt" \
    --output_dir "$CHECKPOINT_DIR/value_heads"

echo "✓ Phase B complete"
echo ""

################################################################################
# Phase C: Conformal Calibration
################################################################################
echo "================================================================================"
echo "Phase C: Conformal Calibration"
echo "================================================================================"

python3 /app/mint/scripts/calibrate.py \
    --pairs_path "$DATA_DIR/pairs.pt" \
    --value_head_dir "$CHECKPOINT_DIR/value_heads" \
    --output_dir "$CHECKPOINT_DIR/calibrators"

echo "✓ Phase C complete"
echo ""

################################################################################
# Phase D: MINT Inference
################################################################################
echo "================================================================================"
echo "Phase D: MINT Inference"
echo "================================================================================"

python3 /app/mint/scripts/mint_inference.py \
    --affordance_dir "$CHECKPOINT_DIR/affordances" \
    --editor_dir "$CHECKPOINT_DIR/editors" \
    --value_head_dir "$CHECKPOINT_DIR/value_heads" \
    --calibrator_dir "$CHECKPOINT_DIR/calibrators"

echo "✓ Phase D complete"
echo ""

################################################################################
# Summary
################################################################################
echo "================================================================================"
echo "MINT Training Pipeline Complete"
echo "================================================================================"
echo ""
echo "Checkpoints: $CHECKPOINT_DIR/"
echo "Results: $RESULTS_DIR/mint/"
echo "Logs: $OUTPUT_DIR/"
echo ""
echo "Next: bash scripts/run_evaluation.sh"

