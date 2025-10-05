#!/bin/bash
set -euo pipefail
# Launch pair generation across GPUs specified in the config (gpu.CUDA_VISIBLE_DEVICES)

CONFIG_PATH="${1:-configs/default.yaml}"

# Extract GPU csv from config; fallback to 0..7 if missing
GPU_CSV="$(python - <<'PY' "$CONFIG_PATH"
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
print(((cfg.get('gpu') or {}).get('CUDA_VISIBLE_DEVICES') or '').strip())
PY
)"
if [[ -z "$GPU_CSV" ]]; then
  GPU_CSV="0,1,2,3,4,5,6,7"
fi
IFS=',' read -r -a GPU_IDS <<< "$GPU_CSV"
NUM_GPUS="${#GPU_IDS[@]}"

echo "=========================================="
echo "MINT: Multi-GPU Pair Generation"
echo "=========================================="
echo "Config: $CONFIG_PATH"
echo "GPUs: ${GPU_IDS[*]} (count=$NUM_GPUS)"
echo "=========================================="

# Create logs directory
mkdir -p logs
# Avoid torchvision import for text-only transformers use
export TRANSFORMERS_NO_TORCHVISION=1

# Launch processes on each configured GPU
for idx in "${!GPU_IDS[@]}"; do
    gpu_id="${GPU_IDS[$idx]}"
    echo "Launching process on GPU $gpu_id (rank $idx)..."
    # Use physical CUDA index directly (no CUDA_VISIBLE_DEVICES remapping)
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python scripts/prepare_pairs.py \
        --config_path="$CONFIG_PATH" \
        --gpu_id="$gpu_id" \
        --worker_rank="$idx" \
        --num_gpus="$NUM_GPUS" \
        --layer_shard=true \
        > "logs/prepare_pairs_gpu${gpu_id}.log" 2>&1 &

    # Store PID by worker rank
    pids[$idx]=$!
    echo "  PID: ${pids[$idx]}"
done

echo "=========================================="
echo "All processes launched. Waiting for completion..."
echo "=========================================="

# Wait for all processes to complete
for idx in "${!GPU_IDS[@]}"; do
    gpu_id="${GPU_IDS[$idx]}"
    wait ${pids[$idx]}
    exit_code=$?
    echo "GPU $gpu_id (rank $idx) completed with exit code $exit_code"
done

echo "=========================================="
echo "All GPUs completed. Merging results..."
echo "=========================================="

# Merge results using the number of configured GPUs
python scripts/merge_pairs.py --config_path="$CONFIG_PATH" --num_gpus="$NUM_GPUS"

echo "=========================================="
echo "Multi-GPU pair generation complete!"
echo "=========================================="
