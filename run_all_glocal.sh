#!/bin/bash

# List of models to run probing on
models=(
  "OpenCLIP_RN50_yfcc15m"
  "OpenCLIP_ViT-L-14_laion400m_e32"
  "OpenCLIP_ViT-L-14_laion2b_s32b_b82k"
)

# Base directories (edit if needed)
DATASET="things"
DATA_ROOT="data"  # e.g., triplets & metadata
PROBING_ROOT="features/things"
IMAGENET_ROOT="features/imagenet"
LOG_DIR="results/things/penultimate"
MODULE="penultimate"
SOURCE="custom"
FORMAT="pt"

# Optimization config
LR=0.01
LAMBDA=0.001
ALPHA=0.1
TAU=0.1
BATCH_SIZE=512
OPTIM="SGD"
DEVICE="cpu"
NUM_PROCESSES=4

# Loop through each model and run glocal probing
for model in "${models[@]}"; do
  echo "🚀 Running glocal probing for: $model"
  python3 main_glocal_probing_efficient.py \
    --model "$model" \
    --source "$SOURCE" \
    --module "$MODULE" \
    --dataset "$DATASET" \
    --data_root "$DATA_ROOT" \
    --imagenet_features_root "$IMAGENET_ROOT" \
    --probing_root "$PROBING_ROOT" \
    --log_dir "$LOG_DIR" \
    --learning_rates "$LR" \
    --lmbdas "$LAMBDA" \
    --alphas "$ALPHA" \
    --taus "$TAU" \
    --contrastive_batch_sizes "$BATCH_SIZE" \
    --optim "$OPTIM" \
    --device "$DEVICE" \
    --features_format "$FORMAT" \
    --num_processes "$NUM_PROCESSES"
done
