#!/bin/bash

models=(
  "openclip_laion2b"
  "openclip_laion400m"
  "official_clip"
)

datasets=(
  "cifar100"
  "cifar100-coarse"
)

# Just point this to where the wrapped .pkl file lives
PROBING_ROOT="features/things"
DATA_ROOT="data"
LOG_DIR="results"
MODULE="penultimate"
SOURCE="custom"
FORMAT="pt"

# Optimization
LR=0.001
LAMBDA=0.001
ALPHA=0.1
TAU=1.0
BATCH_SIZE=512
TRIPLET_BATCH_SIZE=256
OPTIM="SGD"
DEVICE="cpu"
NUM_PROCESSES=4

for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    echo "🚀 Running gLocal probing for: $model on $dataset"

    python3 main_glocal_probing_efficient.py \
      --model "$model" \
      --source "$SOURCE" \
      --module "$MODULE" \
      --dataset "$dataset" \
      --data_root "${DATA_ROOT}/${dataset}" \
      --probing_root "$PROBING_ROOT" \
      --log_dir "${LOG_DIR}/${dataset}/${MODULE}" \
      --learning_rates "$LR" \
      --lmbdas "$LAMBDA" \
      --alphas "$ALPHA" \
      --taus "$TAU" \
      --contrastive_batch_sizes "$BATCH_SIZE" \
      --triplet_batch_size "$TRIPLET_BATCH_SIZE" \
      --optim "$OPTIM" \
      --device "$DEVICE" \
      --features_format "$FORMAT" \
      --num_processes "$NUM_PROCESSES"
  done
done
