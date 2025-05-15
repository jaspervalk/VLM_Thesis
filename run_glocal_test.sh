#!/bin/bash

dataset="cifar100"
model="openclip_laion2b"

MODULE="penultimate"
SOURCE="custom"
TRIPLET_ROOT="data"
FORMAT="pt"
LR=0.01
LAMBDA=0.001
ALPHA=0.1
TAU=0.1
BATCH_SIZE=512
OPTIM="SGD"
DEVICE="cpu"
NUM_PROCESSES=2
DATA_ROOT="data"
PROBING_ROOT="features/${dataset}"
LOG_DIR="logs"
AUX_FEATURES_ROOT="aux_features_root/${dataset}/${model}"

echo "🚀 Test Run: $model on $dataset"
python3 main_glocal_probing_efficient.py \
  --model "$model" \
  --source "$SOURCE" \
  --module "$MODULE" \
  --dataset "$dataset" \
  --data_root "$DATA_ROOT" \
  --aux_features_root "$AUX_FEATURES_ROOT" \
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
