#!/bin/bash

# === Description ===
# This script trains a test transform on 0.01% (~410) triplets using OpenCLIP on CIFAR100,
# and saves it under a custom path to avoid accidental overwrite or caching.

# === Triplet Setup ===
TRIPLET_FILE="triplet_dataset/trainset_0_01pct.npy"
TRIPLET_NAME=$(basename "$TRIPLET_FILE" .npy)_testtransform

# === Model and Data Info ===
MODEL="OpenCLIP_ViT-L-14_laion2b_s32b_b82k"
SOURCE="custom"
MODULE="penultimate"

DATASET="things"
DATA_ROOT="./triplet_dataset"  # Root containing triplets
PROBING_ROOT="./features/things"  # Main probing output location
AUX_FEATURES_ROOT="./features/things/custom/$MODEL/$MODULE"

# === Hyperparams ===
OPTIM="SGD"
ETA=0.001
LAMBDA=0.001
ALPHA=0.1
TAU=1.0
BATCH_SIZE=1024
TRIPLET_BATCH_SIZE=256
EPOCHS=100
BURNIN=20
PATIENCE=15
SIGMA=0.001

# === Misc ===
FEATURES_FORMAT="pt"
NUM_PROCESSES=4
RND_SEED=42
DEVICE="gpu"

# === Custom Output Path to avoid overwrite ===
LOG_DIR="./transforms_check/logs/$TRIPLET_NAME"
OUT_DIR="./test_transforms/$TRIPLET_NAME"

mkdir -p "$LOG_DIR"
mkdir -p "$OUT_DIR"

# === Launch training ===
echo "Training test transform for $MODEL on $DATASET using $TRIPLET_FILE..."
echo "Output path will be: $OUT_DIR"

python main_glocal_probing_efficient.py \
  --data_root "$DATA_ROOT" \
  --aux_features_root "$AUX_FEATURES_ROOT" \
  --dataset "$DATASET" \
  --triplet_file "$TRIPLET_FILE" \
  --model "$MODEL" \
  --module "$MODULE" \
  --source "$SOURCE" \
  --learning_rates "$ETA" \
  --lmbdas "$LAMBDA" \
  --alphas "$ALPHA" \
  --taus "$TAU" \
  --contrastive_batch_sizes "$BATCH_SIZE" \
  --triplet_batch_size "$TRIPLET_BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --burnin "$BURNIN" \
  --patience "$PATIENCE" \
  --sigma "$SIGMA" \
  --device "$DEVICE" \
  --features_format "$FEATURES_FORMAT" \
  --num_processes "$NUM_PROCESSES" \
  --probing_root "$PROBING_ROOT" \
  --log_dir "$LOG_DIR" \
  --optim "$OPTIM" \
  --rnd_seed "$RND_SEED" \
  --custom_out_path "$OUT_DIR"
