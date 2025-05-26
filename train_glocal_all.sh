#!/bin/bash

# === Model and Data ===
MODELS=("OpenCLIP_ViT-L-14_laion2b_s32b_b82k" "OpenCLIP_ViT-L-14_laion400m_e32")
TRIPLET_PCTS=("25pct" "50pct" "75pct")
SOURCE="custom"
MODULE="penultimate"

DATASET="things"
DATA_ROOT="./data"
PROBING_ROOT="./features/things"
DEVICE="gpu"
FEATURES_FORMAT="pt"
NUM_PROCESSES=4
RND_SEED=42

# === Optimization ===
OPTIM="SGD"
ETA=0.001
LAMBDA=0.001
ALPHA=0.1
TAU=1.0
BATCH_SIZE=1024
TRIPLET_BATCH_SIZE=256
EPOCHS=100
BURNIN=20
PATIENCE=100
SIGMA=0.1

for MODEL in "${MODELS[@]}"; do
  for PCT in "${TRIPLET_PCTS[@]}"; do

    TRIPLET_FILE="triplet_dataset/trainset_${PCT}.npy"
    TRIPLET_NAME=$(basename "$TRIPLET_FILE" .npy)
    LOG_DIR="./transforms_reduction/${TRIPLET_NAME}/${MODEL}/${MODULE}"

    AUX_FEATURES_ROOT="./features/things/custom/${MODEL}/${MODULE}"

    echo "============================================================="
    echo "Training with: $MODEL | Triplets: $PCT"
    echo "Logging to: $LOG_DIR"
    echo "============================================================="

    mkdir -p "$LOG_DIR"

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
      --rnd_seed "$RND_SEED"

    echo "Finished training: $MODEL at $PCT triplets"
    echo
  done
done
