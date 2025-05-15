#!/bin/bash

MODEL="OpenCLIP_ViT-L-14_laion2b_s32b_b82k"
SOURCE="custom"
MODULE="penultimate"
OPTIM="SGD"
ETA=0.001
LAMBDA=0.001
ALPHA=0.1
TAU=1.0
BATCH_SIZE=1024
EPOCHS=100
SIGMA=0.01
SEED=42

THINGS_FEATURES="./features/custom/${MODEL}/${MODULE}"
TRANSFORM_DIR="./transforms/${SOURCE}/${MODEL}/${MODULE}/sgd/${ETA}/${LAMBDA}/${ALPHA}/${TAU}/${BATCH_SIZE}"
LOG_DIR="./logs/glocal_2b"
mkdir -p "$LOG_DIR"

echo "Training gLocal for $MODEL on CIFAR100..."
python main_glocal_probing_efficient.py \
  --data_root "./data" \
  --dataset "things" \
  --model "$MODEL" \
  --source "$SOURCE" \
  --module "$MODULE" \
  --device cpu \
  --optim "$OPTIM" \
  --learning_rates "$ETA" \
  --lmbdas "$LAMBDA" \
  --alphas "$ALPHA" \
  --taus "$TAU" \
  --sigma "$SIGMA" \
  --contrastive_batch_sizes "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --triplet_batch_size "$BATCH_SIZE" \
  --aux_features_root "$THINGS_FEATURES" \
  --probing_root "./transforms" \
  --log_dir "$LOG_DIR" \
  --rnd_seed "$SEED"
