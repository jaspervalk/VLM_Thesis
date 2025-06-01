#!/bin/bash

MODEL="OpenCLIP_ViT-L-14_laion2b_s32b_b82k"
SOURCE="custom"
MODULE="penultimate"
DATASET="things"
DATA_ROOT="./data"
AUX_FEATURES_ROOT="./features/things/custom/$MODEL/$MODULE"
PROBING_ROOT="./features/things"
DEVICE="gpu"

OPTIM="SGD"
ETA=0.001
LAMBDA=0.001
ALPHA=0.5
TAU=1.0
BATCH_SIZE=1024
TRIPLET_BATCH_SIZE=256
EPOCHS=50  # for quick debug
SIGMA=0.1
RND_SEED=42

TRIPLET_FILE="data/triplets/train_0.01pct.npy"  # small file for testing

python main_glocal_probing_efficient.py \
  --data_root "$DATA_ROOT" \
  --dataset "$DATASET" \
  --model "$MODEL" \
  --source "$SOURCE" \
  --module "$MODULE" \
  --device "$DEVICE" \
  --optim "$OPTIM" \
  --learning_rates "$ETA" \
  --lmbdas "$LAMBDA" \
  --alphas "$ALPHA" \
  --taus "$TAU" \
  --sigma "$SIGMA" \
  --contrastive_batch_sizes "$BATCH_SIZE" \
  --triplet_batch_size "$TRIPLET_BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --aux_features_root "$AUX_FEATURES_ROOT" \
  --probing_root "$PROBING_ROOT" \
  --log_dir "./logs/glocal_test" \
  --triplet_file "$TRIPLET_FILE" \
  --rnd_seed "$RND_SEED" \
  --custom_out_path "./transforms3/0.01pct/alpha0.5/custom/OpenCLIP_ViT-L-14_laion2b_s32b_b82k/penultimate/sgd/0.001/0.001/0.1/1.0/1024"

