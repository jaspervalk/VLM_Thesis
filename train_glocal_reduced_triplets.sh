#!/bin/bash


TRIPLET_FILE="triplet_dataset/trainset_10pct.npy"
TRIPLET_NAME=$(basename "$TRIPLET_FILE" .npy)
MODEL="OpenCLIP_ViT-L-14_laion400m_e32"
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
ALPHA=0.1
TAU=1.0
BATCH_SIZE=1024
TRIPLET_BATCH_SIZE=256
EPOCHS=100
BURNIN=20
PATIENCE=100
SIGMA=0.1

FEATURES_FORMAT="pt"
NUM_PROCESSES=4
RND_SEED=42

LOG_DIR="transforms2/trainset_10pct/400m"  # change per run, dont forget plz
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
  --custom_out_path "$LOG_DIR" \
  --optim "$OPTIM" \
  --rnd_seed "$RND_SEED" \
  --use_bias

echo ""
echo "Finished! Your new transform should be in:"
echo "  $LOG_DIR"
echo "  (Check for transform.npz in subfolders, should have 'weights', 'bias', 'mean', 'std')"
