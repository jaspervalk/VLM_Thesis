#!/bin/bash
set -e


TRIPLET_FILE="triplet_dataset/trainset_10pct.npy"
MODEL="OpenCLIP_ViT-L-14_laion400m_e32"
SOURCE="custom"
MODULE="penultimate"
DATASET="things"

DATA_ROOT="./data"
AUX_FEATURES_ROOT="./features/things/custom/$MODEL/$MODULE"
PROBING_ROOT="./features/things"
DEVICE="gpu"

OPTIM="SGD"
ETA=0.003
LAMBDA=0.1
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


LOG_DIR="transforms2/trainset_10pct/400m"
mkdir -p "$LOG_DIR"

# Grid of alphas to search. #!/bin/bash
set -e

# Configuration parameters
TRIPLET_FILE="triplet_dataset/trainset_10pct.npy"
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
LAMBDA=0.1
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

LOG_DIR="transforms2/trainset_10pct/400m"
mkdir -p "$LOG_DIR"

# Toggle this to use a separate folder per alpha
ALPHA_SUBDIRS=true

# Alpha grid
ALPHAS=(0.25 0.5)

for ALPHA in "${ALPHAS[@]}"
do
    if [ "$ALPHA_SUBDIRS" = true ]; then
        OUT_DIR="$LOG_DIR/alpha_${ALPHA}"
    else
        OUT_DIR="$LOG_DIR"
    fi
    mkdir -p "$OUT_DIR"
    echo "=== Training transform for alpha=$ALPHA ==="
    echo "Output directory: $OUT_DIR"
    echo "---------------------------------------------------"

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
      --log_dir "$OUT_DIR" \
      --custom_out_path "$OUT_DIR" \
      --optim "$OPTIM" \
      --rnd_seed "$RND_SEED" \
      --use_bias

    echo "Done: alpha=$ALPHA (saved in $OUT_DIR)"
    echo ""
done

echo ""
echo "=== Finished ALL! Your new transforms should be in: ==="
if [ "$ALPHA_SUBDIRS" = true ]; then
    echo "  $LOG_DIR/alpha_*/transform.npz"
else
    echo "  $LOG_DIR/transform.npz"
fi
echo "Check for transform.npz in the expected subfolders (one for each alpha value)."

ALPHAS=(0.05 0.1 0.25 0.5)
# GS loop
for ALPHA in "${ALPHAS[@]}"
do
    echo "=== Training transform for alpha=$ALPHA ==="
    
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

    echo "Done: alpha=$ALPHA"
    echo ""
done

echo ""
echo "Finished ALL! Your new transforms should be in:"
echo "  $LOG_DIR"
echo "Check for transform.npz in the expected subfolders (one for each alpha value)."
