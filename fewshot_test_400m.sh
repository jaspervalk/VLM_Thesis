#!/bin/bash

MODEL="OpenCLIP_ViT-L-14_laion400m_e32"
SOURCE="custom"
MODULE="penultimate"
DATASET="cifar100"
INPUT_DIM=224
N_SHOT=5
N_TEST=25
N_REPS=5
N_CLASSES=100
OPTIM="sgd"
ETA=0.001
LAMBDA=0.001
ALPHA=0.5
TAU=1.0
BATCH_SIZE=1024
DATA_ROOT="./data"
EMBEDDINGS_ROOT="./features"
THINGS_EMBEDDINGS="./features/things/embeddings/model_features_per_source.pkl"
# Let op: TRANSFORMS_ROOT verwijst nu naar de map waarin de 400m map zit
TRANSFORMS_ROOT="transforms2/trainset_01pct/400m"
OUT_DIR="fewshot_results_trainset_01pct_400m"

EXPECTED_TRANSFORM_DIR="$TRANSFORMS_ROOT/$SOURCE/$MODEL/$MODULE/$OPTIM/$ETA/$LAMBDA/$ALPHA/$TAU/$BATCH_SIZE"

# Ensure transform exists
if [ ! -f "$TRANSFORMS_ROOT/transform.npz" ]; then
  echo "Error: $TRANSFORMS_ROOT/transform.npz does not exist! Exiting."
  exit 1
fi

mkdir -p "$EXPECTED_TRANSFORM_DIR"
cp "$TRANSFORMS_ROOT/transform.npz" "$EXPECTED_TRANSFORM_DIR/transform.npz"
echo "Using transform from: $EXPECTED_TRANSFORM_DIR/transform.npz"

echo "Running few-shot with new transform on $DATASET..."

python main_fewshot.py \
  --data_root "$DATA_ROOT" \
  --dataset "$DATASET" \
  --module "$MODULE" \
  --model_names "$MODEL" \
  --sources "$SOURCE" \
  --model_dict_path "model_dict.json" \
  --input_dim "$INPUT_DIM" \
  --n_test "$N_TEST" \
  --n_reps "$N_REPS" \
  --n_classes "$N_CLASSES" \
  --n_shot "$N_SHOT" \
  --out_dir "$OUT_DIR" \
  --embeddings_root "$EMBEDDINGS_ROOT" \
  --transform_type glocal \
  --etas "$ETA" \
  --lmbdas "$LAMBDA" \
  --alphas "$ALPHA" \
  --taus "$TAU" \
  --contrastive_batch_sizes "$BATCH_SIZE" \
  --transforms_root "$TRANSFORMS_ROOT" \
  --things_embeddings_path "$THINGS_EMBEDDINGS" \
  --device cuda \
  --rnd_seed 42
