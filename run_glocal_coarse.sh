#!/bin/bash

# Model & Dataset settings
MODEL="OpenCLIP_ViT-L-14_laion2b_s32b_b82k"
SOURCE="custom"
MODULE="penultimate"
DATASET="cifar100-coarse"
INPUT_DIM=224
N_CLASSES=20  # CIFAR100-Coarse has 20 superclass categories

# Few-shot settings
N_SHOT=5
N_TEST=100
N_REPS=1

# gLocal transform parameters
OPTIM="SGD"
ETA=0.001
LAMBDA=0.001
ALPHA=0.1
TAU=1.0
BATCH_SIZE=1024

# Paths
DATA_ROOT="./data"
EMBEDDINGS_ROOT="./features"
TRANSFORMS_ROOT="./features/transforms"
THINGS_EMBEDDINGS="./features/things/embeddings/model_features_per_source.pkl"
OUT_DIR="./fewshot_results"

# Ensure transform is in expected location
EXPECTED_TRANSFORM_PATH="$TRANSFORMS_ROOT/$SOURCE/$MODEL/$MODULE/sgd/$ETA/$LAMBDA/$ALPHA/$TAU/$BATCH_SIZE"
mkdir -p "$EXPECTED_TRANSFORM_PATH"

# (Optional) If needed, move/copy the transform manually here
# cp ./transforms/$MODEL/visual/transform.npz "$EXPECTED_TRANSFORM_PATH/transform.npz"

echo "[INFO] Running few-shot classification on $DATASET using $MODEL with gLocal transform..."

python main_fewshot.py \
  --data_root "$DATA_ROOT" \
  --dataset "$DATASET" \
  --module "$MODULE" \
  --model_names "$MODEL" \
  --sources "$SOURCE" \
  --overall_source thingsvision \
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

echo "[INFO] Evaluation done."
