#!/bin/bash

# Model and dataset config
MODEL="OpenCLIP_ViT-L-14_laion2b_s32b_b82k"
SOURCE="custom"
MODULE="penultimate"
DATASET="cifar100-coarse"
INPUT_DIM=768

# Few-shot setup
N_SHOT=5
N_TEST=25
N_REPS=5
N_CLASSES=20  # coarse = 20 superclasses

# No transform → ignore transform-specific params
DATA_ROOT="./data"
EMBEDDINGS_ROOT="./features"
THINGS_EMBEDDINGS="./features/things/embeddings/model_features_per_source.pkl"
OUT_DIR="fewshot_results_baseline_cifar100coarse"

# Run few-shot classification WITHOUT gLocal
echo "Running 5-shot classification on $DATASET with $MODEL (no transform)..."

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
  --transform_type without \
  --things_embeddings_path "$THINGS_EMBEDDINGS" \
  --device cuda \
  --rnd_seed 42
