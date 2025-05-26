#!/bin/bash

MODEL="OpenCLIP_ViT-L-14_laion2b_s32b_b82k"
SOURCE="custom"
MODULE="penultimate"
DATASET="cifar100"
INPUT_DIM=224

N_SHOT=5
N_TEST=25
N_REPS=5
N_CLASSES=100

OPTIM="SGD"
ETA=0.001
LAMBDA=0.001
ALPHA=0.1
TAU=1.0
BATCH_SIZE=1024

DATA_ROOT="./data"
EMBEDDINGS_ROOT="./features"
THINGS_EMBEDDINGS="./features/things/embeddings/model_features_per_source.pkl"
TRANSFORM_BASE="./transforms_reduction/trainset_10pct"
TRANSFORMS_ROOT="$TRANSFORM_BASE"

OUT_DIR="fewshot_results_reduction_10pct"

echo "Running 5-shot classification on $DATASET with $MODEL using gLocal (10% triplets)..."

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
