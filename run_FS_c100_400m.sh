#!/bin/bash

# ==== Config ====
MODEL="OpenCLIP_ViT-L-14_laion400m_e32"
SOURCE="custom"
MODULE="penultimate"
DATASET="cifar100"
INPUT_DIM=32  

# Few-shot settings
N_SHOT=5
N_TEST=25
N_REPS=5
N_CLASSES=100

# Paths
DATA_ROOT="./data"
EMBEDDINGS_ROOT="./features"
OUT_DIR="./fewshot_results"

# Clean previous results to ensure no caching errors
RESULT_PATH="${OUT_DIR}/${DATASET}/${SOURCE}/${MODEL}/${MODULE}/0.001/0.001/None/None/None/False/fewshot_results.pkl"
if [ -f "$RESULT_PATH" ]; then
  echo "Removing old result file: $RESULT_PATH"
  rm -f "$RESULT_PATH"
fi

# Remove old embeddings (optional but recommended)
EMBEDDINGS_DIR="${EMBEDDINGS_ROOT}/${DATASET}/${SOURCE}/${MODEL}/${MODULE}"
if [ -d "$EMBEDDINGS_DIR" ]; then
  echo "Removing cached embeddings at: $EMBEDDINGS_DIR"
  rm -rf "$EMBEDDINGS_DIR"
fi

# ==== Run few-shot classification ====
echo "Running few-shot classification on $DATASET using $MODEL without gLocal transform..."

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
  --device cpu \
  --rnd_seed 42

echo "Done."
