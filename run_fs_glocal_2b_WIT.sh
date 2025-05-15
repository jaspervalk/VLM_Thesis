#!/bin/bash

DATASET="cifar100"
EMBEDDINGS_ROOT="./features/${DATASET}/custom"
OUT_DIR="./features/results"
TRANSFORMS="./features/transforms"
MODEL_DICT_PATH="./model_dict.json"  # <== using your local file

MODELS=("CLIP_ViT-L-14_WIT" "OpenCLIP_ViT-L-14_laion2b_s32b_b82k")
SOURCES=("custom" "custom")

for i in "${!MODELS[@]}"; do
  MODEL="${MODELS[$i]}"
  SOURCE="${SOURCES[$i]}"

  echo "[INFO] Running FS glocal evaluation for: $MODEL on $DATASET"

  python main_fewshot.py \
    --data_root ./data \
    --dataset "$DATASET" \
    --model_names "$MODEL" \
    --sources "$SOURCE" \
    --overall_source thingsvision \
    --module penultimate \
    --embeddings_root "$EMBEDDINGS_ROOT" \
    --out_dir "$OUT_DIR" \
    --input_dim 224 \
    --device cuda \
    --n_shot 5 \
    --n_test 100 \
    --n_reps 1 \
    --regressor_type ridge \
    --n_classes 100 \
    --transform_type glocal \
    --transforms_root "$TRANSFORMS" \
    --model_dict_path "$MODEL_DICT_PATH" \
    --etas 0.001 \
    --lmbdas 0.001 \
    --alphas 0.1 \
    --taus 1.0 \
    --contrastive_batch_sizes 1024

  echo "[INFO] Finished FS eval for: $MODEL"
  echo "--------------------------------------------------"
done

echo "All FS evaluations done."
