#!/bin/bash

DATASETS=("cifar100-coarse")
MODELS=("OpenCLIP_ViT-L-14_laion2b_s32b_b82k" "OpenCLIP_ViT-L-14_laion400m_e32" "official_clip")

for DATASET in "${DATASETS[@]}"; do
  for MODEL_NAME in "${MODELS[@]}"; do
    echo "Running fewshot for model: $MODEL_NAME on dataset: $DATASET"

    python main_fewshot.py \
      --data_root "." \
      --task "coarse" \
      --dataset "$DATASET" \
      --module penultimate \
      --model_names "$MODEL_NAME" \
      --sources "custom" \
      --model_dict_path "model_dict.json" \
      --input_dim 224 \
      --n_test 100 \
      --n_reps 5 \
      --n_classes 100 \
      --n_shot 5 \
      --sample_per_superclass \
      --out_dir "fewshot_results" \
      --embeddings_root "." \
      --transform_type without \
      --device cpu \
      --rnd_seed 42
  done
done

echo "All coarse dataset models processed!"
