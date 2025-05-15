#!/bin/bash

DATASET="cifar100-coarse"
MODELS=("OpenCLIP_ViT-L-14_laion400m_e32" "OpenCLIP_ViT-L-14_laion2b_s32b_b82k")

for MODEL in "${MODELS[@]}"; do
  echo "[INFO] Extracting features for: $DATASET / $MODEL"
  python main_clip_feature_extraction.py \
    --data_root ./data \
    --datasets "$DATASET" \
    --model_names "$MODEL" \
    --source custom \
    --features_root ./features \
    --device cuda \
    --batch_size 128
  echo "[INFO] Done with $MODEL on $DATASET"
  echo "----------------------------------"a
done
