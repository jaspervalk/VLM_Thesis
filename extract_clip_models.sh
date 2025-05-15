#!/bin/bash

DATASETS=("cifar100" "cifar100-coarse")
MODEL_NAMES=("official_clip" "openclip_laion400m" "openclip_laion2b")
DEVICE="cpu"
BATCH_SIZE=64
SOURCE="custom"
FEATURES_ROOT="./features"
DATA_ROOT="../human_alignment/datasets"

for dataset in "${DATASETS[@]}"; do
    for model in "${MODEL_NAMES[@]}"; do
        echo "Extracting features for $model on $dataset (source=$SOURCE)"
        python main_clip_feature_extraction.py \
            --datasets "$dataset" \
            --model_names "$model" \
            --source "$SOURCE" \
            --device "$DEVICE" \
            --batch_size "$BATCH_SIZE" \
            --features_root "$FEATURES_ROOT" \
            --data_root "$DATA_ROOT"
    done
done
