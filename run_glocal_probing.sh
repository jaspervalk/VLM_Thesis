#!/bin/bash

PROBING_ROOT="/Users/jaspervalk/Documents/Thesis_project/VLM_Thesis/features"
AUX_FEATURES_ROOT="/Users/jaspervalk/Documents/Thesis_project/VLM_Thesis/aux_features"
LOG_DIR="/Users/jaspervalk/Documents/Thesis_project/VLM_Thesis/logs"

COMMON_ARGS="--module penultimate --source custom --features_format pt --device gpu --log_dir $LOG_DIR"

declare -a CONFIGS=(
    "cifar100 official_clip"
    "cifar100 openclip_laion400m"
    "cifar100-coarse official_clip"
    "cifar100-coarse openclip_laion2b"
    "cifar100-coarse openclip_laion400m"
)

for config in "${CONFIGS[@]}"
do
    IFS=' ' read -r dataset model <<< "$config"

    # Set paths
    AUX_ROOT="$AUX_FEATURES_ROOT/$dataset/$model"
    PROBE_ROOT="$PROBING_ROOT/$dataset/custom/$model/penultimate"
    DATASET_PATH="$DATA_ROOT/$dataset"

    echo "🚀 Running gLocal on $dataset / $model ..."

    python main_glocal_probing_efficient.py \
        --dataset $dataset \
        --model $model \
        --data_root $DATASET_PATH \
        --aux_features_root $AUX_ROOT \
        --probing_root $PROBE_ROOT \
        $COMMON_ARGS
done

