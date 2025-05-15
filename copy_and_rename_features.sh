#!/bin/bash

for DATASET in "cifar100" "cifar100-coarse"; do

  echo "Processing dataset: $DATASET"


  if [[ "$DATASET" == "cifar100" ]]; then
    old_models=("openclip_laion2b" "openclip_laion400m" "official_clip")
    new_models=("OpenCLIP_ViT-L-14_laion2b_s32b_b82k" "OpenCLIP_ViT-L-14_laion400m_e32" "official_clip")
  else
    old_models=("openclip_laion2b" "openclip_laion400m" "official_clip")
    new_models=("OpenCLIP_ViT-L-14_laion2b_s32b_b82k" "OpenCLIP_ViT-L-14_laion400m_e32" "official_clip")
  fi

  for i in "${!old_models[@]}"; do
    old_name="${old_models[$i]}"
    new_name="${new_models[$i]}"

    echo "Processing $DATASET model: $old_name --> $new_name"

    mkdir -p features/${DATASET}/custom/${new_name}/penultimate

    cp features/${DATASET}/custom/${old_name}/penultimate/features.pkl \
       features/${DATASET}/custom/${new_name}/penultimate/embeddings.pkl

    echo "Copied and renamed for: $new_name"
  done

done

echo "All datasets and models processed!"
