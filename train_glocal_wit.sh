#!/bin/bash

MODEL="CLIP_ViT-L-14_WIT"
SOURCE="custom"
MODULE="penultimate"
INPUT_DIM=224
DATASET="things"
DEVICE="cpu"

# Output location for THINGS features
OUT_ROOT="./features/things"
mkdir -p "$OUT_ROOT"

# Extract features
python extract_features.py \
  --data_root "./data" \
  --dataset "$DATASET" \
  --model "$MODEL" \
  --source "$SOURCE" \
  --module "$MODULE" \
  --device "$DEVICE" \
  --batch_size 128 \
  --output_dir "$OUT_ROOT" \
  --input_dim "$INPUT_DIM"
