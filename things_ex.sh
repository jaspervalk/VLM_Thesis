#!/bin/bash

MODEL="OpenCLIP_ViT-L-14_laion400m_e32"
SOURCE="custom"
DATASET="things"
DEVICE="cuda"  # or "cpu" if needed

# Run the extraction
python main_clip_feature_extraction.py \
  --datasets "$DATASET" \
  --model_names "$MODEL" \
  --source "$SOURCE" \
  --device "$DEVICE" \
  --batch_size 256 \
  --features_root "./features" \
  --data_root "./data"
