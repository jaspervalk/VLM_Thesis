#!/bin/bash

MODEL="OpenCLIP_ViT-L-14_laion2b_s32b_b82k"
SOURCE="custom"
MODULE="penultimate"

OPTIM="SGD"
ETA=0.001
LAMBDA=0.001
ALPHA=0.1
TAU=1
BATCH_SIZE=1024

DATA_ROOT="./data"
EMBEDDINGS_THINGS="./features/things/embeddings/model_features_per_source.pkl"
TRANSFORM_OUT="./transforms"

python main_glocal_probing_efficient.py \
  --data_root $DATA_ROOT \
  --dataset things \
  --model $MODEL \
  --source $SOURCE \
  --module $MODULE \
  --device cpu \
  --optim $OPTIM \
  --learning_rates $ETA \
  --lmbdas $LAMBDA \
  --alphas $ALPHA \
  --taus $TAU \
  --contrastive_batch_sizes $BATCH_SIZE \
  --epochs 100 \
  --triplet_batch_size 1024 \
  --aux_features_root $EMBEDDINGS_THINGS \
  --probing_root $TRANSFORM_OUT \
  --num_processes 4 \
  --rnd_seed 42 \
  --log_dir "./logs/glocal_laion2b"
