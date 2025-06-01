#!/bin/bash

MODEL="OpenCLIP_ViT-L-14_laion2b_s32b_b82k"
SOURCE="custom"
MODULE="penultimate"
DATASET="cifar100"
INPUT_DIM=224

N_SHOT=5
N_TEST=25
N_REPS=5
N_CLASSES=100

OPTIM="SGD"
ETA=0.001
LAMBDA=0.001
ALPHA=0.1
TAU=1.0
BATCH_SIZE=1024

DATA_ROOT="./data"
EMBEDDINGS_ROOT="./features"
THINGS_EMBEDDINGS="./features/things/embeddings/model_features_per_source.pkl"
TRANSFORMS_ROOT="./transforms3/random"
OUT_DIR="fewshot_results/transforms3/no_trans"


echo "Running few-shot classification on $DATASET using $MODEL with gLocal transform..."

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
  --etas "$ETA" \
  --lmbdas "$LAMBDA" \
  --alphas "$ALPHA" \
  --taus "$TAU" \
  --contrastive_batch_sizes "$BATCH_SIZE" \
  --transforms_root "$TRANSFORMS_ROOT" \
  --things_embeddings_path "$THINGS_EMBEDDINGS" \
  --device cuda \
  --rnd_seed 42


# #!/bin/bash
# MODEL="OpenCLIP_ViT-L-14_laion2b_s32b_b82k"
# SOURCE="custom"
# MODULE="penultimate"
# DATASET="cifar100"
# INPUT_DIM=224

# N_SHOT=5
# N_TEST=25
# N_REPS=5
# N_CLASSES=100

# OPTIM="sgd"
# ETA=0.001
# LAMBDA=0.001
# ALPHA=0.1
# TAU=1.0
# BATCH_SIZE=1024

# DATA_ROOT="./data"
# EMBEDDINGS_ROOT="./features"
# TRANSFORMS_ROOT="./transforms"  # use transforms, not transform3
# THINGS_EMBEDDINGS="./features/things/embeddings/model_features_per_source.pkl"
# OUT_DIR="fewshot_results/oldtransform/cifar100/"

# echo "Running few-shot classification on cifar100 using $MODEL with OLD transform..."

# python main_fewshot.py \
#   --data_root "$DATA_ROOT" \
#   --dataset "cifar100" \
#   --module "$MODULE" \
#   --model_names "$MODEL" \
#   --sources "$SOURCE" \
#   --model_dict_path "model_dict.json" \
#   --input_dim "$INPUT_DIM" \
#   --n_test "$N_TEST" \
#   --n_reps "$N_REPS" \
#   --n_classes "$N_CLASSES" \
#   --n_shot "$N_SHOT" \
#   --out_dir "$OUT_DIR" \
#   --embeddings_root "$EMBEDDINGS_ROOT" \
#   --transform_type glocal \
#   --etas "$ETA" \
#   --lmbdas "$LAMBDA" \
#   --alphas "$ALPHA" \
#   --taus "$TAU" \
#   --contrastive_batch_sizes "$BATCH_SIZE" \
#   --transforms_root "$TRANSFORMS_ROOT" \
#   --things_embeddings_path "$THINGS_EMBEDDINGS" \
#   --device cuda \
#   --rnd_seed 42
