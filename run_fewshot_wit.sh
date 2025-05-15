#!/bin/bash
set -e


DATASETS=("cifar100" "cifar100-coarse")
MODEL="CLIP_ViT-L-14_WIT"
INPUT_DIM=224 
N_REPS=5
N_TEST=100
N_SHOT=5
OUT_DIR="fewshot_results"

for DATASET in "${DATASETS[@]}"; do
  echo "Running fewshot for model: $MODEL on dataset: $DATASET"

  # Detect coarse task
  if [[ "$DATASET" == "cifar100-coarse" ]]; then
    TASK="coarse"
    N_CLASSES=100  # Still using fine labels, even in coarse mapping
    SAMPLE_SUPERCLASS="--sample_per_superclass"
    TRUE_DATASET="cifar100"
  else
    TASK="none"
    N_CLASSES=100
    SAMPLE_SUPERCLASS=""
    TRUE_DATASET="$DATASET"
  fi

  python main_fewshot.py \
    --data_root "." \
    --task "$TASK" \
    --dataset "$TRUE_DATASET" \
    --module penultimate \
    --model_names "$MODEL" \
    --sources "custom" \
    --model_dict_path "model_dict.json" \
    --input_dim "$INPUT_DIM" \
    --n_test "$N_TEST" \
    --n_reps "$N_REPS" \
    --n_shot "$N_SHOT" \
    --n_classes "$N_CLASSES" \
    $SAMPLE_SUPERCLASS \
    --out_dir "$OUT_DIR" \
    --embeddings_root "." \
    --transform_type without \
    --device cpu \
    --rnd_seed 42
done

echo ""
echo "Few-shot evaluation for $MODEL done."
