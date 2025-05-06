#!/bin/bash

# --- Configuration ---
SOURCE_DATA="dataset/source/BPSK-500Kbps-DATA.h5"
TARGET_DATA="dataset/target/TARGET-BPSK-DATA.h5" # Replace
MODEL_TYPE="resnet" # Or 'lstm'
SEGMENT_LENGTH=1024
BATCH_SIZE=64 # Smaller batch size if memory issues, e.g., 64 or 32
EPOCHS=50
LR=1e-3 # Adjusted from 1e-5, 1e-5 might be too small for Adam from scratch
TOP_LEVEL_RESULTS_DIR="results_main_comparison" # New top-level dir for this whole experiment
EXPERIMENT_NAME="comparison_${MODEL_TYPE}_seg${SEGMENT_LENGTH}_lr${LR}_ep${EPOCHS}" # More descriptive
RESULTS_DIR="${TOP_LEVEL_RESULTS_DIR}/${EXPERIMENT_NAME}" # Specific dir for this run's outputs

SEED=42

# Ensure results directory and its subdirectories exist (main.py will also try to create them)
mkdir -p "${RESULTS_DIR}/models"
mkdir -p "${RESULTS_DIR}/visualizations"
mkdir -p "${RESULTS_DIR}/logs"

echo "===== Running Comparison Experiment ====="
echo "Model: $MODEL_TYPE, Segment Length: $SEGMENT_LENGTH, LR: $LR, Epochs: $EPOCHS"
echo "Results will be saved in: $RESULTS_DIR"

# --- Step 1: Pre-train on Source ---
SAVE_NAME_PRETRAIN="${EXPERIMENT_NAME}_source_pretrained"
PRETRAINED_MODEL_PATH="${RESULTS_DIR}/models/${SAVE_NAME_PRETRAIN}_final.pth.tar"
echo "\n----- Step 1: Pre-training on Source Domain (Save Name: $SAVE_NAME_PRETRAIN) -----"
python -m src.main \
    --source_data $SOURCE_DATA \
    --target_data $TARGET_DATA \
    --model_type $MODEL_TYPE \
    --segment_length $SEGMENT_LENGTH \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --mode pretrain \
    --results_dir "$RESULTS_DIR" \
    --save_name "$SAVE_NAME_PRETRAIN" \
    --seed $SEED \
    --num_classes_source 8 \
    --num_classes_target 8 # Target classes needed by main.py even if not used in pretrain mode directly

if [ ! -f "$PRETRAINED_MODEL_PATH" ]; then
    echo "Pretraining failed or model file not found at $PRETRAINED_MODEL_PATH. Exiting."
    exit 1
fi

# --- Step 2: Evaluate Pre-trained on Target (No Fine-tuning) ---
SAVE_NAME_EVAL_PRETRAINED="${EXPERIMENT_NAME}_eval_pretrained_on_target"
echo "\n----- Step 2: Evaluating Pre-trained on Target (Save Name: $SAVE_NAME_EVAL_PRETRAINED) -----"
python -m src.main \
    --source_data $SOURCE_DATA \
    --target_data $TARGET_DATA \
    --model_type $MODEL_TYPE \
    --segment_length $SEGMENT_LENGTH \
    --batch_size $BATCH_SIZE \
    --epochs 1 \
    --lr $LR \
    --mode eval_pretrained \
    --pretrained_path "$PRETRAINED_MODEL_PATH" \
    --results_dir "$RESULTS_DIR" \
    --save_name "$SAVE_NAME_EVAL_PRETRAINED" \
    --seed $SEED \
    --num_classes_source 8 \
    --num_classes_target 8

# --- Step 3: Fine-tune Classifier Only on Target ---
SAVE_NAME_FINETUNE_CLS="${EXPERIMENT_NAME}_target_finetuned_classifier_only"
echo "\n----- Step 3: Fine-tuning Classifier on Target (Save Name: $SAVE_NAME_FINETUNE_CLS) -----"
python -m src.main \
    --source_data $SOURCE_DATA \
    --target_data $TARGET_DATA \
    --model_type $MODEL_TYPE \
    --segment_length $SEGMENT_LENGTH \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --mode finetune \
    --pretrained_path "$PRETRAINED_MODEL_PATH" \
    --results_dir "$RESULTS_DIR" \
    --save_name "$SAVE_NAME_FINETUNE_CLS" \
    --seed $SEED \
    --num_classes_source 8 \
    --num_classes_target 8
    # The --freeze_backbone_finetune flag in get_args is not used by this script explicitly,
    # as finetune mode now defaults to classifier only.

# --- Step 4: Train on Target Only (From Scratch) ---
SAVE_NAME_TARGET_ONLY="${EXPERIMENT_NAME}_target_only"
echo "\n----- Step 4: Training on Target Only (Save Name: $SAVE_NAME_TARGET_ONLY) -----"
python -m src.main \
    --source_data $SOURCE_DATA \
    --target_data $TARGET_DATA \
    --model_type $MODEL_TYPE \
    --segment_length $SEGMENT_LENGTH \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --mode target_only \
    --results_dir "$RESULTS_DIR" \
    --save_name "$SAVE_NAME_TARGET_ONLY" \
    --seed $SEED \
    --num_classes_source 8 \
    --num_classes_target 8

echo "===== Comparison Experiment Finished ====="