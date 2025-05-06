#!/bin/bash

# --- Configuration ---
SOURCE_DATA="dataset/source/BPSK-500Kbps-DATA.h5"
TARGET_DATA="dataset/target/TARGET-BPSK-DATA.h5" # Replace with your target file name
MODEL_TYPE="resnet" # Or 'lstm'
SEGMENT_LENGTH=1024
BATCH_SIZE=128
EPOCHS=50 # Adjust as needed
LR=1e-5
RESULTS_DIR="results/comparison_${MODEL_TYPE}_seg${SEGMENT_LENGTH}"
SAVE_NAME_BASE="comp_${MODEL_TYPE}_seg${SEGMENT_LENGTH}"
SEED=42

# Ensure results directory exists
mkdir -p $RESULTS_DIR/models

echo "===== Running Comparison Experiment ====="
echo "Model: $MODEL_TYPE, Segment Length: $SEGMENT_LENGTH"
echo "Results will be saved in: $RESULTS_DIR"

# --- Step 1: Pre-train on Source ---
echo "\n----- Step 1: Pre-training on Source Domain -----"
PRETRAINED_MODEL_PATH="${RESULTS_DIR}/models/${SAVE_NAME_BASE}_source_pretrained_source_pretrained_final.pth.tar" # Use _final or _best
python -m src.main \
    --source_data $SOURCE_DATA \
    --target_data $TARGET_DATA \
    --model_type $MODEL_TYPE \
    --segment_length $SEGMENT_LENGTH \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --mode pretrain \
    --results_dir $RESULTS_DIR \
    --save_name "${SAVE_NAME_BASE}_source_pretrained" \
    --seed $SEED \
    --num_classes_source 8 \
    --num_classes_target 8

# Check if pretraining was successful (model file exists)
if [ ! -f "$PRETRAINED_MODEL_PATH" ]; then
    echo "Pretraining failed or model file not found at $PRETRAINED_MODEL_PATH. Exiting."
    exit 1
fi

# --- Step 2: Evaluate Pre-trained on Target (No Fine-tuning) ---
echo "\n----- Step 2: Evaluating Pre-trained on Target (No Fine-tuning) -----"
python -m src.main \
    --source_data $SOURCE_DATA \
    --target_data $TARGET_DATA \
    --model_type $MODEL_TYPE \
    --segment_length $SEGMENT_LENGTH \
    --batch_size $BATCH_SIZE \
    --epochs 1 \
    --lr $LR \
    --mode eval_pretrained \
    --pretrained_path $PRETRAINED_MODEL_PATH \
    --results_dir $RESULTS_DIR \
    --seed $SEED \
    --num_classes_source 8 \
    --num_classes_target 8

# --- Step 3: Fine-tune Pre-trained on Target ---
echo "\n----- Step 3: Fine-tuning on Target Domain -----"
python -m src.main \
    --source_data $SOURCE_DATA \
    --target_data $TARGET_DATA \
    --model_type $MODEL_TYPE \
    --segment_length $SEGMENT_LENGTH \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --mode finetune \
    --pretrained_path $PRETRAINED_MODEL_PATH \
    --results_dir $RESULTS_DIR \
    --save_name "${SAVE_NAME_BASE}_target_finetuned" \
    --seed $SEED \
    --num_classes_source 8 \
    --num_classes_target 8
    # Add --freeze_backbone here if you want to test freezing first

## --- Step 4: Train on Target Only (From Scratch) ---
#echo "\n----- Step 4: Training on Target Only (From Scratch) -----"
#python -m src.main \
#    --source_data $SOURCE_DATA \
#    --target_data $TARGET_DATA \
#    --model_type $MODEL_TYPE \
#    --segment_length $SEGMENT_LENGTH \
#    --batch_size $BATCH_SIZE \
#    --epochs $EPOCHS \
#    --lr $LR \
#    --mode target_only \
#    --results_dir $RESULTS_DIR \
#    --save_name "${SAVE_NAME_BASE}_target_only" \
#    --seed $SEED \
#    --num_classes_source 8 \
#    --num_classes_target 8


echo "===== Comparison Experiment Finished ====="