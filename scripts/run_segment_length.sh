#!/bin/bash

# --- Configuration ---
SOURCE_DATA="dataset/source/BPSK-500Kbps-DATA.h5"
TARGET_DATA="dataset/target/TARGET-BPSK-DATA.h5" # Replace with your target file name
MODEL_TYPE="resnet" # Or 'lstm'
SEGMENT_LENGTHS=(256 512 1024) # Add more lengths as needed
BATCH_SIZE=64
EPOCHS=50 # Adjust as needed
LR=1e-3
BASE_RESULTS_DIR="results/segment_length_${MODEL_TYPE}"
SEED=42

echo "===== Running Segment Length Experiment ====="
echo "Model: $MODEL_TYPE"
echo "Segment Lengths: ${SEGMENT_LENGTHS[@]}"
echo "Results will be saved in subdirectories under: $BASE_RESULTS_DIR"

for SEG_LEN in "${SEGMENT_LENGTHS[@]}"; do
    echo "\n----- Testing Segment Length: $SEG_LEN -----"
    RESULTS_DIR="${BASE_RESULTS_DIR}/seg${SEG_LEN}"
    SAVE_NAME_BASE="seglen_${SEG_LEN}_${MODEL_TYPE}"
    PRETRAINED_MODEL_PATH="${RESULTS_DIR}/models/${SAVE_NAME_BASE}_source_pretrained_final.pth.tar"

    mkdir -p $RESULTS_DIR/models

    # --- Step 1: Pre-train on Source for this segment length ---
    echo "--- Pre-training ---"
    python -m src.main \
        --source_data $SOURCE_DATA \
        --target_data $TARGET_DATA \
        --model_type $MODEL_TYPE \
        --segment_length $SEG_LEN \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --lr $LR \
        --mode pretrain \
        --results_dir $RESULTS_DIR \
        --save_name "${SAVE_NAME_BASE}_source_pretrained" \
        --seed $SEED \
        --num_classes_source 8 \
        --num_classes_target 8

    if [ ! -f "$PRETRAINED_MODEL_PATH" ]; then
        echo "Pretraining failed for segment length $SEG_LEN. Skipping fine-tuning."
        continue # Skip to next segment length
    fi

    # --- Step 2: Fine-tune on Target for this segment length ---
    echo "--- Fine-tuning ---"
    python -m src.main \
        --source_data $SOURCE_DATA \
        --target_data $TARGET_DATA \
        --model_type $MODEL_TYPE \
        --segment_length $SEG_LEN \
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

    # --- Optional Step 3: Target Only for this segment length (for direct comparison) ---
    # echo "--- Target Only Training ---"
    # python -m src.main \
    #     --source_data $SOURCE_DATA \
    #     --target_data $TARGET_DATA \
    #     --model_type $MODEL_TYPE \
    #     --segment_length $SEG_LEN \
    #     --batch_size $BATCH_SIZE \
    #     --epochs $EPOCHS \
    #     --lr $LR \
    #     --mode target_only \
    #     --results_dir $RESULTS_DIR \
    #     --save_name "${SAVE_NAME_BASE}_target_only" \
    #     --seed $SEED \
    #     --num_classes_source 8 \
    #     --num_classes_target 8

done

echo "===== Segment Length Experiment Finished ====="
