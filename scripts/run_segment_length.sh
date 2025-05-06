#!/bin/bash

# --- Configuration ---
SOURCE_DATA="dataset/source/BPSK-500Kbps-DATA.h5"
TARGET_DATA="dataset/target/TARGET-BPSK-DATA.h5" # Replace
MODEL_TYPE="resnet" # Or 'lstm'
SEGMENT_LENGTHS=(256 512 1024)
BATCH_SIZE=64
EPOCHS=50
LR=1e-3
TOP_LEVEL_RESULTS_DIR="results_main_segment_length" # New top-level dir
SEED=42

echo "===== Running Segment Length Experiment ====="
echo "Model: $MODEL_TYPE, Epochs: $EPOCHS, LR: $LR"
echo "Segment Lengths: ${SEGMENT_LENGTHS[@]}"
echo "Results will be saved in subdirectories under: $TOP_LEVEL_RESULTS_DIR"

for SEG_LEN in "${SEGMENT_LENGTHS[@]}"; do
    EXPERIMENT_NAME_SEG="seglen_${SEG_LEN}_${MODEL_TYPE}" # Base name for this segment length run
    # RESULTS_DIR for this specific segment length
    CURRENT_RESULTS_DIR="${TOP_LEVEL_RESULTS_DIR}/${EXPERIMENT_NAME_SEG}"

    echo "\n----- Testing Segment Length: $SEG_LEN (Results in: $CURRENT_RESULTS_DIR) -----"

    mkdir -p "${CURRENT_RESULTS_DIR}/models"
    mkdir -p "${CURRENT_RESULTS_DIR}/visualizations"
    mkdir -p "${CURRENT_RESULTS_DIR}/logs"

    # --- Step 1: Pre-train on Source for this segment length ---
    SAVE_NAME_PRETRAIN="${EXPERIMENT_NAME_SEG}_source_pretrained"
    PRETRAINED_MODEL_PATH="${CURRENT_RESULTS_DIR}/models/${SAVE_NAME_PRETRAIN}_final.pth.tar"
    echo "--- Pre-training (Save Name: $SAVE_NAME_PRETRAIN) ---"
    python -m src.main \
        --source_data $SOURCE_DATA \
        --target_data $TARGET_DATA \
        --model_type $MODEL_TYPE \
        --segment_length $SEG_LEN \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --lr $LR \
        --mode pretrain \
        --results_dir "$CURRENT_RESULTS_DIR" \
        --save_name "$SAVE_NAME_PRETRAIN" \
        --seed $SEED \
        --num_classes_source 8 \
        --num_classes_target 8

    if [ ! -f "$PRETRAINED_MODEL_PATH" ]; then
        echo "Pretraining failed for segment length $SEG_LEN. Skipping fine-tuning."
        continue
    fi

    # --- Step 2: Fine-tune Classifier Only on Target for this segment length ---
    SAVE_NAME_FINETUNE_CLS="${EXPERIMENT_NAME_SEG}_target_finetuned_classifier_only"
    echo "--- Fine-tuning Classifier (Save Name: $SAVE_NAME_FINETUNE_CLS) ---"
    python -m src.main \
        --source_data $SOURCE_DATA \
        --target_data $TARGET_DATA \
        --model_type $MODEL_TYPE \
        --segment_length $SEG_LEN \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --lr $LR \
        --mode finetune \
        --pretrained_path "$PRETRAINED_MODEL_PATH" \
        --results_dir "$CURRENT_RESULTS_DIR" \
        --save_name "$SAVE_NAME_FINETUNE_CLS" \
        --seed $SEED \
        --num_classes_source 8 \
        --num_classes_target 8

    # --- Optional Step 3: Target Only for this segment length ---
    # SAVE_NAME_TARGET_ONLY="${EXPERIMENT_NAME_SEG}_target_only"
    # echo "--- Target Only Training (Save Name: $SAVE_NAME_TARGET_ONLY) ---"
    # python -m src.main \
    #     --source_data $SOURCE_DATA \
    #     --target_data $TARGET_DATA \
    #     --model_type $MODEL_TYPE \
    #     --segment_length $SEG_LEN \
    #     --batch_size $BATCH_SIZE \
    #     --epochs $EPOCHS \
    #     --lr $LR \
    #     --mode target_only \
    #     --results_dir "$CURRENT_RESULTS_DIR" \
    #     --save_name "$SAVE_NAME_TARGET_ONLY" \
    #     --seed $SEED \
    #     --num_classes_source 8 \
    #     --num_classes_target 8
done

echo "===== Segment Length Experiment Finished ====="