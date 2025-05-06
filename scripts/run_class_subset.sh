#!/bin/bash

# --- Configuration ---
SOURCE_DATA="dataset/source/BPSK-500Kbps-DATA.h5"
TARGET_DATA="dataset/target/TARGET-BPSK-DATA.h5" # Replace with your target file name
MODEL_TYPE="resnet" # Or 'lstm'
SEGMENT_LENGTH=1024 # Choose a fixed segment length for this experiment
BATCH_SIZE=64
EPOCHS=50 # Adjust as needed
LR=1e-3
BASE_RESULTS_DIR="results/class_subset_${MODEL_TYPE}_seg${SEGMENT_LENGTH}"
SEED=42

# Define the original labels present in the source data (based on your previous output)
# IMPORTANT: Adjust these if your actual labels are different
ALL_SOURCE_LABELS=(0 2 4 6 8 10 12 14)

echo "===== Running Class Subset Experiment ====="
echo "Model: $MODEL_TYPE, Segment Length: $SEGMENT_LENGTH"
echo "Results will be saved in subdirectories under: $BASE_RESULTS_DIR"

# Loop from 2 classes up to the total number of classes (8)
for K in $(seq 2 8); do
    # Select the first K labels from the list
    CLASS_SUBSET=("${ALL_SOURCE_LABELS[@]:0:$K}")
    CLASS_SUBSET_STR=$(IFS=" "; echo "${CLASS_SUBSET[*]}") # Format for command line argument

    echo "\n----- Testing with $K Source Classes: [${CLASS_SUBSET_STR}] -----"

    RESULTS_DIR="${BASE_RESULTS_DIR}/subset_${K}_classes"
    SAVE_NAME_BASE="subset_${K}_${MODEL_TYPE}"
    PRETRAINED_MODEL_PATH="${RESULTS_DIR}/models/${SAVE_NAME_BASE}_source_pretrained_final.pth.tar"

    mkdir -p $RESULTS_DIR/models

    # --- Step 1: Pre-train on Source Subset ---
    echo "--- Pre-training on ${K} classes ---"
    python -m src.main \
        --source_data $SOURCE_DATA \
        --target_data $TARGET_DATA \
        --model_type $MODEL_TYPE \
        --segment_length $SEGMENT_LENGTH \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --lr $LR \
        --mode pretrain \
        --source_classes $CLASS_SUBSET_STR \
        --num_classes_source $K \
        --num_classes_target 8 \
        --results_dir $RESULTS_DIR \
        --save_name "${SAVE_NAME_BASE}_source_pretrained" \
        --seed $SEED

    if [ ! -f "$PRETRAINED_MODEL_PATH" ]; then
        echo "Pretraining failed for $K classes. Skipping evaluation."
        continue # Skip to next K
    fi

    # --- Step 2: Evaluate Pre-trained Model on Full Target Set ---
    # Note: We use 'eval_pretrained' here to see the direct generalization.
    # Fine-tuning could also be done, but eval shows raw transfer capability.
    echo "--- Evaluating pre-trained model (from $K classes) on full target test set ---"
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
        --num_classes_source $K \
        --num_classes_target 8 \
        --results_dir $RESULTS_DIR \
        --seed $SEED

    # --- Optional: Fine-tune the model pre-trained on K classes ---
    # echo "--- Fine-tuning model (from $K classes) on full target train set ---"
    # python -m src.main \
    #     --source_data $SOURCE_DATA \
    #     --target_data $TARGET_DATA \
    #     --model_type $MODEL_TYPE \
    #     --segment_length $SEGMENT_LENGTH \
    #     --batch_size $BATCH_SIZE \
    #     --epochs $EPOCHS \
    #     --lr $LR \
    #     --mode finetune \
    #     --pretrained_path $PRETRAINED_MODEL_PATH \
    #     --num_classes_source $K \
    #     --num_classes_target 8 \
    #     --results_dir $RESULTS_DIR \
    #     --save_name "${SAVE_NAME_BASE}_target_finetuned" \
    #     --seed $SEED

done

echo "===== Class Subset Experiment Finished ====="