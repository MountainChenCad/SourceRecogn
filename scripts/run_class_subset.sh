#!/bin/bash

# --- Configuration ---
SOURCE_DATA="dataset/source/BPSK-500Kbps-DATA.h5"
TARGET_DATA="dataset/target/TARGET-BPSK-DATA.h5" # Replace
MODEL_TYPE="resnet" # Or 'lstm'
SEGMENT_LENGTH=1024 # Fixed segment length
BATCH_SIZE=64
EPOCHS=50
LR=1e-3
TOP_LEVEL_RESULTS_DIR="results_main_class_subset" # New top-level dir
SEED=42

# IMPORTANT: These are the ORIGINAL label values from your source HDF5 file
# that you want to select for the subsets.
# The data_loader.py will remap these to 0 to K-1 for the model.
ALL_SOURCE_H5_LABELS=(0 2 4 6 8 10 12 14) # Original labels as in the HDF5

echo "===== Running Class Subset Experiment ====="
echo "Model: $MODEL_TYPE, Segment Length: $SEGMENT_LENGTH, Epochs: $EPOCHS, LR: $LR"
echo "Results will be saved in subdirectories under: $TOP_LEVEL_RESULTS_DIR"

# Loop from 2 classes up to the total number of unique source H5 labels (8 in this case)
for K_CLASSES_COUNT in $(seq 2 ${#ALL_SOURCE_H5_LABELS[@]}); do
    # Select the first K_CLASSES_COUNT original H5 labels for this subset
    CURRENT_SOURCE_H5_LABEL_SUBSET=("${ALL_SOURCE_H5_LABELS[@]:0:$K_CLASSES_COUNT}")
    # Format for --source_classes argument (space separated string of original H5 labels)
    SOURCE_CLASSES_ARG_STR=$(IFS=" "; echo "${CURRENT_SOURCE_H5_LABEL_SUBSET[*]}")

    EXPERIMENT_NAME_SUBSET="subset_${K_CLASSES_COUNT}classes_${MODEL_TYPE}_seg${SEGMENT_LENGTH}"
    # RESULTS_DIR for this specific class subset run
    CURRENT_RESULTS_DIR="${TOP_LEVEL_RESULTS_DIR}/${EXPERIMENT_NAME_SUBSET}"

    echo "\n----- Testing with $K_CLASSES_COUNT Source Classes (Original H5 Labels: [${SOURCE_CLASSES_ARG_STR}]) -----"
    echo "----- Results in: $CURRENT_RESULTS_DIR -----"

    mkdir -p "${CURRENT_RESULTS_DIR}/models"
    mkdir -p "${CURRENT_RESULTS_DIR}/visualizations"
    mkdir -p "${CURRENT_RESULTS_DIR}/logs"

    # --- Step 1: Pre-train on Source Subset ---
    SAVE_NAME_PRETRAIN="${EXPERIMENT_NAME_SUBSET}_source_pretrained"
    PRETRAINED_MODEL_PATH="${CURRENT_RESULTS_DIR}/models/${SAVE_NAME_PRETRAIN}_final.pth.tar"
    echo "--- Pre-training on $K_CLASSES_COUNT classes (Save Name: $SAVE_NAME_PRETRAIN) ---"
    python -m src.main \
        --source_data $SOURCE_DATA \
        --target_data $TARGET_DATA \
        --model_type $MODEL_TYPE \
        --segment_length $SEGMENT_LENGTH \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --lr $LR \
        --mode pretrain \
        --source_classes $SOURCE_CLASSES_ARG_STR \
        --num_classes_source $K_CLASSES_COUNT \
        --num_classes_target 8 \
        --results_dir "$CURRENT_RESULTS_DIR" \
        --save_name "$SAVE_NAME_PRETRAIN" \
        --seed $SEED

    if [ ! -f "$PRETRAINED_MODEL_PATH" ]; then
        echo "Pretraining failed for $K_CLASSES_COUNT classes. Skipping further steps for this subset."
        continue
    fi

    # --- Step 2: Evaluate Pre-trained Model on Full Target Set ---
    SAVE_NAME_EVAL_PRETRAINED="${EXPERIMENT_NAME_SUBSET}_eval_pretrained_on_target"
    echo "--- Evaluating pre-trained model (from $K_CLASSES_COUNT classes) on full target (Save Name: $SAVE_NAME_EVAL_PRETRAINED) ---"
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
        --num_classes_source $K_CLASSES_COUNT \
        --num_classes_target 8 \
        --results_dir "$CURRENT_RESULTS_DIR" \
        --save_name "$SAVE_NAME_EVAL_PRETRAINED" \
        --seed $SEED

    # --- Step 3: Fine-tune Classifier Only on Full Target Set ---
    SAVE_NAME_FINETUNE_CLS="${EXPERIMENT_NAME_SUBSET}_target_finetuned_classifier_only"
    echo "--- Fine-tuning Classifier (from $K_CLASSES_COUNT classes pretrain) on full target (Save Name: $SAVE_NAME_FINETUNE_CLS) ---"
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
        --num_classes_source $K_CLASSES_COUNT \
        --num_classes_target 8 \
        --results_dir "$CURRENT_RESULTS_DIR" \
        --save_name "$SAVE_NAME_FINETUNE_CLS" \
        --seed $SEED
done

echo "===== Class Subset Experiment Finished ====="