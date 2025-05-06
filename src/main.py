# src/main.py
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import sys  # For logging
import logging  # For logging

from .utils import set_seed, get_args, load_checkpoint, plot_confusion_matrix, plot_tsne, plot_training_curves
from .data_loader import create_dataloaders
from .models import get_model
from .trainer import Trainer


def main():
    args = get_args()  # This now creates args.logs_dir, args.visualizations_dir, args.models_dir
    set_seed(args.seed)

    # --- Configure Logging for this specific run ---
    # args.save_name is the descriptive name from the shell script (e.g., comp_resnet_seg1024_source_pretrained)
    log_file_name = f"{args.save_name}_{args.mode}.log"  # Add mode to distinguish logs if save_name is reused
    log_file_path = os.path.join(args.logs_dir, log_file_name)

    # Remove existing handlers to avoid duplicate logs if this function were called multiple times in the same process
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file_path, mode='w'),  # mode='w' to overwrite for each specific run
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Script execution started. Logging to: {log_file_path}")
    logging.info("----- Configuration -----")
    for k, v in vars(args).items():
        logging.info(f"{k}: {v}")
    logging.info("-------------------------")

    # --- Determine number of classes for models and loaders ---
    num_classes_for_source_model = args.num_classes_source
    source_class_names_map = []  # Will hold names for source classes
    if args.source_classes is not None and len(args.source_classes) > 0:
        num_classes_for_source_model = len(args.source_classes)
    logging.info(f"Effective number of classes for source model/loader: {num_classes_for_source_model}")

    num_classes_for_target_model = args.num_classes_target
    target_class_names_map = []  # Will hold names for target classes
    logging.info(f"Number of classes for target model/loader: {num_classes_for_target_model}")

    # --- Data Loading ---
    logging.info("Loading source data...")
    source_loader, _, _, source_class_names_map = create_dataloaders(
        h5_path=args.source_data,
        segment_length=args.segment_length,
        stride=args.stride,
        batch_size=args.batch_size,
        model_type=args.model_type,
        is_target=False,
        class_indices=args.source_classes,
        num_expected_classes_for_loader=num_classes_for_source_model
    )
    if not source_class_names_map or len(source_class_names_map) != num_classes_for_source_model:
        logging.warning(
            f"Source class names map not fully populated. Using generic names. Got: {source_class_names_map}")
        source_class_names_map = [f"SrcClass {i}" for i in range(num_classes_for_source_model)]

    logging.info("Loading target data...")
    target_train_loader, target_val_loader, target_test_loader, target_class_names_map = create_dataloaders(
        h5_path=args.target_data,
        segment_length=args.segment_length,
        stride=args.stride,
        batch_size=args.batch_size,
        model_type=args.model_type,
        is_target=True,
        split_ratio=args.target_split_ratio,
        seed=args.seed,
        class_indices=None,
        num_expected_classes_for_loader=num_classes_for_target_model
    )
    if not target_class_names_map or len(target_class_names_map) != num_classes_for_target_model:
        logging.warning(
            f"Target class names map not fully populated. Using generic names. Got: {target_class_names_map}")
        target_class_names_map = [f"TgtClass {i}" for i in range(num_classes_for_target_model)]

    logging.info("Data loading complete.")

    # --- Model Definition ---
    # Models are instantiated within each mode block to ensure correct class numbers.
    criterion = nn.CrossEntropyLoss()

    # --- Helper for plotting ---
    def generate_eval_plots(true_labels, pred_labels, features, class_names_list, num_cls, plot_save_prefix,
                            plot_title_suffix):
        if len(true_labels) > 0:
            # Ensure class_names_list is appropriate for the number of classes in pred_labels/true_labels
            max_label_val = max(np.max(true_labels) if len(true_labels) > 0 else -1,
                                np.max(pred_labels) if len(pred_labels) > 0 else -1)

            current_class_names = class_names_list
            if max_label_val >= len(class_names_list):
                logging.warning(
                    f"Max label {max_label_val} exceeds class names len {len(class_names_list)} for {plot_title_suffix}. Extending generic.")
                current_class_names = list(class_names_list) + [f"Class {i}" for i in
                                                                range(len(class_names_list), max_label_val + 1)]

            # Ensure we only pass as many class names as the model was configured for (num_cls)
            # or as many as needed if max_label_val is higher (already handled by extended_class_names)
            # For the CM, we want labels up to num_cls-1.
            cm_class_names = current_class_names[:num_cls]

            plot_confusion_matrix(true_labels, pred_labels, cm_class_names,
                                  os.path.join(args.visualizations_dir, f"cm_{plot_save_prefix}.png"),
                                  title=f"CM {plot_title_suffix}")
            if features.shape[0] > 0:
                tsne_class_names = current_class_names  # t-SNE plot function handles label mapping
                plot_tsne(features, true_labels, tsne_class_names,
                          os.path.join(args.visualizations_dir, f"tsne_{plot_save_prefix}.png"),
                          title=f"t-SNE {plot_title_suffix}")
        else:
            logging.warning(f"No evaluation data (true_labels empty) to generate plots for {plot_title_suffix}.")

    # --- Mode Execution ---
    # args.save_name is the base name for this stage, e.g., "comp_resnet_seg1024_source_pretrained"
    # args.mode is "pretrain", "finetune", etc.

    if args.mode == 'pretrain':
        logging.info("\n--- Mode: Pre-training on Source Domain ---")
        if source_loader is None:
            logging.error("Source loader is None. Cannot pre-train. Exiting.")
            return

        current_model = get_model(args.model_type, num_classes=num_classes_for_source_model)
        optimizer = optim.Adam(current_model.parameters(), lr=args.lr)
        # Pass args.save_name as the prefix for model files
        trainer = Trainer(current_model, criterion, optimizer, args.device, args.results_dir, args.save_name)
        trainer.train(source_loader, None, args.epochs,  # No val_loader for pretrain simplicity
                      visualizations_dir=args.visualizations_dir,
                      plot_title_prefix=f"Pretrain ({args.model_type})")
        logging.info("Pre-training finished.")
        # Optionally, evaluate pre-trained model on source data (as a sanity check)
        # _, _, src_true, src_pred, src_feat = trainer.evaluate(source_loader, "Eval Pretrained on Source", True)
        # generate_eval_plots(src_true, src_pred, src_feat, source_class_names_map, num_classes_for_source_model,
        #                     f"{args.save_name}_eval_on_source", f"Pretrained on Source ({args.model_type})")


    elif args.mode == 'target_only':
        logging.info("\n--- Mode: Training on Target Domain Only (From Scratch) ---")
        if target_train_loader is None:
            logging.error("Target train loader is None. Exiting.")
            return

        current_model = get_model(args.model_type, num_classes=num_classes_for_target_model)
        optimizer = optim.Adam(current_model.parameters(), lr=args.lr)
        trainer = Trainer(current_model, criterion, optimizer, args.device, args.results_dir, args.save_name)
        trainer.train(target_train_loader, target_val_loader, args.epochs,
                      visualizations_dir=args.visualizations_dir,
                      plot_title_prefix=f"Target Only ({args.model_type})")

        if target_test_loader:
            logging.info("Evaluating on target test set (Target Only)...")
            test_loss, test_acc, true_labels, pred_labels, features = trainer.evaluate(
                target_test_loader, desc="Testing (Target Only)", get_features_for_tsne=True)
            logging.info(f"Target Only - Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
            generate_eval_plots(true_labels, pred_labels, features, target_class_names_map,
                                num_classes_for_target_model,
                                args.save_name, f"Target Only ({args.model_type})")
        else:
            logging.warning("No target test loader available for evaluation (Target Only).")


    elif args.mode == 'finetune' or args.mode == 'eval_pretrained':
        if not args.pretrained_path or not os.path.exists(args.pretrained_path):
            logging.error(f"Pretrained model path not found: {args.pretrained_path}. Exiting.")
            return

        logging.info(f"Loading pre-trained weights from: {args.pretrained_path} for mode {args.mode}")
        checkpoint = torch.load(args.pretrained_path, map_location=args.device)

        loaded_model_fc_out_features = None
        if 'state_dict' in checkpoint:
            pretrained_s_dict = checkpoint['state_dict']
            for key_prefix in ['fc', 'classifier']:  # Common FC layer prefixes
                if f'{key_prefix}.weight' in pretrained_s_dict:
                    loaded_model_fc_out_features = pretrained_s_dict[f'{key_prefix}.weight'].shape[0]
                    break
                elif f'{key_prefix}.bias' in pretrained_s_dict and loaded_model_fc_out_features is None:
                    loaded_model_fc_out_features = pretrained_s_dict[f'{key_prefix}.bias'].shape[0]
                    break
        if loaded_model_fc_out_features is None:
            logging.warning(
                "Could not determine output features of pre-trained model's FC. Assuming it matches source class count used during its training.")
            # This assumption relies on the pretraining call having used the right --num_classes_source if --source_classes was used.
            # For fine-tuning, what matters most is that the backbone is loaded.
            # For eval_pretrained, if FC is different, it should be re-initialized or skipped.
            # Let's assume the pre-trained model was trained with num_classes_for_source_model (from its own args)
            # This is tricky if the shell script passes a different --num_classes_source for eval_pretrained than what was used for pretrain.
            # Best to rely on the loaded_model_fc_out_features. If still None, it means the pretrain used the default num_classes_source.
            # We'll use args.num_classes_source as passed to THIS run of main.py, for loading decision.
            loaded_model_fc_out_features = args.num_classes_source  # Fallback to current run's source class expectation

        current_model = get_model(args.model_type,
                                  num_classes=num_classes_for_target_model)  # Target model always has target classes
        model_dict = current_model.state_dict()

        final_load_dict = {}
        if 'state_dict' in checkpoint:
            pretrained_s_dict = checkpoint['state_dict']
            for k, v in pretrained_s_dict.items():
                is_fc_layer = k.startswith('fc.') or k.startswith('classifier.')
                if is_fc_layer:
                    # Load FC if pre-trained FC matches target model's FC configuration
                    if loaded_model_fc_out_features == num_classes_for_target_model and k in model_dict and model_dict[
                        k].shape == v.shape:
                        final_load_dict[k] = v
                        logging.info(f"Loading FC layer '{k}' (compatible).")
                    else:
                        logging.info(
                            f"Skipping FC layer '{k}' from pre-trained: class/shape mismatch. (Loaded model FC classes: {loaded_model_fc_out_features}, Target model FC classes: {num_classes_for_target_model}). It will be re-initialized.")
                elif k in model_dict and model_dict[k].shape == v.shape:  # Backbone layers
                    final_load_dict[k] = v
                else:
                    logging.warning(f"Skipping layer {k} from pre-trained: not in target model or shape mismatch.")

            current_model.load_state_dict(final_load_dict, strict=False)
            logging.info("Weights loaded into target model architecture.")
        else:
            logging.error("Key 'state_dict' not found in checkpoint. Cannot load weights.")
            return

        current_model = current_model.to(args.device)

        if args.mode == 'finetune':
            # The logic for finetuning only classifier was already implemented based on previous request
            logging.info("\n--- Mode: Fine-tuning on Target Domain (Classifier Only by default) ---")
            if target_train_loader is None:
                logging.error("Target train loader is None. Exiting.")
                return

            logging.info("Freezing backbone layers. Only training the classifier head.")
            for name, param in current_model.named_parameters():
                is_classifier_layer = name.startswith('fc.') or name.startswith('classifier.')
                if is_classifier_layer:
                    param.requires_grad = True
                    logging.info(f"  - Classifier layer '{name}' set to trainable.")
                else:
                    param.requires_grad = False

            trainable_params = [p for p in current_model.parameters() if p.requires_grad]
            if not trainable_params:
                logging.critical(
                    "No trainable parameters found for the classifier head. Check model layer names ('fc.' or 'classifier.'). Unfreezing all as a fallback, but this is not the intended fine-tuning strategy.")
                for param in current_model.parameters(): param.requires_grad = True
                trainable_params = list(current_model.parameters())

            optimizer = optim.Adam(trainable_params, lr=args.lr)
            logging.info(f"Optimizer configured for {len(trainable_params)} trainable parameters (classifier head).")

            trainer = Trainer(current_model, criterion, optimizer, args.device, args.results_dir, args.save_name)
            trainer.train(target_train_loader, target_val_loader, args.epochs,
                          visualizations_dir=args.visualizations_dir,
                          plot_title_prefix=f"Finetune Classifier ({args.model_type})")

            if target_test_loader:
                logging.info("Evaluating fine-tuned model (classifier only) on target test set...")
                test_loss, test_acc, true_labels, pred_labels, features = trainer.evaluate(
                    target_test_loader, desc="Testing (Fine-tuned Classifier)", get_features_for_tsne=True)
                logging.info(f"Fine-tuned Classifier - Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
                generate_eval_plots(true_labels, pred_labels, features, target_class_names_map,
                                    num_classes_for_target_model,
                                    args.save_name, f"Finetuned Classifier ({args.model_type})")
            else:
                logging.warning("No target test loader available for evaluation (Fine-tuned Classifier).")

        elif args.mode == 'eval_pretrained':
            logging.info("\n--- Mode: Evaluating Pre-trained Model on Target Domain (No Fine-tuning) ---")
            if loaded_model_fc_out_features != num_classes_for_target_model:
                logging.warning(
                    f"Pre-trained model's FC layer ({loaded_model_fc_out_features} classes) differs from target ({num_classes_for_target_model}). FC layer in current_model was likely re-initialized.")

            if target_test_loader is None:
                logging.error("Target test loader is None. Exiting.")
                return

            trainer = Trainer(current_model, criterion, None, args.device, args.results_dir,
                              "")  # No optimizer, save_name_prefix not used for model saving here
            logging.info("Evaluating pre-trained model on target test set...")
            test_loss, test_acc, true_labels, pred_labels, features = trainer.evaluate(
                target_test_loader, desc="Testing (Eval Pre-trained)", get_features_for_tsne=True)
            logging.info(f"Eval Pre-trained - Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
            generate_eval_plots(true_labels, pred_labels, features, target_class_names_map,
                                num_classes_for_target_model,
                                args.save_name, f"Eval Pretrained ({args.model_type})")
    else:
        logging.error(f"Unknown mode: {args.mode}")

    logging.info(f"Script execution finished for mode '{args.mode}' with save_name '{args.save_name}'.")


if __name__ == '__main__':
    main()