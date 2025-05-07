# src/main.py
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import sys
import logging

from .utils import set_seed, get_args, load_checkpoint, plot_confusion_matrix, plot_tsne, plot_training_curves
from .data_loader import create_dataloaders
from .models import get_model
from .trainer import Trainer


def main():
    args = get_args()
    set_seed(args.seed)

    # --- Configure Logging for this specific run ---
    log_file_name = f"{args.save_name}_{args.mode}.log"
    log_file_path = os.path.join(args.logs_dir, log_file_name)
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file_path, mode='w'), logging.StreamHandler(sys.stdout)]
    )
    logging.info(f"Script execution started. Logging to: {log_file_path}")
    logging.info("----- Configuration -----")
    for k, v in vars(args).items(): logging.info(f"{k}: {v}")
    logging.info("-------------------------")

    # --- Determine number of classes for models and loaders ---
    num_classes_for_source_model = args.num_classes_source
    if args.source_classes is not None and len(args.source_classes) > 0:
        num_classes_for_source_model = len(args.source_classes)
    logging.info(f"Effective number of classes for source model/loader: {num_classes_for_source_model}")
    num_classes_for_target_model = args.num_classes_target
    logging.info(f"Number of classes for target model/loader: {num_classes_for_target_model}")

    # --- Data Loading ---
    logging.info("Loading source data...")
    source_loader, _, _, source_class_names_map = create_dataloaders(
        h5_path=args.source_data, segment_length=args.segment_length, stride=args.stride,
        batch_size=args.batch_size, model_type=args.model_type, is_target=False,
        class_indices=args.source_classes, num_expected_classes_for_loader=num_classes_for_source_model
    )
    if not source_class_names_map or len(source_class_names_map) != num_classes_for_source_model:
        logging.warning(
            f"Source class names map not fully populated or mismatched. Using generic. Got {len(source_class_names_map)} names for {num_classes_for_source_model} classes.")
        source_class_names_map = [f"SrcClass {i}" for i in range(num_classes_for_source_model)]

    logging.info("Loading target data...")
    target_train_loader, target_val_loader, target_test_loader, target_class_names_map = create_dataloaders(
        h5_path=args.target_data, segment_length=args.segment_length, stride=args.stride,
        batch_size=args.batch_size, model_type=args.model_type, is_target=True,
        split_ratio=args.target_split_ratio, seed=args.seed, class_indices=None,
        num_expected_classes_for_loader=num_classes_for_target_model
    )
    if not target_class_names_map or len(target_class_names_map) != num_classes_for_target_model:
        logging.warning(
            f"Target class names map not fully populated or mismatched. Using generic. Got {len(target_class_names_map)} names for {num_classes_for_target_model} classes.")
        target_class_names_map = [f"TgtClass {i}" for i in range(num_classes_for_target_model)]
    logging.info("Data loading complete.")

    criterion = nn.CrossEntropyLoss()

    def generate_eval_plots(true_labels, pred_labels, features, class_names_list, num_cls_model, plot_save_prefix,
                            plot_title_suffix):
        if len(true_labels) > 0:
            max_label_val_data = max(np.max(true_labels) if len(true_labels) > 0 else -1,
                                     np.max(pred_labels) if len(pred_labels) > 0 else -1)

            # Effective class names for plotting, considering data might have fewer unique labels than model capacity
            effective_class_names = list(class_names_list)  # Start with what's provided
            if max_label_val_data >= len(effective_class_names):
                logging.warning(
                    f"Max label in data ({max_label_val_data}) >= len of provided class names ({len(effective_class_names)}) for {plot_title_suffix}. Extending with generic names.")
                effective_class_names.extend(
                    [f"Label {i}" for i in range(len(effective_class_names), max_label_val_data + 1)])

            # For confusion matrix, use names up to num_cls_model (model's configured output classes)
            cm_plot_names = effective_class_names[:num_cls_model]

            plot_confusion_matrix(true_labels, pred_labels, cm_plot_names,
                                  os.path.join(args.visualizations_dir, f"cm_{plot_save_prefix}.png"),
                                  title=f"CM {plot_title_suffix}")
            if features.shape[0] > 0:
                # For t-SNE, use class names that cover all actual labels present in the data
                tsne_plot_names = effective_class_names
                plot_tsne(features, true_labels, tsne_plot_names,
                          os.path.join(args.visualizations_dir, f"tsne_{plot_save_prefix}.png"),
                          title=f"t-SNE {plot_title_suffix}")
        else:
            logging.warning(f"No evaluation data (true_labels empty) to generate plots for {plot_title_suffix}.")

    # --- Mode Execution ---
    if args.mode == 'pretrain':
        logging.info("\n--- Mode: Pre-training on Source Domain ---")
        if source_loader is None:
            logging.error("Source loader is None. Cannot pre-train. Exiting.")
            return

        current_model = get_model(args.model_type, num_classes=num_classes_for_source_model)
        optimizer = optim.Adam(current_model.parameters(), lr=args.lr)
        trainer = Trainer(current_model, criterion, optimizer, args.device, args.results_dir, args.save_name)
        trainer.train(source_loader, None, args.epochs,  # No val_loader for pretrain
                      visualizations_dir=args.visualizations_dir,
                      plot_title_prefix=f"Pretrain ({args.model_type}) on Source")
        logging.info("Pre-training finished.")

        # --- ADDED: Evaluate pre-trained model on the source domain data itself ---
        logging.info("Evaluating pre-trained model on the Source Domain Data...")
        # Create a new trainer instance for evaluation if optimizer state is not needed, or reuse
        # For simplicity, we can reuse the trainer instance as it only uses model.eval()
        src_test_loss, src_test_acc, src_true_labels, src_pred_labels, src_features = trainer.evaluate(
            source_loader, desc="Testing Pretrained Model on Source Data", get_features_for_tsne=True
        )
        logging.info(f"Pretrained Model on Source Data - Test Loss: {src_test_loss:.4f}, Test Acc: {src_test_acc:.2f}%")

        # Define a save prefix for these source evaluation plots
        source_eval_plot_save_prefix = f"{args.save_name}_eval_on_source_after_pretrain"
        source_eval_plot_title_suffix = f"Pretrained on Source ({args.model_type})"

        generate_eval_plots(src_true_labels, src_pred_labels, src_features,
                            source_class_names_map, num_classes_for_source_model,
                            source_eval_plot_save_prefix, source_eval_plot_title_suffix)
        # --- END OF ADDED SECTION ---

    elif args.mode == 'target_only':
        # ... (target_only logic as before) ...
        logging.info("\n--- Mode: Training on Target Domain Only (From Scratch) ---")
        if target_train_loader is None: logging.error("Target train loader is None. Exiting."); return
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
        # ... (logic for loading pretrained model as before) ...
        if not args.pretrained_path or not os.path.exists(args.pretrained_path):
            logging.error(f"Pretrained model path not found: {args.pretrained_path}. Exiting.");
            return
        logging.info(f"Loading pre-trained weights from: {args.pretrained_path} for mode {args.mode}")
        checkpoint = torch.load(args.pretrained_path, map_location=args.device)
        loaded_model_fc_out_features = None
        if 'state_dict' in checkpoint:
            pretrained_s_dict = checkpoint['state_dict']
            for key_prefix in ['fc', 'classifier']:
                if f'{key_prefix}.weight' in pretrained_s_dict:
                    loaded_model_fc_out_features = pretrained_s_dict[f'{key_prefix}.weight'].shape[0]; break
                elif f'{key_prefix}.bias' in pretrained_s_dict and loaded_model_fc_out_features is None:
                    loaded_model_fc_out_features = pretrained_s_dict[f'{key_prefix}.bias'].shape[0]; break
        if loaded_model_fc_out_features is None:
            logging.warning(
                "Could not determine output features of pre-trained model's FC. Assuming it matches current run's source class expectation.")
            loaded_model_fc_out_features = args.num_classes_source

        current_model = get_model(args.model_type, num_classes=num_classes_for_target_model)
        model_dict = current_model.state_dict()
        final_load_dict = {}
        if 'state_dict' in checkpoint:
            pretrained_s_dict = checkpoint['state_dict']
            for k, v in pretrained_s_dict.items():
                is_fc_layer = k.startswith('fc.') or k.startswith('classifier.')
                if is_fc_layer:
                    if loaded_model_fc_out_features == num_classes_for_target_model and k in model_dict and model_dict[
                        k].shape == v.shape:
                        final_load_dict[k] = v;
                        logging.info(f"Loading FC layer '{k}' (compatible).")
                    else:
                        logging.info(
                            f"Skipping FC layer '{k}' from pre-trained: class/shape mismatch. (Loaded FC: {loaded_model_fc_out_features}, Target FC: {num_classes_for_target_model}). Will be re-initialized.")
                elif k in model_dict and model_dict[k].shape == v.shape:
                    final_load_dict[k] = v
                else:
                    logging.warning(f"Skipping layer {k} from pre-trained: not in target model or shape mismatch.")
            current_model.load_state_dict(final_load_dict, strict=False)
            logging.info("Weights loaded into target model architecture.")
        else:
            logging.error("Key 'state_dict' not found in checkpoint. Cannot load weights."); return
        current_model = current_model.to(args.device)

        if args.mode == 'finetune':
            # ... (finetune logic as before, ensuring only classifier is trained) ...
            logging.info("\n--- Mode: Fine-tuning on Target Domain (Classifier Only by default) ---")
            if target_train_loader is None: logging.error("Target train loader is None. Exiting."); return
            logging.info("Freezing backbone layers. Only training the classifier head.")
            for name, param in current_model.named_parameters():
                is_classifier_layer = name.startswith('fc.') or name.startswith('classifier.')
                param.requires_grad = True if is_classifier_layer else False
                if is_classifier_layer and param.requires_grad: logging.info(
                    f"  - Classifier layer '{name}' set to trainable.")
            trainable_params = [p for p in current_model.parameters() if p.requires_grad]
            if not trainable_params:
                logging.critical("No trainable parameters found for classifier head. Unfreezing all as fallback.");
                [p.requires_grad_() for p in current_model.parameters()];
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
            # ... (eval_pretrained logic as before) ...
            logging.info("\n--- Mode: Evaluating Pre-trained Model on Target Domain (No Fine-tuning) ---")
            if loaded_model_fc_out_features != num_classes_for_target_model: logging.warning(
                f"Pre-trained model's FC ({loaded_model_fc_out_features} cls) differs from target ({num_classes_for_target_model} cls). FC likely re-initialized.")
            if target_test_loader is None: logging.error("Target test loader is None. Exiting."); return
            trainer = Trainer(current_model, criterion, None, args.device, args.results_dir, "")
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