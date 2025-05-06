# src/main.py
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np # Add numpy

from .utils import set_seed, get_args, load_checkpoint, plot_confusion_matrix, plot_tsne # Import plot functions
from .data_loader import create_dataloaders
from .models import get_model
from .trainer import Trainer


def main():
    args = get_args()
    set_seed(args.seed)

    # --- Determine number of classes for models and loaders ---
    num_classes_for_source_model = args.num_classes_source
    if args.source_classes is not None and len(args.source_classes) > 0:
        num_classes_for_source_model = len(args.source_classes)
        print(
            f"Using subset of source classes. Number of classes for source model/loader: {num_classes_for_source_model}")
    else:
        print(
            f"Using all classes from source. Number of classes for source model/loader: {num_classes_for_source_model}")

    num_classes_for_target_model = args.num_classes_target
    print(f"Number of classes for target model/loader: {num_classes_for_target_model}")

    # --- Data Loading ---
    print("Loading source data...")
    source_loader, _, _ = create_dataloaders(
        h5_path=args.source_data,
        segment_length=args.segment_length,
        stride=args.stride,
        batch_size=args.batch_size,
        model_type=args.model_type,
        is_target=False,
        class_indices=args.source_classes,
        num_expected_classes_for_loader=num_classes_for_source_model
    )

    print("Loading target data...")
    target_train_loader, target_val_loader, target_test_loader = create_dataloaders(
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
    print("Data loading complete.")

    # --- Model Definition ---
    print(f"Creating model: {args.model_type}")
    source_model = get_model(args.model_type, num_classes=num_classes_for_source_model)
    target_model = get_model(args.model_type, num_classes=num_classes_for_target_model)
    print(f"Source Model: {num_classes_for_source_model} classes, Target Model: {num_classes_for_target_model} classes")

    criterion = nn.CrossEntropyLoss()

    # --- Mode Execution ---
    if args.mode == 'pretrain':
        print("\n--- Mode: Pre-training on Source Domain ---")
        if source_loader is None:
            print("ERROR: Source loader is None. Cannot pre-train. Exiting.")
            return
        optimizer = optim.Adam(source_model.parameters(), lr=args.lr)
        trainer = Trainer(source_model, criterion, optimizer, args.device, args.results_dir,
                          f"{args.save_name}_source_pretrained")
        # Pass visualizations_dir and num_classes for potential future use, though not typically plotted during pretrain
        trainer.train(source_loader, None, args.epochs,
                      visualizations_dir=args.visualizations_dir,
                      num_classes_for_plot=num_classes_for_source_model)
        print("Pre-training finished.")

    elif args.mode == 'target_only':
        print("\n--- Mode: Training on Target Domain Only (From Scratch) ---")
        if target_train_loader is None:
            print("ERROR: Target train loader is None. Exiting.")
            return

        current_model = get_model(args.model_type, num_classes=num_classes_for_target_model)
        optimizer = optim.Adam(current_model.parameters(), lr=args.lr)
        trainer = Trainer(current_model, criterion, optimizer, args.device, args.results_dir,
                          f"{args.save_name}_target_only")
        trainer.train(target_train_loader, target_val_loader, args.epochs,
                      visualizations_dir=args.visualizations_dir,
                      num_classes_for_plot=num_classes_for_target_model)

        if target_test_loader:
            print("Evaluating on target test set (Target Only)...")
            # Pass get_features_for_tsne=True for evaluation
            test_loss, test_acc, true_labels, pred_labels, features = trainer.evaluate(
                target_test_loader, desc="Testing (Target Only)", get_features_for_tsne=True
            )
            print(f"Target Only - Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

            if len(true_labels) > 0:  # Check if evaluation produced results
                class_names = [f"Class {i}" for i in range(num_classes_for_target_model)]
                plot_confusion_matrix(true_labels, pred_labels, class_names,
                                      os.path.join(args.visualizations_dir, f"cm_{args.save_name}_target_only.png"),
                                      title=f"Confusion Matrix (Target Only - {args.model_type})")
                if features.shape[0] > 0:
                    plot_tsne(features, true_labels, class_names,
                              os.path.join(args.visualizations_dir, f"tsne_{args.save_name}_target_only.png"),
                              title=f"t-SNE (Target Only - {args.model_type})")
        else:
            print("No target test loader available for evaluation (Target Only).")


    elif args.mode == 'finetune' or args.mode == 'eval_pretrained':
        if not args.pretrained_path or not os.path.exists(args.pretrained_path):
            print(f"ERROR: Pretrained model path not found: {args.pretrained_path}. Exiting.")
            return

        print(f"Loading pre-trained weights from: {args.pretrained_path} for mode {args.mode}")
        checkpoint = torch.load(args.pretrained_path, map_location=args.device)

        # Determine num_classes of the loaded pre-trained model's FC layer
        # This is a heuristic: check the shape of the fc.weight or fc.bias
        loaded_model_fc_out_features = None
        if 'state_dict' in checkpoint:
            pretrained_s_dict = checkpoint['state_dict']
            for key in ['fc.weight', 'classifier.weight']:  # Common FC layer names
                if key in pretrained_s_dict:
                    loaded_model_fc_out_features = pretrained_s_dict[key].shape[0]
                    break
            if loaded_model_fc_out_features is None and 'fc.bias' in pretrained_s_dict:  # Try bias
                loaded_model_fc_out_features = pretrained_s_dict['fc.bias'].shape[0]

        if loaded_model_fc_out_features is None:
            print(
                "Warning: Could not determine output features of pre-trained model's FC layer. Assuming it matches source class count.")
            loaded_model_fc_out_features = num_classes_for_source_model

        # Instantiate the target model architecture with the correct number of target classes
        current_model = get_model(args.model_type, num_classes=num_classes_for_target_model)
        model_dict = current_model.state_dict()

        # Filter pretrained_dict: load only backbone, or FC if classes match
        final_load_dict = {}
        if 'state_dict' in checkpoint:
            pretrained_s_dict = checkpoint['state_dict']
            for k, v in pretrained_s_dict.items():
                if k.startswith('fc.') or k.startswith('classifier.'):  # FC layer names
                    if loaded_model_fc_out_features == num_classes_for_target_model and k in model_dict and model_dict[
                        k].shape == v.shape:
                        final_load_dict[k] = v
                        print(f"Loading FC layer '{k}' (compatible).")
                    else:
                        print(
                            f"Skipping FC layer '{k}' from pre-trained: class/shape mismatch (loaded={loaded_model_fc_out_features}, target={num_classes_for_target_model}). Will be re-initialized.")
                elif k in model_dict and model_dict[k].shape == v.shape:
                    final_load_dict[k] = v
                else:
                    print(f"Skipping layer {k} from pre-trained: not in target model or shape mismatch.")

            current_model.load_state_dict(final_load_dict, strict=False)  # strict=False because FC might be missing
            print("Weights loaded into target model architecture.")
        else:
            print("Warning: 'state_dict' not found in checkpoint. Cannot load weights.")

        current_model = current_model.to(args.device)

        if args.mode == 'finetune':
            print("\n--- Mode: Fine-tuning on Target Domain ---")
            if target_train_loader is None:
                print("ERROR: Target train loader is None. Exiting.")
                return

            # Optimizer setup (freeze backbone or not)
            if args.freeze_backbone:
                print("Freezing backbone layers for fine-tuning.")
                for name, param in current_model.named_parameters():
                    if not (name.startswith('fc.') or name.startswith('classifier.')):  # Check common FC names
                        param.requires_grad = False
                    else:  # FC layer should be trainable
                        param.requires_grad = True
                # Ensure there are parameters to optimize
                trainable_params = filter(lambda p: p.requires_grad, current_model.parameters())
                if not list(trainable_params):  # Re-filter to check
                    print("Warning: No trainable parameters found after freezing. Unfreezing all.")
                    for param in current_model.parameters(): param.requires_grad = True

                optimizer = optim.Adam(filter(lambda p: p.requires_grad, current_model.parameters()), lr=args.lr)
            else:
                print("Fine-tuning all layers.")
                for param in current_model.parameters(): param.requires_grad = True
                optimizer = optim.Adam(current_model.parameters(), lr=args.lr)

            trainer = Trainer(current_model, criterion, optimizer, args.device, args.results_dir,
                              f"{args.save_name}_target_finetuned")
            trainer.train(target_train_loader, target_val_loader, args.epochs,
                          visualizations_dir=args.visualizations_dir,
                          num_classes_for_plot=num_classes_for_target_model)

            if target_test_loader:
                print("Evaluating fine-tuned model on target test set...")
                test_loss, test_acc, true_labels, pred_labels, features = trainer.evaluate(
                    target_test_loader, desc="Testing (Fine-tuned)", get_features_for_tsne=True
                )
                print(f"Fine-tuned - Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
                if len(true_labels) > 0:
                    class_names = [f"Class {i}" for i in range(num_classes_for_target_model)]
                    plot_confusion_matrix(true_labels, pred_labels, class_names,
                                          os.path.join(args.visualizations_dir,
                                                       f"cm_{args.save_name}_target_finetuned.png"),
                                          title=f"Confusion Matrix (Fine-tuned - {args.model_type})")
                    if features.shape[0] > 0:
                        plot_tsne(features, true_labels, class_names,
                                  os.path.join(args.visualizations_dir, f"tsne_{args.save_name}_target_finetuned.png"),
                                  title=f"t-SNE (Fine-tuned - {args.model_type})")
            else:
                print("No target test loader available for evaluation (Fine-tuned).")


        elif args.mode == 'eval_pretrained':
            print("\n--- Mode: Evaluating Pre-trained Model on Target Domain (No Fine-tuning) ---")
            if loaded_model_fc_out_features != num_classes_for_target_model:
                print(
                    f"Warning: Pre-trained model's FC layer had {loaded_model_fc_out_features} classes. Target needs {num_classes_for_target_model}. "
                    "The FC layer in the current model was likely re-initialized if not compatible.")

            if target_test_loader is None:
                print("ERROR: Target test loader is None. Exiting.")
                return

            trainer = Trainer(current_model, criterion, None, args.device, args.results_dir,
                              "")  # No optimizer needed for eval
            print("Evaluating model on target test set...")
            test_loss, test_acc, true_labels, pred_labels, features = trainer.evaluate(
                target_test_loader, desc="Testing (Eval Pre-trained)", get_features_for_tsne=True
            )
            print(f"Eval Pre-trained - Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
            if len(true_labels) > 0:
                class_names = [f"Class {i}" for i in range(num_classes_for_target_model)]  # Use target classes
                plot_confusion_matrix(true_labels, pred_labels, class_names,
                                      os.path.join(args.visualizations_dir, f"cm_{args.save_name}_eval_pretrained.png"),
                                      title=f"Confusion Matrix (Eval Pre-trained - {args.model_type})")
                if features.shape[0] > 0:
                    plot_tsne(features, true_labels, class_names,
                              os.path.join(args.visualizations_dir, f"tsne_{args.save_name}_eval_pretrained.png"),
                              title=f"t-SNE (Eval Pre-trained - {args.model_type})")
    else:
        print(f"Unknown mode: {args.mode}")


if __name__ == '__main__':
    main()