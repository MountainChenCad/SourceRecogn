import torch
import torch.nn as nn
import torch.optim as optim
import os

from .utils import set_seed, get_args, load_checkpoint
from .data_loader import create_dataloaders
from .models import get_model
from .trainer import Trainer

def main():
    args = get_args()
    set_seed(args.seed)

    # --- Data Loading ---
    print("Loading data...")
    source_loader, _, _ = create_dataloaders(
        h5_path=args.source_data,
        segment_length=args.segment_length,
        stride=args.stride,
        batch_size=args.batch_size,
        model_type=args.model_type,
        is_target=False,
        class_indices=args.source_classes # Use subset if specified
    )
    target_train_loader, target_val_loader, target_test_loader = create_dataloaders(
        h5_path=args.target_data,
        segment_length=args.segment_length,
        stride=args.stride, # Use same stride for consistency, maybe smaller for target?
        batch_size=args.batch_size,
        model_type=args.model_type,
        is_target=True,
        split_ratio=args.target_split_ratio,
        seed=args.seed
    )
    print("Data loading complete.")

    # Determine number of classes dynamically if source_classes is used
    num_classes_source = args.num_classes_source
    if args.source_classes is not None:
        num_classes_source = len(args.source_classes)
        print(f"Adjusted source class count based on --source_classes: {num_classes_source}")


    # --- Model Definition ---
    print(f"Creating model: {args.model_type}")
    # Source model (used for pretraining)
    source_model = get_model(args.model_type, num_classes=num_classes_source)

    # Target model (can be same architecture but different final layer size if needed)
    # For this setup, assume target also has 8 classes. If different, adjust num_classes_target.
    target_model = get_model(args.model_type, num_classes=args.num_classes_target)
    print(f"Source Model: {num_classes_source} classes, Target Model: {args.num_classes_target} classes")

    criterion = nn.CrossEntropyLoss()

    # --- Mode Execution ---
    if args.mode == 'pretrain':
        print("\n--- Mode: Pre-training on Source Domain ---")
        optimizer = optim.Adam(source_model.parameters(), lr=args.lr)
        trainer = Trainer(source_model, criterion, optimizer, args.device, args.results_dir, f"{args.save_name}_source_pretrained")
        trainer.train(source_loader, None, args.epochs) # No validation during pretraining for simplicity
        print("Pre-training finished.")

    elif args.mode == 'target_only':
        print("\n--- Mode: Training on Target Domain Only (From Scratch) ---")
        optimizer = optim.Adam(target_model.parameters(), lr=args.lr)
        trainer = Trainer(target_model, criterion, optimizer, args.device, args.results_dir, f"{args.save_name}_target_only")
        trainer.train(target_train_loader, target_val_loader, args.epochs)
        print("Evaluating on target test set...")
        test_loss, test_acc = trainer.evaluate(target_test_loader, desc="Testing (Target Only)")
        print(f"Target Only - Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    elif args.mode == 'finetune' or args.mode == 'eval_pretrained':
        # Requires loading pretrained weights
        if not args.pretrained_path or not os.path.exists(args.pretrained_path):
            raise FileNotFoundError(f"Pretrained model path not found or not specified: {args.pretrained_path}")

        # Load pre-trained weights into the target model architecture
        print(f"Loading pre-trained weights from: {args.pretrained_path}")
        # Load the state dict from the source-trained model
        checkpoint = torch.load(args.pretrained_path, map_location=args.device)
        pretrained_dict = checkpoint['state_dict']

        # Adapt for potential differences in the final classification layer
        model_dict = target_model.state_dict()
        # 1. filter out unnecessary keys (like the final fc layer if class numbers differ)
        # NOTE: This assumes the final layer is named 'fc'. Adjust if your model uses a different name.
        pretrained_dict_filtered = {k: v for k, v in pretrained_dict.items() if k in model_dict and not k.startswith('fc.')}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict_filtered)
        # 3. load the new state dict
        target_model.load_state_dict(model_dict, strict=False) # strict=False allows loading partial weights
        print("Pre-trained backbone weights loaded.")
        if any(k.startswith('fc.') for k in pretrained_dict_filtered.keys()):
             print("Warning: FC layer weights might have been loaded unexpectedly if names matched.")

        target_model = target_model.to(args.device) # Ensure model is on correct device

        if args.mode == 'finetune':
            print("\n--- Mode: Fine-tuning on Target Domain ---")
            # Setup optimizer - potentially different LR for backbone and head
            if args.freeze_backbone:
                print("Freezing backbone layers.")
                for param in target_model.parameters():
                    param.requires_grad = False
                # Unfreeze the final classification layer (assuming it's named 'fc')
                if hasattr(target_model, 'fc'):
                    for param in target_model.fc.parameters():
                        param.requires_grad = True
                    optimizer = optim.Adam(filter(lambda p: p.requires_grad, target_model.parameters()), lr=args.lr)
                    print("Optimizer set for classifier head only.")
                else:
                     print("Warning: Could not find 'fc' layer to unfreeze. Fine-tuning might not work as expected.")
                     optimizer = optim.Adam(filter(lambda p: p.requires_grad, target_model.parameters()), lr=args.lr) # Finetune whatever is available
            else:
                print("Fine-tuning all layers.")
                optimizer = optim.Adam(target_model.parameters(), lr=args.lr) # Finetune all layers

            trainer = Trainer(target_model, criterion, optimizer, args.device, args.results_dir, f"{args.save_name}_target_finetuned")
            trainer.train(target_train_loader, target_val_loader, args.epochs)
            print("Evaluating fine-tuned model on target test set...")
            test_loss, test_acc = trainer.evaluate(target_test_loader, desc="Testing (Fine-tuned)")
            print(f"Fine-tuned - Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

        elif args.mode == 'eval_pretrained':
            print("\n--- Mode: Evaluating Pre-trained Model on Target Domain (No Fine-tuning) ---")
            # Note: This assumes the pre-trained model's output classes match the target classes.
            # If not, the accuracy calculation might be meaningless without adapting the head.
            # For simplicity here, we assume they match (both 8).
            if num_classes_source != args.num_classes_target:
                print(f"Warning: Pre-trained model has {num_classes_source} classes, target needs {args.num_classes_target}. Evaluation might be inaccurate without head adaptation.")
                # A proper evaluation would involve training a new head on the frozen features.
                # For this script, we just proceed with the loaded model.

            trainer = Trainer(target_model, criterion, None, args.device, args.results_dir, "") # No optimizer needed
            print("Evaluating pre-trained model on target test set...")
            test_loss, test_acc = trainer.evaluate(target_test_loader, desc="Testing (Pre-trained)")
            print(f"Pre-trained (No Fine-tune) - Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    else:
        print(f"Unknown mode: {args.mode}")

if __name__ == '__main__':
    main()