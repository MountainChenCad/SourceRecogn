import torch
import random
import numpy as np
import os
import argparse
import json

def set_seed(seed):
    """Sets random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """Saves model and training parameters."""
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Loads model and training parameters."""
    if os.path.exists(checkpoint_path):
        print(f"=> Loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer:
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except ValueError as e:
                 print(f"Warning: Could not load optimizer state: {e}. Starting fresh.")
        print(f"=> Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint.get('epoch', '?')})")
        return checkpoint.get('epoch', 0)
    else:
        print(f"=> No checkpoint found at '{checkpoint_path}'")
        return 0

def get_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description='Transfer Learning for Radiation Source Recognition')

    # Data args
    parser.add_argument('--source_data', type=str, required=True, help='Path to source HDF5 file')
    parser.add_argument('--target_data', type=str, required=True, help='Path to target HDF5 file')
    parser.add_argument('--segment_length', type=int, default=1024, help='Length of signal segments')
    parser.add_argument('--stride', type=int, default=None, help='Stride for segmentation (default: segment_length)')
    parser.add_argument('--target_split_ratio', type=float, nargs=3, default=[0.7, 0.15, 0.15], help='Target train/val/test split ratio')
    parser.add_argument('--source_classes', type=int, nargs='+', default=None, help='Subset of source classes to use (indices, e.g., 0 2 4)')
    parser.add_argument('--num_classes_source', type=int, default=8, help='Number of classes in the source domain')
    parser.add_argument('--num_classes_target', type=int, default=8, help='Number of classes in the target domain')

    # Model args
    parser.add_argument('--model_type', type=str, default='resnet', choices=['resnet', 'lstm'], help='Backbone model type')
    # Add model specific args if needed (e.g., lstm_hidden_size, resnet_layers)

    # Training args
    parser.add_argument('--mode', type=str, required=True, choices=['pretrain', 'finetune', 'target_only', 'eval_pretrained'], help='Execution mode')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--freeze_backbone', action='store_true', help='Freeze backbone during fine-tuning')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Paths and Saving
    parser.add_argument('--results_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path to load pre-trained model weights')
    parser.add_argument('--save_name', type=str, default='model', help='Base name for saving models')

    args = parser.parse_args()
    args.device = torch.device("cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu")
    args.stride = args.stride if args.stride is not None else args.segment_length # Default stride

    # Create results directories if they don't exist
    os.makedirs(os.path.join(args.results_dir, 'models'), exist_ok=True)

    print("----- Configuration -----")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("-------------------------")

    return args
