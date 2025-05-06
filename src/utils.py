# src/utils.py
import torch
import random
import numpy as np
import os
import argparse
# import json # Not used, can be removed
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.manifold import TSNE
import logging  # Import logging

# Configure basic logging (will be reconfigured in main.py for file output)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def set_seed(seed):
    """Sets random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """Saves model and training parameters."""
    logging.info("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Loads model and training parameters."""
    if os.path.exists(checkpoint_path):
        logging.info(f"=> Loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)  # Ensure loading to CPU if model was on GPU
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer:
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except ValueError as e:
                logging.warning(f"Could not load optimizer state: {e}. Starting fresh.")
        logging.info(f"=> Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint.get('epoch', '?')})")
        return checkpoint.get('epoch', 0)
    else:
        logging.error(f"=> No checkpoint found at '{checkpoint_path}'")
        return 0


def plot_confusion_matrix(y_true, y_pred, class_names, save_path, title='Confusion Matrix'):
    if len(y_true) == 0 or len(y_pred) == 0:
        logging.warning("Empty true or predicted labels for confusion matrix. Skipping plot.")
        return
    cm = sk_confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    plt.figure(figsize=(max(8, len(class_names)), max(6, int(len(class_names) * 0.8))))  # Dynamic figure size
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    try:
        plt.savefig(save_path, dpi=300)
        logging.info(f"Confusion matrix saved to {save_path}")
    except Exception as e:
        logging.error(f"Error saving confusion matrix: {e}")
    plt.close()


def plot_tsne(features, labels, class_names, save_path, title='t-SNE Visualization of Features',
              perplexity=30, n_iter=1000, random_state=42):
    if features.shape[0] == 0:
        logging.warning("No features to plot for t-SNE.")
        return

    num_samples = features.shape[0]
    if num_samples <= 1:  # TSNE needs more than 1 sample
        logging.warning(f"t-SNE requires more than 1 sample, got {num_samples}. Skipping plot.")
        return

    # Adjust perplexity: must be less than n_samples
    effective_perplexity = min(perplexity, num_samples - 1)
    if effective_perplexity <= 0:  # if num_samples was 1, perplexity becomes 0
        effective_perplexity = 1  # or some small default like 5 if num_samples > 5
        if num_samples < 5: effective_perplexity = max(1, num_samples - 1)

    if effective_perplexity != perplexity:
        logging.warning(
            f"Adjusted t-SNE perplexity from {perplexity} to {effective_perplexity} due to sample size {num_samples}.")

    logging.info(f"Running t-SNE (perplexity={effective_perplexity}, n_iter={n_iter})... this may take a while.")
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=effective_perplexity,
                n_iter=n_iter, init='pca', learning_rate='auto', n_jobs=-1)  # Use n_jobs for speed

    try:
        tsne_results = tsne.fit_transform(features)
    except Exception as e:
        logging.error(f"Error during t-SNE fitting: {e}")
        return

    plt.figure(figsize=(12, 10))
    unique_labels_in_data = np.unique(labels)

    # Ensure class_names array is long enough, pad if necessary
    # This handles cases where labels might be like [0, 2] but class_names only has 2 entries.
    max_label_val = np.max(unique_labels_in_data) if len(unique_labels_in_data) > 0 else -1

    if max_label_val >= len(class_names):
        logging.warning(
            f"Max label value ({max_label_val}) exceeds length of provided class_names ({len(class_names)}). Using generic names for missing labels.")
        # Extend class_names with generic names
        extended_class_names = list(class_names) + [f"Class {i}" for i in range(len(class_names), max_label_val + 1)]
    else:
        extended_class_names = class_names

    colors = plt.cm.get_cmap('viridis', len(unique_labels_in_data))

    for i, label_val in enumerate(unique_labels_in_data):
        idx = labels == label_val
        class_name_for_legend = extended_class_names[label_val] if label_val < len(
            extended_class_names) else f"Unknown Class {label_val}"
        plt.scatter(tsne_results[idx, 0], tsne_results[idx, 1],
                    color=colors(i), label=class_name_for_legend, alpha=0.7)

    plt.title(title)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    if len(unique_labels_in_data) > 0: plt.legend(loc='best')  # Only show legend if there's data
    plt.tight_layout()
    try:
        plt.savefig(save_path, dpi=300)
        logging.info(f"t-SNE plot saved to {save_path}")
    except Exception as e:
        logging.error(f"Error saving t-SNE plot: {e}")
    plt.close()


def plot_training_curves(history, save_path_prefix, title_prefix=''):
    """
    Plots and saves training and validation loss and accuracy curves.
    Args:
        history (dict): Dictionary containing lists: 'train_loss', 'train_acc', 'val_loss', 'val_acc'.
        save_path_prefix (str): Prefix for saving plot files (e.g., path/to/visualizations/exp_stage).
        title_prefix (str): Prefix for plot titles.
    """
    epochs_range = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Training Loss')
    if 'val_loss' in history and history['val_loss']:
        plt.plot(epochs_range, history['val_loss'], label='Validation Loss')
    plt.title(f'{title_prefix} Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_acc'], label='Training Accuracy')
    if 'val_acc' in history and history['val_acc']:
        plt.plot(epochs_range, history['val_acc'], label='Validation Accuracy')
    plt.title(f'{title_prefix} Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    save_path = f"{save_path_prefix}_train_val_curves.png"
    try:
        plt.savefig(save_path, dpi=300)
        logging.info(f"Training curves saved to {save_path}")
    except Exception as e:
        logging.error(f"Error saving training curves: {e}")
    plt.close()

def get_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description='Transfer Learning for Radiation Source Recognition')

    parser.add_argument('--source_data', type=str, required=True, help='Path to source HDF5 file')
    parser.add_argument('--target_data', type=str, required=True, help='Path to target HDF5 file')
    parser.add_argument('--segment_length', type=int, default=1024, help='Length of signal segments')
    parser.add_argument('--stride', type=int, default=None, help='Stride for segmentation (default: segment_length)')
    parser.add_argument('--target_split_ratio', type=float, nargs=3, default=[0.7, 0.15, 0.15],
                        help='Target train/val/test split ratio')
    parser.add_argument('--source_classes', type=int, nargs='+', default=None,
                        help='Subset of source classes to use (original values from HDF5)')
    parser.add_argument('--num_classes_source', type=int, default=8,
                        help='Number of classes in the source domain (after potential subset selection and remapping)')
    parser.add_argument('--num_classes_target', type=int, default=8,
                        help='Number of classes in the target domain (after remapping)')
    parser.add_argument('--model_type', type=str, default='resnet', choices=['resnet', 'lstm'],
                        help='Backbone model type')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['pretrain', 'finetune', 'target_only', 'eval_pretrained'], help='Execution mode')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--freeze_backbone_finetune', action='store_true',
                        help='Force freeze backbone during fine-tuning (classifier only). Default is to fine-tune classifier only for "finetune" mode.')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--results_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path to load pre-trained model weights')
    parser.add_argument('--save_name', type=str, default='model',
                        help='Base name for saving models and logs, set by shell script for each stage')

    args = parser.parse_args()
    args.device = torch.device("cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu")
    args.stride = args.stride if args.stride is not None else args.segment_length

    # Create results, models, visualizations, and logs directories
    # args.save_name will be specific to the experiment stage, e.g., "comp_resnet_seg1024_source_pretrained"
    # results_dir will be like "results/comparison_resnet_seg1024"

    # The main results directory for this specific run (e.g. results/comparison_resnet_seg1024/comp_resnet_seg1024_source_pretrained)
    # No, args.results_dir is the top-level for the experiment (e.g., results/comparison_resnet_seg1024)
    # args.save_name is more like a file prefix.

    os.makedirs(args.results_dir, exist_ok=True)  # Top level e.g. results/comparison_resnet_seg1024
    args.models_dir = os.path.join(args.results_dir, 'models')  # e.g. results/comparison_resnet_seg1024/models
    args.visualizations_dir = os.path.join(args.results_dir,
                                           'visualizations')  # e.g. results/comparison_resnet_seg1024/visualizations
    args.logs_dir = os.path.join(args.results_dir, 'logs')  # e.g. results/comparison_resnet_seg1024/logs

    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.visualizations_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)

    # Note: Logging to file will be configured in main.py using args.logs_dir and args.save_name
    return args