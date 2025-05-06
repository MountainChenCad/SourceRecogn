import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

def load_h5_data(filepath, class_indices=None):
    """Loads Data_IQ and Label from HDF5 file, optionally filtering by class indices."""
    try:
        with h5py.File(filepath, 'r') as f:
            all_iq_data = f['Data_IQ'][:]
            all_labels = f['Label'][:]
            # Optional: Load class names if needed later
            # class_names = [c.decode('utf-8') for c in f['classes'][:]]

        if class_indices is not None:
            print(f"Filtering data for class indices: {class_indices}")
            # Find original indices matching the desired class labels
            mask = np.isin(all_labels, class_indices)
            iq_data = all_iq_data[mask]
            labels = all_labels[mask]

             # --- Re-map labels to be contiguous from 0 ---
            unique_labels = sorted(list(set(labels)))
            label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
            labels = np.array([label_map[l] for l in labels])
            print(f"Selected {len(labels)} samples. New label mapping: {label_map}")
            # --------------------------------------------

        else:
            iq_data = all_iq_data
            labels = all_labels

        print(f"Loaded data from {filepath}. IQ shape: {iq_data.shape}, Labels shape: {labels.shape}")
        return iq_data, labels #, class_names

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        raise
    except Exception as e:
        print(f"Error loading HDF5 file {filepath}: {e}")
        raise


def segment_data(iq_data, labels, segment_length, stride):
    """Segments long IQ signals into smaller chunks."""
    num_samples, num_channels, total_length = iq_data.shape
    segments = []
    segment_labels = []

    print(f"Segmenting data: {num_samples} samples, segment length {segment_length}, stride {stride}")
    for i in range(num_samples):
        label = labels[i]
        signal = iq_data[i] # Shape (2, total_length)
        start = 0
        while start + segment_length <= total_length:
            segment = signal[:, start : start + segment_length]
            segments.append(segment)
            segment_labels.append(label)
            start += stride

    segments = np.array(segments, dtype=np.float32) # Shape (total_segments, 2, segment_length)
    segment_labels = np.array(segment_labels, dtype=np.int64)

    print(f"Segmentation complete. Total segments: {len(segments)}")
    return segments, segment_labels


class SignalDataset(Dataset):
    """PyTorch Dataset for segmented signal data."""
    def __init__(self, segments, labels, model_type='resnet'):
        self.segments = segments
        self.labels = labels
        self.model_type = model_type
        assert len(self.segments) == len(self.labels), "Segments and labels must have the same length!"

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        segment = self.segments[idx] # Shape (2, segment_length)
        label = self.labels[idx]

        # Reshape for LSTM if needed: (seq_len, features)
        # LSTM expects input shape (batch_size, seq_len, input_size)
        # Our segment is (channels, seq_len) -> needs transpose -> (seq_len, channels)
        if self.model_type == 'lstm':
            segment = segment.transpose(1, 0) # Shape (segment_length, 2)

        return torch.from_numpy(segment), torch.tensor(label, dtype=torch.long)


def create_dataloaders(h5_path, segment_length, stride, batch_size,
                       model_type='resnet', is_target=False, split_ratio=None,
                       seed=42, class_indices=None):
    """Loads, segments, splits data, and creates DataLoaders."""

    iq_data, labels = load_h5_data(h5_path, class_indices)
    segments, segment_labels = segment_data(iq_data, labels, segment_length, stride)

    if not is_target:
        # Source domain: Use all data for training (no split needed for pre-training)
        print("Creating DataLoader for source domain (all data for training)")
        dataset = SignalDataset(segments, segment_labels, model_type)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        return loader, None, None # Return None for val/test loaders
    else:
        # Target domain: Split into train, validation, test
        if split_ratio is None:
            split_ratio = [0.7, 0.15, 0.15] # Default split
        assert abs(sum(split_ratio) - 1.0) < 1e-6, "Split ratios must sum to 1"
        print(f"Splitting target data: Train={split_ratio[0]*100}%, Val={split_ratio[1]*100}%, Test={split_ratio[2]*100}%")

        num_segments = len(segments)
        indices = np.arange(num_segments)

        # Split indices first
        train_indices, remaining_indices = train_test_split(
            indices, train_size=split_ratio[0], random_state=seed, stratify=segment_labels
        )
        # Adjust remaining split ratio for val/test from the remainder
        if split_ratio[1] + split_ratio[2] > 1e-6: # Avoid division by zero if no val/test split
            val_test_ratio = split_ratio[2] / (split_ratio[1] + split_ratio[2])
            val_indices, test_indices = train_test_split(
                remaining_indices,
                test_size=val_test_ratio,
                random_state=seed,
                stratify=segment_labels[remaining_indices] # Stratify on the remaining labels
            )
        else: # Handle case where only train split is needed (e.g., ratio [1.0, 0.0, 0.0])
            val_indices, test_indices = np.array([]), np.array([])


        print(f"Split sizes: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")

        train_dataset = SignalDataset(segments[train_indices], segment_labels[train_indices], model_type)
        val_dataset = SignalDataset(segments[val_indices], segment_labels[val_indices], model_type) if len(val_indices) > 0 else None
        test_dataset = SignalDataset(segments[test_indices], segment_labels[test_indices], model_type) if len(test_indices) > 0 else None

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True) if val_dataset else None
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True) if test_dataset else None

        return train_loader, val_loader, test_loader