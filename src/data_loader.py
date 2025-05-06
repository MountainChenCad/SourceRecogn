# src/data_loader.py

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


def load_h5_data(filepath, class_indices=None, num_expected_classes=None):
    """
    Loads Data_IQ and Label from HDF5 file.
    - If class_indices is provided, filters for those original labels and remaps them to 0 to K-1.
    - If class_indices is None, uses all labels and remaps them to 0 to K-1.
    - num_expected_classes is the number of classes the model (and thus the remapped labels) should have.
    """
    try:
        with h5py.File(filepath, 'r') as f:
            all_iq_data = f['Data_IQ'][:]
            original_labels_from_file = f['Label'][:]  # Labels as they are in the HDF5 file

        iq_data = all_iq_data

        if class_indices is not None and len(class_indices) > 0:
            # User wants a subset of classes, specified by their original values in HDF5
            print(f"Filtering data for original class values: {class_indices} from HDF5 file: {filepath}")

            mask = np.isin(original_labels_from_file, class_indices)
            iq_data = all_iq_data[mask]
            labels_to_remap = original_labels_from_file[mask]

            if len(labels_to_remap) == 0:
                print(f"Warning: No samples found for class_indices {class_indices} in {filepath}.")
                return np.array([]), np.array([], dtype=np.int64)

            unique_selected_original_labels = sorted(list(set(labels_to_remap)))

            # The number of classes for the model will be the count of these unique selected labels
            if num_expected_classes is not None and len(unique_selected_original_labels) != num_expected_classes:
                print(
                    f"CRITICAL WARNING: Number of unique selected original labels ({len(unique_selected_original_labels)}) "
                    f"does not match num_expected_classes for the model ({num_expected_classes}). This will cause errors.")

            label_map = {old_label: new_label for new_label, old_label in enumerate(unique_selected_original_labels)}
            processed_labels = np.array([label_map[l] for l in labels_to_remap], dtype=np.int64)
            print(
                f"Selected {len(processed_labels)} samples. Original labels subset: {unique_selected_original_labels}. New remapped labels (0 to K-1) mapping: {label_map}")

        else:
            # No class_indices specified, use all labels from the file.
            # These labels still need to be remapped to be 0 to K-1 for the model.
            print(f"Using all labels from HDF5 file: {filepath}")
            labels_to_remap = original_labels_from_file

            if len(labels_to_remap) == 0:
                print(f"Warning: No labels found in {filepath}.")
                return np.array([]), np.array([], dtype=np.int64)

            unique_original_labels_in_file = sorted(list(set(labels_to_remap)))

            if num_expected_classes is not None and len(unique_original_labels_in_file) != num_expected_classes:
                print(
                    f"CRITICAL WARNING: Number of unique original labels in file ({len(unique_original_labels_in_file)}) "
                    f"does not match num_expected_classes for the model ({num_expected_classes}). This will cause errors.")

            label_map = {old_label: new_label for new_label, old_label in enumerate(unique_original_labels_in_file)}
            processed_labels = np.array([label_map[l] for l in labels_to_remap], dtype=np.int64)

            is_already_contiguous = (
                        list(unique_original_labels_in_file) == list(range(len(unique_original_labels_in_file))))
            if not is_already_contiguous:
                print(
                    f"Remapped all original labels from file. Original unique labels: {unique_original_labels_in_file}. New remapped labels (0 to K-1) mapping: {label_map}")
            else:
                print(
                    f"Original labels from file {unique_original_labels_in_file} are already effectively 0-indexed and contiguous.")

        print(
            f"Loaded data from {filepath}. IQ shape: {iq_data.shape}, Processed Labels shape: {processed_labels.shape}")
        if len(processed_labels) > 0:
            print(f"Min processed label: {np.min(processed_labels)}, Max processed label: {np.max(processed_labels)}")
            num_unique_final_labels = len(np.unique(processed_labels))
            print(f"Number of unique processed labels: {num_unique_final_labels}")

            if num_expected_classes is not None:
                if num_unique_final_labels > num_expected_classes:  # More unique labels than model classes
                    print(
                        f"CRITICAL WARNING: Found {num_unique_final_labels} unique processed labels, but model expects {num_expected_classes}. Labels will exceed model output range.")
                if np.max(processed_labels) >= num_expected_classes:
                    print(
                        f"CRITICAL WARNING: Max processed label ({np.max(processed_labels)}) is out of bounds for a model expecting "
                        f"{num_expected_classes} classes (i.e., labels 0 to {num_expected_classes - 1}).")
        else:
            print("No labels processed (empty dataset).")

        return iq_data, processed_labels

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        raise
    except Exception as e:
        print(f"Error loading HDF5 file {filepath}: {e}")
        raise


def segment_data(iq_data, labels, segment_length, stride):
    if iq_data.ndim == 0 or iq_data.size == 0 or labels.ndim == 0 or labels.size == 0:
        print("Warning: Empty iq_data or labels passed to segment_data. Returning empty arrays.")
        return np.array([]), np.array([], dtype=np.int64)

    num_samples, num_channels, total_length = iq_data.shape
    segments = []
    segment_labels = []

    print(f"Segmenting data: {num_samples} samples, segment length {segment_length}, stride {stride}")
    for i in range(num_samples):
        label = labels[i]
        signal = iq_data[i]
        start = 0
        while start + segment_length <= total_length:
            segment = signal[:, start: start + segment_length]
            segments.append(segment)
            segment_labels.append(label)
            start += stride

    if not segments:
        print("Warning: No segments created from data. Check segment_length and total_length.")
        return np.array([]), np.array([], dtype=np.int64)

    segments = np.array(segments, dtype=np.float32)
    segment_labels = np.array(segment_labels, dtype=np.int64)

    print(f"Segmentation complete. Total segments: {len(segments)}")
    return segments, segment_labels


class SignalDataset(Dataset):
    def __init__(self, segments, labels, model_type='resnet'):
        self.segments = segments
        self.labels = labels
        self.model_type = model_type
        if len(self.segments) != len(self.labels):
            raise ValueError(
                f"Segments and labels must have the same length! Got {len(self.segments)} and {len(self.labels)}")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        segment = self.segments[idx]
        label = self.labels[idx]
        if self.model_type == 'lstm':
            segment = segment.transpose(1, 0)
        return torch.from_numpy(segment), torch.tensor(label, dtype=torch.long)


def create_dataloaders(h5_path, segment_length, stride, batch_size,
                       model_type='resnet', is_target=False, split_ratio=None,
                       seed=42, class_indices=None, num_expected_classes_for_loader=None):
    iq_data, labels = load_h5_data(
        h5_path,
        class_indices=class_indices,
        num_expected_classes=num_expected_classes_for_loader
    )

    if iq_data.size == 0:  # Check if iq_data is empty (implies labels might also be)
        print(f"Warning: No data loaded from {h5_path} with current settings. Returning None for DataLoaders.")
        return None, None, None  # Adjusted to return three Nones for consistency

    segments, segment_labels = segment_data(iq_data, labels, segment_length, stride)

    if segments.size == 0:
        print(f"Warning: No segments created for {h5_path}. Returning None for DataLoaders.")
        return None, None, None  # Adjusted for consistency

    if not is_target:
        print(f"Creating DataLoader for source domain ({h5_path}, all data for training)")
        dataset = SignalDataset(segments, segment_labels, model_type)
        if len(dataset) == 0:
            print(f"Warning: Source dataset for {h5_path} is empty after processing.")
            # Return an empty DataLoader or handle as error
            return DataLoader(dataset, batch_size=batch_size) if len(dataset) > 0 else None, None, None
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        return loader, None, None
    else:
        # Target domain splitting logic (simplified for brevity, ensure your original logic for stratification is sound)
        if split_ratio is None: split_ratio = [0.7, 0.15, 0.15]
        assert abs(sum(split_ratio) - 1.0) < 1e-6, "Split ratios must sum to 1"

        num_total_segments = len(segments)
        indices = np.arange(num_total_segments)

        # Simplified split - in practice, ensure stratification works with small classes
        stratify_array = segment_labels if len(np.unique(segment_labels)) > 1 and min(
            np.unique(segment_labels, return_counts=True)[1]) >= (
                                               1 if split_ratio[1] + split_ratio[2] == 0 else 2) else None

        train_indices, remaining_indices = train_test_split(
            indices, train_size=split_ratio[0], random_state=seed, stratify=stratify_array
        )

        val_indices, test_indices = np.array([]), np.array([])
        if len(remaining_indices) > 0 and (split_ratio[1] + split_ratio[2] > 1e-6):
            val_test_ratio = split_ratio[2] / (split_ratio[1] + split_ratio[2]) if (split_ratio[1] + split_ratio[
                2]) > 0 else 0
            stratify_remaining = segment_labels[remaining_indices] if stratify_array is not None and len(
                np.unique(segment_labels[remaining_indices])) > 1 and min(
                np.unique(segment_labels[remaining_indices], return_counts=True)[1]) >= (
                                                                          1 if val_test_ratio == 0 or val_test_ratio == 1 else 2) else None
            val_indices, test_indices = train_test_split(
                remaining_indices, test_size=val_test_ratio, random_state=seed, stratify=stratify_remaining
            )
        elif len(remaining_indices) > 0:  # if no val/test split, remaining goes to train or is unused
            if split_ratio[0] == 1.0:  # All data for training
                train_indices = indices
            # else: # Some data might be unused if splits don't sum to 1 and are not handled
            #     print("Warning: Some data might be unused due to split configuration.")

        print(
            f"Target split sizes ({h5_path}): Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")

        train_dataset = SignalDataset(segments[train_indices], segment_labels[train_indices], model_type) if len(
            train_indices) > 0 else None
        val_dataset = SignalDataset(segments[val_indices], segment_labels[val_indices], model_type) if len(
            val_indices) > 0 else None
        test_dataset = SignalDataset(segments[test_indices], segment_labels[test_indices], model_type) if len(
            test_indices) > 0 else None

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2,
                                  pin_memory=True) if train_dataset else None
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2,
                                pin_memory=True) if val_dataset else None
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2,
                                 pin_memory=True) if test_dataset else None

        return train_loader, val_loader, test_loader