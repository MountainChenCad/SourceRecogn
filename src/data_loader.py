# src/data_loader.py
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import logging


def load_h5_data(filepath, class_indices=None, num_expected_classes=None):
    try:
        with h5py.File(filepath, 'r') as f:
            all_iq_data = f['Data_IQ'][:]
            original_labels_from_file = f['Label'][:]  # Labels as they are in the HDF5 file

            loaded_h5_class_names = []
            if 'classes' in f:
                try:
                    loaded_h5_class_names = [c.decode('utf-8') if isinstance(c, bytes) else str(c) for c in
                                             f['classes'][:]]
                    logging.info(
                        f"Successfully loaded {len(loaded_h5_class_names)} class names from 'classes' dataset in {filepath}.")
                except Exception as e:
                    logging.warning(
                        f"Could not decode class names from {filepath}: {e}. Will use numeric labels if mapping fails.")
            else:
                logging.info(f"No 'classes' dataset found in {filepath}. Will use numeric labels.")

        iq_data = all_iq_data

        # Determine the set of original HDF5 label values that will be used
        if class_indices is not None and len(class_indices) > 0:
            logging.info(f"Filtering data for original HDF5 class values: {class_indices}")
            mask = np.isin(original_labels_from_file, class_indices)
            iq_data = all_iq_data[mask]
            active_original_labels = original_labels_from_file[mask]  # Labels from HDF5 that are selected
            if len(active_original_labels) == 0:
                logging.warning(f"No samples found for class_indices {class_indices} in {filepath}.")
                return np.array([]), np.array([], dtype=np.int64), []
            unique_active_original_labels = sorted(list(set(active_original_labels)))
        else:
            logging.info(f"Using all labels from HDF5 file: {filepath}")
            active_original_labels = original_labels_from_file
            if len(active_original_labels) == 0:
                logging.warning(f"No labels found in {filepath}.")
                return np.array([]), np.array([], dtype=np.int64), []
            unique_active_original_labels = sorted(list(set(active_original_labels)))

        # --- Remap active original labels to 0 to K-1 for the model ---
        # K is num_expected_classes or len(unique_active_original_labels)

        num_model_classes = num_expected_classes
        if num_model_classes is None:
            num_model_classes = len(unique_active_original_labels)
            logging.info(f"num_expected_classes not provided, set to found unique labels: {num_model_classes}")

        if len(unique_active_original_labels) != num_model_classes:
            logging.warning(f"Number of unique active original labels ({len(unique_active_original_labels)}) "
                            f"does not match num_model_classes ({num_model_classes}). This might lead to issues if not intended.")
            # If using a subset via class_indices, num_model_classes should be len(unique_active_original_labels)
            # If using all, num_model_classes should be len(unique_active_original_labels) (if num_expected_classes was None)
            # or num_expected_classes (if provided, and they should match)

        # label_map: maps from an original HDF5 label value (from unique_active_original_labels)
        # to a new model label (0 to num_model_classes-1)
        label_map = {orig_label: new_label for new_label, orig_label in enumerate(unique_active_original_labels)}

        # Ensure label_map only contains keys that are in unique_active_original_labels
        # and values are within 0 to num_model_classes-1
        # This should be correct if unique_active_original_labels is what's used to define the mapping.
        # If len(unique_active_original_labels) > num_model_classes, some original labels won't be used.
        # If len(unique_active_original_labels) < num_model_classes, model expects more classes than available.

        processed_labels = np.array([label_map[l] for l in active_original_labels if l in label_map], dtype=np.int64)

        # --- Construct final_class_names_for_model (length num_model_classes) ---
        final_class_names_for_model = [f"Label {i}" for i in range(num_model_classes)]  # Default to numeric names

        if loaded_h5_class_names:
            # Try to map names. This is the tricky part.
            # Assumption 1: If create_target_h5.py was used, HDF5 'classes' are for HDF5 labels 0, 1, 2...
            # Assumption 2: For source, HDF5 'classes' might be for its own internal 0..7, while 'Label' is 0,2,4..

            # Let's try a direct mapping if original HDF5 labels were simple 0-indexed
            # This is more likely for target_domain.h5
            all_original_h5_labels_sorted = sorted(list(set(original_labels_from_file)))

            # Heuristic: if HDF5 labels are 0,1,2... and match length of loaded_h5_class_names
            is_h5_simple_0_indexed = (
                        all_original_h5_labels_sorted == list(range(len(all_original_h5_labels_sorted))) and
                        len(all_original_h5_labels_sorted) == len(loaded_h5_class_names))

            for orig_hdf5_label_val, model_label_idx in label_map.items():
                # orig_hdf5_label_val is one of the unique_active_original_labels
                # model_label_idx is its remapped value (0 to K-1)
                if model_label_idx < num_model_classes:  # Ensure we don't write out of bounds
                    found_name = None
                    if is_h5_simple_0_indexed:
                        # If HDF5 labels (all of them, not just active) were 0,1,2...N-1
                        # and loaded_h5_class_names has N names, then names[orig_hdf5_label_val] is the name.
                        if orig_hdf5_label_val < len(loaded_h5_class_names):
                            found_name = loaded_h5_class_names[orig_hdf5_label_val]
                    else:
                        # More complex case (e.g., source BPSK file where labels are 0,2,4...)
                        # Try to find orig_hdf5_label_val in the sorted unique labels from the HDF5 file
                        # and use its index to get the name from loaded_h5_class_names,
                        # IF loaded_h5_class_names corresponds to this sorted order.
                        try:
                            idx_in_h5_sorted_labels = all_original_h5_labels_sorted.index(orig_hdf5_label_val)
                            if idx_in_h5_sorted_labels < len(loaded_h5_class_names):
                                found_name = loaded_h5_class_names[idx_in_h5_sorted_labels]
                        except ValueError:
                            pass  # orig_hdf5_label_val not in all_original_h5_labels_sorted (should not happen if logic is right)

                    if found_name:
                        final_class_names_for_model[model_label_idx] = found_name
                    # If found_name is still None, it will keep the default "Label X"

        logging.info(f"File: {filepath} - Remapped active HDF5 labels {unique_active_original_labels} to model labels "
                     f"using map: {label_map}. Resulting model class names (len {len(final_class_names_for_model)}): {final_class_names_for_model}")

        # Sanity checks (convert prints to logging)
        logging.info(
            f"Loaded data from {filepath}. IQ shape: {iq_data.shape}, Processed Labels shape: {processed_labels.shape}")
        if len(processed_labels) > 0:
            min_proc_label, max_proc_label = np.min(processed_labels), np.max(processed_labels)
            logging.info(f"Min processed label: {min_proc_label}, Max processed label: {max_proc_label}")
            num_unique_final_labels = len(np.unique(processed_labels))
            logging.info(f"Number of unique processed labels: {num_unique_final_labels}")

            if max_proc_label >= num_model_classes:
                logging.critical(
                    f"Max processed label ({max_proc_label}) is out of bounds for model (0 to {num_model_classes - 1}). This indicates a bug in label remapping or class counting.")
        elif len(active_original_labels) > 0:  # Had original labels but processed_labels is empty
            logging.warning(
                "Processed labels array is empty despite having active original labels. Check label_map logic.")
        else:  # No active labels to begin with
            logging.info("No active labels to process from this file with current settings.")

        return iq_data, processed_labels, final_class_names_for_model

    except FileNotFoundError:
        logging.error(f"File not found at {filepath}")
        raise
    except Exception as e:
        logging.error(f"Error loading HDF5 file {filepath}: {e}")
        raise

def segment_data(iq_data, labels, segment_length, stride):
    if iq_data.ndim == 0 or iq_data.size == 0 or labels.ndim == 0 or labels.size == 0:
        logging.warning("Empty iq_data or labels passed to segment_data. Returning empty arrays.")
        return np.array([]), np.array([], dtype=np.int64)

    num_samples, num_channels, total_length = iq_data.shape
    segments = []
    segment_labels = []

    logging.info(f"Segmenting data: {num_samples} samples, segment length {segment_length}, stride {stride}")
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
        logging.warning("No segments created from data. Check segment_length and total_length.")
        return np.array([]), np.array([], dtype=np.int64)

    segments = np.array(segments, dtype=np.float32)
    segment_labels = np.array(segment_labels, dtype=np.int64)

    logging.info(f"Segmentation complete. Total segments: {len(segments)}")
    return segments, segment_labels


class SignalDataset(Dataset):  # (No changes needed here beyond what was already there)
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
        if self.model_type == 'lstm': segment = segment.transpose(1, 0)
        return torch.from_numpy(segment), torch.tensor(label, dtype=torch.long)


def create_dataloaders(h5_path, segment_length, stride, batch_size,
                       model_type='resnet', is_target=False, split_ratio=None,
                       seed=42, class_indices=None, num_expected_classes_for_loader=None):
    iq_data, labels, class_names = load_h5_data(  # Get class_names
        h5_path,
        class_indices=class_indices,
        num_expected_classes=num_expected_classes_for_loader
    )

    if iq_data.size == 0:
        logging.warning(f"No data loaded from {h5_path} with current settings. Returning None for DataLoaders.")
        return None, None, None, []  # Return empty list for class_names

    segments, segment_labels = segment_data(iq_data, labels, segment_length, stride)

    if segments.size == 0:
        logging.warning(f"No segments created for {h5_path}. Returning None for DataLoaders.")
        return None, None, None, []  # Return empty list for class_names

    # ... (rest of create_dataloaders logic as in your provided file, but return class_names at the end)
    if not is_target:
        logging.info(f"Creating DataLoader for source domain ({h5_path}, all data for training)")
        dataset = SignalDataset(segments, segment_labels, model_type)
        if len(dataset) == 0:
            logging.warning(f"Source dataset for {h5_path} is empty after processing.")
            return (DataLoader(dataset, batch_size=batch_size) if len(dataset) > 0 else None), None, None, class_names
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        return loader, None, None, class_names  # Pass class_names
    else:
        # ... (target splitting logic from your file) ...
        if split_ratio is None: split_ratio = [0.7, 0.15, 0.15]
        assert abs(sum(split_ratio) - 1.0) < 1e-6, "Split ratios must sum to 1"
        num_total_segments = len(segments)
        indices = np.arange(num_total_segments)
        stratify_array = segment_labels if len(np.unique(segment_labels)) > 1 and min(
            np.unique(segment_labels, return_counts=True)[1]) >= (
                                               1 if split_ratio[1] + split_ratio[2] == 0 else 2) else None
        train_indices, remaining_indices = train_test_split(indices, train_size=split_ratio[0], random_state=seed,
                                                            stratify=stratify_array)
        val_indices, test_indices = np.array([]), np.array([])
        if len(remaining_indices) > 0 and (split_ratio[1] + split_ratio[2] > 1e-6):
            val_test_ratio = split_ratio[2] / (split_ratio[1] + split_ratio[2]) if (split_ratio[1] + split_ratio[
                2]) > 0 else 0
            stratify_remaining = segment_labels[remaining_indices] if stratify_array is not None and len(
                np.unique(segment_labels[remaining_indices])) > 1 and min(
                np.unique(segment_labels[remaining_indices], return_counts=True)[1]) >= (
                                                                          1 if val_test_ratio == 0 or val_test_ratio == 1 else 2) else None
            val_indices, test_indices = train_test_split(remaining_indices, test_size=val_test_ratio, random_state=seed,
                                                         stratify=stratify_remaining)
        elif len(remaining_indices) > 0 and split_ratio[0] == 1.0:
            train_indices = indices

        logging.info(
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
        return train_loader, val_loader, test_loader, class_names  # Pass class_names