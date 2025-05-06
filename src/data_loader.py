# src/data_loader.py
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import logging  # Import logging


def load_h5_data(filepath, class_indices=None, num_expected_classes=None):
    try:
        with h5py.File(filepath, 'r') as f:
            all_iq_data = f['Data_IQ'][:]
            original_labels_from_file = f['Label'][:]

            # Attempt to load class names
            loaded_class_names = []
            if 'classes' in f:
                try:
                    loaded_class_names = [c.decode('utf-8') if isinstance(c, bytes) else str(c) for c in
                                          f['classes'][:]]
                except Exception as e:
                    logging.warning(f"Could not decode class names from {filepath}: {e}. Using generic names.")

        # ... (existing label processing and remapping logic from your file) ...
        iq_data = all_iq_data
        final_class_names_for_model = []  # Names corresponding to the remapped 0 to K-1 labels

        if class_indices is not None and len(class_indices) > 0:
            logging.info(f"Filtering data for original class values: {class_indices} from HDF5 file: {filepath}")
            mask = np.isin(original_labels_from_file, class_indices)
            iq_data = all_iq_data[mask]
            labels_to_remap = original_labels_from_file[mask]

            if len(labels_to_remap) == 0:
                logging.warning(f"No samples found for class_indices {class_indices} in {filepath}.")
                return np.array([]), np.array([], dtype=np.int64), []

            unique_selected_original_labels = sorted(list(set(labels_to_remap)))
            if num_expected_classes is not None and len(unique_selected_original_labels) != num_expected_classes:
                logging.critical(
                    f"Num unique selected original labels ({len(unique_selected_original_labels)}) != num_expected_classes ({num_expected_classes}).")

            label_map = {old_label: new_label for new_label, old_label in enumerate(unique_selected_original_labels)}
            processed_labels = np.array([label_map[l] for l in labels_to_remap], dtype=np.int64)

            # Create final_class_names_for_model based on the remapped labels
            # It should have length `num_expected_classes` or `len(unique_selected_original_labels)`
            # And its indices should correspond to the new remapped labels (0, 1, 2...)
            # The name at final_class_names_for_model[new_label] should be the name of original_label

            final_class_names_for_model = [""] * len(unique_selected_original_labels)
            if loaded_class_names:
                original_labels_in_h5_file_sorted = sorted(list(set(original_labels_from_file)))
                # Map from original HDF5 label value to its HDF5 name string
                original_h5_label_to_name_map = {orig_lbl_val: loaded_class_names[orig_idx]
                                                 for orig_idx, orig_lbl_val in
                                                 enumerate(original_labels_in_h5_file_sorted)
                                                 if orig_idx < len(loaded_class_names)}

                for original_label_val, new_label_idx in label_map.items():
                    if original_label_val in original_h5_label_to_name_map:
                        final_class_names_for_model[new_label_idx] = original_h5_label_to_name_map[original_label_val]
                    else:  # Fallback if name not found for some reason
                        final_class_names_for_model[new_label_idx] = f"OrigClass {original_label_val}"
            else:  # No class names in HDF5
                for original_label_val, new_label_idx in label_map.items():
                    final_class_names_for_model[new_label_idx] = f"Class {new_label_idx} (Orig {original_label_val})"

            logging.info(
                f"Selected {len(processed_labels)} samples. Remapped labels mapping: {label_map}. Final model class names: {final_class_names_for_model}")

        else:  # Use all labels
            logging.info(f"Using all labels from HDF5 file: {filepath}")
            labels_to_remap = original_labels_from_file
            if len(labels_to_remap) == 0:
                logging.warning(f"No labels found in {filepath}.")
                return np.array([]), np.array([], dtype=np.int64), []

            unique_original_labels_in_file = sorted(list(set(labels_to_remap)))
            if num_expected_classes is not None and len(unique_original_labels_in_file) != num_expected_classes:
                logging.critical(
                    f"Num unique original labels in file ({len(unique_original_labels_in_file)}) != num_expected_classes ({num_expected_classes}).")

            label_map = {old_label: new_label for new_label, old_label in enumerate(unique_original_labels_in_file)}
            processed_labels = np.array([label_map[l] for l in labels_to_remap], dtype=np.int64)

            # Create final_class_names_for_model for all labels
            final_class_names_for_model = [""] * len(unique_original_labels_in_file)
            if loaded_class_names:
                # Assuming loaded_class_names corresponds to the sorted unique original labels in the HDF5 file
                # This might be fragile if 'classes' dataset doesn't align perfectly with 'Label' values
                # A safer way would be if 'classes' dataset also had an associated mapping or if its order was guaranteed
                for original_label_val, new_label_idx in label_map.items():
                    # Find the original index of original_label_val in the sorted unique labels of the HDF5 to get its name
                    try:
                        original_h5_idx = list(original_labels_from_file).index(
                            original_label_val)  # This is not robust, need a better map
                        # A better approach: if 'classes' in HDF5 corresponds to labels 0..N-1 from HDF5's perspective
                        # And those HDF5 labels are [l0, l1, l2...], then loaded_class_names[i] is name for li
                        # We need to map our `original_label_val` (which is one of l0,l1,l2) to its name.
                        # If `loaded_class_names` was created by `create_target_h5.py`, it means `loaded_class_names[i]` is the name for label `i` from that file.
                        # So, `original_label_val` (e.g. 0, 2, 4...) should directly index `loaded_class_names` if they were 0-indexed in the H5.
                        # Given `create_target_h5.py` assigns labels 0,1,2... and class names in that order.
                        if original_label_val < len(
                                loaded_class_names):  # This is true if original_labels are 0-indexed and contiguous
                            final_class_names_for_model[new_label_idx] = loaded_class_names[original_label_val]
                        else:  # Fallback if original H5 labels were not 0-indexed or names are missing
                            final_class_names_for_model[new_label_idx] = f"OrigClass {original_label_val}"
                    except ValueError:
                        final_class_names_for_model[new_label_idx] = f"OrigClass {original_label_val}"  # Fallback
            else:  # No class names in HDF5
                for original_label_val, new_label_idx in label_map.items():
                    final_class_names_for_model[new_label_idx] = f"Class {new_label_idx} (Orig {original_label_val})"

            logging.info(
                f"Processed all labels. Remapped mapping: {label_map}. Final model class names: {final_class_names_for_model}")

        # ... (existing print statements for label checks - convert to logging) ...
        logging.info(
            f"Loaded data from {filepath}. IQ shape: {iq_data.shape}, Processed Labels shape: {processed_labels.shape}")
        if len(processed_labels) > 0:
            logging.info(
                f"Min processed label: {np.min(processed_labels)}, Max processed label: {np.max(processed_labels)}")
            num_unique_final_labels = len(np.unique(processed_labels))
            logging.info(f"Number of unique processed labels: {num_unique_final_labels}")
            if num_expected_classes is not None:
                if num_unique_final_labels > num_expected_classes:
                    logging.critical(
                        f"Found {num_unique_final_labels} unique labels, but model expects {num_expected_classes}.")
                if np.max(processed_labels) >= num_expected_classes:
                    logging.critical(
                        f"Max processed label ({np.max(processed_labels)}) is out of bounds for model (0 to {num_expected_classes - 1}).")

        # Ensure final_class_names_for_model has length num_expected_classes if provided, pad if necessary
        if num_expected_classes is not None and len(final_class_names_for_model) < num_expected_classes:
            logging.warning(
                f"Number of derived class names ({len(final_class_names_for_model)}) is less than num_expected_classes ({num_expected_classes}). Padding with generic names.")
            final_class_names_for_model.extend(
                [f"GenClass {i}" for i in range(len(final_class_names_for_model), num_expected_classes)])
        elif num_expected_classes is not None and len(final_class_names_for_model) > num_expected_classes:
            logging.warning(
                f"Number of derived class names ({len(final_class_names_for_model)}) is more than num_expected_classes ({num_expected_classes}). Truncating.")
            final_class_names_for_model = final_class_names_for_model[:num_expected_classes]

        return iq_data, processed_labels, final_class_names_for_model

    except FileNotFoundError:
        logging.error(f"File not found at {filepath}")
        raise
    except Exception as e:
        logging.error(f"Error loading HDF5 file {filepath}: {e}")
        raise


# ... (segment_data and SignalDataset as before, ensure prints are converted to logging) ...

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