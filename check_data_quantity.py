import h5py
import numpy as np
import os


def count_total_segments(h5_filepath, segment_length, stride, data_iq_key='Data_IQ'):
    """
    Counts the total number of segments that can be generated from an HDF5 file.
    Assumes Data_IQ shape is (num_raw_samples, num_channels, total_sample_length).
    """
    if not os.path.exists(h5_filepath):
        print(f"Error: File not found at {h5_filepath}")
        return 0, 0, []

    total_segments_count = 0
    raw_sample_count = 0
    original_lengths = []

    try:
        with h5py.File(h5_filepath, 'r') as f:
            if data_iq_key not in f:
                print(f"Error: Dataset '{data_iq_key}' not found in {h5_filepath}.")
                return 0, 0, []

            data_iq = f[data_iq_key]
            raw_sample_count = data_iq.shape[0]

            for i in range(raw_sample_count):
                # Assuming shape is (num_samples, num_channels, total_length)
                # For audio from WAV, it's likely (num_samples, num_channels, max_length_padded)
                # For source, it's (8, 2, 10000000)
                sample_length = data_iq.shape[2]  # Length of the I or Q channel data
                original_lengths.append(sample_length)

                num_segments_for_this_sample = 0
                start = 0
                while start + segment_length <= sample_length:
                    num_segments_for_this_sample += 1
                    start += stride
                total_segments_count += num_segments_for_this_sample

        return raw_sample_count, total_segments_count, original_lengths

    except Exception as e:
        print(f"Error processing file {h5_filepath}: {e}")
        return 0, 0, []


# --- Configuration ---
SOURCE_H5_FILE = "dataset/source/BPSK-500Kbps-DATA.h5"
# 请确保这个文件名与你用 create_target_h5.py 生成的目标文件名一致
TARGET_H5_FILE = "dataset/target/TARGET-BPSK-DATA.h5"

SEGMENT_LENGTH = 1024  # 使用你训练时常用的片段长度
STRIDE = 1024  # 使用你训练时常用的步长 (如果重叠，可以调整)

print(f"Comparing data quantity with segment_length={SEGMENT_LENGTH}, stride={STRIDE}\n")

# --- Source Domain ---
print("--- Source Domain Analysis ---")
src_raw_count, src_total_segments, src_orig_lengths = count_total_segments(SOURCE_H5_FILE, SEGMENT_LENGTH, STRIDE)
if src_raw_count > 0:
    print(f"File: {SOURCE_H5_FILE}")
    print(f"  Number of raw samples: {src_raw_count}")
    if src_orig_lengths:
        unique_src_lengths = list(set(src_orig_lengths))
        print(
            f"  Original sample length(s) per IQ channel: {unique_src_lengths[0] if len(unique_src_lengths) == 1 else unique_src_lengths}")
    print(f"  Total potential segments: {src_total_segments}")
else:
    print(f"Could not analyze source file: {SOURCE_H5_FILE}")

# --- Target Domain ---
print("\n--- Target Domain Analysis ---")
tgt_raw_count, tgt_total_segments, tgt_orig_lengths = count_total_segments(TARGET_H5_FILE, SEGMENT_LENGTH, STRIDE)
if tgt_raw_count > 0:
    print(f"File: {TARGET_H5_FILE}")
    print(f"  Number of raw samples: {tgt_raw_count}")
    if tgt_orig_lengths:
        unique_tgt_lengths = list(set(tgt_orig_lengths))
        # For target_domain.h5, all samples are padded to the same max_length
        print(
            f"  Padded sample length(s) per channel: {unique_tgt_lengths[0] if len(unique_tgt_lengths) == 1 else unique_tgt_lengths}")
    print(f"  Total potential segments: {tgt_total_segments}")
else:
    print(f"Could not analyze target file: {TARGET_H5_FILE}")

# --- Comparison ---
print("\n--- Conclusion ---")
if src_raw_count > 0 and tgt_raw_count > 0:
    if tgt_total_segments < src_total_segments:
        diff = src_total_segments - tgt_total_segments
        percentage = (diff / src_total_segments) * 100 if src_total_segments > 0 else 0
        print(
            f"Yes, the target domain ({tgt_total_segments} segments) has less data than the source domain ({src_total_segments} segments).")
        print(f"The difference is {diff} segments ({percentage:.2f}% less).")
    elif tgt_total_segments > src_total_segments:
        diff = tgt_total_segments - src_total_segments
        percentage = (diff / src_total_segments) * 100 if src_total_segments > 0 else float('inf')
        print(
            f"No, the target domain ({tgt_total_segments} segments) has MORE data than the source domain ({src_total_segments} segments).")
        print(f"The difference is {diff} segments ({percentage:.2f}% more).")
    else:
        print(
            f"The target domain ({tgt_total_segments} segments) and source domain ({src_total_segments} segments) have approximately the same amount of segmentable data.")
else:
    print("Could not perform comparison due to issues analyzing one or both files.")
