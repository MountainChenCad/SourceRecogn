import h5py
import numpy as np
import sys

def print_hdf5_item_details(item, indent='', show_data_limit=5):
    """
    Recursively prints details of HDF5 groups and datasets,
    including attributes and a sample of data.
    """
    name = item.name
    print(f"{indent}Name: {name}")

    # Print Attributes
    if item.attrs:
        print(f"{indent}  Attributes:")
        for key, val in item.attrs.items():
            print(f"{indent}    {key}: {val}")
        print(f"{indent}  ---") # Separator

    if isinstance(item, h5py.Group):
        print(f"{indent}Type: Group")
        print(f"{indent}Members:")
        for key in item:
            print_hdf5_item_details(item[key], indent + '  ', show_data_limit)

    elif isinstance(item, h5py.Dataset):
        print(f"{indent}Type: Dataset")
        print(f"{indent}Shape: {item.shape}")
        print(f"{indent}Dtype: {item.dtype}")

        if show_data_limit > 0 and item.size > 0: # Only show data if limit > 0 and dataset not empty
            print(f"{indent}Data Sample (first {show_data_limit} elements/slice):")
            try:
                # Handle different dimensions for slicing preview
                if item.ndim == 0: # Scalar dataset
                     sample = item[()]
                elif item.ndim == 1:
                    limit = min(show_data_limit, item.shape[0])
                    sample = item[:limit]
                elif item.ndim == 2:
                    limit0 = min(show_data_limit, item.shape[0])
                    limit1 = min(show_data_limit, item.shape[1])
                    sample = item[:limit0, :limit1]
                elif item.ndim == 3: # Like Data_IQ
                    limit0 = min(show_data_limit, item.shape[0])
                    # For IQ data (dim 1 often 2), show both channels
                    limit1 = item.shape[1]
                    limit2 = min(show_data_limit, item.shape[2])
                    # Show maybe just the first sample's beginning
                    sample = item[0, :limit1, :limit2]
                    print(f"{indent}  (Showing start of first sample along dim 0)")
                else: # Higher dimensions
                    # Create slice objects for the first 'show_data_limit' elements along each dim
                    slices = tuple(slice(min(show_data_limit, dim_size)) for dim_size in item.shape)
                    sample = item[slices]

                # Handle string data (often stored as bytes)
                if np.issubdtype(item.dtype, np.bytes_) or item.dtype == object:
                    try:
                        # Attempt decoding if it looks like bytes
                        if isinstance(sample, np.ndarray) and sample.size > 0:
                            # Decode individual elements if they are bytes
                           decoded_sample = np.array([s.decode('utf-8', errors='replace') if isinstance(s, bytes) else s for s in sample.flat]).reshape(sample.shape)
                           print(f"{indent}  {decoded_sample}")
                        elif isinstance(sample, bytes):
                            print(f"{indent}  {sample.decode('utf-8', errors='replace')}")
                        else: # Already string or other object
                            print(f"{indent}  {sample}")
                    except Exception as decode_err:
                        print(f"{indent}  (Could not auto-decode string data, showing raw): {sample}")
                        # print(f"{indent}  Decode Error: {decode_err}") # Optional: show decode error
                else: # Numerical data
                    print(f"{indent}  {sample}")

            except Exception as e:
                print(f"{indent}  Error reading data sample: {e}")
    else:
        print(f"{indent}Type: Unknown")

    print("-" * (len(indent) + 20)) # Separator between items

# --- Main part of the script ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python show_h5_detailed.py <your_file.h5> [show_data_limit]")
        sys.exit(1)

    filename = sys.argv[1]
    data_limit = 50 # Default number of elements/slice dimension to show
    if len(sys.argv) > 2:
        try:
            data_limit = int(sys.argv[2])
        except ValueError:
            print("Warning: Invalid number for show_data_limit, using default (5).")


    try:
        with h5py.File(filename, 'r') as f:
            print(f"--- Detailed Information for {filename} ---")
            print_hdf5_item_details(f, show_data_limit=data_limit) # Start from the root group '/'
            print("--- End of Information ---")

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
    except Exception as e:
        print(f"An error occurred while reading '{filename}': {e}")
