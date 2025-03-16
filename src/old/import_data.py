import os
import xarray as xr
import pandas as pd
from io import StringIO

def load_data(file_path):
    """
    Reads an ion-chromatograph file and returns an xarray Dataset.
    
    The file contains a metadata header followed by CSV data with three columns:
        Time(min)   Step(sec)   Value(µS)
    This function:
      - Extracts the metadata (using the "Sample" field for a sample coordinate)
      - Reads the CSV without using the first column as an index
      - Converts Time(min) to seconds
      - Ignores the Step(sec) column
      - Creates an xarray Dataset with a "time" coordinate (in seconds)
        and a "sample" coordinate (with the sample ID)
      - Adds units and descriptive metadata to all variables and coordinates.
    
    Parameters:
        file_path (str): Path to the ion-chromatograph data file.
    
    Returns:
        xr.Dataset: Dataset with dimensions "time" and "sample", and a data variable "Conductance".
    """
    # Open the file using an encoding that avoids Unicode errors.
    with open(file_path, 'r', encoding='latin-1') as f:
        lines = f.readlines()
    
    # Separate metadata lines and CSV data lines.
    metadata_lines = []
    data_lines = []
    in_data = False
    for line in lines:
        if "Raw Data:" in line:
            in_data = True
            continue  # Skip the marker line.
        if in_data:
            data_lines.append(line)
        else:
            metadata_lines.append(line)
    
    # Parse metadata lines (assume key-value pairs separated by tabs).
    metadata = {}
    for line in metadata_lines:
        line = line.strip()
        if not line:
            continue
        if "\t" in line:
            parts = line.split("\t")
            if len(parts) >= 2:
                key = parts[0].strip().strip(':')
                value = parts[1].strip()
                metadata[key] = value
    
    # Get the sample ID; default to "unknown" if not found.
    sample_id = metadata.get("Sample", "unknown")
    
    # Join CSV lines and read into a pandas DataFrame without assigning an index.
    csv_str = "\n".join(data_lines)
    df = pd.read_csv(StringIO(csv_str), delimiter='\t', index_col=False)
    
    # If the DataFrame has an index set from the file, reset it.
    if df.index.name == "Time(min)":
        df = df.reset_index()
    
    # Convert "Time(min)" to seconds.
    df['time_sec'] = df['Time(min)'] * 60
    
    # Create DataArrays for the coordinates with metadata.
    time_coord = xr.DataArray(
        df['time_sec'].values,
        dims=["time"],
        attrs={"units": "s", "long_name": "Time (seconds)"}
    )
    sample_coord = xr.DataArray(
        [sample_id],
        dims=["sample"],
        attrs={"long_name": "Sample ID"}
    )
    
    # Create the DataArray for Conductance with metadata.
    conductance = xr.DataArray(
        df[['Value(µS)']].values.reshape(-1, 1),
        dims=["time", "sample"],
        attrs={"units": "µS", "long_name": "Conductance"}
    )
    
    # Build the xarray Dataset.
    ds = xr.Dataset(
        data_vars={
            "conductance": conductance
        },
        coords={
            "time": time_coord,
            "sample": sample_coord
        }
    )
    
    return ds


def load_all_data(directory_path):
    """
    Imports all ion-chromatograph data files from a given directory and returns
    a single xarray Dataset that combines each file along a new "sample" coordinate.
    
    Each file is loaded using the load_dionex_data() function (which extracts the
    "Sample" from the file's metadata and builds a dataset with coordinates "time" and "sample").
    
    If duplicate sample names are detected, they are mangled by appending an underscore
    and a counter (e.g. "S17" -> "S17_1", "S17_2", etc.).
    
    After concatenation, the "sample" coordinate is sorted in increasing alphanumeric order.
    
    Parameters:
        directory_path (str): Path to the directory containing the data files.
        
    Returns:
        xr.Dataset: A combined dataset with dimensions "time" and "sample", where the data variable
                    "conductance" contains the measurements from all files.
    """
    datasets = []
    sample_counts = {}  # Dictionary to track occurrences of sample names.
    
    # Iterate over all files in the directory.
    for filename in os.listdir(directory_path):
        # Adjust the extension filter as needed (here we assume files end with .TXT)
        if filename.lower().endswith('.txt'):
            full_path = os.path.join(directory_path, filename)
            ds = load_data(full_path)
            # Extract sample name (assumes ds['sample'] is a one-element array)
            sample_name = ds['sample'].values.item()
            # Check for duplicates and mangle if necessary.
            if sample_name in sample_counts:
                sample_counts[sample_name] += 1
                new_sample_name = f"{sample_name}_{sample_counts[sample_name]}"
            else:
                sample_counts[sample_name] = 0
                new_sample_name = sample_name
            
            # Reassign the sample coordinate in this dataset.
            ds = ds.assign_coords(sample=[new_sample_name])
            datasets.append(ds)
    
    if not datasets:
        raise ValueError("No data files found in the provided directory.")
    
    # Concatenate along the 'sample' dimension.
    combined_ds = xr.concat(datasets, dim="sample")
    
    # Reorder the 'sample' coordinate in increasing alphanumeric order.
    combined_ds = combined_ds.sortby("sample")
    
    return combined_ds
