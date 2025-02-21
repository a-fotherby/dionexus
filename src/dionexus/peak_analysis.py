import numpy as np
from scipy.signal import find_peaks, peak_widths

def detect_peaks_and_integrate(ds, prominence=0.001, rel_height=0.5, max_peaks=None):
    """
    Detects peaks in the 'conductance' time series for each sample in the given xarray Dataset,
    subtracts a baseline from each peak before integration, and calculates the area under each peak.
    
    For each sample, the function:
      - Detects peaks using scipy.signal.find_peaks.
      - Optionally caps the number of detected peaks (max_peaks) by selecting those with highest prominence.
      - Uses scipy.signal.peak_widths (with the specified rel_height) to determine the left/right boundaries of each peak.
      - Subtracts a baseline (linearly interpolated between the boundaries) from the conductance signal before integration.
      - Calculates the area under the baseline-corrected peak using trapezoidal integration.
      - Prints out the number of peaks and their retention times.
    
    The following new variables are added to the dataset:
      - peak_count: number of peaks detected for each sample.
      - peak_retention_time: the time (in s) at which each peak reaches its maximum.
      - peak_area: the integrated area (in µS*s) under each detected peak (baseline-corrected).
      - peak_start_time: the start time (in s) of the peak (left boundary).
      - peak_end_time: the end time (in s) of the peak (right boundary).
    
    Parameters:
      ds (xr.Dataset): The input dataset with coordinates "time" (s) and "sample",
                       and a variable "conductance".
      prominence (float): Minimum prominence required for a peak (default 0.001).
      rel_height (float): The relative height at which the peak width is measured (default 0.5).
                          Lowering this value will measure a wider base.
      max_peaks (int, optional): Maximum number of peaks to keep per sample. If more peaks are detected,
                          only the ones with the highest prominence are retained.
    
    Returns:
      xr.Dataset: The original dataset augmented with the new peak detection variables.
    """
    # Get the time vector (assumed to be common to all samples)
    time = ds['time'].values
    n_time = len(time)
    
    # Determine number of samples.
    samples = ds['sample'].values
    n_samples = len(samples)
    
    # We'll collect peak info for each sample in a list.
    sample_peak_info = []
    overall_max_peaks = 0
    
    # Loop over samples.
    for sample in samples:
        # Extract the conductance time series for the current sample and squeeze to 1D.
        data = ds['conductance'].sel(sample=sample).squeeze().values
        
        # Find peaks using the given prominence.
        peaks, properties = find_peaks(data, prominence=prominence)
        
        # Cap the number of peaks if max_peaks is provided.
        if max_peaks is not None and len(peaks) > max_peaks:
            # Select indices corresponding to the highest prominences.
            sort_idx = np.argsort(properties['prominences'])[-max_peaks:]
            peaks = peaks[sort_idx]
            for key in properties:
                properties[key] = properties[key][sort_idx]
            # Re-sort peaks by their time (ascending order)
            sort_time_idx = np.argsort(time[peaks])
            peaks = peaks[sort_time_idx]
            for key in properties:
                properties[key] = properties[key][sort_time_idx]
        
        # Determine peak widths; returns widths, width_heights, left_ips, right_ips.
        width_results = peak_widths(data, peaks, rel_height=rel_height)
        left_ips = width_results[2]  # fractional indices for left boundary
        right_ips = width_results[3]  # fractional indices for right boundary
        
        # Compute retention times (time at the peak's maximum).
        retention_times = time[peaks]
        
        # For each detected peak, compute the baseline-corrected area.
        areas = []
        start_times = []
        end_times = []
        for i, peak in enumerate(peaks):
            # Convert fractional indices to integer boundaries.
            left_idx = int(np.floor(left_ips[i]))
            right_idx = int(np.ceil(right_ips[i]))
            left_idx = max(left_idx, 0)
            right_idx = min(right_idx, n_time - 1)
            
            # Define the time segment for the peak.
            time_segment = time[left_idx:right_idx+1]
            # Estimate the baseline by linear interpolation between the values at the boundaries.
            baseline = np.interp(time_segment, [time[left_idx], time[right_idx]], [data[left_idx], data[right_idx]])
            # Subtract the baseline from the signal.
            corrected_signal = data[left_idx:right_idx+1] - baseline
            # Compute the area using the trapezoidal rule.
            area = np.trapz(corrected_signal, x=time_segment)
            areas.append(area)
            start_times.append(time[left_idx])
            end_times.append(time[right_idx])
        
        num_peaks = len(peaks)
        if num_peaks > overall_max_peaks:
            overall_max_peaks = num_peaks
        
        print(f"Sample {sample}: {num_peaks} peaks detected. Retention times (s): {retention_times}")
        
        sample_peak_info.append({
            'num_peaks': num_peaks,
            'retention_times': retention_times,
            'areas': np.array(areas),
            'start_times': np.array(start_times),
            'end_times': np.array(end_times)
        })
    
    # Create uniform arrays with shape (n_samples, overall_max_peaks) and pad with NaN where needed.
    retention_array = np.full((n_samples, overall_max_peaks), np.nan)
    area_array = np.full((n_samples, overall_max_peaks), np.nan)
    start_array = np.full((n_samples, overall_max_peaks), np.nan)
    end_array = np.full((n_samples, overall_max_peaks), np.nan)
    
    for i, info in enumerate(sample_peak_info):
        n_peaks = info['num_peaks']
        if n_peaks > 0:
            retention_array[i, :n_peaks] = info['retention_times']
            area_array[i, :n_peaks] = info['areas']
            start_array[i, :n_peaks] = info['start_times']
            end_array[i, :n_peaks] = info['end_times']
    
    # Add new variables to the dataset.
    ds = ds.assign(
        peak_count=("sample", np.array([info['num_peaks'] for info in sample_peak_info])),
        peak_retention_time=(("sample", "peak"), retention_array),
        peak_area=(("sample", "peak"), area_array),
        peak_start_time=(("sample", "peak"), start_array),
        peak_end_time=(("sample", "peak"), end_array)
    )
    
    # Add attributes (metadata) for these new variables.
    ds['peak_count'].attrs = {"long_name": "Number of Peaks Detected"}
    ds['peak_retention_time'].attrs = {"units": "s", "long_name": "Peak Retention Time"}
    ds['peak_area'].attrs = {"units": "µS*s", "long_name": "Area Under Peak (baseline-corrected)"}
    ds['peak_start_time'].attrs = {"units": "s", "long_name": "Peak Start Time"}
    ds['peak_end_time'].attrs = {"units": "s", "long_name": "Peak End Time"}
    
    return ds


def assign_peak_identities(ds, theoretical_df, tolerance=20):
    """
    Assigns identities to each detected peak in the dataset based on theoretical retention times.
    
    For each detected peak (in ds['peak_retention_time']), this function:
      - Computes the absolute difference between the detected retention time and each theoretical retention time 
        (from theoretical_df).
      - If the smallest difference is within the specified tolerance (in seconds), assigns the corresponding 
        analyte name (from theoretical_df['Analyte']) to that peak.
      - Otherwise, labels the peak as "unknown".
    
    The theoretical_df is expected to have at least the following columns:
        - "Analyte": The name/identity of the peak.
        - "RetentionTime": The theoretical retention time (in seconds) for that analyte.
    
    Parameters:
      ds (xr.Dataset): The xarray dataset containing the variable 'peak_retention_time' (dimensions: sample x peak).
      theoretical_df (pd.DataFrame): DataFrame with columns "Analyte" and "RetentionTime".
      tolerance (float): The maximum allowed difference (in seconds) between a detected and theoretical retention time.
    
    Returns:
      xr.Dataset: The input dataset augmented with a new variable "peak_identity" (dimensions: sample x peak),
                  which contains the assigned identity (string) for each peak.
    """
    # Extract the detected peak retention times (assumed shape: (n_samples, n_peaks)).
    detected_rts = ds['peak_retention_time'].values  
    n_samples, n_peaks = detected_rts.shape
    
    # Prepare an array of type object to hold string identities.
    peak_identity = np.empty((n_samples, n_peaks), dtype=object)
    
    # Loop over each sample and each detected peak.
    for i in range(n_samples):
        for j in range(n_peaks):
            rt = detected_rts[i, j]
            # If the retention time is NaN (i.e. no peak was detected in that slot), leave blank.
            if np.isnan(rt):
                peak_identity[i, j] = ""
            else:
                # Compute the absolute differences with theoretical retention times.
                diffs = np.abs(theoretical_df['Ret. Time'].values - rt)
                min_diff = np.min(diffs)
                if min_diff <= tolerance:
                    # Choose the analyte with the minimum difference.
                    idx = np.argmin(diffs)
                    peak_identity[i, j] = theoretical_df['Analyte'].iloc[idx]
                else:
                    peak_identity[i, j] = "unknown"
    
    # Assign the new variable to the dataset.
    ds = ds.assign(peak_identity=(("sample", "peak"), peak_identity))
    
    # Optionally add metadata attributes.
    ds['peak_identity'].attrs = {
        "long_name": "Assigned Peak Identity",
        "description": ("Peak identities are assigned based on the detected retention time "
                        "compared with theoretical retention times within a tolerance of "
                        f"{tolerance} s.")
    }
    
    return ds