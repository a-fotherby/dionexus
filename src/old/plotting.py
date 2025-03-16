
import numpy as np
import matplotlib.pyplot as plt

def plot_sample_peaks(ds, sample=None):
    """
    Plots the conductance curve for a given sample and highlights the detected peaks.
    
    The dataset is expected to include:
      - A "time" coordinate (in seconds) with a descriptive 'long_name'.
      - A "conductance" variable with associated 'long_name' and 'units'.
      - Peak detection variables for each sample: 
          "peak_retention_time", "peak_start_time", and "peak_end_time"
      - Optionally, a "peak_identity" variable (dimensions: sample x peak) that
        contains the assigned analyte names for each peak.
    
    Parameters:
      ds (xr.Dataset): Combined xarray dataset containing the data and peak information.
      sample (str, optional): The sample identifier to plot. If None, the first sample is used.
      
    The function:
      - Plots the conductance time series.
      - Shades the region corresponding to each detected peak.
      - Marks the retention time (the time of peak maximum) with a red circle.
      - Annotates each peak using the identity from "peak_identity" if available, 
        otherwise labels with the peak number.
    """
    # Choose sample: if not provided, use the first sample.
    if sample is None:
        sample = ds['sample'].values[0]
    
    # Select the data for the chosen sample.
    sample_ds = ds.sel(sample=sample)
    time = sample_ds['time'].values
    conductance = sample_ds['conductance'].values
    
    # Retrieve peak detection variables.
    retention_times = sample_ds['peak_retention_time'].squeeze().values
    start_times = sample_ds['peak_start_time'].squeeze().values
    end_times = sample_ds['peak_end_time'].squeeze().values
    
    # Optionally, retrieve peak identity if available.
    if 'peak_identity' in ds:
        peak_identities = sample_ds['peak_identity'].squeeze().values
    else:
        peak_identities = None
    
    # Filter out any padded NaN entries.
    valid = ~np.isnan(retention_times)
    retention_times = retention_times[valid]
    start_times = start_times[valid]
    end_times = end_times[valid]
    if peak_identities is not None:
        peak_identities = np.array(peak_identities)[valid]
    
    plt.figure(figsize=(10, 6))
    plt.plot(time, conductance, label="Conductance", color="blue")
    
    # Loop over each detected peak.
    for i, (rt, st, et) in enumerate(zip(retention_times, start_times, end_times)):
        # Shade the peak region.
        mask = (time >= st) & (time <= et)
        plt.fill_between(time, conductance, where=mask, color='orange', alpha=0.3,
                         label="Peak span" if i == 0 else None)
        # Mark the retention time with a red circle.
        peak_val = np.interp(rt, time, conductance)
        plt.plot(rt, peak_val, "ro", markersize=8, label="Retention time" if i == 0 else None)
        
        # Determine the label: if peak_identity exists and is non-empty, use it; else use peak number.
        if peak_identities is not None and peak_identities[i]:
            label_text = f"{peak_identities[i]}"
        else:
            label_text = f"Peak {i+1}"
            
        # Annotate the peak.
        plt.annotate(label_text, (rt, peak_val), textcoords="offset points",
                     xytext=(0,10), ha='center', fontsize=9)
    
    # Use metadata for axis labels.
    time_label = sample_ds['time'].attrs.get('long_name', 'Time (s)')
    conductance_label = sample_ds['conductance'].attrs.get('long_name', 'Conductance')
    conductance_units = sample_ds['conductance'].attrs.get('units', '')
    
    plt.xlabel(time_label)
    plt.ylabel(f"{conductance_label} ({conductance_units})")
    plt.title(f"Conductance Curve and Detected Peaks for Sample {sample}")
    plt.legend()
    plt.tight_layout()
    plt.show()