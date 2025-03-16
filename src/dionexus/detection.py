import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths

def detect_peaks_in_signal(time, signal, min_height=0.1, prominence=0.05, distance=5,
                             max_width=None, min_sharpness=None,
                             width_rel=0.5, extension_factor=0.0):
    """
    Detect peaks in a 1D signal and filter out broad, flat peaks.
    
    This function now allows you to specify:
      - width_rel: the relative height at which the left/right bounds are computed (lower values extend the bounds)
      - extension_factor: an additional fraction of the computed width to extend the bounds.
    
    Parameters:
        time (np.array): 1D array of time points.
        signal (np.array): 1D array of conductance values.
        min_height (float): Minimum height (in µS) for a peak.
        prominence (float): Minimum prominence required.
        distance (int): Minimum number of data points between peaks.
        max_width (float, optional): Maximum allowed peak width (in same time units as `time`).
        min_sharpness (float, optional): Minimum allowed ratio of peak_height to width.
        width_rel (float): Relative height for computing peak widths (e.g. 0.5 by default; lower values extend bounds).
        extension_factor (float): Fraction of the computed width to further extend left/right boundaries.
        
    Returns:
        peaks_df (pd.DataFrame): DataFrame with detected and filtered peak properties:
            Columns: 'peak_index', 'peak_time', 'peak_height', 'prominence', 
                     'width', 'sharpness', 'left_time', 'right_time'
    """
    peaks, properties = find_peaks(signal, height=min_height, prominence=prominence, distance=distance)
    
    if len(peaks) > 0:
        # Calculate widths at a given relative height (width_rel) and get left/right interpolation indices
        results_half = peak_widths(signal, peaks, rel_height=width_rel)
        widths_samples = results_half[0]
        left_ips = results_half[2]
        right_ips = results_half[3]
        
        dt = np.median(np.diff(time))
        widths_time = widths_samples * dt
        # Convert left_ips and right_ips to time using interpolation
        left_time = np.interp(left_ips, np.arange(len(time)), time)
        right_time = np.interp(right_ips, np.arange(len(time)), time)
        
        # Extend the boundaries further by a fraction of the width if requested
        if extension_factor > 0:
            left_time = np.maximum(left_time - extension_factor * widths_time, time[0])
            right_time = np.minimum(right_time + extension_factor * widths_time, time[-1])
    else:
        widths_time = np.array([])
        left_time = np.array([])
        right_time = np.array([])
    
    peak_data = {
        'peak_index': peaks,
        'peak_time': time[peaks] if len(peaks) > 0 else np.array([]),
        'peak_height': properties['peak_heights'] if len(peaks) > 0 else np.array([]),
        'prominence': properties['prominences'] if len(peaks) > 0 else np.array([]),
        'width': widths_time,
        'left_time': left_time,
        'right_time': right_time,
    }
    peaks_df = pd.DataFrame(peak_data)
    
    # Only proceed if peaks were detected
    if not peaks_df.empty:
        # Calculate sharpness: height-to-width ratio
        peaks_df['sharpness'] = peaks_df['peak_height'] / peaks_df['width']
        
        # Filter by maximum width (exclude broad peaks)
        if max_width is not None:
            peaks_df = peaks_df[peaks_df['width'] <= max_width]
        
        # Filter by minimum sharpness (exclude flat peaks)
        if min_sharpness is not None:
            peaks_df = peaks_df[peaks_df['sharpness'] >= min_sharpness]
    
    return peaks_df
