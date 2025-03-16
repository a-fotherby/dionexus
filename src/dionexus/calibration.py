import pandas as pd
import numpy as np

def load_calibration_csv(csv_file):
    """
    Load the calibration CSV file containing expected elution order.
    Expected columns: 'Analyte' and (optionally) 'ExpectedRetentionTime'.
    The order of rows defines the expected elution order.
    
    Returns:
        calib_df (pd.DataFrame): DataFrame with calibration information.
    """
    calib_df = pd.read_csv(csv_file)
    return calib_df

def assign_calibration_peaks(peaks_df, calib_df):
    """
    Assign analyte identities to peaks in a calibration sample based on expected elution order.
    
    Parameters:
        peaks_df (pd.DataFrame): DataFrame of detected peaks for a calibration sample.
                                  Must have a 'peak_time' column.
        calib_df (pd.DataFrame): Calibration DataFrame with at least a column 'Analyte'.
        
    Returns:
        peaks_df (pd.DataFrame): Updated DataFrame with an 'assigned_analyte' column.
                                 Peaks are assigned in order of increasing retention time.
    """
    # Sort detected peaks by retention time
    peaks_df = peaks_df.sort_values(by='peak_time').reset_index(drop=True)
    
    num_expected = len(calib_df)
    assigned = []
    for i in range(len(peaks_df)):
        if i < num_expected:
            analyte = calib_df.iloc[i]['Analyte']
        else:
            analyte = None  # Extra peaks are not assigned
        assigned.append(analyte)
    peaks_df['assigned_analyte'] = assigned
    return peaks_df

def average_calibration_retention_times(calib_peak_results):
    """
    Calculate average retention times for each analyte based on calibration samples.
    
    Parameters:
        calib_peak_results (dict): Dictionary where keys are sample names and values
                                   are DataFrames of detected peaks (with 'assigned_analyte' and 'peak_time').
            
    Returns:
        avg_retention (dict): Mapping from analyte to average retention time.
    """
    analyte_times = {}
    for sample, df in calib_peak_results.items():
        for analyte in df['assigned_analyte'].dropna().unique():
            times = df.loc[df['assigned_analyte'] == analyte, 'peak_time'].values
            analyte_times.setdefault(analyte, []).extend(times)
    
    avg_retention = {analyte: np.mean(times) for analyte, times in analyte_times.items() if len(times) > 0}
    return avg_retention
