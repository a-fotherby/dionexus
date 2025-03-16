def assign_unknown_peaks(peaks_df, avg_retention, tolerance=30):
    """
    Assign peaks in unknown samples to analytes based on calibrated average retention times.
    
    Parameters:
        peaks_df (pd.DataFrame): DataFrame of detected peaks for an unknown sample.
                                 Must include a 'peak_time' column.
        avg_retention (dict): Dictionary mapping analyte names to average retention times.
        tolerance (float): Maximum allowed difference (in seconds) for assignment.
        
    Returns:
        peaks_df (pd.DataFrame): Updated DataFrame with an 'assigned_analyte' column.
                                 Peaks outside tolerance remain unassigned (None).
    """
    assigned = []
    for pt in peaks_df['peak_time']:
        assigned_analyte = None
        min_diff = float('inf')
        for analyte, ref_time in avg_retention.items():
            diff = abs(pt - ref_time)
            if diff < tolerance and diff < min_diff:
                min_diff = diff
                assigned_analyte = analyte
        assigned.append(assigned_analyte)
    peaks_df['assigned_analyte'] = assigned
    return peaks_df
