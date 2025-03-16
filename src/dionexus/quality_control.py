def quality_filter(peaks_df, max_asymmetry=2.0):
    """
    Filter peaks based on quality control criteria.
    
    For this example, if a peak has an 'asymmetry' value (assumed to be computed elsewhere)
    that exceeds max_asymmetry, the peak is flagged as failing QC.
    
    Parameters:
        peaks_df (pd.DataFrame): DataFrame of detected peaks.
        max_asymmetry (float): Maximum allowed asymmetry factor.
        
    Returns:
        peaks_df (pd.DataFrame): DataFrame with an added 'QC_flag' column (True if passing QC).
    """
    if 'asymmetry' not in peaks_df.columns:
        # If asymmetry hasn't been computed, assume QC passes.
        peaks_df['QC_flag'] = True
    else:
        peaks_df['QC_flag'] = peaks_df['asymmetry'] <= max_asymmetry
    return peaks_df
