import matplotlib.pyplot as plt
import numpy as np

def plot_chromatogram(time, signal, peaks_df, sample_name=None):
    """
    Plot the chromatogram with detected peaks annotated, including integration regions.
    
    Instead of annotating the integration area, this function annotates each peak
    with its assigned analyte identity.
    
    Parameters:
        time (np.array): 1D array of time points.
        signal (np.array): 1D array of conductance values.
        peaks_df (pd.DataFrame): DataFrame of detected peaks with columns including
                                 'peak_time', 'peak_height', 'left_time', 'right_time',
                                 and 'assigned_analyte'.
        sample_name (str, optional): Identifier for the sample (displayed in the title).
    """
    plt.figure(figsize=(10, 6))
    plt.plot(time, signal, label='Chromatogram', color='blue')
    
    if peaks_df is not None and not peaks_df.empty:
        # Plot detected peaks as red crosses
        plt.plot(peaks_df['peak_time'], peaks_df['peak_height'], 'rx', markersize=10, label='Detected Peaks')
        
        # For each peak, fill the integration region and annotate with assigned analyte
        for _, row in peaks_df.iterrows():
            lt = row['left_time']
            rt = row['right_time']
            # Create mask for the integration window
            mask = (time >= lt) & (time <= rt)
            # Fill the integration region
            plt.fill_between(time[mask], signal[mask], alpha=0.3, color='orange',
                             label='Peak Area' if _ == 0 else "")
            # Get the assigned analyte for annotation
            assigned = row.get('assigned_analyte', '')
            if assigned is None:
                assigned = ''
            # Annotate near the center of the integration window
            mid_time = (lt + rt) / 2
            plt.text(mid_time, row['peak_height'], assigned,
                     ha='center', va='bottom', fontsize=8, color='darkred')
    
    title = f"Chromatogram - Sample {sample_name}" if sample_name else "Chromatogram"
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Conductance (µS)")
    plt.legend()
    plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_analytes_by_sample(ds):
    """
    Create a grouped bar chart with one entry per sample.
    
    For each sample (x-axis), the chart shows bars (with error bars) for each analyte
    representing the average concentration (in mM). Each sample's x-axis has its own labels.
    
    Parameters:
        ds (xr.Dataset): xarray Dataset containing variables:
            'concentration', 'concentration_error', 'assigned_analyte'
            with dimensions ("sample", "peak").
    """
    # Convert the dataset to a DataFrame and reset index.
    df = ds.to_dataframe().reset_index()
    # Keep only rows where an analyte was assigned.
    df = df[df['assigned_analyte'].notnull()]
    
    # Group by sample and analyte, computing mean and standard deviation.
    grouped = df.groupby(['sample', 'assigned_analyte'])
    summary = grouped.agg(
        concentration_mean=('concentration', 'mean'),
        concentration_std=('concentration', 'std'),
        n=('concentration', 'count')
    ).reset_index()
    
    # Compute standard error; if only one measurement, the std will be NaN.
    summary['concentration_se'] = summary['concentration_std'] / np.sqrt(summary['n'])
    
    # Pivot the summary so that rows are samples and columns are analytes.
    pivot_mean = summary.pivot(index='sample', columns='assigned_analyte', values='concentration_mean')
    pivot_se = summary.pivot(index='sample', columns='assigned_analyte', values='concentration_se')
    
    # Ensure samples are sorted.
    pivot_mean = pivot_mean.sort_index()
    pivot_se = pivot_se.reindex(pivot_mean.index)
    
    samples = pivot_mean.index.tolist()
    analytes = pivot_mean.columns.tolist()
    
    x = np.arange(len(samples))  # positions on the x-axis for each sample
    width = 0.8 / len(analytes)   # bar width for grouped bars
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, analyte in enumerate(analytes):
        means = pivot_mean[analyte].values
        errors = pivot_se[analyte].values
        # Compute an offset for each analyte's bar within a sample group.
        offset = (i - (len(analytes)-1)/2) * width
        ax.bar(x + offset, means, width, yerr=errors, capsize=5, label=analyte)
    
    ax.set_xticks(x)
    ax.set_xticklabels(samples)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Concentration (mM)")
    ax.set_title("Concentration by Sample (Grouped by Analyte)")
    ax.legend(title="Analyte")
    plt.show()


def plot_samples_by_analyte(ds):
    """
    Create individual bar charts for each analyte showing the concentration in each sample.
    
    For each analyte, a separate figure (or subplot) is generated with its own x-axis labels,
    showing the average concentration (in mM) with error bars representing the standard error.
    
    Parameters:
        ds (xr.Dataset): xarray Dataset containing variables:
            'concentration', 'concentration_error', 'assigned_analyte'
            with dimensions ("sample", "peak").
    """
    # Convert dataset to DataFrame.
    df = ds.to_dataframe().reset_index()
    # Keep only rows where an analyte was assigned.
    df = df[df['assigned_analyte'].notnull()]
    
    # Group by analyte and sample.
    grouped = df.groupby(['assigned_analyte', 'sample'])
    summary = grouped.agg(
        concentration_mean=('concentration', 'mean'),
        concentration_std=('concentration', 'std'),
        n=('concentration', 'count')
    ).reset_index()
    summary['concentration_se'] = summary['concentration_std'] / np.sqrt(summary['n'])
    
    analytes = summary['assigned_analyte'].unique()
    
    # Create a separate figure for each analyte.
    for analyte in analytes:
        sub_df = summary[summary['assigned_analyte'] == analyte]
        sub_df = sub_df.sort_values(by='sample')
        x = np.arange(len(sub_df))
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x, sub_df['concentration_mean'], yerr=sub_df['concentration_se'], capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(sub_df['sample'], rotation=45, ha='right')
        ax.set_xlabel("Sample")
        ax.set_ylabel("Concentration (mM)")
        ax.set_title(f"Concentration for {analyte} Across Samples")
        plt.tight_layout()
        plt.show()
