
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def assign_standard_concentrations(ds, csv_file, calib_standard_samples, dilution_factors):
    """
    Constructs calibration concentrations for each peak based on the assigned peak identities 
    (in ds["peak_identity"]) and stock concentrations from a CSV file.
    
    The CSV file must have columns:
      Name,Approx. Concentration,Molar Mass (g/mol),Concentration (mM)
    where the "Concentration (mM)" column gives the stock concentration (for G-100).
    
    For each sample in ds, if the sample is one of the calibration standards (as provided in 
    calib_standard_samples), its dilution factor (from dilution_factors) is used to compute the expected 
    concentration for each peak by:
    
        concentration = stock_concentration * dilution_factor
    
    The peak’s species is determined by matching the value in ds["peak_identity"] to the "Name" in the CSV.
    For non‑calibration samples or peaks with no valid identity, the concentration is set to NaN.
    
    Parameters:
      ds (xr.Dataset): Dataset containing a variable "peak_identity" (dims: sample x peak).
      csv_file (str): Path to the CSV file containing standard data.
      calib_standard_samples (list or tuple): List of sample names (from ds["sample"]) that are calibration standards.
      dilution_factors (list or tuple): List of dilution factors corresponding to each calibration standard sample.
           For example, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0].
           
    Returns:
      xr.Dataset: The input dataset augmented with a new variable "concentration" (dims: sample x peak)
                  containing the computed concentrations in mM.
    """
    # Read the CSV file and set "Name" as the index.
    std_df = pd.read_csv(csv_file)
    std_df.set_index("Name", inplace=True)
    
    # Create a mapping from species name to its stock concentration (in mM)
    stock_conc = std_df["Concentration (mM)"].to_dict()
    
    # Extract the peak identities; assumed to be (n_samples, n_peaks).
    peak_ids = ds["peak_identity"].values
    
    # Get all sample names from the dataset.
    all_samples = ds["sample"].values
    n_samples = len(all_samples)
    n_peaks = peak_ids.shape[1]
    
    # Initialize an array for the concentration values.
    conc_arr = np.full((n_samples, n_peaks), np.nan)
    
    # Loop over each sample.
    for j, sample in enumerate(all_samples):
        # If sample is a calibration standard, retrieve its dilution factor; otherwise, leave factor as None.
        if sample in calib_standard_samples:
            idx = calib_standard_samples.index(sample)
            factor = dilution_factors[idx]
        else:
            factor = None
        
        # For each peak in this sample:
        for i in range(n_peaks):
            analyte = peak_ids[j, i]
            # If no analyte was assigned or not a calibration standard, set NaN.
            if analyte == "" or pd.isna(analyte) or factor is None:
                conc_arr[j, i] = np.nan
            else:
                # If the analyte is not found in the CSV, warn and assign NaN.
                if analyte not in stock_conc:
                    print(f"Warning: Analyte '{analyte}' in sample '{sample}' not found in CSV; skipping.")
                    conc_arr[j, i] = np.nan
                else:
                    conc_arr[j, i] = stock_conc[analyte] * factor
    
    # Add the computed concentration array to the dataset along the existing "peak" coordinate.
    ds = ds.assign(concentration=(("sample", "peak"), conc_arr))
    ds["concentration"].attrs = {
         "units": "mM",
         "long_name": "Calibration Concentration (mM)",
         "description": ("For each calibration standard sample, the concentration is computed by multiplying the stock "
                         "concentration (from the CSV) by the specified dilution factor, matched to the peak identity "
                         "in 'peak_identity'. Non-calibration samples and peaks with missing or unrecognized identities "
                         "are assigned NaN.")
    }
    return ds



def calibrate_species(ds, calib_standard_samples, marker='o', linestyle='-', colors=None):
    """
    For each analyte found in the calibration standard samples (as given by the variable 
    "peak_identity"), this function:
      - Extracts the concentration (from ds['concentration']) and measured Area (from ds['Area']) 
        for that analyte in the calibration standards.
      - Treats NaN values as zero.
      - Plots a scatter of concentration (x-axis) versus Area (y-axis) on a single figure.
      - Fits a linear regression to the calibration data and plots the regression line.
      - Annotates the plot with the regression equation and R².
    
    Parameters:
      ds (xr.Dataset): The dataset which must include:
                         - "peak_identity" (dims: sample x peak) with assigned analyte names.
                         - "concentration" (dims: sample x peak) containing calibration concentrations 
                           (for calibration standard samples; other samples are NaN).
                         - "Area" (dims: sample x peak) containing the measured areas.
                         - A coordinate "sample" listing sample names.
      calib_standard_samples (list or tuple): List of sample names (from ds["sample"]) that are the calibration standards.
      marker (str): Marker style for scatter plot (default 'o').
      linestyle (str): Line style for regression lines (default '-').
      colors (list): Optional list of colors for the analytes; if None, matplotlib defaults are used.
    
    Returns:
      dict: A dictionary mapping each analyte to its regression coefficients:
            { analyte: {"slope": ..., "intercept": ..., "R2": ...}, ... }
    """
    # Select calibration standard samples.
    calib_ds = ds.sel(sample=calib_standard_samples)
    
    # Extract the peak identities from the calibration standards.
    # This array has shape (n_calib, n_peaks)
    peak_ids = calib_ds["peak_identity"].values
    # Flatten and get unique analyte names, ignoring empty strings.
    unique_analytes = np.unique(peak_ids)
    unique_analytes = [a for a in unique_analytes if a != "" and not pd.isna(a)]
    
    coeffs = {}  # to store regression coefficients for each analyte.
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Loop over each analyte.
    for idx, analyte in enumerate(unique_analytes):
        conc_list = []
        area_list = []
        # Loop over each calibration sample.
        for sample in calib_standard_samples:
            sample_data = calib_ds.sel(sample=sample)
            # Find peaks with this analyte.
            # We assume that there is only one calibration peak per analyte per sample.
            da_peak = sample_data["peak_identity"]
            sel = da_peak == analyte
            if sel.sum() == 0:
                continue  # no peak for this analyte in this sample.
            # Get the first occurrence.
            conc_val = sample_data["concentration"].where(sel, drop=True).values[0]
            area_val = sample_data["peak_area"].where(sel, drop=True).values[0]
            # Treat NaNs as zero.
            if np.isnan(conc_val):
                conc_val = 0.0
            if np.isnan(area_val):
                area_val = 0.0
            conc_list.append(conc_val)
            area_list.append(area_val)
        
        if len(conc_list) == 0:
            continue  # Skip analytes with no calibration data.
        
        # Convert lists to arrays and sort by increasing concentration.
        conc_arr = np.array(conc_list)
        area_arr = np.array(area_list)
        sort_idx = np.argsort(conc_arr)
        conc_arr = conc_arr[sort_idx]
        area_arr = area_arr[sort_idx]
        
        # Plot scatter.
        if colors is None:
            color = None
        else:
            color = colors[idx % len(colors)]
        ax.scatter(conc_arr, area_arr, label=analyte, marker=marker, color=color)
        
        # Fit linear regression.
        model = LinearRegression()
        conc_reshaped = conc_arr.reshape(-1, 1)
        model.fit(conc_reshaped, area_arr)
        y_pred = model.predict(conc_reshaped)
        ax.plot(conc_arr, y_pred, linestyle=linestyle, color=color)
        
        r_squared = model.score(conc_reshaped, area_arr)
        slope = model.coef_[0]
        intercept = model.intercept_
        coeffs[analyte] = {"slope": slope, "intercept": intercept, "R2": r_squared}
        
        # Annotate the plot.
        ax.text(0.05, 0.95 - idx*0.05, 
                f"{analyte}: y = {slope:.2f}x + {intercept:.2f} (R² = {r_squared:.2f})",
                transform=ax.transAxes, color=color, fontsize=9, verticalalignment='top')
    
    ax.set_xlabel("Concentration (mM)")
    ax.set_ylabel("Area (µS*min)")
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    return coeffs


def compute_concentrations(ds, coeffs, calib_standard_samples):
    """
    Computes concentrations for each peak in the dataset using calibration regression coefficients,
    while preserving the calibration standard samples' pre-assigned concentrations. For non-calibration 
    samples, the regression model (Area = slope * concentration + intercept) is used:
    
         concentration = (Area - intercept) / slope
    
    and the concentration error is estimated by combining both the uncertainty in the peak integration 
    and the calibration regression error. Specifically, if:
    
         conc = (Area - intercept) / slope
         error_area = peak_area_error / |slope|
         error_calib = |conc| * (1 - R²)
    
    then the total error is computed as:
    
         concentration_error = sqrt( (error_area)² + (error_calib)² )
    
    For calibration standard samples (given in calib_standard_samples), the pre-assigned concentrations 
    are preserved, and the error is set to 0.
    
    Parameters:
      ds (xr.Dataset): The dataset containing:
           - "peak_identity" (dims: sample x peak): assigned analyte names for each peak.
           - "peak_area" (dims: sample x peak): measured peak areas.
           - "peak_area_error" (dims: sample x peak): integration error for each peak.
           - "concentration" (dims: sample x peak): pre-assigned concentrations for calibration standards.
           - A coordinate "sample".
      coeffs (dict): A dictionary mapping analyte names to regression coefficients, e.g.:
                     { "Fluoride": {"slope": ..., "intercept": ..., "R2": ...}, ... }
      calib_standard_samples (list or tuple): List of sample names (from ds["sample"]) that are calibration standards.
    
    Returns:
      xr.Dataset: The input dataset augmented with two new variables:
                 - "concentration" (dims: sample x peak): computed (or preserved) concentration (mM).
                 - "concentration_error" (dims: sample x peak): estimated error in concentration.
    """
    # Extract arrays for peak identity, peak_area, and peak_area_error.
    peak_ids = ds["peak_identity"].values   # shape: (n_samples, n_peaks)
    areas = ds["peak_area"].values            # shape: (n_samples, n_peaks)
    area_errs = ds["peak_area_error"].values  # shape: (n_samples, n_peaks)
    
    n_samples, n_peaks = peak_ids.shape
    conc_arr = np.full((n_samples, n_peaks), np.nan)
    error_arr = np.full((n_samples, n_peaks), np.nan)
    
    # Get all sample names.
    all_samples = ds["sample"].values
    
    for i, sample in enumerate(all_samples):
        if sample in calib_standard_samples:
            # For calibration standards, preserve the pre-assigned concentration and set error to 0.
            conc_arr[i, :] = ds["concentration"].sel(sample=sample).values
            error_arr[i, :] = 0.0
        else:
            # For non-calibration samples, compute concentration using regression.
            for j in range(n_peaks):
                analyte = peak_ids[i, j]
                if analyte == "" or pd.isna(analyte):
                    print(f"Warning: Sample '{sample}', peak {j+1} has no assigned identity; skipping concentration computation.")
                    continue
                if analyte not in coeffs:
                    print(f"Warning: Sample '{sample}', peak {j+1} has identity '{analyte}' not found in regression coefficients; skipping concentration computation.")
                    continue
                
                slope = coeffs[analyte]["slope"]
                intercept = coeffs[analyte]["intercept"]
                r2 = coeffs[analyte].get("R2", np.nan)
                if slope == 0:
                    print(f"Warning: Regression slope for '{analyte}' is zero in sample '{sample}', peak {j+1}; skipping concentration computation.")
                    continue
                
                measured_area = areas[i, j]
                measured_area_error = area_errs[i, j]
                if np.isnan(measured_area):
                    continue
                
                # Compute concentration using the calibration model.
                conc = (measured_area - intercept) / slope
                conc_arr[i, j] = conc
                
                # Calculate error components:
                # 1. From the peak area uncertainty:
                error_area = measured_area_error / np.abs(slope)
                # 2. From the calibration regression error:
                error_calib = np.abs(conc) * (1 - r2) if not np.isnan(r2) else np.nan
                # Combine the two errors in quadrature:
                error_arr[i, j] = np.sqrt(error_area**2 + error_calib**2)
    
    ds = ds.assign(concentration=(("sample", "peak"), conc_arr),
                   concentration_error=(("sample", "peak"), error_arr))
    ds["concentration"].attrs = {
         "long_name": "Computed Concentration (mM)",
         "description": ("For non-calibration samples, concentration is computed from the measured peak area "
                         "using the calibration regression model (concentration = (Area - intercept)/slope) based on "
                         "the peak's identity from 'peak_identity'. For calibration standard samples, the pre-assigned "
                         "concentrations are preserved.")
    }
    ds["concentration_error"].attrs = {
         "long_name": "Computed Concentration Uncertainty (mM)",
         "description": ("The concentration error is estimated by combining the integration uncertainty from "
                         "'peak_area_error' and the calibration regression error. Specifically, if:\n"
                         "  conc = (Area - intercept) / slope,\n"
                         "  error_area = peak_area_error / |slope|,\n"
                         "  error_calib = |conc| * (1 - R²),\n"
                         "then:\n"
                         "  concentration_error = sqrt(error_area² + error_calib²).\n"
                         "For calibration standard samples, the error is set to 0.")
    }
    return ds