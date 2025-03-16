import xarray as xr
import pandas as pd
from . import detection
from . import calibration
from . import assignment
from . import quality_control  # renamed module for QC filtering
from . import visualization
import numpy as np
import scipy.integrate as integrate
from matplotlib import pyplot as plt
from scipy import stats

class IonChromatographyAnalyzer:
    def __init__(self, dataset: xr.Dataset, calibration_file: str = None):
        """
        Initialize the analyzer.
        
        Parameters:
            dataset (xr.Dataset): xarray dataset containing 'conductance' (time x sample) and 'time' (time).
            calibration_file (str): Path to the calibration CSV file.
        """
        self.dataset = dataset
        self.calibration_file = calibration_file
        self.calib_df = None
        if calibration_file:
            self.calib_df = calibration.load_calibration_csv(calibration_file)
        self.peak_results = {}  # dict: sample name -> DataFrame of detected peaks
        # Determine sample names from dataset coordinates, if available
        if 'sample' in dataset.coords:
            self.sample_names = list(dataset.coords['sample'].values)
        else:
            self.sample_names = list(range(dataset.dims['sample']))
        self.calib_avg_retention = {}  # to be computed from calibration samples

    def detect_peaks(self, calibration_samples=None, calib_criteria=None, unknown_criteria=None):
        """
        Detect peaks in the conductance-time series for each sample,
        applying different detection criteria for calibration samples vs. non-calibration samples.
        
        Parameters:
            calibration_samples (list, optional):
                List of sample names (or indices) that are calibration samples.
                These samples will use the 'calib_criteria' parameters.
            calib_criteria (dict, optional):
                Dictionary of keyword arguments for peak detection for calibration samples.
                Example keys: min_height, prominence, distance, max_width, min_sharpness.
                Defaults to: 
                    {'min_height': 0.1, 'prominence': 0.05, 'distance': 5, 'max_width': 10, 'min_sharpness': 0.05}
            unknown_criteria (dict, optional):
                Dictionary of keyword arguments for peak detection for non-calibration samples.
                Defaults to: 
                    {'min_height': 0.1, 'prominence': 0.05, 'distance': 5, 'max_width': 15, 'min_sharpness': 0.03}
        
        The function loops over all samples in the dataset and uses the appropriate set of criteria for each sample.
        """
        time = self.dataset['time'].values
        conductance = self.dataset['conductance'].values  # 2D array: time x sample
        
        # Set default criteria if not provided
        if calib_criteria is None:
            calib_criteria = dict(min_height=0.1, prominence=0.05, distance=5, max_width=10, min_sharpness=0.05)
        if unknown_criteria is None:
            unknown_criteria = dict(min_height=0.1, prominence=0.05, distance=5, max_width=15, min_sharpness=0.03)
        
        # If calibration_samples is None, assume an empty list (all samples are non-calibration)
        if calibration_samples is None:
            calibration_samples = []
        
        for i, sample in enumerate(self.sample_names):
            signal = conductance[:, i]
            # Choose criteria based on whether the sample is a calibration sample
            if sample in calibration_samples:
                criteria = calib_criteria
            else:
                criteria = unknown_criteria
            
            peaks_df = detection.detect_peaks_in_signal(
                time, signal,
                **criteria
            )
            self.peak_results[sample] = peaks_df


    import numpy as np
    import scipy.integrate as integrate

    def calculate_peak_area(self):
        """
        For each sample, calculate the peak area by integrating the signal between the
        left and right boundaries (provided by detect_peaks_in_signal).
        The area is computed using the trapezoidal rule via scipy.integrate.trapezoid 
        and added as new columns 'peak_area' and 'peak_area_error' in the corresponding
        peak results DataFrame.
        
        The error is estimated based on the baseline noise. For each peak, we define a 
        baseline window (10% of the integration window) immediately before and after the peak.
        The noise (σ_noise) is estimated as the average standard deviation in these windows.
        Then, the error in the area is approximated as:
        
            σ_area = σ_noise * sqrt(N) * Δt,
        
        where N is the number of points in the integration window and Δt is the median time step.
        """
        time = self.dataset['time'].values
        conductance = self.dataset['conductance'].values  # 2D array: time x sample
        dt = np.median(np.diff(time))
        
        for sample in self.sample_names:
            peaks_df = self.peak_results.get(sample)
            if peaks_df is None or peaks_df.empty:
                continue
            sample_index = self.sample_names.index(sample)
            signal = conductance[:, sample_index]
            areas = []
            area_errors = []
            for idx, row in peaks_df.iterrows():
                lt = row['left_time']
                rt = row['right_time']
                # Create a mask for the integration window
                mask = (time >= lt) & (time <= rt)
                N_points = np.sum(mask)
                if N_points < 2:
                    area = 0
                    area_err = 0
                else:
                    area = integrate.trapezoid(signal[mask], time[mask])
                    
                    # Define baseline windows: 10% of the integration window on either side.
                    window_size = 0.1 * (rt - lt)
                    # Left baseline window: from max(time[0], lt - window_size) to lt.
                    mask_left = (time >= max(time[0], lt - window_size)) & (time < lt)
                    # Right baseline window: from rt to min(time[-1], rt + window_size)
                    mask_right = (time > rt) & (time <= min(time[-1], rt + window_size))
                    if np.sum(mask_left) > 1 and np.sum(mask_right) > 1:
                        sigma_left = np.std(signal[mask_left])
                        sigma_right = np.std(signal[mask_right])
                        sigma_noise = (sigma_left + sigma_right) / 2.0
                    else:
                        # Fallback: use the std of the integration window (may be overestimated if peak dominates)
                        sigma_noise = np.std(signal[mask])
                    
                    # Estimate error in area: σ_area = σ_noise * sqrt(N) * dt
                    area_err = sigma_noise * np.sqrt(N_points) * dt
                areas.append(area)
                area_errors.append(area_err)
            peaks_df['peak_area'] = areas
            peaks_df['peak_area_error'] = area_errors
            self.peak_results[sample] = peaks_df



    def calibrate(self, calibration_samples=['A','B','C','D','E','F','G']):
        """
        Use calibration samples to assign analyte identities and compute average retention times.
        
        Parameters:
            calibration_samples (list): List of sample names (or indices) that are calibration standards.
        """
        if self.calib_df is None:
            raise ValueError("Calibration CSV file not provided.")
        calib_peak_results = {}
        for sample in calibration_samples:
            if sample not in self.peak_results:
                raise ValueError(f"Calibration sample {sample} not found in dataset.")
            peaks_df = self.peak_results[sample]
            peaks_df = calibration.assign_calibration_peaks(peaks_df, self.calib_df)
            calib_peak_results[sample] = peaks_df
            self.peak_results[sample] = peaks_df  # update with assigned analyte
        self.calib_avg_retention = calibration.average_calibration_retention_times(calib_peak_results)

    def assign_peaks(self, ret_time_tolerance=30):
        """
        Assign peaks in non-calibration samples based on calibrated average retention times.
        
        Parameters:
            ret_time_tolerance (float): Maximum allowed difference (in seconds) for matching.
        """
        for sample in self.sample_names:
            # Skip calibration samples (if the sample name is in calibration CSV)
            if self.calib_df is not None and sample in self.calib_df['Analyte'].values:
                continue
            peaks_df = self.peak_results.get(sample)
            if peaks_df is not None and not peaks_df.empty:
                peaks_df = assignment.assign_unknown_peaks(peaks_df, self.calib_avg_retention,
                                                           tolerance=ret_time_tolerance)
                self.peak_results[sample] = peaks_df

    def filter_peaks(self, max_asymmetry=2.0):
        """
        Apply quality control filtering on detected peaks.
        
        Parameters:
            max_asymmetry (float): Maximum allowed asymmetry factor for a valid peak.
        """
        for sample in self.sample_names:
            peaks_df = self.peak_results.get(sample)
            if peaks_df is not None and not peaks_df.empty:
                peaks_df = quality_control.quality_filter(peaks_df, max_asymmetry=max_asymmetry)
                self.peak_results[sample] = peaks_df


    def get_results_xarray(self):
        """
        Compile the detected and assigned peaks into a single xarray Dataset,
        with one entry per sample and a fixed number of peaks (padded with NaNs).
        Each sample will have a "peak" dimension with missing values padded with NaNs.
        
        Returns:
            ds (xr.Dataset): Dataset with dimensions "sample" and "peak", with variables:
                peak_time, peak_height, prominence, width, left_time, right_time,
                sharpness, peak_area, peak_area_error, assigned_analyte, QC_flag.
        """
        import xarray as xr
        import numpy as np

        samples = []
        # Prepare lists of lists for each variable
        peak_time_list = []
        peak_height_list = []
        prominence_list = []
        width_list = []
        left_time_list = []
        right_time_list = []
        sharpness_list = []
        peak_area_list = []
        peak_area_error_list = []  # new list for error in peak area
        assigned_analyte_list = []
        QC_flag_list = []

        for sample in self.sample_names:
            samples.append(sample)
            df = self.peak_results.get(sample)
            if df is not None and not df.empty:
                peak_time_list.append(df['peak_time'].tolist())
                peak_height_list.append(df['peak_height'].tolist())
                prominence_list.append(df['prominence'].tolist())
                width_list.append(df['width'].tolist())
                left_time_list.append(df['left_time'].tolist())
                right_time_list.append(df['right_time'].tolist())
                sharpness_list.append(df['sharpness'].tolist() if 'sharpness' in df.columns else [])
                peak_area_list.append(df['peak_area'].tolist() if 'peak_area' in df.columns else [])
                peak_area_error_list.append(df['peak_area_error'].tolist() if 'peak_area_error' in df.columns else [])
                assigned_analyte_list.append(df['assigned_analyte'].tolist() if 'assigned_analyte' in df.columns else [])
                QC_flag_list.append(df['QC_flag'].tolist() if 'QC_flag' in df.columns else [])
            else:
                # No peaks detected for this sample.
                peak_time_list.append([])
                peak_height_list.append([])
                prominence_list.append([])
                width_list.append([])
                left_time_list.append([])
                right_time_list.append([])
                sharpness_list.append([])
                peak_area_list.append([])
                peak_area_error_list.append([])
                assigned_analyte_list.append([])
                QC_flag_list.append([])

        # Determine the maximum number of peaks across all samples.
        max_peaks = max((len(lst) for lst in peak_time_list), default=0)

        # Helper function: pad each list to the fixed length with np.nan.
        def pad_list(lst, length, fill_value=np.nan):
            return lst + [fill_value] * (length - len(lst))

        # Pad each sample's list to have the same length.
        peak_time_arr = np.array([pad_list(lst, max_peaks, np.nan) for lst in peak_time_list])
        peak_height_arr = np.array([pad_list(lst, max_peaks, np.nan) for lst in peak_height_list])
        prominence_arr = np.array([pad_list(lst, max_peaks, np.nan) for lst in prominence_list])
        width_arr = np.array([pad_list(lst, max_peaks, np.nan) for lst in width_list])
        left_time_arr = np.array([pad_list(lst, max_peaks, np.nan) for lst in left_time_list])
        right_time_arr = np.array([pad_list(lst, max_peaks, np.nan) for lst in right_time_list])
        sharpness_arr = np.array([pad_list(lst, max_peaks, np.nan) for lst in sharpness_list])
        peak_area_arr = np.array([pad_list(lst, max_peaks, np.nan) for lst in peak_area_list])
        peak_area_error_arr = np.array([pad_list(lst, max_peaks, np.nan) for lst in peak_area_error_list])
        assigned_analyte_arr = np.array([pad_list(lst, max_peaks, np.nan) for lst in assigned_analyte_list])
        QC_flag_arr = np.array([pad_list(lst, max_peaks, np.nan) for lst in QC_flag_list])

        # Create the xarray Dataset with dimensions "sample" and "peak"
        ds = xr.Dataset({
            "peak_time": (("sample", "peak"), peak_time_arr),
            "peak_height": (("sample", "peak"), peak_height_arr),
            "prominence": (("sample", "peak"), prominence_arr),
            "width": (("sample", "peak"), width_arr),
            "left_time": (("sample", "peak"), left_time_arr),
            "right_time": (("sample", "peak"), right_time_arr),
            "sharpness": (("sample", "peak"), sharpness_arr),
            "peak_area": (("sample", "peak"), peak_area_arr),
            "peak_area_error": (("sample", "peak"), peak_area_error_arr),
            "assigned_analyte": (("sample", "peak"), assigned_analyte_arr),
            "QC_flag": (("sample", "peak"), QC_flag_arr)
        }, coords={"sample": samples, "peak": np.arange(max_peaks)})

        return ds


    def plot_chromatogram(self, sample):
        """
        Plot the chromatogram and detected peaks for a given sample,
        including a visual representation of the peak area integration.
        
        Parameters:
            sample: Sample name or index.
        """
        if sample not in self.sample_names:
            raise ValueError(f"Sample {sample} not found in dataset.")
        time = self.dataset['time'].values
        sample_index = self.sample_names.index(sample)
        signal = self.dataset['conductance'].values[:, sample_index]
        peaks_df = self.peak_results.get(sample)
        visualization.plot_chromatogram(time, signal, peaks_df, sample_name=sample)


    def calculate_calibration_curve(self, calib_conc_csv, dilution_factors, calibration_samples=None):
        """
        Calculate a calibration curve for each analyte using calibration standards.
        
        The CSV file 'calib_conc_csv' should have the following columns:
            Name,Approx. Concentration,Molar Mass (g/mol),Concentration (mM)
        The 'Concentration (mM)' column is used as the true standard concentration.
        
        The dilution_factors parameter is a list of dilution factors for each calibration sample.
        The effective concentration for each calibration sample is computed as:
            effective_concentration = true_concentration * dilution_factor
        
        For each analyte, this method gathers the measured peak area from each calibration sample
        (using the 'assigned_analyte' from the peak detection) and performs a linear regression 
        of peak area versus effective concentration.
        
        It returns a dictionary of calibration fit results for each analyte (including slope, intercept, 
        r_value, p_value, and standard error) and also plots all the fits on a single figure.
        
        Parameters:
            calib_conc_csv (str): Path to the CSV file with calibration concentrations.
            dilution_factors (list): List of dilution factors for each calibration sample.
            calibration_samples (list, optional): List of calibration sample identifiers.
                If None, defaults to ['A', 'B', 'C', 'D', 'E', 'F', 'G'].
        
        Returns:
            fit_results (dict): A dictionary with analyte names as keys and a dictionary of fit parameters as values.
        """
        # Load the calibration concentration data
        calib_df = pd.read_csv(calib_conc_csv)
        
        if calibration_samples is None:
            calibration_samples = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        
        # Create a dictionary to store effective concentrations and measured areas for each analyte.
        calibration_data = {}
        for analyte in calib_df['Name']:
            calibration_data[analyte] = {'conc': [], 'area': []}
        
        # Loop over each calibration sample (order of dilution_factors must match calibration_samples)
        for i, sample in enumerate(calibration_samples):
            if sample not in self.peak_results:
                continue
            sample_df = self.peak_results[sample]
            dilution = dilution_factors[i]
            for analyte in calibration_data.keys():
                # Get the peak(s) assigned to the analyte
                subset = sample_df[sample_df['assigned_analyte'] == analyte]
                if not subset.empty:
                    measured_area = subset['peak_area'].mean()
                    # Get the true concentration (in mM) from the CSV
                    true_conc = calib_df.loc[calib_df['Name'] == analyte, 'Concentration (mM)'].values[0]
                    # Compute the effective concentration in the calibration sample.
                    effective_conc = true_conc * dilution
                    calibration_data[analyte]['conc'].append(effective_conc)
                    calibration_data[analyte]['area'].append(measured_area)
        
        # For each analyte, perform a linear fit of peak area vs. effective concentration.
        fit_results = {}
        plt.figure(figsize=(10, 8))
        colors = plt.cm.tab10.colors  # use a colormap with at least 10 colors
        for idx, analyte in enumerate(calibration_data.keys()):
            x = calibration_data[analyte]['conc']
            y = calibration_data[analyte]['area']
            if len(x) < 2:
                continue  # skip analytes with insufficient data
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            fit_results[analyte] = {
                'slope': slope,
                'intercept': intercept,
                'r_value': r_value,
                'p_value': p_value,
                'std_err': std_err
            }
            # Plot the data points
            plt.scatter(x, y, color=colors[idx % len(colors)], label=f"{analyte} Data")
            # Plot the fit line
            x_fit = np.linspace(min(x), max(x), 100)
            y_fit = slope * x_fit + intercept
            plt.plot(x_fit, y_fit, color=colors[idx % len(colors)], linestyle='--', label=f"{analyte} Fit")
        
        plt.xlabel("Effective Concentration (mM)")
        plt.ylabel("Peak Area")
        plt.title("Calibration Curves for Analytes")
        plt.legend()
        plt.show()
        
        return fit_results


    def calculate_concentrations(self, ds, calibration_fit):
        """
        Calculate concentrations for each detected peak in the xarray Dataset based on the calibration curves.
        
        For each peak with a valid assigned analyte, the concentration (in mM) is calculated using:
        
            concentration = (peak_area - intercept) / slope
            
        and the uncertainty is propagated from both the peak fitting (peak_area_error) and the calibration fit (std_err):
        
            concentration_error = sqrt((peak_area_error / slope)^2 + ((peak_area - intercept) * std_err / slope^2)^2)
        
        Parameters:
            ds (xr.Dataset): The xarray Dataset with dimensions ("sample", "peak") containing:
                - 'peak_area': the measured peak area,
                - 'assigned_analyte': the assigned analyte identity for each peak,
                - optionally, 'peak_area_error': the error from peak fitting.
            calibration_fit (dict): A dictionary with calibration fit parameters for each analyte, e.g.:
                {
                'Fluoride': {'slope': ..., 'intercept': ..., 'std_err': ...},
                ...
                }
                
        Returns:
            ds_out (xr.Dataset): A new xarray Dataset with additional variables:
                - 'concentration': calculated concentration for each peak (mM),
                - 'concentration_error': propagated uncertainty for each concentration.
        """
        import numpy as np
        import xarray as xr

        # Get measured peak area and assigned analyte arrays.
        peak_area = ds['peak_area'].values    # shape (sample, peak)
        assigned = ds['assigned_analyte'].values  # shape (sample, peak)
        
        # Get peak_area_error if available; otherwise, assume zeros.
        if 'peak_area_error' in ds:
            peak_area_error = ds['peak_area_error'].values
        else:
            peak_area_error = np.zeros_like(peak_area)
        
        # Initialize output arrays.
        conc = np.full(peak_area.shape, np.nan, dtype=float)
        conc_err = np.full(peak_area.shape, np.nan, dtype=float)
        
        n_samples, n_peaks = peak_area.shape
        for i in range(n_samples):
            for j in range(n_peaks):
                analyte = assigned[i, j]
                # Only process if a valid analyte is assigned and it exists in the calibration fit dictionary.
                if isinstance(analyte, str) and analyte in calibration_fit:
                    fit = calibration_fit[analyte]
                    slope = fit['slope']
                    intercept = fit['intercept']
                    std_err = fit['std_err']  # uncertainty in the slope from the calibration fit
                    A = peak_area[i, j]
                    sigma_A = peak_area_error[i, j]
                    # Calculate concentration.
                    conc_val = (A - intercept) / slope
                    conc[i, j] = conc_val
                    # Propagate errors from both A and the slope.
                    conc_err[i, j] = np.sqrt((sigma_A / slope)**2 + (((A - intercept) * std_err) / (slope**2))**2)
        
        # Create a new dataset with the concentration data.
        ds_out = ds.copy()
        ds_out['concentration'] = (('sample', 'peak'), conc)
        ds_out['concentration_error'] = (('sample', 'peak'), conc_err)
        
        return ds_out
