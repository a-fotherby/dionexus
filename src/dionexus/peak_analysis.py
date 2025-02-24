import numpy as np
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import uniform_filter1d
from scipy.special import erfc
from lmfit import Model


def exgauss(x, amplitude, center, sigma, tau):
    """
    Exponentially modified Gaussian (ExGaussian) function.
    
    Parameters:
      amplitude : float
          Peak amplitude.
      center : float
          Center of the Gaussian component.
      sigma : float
          Standard deviation of the Gaussian.
      tau : float
          Exponential decay parameter; controls asymmetry.
          
    Returns:
      np.array : The ExGaussian function evaluated at x.
    """
    arg = (sigma/tau - (x - center)/sigma) / np.sqrt(2)
    return (amplitude/(2*tau)) * np.exp((sigma**2/(2*tau**2)) - (x - center)/tau) * erfc(arg)

def fit_peaks_exgauss_adaptive(ds, prominence=0.001, max_peaks=None,
                               factor_pre=1.0, factor_post=2.0,
                               smooth_window=5):
    """
    For each sample in the dataset, detect peaks and use an adaptive window (based on the 
    measured peak widths) to fit an exponentially modified Gaussian (exgauss) model.
    
    Then, for each pair of consecutive peaks that have overlapping windows, adjust the 
    earlier peak’s end time by detecting a valley (using a smoothed signal) in the overlapping 
    region. Finally, all peak properties (retention time, area, asymmetry) are recalculated 
    from the raw data over the final adjusted window, ensuring that the retention time lies 
    within the [start, end] boundaries and that all properties reflect the non-overlapping window.
    
    New variables added to the dataset:
      - peak_count
      - peak_retention_time
      - peak_area
      - peak_area_error     <-- new variable: integration error
      - peak_asymmetry (calculated from raw data as (end - retention)/(retention - start))
      - peak_start_time
      - peak_end_time
    
    Parameters:
      ds (xr.Dataset): Input dataset with coordinates "time" (s) and "sample", and a variable "conductance".
      prominence (float): Parameter for initial peak detection.
      max_peaks (int, optional): Maximum number of peaks to fit per sample.
      factor_pre (float): Factor to multiply the left half-width to determine window start.
      factor_post (float): Factor to multiply the right half-width to determine window end.
      smooth_window (int): Window size for smoothing the signal (using a moving average) during valley detection.
    
    Returns:
      xr.Dataset: The dataset augmented with new peak attributes.
    """
    # Helper: smooth the signal using a moving average.
    def smooth_signal(signal, window_size):
        if window_size < 2:
            return signal
        return uniform_filter1d(signal, size=window_size)
    
    # Helper: find a valley (local minimum) in the overlapping region, using the smoothed signal.
    def find_valley(time_arr, data_arr, start_time, end_time, smooth_size):
        mask = (time_arr >= start_time) & (time_arr <= end_time)
        t_region = time_arr[mask]
        s_region = data_arr[mask]
        if len(t_region) < 3:
            return start_time
        s_smooth = smooth_signal(s_region, smooth_size)
        valley_idx = np.argmin(s_smooth)
        return t_region[valley_idx]
    
    # Helper: compute the numerical integral of the model function over x.
    def model_integral(x, amp, center, sigma, tau):
        return np.trapz(exgauss(x, amp, center, sigma, tau), x)
    
    # Helper: numerical partial derivative using central differences.
    def partial_derivative(x, amp, center, sigma, tau, param, delta):
        # param should be one of 'amp', 'center', 'sigma', 'tau'
        if param == 'amp':
            I_plus = model_integral(x, amp + delta, center, sigma, tau)
            I_minus = model_integral(x, amp - delta, center, sigma, tau)
        elif param == 'center':
            I_plus = model_integral(x, amp, center + delta, sigma, tau)
            I_minus = model_integral(x, amp, center - delta, sigma, tau)
        elif param == 'sigma':
            I_plus = model_integral(x, amp, center, sigma + delta, tau)
            I_minus = model_integral(x, amp, center, sigma - delta, tau)
        elif param == 'tau':
            I_plus = model_integral(x, amp, center, sigma, tau + delta)
            I_minus = model_integral(x, amp, center, sigma, tau - delta)
        else:
            I_plus = I_minus = 0
        return (I_plus - I_minus) / (2 * delta)
    
    # Get common time vector and sample IDs.
    time = ds['time'].values
    samples = ds['sample'].values
    n_samples = len(samples)
    
    sample_info = []
    overall_max_peaks = 0
    
    # Create LMFit model.
    exgauss_model = Model(exgauss, independent_vars=['x'])
    
    for sample in samples:
        data = ds['conductance'].sel(sample=sample).squeeze().values
        
        # Preliminary peak detection.
        peaks, properties = find_peaks(data, prominence=prominence)
        if max_peaks is not None and len(peaks) > max_peaks:
            sort_idx = np.argsort(properties['prominences'])[-max_peaks:]
            peaks = peaks[sort_idx]
            properties = {k: properties[k][sort_idx] for k in properties}
            sort_time_idx = np.argsort(time[peaks])
            peaks = peaks[sort_time_idx]
            properties = {k: properties[k][sort_time_idx] for k in properties}
        
        # Lists to store fitted properties.
        fitted_centers = []
        fitted_areas = []
        fitted_asymmetries = []
        fitted_starts = []
        fitted_ends = []
        # New: store LM fit parameters and their uncertainties.
        fitted_params = []  # each entry is a dict with keys: amp, amp_err, center, center_err, sigma, sigma_err, tau, tau_err
        
        # For each detected peak, perform adaptive fitting.
        for i, peak in enumerate(peaks):
            peak_center = time[peak]
            width_results = peak_widths(data, [peak], rel_height=0.5)
            left_bound = time[int(np.floor(width_results[2][0]))]
            right_bound = time[int(np.ceil(width_results[3][0]))]
            left_half_width = peak_center - left_bound
            right_half_width = right_bound - peak_center  # fixed calculation
            window_start = peak_center - factor_pre * left_half_width
            window_end = peak_center + factor_post * right_half_width
            window_start = max(window_start, time[0])
            window_end = min(window_end, time[-1])
            
            mask = (time >= window_start) & (time <= window_end)
            xdata = time[mask]
            ydata = data[mask]
            if len(xdata) < 5:
                continue
            
            amplitude_guess = ydata.max()
            sigma_guess = left_half_width if left_half_width > 0 else 1.0
            tau_guess = right_half_width if right_half_width > 0 else 1.0
            params = exgauss_model.make_params(amplitude=amplitude_guess,
                                               center=peak_center,
                                               sigma=sigma_guess,
                                               tau=tau_guess)
            try:
                result = exgauss_model.fit(ydata, params, x=xdata)
            except Exception:
                continue
            
            # Use the raw maximum in the window as the initial retention time.
            raw_max_time = xdata[np.argmax(ydata)]
            sigma_fit = result.params['sigma'].value
            tau_fit = result.params['tau'].value
            amplitude_fit = result.params['amplitude'].value
            
            # Retrieve uncertainties (if not available, use a small value).
            amp_err = result.params['amplitude'].stderr if result.params['amplitude'].stderr is not None else 1e-6
            center_err = result.params['center'].stderr if result.params['center'].stderr is not None else 1e-6
            sigma_err = result.params['sigma'].stderr if result.params['sigma'].stderr is not None else 1e-6
            tau_err = result.params['tau'].stderr if result.params['tau'].stderr is not None else 1e-6
            
            # Store the initial properties.
            fitted_centers.append(raw_max_time)
            fitted_areas.append(np.trapz(ydata, x=xdata))
            fitted_asymmetries.append(tau_fit / sigma_fit if sigma_fit != 0 else np.nan)
            fitted_starts.append(xdata[0])
            fitted_ends.append(xdata[-1])
            fitted_params.append({
                'amp': amplitude_fit,
                'amp_err': amp_err,
                'center': result.params['center'].value,
                'center_err': center_err,
                'sigma': sigma_fit,
                'sigma_err': sigma_err,
                'tau': tau_fit,
                'tau_err': tau_err
            })
        
        # Adjust overlapping windows.
        for j in range(len(fitted_centers) - 1):
            if fitted_ends[j] > fitted_starts[j+1]:
                overlap_start = fitted_starts[j+1]
                overlap_end = fitted_ends[j]
                new_end = find_valley(time, data, overlap_start, overlap_end, smooth_size=smooth_window)
                # Ensure new_end is strictly less than next peak's start.
                new_end = min(new_end, fitted_starts[j+1] - 1e-6)
                fitted_ends[j] = new_end
        
        # Estimate background noise for this sample:
        # Build a mask that is True where any peak window is active.
        peak_mask = np.zeros_like(time, dtype=bool)
        for win_start, win_end in zip(fitted_starts, fitted_ends):
            peak_mask |= ((time >= win_start) & (time <= win_end))
        background_mask = ~peak_mask
        if np.any(background_mask):
            noise_est = np.std(data[background_mask])
        else:
            noise_est = np.std(data)
        # Assume uniform sampling (or use median diff).
        dx = np.median(np.diff(time))
        
        # Now, recalculate all peak properties using the final (adjusted) window for each peak,
        # and propagate LM fit errors to estimate the integration error.
        recalculated_centers = []
        recalculated_areas = []
        recalculated_asym = []
        recalculated_starts = []
        recalculated_ends = []
        recalculated_area_errors = []
        
        for j in range(len(fitted_centers)):
            win_start = fitted_starts[j]
            win_end = fitted_ends[j]
            mask = (time >= win_start) & (time <= win_end)
            if np.sum(mask) < 2:
                continue
            x_win = time[mask]
            y_win = data[mask]
            # Recalculate retention time as raw maximum in the adjusted window.
            new_center = x_win[np.argmax(y_win)]
            new_area = np.trapz(y_win, x=x_win)
            # Recalculate asymmetry as (win_end - new_center) / (new_center - win_start)
            new_asym = (win_end - new_center) / (new_center - win_start) if (new_center - win_start) != 0 else np.nan
            
            # Retrieve LM fit parameters for this peak.
            params_dict = fitted_params[j]
            amp = params_dict['amp']
            center_fit = params_dict['center']
            sigma = params_dict['sigma']
            tau = params_dict['tau']
            amp_err = params_dict['amp_err']
            center_err = params_dict['center_err']
            sigma_err = params_dict['sigma_err']
            tau_err = params_dict['tau_err']
            
            # Compute the model integral over the final window using the LM fit parameters.
            I0 = model_integral(x_win, amp, center_fit, sigma, tau)
            # Use the LM fit parameter uncertainties as finite differences.
            dI_damp = partial_derivative(x_win, amp, center_fit, sigma, tau, 'amp', amp_err)
            dI_dcenter = partial_derivative(x_win, amp, center_fit, sigma, tau, 'center', center_err)
            dI_dsigma = partial_derivative(x_win, amp, center_fit, sigma, tau, 'sigma', sigma_err)
            dI_dtau = partial_derivative(x_win, amp, center_fit, sigma, tau, 'tau', tau_err)
            
            # Propagate LM fit uncertainties.
            error_model = np.sqrt((dI_damp * amp_err)**2 +
                                  (dI_dcenter * center_err)**2 +
                                  (dI_dsigma * sigma_err)**2 +
                                  (dI_dtau * tau_err)**2)
            
            # Propagate background noise error over the integration window.
            error_bg = noise_est * np.sqrt(len(x_win)) * dx
            
            # Total error (combine in quadrature).
            total_error = np.sqrt(error_model**2 + error_bg**2)
            
            recalculated_centers.append(new_center)
            recalculated_areas.append(new_area)
            recalculated_asym.append(new_asym)
            recalculated_starts.append(win_start)
            recalculated_ends.append(win_end)
            recalculated_area_errors.append(total_error)
        
        n_peaks_fitted = len(recalculated_centers)
        if n_peaks_fitted > overall_max_peaks:
            overall_max_peaks = n_peaks_fitted
        
        sample_info.append({
            'n_peaks': n_peaks_fitted,
            'centers': np.array(recalculated_centers),
            'areas': np.array(recalculated_areas),
            'area_errors': np.array(recalculated_area_errors),
            'asymmetries': np.array(recalculated_asym),
            'starts': np.array(recalculated_starts),
            'ends': np.array(recalculated_ends)
        })
    
    # Assemble the results into uniform arrays (samples x peaks).
    centers_array = np.full((n_samples, overall_max_peaks), np.nan)
    areas_array = np.full((n_samples, overall_max_peaks), np.nan)
    area_errors_array = np.full((n_samples, overall_max_peaks), np.nan)
    asym_array = np.full((n_samples, overall_max_peaks), np.nan)
    starts_array = np.full((n_samples, overall_max_peaks), np.nan)
    ends_array = np.full((n_samples, overall_max_peaks), np.nan)
    
    for i, info in enumerate(sample_info):
        n = info['n_peaks']
        if n > 0:
            centers_array[i, :n] = info['centers']
            areas_array[i, :n] = info['areas']
            area_errors_array[i, :n] = info['area_errors']
            asym_array[i, :n] = info['asymmetries']
            starts_array[i, :n] = info['starts']
            ends_array[i, :n] = info['ends']
    
    ds = ds.assign(
        peak_count=("sample", np.array([info['n_peaks'] for info in sample_info])),
        peak_retention_time=(("sample", "peak"), centers_array),
        peak_area=(("sample", "peak"), areas_array),
        peak_area_error=(("sample", "peak"), area_errors_array),
        peak_asymmetry=(("sample", "peak"), asym_array),
        peak_start_time=(("sample", "peak"), starts_array),
        peak_end_time=(("sample", "peak"), ends_array)
    )
    
    ds['peak_count'].attrs = {"long_name": "Number of Fitted Peaks"}
    ds['peak_retention_time'].attrs = {"units": "s", "long_name": "Fitted Peak Retention Time (raw maximum)"}
    ds['peak_area'].attrs = {"units": "µS*s", "long_name": "Fitted Peak Area (model-based, raw integration)"}
    ds['peak_area_error'].attrs = {
        "units": "µS*s",
        "long_name": "Estimated Error in Fitted Peak Area",
        "description": "Error propagated from LM fit parameter uncertainties and background noise over the final integration window"
    }
    ds['peak_asymmetry'].attrs = {
        "long_name": "Fitted Peak Asymmetry",
        "description": "Calculated as (peak_end_time - retention_time) / (retention_time - peak_start_time)"
    }
    ds['peak_start_time'].attrs = {"units": "s", "long_name": "Fitted Peak Start Time"}
    ds['peak_end_time'].attrs = {"units": "s", "long_name": "Fitted Peak End Time"}
    
    return ds


def flag_peaks_by_asymmetry(ds, threshold=2.0):
    """
    Flags peaks in each sample of the dataset whose calculated asymmetry exceeds a given threshold.
    
    The dataset is assumed to contain a variable 'peak_asymmetry' with dimensions ("sample", "peak"),
    where each value represents the asymmetry (e.g., tau/sigma) for that peak. For each sample,
    any peak with an asymmetry greater than the threshold is flagged.
    
    Parameters:
      ds (xr.Dataset): The xarray dataset containing the 'peak_asymmetry' variable and a coordinate 'sample'.
      threshold (float): The asymmetry threshold above which a peak is considered flagged. Default is 2.0.
    
    Returns:
      dict: A dictionary mapping sample identifiers to a list of peak indices (0-indexed) that have asymmetry > threshold.
            (For display, these indices could be shifted to 1-indexing if preferred.)
    """
    flagged_peaks = {}
    # Get the asymmetry values and sample names.
    asym_array = ds['peak_asymmetry'].values  # shape (n_samples, n_peaks)
    samples = ds['sample'].values
    
    for i, sample in enumerate(samples):
        # For each sample, find peak indices where asymmetry > threshold.
        sample_asym = asym_array[i, :]
        peak_indices = np.where(sample_asym > threshold)[0]
        if peak_indices.size > 0:
            flagged_peaks[sample] = peak_indices.tolist()
            print(f"Sample {sample} has peaks {peak_indices.tolist()} with asymmetry > {threshold}.")
    
    return flagged_peaks


def assign_analyte_names(ds, virtual_df, calib_standard_samples, tolerance=10.0):
    """
    Assigns analyte names to each detected peak based on calibration standards and a virtual 
    calibration DataFrame.
    
    The virtual_df should have at least the following columns:
      - 'Analyte': A string label indicating the analyte. The row order defines the expected elution order.
      - 'Retention_Time': The theoretical retention time (in seconds) for that analyte (used only to define order).
    
    The function uses the calibration standard samples (provided as a list of sample names from ds['sample'])
    to extract detected peak retention times. For each calibration sample, the detected peaks are sorted in increasing 
    order. The median retention time for each peak index (from 1 to the minimum number of detected peaks among
    the calibration samples) is computed. These median values define the expected retention times for the 
    elution order.
    
    Then, for every sample in ds, each detected peak (from ds['peak_retention_time']) is compared to these expected 
    median retention times. If the absolute difference is within the specified tolerance (default 10 s), the corresponding 
    analyte name (from the virtual_df 'Analyte' column) is assigned; otherwise, the peak is left unassigned (empty string).
    
    Parameters:
      ds (xr.Dataset): Dataset containing a variable 'peak_retention_time' (dims: sample x peak).
      virtual_df (pd.DataFrame): Calibration DataFrame with columns 'Analyte' and 'Retention_Time'.
      calib_standard_samples (list or tuple): List of sample names (from ds['sample']) to be used as calibration standards.
      tolerance (float): Tolerance in seconds for matching a detected peak to the expected retention time.
    
    Returns:
      xr.Dataset: The dataset augmented with a new variable 'analyte_assignment' (dims: sample x peak)
                  containing the assigned analyte names (or an empty string if no match is found).
    """
    # Extract calibration peaks from each calibration standard sample.
    calib_peaks_list = []
    for sample in calib_standard_samples:
        # Get the retention times for the given calibration sample.
        rt = ds['peak_retention_time'].sel(sample=sample).values
        # Remove NaN values and sort in ascending order.
        rt = np.sort(rt[~np.isnan(rt)])
        calib_peaks_list.append(rt)
    
    # Determine the minimum number of peaks detected among the calibration samples.
    min_peaks = min(len(rts) for rts in calib_peaks_list)
    if min_peaks == 0:
        raise ValueError("No calibration peaks were detected in one or more of the provided calibration standard samples.")
    
    # Compute the median retention time for each peak order (1 to min_peaks) across calibration samples.
    median_calib_rt = []
    for i in range(min_peaks):
        times = [rts[i] for rts in calib_peaks_list if len(rts) > i]
        median_calib_rt.append(np.median(times))
    median_calib_rt = np.array(median_calib_rt)
    
    if median_calib_rt.size == 0:
        raise ValueError("No calibration peak retention times could be computed from the provided standards.")
    
    # Use the analyte names from the virtual_df.
    # We assume that the virtual_df rows are in the elution order.
    expected_analytes = virtual_df['Analyte'].iloc[:min_peaks].values
    print(expected_analytes)
    
    # Now assign analyte names to each detected peak in the full dataset.
    detected_rt = ds['peak_retention_time'].values  # shape: (n_samples, n_peaks)
    n_samples, n_peaks = detected_rt.shape
    assignment = np.empty((n_samples, n_peaks), dtype=object)
    assignment[:] = ""  # initialize with empty strings
    
    for i in range(n_samples):
        for j in range(n_peaks):
            rt = detected_rt[i, j]
            if np.isnan(rt):
                assignment[i, j] = ""
            else:
                # Calculate the difference to each expected calibration retention time.
                diffs = np.abs(rt - median_calib_rt)
                if diffs.size == 0:
                    assignment[i, j] = ""
                else:
                    min_idx = np.argmin(diffs)
                    if diffs[min_idx] <= tolerance:
                        assignment[i, j] = expected_analytes[min_idx]
                    else:
                        assignment[i, j] = ""
    
    ds = ds.assign(peak_identity=(("sample", "peak"), assignment))
    ds['peak_identity'].attrs = {
        "long_name": "Assigned Analyte Name",
        "description": (f"Each detected peak's retention time is matched to the median retention times "
                        f"from the calibration standard samples (provided in calib_standard_samples) based on the "
                        f"expected elution order in virtual_df. A match is assigned if the absolute difference is "
                        f"within {tolerance} s; otherwise, the peak remains unassigned.")
    }
    return ds