import numpy as np
from scipy.stats import t
from scipy.fftpack import dct, idct
from scipy.spatial.transform import Rotation as R

def dag_rescale_bw(data_in, **kwargs):
    '''dag_rescale_bw    
    rescale data between 2 values

    data_in     data to rescale
    old_min     minimum value of data_in
    old_max     maximum value of data_in
    new_min     minimum value of rescaled data
    new_max     maximum value of rescaled data
    log         log spacing?
    '''
    data_out = np.copy(data_in)
    old_min = kwargs.get('old_min', np.nanmin(data_in))
    old_max = kwargs.get('old_max', np.nanmax(data_in))
    new_min = kwargs.get('new_min', 0)
    new_max = kwargs.get('new_min', 1)
    do_log = kwargs.get('log', False)    
    data_out[data_in<old_min] = old_min
    data_out[data_in>old_max] = old_max    
    data_out = (data_out - old_min) / (old_max - old_min) # Scaled bw 0 and 1
    data_out = data_out * (new_max-new_min) + new_min # Scale bw new values
    if do_log:
        data_out = np.log(data_out+1)
        data_out /= np.nanmax(data_out)
    return data_out

def dag_get_rsq(tc_target, tc_fit):
    '''dag_get_rsq
    Calculate the rsq (R squared)
    Of a fit time course (tc_fit), on a target (tc_target)    
    '''
    ss_res = np.sum((tc_target-tc_fit)**2, axis=-1)
    ss_tot = np.sum(
        (tc_target-tc_target.mean(axis=-1)[...,np.newaxis])**2, 
        axis=-1
        )
    rsq = 1-(ss_res/ss_tot)

    return rsq
    
def dag_get_corr(a, b):
    '''dag_get_corr
    '''
    corr = np.corrcoef(a,b)[0,1]
    return corr


def dag_detrending(ts, detrend_method='poly', detrend_param=0, normalize_method='psc', baseline_pt=None):
    """
    Detrend a time series using either DCT-based or polynomial detrending, with optional normalization.

    Parameters
    ----------
    ts : numpy.ndarray
        Input time series data with shape (n_voxels, n_time_points).
    detrend_method : str, optional
        Detrending method. Options:
            - 'dct': Remove the first `detrend_param` DCT components.
            - 'poly': Remove a polynomial trend of degree `detrend_param` (e.g., 1 for linear detrending).
        Default is 'dct'.
    detrend_param : int or False, optional
        Parameter for detrending. For 'dct', it is the number of DCT components to remove.
        For 'poly', it is the degree of the polynomial to remove.
        If 0 or False, no detrending is performed.
    normalize_method : str, optional
        Normalization method applied after detrending. Options are:
            - 'psc': Percentage Signal Change (requires an external function `dag_psc`).
            - 'zscore': Z-score normalization.
            - None: No normalization.
        Default is 'psc'.
    baseline_pt : int, list, numpy.ndarray, or None, optional
        Baseline points for PSC calculation. If an integer, baseline values from 0 to baseline_pt are used.
        If a list of two values, the range [start, stop] is used.
        If more than two values, these specific points are used.
        Only used if normalize_method is 'psc'.

    Returns
    -------
    numpy.ndarray
        Detrended (and optionally normalized) time series data with the same shape as input.
    """
    # Ensure ts is a 2D array (n_voxels, n_time_points)
    if ts.ndim == 1:
        ts = ts.reshape(1, -1)
    
    # No detrending requested
    if not detrend_param:
        detrended_ts = ts.copy()
    
    # Polynomial detrending
    elif detrend_method == 'poly':
        n_voxels, n_time = ts.shape
        x = np.arange(n_time)
        deg = detrend_param
        
        # Build the Vandermonde matrix for polynomial terms:
        # Each column corresponds to a power of x, starting with the highest degree.
        V = np.vander(x, N=deg+1, increasing=False)  # shape: (n_time, deg+1)
        
        # Compute the pseudoinverse of the Vandermonde matrix once.
        V_pinv = np.linalg.pinv(V)  # shape: (deg+1, n_time)
        
        # Compute polynomial coefficients for all voxels simultaneously.
        # ts has shape (n_voxels, n_time) and V_pinv.T has shape (n_time, deg+1)
        coeffs = ts.dot(V_pinv.T)  # shape: (n_voxels, deg+1)
        
        # Evaluate the polynomial trend for each voxel.
        # V.T has shape (deg+1, n_time), so the product yields (n_voxels, n_time)
        trend = coeffs.dot(V.T)
        
        # Subtract the trend from the original data.
        ts_mean = np.mean(ts, axis=1, keepdims=True)
        # remean...
        detrended_ts = ts - trend + ts_mean


    # DCT-based detrending
    elif detrend_method == 'dct':
        # Remove mean for proper DCT operation
        ts_mean = np.mean(ts, axis=1, keepdims=True)
        ts_centered = ts - ts_mean
        
        # Compute the DCT along the time axis (type II with orthonormal norm)
        dct_coeffs = dct(ts_centered, type=2, norm='ortho', axis=1)
        
        # Zero out the first `detrend_param` coefficients to remove low-frequency trends
        dct_coeffs[:, :detrend_param] = 0
        
        # Reconstruct the detrended time series via inverse DCT and add the mean back
        detrended_ts = idct(dct_coeffs, type=2, norm='ortho', axis=1) + ts_mean
    else:
        raise ValueError("Invalid detrend_method. Choose 'dct' or 'poly'.")
    
    # Apply normalization if requested
    if normalize_method == 'psc':
        # Note: Assumes that a function `dag_psc` is defined elsewhere.
        detrended_ts = dag_psc(detrended_ts, baseline_pt)
    elif normalize_method == 'zscore':
        detrended_ts = (detrended_ts - np.mean(detrended_ts, axis=1, keepdims=True)) / \
                        np.std(detrended_ts, axis=1, ddof=1, keepdims=True)
    elif normalize_method is None:
        pass  # No normalization
    else:
        raise ValueError("Invalid normalize_method. Choose 'psc', 'zscore', or None.")
    
    return detrended_ts

def dag_psc(ts_in, baseline_pt=None):
    """
    Calculate Percentage Signal Change (PSC) for the input time series.

    Parameters:
    - ts_in (numpy.ndarray): Input time series data, shape (n_vx, n_time_points).
    - baseline_pt (np.ndarray, int, list, or None, optional): Baseline points used for PSC calculation. If None, all points are considered as baseline.
        If 1 value: taske baseline values from 0 to baseline_pt
        If 2 value: it represents the range of points [start, stop] as baseline.
        If more   : it represents specific points as baseline.

    Returns:
    - numpy.ndarray: Time series data after Percentage Signal Change (PSC) normalization, shape (n_vx, n_time_points).
    """
    if ts_in.ndim == 1:
        ts_in = ts_in.reshape(-1, 1)

    # Define the baseline points
    if baseline_pt is None:
        baseline_pt = np.arange(ts_in.shape[-1])
    elif isinstance(baseline_pt, (int, np.integer)):
        baseline_pt = np.arange(0, baseline_pt)
    elif len(baseline_pt) == 2:
        baseline_pt = np.arange(baseline_pt[0], baseline_pt[1], dtype=int)
    else:
        baseline_pt = np.array(list(baseline_pt), dtype=int)

    # Calculate the mean of baseline points
    baseline_mean = np.mean(ts_in[:, baseline_pt], axis=1, keepdims=True)

    # Perform Percentage Signal Change (PSC) normalization
    ts_out = (ts_in - baseline_mean) / baseline_mean * 100

    # Handle NaN values resulting from division by zero
    nan_rows = np.isnan(ts_out).any(axis=1)
    ts_out[nan_rows, :] = 0

    return ts_out

def dag_paired_ttest(x, y, **kwargs):
    '''dag_paired_ttest
    sim#ple paired t-test, with option to override to correct for voxel-to-surface upsampling
    
    ow_n                    Specify the 'n' by hand
    upsampling_factor       n will be obtained from len(x)/upsampling factor 
    side                    'two-sided', 'greater', 'less'. Which sided test.

    # Test that it works (not doing adjusting)
    x= np.random.rand(100)
    y = x +  (np.random.rand(100) -.5 )     
    for side in ['greater', 'less', 'two-sided']:
        print(dag_paired_ttest(x, y,side=side))
        print(stats.ttest_rel(x, y, alternative=side))
    '''
    ow_n = kwargs.get('ow_n', None)
    upsampling_factor = kwargs.get('upsampling_factor', None)
    side = kwargs.get('side', 'two-sided') # 

    actual_n = len(x)    
    if upsampling_factor is not None:
        n = actual_n / upsampling_factor
    elif ow_n is not None:
        n = ow_n # overwrite the n that is given
    else:
        n = actual_n
    df = n-1 # Degrees of freedom = n-1 
    
    diffs = x - y
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)  # Delta degrees of freedom
    standard_error = std_diff / np.sqrt(n)
    t_statistic = mean_diff / standard_error

    p_value = dag_t_to_p(t_statistic, df, side)

    stats = {
        'n'             : n,
        'mean_diff'     : mean_diff,
        't_statistic'   : t_statistic,
        'p_value'       : p_value,
        'df'            : df,
    }
    return stats


def dag_rapid_slope(x,y):
    '''dag_rapid_slope
    Calculate the slope as quickly as possible
    '''
    x_mean = x.mean()
    y_mean = y.mean()

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    # Calculate the slope...
    slope = numerator / denominator
    return slope

def dag_rapid_slope_intercept(x,y):
    '''dag_rapid_slope_intercept
    Calculate the slope and intercept as quickly as possible
    '''
    x_mean = x.mean()
    y_mean = y.mean()

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    # Calculate the slope...
    slope = numerator / denominator
    # Calculate the intercept
    intercept = y_mean - (slope * x_mean)
    
    return slope, intercept

def dag_slope_test(x,y, **kwargs):
    '''ncsf_slope
    Calculate the slope, intercept, associated t-stat, and p-values
    
    Options:
    ow_n                Overwrite the "n" to your own
    upsampling_factor       n will be obtained from len(x)/upsampling factor 
    side                    'two-sided', 'greater', 'less'. Which sided test.

    '''
    ow_n = kwargs.get('ow_n', None)
    upsampling_factor = kwargs.get('upsampling_factor', None)
    side = kwargs.get('side', 'two-sided') # 

    actual_n = len(x)    
    if upsampling_factor is not None:
        n = actual_n / upsampling_factor
    elif ow_n is not None:
        n = ow_n # overwrite the n that is given
    else:
        n = actual_n
    df = n-2 # Degrees of freedom

    x_mean = x.mean()
    y_mean = y.mean()

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    # Calculate the slope...
    slope = numerator / denominator
    # Calculate the intercept
    intercept = y_mean - (slope * x_mean)
    # Calculate predictions
    y_pred  = slope * x + intercept
    # Now calculate the residuals 
    residuals = y - y_pred

    # Standard error of the slope
    std_error_slope = np.sqrt(np.sum(residuals**2) / (df * np.sum((x - x_mean) ** 2)))

    # t-statistic
    t_statistic = slope / std_error_slope

    p_value = dag_t_to_p(t_statistic, df, side)

    stats = {
        'n' : n,
        'df' : df,
        'slope' : slope, 
        'intercept' : intercept, 
        't_statistic' : t_statistic, 
        'p_value' : p_value,
    }
    return stats

def dag_t_to_p(t_statistic, df, side):
    # Caclulate the p-value
    if side=='two-sided':
        p_value = 2 * (1 - t.cdf(np.abs(t_statistic), df))
    elif side=='less':
        p_value = t.cdf(t_statistic, df)
    elif side=='greater':
        p_value = t.cdf(-t_statistic, df)        
    return p_value



def dag_coord_convert(a,b,old2new):
    ''' 
    Convert cartesian to polar and vice versa
    >> a,b          x,y or eccentricity, polar
    >> old2new      direction of conversion ('pol2cart' or 'cart2pol') 
    '''
    if old2new=="pol2cart":
        x = a * np.cos(b)
        y = a * np.sin(b)

        new_a = x
        new_b = y
    
    elif old2new=="cart2pol":            
        ecc = np.sqrt( a**2 + b**2 ) # Eccentricity
        pol = np.arctan2( b, a ) # Polar angle
        new_a = ecc
        new_b = pol
        
    return new_a, new_b

def dag_coord_rot(coords, angles):
    '''
    Rotate coordinates
    '''        
    r = R.from_euler('xyz', angles, degrees=True)
    transformed_coords = r.apply(coords)
    return transformed_coords

def dag_coord_convert3d(a,b,c,old2new):
    ''' 
    Convert cartesian to polar and vice versa
    >> a,b,c          x,y,z or eccentricity, polar, azimuth
    >> old2new      direction of conversion ('pol2cart' or 'cart2pol') 
    '''
    if old2new=="pol2cart":
        x = a * np.sin(b) * np.cos(c)
        y = a * np.sin(b) * np.sin(c)
        z = a * np.cos(b)

        new_a = x
        new_b = y
        new_c = z
    
    elif old2new=="cart2pol":            
        ecc = np.sqrt( a**2 + b**2 + c**2 ) # Eccentricity
        pol = np.arccos( c/ecc ) # Polar angle
        azi = np.arctan2( b, a ) # Azimuthal angle
        new_a = ecc
        new_b = pol
        new_c = azi
        
    return new_a, new_b, new_c

def dag_pol_difference(pol, ref_pol):
    abs_diff = np.abs(ref_pol - pol)
    abs_diff = np.min(abs_diff, 2*np.pi-abs_diff)
    return abs_diff

def dag_merid_idx(x, y, wedge_angle=15, angle_type='deg', **kwargs):
    """
    Categorize points based on their position relative to specified meridians.

    Parameters:
    - x: NumPy array of x-coordinates
    - y: NumPy array of y-coordinates
    - wedge_angle: Number of degrees around each meridian center (+/-)
    - angly_type: is wedge_angle specified in degrees or radians

    Returns:
    - Dictionary with meridians as keys and boolean NumPy arrays indicating points within each meridian's range
    """
    label_list = kwargs.get('label_list', ['right', 'upper', 'left', 'lower'])
    # Define meridian centers
    merid_centers = {'right': 0, 'upper': np.pi/2, 'left': np.pi, 'lower': -np.pi/2}
    if angle_type=='deg':
        # Convert degrees around meridian to rad
        wedge_angle *= np.pi/180
    # Calculate polar angle
    pol = np.arctan2(y, x) 
    
    merid_idx = {}
    for merid,merid_center in merid_centers.items():        
        # Get difference from meridian centre
        abs_diff = np.abs(merid_center - pol)
        abs_diff = np.min([abs_diff, 2*np.pi-abs_diff], axis=0)
        # print(abs_diff.shape)
        merid_idx[merid] = abs_diff <= wedge_angle

    # Sanity check:
    total_true = 0
    for m,m_idx in merid_idx.items():
        total_true += m_idx.sum()
    # print(f'Total true = {total_true}, total vx = {x.shape[0]}')
    
    # Collapse LR? 
    merid_idx['horizontal'] = merid_idx['left'] | merid_idx['right']
    merid_idx['vertical'] = merid_idx['upper'] | merid_idx['lower']        
    merid_label = np.full(x.shape[0], 'na', dtype='object')
    
    for label in label_list:
        merid_label[merid_idx[label]] = label
    merid_idx['label'] = merid_label

    return merid_idx



def dag_pol_to_clock(pol):
    # Convert angles to the range [0, 2*pi)
    # rotate by 90
    pol = pol + np.pi/2
    pol = np.mod(pol, 2 * np.pi)

    # Convert angles to the range [0, 12)
    clock_values = (pol / (2 * np.pi)) * 12
    return clock_values


def dag_weighted_mean(w,x, axis='all'):
    # w_mean = np.sum(w * x) / np.sum(w) # original form
    if axis=='all':
        w_mean = np.nansum(w * x) / np.nansum(w)
    else:
        w_mean = np.nansum(w * x, axis=axis) / np.nansum(w, axis=axis)

    return w_mean


def dag_get_pos_change(old_x, old_y, new_x, new_y):
    dx = new_x - old_x
    dy = new_y - old_y
    dsize = np.sqrt(dx**2 + dy**2)
    return dsize