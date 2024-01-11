import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.cm
import xdem
from scipy import stats


def uncertainty_analysis(
    dh,
    ground_control_vector,
    subsample = None,
    n_variograms = None,
    xscale_range_split = None,
    ylim = None,
    maxlag = None,
    parallelism = 1,
    FILTER_OUTLIERS = True,
    SIMPLE_FILTER = True,
    simple_filter_threshold = None
):
    """Helper function to analyze uncertainty of a DoD.

    Some parameters that make sense:
    subsample = 1800
    n_variograms = 10
    xscale_range_split = [200]
    maxlag = 1000 #But try without providing this first.
    parallelism = 32 (how many cores do you have?)
    FILTER_OUTLIERS = False (you probably don't want to do this now.)

    Args:
        dh (xdem.dDEM): A dDEM.
        ground_control_vector (gu.Vector): Polygons defining your bareground area.
        subsample (int, optional): number of samples to randomly draw from the values. See documentation for xdem.spatialstats.sample_empirical_variogram. Defaults to None.
        n_variograms (int, optional): number of independent empirical variogram estimations. See documentation for xdem.spatialstats.sample_empirical_variogram. Defaults to None.
        xscale_range_split (list of int, optional): List of ranges at which to split the variogram figure. See documentation for xdem.spatialstats.plot_vgm. Defaults to None.
        ylim (float, optional): ylim for matplitlib plotting of spatiovariogram. See documentation for xdem.spatialstats.plot_vgm. Defaults to None.
        maxlag (int, optional): Maximum distance between coordinates to calculate variance. See documentation for xdem.spatialstats.sample_empirical_variogram. Defaults to None.
        parallelism (int, optional): Number of processing cores. See documentation for xdem.spatialstats.sample_empirical_variogram. Defaults to 1.
        FILTER_OUTLIERS (bool, optional): Whether or not to filter outliers. Defaults to True.
        SIMPLE_FILTER (bool, optional): If FILTER_OUTLIERS, this switch determines if you do complex or simple filtering. If SIMPLE_FILTER, you must provide a value for simple_filter_threshold, else, outliers are removed outside Median +/- 4*nmad. Defaults to True.
        simple_filter_threshold (_type_, optional): Threshold for simple outlier filtering. Provide a single value like 50  to remove all values between -50 and 50. Defaults to None.

    Returns:
        _type_: _description_
    """
    figs = []

    gcas_mask = ground_control_vector.create_mask(dh)
    # Masked array containing all data
    all_values_masked = dh.data.copy()
    all_values = all_values_masked.filled(np.nan)
    # Masked array that we set to only contain data in stable areas (ground control areas)
    stable_values_masked = dh.data.copy()
    stable_values_masked.mask = np.ma.mask_or(stable_values_masked.mask, ~gcas_mask)
    stable_values = stable_values_masked.filled(np.nan)
    
    plt.figure(figsize=(4, 3))
    cmap = copy.copy(matplotlib.cm.get_cmap("RdYlBu"))
    cmap.set_bad('grey',1.)
    
    # # maybe plot something else here?
    _ = plt.gca().imshow(stable_values_masked.squeeze(),cmap=cmap, vmin=-4, vmax=4)
    plt.gca().set_title('Elevation differences (m)')
    figs.append(plt.gcf())

    if FILTER_OUTLIERS:
        if SIMPLE_FILTER:
            stable_values_filt = np.where(np.abs(stable_values) < simple_filter_threshold, stable_values, np.nan)
            all_values_filt = np.where(np.abs(all_values) < simple_filter_threshold, all_values, np.nan)
        else:
            low = np.nanmedian(all_values) - 4*xdem.spatialstats.nmad(all_values)
            high = np.nanmedian(all_values) + 4*xdem.spatialstats.nmad(all_values)
            stable_values_filt = np.where(stable_values < high, stable_values, np.nan)
            stable_values_filt = np.where(stable_values_filt > low, stable_values_filt, np.nan)
            all_values_filt = np.where(all_values < high, all_values, np.nan)
            all_values_filt = np.where(all_values_filt > low, all_values_filt, np.nan)
    else:
        stable_values_filt = stable_values
        all_values_filt = all_values

    df = xdem.spatialstats.sample_empirical_variogram(
        values=stable_values_filt,
        gsd=dh.res[0],
        subsample=subsample,
        n_jobs=parallelism,
        n_variograms=n_variograms,
        maxlag=maxlag
    )
    
    if ylim:
        xdem.spatialstats.plot_variogram(
            df,
            xscale_range_split=xscale_range_split,
            ylim=ylim
        )
    else:
        xdem.spatialstats.plot_variogram(
            df,
            xscale_range_split=xscale_range_split
        )
    figs.append(plt.gcf())
    # plt.show(block=False)
    fun, params = xdem.spatialstats.fit_sum_model_variogram(['Sph'], empirical_variogram=df)
    if ylim:
        xdem.spatialstats.plot_variogram(
            df,
            list_fit_fun=[fun],
            list_fit_fun_label=['Single-range model'],
            xscale_range_split=xscale_range_split,
            ylim=ylim
        )
    else:
        xdem.spatialstats.plot_variogram(
            df,
            list_fit_fun=[fun],
            list_fit_fun_label=['Single-range model'],
            xscale_range_split=xscale_range_split
        )
    figs.append(plt.gcf())
    # plt.show(block=False)

    results_dict = {
        "Range": params['range'].iloc[0],
        "Sill": params['psill'].iloc[0],
        "Interval": dh.interval,
        "NMAD": xdem.spatialstats.nmad(stable_values_filt),
        "Mean": float(np.nanmean(stable_values_filt)),
        "Median": float(np.nanmedian(stable_values_filt)),
        "RMSE": xdem.spatialstats.rmse(stable_values_filt),
        "StdDev": float(np.nanstd(stable_values_filt)),
        "Max": float(np.nanmax(stable_values_filt)),
        "Min": float(np.nanmin(stable_values_filt)),
        "Count of stable pixels": np.count_nonzero(~np.isnan(stable_values_filt)),
        "Count of all pixels": np.count_nonzero(~np.isnan(all_values_filt)),
        "Outlier lower limit": (-simple_filter_threshold if SIMPLE_FILTER else low) if FILTER_OUTLIERS else np.nan,
        "Outlier upper limit": (simple_filter_threshold if SIMPLE_FILTER else high) if FILTER_OUTLIERS else np.nan,

        "pre-filter": {            
            "NMAD": xdem.spatialstats.nmad(stable_values),
            "Mean": float(np.nanmean(stable_values)),
            "Median": float(np.nanmedian(stable_values)),
            "RMSE": xdem.spatialstats.rmse(stable_values),
            "StdDev": float(np.nanstd(stable_values)),
            "Max": float(np.nanmax(stable_values)),
            "Min": float(np.nanmin(stable_values)),
            "Count of stable pixels": np.count_nonzero(~np.isnan(stable_values)),
            "Count of all pixels": np.count_nonzero(~np.isnan(all_values))
        }
    }
    #('count of pixels before filtering' - 'count of pixels after filtering')/'count of pixels before filtering'
    results_dict["Percentage all pixels filtered/removed"] = (
        (results_dict['pre-filter']['Count of all pixels'] - results_dict['Count of all pixels']) / results_dict['pre-filter']['Count of all pixels']
    )
    results_dict["Percentage stable pixels filtered/removed"] = (
        (results_dict['pre-filter']['Count of stable pixels'] - results_dict['Count of stable pixels']) / results_dict['pre-filter']['Count of stable pixels']
    )
    return results_dict, figs
    