from rasterio.enums import Resampling
import xarray as xr
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import median_absolute_deviation

def hi():
    print("Hi")
    
def create_dem_dataset(
    dem_dict,
    reference_dem_key,
    resampling_alg = Resampling.cubic
):
    """
    Take a dictionary of any keys to xarray.DataArrays, and given the key of the reference,
    reproject all rasters to the reference raster.
    
    dem_files_dict: dictionary of pd.Datetime to xarray.Dataset of DEM opened with rioxarray
    reference_dem_key: key in the dem_files_dict pointing to the DEM to be used as reference DEM
    resampling_alg: rasterio.enums.Resampling algorithm used to reproject all DEMs to the reference DEM
    """
    reference_dem = dem_dict[reference_dem_key]
    
    for key,dem in dem_dict.items():
        if key is not reference_dem_key:
            dem_dict[key] = dem.rio.reproject_match(reference_dem, resampling=resampling_alg)
    
    return xr.Dataset(dem_dict)

def assign_dem_uncertainty_values(dem_dataset, static_area_geometries, reference_dem_key):
    """
    Take a xarray.Dataset and for each data variable, extract the values of pixels within
    the provided geometries. Calculate statistics for these pixels using control values taken
    from the DataArray indicated by the provided "reference_dem_key" data variable name. The
    reference DEM is regarded as "truth", or the ground control values. Add the statistical 
    data to the attrs dictionary of each DataArray in the dataset.
    
    dem_dataset: Dataset with DEM data variables to assign statistics to
    reference_dem_key: key in the dem_files_dict pointing to the DEM to be used as reference DEM
    static_area_geometries: geometries containing static areas. CRS of geometries should match
                            that of the all the DataArrays in the dataset.
    """
    #
    dem_dataset_clipped = dem_dataset.rio.clip(static_area_geometries.dropna())
#     faster? way
#     [ dem_dataset.rio.clip_box(*geometry.bounds).rio.clip([geometry]) 
#          for geometry in gca_gdf['geometry'] 
#     ]
    
    for var in dem_dataset_clipped:
        if var is not reference_dem_key:
            static_noise_dataarray = dem_dataset_clipped[var] - dem_dataset_clipped[reference_dem_key]
            static_noise = static_noise_dataarray.values.flatten()
            static_noise = static_noise[~np.isnan(static_noise)]
            dem_dataset[var].attrs['RMSE'] = mean_squared_error(
                static_noise,
                np.full(len(static_noise), 0), 
                squared=False
            )
            dem_dataset[var].attrs['MSE'] = mean_squared_error(
                static_noise,
                np.full(len(static_noise), 0), 
                squared=True
            )
            dem_dataset[var].attrs['Median'] = np.median(static_noise)
            dem_dataset[var].attrs['NMAD'] = median_absolute_deviation(static_noise)
            dem_dataset[var].attrs['84th Percentile'] = np.percentile(static_noise, 84)
            dem_dataset[var].attrs['16th Percentile'] = np.percentile(static_noise, 16)
            
    return dem_dataset