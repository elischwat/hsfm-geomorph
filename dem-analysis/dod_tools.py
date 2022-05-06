from rasterio.enums import Resampling
import xarray as xr
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import median_absolute_deviation
# import skgstat as skg

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


def variogram_calculation(dataset, geom):
    "Returns: (range, variance)"
    sv_df = pd.DataFrame({
        'y': [],
        'x': [],
        'value': []
    })
    data = dataset.rio.clip_box(*geom.bounds).rio.clip([geom]).squeeze()
    if len(pd.Series(data.values.flatten()).dropna()) > 0:
        for y in data.coords['y']:
            for x in data.coords['x']:
                value = data.loc[y, x].item()
                if not np.isnan(value):
                    sv_df = sv_df.append({
                            'y': y.item(),
                            'x': x.item(),
                            'value': value
                        }, 
                        ignore_index=True
                    )
        if len(sv_df) > 2 and len(sv_df['value'].dropna()) > 0:
            try:
                variogram = skg.Variogram(
                    sv_df[['x','y']].values, 
                    sv_df['value'].values
                )
            except:
                print(sv_df[['x','y']].values)
                print()
                print(sv_df['value'].values)
                
            return variogram.cof[0], np.var(sv_df['value'])
        else:
            return np.nan, np.nan
    else:
        return np.nan, np.nan    
    
def assign_dod_uncertainty_statistics(dod_dataset, static_area_geometries, sc_geometry_area_threshold=1000):
    """
    Take an xarray.Dataset and for each data variable, extract the values of 
    pixels within the provided geometries. Calculate statistics for these pixels
    assuming that the true values are 0. Add the statistical data to the attrs
    dictionay of each DataArray in the dataset.
    
    dod_dataset: Dataset with DoD data variables to assign statistics to
    static_area_geometries: geometries containing static areas. CRS of geometries should match
                            that of the all the DataArrays in the dataset.
    """
    dod_dataset_clipped = dod_dataset.rio.clip(static_area_geometries.dropna())
    for var in dod_dataset_clipped:
        static_noise = dod_dataset_clipped[var].values.flatten()
        static_noise = static_noise[~np.isnan(static_noise)]
        dod_dataset[var].attrs['RMSE'] = mean_squared_error(
            static_noise,
            np.full(len(static_noise), 0), 
            squared=False
        )
        dod_dataset[var].attrs['MSE'] = mean_squared_error(
            static_noise,
            np.full(len(static_noise), 0), 
            squared=True
        )
        dod_dataset[var].attrs['Median'] = np.median(static_noise)
        dod_dataset[var].attrs['Mean'] = np.mean(static_noise)
        dod_dataset[var].attrs['Standard Deviation'] = np.std(static_noise)
        dod_dataset[var].attrs['NMAD'] = median_absolute_deviation(static_noise)
        dod_dataset[var].attrs['Variance'] = np.var(static_noise)
        dod_dataset[var].attrs['84th Percentile'] = np.percentile(static_noise, 84)
        dod_dataset[var].attrs['16th Percentile'] = np.percentile(static_noise, 16)
        
        #calculate average errors variance and average errors variogram range 
        #    Note that the variance and variogram range are calculated separately for each static area geometry
        #    and then averaged
        # variogram_geoms = static_area_geometries.dropna()
        # variogram_geoms = variogram_geoms[variogram_geoms.area > sc_geometry_area_threshold]
        # ranges, variances = zip(*[variogram_calculation(dod_dataset_clipped[var], geom) for geom in variogram_geoms])
        # dod_dataset[var].attrs['Average Variogram Range'] = pd.Series(ranges).dropna().mean()
        # dod_dataset[var].attrs['Average Variogram Variance'] = pd.Series(variances).dropna().mean()
        
        
    return dod_dataset

def assign_dem_uncertainty_values(dem_dataset, static_area_geometries, reference_dem_key, sc_geometry_area_threshold=1000):
    """
    Take a xarray.Dataset and for each data variable, extract the values of pixels within
    the provided geometries. Calculate statistics for these pixels using control values taken
x    reference DEM is regarded as "truth", or the ground control values. Add the statistical 
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
        if var != reference_dem_key:
            print(f'Calculating var {var} with reference {reference_dem_key}, {var==reference_dem_key}')
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
            dem_dataset[var].attrs['Mean'] = np.mean(static_noise)
            dem_dataset[var].attrs['Standard Deviation'] = np.std(static_noise)
            dem_dataset[var].attrs['NMAD'] = median_absolute_deviation(static_noise)
            dem_dataset[var].attrs['Variance'] = np.var(static_noise)
            dem_dataset[var].attrs['84th Percentile'] = np.percentile(static_noise, 84)
            dem_dataset[var].attrs['16th Percentile'] = np.percentile(static_noise, 16)
            
    return dem_dataset

def get_dod_keys_for_valley(name, valleys_metadata_dict):
    relevant_dods = valleys_metadata_dict[name]['dod pairs'] + [
        valleys_metadata_dict[name]['bounding dod pair']
    ]
    # return [(int(yr1), int(yr2)) for yr1, yr2 in relevant_dods]
    return [(yr1, yr2) for yr1, yr2 in relevant_dods]

def apply_shared_mask(dataset):
    """
    Apply a shared mask across datasets so that different DoDs for a given area 
    don't contribute differing number of pixels to mass wasted calculations.
    """
    vars_list = list(dataset.data_vars)
    # create an initial mask with the first data variable in the dataset
    combined_mask = np.isnan(dataset[vars_list[0]])
    # iterate over the rest of the data variables, modifying the nan mask each time
    for var in vars_list[1:]:
        combined_mask = combined_mask | np.isnan(dataset[var])
    # iterate over the data variables, applying the combined mask
    for var in vars_list:
        dataset[var] = dataset[var].where(~combined_mask)
    return dataset

def calculate_mass_wasted(
    dod_dataset, 
    polygons_gdf, 
    carry_columns, 
    carry_attrs, 
    valleys_metadata_dict,
    share_mask
):
    """
    
    dod_dataset: Dataset with DoDs as data variables
    polygons: GeoDataframe to calculate mass wasted for
    carry_columns: List of column names corresponding to columns in polygons_gdf that should be copied
                    into the new dataframe (With mass wasted information) that will be returned
    carry_attrs: List of attribute keys corresponding to attributes in the dataset data variable Dataarrays
                that should be copied into the new dataframe that will be returned
    valleys_metadata_dict: dictionary of valley metadata describing which DoDs apply to which polygons.
                            Data is mapped between the dictionary and the polygons gdf using the 'name' 
                            column
    share_mask: whether to share a nan/nodata mask across DODs when clipping by erosion polygons
    """

    
    mass_wasted_df = pd.DataFrame()
    
    for index, p in polygons_gdf.iterrows():
        #filter the dod_dataset so that it only contains DoDs relevant to the valley of this geometry
        relevant_keys = get_dod_keys_for_valley(p['name'], valleys_metadata_dict)
        dataset = dod_dataset[relevant_keys]
        dataset = dataset.rio.clip_box(*p['geometry'].bounds).rio.clip([p['geometry']])
        
        if share_mask:
            dataset = apply_shared_mask(dataset.copy(deep=True))

        pixel_area = dataset.rio.resolution()[0] * dataset.rio.resolution()[1]
        if pixel_area < 0:
            pixel_area = -pixel_area

        for data_var in dataset.data_vars:
            n_valid_pixels = len(pd.Series(dataset[data_var].values.flatten()).dropna())
            pixel_area = dataset[data_var].rio.resolution()[0] * dataset[data_var].rio.resolution()[1]
            if pixel_area < 0:
                pixel_area = - pixel_area
                
            mass_wasted_data = {}    
            mass_wasted_data['mass_wasted'] = dataset[data_var].sum().values * pixel_area
            mass_wasted_data['dataset_name'] = dataset[data_var].name
            mass_wasted_data['n_valid_pixels'] = n_valid_pixels
            mass_wasted_data['pixel_area'] = pixel_area
            for col in carry_columns:
                mass_wasted_data[col] = p[col]
            for attr in carry_attrs:
                mass_wasted_data[attr] = dataset[data_var].attrs[attr]
            
            mass_wasted_df = mass_wasted_df.append(mass_wasted_data, ignore_index=True)
    return mass_wasted_df
