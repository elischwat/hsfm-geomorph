# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import os
import rioxarray as rix
import pandas as pd
import geopandas as gpd
from datetime import datetime
import altair as alt
from altair import datum

import xarray as xr
import numpy as np
import holoviews as hv
import hvplot.xarray
import geoviews as gv
from holoviews.operation.datashader import regrid
import cartopy.crs

hv.extension('bokeh')
# -

# # Load and organize data

# ## DEMs

# Using the QGIS mt_baker_mass_wasted project, I observed the coverage of all DEMs for each glacial valley:
#
# ### Create dictionary describing DEM coverage for each valley
#
# Deming
#     70 (poorish coverage) 
#     79
#     90 (poor coverage)
#     91
#
# Thunder
#     70
#     79
#
# Coleman/Roosavelt
#     70
#     79
#     87 (poor coverage)
#     90
#
# Mazama
#     70
#     77 (bad error?)
#     79
#
# Rainbow (seems very active compared to others)
#     70
#     74
#     79
#     91
#
# Park Glacier
#     70
#     74 (poor coverage)
#     79
#     91
#
# Boulder Glacier
#     67 (poor coverage)
#     70
#     74 (poor coverage)
#     79
#     87
#     91
#
# Talum Glaciers
#     70
#     77
#     79
#     87
#     91
#
# Squak
#     70
#     77
#     79
#     91
#
# Easton
#     70
#     74 (poor coverage)
#     79
#     90 (poor coverage)
#     91

# Create a dictionary describing coverage, only using DEMs with better-than-poor coverage

valleys_metadata_dict = {
    'Deming': {
        'years': [1970, 1979, 1991, 2015],
        'max_coverage_year': 1979
    },
    'Thunder': {
        'years': [1970, 1979, 2015],
        'max_coverage_year': 1979
    },
    'Coleman/Roosevelt': {
        'years': [1970, 1979, 1990, 2015],
        'max_coverage_year': 1979
    },
    'Mazama': {
        'years': [1970, 1979, 2015], #removed 1977 after the fact due to apparent systematic error
        'max_coverage_year': 1979
    },
    'Rainbow': {
        'years': [1970, 1979, 1991, 2015],
        'max_coverage_year': 1970
    },
    'Park': {
        'years': [1970, 1979, 1991, 2015],
        'max_coverage_year': 1970
    },
    'Boulder': {
        'years': [1970, 1979, 1987, 1991, 2015],
        'max_coverage_year': 1970
    },
    'Talum': {
        'years': [1970, 1977, 1979, 1987, 1991, 2015],
        'max_coverage_year': 1979
    },
    'Squak': {
        'years': [1970, 1977, 1979, 1991, 2015],
        'max_coverage_year': 1979
    },
    'Easton': {
        'years': [1970, 1979, 1991, 2015],
        'max_coverage_year': 1979
    },
}

# Create a dictionary of DEM file paths by year

reference_dem_file = '/data2/elilouis/hsfm-geomorph/data/reference_dem_highres/baker/raw_tifs/baker_2015/2015.tif'
dem_files_dict = {
    1970: "/data2/elilouis/generate_ee_dems_baker/mixed_timesift/individual_clouds/70_9.0/dem.tif",
    1974: "/data2/elilouis/generate_ee_dems_baker/mixed_timesift/individual_clouds/74_8.0/dem.tif",
    1977: "/data2/elilouis/generate_ee_dems_baker/mixed_timesift/individual_clouds/77_9.0/dem.tif",
    1979: "/data2/elilouis/generate_ee_dems_baker/mixed_timesift/individual_clouds/79_10.0/dem.tif",
    1987: "/data2/elilouis/generate_ee_dems_baker/mixed_timesift/individual_clouds/87_8.0/dem.tif",
    1990: "/data2/elilouis/generate_ee_dems_baker/mixed_timesift/individual_clouds/90_9.0/dem.tif",
    1991: "/data2/elilouis/generate_ee_dems_baker/mixed_timesift/individual_clouds/91_9.0/dem.tif",
    2015: reference_dem_file
}

# Check that all these file paths exists

for k,file in dem_files_dict.items():
    print(k, f'Exists: {os.path.exists(file)}', 'Resolution: ', rix.open_rasterio(file).rio.resolution())

# Load reference DEM and regrid to 1 meter. This grid will be used to grid all DEMs. The same grid is necessary for differencing

# %%time
reference_dem = rix.open_rasterio(reference_dem_file, masked=True, chunks=1000).squeeze()
reference_dem = reference_dem.rio.reproject(reference_dem.rio.crs, resolution=(1,1))
reference_dem.name = 'reference dem'

# In case you dont want to reproject, the file is saved and reloaded in here

regridded_1m_reference_dem_fn = reference_dem_file.replace('2015.tif', '2015_1m.tif')
reference_dem.rio.to_raster(regridded_1m_reference_dem_fn)
# !ls $regridded_1m_reference_dem_fn
reference_dem = rix.open_rasterio(regridded_1m_reference_dem_fn, masked=True, chunks=1000).squeeze()
reference_dem.name = 'reference dem'

# Load all DEMs into a dictionary, reprojecting each onto the reference grid

# %%time
dem_dict = {
    k: rix.open_rasterio(v, masked=True).squeeze().rio.reproject_match(reference_dem)
    for k, v in dem_files_dict.items()
}

{k:v.rio.resolution() for k,v in dem_dict.items()}

# ### Create DEM xarray.Dataset

dem_dataset = xr.Dataset(dem_dict)

# ### Calculate Uncertainty for DEMs

# Read in Ground Control Area (GCA) polygons

gca_gdf = gpd.read_file('/data2/elilouis/hsfm-geomorph/data/mt_baker_mass_wasted/gcas.geojson')
gca_gdf.plot()

# Clip all DEM datasets by GCA polygons

dem_dataset_clipped =  dem_dataset.rio.clip(gca_gdf.geometry)

for var in dem_dataset_clipped.data_vars:
    print(dem_dataset_clipped[var].shape)

# ### Add Uncertainty ('RMSE') attribute to DEM xarray.Dataset
# Calculate RMSE and add as an attribute to the dem_dataset DataArray, also plot the Ground Control Area difference distributions

import seaborn as sns


def get_rmse(values):
    data_series = pd.Series(values.flatten()).dropna()
    rms = np.sqrt(data_series.apply(lambda x: x**2).sum()/len(data_series))
    return rms


fig,axes = plt.subplots(len(list(dem_dataset_clipped.data_vars)[:-1]), figsize=(5,15), sharex=True, sharey=True)
for ax, var in zip(axes, list(dem_dataset_clipped.data_vars)[:-1]):
    data = dem_dataset_clipped[var] - dem_dataset_clipped[list(dem_dataset_clipped.data_vars)[-1]]
    sns.distplot(data.values.flatten(), kde=False, ax = ax)
    rmse = get_rmse(data.values)
    dem_dataset_clipped[var].attrs['RMSE'] = rmse
    dem_dataset[var].attrs['RMSE'] = rmse
    ax.set_title(str(var) + ", RMSE: {:.2f}".format(rmse))
    ax.set_xlim(-10,10)

for var in dem_dataset.data_vars:
    print(dem_dataset[var].name)
    if dem_dataset[var].attrs.get('RMSE'):
        print(dem_dataset[var].attrs['RMSE'])
    else:
        print('no error calculated')
    print()

# ## DoDs

# ### Load valley bounds 

valley_bounds = gpd.read_file('/data2/elilouis/hsfm-geomorph/data/mt_baker_mass_wasted/valley_bounds.geojson')

# Add pairs of years to the valleys_metadata_dict

for k, v in valleys_metadata_dict.items():
    def get_dod_pairs_from_list(list_of_years):
        return [
            (list_of_years[i], list_of_years[i+1]) for i in range(0, len(list_of_years)-1)
        ]    
    valleys_metadata_dict[k]['dod pairs'] = get_dod_pairs_from_list(v['years'])
    valleys_metadata_dict[k]['bounding dod pair'] = (v['years'][0], v['years'][-1])
valleys_metadata_dict

# ### Create DoD xarray.Dataset 
#
# (really a dictionary of valley names to xarray.Dataset s)

# Create one dod xarray.Dataset of DoDs for each valley. Create one DOD for each dod pair in the valleys_metadata_dict, and one for the bounding pair as well.

all([np.nan, 5]), all([5, 5])


def combine_rmse(rmses):
    rmses_series = pd.Series(rmses).where(pd.notnull(rmses), None)
    if not all(rmses_series):
        return np.nan
    else: 
        return np.sqrt(
            (
                ((1 / len(rmses)) * rmses_series.apply(lambda x: x**2)).sum()
            )
        )


# ### Manually add the RMSE for the LIDAR data (assume 10cm for now)

dem_dataset[2015].attrs['RMSE'] = 0.10

[dem_dataset[var].attrs for var in dem_dataset.data_vars]

# +
dod_dataset_dict = {}
def add_data_var_to_dataset(dataset, new_name, new_dataarray):
    if '_file_obj' not in new_dataset.attrs.keys():
        new_dataarray['_file_obj'] = None
    dataset[new_name] = new_dataarray
    return dataset
    
for valley_name, valley_metadata in valleys_metadata_dict.items():
    print(f'Creating DODs for {valley_name}...')
    new_dataset = xr.Dataset()
    valley_box = valley_bounds[valley_bounds.name==valley_name].geometry.iloc[0].bounds
    # add all the dod data for sequential dods
    for first_year, second_year in valley_metadata['dod pairs']:
        dod = dem_dataset[second_year].rio.clip_box(*valley_box) - dem_dataset[first_year].rio.clip_box(*valley_box)
        dod.attrs['RMSE'] = combine_rmse(
            [
                dem_dataset[second_year].attrs.get('RMSE'), 
                dem_dataset[first_year].attrs.get('RMSE') 
            ]
        )
        new_dataset = add_data_var_to_dataset(
            new_dataset,
            f"{first_year}-{second_year}",
            dod
        )
    
    # add the dod data for the "bounding" dod
    first_year, last_year = valley_metadata['bounding dod pair']
    bounding_dod = dem_dataset[last_year].rio.clip_box(*valley_box) - dem_dataset[first_year].rio.clip_box(*valley_box)
    bounding_dod.attrs['RMSE'] = combine_rmse([
        dem_dataset[last_year].attrs.get('RMSE'),
        dem_dataset[first_year].attrs.get('RMSE'),
        
    ])
    new_dataset = add_data_var_to_dataset(
        new_dataset,
        f"Bounding {first_year}-{last_year}",
        bounding_dod
    )
    
    dod_dataset_dict[valley_name] = new_dataset
# -

dod_dataset_dict.keys()

for var in dem_dataset.data_vars:
    print(var)
    print(dem_dataset[var].attrs.get('RMSE'))
    print()

for key, value in dod_dataset_dict.items():
    for var in value.data_vars:
        print(var + ':\t\t' + 
              str(value[var].attrs.get('RMSE'))
             )

# ### Crop out glaciers from DODs
#
# Use the glacier polygons from both years that define the DOD

# Load glacier polygons

glacier_polygons = gpd.read_file(
    '/data2/elilouis/hsfm-geomorph/data/mt_baker_mass_wasted/glacier_polygons_combined.geojson'
)
# fill in empty geometries...
from shapely.geometry.collection import GeometryCollection
glacier_polygons['geometry'] = glacier_polygons.geometry.apply(lambda x: x if x else GeometryCollection())
glacier_polygons = glacier_polygons.to_crs(epsg=32610)

# Iterate over DOD datasets, and iterate over each DOD/data variable within each dataset, masking out glacier pixels using both years

for valley_name, dataset in dod_dataset_dict.items():
    for dod_name in dataset.data_vars:
        first_year = dod_name.replace('Bounding ', '').split('-')[0]
        second_year = dod_name.replace('Bounding ', '').split('-')[1]
        relevant_polygons = glacier_polygons[glacier_polygons.year.isin([first_year, second_year])]
        print(f"For years {first_year} and {second_year}, found {len(relevant_polygons)} glacier polygons from years {relevant_polygons.year.unique()} for masking.")
        dod_dataset_dict[valley_name][dod_name] = dod_dataset_dict[valley_name][dod_name].rio.clip(
            glacier_polygons.geometry, invert=True
        )

# # Calculate mass wasted

# ## Load Erosion polygons

erosion_polygons = gpd.read_file(
    '/data2/elilouis/hsfm-geomorph/data/mt_baker_mass_wasted/erosion_polygons.geojson'
).assign(fluvial=False)
fluvial_erosion_polygons = gpd.read_file(
    '/data2/elilouis/hsfm-geomorph/data/mt_baker_mass_wasted/fluvial_erosion_polygons.geojson'
).assign(fluvial=True)
polygons = pd.concat([erosion_polygons, fluvial_erosion_polygons]).reset_index(drop=True)


# Define function that will apply a shared mask across datasets so that different DoDs for a given area dont contribute differing number of pixels to mass wasted calculations

def apply_shared_mask(dataset):
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


# For each polygon, calculate mass wasted using each possible DOD. DODs are generated for each pair of DEMs available for the given glacial valley. The `name` column is used to match between the DEM dictionary and the erosion polygon dataframes

# +
# %%time
mass_wasted_df = pd.DataFrame()

for index, p in polygons.iterrows():
    dataset = dod_dataset_dict[p['name']].rio.clip([p['geometry']])
    dataset = apply_shared_mask(dataset)
    
    pixel_area = dataset.rio.resolution()[0] * dataset.rio.resolution()[1]
    if pixel_area < 0:
        pixel_area = -pixel_area
    
    for data_var in dataset.data_vars:
        n_valid_pixels = len(pd.Series(dataset[data_var].values.flatten()).dropna())
        pixel_area = dataset[data_var].rio.resolution()[0] * dataset[data_var].rio.resolution()[1]
        if pixel_area < 0:
            pixel_area = - pixel_area
        mass_wasted_df = mass_wasted_df.append({
                'name': p['name'],
                'mass_wasted': dataset[data_var].sum().values * pixel_area,
                'dataset_name': dataset[data_var].name,
                'fluvial': p['fluvial'],
                'id': p['id'],
                'n_valid_pixels': n_valid_pixels,
                'dod_raster_rmse': dataset[data_var].attrs['RMSE'],
                'pixel_area': pixel_area
            }, 
            ignore_index=True
        )

mass_wasted_df
# -

# Calculate mass wasted volumetric uncertainty using equation 22 from Scott Anderson, assuming only random error (no systematic error)
#
# $$\sigma_v = n L^2 \sqrt{\sigma^2_{rms} / n }$$
#
# where $n$ is number of pixels used in measurement, $L^2$ is the pixel area, and $\sigma_{rms}$ is the root mean squared error of the DOD.

mass_wasted_df['mass_wasted_uncertainty'] = (
    mass_wasted_df['n_valid_pixels'] * mass_wasted_df['pixel_area']
)*np.sqrt(
    mass_wasted_df['dod_raster_rmse']**2 / mass_wasted_df['n_valid_pixels']
)

mass_wasted_df

# Convert the fluvial column from int (0/1) to bool, make sure all values in id column have a value, parse years and bounding (or not) from dataset_name column, and lastly assign dummy pd.datetime dates for the `start_year` and `end_year` data columns.
#
# I do this so Altair can do a better job plotting.
#
# Note that I only have to do the parsing of "dataset_name" column to start and end years because of what i do in the loop above, and how the DODs are organized into a dataset with names

# +
mass_wasted_df['fluvial'] = mass_wasted_df['fluvial'].astype(bool)
mass_wasted_df['id'] = mass_wasted_df['id'].apply(lambda x: 0 if not x else x)

mass_wasted_df['start_year'] = mass_wasted_df['dataset_name'].apply(lambda x:
    int(x.split('Bounding ')[1].split('-')[0]) if 'Bounding ' in x else int(x.split('-')[0])
)
mass_wasted_df['end_year'] = mass_wasted_df['dataset_name'].apply(lambda x:
    int(x.split('Bounding ')[1].split('-')[1]) if 'Bounding ' in x else int(x.split('-')[1])
)
mass_wasted_df['Start Date'] = mass_wasted_df['start_year'].apply(lambda x: datetime(int(x), 1, 1))
mass_wasted_df['End Date'] = mass_wasted_df['end_year'].apply(lambda x: datetime(int(x), 1, 1))
mass_wasted_df['Measurement'] = mass_wasted_df['dataset_name'].apply(lambda x: 'Bounding' if 'Bounding' in x else 'Normal')
mass_wasted_df


# -

# Calculate polygon-specific sums using each individual time-interval measurement. These will be compared to the "bounding" measurements already generated.

# ## Calculate combined uncertainty... 
# when adding mass wasted values, for individual erosion polygons, across multiple time periods (calculating "sum of intervals")
#
# ToDo: Is this correct??

def combine_annual_uncertainty(uncertainties):
    return np.sqrt((uncertainties**2).sum())


sum_of_intervals = mass_wasted_df[mass_wasted_df['Measurement']=='Normal']
sum_of_intervals = sum_of_intervals.groupby(['fluvial', 'id', 'name']).agg({
     'mass_wasted':'sum', 
     'Start Date':'min', 
     'End Date':'max', 
    'mass_wasted_uncertainty': combine_annual_uncertainty
}).reset_index()
sum_of_intervals['Measurement'] = 'sum of intervals'

mass_wasted_df = mass_wasted_df[sum_of_intervals.columns]
mass_wasted_df = pd.concat([mass_wasted_df, sum_of_intervals])

mass_wasted_df[mass_wasted_df.name=='Rainbow']

# # Calculate mass wasted rates

# ToDo: Confirm - can i calculate mass wasted rate uncertainties by simply dividing the uncertainties by the number of years??

# +
from dateutil.relativedelta import relativedelta
mass_wasted_df['N Years'] = mass_wasted_df.apply(lambda x: relativedelta(x['End Date'], x['Start Date']).years, axis=1)
mass_wasted_df['mass_wasted_rate'] = mass_wasted_df['mass_wasted']/mass_wasted_df['N Years']
mass_wasted_df['mass_wasted_rate_uncertainty'] = mass_wasted_df['mass_wasted_uncertainty']/mass_wasted_df['N Years']

mass_wasted_df
# -

# # Plot: Mass wasted

# ## Fluvial Erosion

# New version (with uncertainties):

mass_wasted_df.head()

# New version (with uncertainties):

# +
src_normal = mass_wasted_df[
    (mass_wasted_df.fluvial) & (mass_wasted_df.Measurement == 'Normal')
]
src_normal = src_normal.groupby(['name', 'Start Date', 'End Date']).agg({
    'mass_wasted'  : 'sum',
    'mass_wasted_uncertainty'  : 'sum',
    'mass_wasted_rate': 'sum',
    'mass_wasted_rate_uncertainty': combine_annual_uncertainty
    
}).reset_index()

#half way between the bounding dates, just for plotting convenience
src_normal['uncertainty_date'] = src_normal['Start Date'] + (
    src_normal['End Date'] - src_normal['Start Date']
) / 2
src_normal['mass_wasted_rate_uncertainty_upper_bound'] = src_normal['mass_wasted_rate'] + src_normal['mass_wasted_rate_uncertainty']
src_normal['mass_wasted_rate_uncertainty_lower_bound'] = src_normal['mass_wasted_rate'] - src_normal['mass_wasted_rate_uncertainty']

# +
src_totals = mass_wasted_df[
    (mass_wasted_df.fluvial) & (mass_wasted_df.Measurement != 'Normal')
]
src_totals = src_totals.groupby(['name', 'Measurement', 'Start Date', 'End Date']).agg({
    'mass_wasted'  : 'sum',
    'mass_wasted_uncertainty'  : 'sum',
    'mass_wasted_rate': 'sum',
    'mass_wasted_rate_uncertainty': combine_annual_uncertainty
    
}).reset_index()
src_totals['mass_wasted_uncertainty_upper_bound'] = src_totals['mass_wasted'] + src_totals['mass_wasted_uncertainty']
src_totals['mass_wasted_uncertainty_lower_bound'] = src_totals['mass_wasted'] - src_totals['mass_wasted_uncertainty']

src_totals['mass_wasted_rate_uncertainty_upper_bound'] = src_totals['mass_wasted_rate'] + src_totals['mass_wasted_rate_uncertainty']
src_totals['mass_wasted_rate_uncertainty_lower_bound'] = src_totals['mass_wasted_rate'] - src_totals['mass_wasted_rate_uncertainty']

# +
### Combine 3 charts into one
### 
### Each chart requires error bars. 
###
###


### Time-series of mass wasted
###
chart_normal = alt.Chart(src_normal).mark_bar(opacity=0.5).encode(
    alt.X('Start Date:T'),
    alt.X2('End Date:T'),
    alt.Y('mass_wasted_rate', axis=alt.Axis(
#         format=".0e"
    ))
).properties(
    width=150,
    height=150
)

chart_normal_bars = alt.Chart(src_normal).mark_bar(color='black', width=2).encode(
    alt.X('uncertainty_date:T', title='Date'),
    alt.Y('mass_wasted_rate_uncertainty_lower_bound:Q', title = 'Mass Wasted Rate (m^3/year)', axis=alt.Axis(
#         format=".0e"
    )),
    alt.Y2('mass_wasted_rate_uncertainty_upper_bound:Q')
).properties(
    width=150,
    height=150
)

chart_normal_with_bars = (chart_normal + chart_normal_bars).facet(
    column='name'
).resolve_scale(
    y='independent'
)

### Net mass wasted rate measurements, two ways
###
chart_total_rates = alt.Chart(src_totals).mark_bar(opacity=0.5).encode(
    alt.X('Measurement:N', title='Net mass wasted method'),
    alt.Y('mass_wasted_rate:Q', title='Mass wasted rate (m^3/year)'),
).properties(
    width=150,
    height=150
)

chart_total_rates_bars = alt.Chart(src_totals).mark_bar(color='black', width=5).encode(
    alt.X('Measurement:N'),
    alt.Y('mass_wasted_rate_uncertainty_lower_bound:Q'),
    alt.Y2('mass_wasted_rate_uncertainty_upper_bound:Q'),
).properties(
    width=150,
    height=150
)
chart_total_rates_with_bars = (chart_total_rates+chart_total_rates_bars).facet(
    column='name',
).resolve_scale(
    y='independent'
)

### Net mass wasted measurements, two ways
###
chart_total = alt.Chart(src_totals).mark_bar(opacity=0.5).encode(
    alt.X('Measurement:N', title='Net mass wasted method'),
    alt.Y('mass_wasted:Q', title='Mass wasted (m^3)'),
).properties(
    width=150,
    height=150
)

chart_total_bars = alt.Chart(src_totals).mark_bar(color='black', width=5).encode(
    alt.X('Measurement:N'),
    alt.Y('mass_wasted_uncertainty_lower_bound:Q'),
    alt.Y2('mass_wasted_uncertainty_upper_bound:Q'),
).properties(
    width=150,
    height=150
)
chart_total_with_bars = (chart_total+chart_total_bars).facet(
    column='name',
).resolve_scale(
    y='independent'
)

chart_normal_with_bars & chart_total_rates_with_bars & chart_total_with_bars

# -

# ## Hillslope Erosion

# +
# NOTE: no need to groupby and sum across polygons, Altair will do this for us

src_normal = mass_wasted_df[
    (~mass_wasted_df.fluvial) & (mass_wasted_df.Measurement == 'Normal')
]
src_normal = src_normal.groupby(['name', 'Start Date', 'End Date']).sum().reset_index()

chart_normal = alt.Chart(src_normal).mark_bar().encode(
    alt.X('Start Date:T'),
    alt.X2('End Date:T'),
    alt.Y('mass_wasted_rate', axis=alt.Axis(
#         format=".0e"
    ))
).properties(
    width=150,
    height=150
).facet(
    row='name',
).resolve_scale(
    y='independent'
)

src_totals = mass_wasted_df[
    (~mass_wasted_df.fluvial) & (mass_wasted_df.Measurement != 'Normal')
]
src_totals = src_totals.groupby(['name', 'Measurement', 'Start Date', 'End Date']).sum().reset_index()

chart_totals_rate_and_volume = alt.Chart(src_totals).transform_fold(
    ['mass_wasted_rate', 'mass_wasted']
).mark_bar().encode(
    alt.X('Measurement:N'),
    alt.Y('value:Q', axis=alt.Axis(
#         format=".0e"
    ))
).properties(
    width=150,
    height=150
).facet(
    row='name',
    column='key:N'
).resolve_scale(
    y='independent'
)

(chart_normal | chart_totals_rate_and_volume).properties(title='Mass wasted by hillslope processes')
# -

# ## Combined Erosion

# +
# NOTE: no need to groupby and sum across polygons, Altair will do this for us

src_normal = mass_wasted_df[
    mass_wasted_df.Measurement == 'Normal'
]
src_normal = src_normal.groupby(['name', 'Start Date', 'End Date']).sum().reset_index()

chart_normal = alt.Chart(src_normal).mark_bar().encode(
    alt.X('Start Date:T'),
    alt.X2('End Date:T'),
    alt.Y('mass_wasted_rate', axis=alt.Axis(
#         format=".0e"
    ))
).properties(
    width=150,
    height=150
).facet(
    row='name',
).resolve_scale(
    y='independent'
)

src_totals = mass_wasted_df[
    mass_wasted_df.Measurement != 'Normal'
]
src_totals = src_totals.groupby(['name', 'Measurement', 'Start Date', 'End Date']).sum().reset_index()

chart_totals_rate_and_volume = alt.Chart(src_totals).transform_fold(
    ['mass_wasted_rate', 'mass_wasted']
).mark_bar().encode(
    alt.X('Measurement:N'),
    alt.Y('value:Q', axis=alt.Axis(
#         format=".0e"
    ))
).properties(
    width=150,
    height=150
).facet(
    row='name',
    column='key:N'
).resolve_scale(
    y='independent'
)

(chart_normal | chart_totals_rate_and_volume).properties(title='Mass wasted by all processes')


# -

# # Prep data for plotting DOD maps
#
# Prepare a few functions that serve up data for a specified valley

def get_dods(valley_name, threshold=None):
    """
    Return list of DODs.
    """
    dod_dataset = dod_dataset_dict[valley_name]
    
    if threshold:
        dods = []
        for data_var in dod_dataset.data_vars:
            data = dod_dataset[data_var]
            data = data.where(np.abs(data) > threshold, 0.0)
            dods.append(data)
        return dods
    else:
        return dods


def get_polys_to_plot(valley_name):
    "Return geodataframe of polygons."
    hillslope_polys_to_plot = polygons[polygons.name == valley_name]
    hillslope_polys_to_plot = hillslope_polys_to_plot[~hillslope_polys_to_plot.fluvial]
    fluvial_polys_to_plot =  polygons[polygons.name == valley_name]
    fluvial_polys_to_plot = fluvial_polys_to_plot[fluvial_polys_to_plot.fluvial]

    hillslope_erosion_gvpolys = gv.Polygons(
        hillslope_polys_to_plot.to_crs(epsg=4326), 
        vdims=['name']
    ).opts(
        projection=cartopy.crs.UTM(zone=10),
        fill_alpha=0.0,
        line_color='red'
    )
    
    fluvial_erosion_gvpolys = gv.Polygons(
        fluvial_polys_to_plot.to_crs(epsg=4326), 
        vdims=['name']
    ).opts(
        projection=cartopy.crs.UTM(zone=10),
        fill_alpha=0.0,
        line_color='blue'
    )
    return hillslope_erosion_gvpolys, fluvial_erosion_gvpolys


len(colors), len(levels)

# +
colors = [
    '#5e3c99',
    '#8068b0',
    '#a195c7',
    '#c0bada',
    '#dcd9e9',
    '#f7f7f7',
    '#fadebc',
    '#fcc581',
    '#f9a74f',
    '#f08428',
    '#e66101'
]
levels = [
    np.inf,
    30,
    12,
    6.66,
    3.33,
    1.3,
    -1.3,
    -3.33,
    -6.66,
    -12,
    -30,
    -np.inf
]

def plot_multiple_diff_maps(dods, hillslope_polys, fluvial_polys):
    def get_diff_map(dod, hillslope_polys, fluvial_polys):
        return rasterize(dod.hvplot.image(
                'x', 
                'y', 
                color_key=levels, 
#                 width=500,
            #     geo=True,
            #     project=True, 
            #     rasterize=True,
            #     dynamic=True,
            #     crs=str(dod.rio.crs)
            ).opts(title=dod.name, aspect=1)
                  ) * hillslope_polys * fluvial_polys

    combined_diff_map = get_diff_map(dods[0], hillslope_polys, fluvial_polys)
    for dod in dods[1:]:
        combined_diff_map = combined_diff_map + get_diff_map(dod, hillslope_polys, fluvial_polys)
    return combined_diff_map


# -

def plot(name):
    return plot_multiple_diff_maps(
        get_dods(name, threshold=1.3), 
        *get_polys_to_plot(name)
    )


valleys_metadata_dict.keys()

# +
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, BoundaryNorm
import matplotlib as mpl
colors = [
    '#5e3c99',
    '#8068b0',
    '#a195c7',
    '#c0bada',
    '#dcd9e9',
    '#f7f7f7',
    '#fadebc',
    '#fcc581',
    '#f9a74f',
    '#f08428',
    '#e66101'
]
colors.reverse()
levels = [
#     -60, #-np.inf,
    -30,
    -12,
    -6.66,
    -3.33,
    -1.3,
    1.3,
    3.33,
    6.66,
    12,
    30,
#     60, #np.inf,
]
cmap = (mpl.colors.ListedColormap(colors)
#         .with_extremes(over='red', under='blue')
       )

cmap = ListedColormap(colors)
cmap.set_bad(color="grey")
boundaries = levels
norm = BoundaryNorm(boundaries, cmap.N, clip=False, extend='max')

# -

len(colors),len(levels)

import matplotlib.pyplot as plt
def plot_dod_dataset(dataset, valley_name):
    fig, axes = plt.subplots(1, len(dataset.data_vars), 
                             figsize=(24,6), 
                             sharex=True, sharey=True) # Create a figure with a single axes.
    for ax, var in zip(axes, dataset.data_vars):
        im = dataset[var].plot.imshow(ax=ax, cmap=cmap, norm=norm, add_colorbar=False)
        erosion_polygons[erosion_polygons.name==valley_name].plot(ax=ax, facecolor="none", edgecolor="black")
        fluvial_erosion_polygons[fluvial_erosion_polygons.name==valley_name].plot(ax=ax, facecolor="none", edgecolor="black")
        ax.set_title(var)
        ax.set_xlabel(None)
        ax.set_ylabel(None)
    fig.text(
        0.5, 0.04, 
        'X (UTM 10N)', ha='center')
    fig.text(
        -0.01, 0.5, 
        'Y (UTM 10N)', va='center', rotation='vertical')
    plt.tight_layout()


plot_dod_dataset(dod_dataset_dict['Deming'], 'Deming')

plot_dod_dataset(dod_dataset_dict['Easton'], 'Easton')

plot_dod_dataset(dod_dataset_dict['Coleman/Roosevelt'], 'Coleman/Roosevelt')

plot_dod_dataset(dod_dataset_dict['Mazama'], 'Mazama')

plot_dod_dataset(dod_dataset_dict['Rainbow'], 'Rainbow')

plot_dod_dataset(dod_dataset_dict['Park'], 'Park')

plot_dod_dataset(dod_dataset_dict['Squak'], 'Squak')

plot_dod_dataset(dod_dataset_dict['Talum'], 'Talum')

plot_dod_dataset(dod_dataset_dict['Thunder'], 'Thunder')




