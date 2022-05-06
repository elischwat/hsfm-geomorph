# ---
# jupyter:
#   jupytext:
#     tSext_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2

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
from rasterio.enums import Resampling

from dateutil.relativedelta import relativedelta
from shapely.geometry.collection import GeometryCollection
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib as mpl

pd.options.mode.chained_assignment = None  # default='warn'

# +

# requires dod_tools.py file in the same directory as this notebook
import dod_tools
# -

# # Load and organize data

# ## DEMs

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
#     
# Using the QGIS mt_baker_mass_wasted project, I observed the coverage of all DEMs for each glacial valley:

# Without 1950s dataset

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
        'years': [1970, 1979, 2015], #[1970, 1977, 1979, 2015]
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

# With 1950 dataset:

valleys_metadata_dict = {
    'Deming': {
        'years': [1950, 1970, 1979, 1991, 2015],
        'max_coverage_year': 1950
    },
    'Thunder': {
        'years': [1950, 1970, 1979, 2015],
        'max_coverage_year': 1950
    },
    'Coleman/Roosevelt': {
        'years': [1950, 1970, 1979, 1990, 2015],
        'max_coverage_year': 1950
    },
    'Mazama': {
        'years': [1950, 1970, 1979, 2015], #removed 1977 after the fact due to apparent systematic error
        'max_coverage_year': 1950
    },
    'Rainbow': {
        'years': [1950, 1970, 1979, 1991, 2015],
        'max_coverage_year': 1950
    },
    'Park': {
        'years': [1950, 1970, 1979, 1991, 2015],
        'max_coverage_year': 1950
    },
    'Boulder': {
        'years': [1950, 1970, 1979, 1987, 1991, 2015],
        'max_coverage_year': 1950
    },
    'Talum': {
        'years': [1950, 1970, 1977, 1979, 1987, 1991, 2015],
        'max_coverage_year': 1950
    },
    'Squak': {
        'years': [1950, 1970, 1977, 1979, 1991, 2015],
        'max_coverage_year': 1950
    },
    'Easton': {
        'years': [1950, 1970, 1979, 1991, 2015],
        'max_coverage_year': 1950
    },
}

years = set()
for v in valleys_metadata_dict.values():
    years = years.union(set(v['years']))
years

# Add pairs of years to the valleys_metadata_dict
def get_dod_pairs_from_list(list_of_years):
    return [
        (list_of_years[i], list_of_years[i+1]) for i in range(0, len(list_of_years)-1)
    ]    
for k, v in valleys_metadata_dict.items():
    valleys_metadata_dict[k]['dod pairs'] = get_dod_pairs_from_list(v['years'])
    valleys_metadata_dict[k]['bounding dod pair'] = (v['years'][0], v['years'][-1])
del k,v
valleys_metadata_dict

# ### Create DEM xarray.Dataset

# Create a dictionary of DEM file paths by year

# Without 1950s dataset

# +
dem_files_dict = {
    1970: "/data2/elilouis/generate_ee_dems_baker/mixed_timesift/individual_clouds/70_9.0/dem.tif",
    1974: "/data2/elilouis/generate_ee_dems_baker/mixed_timesift/individual_clouds/74_8.0/dem.tif",
    1977: "/data2/elilouis/generate_ee_dems_baker/mixed_timesift/individual_clouds/77_9.0/dem.tif",
    1979: "/data2/elilouis/generate_ee_dems_baker/mixed_timesift/individual_clouds/79_10.0/dem.tif",
    1987: "/data2/elilouis/generate_ee_dems_baker/mixed_timesift/individual_clouds/87_8.0/dem.tif",
    1990: "/data2/elilouis/generate_ee_dems_baker/mixed_timesift/individual_clouds/90_9.0/dem.tif",
    1991: "/data2/elilouis/generate_ee_dems_baker/mixed_timesift/individual_clouds/91_9.0/dem.tif",
    2015: "/data2/elilouis/hsfm-geomorph/data/reference_dem_highres/baker/raw_tifs/baker_2015/2015.tif"
}
dem_dict = {
    k: rix.open_rasterio(v, masked=True, chunk=1000) for k,v in dem_files_dict.items()
    
}
# -

# With 1950s dataset

# +
dem_files_dict = {
    1950: "/data2/elilouis/timesift/baker-ee/mixed_timesift/individual_clouds/50_9.0_2.0/cluster0/0/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/EE_baker_1950_corrected_dem_align/EE_baker_1950_corrected_baker_2015_utm_m_nuth_x-2.31_y-2.09_z+1.80_align.tif",
    1970: "/data2/elilouis/generate_ee_dems_baker/mixed_timesift/individual_clouds/70_9.0/dem.tif",
    1974: "/data2/elilouis/generate_ee_dems_baker/mixed_timesift/individual_clouds/74_8.0/dem.tif",
    1977: "/data2/elilouis/generate_ee_dems_baker/mixed_timesift/individual_clouds/77_9.0/dem.tif",
    1979: "/data2/elilouis/generate_ee_dems_baker/mixed_timesift/individual_clouds/79_10.0/dem.tif",
    1987: "/data2/elilouis/generate_ee_dems_baker/mixed_timesift/individual_clouds/87_8.0/dem.tif",
    1990: "/data2/elilouis/generate_ee_dems_baker/mixed_timesift/individual_clouds/90_9.0/dem.tif",
    1991: "/data2/elilouis/generate_ee_dems_baker/mixed_timesift/individual_clouds/91_9.0/dem.tif",
    2015: "/data2/elilouis/hsfm-geomorph/data/reference_dem_highres/baker/raw_tifs/baker_2015/2015.tif"
}
dem_dict = {
    k: rix.open_rasterio(v, masked=True, chunk=1000) for k,v in dem_files_dict.items()
    
}
# -

# Load reference DEM and regrid to desired resolution. This grid will be used to grid all DEMs.
# ### NOTE: this is where resolution (and processing speed) is determined 

# %%time
dem_dict[2015] = dem_dict[2015].rio.reproject(
    dem_dict[2015].rio.crs, 
    resolution=(1,1),
    resampling=Resampling.cubic
)

# Create DEM dataset

# %%time
dem_dataset = dod_tools.create_dem_dataset(dem_dict, 2015, resampling_alg=Resampling.cubic)


# Read DEM Dataset

# ### Define function for calculating uncertainty
#
# Mass wasted volumetric uncertainty using equation 22 from Scott Anderson, assuming random error and systematic error, but no spatially correlated error.
#
# $$\sigma_v = n L^2 \sqrt{\frac{\sigma_{rms}^2}{n} + \sigma_{sys}^2 + \frac{\sigma_{sc}^2}{n} \frac{\pi a_i^2}{5L^2}}$$
#
# where $n$ is number of pixels used in measurement, $L^2$ is the pixel area, $\sigma_{rms}$ is the root mean squared error of the DOD, and $\sigma_{sys}$ is the systematic error of the DOD which we assume is equal to the mean error, $\sigma_{sc}$ is the partial sill of the fitted spherical semivariogram, and $a_i$ is the range of the fitted spherical semivariogram.

def volumetric_uncertainty(n_pixels, pixel_area, rmse, me, p_sill, variogram_range):
    return n_pixels*pixel_area*np.sqrt(
        rmse**2/n_pixels + 
        me**2 + 
        (p_sill**2/n_pixels)*(
            (np.pi*variogram_range**2)/(5*pixel_area))
    )


# ### Add Uncertainty related attributes to DEM xarray.Dataset
# Calculate RMSE and add as an attribute to the dem_dataset DataArray, also plot the Ground Control Area difference distributions

# ## NOTE: choose GCAs which will determine error HERE

static_geoms_gdf = gpd.read_file('/data2/elilouis/hsfm-geomorph/data/mt_baker_mass_wasted/gcas_expanded.geojson')
dem_dataset = dod_tools.assign_dem_uncertainty_values(
    dem_dataset, 
    static_geoms_gdf['geometry'],
    2015
)

df = pd.DataFrame({
    'RMSE': [dem_dataset[var].attrs.get('RMSE') for var in dem_dataset],
    'Year': [datetime(year=int(var), month=1, day=1) for var in dem_dataset],
    'Median': [dem_dataset[var].attrs.get('Median') for var in dem_dataset],
    'Mean': [dem_dataset[var].attrs.get('Mean') for var in dem_dataset],
    'Standard Deviation': [dem_dataset[var].attrs.get('Standard Deviation') for var in dem_dataset],
    'NMAD': [dem_dataset[var].attrs.get('NMAD') for var in dem_dataset],
    '84th Percentile': [dem_dataset[var].attrs.get('84th Percentile') for var in dem_dataset],
    '16th Percentile': [dem_dataset[var].attrs.get('16th Percentile') for var in dem_dataset],
    'Variance': [dem_dataset[var].attrs.get('Variance') for var in dem_dataset],
    'Average Variogram Range': [dem_dataset[var].attrs.get('Average Variogram Range') for var in dem_dataset],
    'Average Variogram Variance': [dem_dataset[var].attrs.get('Average Variogram Variance') for var in dem_dataset]
})

src = df[~df.Year.isin([
    datetime(2015,1,1),
    datetime(1974,1,1)
])]
alt.Chart(src).transform_fold(
    ['NMAD', 'RMSE', 'Mean', 'Variance', 'Average Variogram Range']
).mark_bar().encode(
    alt.X('Year:T', title='Date'),
    alt.Y('value:Q', title=None)
).properties(
    height=75,
).facet(
    row=alt.Row('key:N', title=None),
).resolve_scale(y='independent').properties(
    title='DEM Error Metrics'
)

# +
src = df[~df.Year.isin([
    datetime(2015,1,1),
    datetime(1974,1,1)
])]
median_points = alt.Chart(src).mark_circle(size=100).encode(
    alt.X('Year:T'),
    alt.Y('Median:Q'),
)

spread_bars = alt.Chart(src).mark_bar(size=2, opacity=.75).encode(
    alt.X('Year:T', title='Date'),
    alt.Y('16th Percentile:Q', scale=alt.Scale(domain=(-6.5,6.5))),
    alt.Y2('84th Percentile:Q', title=['DEM Error (m)'])
)

spread_bar_bottoms = alt.Chart(src).mark_tick(size=10, opacity=.75).encode(
    alt.X('Year:T', title='Date'),
    alt.Y('16th Percentile:Q'),
)

spread_bar_tops = alt.Chart(src).mark_tick(size=10, opacity=.75).encode(
    alt.X('Year:T', title='Date'),
    alt.Y('84th Percentile:Q'),
)

(median_points + spread_bars + spread_bar_bottoms + spread_bar_tops).properties(
    title = 'DEM Error, Median and 16-84% Spread',
    height=150
)


# + [markdown] tags=[]
# ## DoDs
# -

# ### Create DoD xarray.Dataset

# Get all DOD year combinations

# +
def flatten(t):
    return [item for sublist in t for item in sublist]

unique_dod_year_combinations = set(flatten([ 
    k['dod pairs'] for k in valleys_metadata_dict.values()
]))
unique_dod_year_combinations = {(years[0], years[1]) for years in unique_dod_year_combinations}
[x for x in unique_dod_year_combinations]
# -

unique_bounding_dod_year_combinations = set([ 
    k['bounding dod pair'] for k in valleys_metadata_dict.values()
])
unique_bounding_dod_year_combinations

dod_dataset = xr.Dataset()
for yr_range in unique_bounding_dod_year_combinations.union(unique_dod_year_combinations):
    print(f"Generating DOD: {str(yr_range)}")
    dod_dataset[yr_range] = dem_dataset[yr_range[1]] - dem_dataset[yr_range[0]]

# ### Crop out glaciers from DODs

# Load glacier polygons

glacier_polygons = gpd.read_file(
    '/data2/elilouis/hsfm-geomorph/data/mt_baker_mass_wasted/glacier_polygons_combined.geojson'
)
# fill in empty geometries...
glacier_polygons['geometry'] = glacier_polygons.geometry.apply(lambda x: x if x else GeometryCollection())
glacier_polygons = glacier_polygons.to_crs(epsg=32610)
glacier_polygons = glacier_polygons[~glacier_polygons.geometry.is_empty]

# Iterate over DOD datasets, and iterate over each DOD/data variable within each dataset, masking out glacier pixels using both years.

for yr1, yr2 in dod_dataset.data_vars:
    relevant_polygons = glacier_polygons[glacier_polygons.year.astype(int).isin([yr1, yr2])]
    print(f"For years {yr1} and { yr2}, found {len(relevant_polygons)} glacier polygons from years {relevant_polygons.year.unique()} for masking.")
    dod_dataset[(yr1, yr2)] = dod_dataset[(yr1, yr2)].rio.clip(
        glacier_polygons.geometry, invert=True
    )

# + [markdown] tags=[]
# ### Save DOD static area datasets to disk
# -

dod_dataset_clipped = dod_dataset.rio.clip(
    static_geoms_gdf.geometry.dropna()
)

# ### Note: Input an output path for files containing DODs masked to isolate static area pixels

static_area_dod_tif_output_path='/data2/elilouis/hsfm-geomorph/data/mt_baker_mass_wasted/static_area_dods/'

if not os.path.exists(static_area_dod_tif_output_path):
    os.makedirs(static_area_dod_tif_output_path)


for var in dod_dataset_clipped:
    years_string = '_'.join([str(f) for f in var])

    new_ds = dod_dataset_clipped[var]
    new_ds.name = years_string
    
    new_path = os.path.join(
            static_area_dod_tif_output_path,
            years_string + '.tif'
        )
    print(f'Saving to {new_path}')
    new_ds.rio.to_raster(
        new_path,
        driver='GTiff'
    )

# ls -lah /data2/elilouis/hsfm-geomorph/data/mt_baker_mass_wasted/static_area_dods/

# ### Calculate uncertainty metrics

dod_dataset = dod_tools.assign_dod_uncertainty_statistics(dod_dataset, static_geoms_gdf.geometry)

# ### Manually add in spatially-correlated error variables

# Without 1950

sp_corr_error_params = {
    (1970, 1977): {"P Sill": 0.2501 , "Range": 72.1294},
    (1970, 1979): {"P Sill": 0.3384 , "Range": 47.9692},
    (1970, 2015): {"P Sill": 0.1051 , "Range": 44.017},
    (1977, 1979): {"P Sill": 0.5308 , "Range": 66.4927},
    (1979, 1987): {"P Sill": 0.9463 , "Range": 108.9814},
    (1979, 1990): {"P Sill": 0.5886 , "Range": 33.3059},
    (1979, 1991): {"P Sill": 0.5472 , "Range": 61.857},
    (1979, 2015): {"P Sill": 0.4309 , "Range": 40.8339},
    (1987, 1991): {"P Sill": 0.2718 , "Range": 83.6205},
    (1990, 2015): {"P Sill": 0.5144 , "Range": 40.6104},
    (1991, 2015): {"P Sill": 0.389 , "Range":  62.4523}
}

# With 1950

sp_corr_error_params = {
    (1950, 1970): {"P Sill": 6.2326, "Range": 20.8186},
    (1950, 2015): {"P Sill": 13.3341, "Range": 909.646},
    (1970, 1977): {"P Sill": 0.2501 , "Range": 72.1294},
    (1970, 1979): {"P Sill": 0.3384 , "Range": 47.9692},
    (1977, 1979): {"P Sill": 0.5308 , "Range": 66.4927},
    (1979, 1987): {"P Sill": 0.9463 , "Range": 108.9814},
    (1979, 1990): {"P Sill": 0.5886 , "Range": 33.3059},
    (1979, 1991): {"P Sill": 0.5472 , "Range": 61.857},
    (1979, 2015): {"P Sill": 0.4309 , "Range": 40.8339},
    (1987, 1991): {"P Sill": 0.2718 , "Range": 83.6205},
    (1990, 2015): {"P Sill": 0.5144 , "Range": 40.6104},
    (1991, 2015): {"P Sill": 0.389 , "Range":  62.4523}
}

for var in dod_dataset.data_vars:
    dod_dataset[var].attrs['Variogram Range'] = sp_corr_error_params[var]['Range']
    dod_dataset[var].attrs['Variogram Partial Sill'] = sp_corr_error_params[var]['P Sill']

# ### Look at Uncertainty Params

df = pd.DataFrame({
    'RMSE': [dod_dataset[var].attrs.get('RMSE') for var in dod_dataset],
    'Year': [var for var in dod_dataset],
    'Median': [dod_dataset[var].attrs.get('Median') for var in dod_dataset],
    'Mean': [dod_dataset[var].attrs.get('Mean') for var in dod_dataset],
    'Standard Deviation': [dod_dataset[var].attrs.get('Standard Deviation') for var in dod_dataset],
    'NMAD': [dod_dataset[var].attrs.get('NMAD') for var in dod_dataset],
    '84th Percentile': [dod_dataset[var].attrs.get('84th Percentile') for var in dod_dataset],
    '16th Percentile': [dod_dataset[var].attrs.get('16th Percentile') for var in dod_dataset],
    'Variance': [dod_dataset[var].attrs.get('Variance') for var in dod_dataset],
    'Variogram Range': [dod_dataset[var].attrs.get('Variogram Range') for var in dod_dataset],
    'Variogram Partial Sill': [dod_dataset[var].attrs.get('Variogram Partial Sill') for var in dod_dataset]
})

src = df.copy()
src['Year'] = src['Year'].apply(lambda x: f"{x[0]}-{x[1]}")
alt.Chart(src).transform_fold(
    ['NMAD', 'RMSE', 'Mean', 'Variance', 'Variogram Range', 'Variogram Partial Sill']
).mark_bar().encode(
    alt.X('Year:O', title='Date', axis=alt.Axis(labelAngle=-70)),
    alt.Y('value:Q', title=None)
).properties(
    height=75, width=400
).facet(
    row=alt.Row('key:N', title=None),
).resolve_scale(y='independent').properties(
    title='DoD Error Metrics'
)

src = df.copy()
src['Year'] = src['Year'].apply(lambda x: f"{x[0]}-{x[1]}")
alt.Chart(src).transform_fold(
    ['RMSE', 'Mean', 'Variogram Range', 'Variogram Partial Sill']
).mark_bar().encode(
    alt.X('Year:O', title='Date', axis=alt.Axis(labelAngle=-70)),
    alt.Y('value:Q', title=None)
).properties(
    height=75, width=400
).facet(
    row=alt.Row('key:N', title=None),
).resolve_scale(y='independent')

# +
src = df.copy()
src['Year'] = src['Year'].apply(lambda x: f"{x[0]}-{x[1]}")
median_points = alt.Chart(src).mark_circle(size=100).encode(
    alt.X('Year:O'),
    alt.Y('Median:Q'),
)

spread_bars = alt.Chart(src).mark_bar(size=2, opacity=.75).encode(
    alt.X('Year:O', title='Date', axis=alt.Axis(labelAngle=-70)),
    alt.Y('16th Percentile:Q', scale=alt.Scale(domain=(-6.5,6.5))),
    alt.Y2('84th Percentile:Q', title=['DEM Error (m)'])
)

spread_bar_bottoms = alt.Chart(src).mark_tick(size=10, opacity=.75).encode(
    alt.X('Year:O', title='Date'),
    alt.Y('16th Percentile:Q'),
)

spread_bar_tops = alt.Chart(src).mark_tick(size=10, opacity=.75).encode(
    alt.X('Year:O', title='Date'),
    alt.Y('84th Percentile:Q'),
)

(median_points + spread_bars + spread_bar_bottoms + spread_bar_tops).properties(
    title = 'DoD Error, Median and 16-84% Spread',
    height=150, width=400
)
# -

# ## Calculate mass wasted

# ### Load Erosion polygons

erosion_polygons = gpd.read_file(
    '/data2/elilouis/hsfm-geomorph/data/mt_baker_mass_wasted/erosion_polygons.geojson'
).assign(fluvial=False)
fluvial_erosion_polygons = gpd.read_file(
    '/data2/elilouis/hsfm-geomorph/data/mt_baker_mass_wasted/fluvial_erosion_polygons.geojson'
).assign(fluvial=True)
polygons = pd.concat([erosion_polygons, fluvial_erosion_polygons]).reset_index(drop=True)

# ### Calculate mass wasted in each the hand drawn erosion polygons

# For each polygon, calculate mass wasted using each possible DOD. DODs are generated for each pair of DEMs available for the given glacial valley. The `name` column is used to match between the DEM dictionary and the erosion polygon dataframes.
#
# Calculate mass wasted using a shared NaN/nodata mask across all DODs for one erosion polygon and also calculate using as much data is available in each DOD (probably will lead to a different number of valid pixels for mass wasted calculations for individual polygons for different DoDs).

polygons.columns

# %%time
mass_wasted_df_shared_mask = dod_tools.calculate_mass_wasted(
    dod_dataset, 
    polygons, 
    carry_columns = ['name', 'fluvial', 'id'], 
    carry_attrs = ['RMSE', 'Mean', "Variance", "Variogram Range", "Variogram Partial Sill"], 
    valleys_metadata_dict = valleys_metadata_dict,
    share_mask = True
)
mass_wasted_df_not_shared_mask = dod_tools.calculate_mass_wasted(
    dod_dataset, 
    polygons, 
    carry_columns = ['name', 'fluvial', 'id'], 
    carry_attrs = ['RMSE', 'Mean', "Variance", "Variogram Range", "Variogram Partial Sill"], 
    valleys_metadata_dict = valleys_metadata_dict,
    share_mask = False
)
mass_wasted_df_shared_mask['shared_mask'] = True
mass_wasted_df_not_shared_mask['shared_mask'] = False
mass_wasted_df = pd.concat([mass_wasted_df_not_shared_mask, mass_wasted_df_shared_mask])

# If the end/start years are 1970/2015, call the measurement "bounding" otherwise "normal

# Without 1950s dataset

BOUNDING_PAIR_TUPLE = (1970, 2015)

# With 1950s dataset

BOUNDING_PAIR_TUPLE = (1950, 2015)

mass_wasted_df['Measurement'] = mass_wasted_df['dataset_name'].apply(
    lambda x: 'Bounding' if x == BOUNDING_PAIR_TUPLE else "Normal"
)

# Convert the fluvial column from int (0/1) to bool, make sure all values in id column have a value, parse years and bounding (or not) from dataset_name column, and lastly assign dummy pd.datetime dates for the `start_year` and `end_year` data columns.
#
# I do this so Altair can do a better job plotting.
#
# Note that I only have to do the parsing of "dataset_name" column to start and end years because of what i do in the loop above, and how the DODs are organized into a dataset with names

# +

mass_wasted_df['fluvial'] = mass_wasted_df['fluvial'].astype(bool)
mass_wasted_df['id'] = mass_wasted_df['id'].apply(lambda x: 0 if not x else x)

mass_wasted_df['start_year'] = mass_wasted_df['dataset_name'].apply(lambda x: x[0])
mass_wasted_df['end_year'] = mass_wasted_df['dataset_name'].apply(lambda x: x[1])

mass_wasted_df['Start Date'] = mass_wasted_df['start_year'].apply(lambda x: datetime(int(x), 1, 1))
mass_wasted_df['End Date'] = mass_wasted_df['end_year'].apply(lambda x: datetime(int(x), 1, 1))

mass_wasted_df['N Years'] = mass_wasted_df.apply(lambda x: relativedelta(x['End Date'], x['Start Date']).years, axis=1)
mass_wasted_df
# -

mass_wasted_df = mass_wasted_df[mass_wasted_df['n_valid_pixels'] != 0]

# ### Calculate volumetric uncertainty in mass wasted associated with each erosion polygon

mass_wasted_df['mass_wasted_uncertainty'] = mass_wasted_df.apply(
    lambda x: volumetric_uncertainty(x.n_valid_pixels, x.pixel_area, x['RMSE'], x['Mean'], x['Variogram Partial Sill'], x['Variogram Range']),
    axis=1
)

mass_wasted_df.head(3)

# ### Drop unnecesary statistical columns now

mass_wasted_df = mass_wasted_df[[
    'dataset_name', 'name', 
    'Start Date', 'End Date', 
    'pixel_area', 'id', 'fluvial', 'Measurement', 'n_valid_pixels', 'shared_mask',
    'mass_wasted', 'mass_wasted_uncertainty', 
]]

mass_wasted_df.head()


def sum_in_quadrature(ls):
    return np.sqrt((pd.Series(ls)**2).sum())


# ### Calculate mass wasted for each valley,
#
# ... by DoD/Timespan, by the type of erosion represented (fluvial/hillslope), by the measurement type (interval or bounding), and by whether the calculation was made with a shared nodata mask or not

# +
mass_wasted_df = mass_wasted_df.groupby([
    'dataset_name', 
    'name',
    'Start Date', 
    'End Date', 
    'pixel_area',
    'fluvial',
    'Measurement',
    'shared_mask'
]).agg({
    'mass_wasted': 'sum',
    'mass_wasted_uncertainty': sum_in_quadrature,
    
}).reset_index()
mass_wasted_df
# -

# ### Calculate mass wasted by sum-of-intervals
#
# to compare to the bounding calculations,

all_normal_measurements = mass_wasted_df[mass_wasted_df.Measurement == 'Normal']
sum_of_intervals_measurements = all_normal_measurements.groupby(['name', 'fluvial', 'shared_mask']).agg({
    'Start Date': 'min',
    'End Date': 'max',
    'mass_wasted': 'sum',
    'mass_wasted_uncertainty': sum_in_quadrature,
}).reset_index()
sum_of_intervals_measurements['Measurement'] = 'sum of intervals'

mass_wasted_df = pd.concat([mass_wasted_df, sum_of_intervals_measurements])

# # Plot: Mass wasted

# ## Overview of Erosion

# ### All of Baker

overview_src = mass_wasted_df[
    (mass_wasted_df.Measurement == 'Bounding') & ~(mass_wasted_df.shared_mask)
]

overview_src = overview_src.groupby('fluvial').agg({
        'mass_wasted': 'sum',
        'mass_wasted_uncertainty': sum_in_quadrature,
}).reset_index()
overview_src['Type'] = overview_src['fluvial'].apply(lambda x: 'Fluvial' if x else 'Hillslope')
overview_src['mass_wasted_uncertainty_upper_bound'] = overview_src['mass_wasted'] + overview_src['mass_wasted_uncertainty']
overview_src['mass_wasted_uncertainty_lower_bound'] = overview_src['mass_wasted'] - overview_src['mass_wasted_uncertainty']

# +
src = overview_src

mass_wasted_bars = alt.Chart(src).mark_bar().encode(
    alt.X('Type', title=None, axis=alt.Axis(labels=False)),
    alt.Y('mass_wasted', title='Mass wasted (cubic meters)'),
    alt.Color(
        'Type',  
        legend=alt.Legend(orient='top'),
         scale=alt.Scale(
             domain=['Fluvial', 'Hillslope'],
            range=['orange', 'red']
         )
     )
)

mass_wasted_errors = alt.Chart(src).mark_bar(color='black', width=2).encode(
    alt.X('Type:N'),
    alt.Y('mass_wasted_uncertainty_lower_bound:Q'),
    alt.Y2('mass_wasted_uncertainty_upper_bound:Q'),
)

mass_wasted_errors_top = alt.Chart(src).mark_tick(color='black', width=6).encode(
    alt.X('Type:N'),
    alt.Y('mass_wasted_uncertainty_upper_bound:Q'),
)

mass_wasted_errors_bottom = alt.Chart(src).mark_tick(color='black', width=8).encode(
    alt.X('Type:N'),
    alt.Y('mass_wasted_uncertainty_lower_bound:Q'),
)

(mass_wasted_bars + mass_wasted_errors 
 + mass_wasted_errors_top + mass_wasted_errors_bottom
)
# -

# ### By Valley

overview_src_byvalley = mass_wasted_df[
    (mass_wasted_df.Measurement == 'Bounding') & ~(mass_wasted_df.shared_mask)
]
overview_src_byvalley['Type'] = overview_src_byvalley['fluvial'].apply(lambda x: 'Fluvial' if x else 'Hillslope')
overview_src_byvalley['mass_wasted_uncertainty_upper_bound'] = overview_src_byvalley['mass_wasted'] + overview_src_byvalley['mass_wasted_uncertainty']
overview_src_byvalley['mass_wasted_uncertainty_lower_bound'] = overview_src_byvalley['mass_wasted'] - overview_src_byvalley['mass_wasted_uncertainty']

# +
src = overview_src_byvalley[~overview_src_byvalley['name'].isin(['Deming', 'Mazama', 'Coleman/Roosevelt', 'Rainbow'])]

mass_wasted_bars = alt.Chart(src).mark_bar().encode(
    alt.X('Type', title=None, axis=alt.Axis(labels=False)),
    alt.Y('mass_wasted', title='Mass wasted (cubic meters)'),
    alt.Color(
        'Type',  
        legend=alt.Legend(orient='top'),
         scale=alt.Scale(
             domain=['Fluvial', 'Hillslope'],
            range=['orange', 'red']
         )
     )
)

mass_wasted_errors = alt.Chart(src).mark_bar(color='black', width=2).encode(
    alt.X('Type:N'),
    alt.Y('mass_wasted_uncertainty_lower_bound:Q'),
    alt.Y2('mass_wasted_uncertainty_upper_bound:Q'),
)

mass_wasted_errors_top = alt.Chart(src).mark_tick(color='black', width=6).encode(
    alt.X('Type:N'),
    alt.Y('mass_wasted_uncertainty_upper_bound:Q'),
)

mass_wasted_errors_bottom = alt.Chart(src).mark_tick(color='black', width=8).encode(
    alt.X('Type:N'),
    alt.Y('mass_wasted_uncertainty_lower_bound:Q'),
)

mass_wasted_smaller_valleys = (mass_wasted_bars + mass_wasted_errors 
 + mass_wasted_errors_top + mass_wasted_errors_bottom
).facet(
    column=alt.Column('name', title=None)
)

# +
src = overview_src_byvalley[overview_src_byvalley['name'].isin(['Deming', 'Mazama', 'Coleman/Roosevelt', 'Rainbow'])]

mass_wasted_bars = alt.Chart(src).mark_bar().encode(
    alt.X('Type', title=None, axis=alt.Axis(labels=False)),
    alt.Y('mass_wasted', title='Mass wasted (cubic meters)'),
    alt.Color(
        'Type',  
        legend=alt.Legend(orient='top'),
         scale=alt.Scale(
             domain=['Fluvial', 'Hillslope'],
            range=['orange', 'red']
         )
     )
)

mass_wasted_errors = alt.Chart(src).mark_bar(color='black', width=2).encode(
    alt.X('Type:N'),
    alt.Y('mass_wasted_uncertainty_lower_bound:Q'),
    alt.Y2('mass_wasted_uncertainty_upper_bound:Q'),
)

mass_wasted_errors_top = alt.Chart(src).mark_tick(color='black', width=6).encode(
    alt.X('Type:N'),
    alt.Y('mass_wasted_uncertainty_upper_bound:Q'),
)

mass_wasted_errors_bottom = alt.Chart(src).mark_tick(color='black', width=8).encode(
    alt.X('Type:N'),
    alt.Y('mass_wasted_uncertainty_lower_bound:Q'),
)

mass_wasted_larger_valleys = (mass_wasted_bars + mass_wasted_errors 
 + mass_wasted_errors_top + mass_wasted_errors_bottom
).facet(
    column=alt.Column('name', title=None)
)
# -

mass_wasted_smaller_valleys | mass_wasted_larger_valleys

# ## Fluvial Erosion

# First we calculate uncertainty in mass wasted columes and then the upper and lower bounds determined by the RMS calculation.

# ### Time series mass wasted

# +
src_normal = mass_wasted_df[
    (mass_wasted_df.fluvial) & (mass_wasted_df.Measurement == 'Normal')
]
src_normal['mass_wasted_uncertainty_upper_bound'] = src_normal['mass_wasted'] + src_normal['mass_wasted_uncertainty']
src_normal['mass_wasted_uncertainty_lower_bound'] = src_normal['mass_wasted'] - src_normal['mass_wasted_uncertainty']

# We need this to plot the error bars in a good place
src_normal['Average Date'] = src_normal['Start Date'] + ((src_normal['End Date'] - src_normal['Start Date']) / 2).dt.ceil('D')

mass_wasted_bars = alt.Chart(src_normal).mark_bar(opacity=0.5).encode(
    alt.X('Start Date:T', title='Date'),
    alt.X2('End Date:T'),
    alt.Y('mass_wasted:Q', title='Mass Wasted (cubic meters)')
).properties(width=150, height=150)

mass_wasted_errors = alt.Chart(src_normal).mark_bar(color='black', width=2).encode(
    alt.X('Average Date:T'),
    alt.Y('mass_wasted_uncertainty_lower_bound'),
    alt.Y2('mass_wasted_uncertainty_upper_bound'),
)

mass_wasted_errors_top = alt.Chart(src_normal).mark_tick(color='black', width=6).encode(
    alt.X('Average Date:T'),
    alt.Y('mass_wasted_uncertainty_upper_bound'),
)

mass_wasted_errors_bottom = alt.Chart(src_normal).mark_tick(color='black', width=8).encode(
    alt.X('Average Date:T'),
    alt.Y('mass_wasted_uncertainty_lower_bound'),
)

(mass_wasted_bars+mass_wasted_errors + mass_wasted_errors_top+ mass_wasted_errors_bottom).facet(
    column='name', row='shared_mask'
).resolve_scale(x='independent', y='independent')

# -

# ### Entire interval mass wasted

# +
src_normal = mass_wasted_df[
    (mass_wasted_df.fluvial) & (mass_wasted_df.Measurement != 'Normal')
]
src_normal['mass_wasted_uncertainty_upper_bound'] = src_normal['mass_wasted'] + src_normal['mass_wasted_uncertainty']
src_normal['mass_wasted_uncertainty_lower_bound'] = src_normal['mass_wasted'] - src_normal['mass_wasted_uncertainty']

# We need this to plot the error bars in a good place
src_normal['Average Date'] = src_normal['Start Date'] + ((src_normal['End Date'] - src_normal['Start Date']) / 2).dt.ceil('D')
src_normal.head()

mass_wasted_bars = alt.Chart(src_normal).mark_bar(opacity=0.5).encode(
    alt.X('Measurement:N'),
#     alt.X2('End Date:T'),
    alt.Y('mass_wasted:Q', title='Mass Wasted (cubic meters)')
).properties(width=150, height=150)

mass_wasted_errors = alt.Chart(src_normal).mark_bar(color='black', width=2).encode(
    alt.X('Measurement:N'),
    alt.Y('mass_wasted_uncertainty_lower_bound'),
    alt.Y2('mass_wasted_uncertainty_upper_bound'),
)

mass_wasted_errors_top = alt.Chart(src_normal).mark_tick(color='black', width=6).encode(
    alt.X('Measurement:N'),
    alt.Y('mass_wasted_uncertainty_upper_bound'),
)

mass_wasted_errors_bottom = alt.Chart(src_normal).mark_tick(color='black', width=8).encode(
    alt.X('Measurement:N'),
    alt.Y('mass_wasted_uncertainty_lower_bound'),
)

(mass_wasted_bars+mass_wasted_errors + mass_wasted_errors_top+ mass_wasted_errors_bottom).facet(
    column='name', row='shared_mask'
).resolve_scale(x='independent', y='independent')
# -

# ## Hillslope Erosion

# ### Time series mass wasted

# +
src_normal = mass_wasted_df[
    ~(mass_wasted_df.fluvial) & (mass_wasted_df.Measurement == 'Normal')
]
src_normal['mass_wasted_uncertainty_upper_bound'] = src_normal['mass_wasted'] + src_normal['mass_wasted_uncertainty']
src_normal['mass_wasted_uncertainty_lower_bound'] = src_normal['mass_wasted'] - src_normal['mass_wasted_uncertainty']

# We need this to plot the error bars in a good place
src_normal['Average Date'] = src_normal['Start Date'] + ((src_normal['End Date'] - src_normal['Start Date']) / 2).dt.ceil('D')

mass_wasted_bars = alt.Chart(src_normal).mark_bar(opacity=0.5).encode(
    alt.X('Start Date:T', title='Date'),
    alt.X2('End Date:T'),
    alt.Y('mass_wasted:Q', title='Mass Wasted (cubic meters)')
).properties(width=150, height=150)

mass_wasted_errors = alt.Chart(src_normal).mark_bar(color='black', width=2).encode(
    alt.X('Average Date:T'),
    alt.Y('mass_wasted_uncertainty_lower_bound'),
    alt.Y2('mass_wasted_uncertainty_upper_bound'),
)

mass_wasted_errors_top = alt.Chart(src_normal).mark_tick(color='black', width=6).encode(
    alt.X('Average Date:T'),
    alt.Y('mass_wasted_uncertainty_upper_bound'),
)

mass_wasted_errors_bottom = alt.Chart(src_normal).mark_tick(color='black', width=8).encode(
    alt.X('Average Date:T'),
    alt.Y('mass_wasted_uncertainty_lower_bound'),
)

(mass_wasted_bars+mass_wasted_errors + mass_wasted_errors_top+ mass_wasted_errors_bottom).facet(
    column='name', row='shared_mask'
).resolve_scale(x='independent', y='independent')
# -

# ### Entire interval mass wasted

# +
src_normal = mass_wasted_df[
    ~(mass_wasted_df.fluvial) & (mass_wasted_df.Measurement != 'Normal')
]
src_normal['mass_wasted_uncertainty_upper_bound'] = src_normal['mass_wasted'] + src_normal['mass_wasted_uncertainty']
src_normal['mass_wasted_uncertainty_lower_bound'] = src_normal['mass_wasted'] - src_normal['mass_wasted_uncertainty']

# We need this to plot the error bars in a good place
src_normal['Average Date'] = src_normal['Start Date'] + ((src_normal['End Date'] - src_normal['Start Date']) / 2).dt.ceil('D')
src_normal.head()

mass_wasted_bars = alt.Chart(src_normal).mark_bar(opacity=0.5).encode(
    alt.X('Measurement:N'),
#     alt.X2('End Date:T'),
    alt.Y('mass_wasted:Q', title='Mass Wasted (cubic meters)')
).properties(width=150, height=150)

mass_wasted_errors = alt.Chart(src_normal).mark_bar(color='black', width=2).encode(
    alt.X('Measurement:N'),
    alt.Y('mass_wasted_uncertainty_lower_bound'),
    alt.Y2('mass_wasted_uncertainty_upper_bound'),
)

mass_wasted_errors_top = alt.Chart(src_normal).mark_tick(color='black', width=6).encode(
    alt.X('Measurement:N'),
    alt.Y('mass_wasted_uncertainty_upper_bound'),
)

mass_wasted_errors_bottom = alt.Chart(src_normal).mark_tick(color='black', width=8).encode(
    alt.X('Measurement:N'),
    alt.Y('mass_wasted_uncertainty_lower_bound'),
)

(mass_wasted_bars+mass_wasted_errors + mass_wasted_errors_top+ mass_wasted_errors_bottom).facet(
    column='name', row='shared_mask'
).resolve_scale(x='independent', y='independent')
# -

# ## Combined Erosion

# ### Plotted with fluvial and hillslope erosion summed together

mass_wasted_combined_df = mass_wasted_df.groupby([
    'name', 'Start Date', 'End Date', 'Measurement', 'shared_mask'
]).agg({
    'mass_wasted': 'sum',
    'mass_wasted_uncertainty': sum_in_quadrature,   
}).reset_index()

# +
src_normal = mass_wasted_combined_df[
    (mass_wasted_combined_df.Measurement == 'Normal')
]
src_normal['mass_wasted_uncertainty_upper_bound'] = src_normal['mass_wasted'] + src_normal['mass_wasted_uncertainty']
src_normal['mass_wasted_uncertainty_lower_bound'] = src_normal['mass_wasted'] - src_normal['mass_wasted_uncertainty']

# We need this to plot the error bars in a good place
src_normal['Average Date'] = src_normal['Start Date'] + ((src_normal['End Date'] - src_normal['Start Date']) / 2).dt.ceil('D')

mass_wasted_bars = alt.Chart(src_normal).mark_bar(opacity=0.5).encode(
    alt.X('Start Date:T', title='Date'),
    alt.X2('End Date:T'),
    alt.Y('mass_wasted:Q', title='Mass Wasted (cubic meters)')
).properties(width=200, height=200)

mass_wasted_errors = alt.Chart(src_normal).mark_bar(color='black', width=2).encode(
    alt.X('Average Date:T'),
    alt.Y('mass_wasted_uncertainty_lower_bound'),
    alt.Y2('mass_wasted_uncertainty_upper_bound'),
)

mass_wasted_errors_top = alt.Chart(src_normal).mark_tick(color='black', width=6).encode(
    alt.X('Average Date:T'),
    alt.Y('mass_wasted_uncertainty_upper_bound'),
)

mass_wasted_errors_bottom = alt.Chart(src_normal).mark_tick(color='black', width=8).encode(
    alt.X('Average Date:T'),
    alt.Y('mass_wasted_uncertainty_lower_bound'),
)

(mass_wasted_bars+mass_wasted_errors + mass_wasted_errors_top+ mass_wasted_errors_bottom).facet(
    column='name', row='shared_mask'
).resolve_scale(x='independent', y='independent')

# +
src_normal = mass_wasted_combined_df[
    (mass_wasted_combined_df.Measurement != 'Normal')
]
src_normal['mass_wasted_uncertainty_upper_bound'] = src_normal['mass_wasted'] + src_normal['mass_wasted_uncertainty']
src_normal['mass_wasted_uncertainty_lower_bound'] = src_normal['mass_wasted'] - src_normal['mass_wasted_uncertainty']

# We need this to plot the error bars in a good place
src_normal['Average Date'] = src_normal['Start Date'] + ((src_normal['End Date'] - src_normal['Start Date']) / 2).dt.ceil('D')
src_normal.head()

mass_wasted_bars = alt.Chart(src_normal).mark_bar(opacity=0.5).encode(
    alt.X('Measurement:N', axis=alt.Axis(labelAngle=0)),
#     alt.X2('End Date:T'),
    alt.Y('mass_wasted:Q', title='Mass Wasted (cubic meters)')
).properties(width=200, height=200)

mass_wasted_errors = alt.Chart(src_normal).mark_bar(color='black', width=2).encode(
    alt.X('Measurement:N'),
    alt.Y('mass_wasted_uncertainty_lower_bound'),
    alt.Y2('mass_wasted_uncertainty_upper_bound'),
)

mass_wasted_errors_top = alt.Chart(src_normal).mark_tick(color='black', width=6).encode(
    alt.X('Measurement:N'),
    alt.Y('mass_wasted_uncertainty_upper_bound'),
)

mass_wasted_errors_bottom = alt.Chart(src_normal).mark_tick(color='black', width=8).encode(
    alt.X('Measurement:N'),
    alt.Y('mass_wasted_uncertainty_lower_bound'),
)

(mass_wasted_bars+mass_wasted_errors + mass_wasted_errors_top+ mass_wasted_errors_bottom).facet(
    column='name', row='shared_mask'
).resolve_scale(x='independent', y='independent')
# -

# # Prep data for plotting DOD maps

dod_dataset_plotting = dod_dataset.copy()

# ## Replace data var name tuples with strings

for var in dod_dataset_plotting.data_vars:
    dod_dataset_plotting[
        f"{var[0]}-{var[1]}"
    ] = dod_dataset_plotting[var]
    dod_dataset_plotting = dod_dataset_plotting.drop_vars([var])


# ## Add Hillshade layer to dataset

def calculate_hillshade(array, azimuth=315, angle_altitude=45):
    azimuth = 360.0 - azimuth

    x, y = np.gradient(array)
    slope = np.pi / 2.0 - np.arctan(np.sqrt(x * x + y * y))
    aspect = np.arctan2(-x, y)
    azimuthrad = azimuth * np.pi / 180.0
    altituderad = angle_altitude * np.pi / 180.0

    shaded = np.sin(altituderad) * np.sin(slope) + np.cos(altituderad) * np.cos(
        slope
    ) * np.cos((azimuthrad - np.pi / 2.0) - aspect)

    hillshade = 255 * (shaded + 1) / 2

    return hillshade


dod_dataset_plotting['hillshade'] = (
    dem_dataset[2015].squeeze().dims,
    calculate_hillshade(dem_dataset[2015].squeeze().values)
)

# ### Load valley bounds 
#
# Mainly for plotting erosion polygons close up in the context of the valley

valley_bounds = gpd.read_file('/data2/elilouis/hsfm-geomorph/data/mt_baker_mass_wasted/valley_bounds.geojson')

# Prepare a few functions that serve up data for a specified valley

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
colors.reverse()

levels = [
    -24,
    -12,
    -6.66,
    -3.33,
    -1.3,
    1.3,
    3.33,
    6.66,
    12,
    24,
]
cmap = ListedColormap(colors)
cmap.set_bad(color="grey")
norm = BoundaryNorm(
    levels, 
    cmap.N, 
    clip=False, 
    extend='max'
)
# -


len(colors),len(levels)

def plot_dod_dataset(dataset, valley_name, discrete_colorbar=False, figsize = (20,10), vmin=-20, vmax=20):
    geoms = valley_bounds[valley_bounds.name == valley_name].geometry
    assert len(geoms) == 1
    geom = geoms.iloc[0]
    dataset = dataset.rio.clip_box(*geom.bounds)
    hillshade = dataset['hillshade']
    relevant_keys = dod_tools.get_dod_keys_for_valley(valley_name, valleys_metadata_dict)
    relevant_keys = [f"{x}-{y}" for x,y in relevant_keys]
    dataset = dataset[relevant_keys] 
    
    fig, axes = plt.subplots(1, 
                             len(dataset.data_vars), 
                             figsize=figsize, 
                             sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, var in zip(axes, dataset.data_vars):
        ax.set_aspect('equal')
        
        hillshade.plot.imshow(ax=ax, cmap='gray', alpha=0.75, add_colorbar=False)
        if discrete_colorbar:
            im = dataset[var].squeeze().plot.imshow(ax=ax, 
                                                cmap=cmap, norm=norm, 
                                        alpha=0.75, add_colorbar=False)
        else:
            im = dataset[var].squeeze().plot.imshow(ax=ax,
                                                    cmap='PuOr', vmin=vmin, vmax=vmax,
                                                    alpha=0.75, add_colorbar=False)
        pd.concat([
            erosion_polygons[erosion_polygons.name==valley_name],
            fluvial_erosion_polygons[fluvial_erosion_polygons.name==valley_name]
        ]).plot(ax=ax, linewidth=0.5, facecolor="none", edgecolor="black")
        
        ax.set_title(var, fontsize=11)
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.ticklabel_format(useOffset=False, style='plain')
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right', fontsize=9)
        plt.setp(ax.get_yticklabels(), fontsize=9)
        
    # plt.colorbar(im,ax=axes.ravel().tolist(),
    #              fraction=0.02, 
    #              pad=0.04
    #             )

    return fig, ax, im


# +
fig, ax, im = plot_dod_dataset(dod_dataset_plotting, 'Deming', figsize=(25, 5))
# fig.savefig('Deming.png', dpi=1600, bbox_inches = "tight")

plt.tight_layout()
fig.canvas.draw()
# cax = fig.add_axes([ax.get_position().x0, ax.get_position().y0 + 0.12, ax.get_position().width, 0.02])

cax = fig.add_axes([
    ax.get_position().x0 + 1.05*ax.get_position().width,  # bottom left corner of colorbar relative to furthest right ax
    ax.get_position().y0, 
    0.005, 
    ax.get_position().height
])
cbar = plt.colorbar(im, orientation='vertical', cax=cax)

# +
fig, ax, im = plot_dod_dataset(dod_dataset_plotting, 'Easton', figsize=(10,25))
# fig.savefig('Easton.png', dpi=1600, bbox_inches = "tight")

plt.tight_layout()
fig.canvas.draw()
# cax = fig.add_axes([ax.get_position().x0, ax.get_position().y0 + 0.12, ax.get_position().width, 0.02])

cax = fig.add_axes([
    ax.get_position().x0 + 1.05*ax.get_position().width,  # bottom left corner of colorbar relative to furthest right ax
    ax.get_position().y0, 
    0.01, 
    ax.get_position().height
])
cbar = plt.colorbar(im, orientation='vertical', cax=cax)

# +
fig, ax, im = plot_dod_dataset(dod_dataset_plotting, 'Coleman/Roosevelt', figsize=(25, 5))
# fig.savefig('Coleman-Roosevelt.png', dpi=1600, bbox_inches = "tight")

plt.tight_layout()
fig.canvas.draw()
# cax = fig.add_axes([ax.get_position().x0, ax.get_position().y0 + 0.12, ax.get_position().width, 0.02])

cax = fig.add_axes([
    ax.get_position().x0 + 1.05*ax.get_position().width,  # bottom left corner of colorbar relative to furthest right ax
    ax.get_position().y0, 
    0.01, 
    ax.get_position().height
])
cbar = plt.colorbar(im, orientation='vertical', cax=cax)

# +
fig, ax, im = plot_dod_dataset(dod_dataset_plotting, 'Mazama', figsize=(10, 5))
# fig.savefig('Mazama.png', dpi=1600, bbox_inches = "tight")
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
fig.canvas.draw()
# cax = fig.add_axes([ax.get_position().x0, ax.get_position().y0 + 0.12, ax.get_position().width, 0.02])

cax = fig.add_axes([
    ax.get_position().x0 + 1.05*ax.get_position().width,  # bottom left corner of colorbar relative to furthest right ax
    ax.get_position().y0, 
    0.01, 
    ax.get_position().height
])
cbar = plt.colorbar(im, orientation='vertical', cax=cax)
# -

fig = plot_dod_dataset(dod_dataset_plotting, 'Rainbow')
# fig.savefig('Rainbow.png', dpi=1600, bbox_inches = "tight")

fig = plot_dod_dataset(dod_dataset_plotting, 'Park')
# fig.savefig('Park.png', dpi=1600, bbox_inches = "tight")

fig = plot_dod_dataset(dod_dataset_plotting, 'Squak')
# fig.savefig('Squak.png', dpi=1600, bbox_inches = "tight")

# +
fig, ax, im = plot_dod_dataset(dod_dataset_plotting, 'Talum', figsize=(15, 5), vmin=-10, vmax=10)
# fig.savefig('Talum.png', dpi=1600, bbox_inches = "tight")

plt.tight_layout()

fig.canvas.draw()
# cax = fig.add_axes([ax.get_position().x0, ax.get_position().y0 + 0.12, ax.get_position().width, 0.02])

cax = fig.add_axes([
    ax.get_position().x0 + 1.05*ax.get_position().width,  # bottom left corner of colorbar relative to furthest right ax
    ax.get_position().y0, 
    0.01, 
    ax.get_position().height
])
cbar = plt.colorbar(im, orientation='vertical', cax=cax)

# +
fig, ax, im = plot_dod_dataset(dod_dataset_plotting, 'Thunder', figsize=(15, 5), vmin=-10, vmax=10)
# fig.savefig('Thunder.png', dpi=1600, bbox_inches = "tight")

plt.tight_layout()

fig.canvas.draw()
# cax = fig.add_axes([ax.get_position().x0, ax.get_position().y0 + 0.12, ax.get_position().width, 0.02])

cax = fig.add_axes([
    ax.get_position().x0 + 1.05*ax.get_position().width,  # bottom left corner of colorbar relative to furthest right ax
    ax.get_position().y0, 
    0.01, 
    ax.get_position().height
])
cbar = plt.colorbar(im, orientation='vertical', cax=cax)
# -


# # Examine Slope/Area relationship of erosion polygons

from pysheds.grid import Grid
from matplotlib import colors, cm, pyplot as plt

# Pull in a lower res DEM because the normal one is too high

slope_area_dem_fn = '/data2/elilouis/hsfm-geomorph/data/reference_dem_highres/baker/raw_tifs/baker_2015/2015_10m.tif'

# Use Pysheds to generate an array of accumulation/drainage area values

# +
# create slope and drainage area rasters for the reference dem

path_to_slope_raster = slope_area_dem_fn.replace('.tif', '_slope.tif')
path_to_darea_raster = slope_area_dem_fn.replace('.tif', '_drainage_area.tif')

grid = Grid.from_raster(slope_area_dem_fn)
dem = grid.read_raster(slope_area_dem_fn)
pit_filled_dem = grid.fill_pits(dem)
flooded_dem = grid.fill_depressions(pit_filled_dem)
inflated_dem = grid.resolve_flats(flooded_dem)
fdir = grid.flowdir(inflated_dem) #May need to pass in a dirmap dirmap=dirmap
acc = grid.accumulation(fdir)
slope = grid.cell_slopes(dem, fdir)
# -


Grid.from_raster(acc).to_raster(acc, 'acc.tif')
Grid.from_raster(slope).to_raster(slope, 'slope.tif')

# Open the low res DEM with rioxarray and then use the object and replace the values with the accumulation values. Note the above files generated are saved to file names that are used in the cell below.

lowres_dem = rix.open_rasterio(slope_area_dem_fn, masked=True).squeeze()
lowres_darea = rix.open_rasterio('acc.tif', masked=True).squeeze()
lowres_slope = rix.open_rasterio('slope.tif', masked=True).squeeze()

lowres_dem.plot.hist()

lowres_darea.plot.hist()

lowres_dem.plot.hist()

lowres_darea.values = np.ma.masked_where(np.isnan(lowres_dem.values), lowres_darea.values) 
lowres_slope.values = np.ma.masked_where(np.isnan(lowres_dem.values), lowres_slope.values) 

lowres_darea.plot()

lowres_slope.plot(vmin=-5, vmax=0)

lowres_dem.plot()

darea_vals = lowres_darea.values.flatten() 
slope_vals = lowres_slope.values.flatten()

# +

plt.scatter(darea_vals, slope_vals, s=0.00001, color='k')
plt.xlim(10,10e6)
plt.ylim(0,3)
plt.xscale('log')
# ax.set_xscale('log')
# ax.set_yscale('log')
# plt.show()

# -

df = pd.DataFrame.from_dict({
    'drainage area': darea_vals,
    'slope': slope_vals
}).dropna()

bins = [10**x for x in np.arange(0, 8, 0.5)]

binned_medians_df = pd.DataFrame(df.groupby(pd.cut(df['drainage area'], bins=bins))['slope'].median()).reset_index()
binned_medians_df['drainage area'] = binned_medians_df['drainage area'].apply(lambda interval: np.mean([interval.left, interval.right]))
binned_medians_df

# +

plt.scatter(darea_vals, slope_vals, s=0.00001, color='k')
plt.scatter(binned_medians_df['drainage area'], binned_medians_df['slope'])
plt.xlim(10,10e6)
plt.ylim(0,3)
plt.xscale('log')
# ax.set_xscale('log')
# ax.set_yscale('log')
# plt.show()

# -

# ### Crop the darea and slope rasters by polygons

# +
lowres_darea_fluvial_erosion_clipped = lowres_darea.rio.clip(fluvial_erosion_polygons.geometry)
lowres_darea_hillslope_erosion_clipped = lowres_darea.rio.clip(erosion_polygons.geometry)

lowres_slope_fluvial_erosion_clipped = lowres_slope.rio.clip(fluvial_erosion_polygons.geometry)
lowres_slope_hillslope_erosion_clipped = lowres_slope.rio.clip(erosion_polygons.geometry)

# +
df_fluvial = pd.DataFrame.from_dict({
    'drainage area': lowres_darea_fluvial_erosion_clipped.values.flatten(),
    'slope': lowres_slope_fluvial_erosion_clipped.values.flatten()
}).dropna()

df_hillslope = pd.DataFrame.from_dict({
    'drainage area': lowres_darea_hillslope_erosion_clipped.values.flatten(),
    'slope': lowres_slope_hillslope_erosion_clipped.values.flatten()
}).dropna()

# +
binned_medians_df_fluvial = pd.DataFrame(df_fluvial.groupby(pd.cut(df_fluvial['drainage area'], bins=bins))['slope'].median()).reset_index()
binned_medians_df_fluvial['drainage area'] = binned_medians_df_fluvial['drainage area'].apply(lambda interval: np.mean([interval.left, interval.right]))

binned_medians_df_hillslope = pd.DataFrame(df_hillslope.groupby(pd.cut(df_hillslope['drainage area'], bins=bins))['slope'].median()).reset_index()
binned_medians_df_hillslope['drainage area'] = binned_medians_df_hillslope['drainage area'].apply(lambda interval: np.mean([interval.left, interval.right]))

# +
plt.scatter(darea_vals, slope_vals, s=0.0001, alpha=0.5, color='k', label='All pixels')\

plt.scatter(
    lowres_darea_fluvial_erosion_clipped.values,
    lowres_slope_fluvial_erosion_clipped.values,
    label='Fluvial Erosion',
    s=2,
    alpha=0.3,
    # color='blue'
)

plt.scatter(
    lowres_darea_hillslope_erosion_clipped.values,
    lowres_slope_hillslope_erosion_clipped.values,
    label='Hillslope Erosion',
    s=2,
    alpha=0.3,
    # color='orange'
)

plt.scatter(binned_medians_df['drainage area'], binned_medians_df['slope'], 
            color='k')
plt.scatter(binned_medians_df_fluvial['drainage area'], binned_medians_df_fluvial['slope'], 
            color='lightblue')
plt.scatter(binned_medians_df_hillslope['drainage area'], binned_medians_df_hillslope['slope'], 
            color='red')

plt.xlim(10,10e6)
plt.ylim(0,1)
plt.xscale('log')
plt.legend()
# ax.set_xscale('log')
# ax.set_yscale('log')
# plt.show()
# -


