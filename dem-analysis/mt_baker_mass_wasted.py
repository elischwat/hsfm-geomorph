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
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry.collection import GeometryCollection
import rioxarray as rix

import matplotlib.pyplot as plt
import seaborn as sns
import hvplot
import altair as alt
import geoviews as gv
from cartopy import crs as ccrs
import holoviews as hv

from holoviews import opts
from holoviews.operation.datashader import regrid

hv.extension('bokeh')
# -

BASE_PATH = "/data2/elilouis/generate_ee_dems_baker/mixed_timesift/individual_clouds/"
def get_resource(other_path):
    return os.path.join(BASE_PATH, other_path)
def see_resources():
    return os.listdir(BASE_PATH)


see_resources()

# ## Load the 1970 - 2009 DOD

dod_file = get_resource('70_9.0/dod.tif')

dod = rix.open_rasterio(dod_file, masked=True).squeeze()
dod = - dod

plt.figure(figsize=(10,10))
plt.imshow(dod.values[::10,::10], cmap='PuOr', vmin=-30,vmax=30)
plt.colorbar()

# ## Create forest-masked version
# Uses dem_coreg library dem_mask.py

# !dem_mask.py {dod_file} --nlcd --nlcd_filter not_forest 

# !mv \
#     /data2/elilouis/generate_ee_dems_baker/mixed_timesift/individual_clouds/70_9.0/dod_ref.tif \
#     /data2/elilouis/generate_ee_dems_baker/mixed_timesift/individual_clouds/70_9.0/dod_masked_forest.tif

dod_masked_forest = rix.open_rasterio(
    "/data2/elilouis/generate_ee_dems_baker/mixed_timesift/individual_clouds/70_9.0/dod_masked_forest.tif",
    masked=True
).squeeze()
dod_masked_forest = - dod_masked_forest

plt.figure(figsize=(10,10))
plt.imshow(dod_masked_forest.values[::10,::10], cmap='PuOr', vmin=-30,vmax=30)
plt.colorbar()

# ## Create glacier-masked version

glacier_polygons = gpd.read_file('/data2/elilouis/hsfm-geomorph/data/mt_baker_mass_wasted/glacier_polygons_1970.geojson')
glacier_polygons = glacier_polygons.geometry.apply(lambda x: x if x else GeometryCollection()).to_crs(epsg=32610)

dod_masked_glaciers = dod.rio.clip(glacier_polygons.geometry, invert=True)

plt.figure(figsize=(10,10))
plt.imshow(dod_masked_glaciers.values[::10,::10], cmap='PuOr', vmin=-30,vmax=30)
plt.colorbar()

# ## Create glacier and forest masked version

dod_masked_forest_glaciers = dod_masked_forest.rio.clip(glacier_polygons.geometry, invert=True)

plt.figure(figsize=(10,10))
plt.imshow(dod_masked_forest_glaciers.values[::10,::10], cmap='PuOr', vmin=-30,vmax=30)
plt.colorbar()

# ## Load Erosion Polygons

erosion_polygons = gpd.read_file('/data2/elilouis/hsfm-geomorph/data/mt_baker_mass_wasted/erosion_polygons.geojson')
fluvial_erosion_polygons = gpd.read_file('/data2/elilouis/hsfm-geomorph/data/mt_baker_mass_wasted/fluvial_erosion_polygons.geojson')

fluvial_erosion_polygons.to_file(
    '/data2/elilouis/hsfm-geomorph/data/mt_baker_mass_wasted/fluvial_erosion_polygons.geojson',
    driver='GeoJSON'
)

erosion_polygons['fluvial'] = False
fluvial_erosion_polygons['fluvial'] = True



polygons = polygons.drop(['_count', '_sum', '_mean'], axis=1)

polygons = polygons.reset_index(drop=True)

polygons.sort_values('name')

# # Analysis with a single DOD

# ### Calculate mass wasted in each

# Available rasters:

# + jupyter={"outputs_hidden": true}
polygons['pixel_count'] = polygons.geometry.apply(
    lambda geom: np.count_nonzero(~np.isnan(
        dod.rio.clip([geom])
    ))
)
# -

polygons['mass_wasted_raw_dod'] = polygons.geometry.apply(
    lambda geom:
    (
        dod.rio.clip([geom]).sum()*(
            dod.rio.resolution()[0]*(-dod.rio.resolution()[1])
        )
    ).values
)

polygons['mass_wasted_masked_forest'] = polygons.geometry.apply(
    lambda geom:
    (
        dod_masked_glaciers.rio.clip([geom]).sum()*(
            dod.rio.resolution()[0]*(-dod.rio.resolution()[1])
        )
    ).values
)

polygons['estimate_diff'] = polygons['mass_wasted_masked_forest']/polygons['mass_wasted_raw_dod']

polygons.sort_values('estimate_diff').head()

polygons_by_locandtype = polygons.groupby(['name', 'fluvial']).sum().reset_index()

polygons_by_locandtype

polygons_by_locandtype['type'] = polygons_by_locandtype['fluvial'].apply(
    lambda x: 'fluvial' if x else 'hillslope'
)
polygons_by_locandtype = pd.melt(polygons_by_locandtype, id_vars=['name', 'type', 'pixel_count'], value_vars=[
    'mass_wasted_raw_dod',
    'mass_wasted_masked_forest'
])
polygons_by_locandtype['masked'] = polygons_by_locandtype['variable'].apply(
    lambda x: False if x == 'mass_wasted_raw_dod' else True
)

polygons_by_locandtype

# pd.options.display.float_format = '{:,.0f}'.format
display_df = polygons_by_locandtype.pivot(index=['name', 'type', 'pixel_count'], columns='variable', values='value')
display_df = display_df.rename(mapper = {
    'mass_wasted_masked_forest': 'Mass Wasted (cubic meters), Forest Removed',
    'mass_wasted_raw_dod': 'Mass Wasted (cubic meters)',
    'name': 'Glacial Valley'
    },
    axis='columns'
)
display_df = display_df.reset_index()

# +

print(len(display_df))
# -

# ### Calculate uncertainty

# Uncertainty equation, assuming NO systematic uncertainty

# $\sigma_v = n L^2 \sqrt{\sigma_{rms}^2 \big/ n}$

# +
rms = 0.95 # meters
L = 1.00 # meters

display_df['Mass Wasted Uncertainty (cubic meters)'] = display_df['pixel_count'].apply(
    lambda n: n*L**2*np.sqrt(rms**2 / n)
)
display_df = display_df.rename(mapper={'name': 'Glacial Valley', 'pixel_count': 'Pixel Count'}, axis='columns')

# -

display_df

display_df.to_html()

# +
src = polygons_by_locandtype

src = src[src['name'].isin(['Deming', 'Mazama'])]

domain = ['fluvial', 'hillslope']
range_ = ['#FFA500', '#FF0000']


alt.Chart(src).mark_bar().encode(
    alt.X('masked:N'),
#     alt.Y('value', scale = alt.Scale(type='symlog')),
    alt.Y('value'),
    alt.Color('type', scale=alt.Scale(domain=domain, range=range_))
).facet('name')

# +
src = polygons_by_locandtype

src = src[~src['name'].isin(['Deming', 'Mazama'])]

domain = ['fluvial', 'hillslope']
range_ = ['#FFA500', '#FF0000']

alt.Chart(src).mark_bar().encode(
    alt.X('masked:N'),
#     alt.Y('value', scale = alt.Scale(type='symlog')),
    alt.Y('value'),
    alt.Color('type', scale=alt.Scale(domain=domain, range=range_))
).facet('name')
# -

# # Analysis with all the DODs

# ## Organize and load data

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
        'years': [1970, 1977, 1979, 2015],
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

# ### Create dictionary of DEMs by year

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
reference_dem_file = '/data2/elilouis/hsfm-geomorph/data/reference_dem_highres/baker/raw_tifs/baker_2015/2015.tif'

for k,v in dem_files_dict.items():
    print(k, os.path.exists(v))

for year, file in dem_files_dict.items():
    # !gdalinfo {file} | grep "Pixel Size"

# ### Load the reference_dem and regrid it to 1m. This grid will be used for all the other DEMs

reference_dem = rix.open_rasterio(reference_dem_file, masked=True, chunks=1000).squeeze()
reference_dem = reference_dem.rio.reproject(reference_dem.rio.crs, resolution=(1,1))
reference_dem.name = 'reference dem'
reference_dem

# ### Load dems into dictionary

dem_dict = {
    k: rix.open_rasterio(v, masked=True).squeeze().rio.reproject_match(reference_dem)
    for k, v in dem_files_dict.items()
}

{k:v.rio.resolution() for k,v in dem_dict.items()}

difference_data_df = pd.DataFrame()

# ## Calculate mass wasted for each polygon and DOD combo

# Get mass-wasted calculations for each polygon, for each year of DEM coverage associated with the valley name that belongs to each polygon

# %%time
for idx, p in polygons.iterrows():
    years = valleys_metadata_dict[p['name']]['years']
    for i in range(0, len(years) - 1):
        first_year = years[i]
        second_year = years[i + 1]
        # clip box and then clup to be more memory efficient, per rioxarray docs recommendation
        first_dem = dem_dict[first_year].rio.clip_box(*p.geometry.bounds).rio.clip([p.geometry])
        second_dem = dem_dict[second_year].rio.clip_box(*p.geometry.bounds).rio.clip([p.geometry])
        if not first_dem.rio.shape[0]:
            print('missing data')
        diff_dem = second_dem - first_dem
        mass_wasted = diff_dem.sum() * diff_dem.rio.resolution()[0] * -diff_dem.rio.resolution()[1]
        difference_data_df = difference_data_df.append(pd.DataFrame({
            'idx': [idx],
            'id': [p['id']],
            'polygon': [p['geometry']],
            'name': [p['name']],
            'fluvial': [p['fluvial']],
            'start year': [first_year],
            'end year': [second_year],
            'mass wasted (cubic meters)': [mass_wasted.values.item()]
        }))

# ## Plot erosion data by polygon for multiple time intervals

fluvial_erosion_df = difference_data_df[difference_data_df.fluvial]
hillslope_erosion_df = difference_data_df[~difference_data_df.fluvial]

# ### Fluvial Erosion

# +
src = fluvial_erosion_df.drop('polygon', axis='columns')

alt.Chart(src).mark_bar().encode(
    alt.X('start year:O'),
    alt.X2('end year:O'),
    alt.Y('mass wasted (cubic meters)')
).properties(width=150, height=150).facet(
    'name',
    columns=5
).resolve_scale(
    x='independent', 
    y='independent'
)
# -

# ### Hillslope Erosion (by polygon)

src = hillslope_erosion_df.drop('polygon', axis='columns')
src['id'] = src['id'].astype(int)
alt.Chart(src).mark_bar().encode(
    alt.X('start year:O'),
    alt.X2('end year:O'),
    alt.Y('mass wasted (cubic meters)')
).properties(
    width=150,
    height=150
).facet(
    row='name',
    column=alt.Column('id:Q')
).resolve_scale(
    x='independent', 
    y='independent'
)

# ### Hillslope Erosion (by valley, polygons summed)

# +
src = hillslope_erosion_df.drop('polygon', axis='columns')
src['id'] = src['id'].astype(int)
src = src.groupby(['name', 'start year', 'end year']).sum().drop(['idx', 'id', 'fluvial'], axis='columns').reset_index()

alt.Chart(src).mark_bar().encode(
    alt.X('start year:O'),
    alt.X2('end year:O'),
    alt.Y('mass wasted (cubic meters)')
).properties(
    width=150,
    height=150
).facet(
    'name',
    columns=4
).resolve_scale(y='independent')

# +
# import holoviews as hv

# from holoviews import opts
# from holoviews.operation.datashader import regrid

# hv.extension('bokeh')
# -

# ## Plot Repeat DoD Maps 

valley_bounds = gpd.read_file('/data2/elilouis/hsfm-geomorph/data/mt_baker_mass_wasted/valley_bounds.geojson')


def get_dods(valley_name):
    valley_bbox = valley_bounds[valley_bounds.name==valley_name].geometry.iloc[0]
    print(f'{valley_name} valley bbox: {valley_bbox.bounds}')
    years = valleys_metadata_dict[valley_name]['years']
    print(f'{valley_name} years available: {years}')
    dods = []
    for i in range(0, len(years)-1):
        earlier_dem = dem_dict[years[i]]
        later_dem = dem_dict[years[i+1]]
        this_dod = later_dem.rio.clip_box(*valley_bbox.bounds) - earlier_dem.rio.clip_box(*valley_bbox.bounds)
        this_dod.name = f"{years[i+1]} - {years[i]}"
        print(f'Creating dod {this_dod.name}')
        this_dod['_file_obj'] = None
        dods.append(this_dod)
    # add in the bounding DOD
    earlier_dem = dem_dict[years[0]]
    later_dem = dem_dict[years[-1]]
    bounding_dod = later_dem.rio.clip_box(*valley_bbox.bounds) - earlier_dem.rio.clip_box(*valley_bbox.bounds)
    bounding_dod.name = f"{years[0]} - {years[-1]}"
    print(f'Creating dod {bounding_dod.name}')
    bounding_dod['_file_obj'] = None
    dods.append(bounding_dod)
    return dods


def get_polys_to_plot(valley_name):
    hillslope_polys_to_plot = erosion_polygons[erosion_polygons.name == 'Easton']
    fluvial_polys_to_plot =  fluvial_erosion_polygons[fluvial_erosion_polygons.name == 'Easton']

    hillslope_erosion_gvpolys = gv.Polygons(
        hillslope_polys_to_plot.to_crs(epsg=4326), 
        vdims=['name']
    ).opts(
        projection=ccrs.UTM(zone=10),
        fill_alpha=0.0,
        line_color='red'
    )
    
    fluvial_erosion_gvpolys = gv.Polygons(
        fluvial_polys_to_plot.to_crs(epsg=4326), 
        vdims=['name']
    ).opts(
        projection=ccrs.UTM(zone=10),
        fill_alpha=0.0,
        line_color='blue'
    )
    return hillslope_erosion_gvpolys, fluvial_erosion_gvpolys


def get_diff_map(dod, hillslope_polys, fluvial_polys):
    
    return regrid(hv.Image(dod).opts(cmap='PuOr', clim=(-10,10), 
                                     width=300, height=400, 
                                     title=dod.name)) * hillslope_polys * fluvial_polys
def plot_multiple_diff_maps(dods, hillslope_polys, fluvial_polys):
    combined_diff_map = get_diff_map(dods[0], hillslope_polys, fluvial_polys)
    for dod in dods[1:]:
        combined_diff_map = combined_diff_map + get_diff_map(dod, hillslope_polys, fluvial_polys)
    return combined_diff_map


dods = get_dods('Easton')
hillslope_erosion_gvpolys, fluvial_erosion_gvpolys = get_polys_to_plot('Easton')
plot_multiple_diff_maps(dods, hillslope_erosion_gvpolys, fluvial_erosion_gvpolys)


