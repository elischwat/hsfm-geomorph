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

# Using data prepared in the `collect_and_prepare_rasters.py` file

import rioxarray as rix
import os
import matplotlib.pyplot as plt
import pandas as pd
import altair as alt
import geopandas as gpd
alt.data_transformers.disable_max_rows()


# difference_dem_dir = "/data2/elilouis/hsfm-geomorph/data/mt_hood_eliot_glacier/difference_dems_glacier_nlcd_masked/"
difference_dem_dir = "/data2/elilouis/hsfm-geomorph/data/mt_hood_eliot_glacier/difference_dems_glacier_studyarea_masked/"

# # Load DOD rasters

# +
rasters = dict([
    (f.replace('.tif', ''), rix.open_rasterio(os.path.join(difference_dem_dir, f)))
    for f in os.listdir(difference_dem_dir) if f.endswith('.tif')
])

rasters.keys()
# -

# # Load Slope and Area Rasters (the regridded versions)

other_rasters = {
    'slope': rix.open_rasterio("/data2/elilouis/hsfm-geomorph/data/mt_hood_eliot_glacier/rasters_regridded/slope.tif"),
    'area': rix.open_rasterio("/data2/elilouis/hsfm-geomorph/data/mt_hood_eliot_glacier/rasters_regridded/flowdir.tif")
}

# # Visualize DOD Rasters

fig, axes = plt.subplots(
    ncols=5,
    nrows=int(len(rasters)/5),
    figsize=(20,5),
    sharex = True,
    sharey = True,
    constrained_layout=True
)
ims = []
for (date_str,raster), ax in zip(rasters.items(), axes):
    ax.set_title(date_str)
    ax.set_aspect('equal')
    im = raster.plot(
        ax=ax,
        add_labels = False,
        add_colorbar = False,
        vmin = -10,
        vmax = 10,
        cmap='PuOr'
    )
    ims.append(im)
cbar = fig.colorbar(ims[0], ax=axes, shrink=0.5)

# # Visualize distributions of DOD rasters

import seaborn as sns

fig, axes = plt.subplots(
    ncols=5,
    nrows=int(len(rasters)/5),
    figsize=(20,5),
    sharex = True,
    sharey = True,
    constrained_layout=True
)
ims = []
for (date_str,raster), ax in zip(rasters.items(), axes):
    ax.set_title(date_str)
    im = sns.distplot(raster.values, ax=ax, kde=False, bins=200)
    ax.set_xlim(-20,20)
    ims.append(im)

# # Create tabular data with difference data and slope/area

df = pd.DataFrame(dict(
    [(k, raster.values.flatten()) for k, raster in rasters.items()]
))
df['slope'] = other_rasters['slope'].values.flatten()
df['area'] = other_rasters['area'].values.flatten()
df = df.dropna()

df.head()

# # Plot Slope and Area Distributions

import seaborn as sns

sns.distplot(df.slope, kde=False)

sns.distplot(df.area, kde=False)
plt.gca().set_yscale('log')

# # Plot each DOD dataset on the slope-area diagram

# +

import holoviews as hv
import holoviews.operation.datashader as hd
from holoviews.plotting.links import DataLink

hv.extension('bokeh')

# -

def slope_area_diff_plot(df, diff_year):
    return hv.Points(
        data=df,
        kdims=['slope', 'area'],
        vdims=[diff_year],
    ).opts(
        color=diff_year, 
        cmap='viridis', 
        width=500, 
        height=500, 
        colorbar=True, 
        clim=(-20,20),
#         size=1,
#         alpha=0.1
    )


slope_area_diff_plot(df, '1975-09-1967-09')

slope_area_diff_plot(df, '1977-10-1975-09')

slope_area_diff_plot(df, '1980-10-1977-10')


slope_area_diff_plot(df, '1990-09-1980-10')

slope_area_diff_plot(df, '2009-1990-09')

# # Plot each DOD dataset with area

# +
from holoviews import opts
from holoviews.operation.datashader import datashade

(
    hv.Points(df[['area', '1975-09-1967-09']]).opts(ylim=(-20, 20)) +
    hv.Points(df[['area', '1977-10-1975-09']]).opts(ylim=(-20, 20)) +
    hv.Points(df[['area', '1980-10-1977-10']]).opts(ylim=(-20, 20)) +
    hv.Points(df[['area', '1990-09-1980-10']]).opts(ylim=(-20, 20)) +
    hv.Points(df[['area', '2009-1990-09']]).opts(ylim=(-20, 20))
).opts(shared_axes=True)

# -

# # Plot each DOD dataset with slope

# +
from holoviews import opts

(
    datashade(hv.Points(df[['slope', '1975-09-1967-09']])).opts(xlim=(0, 90), ylim=(-20, 20)) +
    datashade(hv.Points(df[['slope', '1977-10-1975-09']])).opts(xlim=(0, 90), ylim=(-20, 20)) +
    datashade(hv.Points(df[['slope', '1980-10-1977-10']])).opts(xlim=(0, 90), ylim=(-20, 20)) +
    datashade(hv.Points(df[['slope', '1990-09-1980-10']])).opts(xlim=(0, 90), ylim=(-20, 20)) +
    datashade(hv.Points(df[['slope', '2009-1990-09']])).opts(xlim=(0, 90), ylim=(-20, 20))
).opts(shared_axes=True)

# -

# # Calculate net mass change

net_change_df = pd.DataFrame(df.drop(['slope', 'area'], axis=1).sum()).reset_index().rename({'index': 'Timespan', 0: 'Linear net change (m)'}, axis=1)

net_change_df['Volumetric net change (m^3)'] = 2*2*net_change_df['Linear net change (m)']

# ## Break down net change across many years into annual rates
#
# Split net change across time spans equally for each year

import numpy as np


def create_annual_dataset(start_year, end_year, df, key):
    return list(range(start_year, end_year)), np.full(
        len(list(range(start_year, end_year))),
        df[df.Timespan==key]["Volumetric net change (m^3)"].iloc[0]/len(list(range(start_year, end_year)))
    )


net_change_df

all_years = []
all_net_changes = []
for yrs, netchanges in [
    create_annual_dataset(1967, 1975, net_change_df, '1975-09-1967-09'),
    create_annual_dataset(1975, 1977, net_change_df, '1977-10-1975-09'),
    create_annual_dataset(1977, 1980, net_change_df, '1980-10-1977-10'),
    create_annual_dataset(1980, 1990, net_change_df, '1990-09-1980-10'),
    create_annual_dataset(1990, 2009, net_change_df, '2009-1990-09')
]:
    all_years = all_years + list(yrs)
    all_net_changes = all_net_changes + list(netchanges)
annual_net_change_df = pd.DataFrame({
    'Year': all_years,
    'Volumetric net change (m^3)': all_net_changes
})

# +
alt.Chart(net_change_df).mark_line(point=True).encode(
    x='Timespan:O',
    y = 'Volumetric net change (m^3)'
    
)
# -

alt.Chart(annual_net_change_df).mark_line(point=True).encode(
    x='Year:Q',
    y = 'Volumetric net change (m^3)'
)

# # Calculate glacial area change

glacier_gdf = gpd.read_file(
    "/data2/elilouis/hsfm-geomorph/data/mt_hood_eliot_glacier/glacier_polygons.shp"
)

glacier_gdf

glacier_gdf['area'] = glacier_gdf.geometry.area

glacier_gdf = glacier_gdf.groupby('id').sum().reset_index()
glacier_gdf

glacier_gdf['area change'] = glacier_gdf.area.diff()

glacier_gdf

annual_glacier_change_df = pd.concat([
    pd.DataFrame({
            'Year': range(start_year, end_year),
            'Glacier area change (m^2)': np.full(
                len(range(start_year, end_year)), 
                glacier_gdf[glacier_gdf.id==end_year]['area change'].iloc[0]/len(range(start_year, end_year))
            )
        }) 
    for start_year, end_year in [
        (1967, 1975),
        (1975, 1977),
        (1977, 1980),
        (1980, 1990),
        (1990, 2009)
    ]
])

change_df = pd.merge(annual_glacier_change_df, annual_net_change_df, on='Year')

# +
base = alt.Chart(change_df).encode(
    alt.X('Year:Q')
)

sed_change = base.mark_line(color='#2ca02c').encode(
    alt.Y(
        'Volumetric net change (m^3)', 
        axis=alt.Axis(titleColor='#2ca02c', title='Sediment volumetric change rate (m^3/year)'), 
        scale=alt.Scale(domain=[-500000, 500000])
    )
)

glacier_change = base.mark_line(color='#4c78a8').encode(
    alt.Y(
        'Glacier area change (m^2)', 
        axis=alt.Axis(titleColor='#4c78a8', title='Glacier area change rate (m^2/year)'), 
        scale=alt.Scale(domain=[-30000, 30000])
    )
)

zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='grey').encode(y='y')

alt.layer(sed_change, glacier_change, zero_line).resolve_scale(
    y='independent'
).properties(
    title='Annual rate of change in sediment volume and glacier area'
)
# -

# # Profile Analysis on LIA moraine wall

lines_gdf = gpd.read_file('/data2/elilouis/hsfm-geomorph/data/mt_hood_eliot_glacier/profiles.shp')
lines_gdf['id'] = lines_gdf.index
lines_gdf

import glob
glob.glob("/data2/elilouis/hsfm-geomorph/data/mt_hood_eliot_glacier/rasters_regridded/*.tif")

import profiling_tools
from pathlib import Path


dem_files = [
    '/data2/elilouis/hsfm-geomorph/data/mt_hood_eliot_glacier/rasters_regridded/1967-09.tif',
    '/data2/elilouis/hsfm-geomorph/data/mt_hood_eliot_glacier/rasters_regridded/1975-09.tif',
    '/data2/elilouis/hsfm-geomorph/data/mt_hood_eliot_glacier/rasters_regridded/1977-10.tif',
    '/data2/elilouis/hsfm-geomorph/data/mt_hood_eliot_glacier/rasters_regridded/1980-10.tif',
    '/data2/elilouis/hsfm-geomorph/data/mt_hood_eliot_glacier/rasters_regridded/1990-09.tif',
    '/data2/elilouis/hsfm-geomorph/data/mt_hood_eliot_glacier/rasters_regridded/2009.tif'
]

# +
df_list = []

for key, linestring in lines_gdf.iterrows():
    for dem_file in dem_files:
        one_profile_df = profiling_tools.get_raster_values_for_line(
            profiling_tools.increase_line_complexity(linestring.geometry, 100), 
            dem_file, 
            band=1
        )
        one_profile_df['Date'] = Path(dem_file).stem
        one_profile_df['Profile'] = key
        df_list.append(one_profile_df)
profiles_df = pd.concat(df_list)
# -

alt.Chart(profiles_df).mark_line().encode(
    alt.X('path_distance'), 
    alt.Y('raster_value', scale=alt.Scale(zero=False)),
    alt.Color('Date:O'),
    alt.Facet('Profile', columns=3)
).properties(width=300, height=200).resolve_scale(y='independent')


