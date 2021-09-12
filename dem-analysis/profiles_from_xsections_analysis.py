# %% [markdown]
# # Profiles from X-sections Analysis, Baker

# %%
import sys
sys.path.append('/home/elilouis/hsfm-geomorph/dem-analysis/')
import profiling_tools

import os
import glob
import json

import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray as rix
from shapely.geometry import Point
from shapely import geometry

import seaborn as sns
import contextily as ctx

import altair as alt
alt.data_transformers.disable_max_rows()

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 10.0)


# %% [markdown]
# Read in manually generated cross-section lines
#
# NOTE: Cross sections must be in order from downstream to upstream

# %% [markdown]
# # Load files

# %%
xsection_files = glob.glob(
    '/data2/elilouis/hsfm-geomorph/data/mt_baker_long_profiles/valley_xsections/*.json'
)

# %% [markdown]
# Create labeled versions

# %%
# mkdir /data2/elilouis/hsfm-geomorph/data/mt_baker_long_profiles/valley_xsections_ordered/

# %%
for f in xsection_files:
    gdf = gpd.read_file(f)
    gdf['id'] = gdf.index
    gdf.to_file(f.replace('valley_xsections', 'valley_xsections_ordered'), driver='GeoJSON')

# %%
all_historical_dem_files = glob.glob( 
    "/data2/elilouis/generate_ee_dems_baker/mixed_timesift/individual_clouds/*/cluster*/1/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/*align.tif"
)
modern_dem_files = [
    "/data2/elilouis/hsfm-geomorph/data/reference_dem_highres/baker/raw_tifs/baker_2015/2015.tif"
]

# %%
dem_stats_files = [
    f.replace('align.tif', 'align_stats.json') for f in all_historical_dem_files
]

# %%
UPPER_NMAD_LIMIT = 2.5

# %% [markdown]
# Filter for good historicals (low NMAD)

# %% [markdown]
# Should this be `after_filt` of `after`?

# %%
nmads = []
good_historical_dem_files = []

for dem_file, stats_file in zip(all_historical_dem_files, dem_stats_files):
    with open(stats_file) as src:
        x = json.load(src)
#         nmad = x['after']['nmad']
        nmad = x['after_filt']['nmad']
        nmads.append(nmad)
        print(nmad)
        if nmad < UPPER_NMAD_LIMIT:
            good_historical_dem_files.append(dem_file)

# %% [markdown]
# This is how many DEMs were dropped due to poor NMAD.

# %%
len(all_historical_dem_files) - len(good_historical_dem_files)

# %% [markdown]
# Create a list of all the DEMs we will use

# %%
usable_dem_files = good_historical_dem_files + modern_dem_files

# %%
usable_dem_files

# %% [markdown]
# # Extract error/NMAD of DEMs and set an upper limit

# %%
sns.distplot(nmads, bins=100, kde=False, hist=True)
plt.xlabel('NMAD (m)')
plt.ylabel('DEM count')

# %% [markdown]
# # Extract data from DEMs

# %%
profiles_gdf = gpd.GeoDataFrame()

# %%
for dem_file in usable_dem_files:
    date_str = dem_file.split('/')[6]
    for xsection_file in xsection_files:                
        # Check if DEM and xsections even intersect
        xsection_geoms = gpd.read_file(xsection_file)
        dem_bounding_polygon = geometry.box(*rix.open_rasterio(dem_file).rio.bounds())
        dem_and_xsections_intersect = any(
            xsection_geoms.geometry.apply(lambda x: dem_bounding_polygon.intersects(x))
        )
        if dem_and_xsections_intersect:          
            print(f'Extracting xsections from xsection file {xsection_file}')

            gdf = profiling_tools.get_valley_centerline_from_xsections(xsection_file, dem_file)
            gdf['Date'] = date_str
            
            stats_file_path = dem_file.replace('align.tif', 'align_stats.json')
            #this first logic test makes sure that the stats file actually exists (if we are working on the reference DEM,
            # then the above replace code won't actually do anything and we will try to open a DEM tif file as the JSON)
            if 'align_stats.json' in stats_file_path and os.path.exists(stats_file_path):
                    with open(stats_file_path) as src:
                        x = json.load(src)
                        nmad = x['after']['nmad']
            else:
                print(f'No statistics file available for file {dem_file}, setting NMAD to NaN')
                nmad = np.nan
                
            gdf['nmad'] = nmad
            gdf['Location'] = xsection_file.split('/')[-1].replace('.json', '')
            assert np.nan not in gdf['Location'],"How is Location NAN!?"

            profiles_gdf = profiles_gdf.append(gdf)
        else:
            print(f'DEM files and xsections do not intersect (xsection file {xsection_file})')
            
    print()

# %%
profiles_gdf.crs

# %%
profiles_gdf

# %% [markdown]
# # Plot profile view

# %% [markdown]
# Take advantage of Altair's facet grid (obscures detail)

# %%
src = profiles_gdf.drop('geometry', axis=1)
alt.Chart(src).mark_line().encode(
    alt.X('Upstream Distance:Q'),
    alt.Y('Elevation:Q', scale=alt.Scale(zero=False)),
    alt.Color('Date:N'),
    alt.Facet('Location', columns=3)
).resolve_scale(x='independent', y='independent')

# %% [markdown]
# Better detail view with loops (more code)

# %%
for key, src in profiles_gdf.drop('geometry', axis=1).groupby('Location'):
    display(
        alt.Chart(src).mark_line().encode(
            alt.X('Upstream Distance:Q'),
            alt.Y('Elevation:Q', scale=alt.Scale(zero=False)),
            alt.Color('Date:N'),
        ).resolve_scale(x='independent', y='independent').properties(
            width=1500,
            height=1000,
            title=key
        )
    )

# %% [markdown]
# Better detail view with error bars.
#
# Right now i can only do this with one dataset (location and date) at a time 

# %%
profiles_gdf['Elevation Upper Confidence Limit'] = profiles_gdf['Elevation'] + profiles_gdf['nmad']
profiles_gdf['Elevation Lower Confidence Limit'] = profiles_gdf['Elevation'] - profiles_gdf['nmad']

# %%
test = pd.concat([
    profiles_gdf.groupby(['Location', 'Date']).get_group(('boulder1', '70_09')),
    profiles_gdf.groupby(['Location', 'Date']).get_group(('boulder1', 'reference_dem_highres'))
])

# %%
profiles_gdf.nmad.unique()

# %%
src = test[test['Upstream Distance'] > 3000]

line = alt.Chart(src).mark_line().encode(
    x='Upstream Distance',
    y=alt.Y('Elevation', scale=alt.Scale(zero=False)),
    color='Date:N'
).properties(width=1500, height = 500)

band = alt.Chart(src).mark_area(
    opacity=0.25
).encode(
    x='Upstream Distance:Q',
    y=alt.Y('Elevation Lower Confidence Limit', scale=alt.Scale(zero=False)),
    y2='Elevation Upper Confidence Limit',
    color='Date:N'
).properties(width=1000, height = 500)

line + band

# %% [markdown]
# # Plot map view

# %%
for loc, gdf in profiles_gdf.groupby('Location'):
    # Replace Point(nan, nan) with None
    gdf.geometry = gdf.geometry.apply(lambda point: None if all([np.isnan(point.x), np.isnan(point.y)]) else point)

    line_gdf = gpd.GeoDataFrame(gdf.groupby('Date').apply(lambda x: geometry.LineString(x.geometry.dropna().tolist())).reset_index().rename({0:'geometry'}, axis=1), crs=gdf.crs)
    line_gdf = line_gdf[~line_gdf.geometry.is_empty]
    line_gdf['Location'] = loc
    ax = line_gdf.plot(column='Date', legend=True)
    plt.title(loc)
    ctx.add_basemap(ax, zoom=14, crs=line_gdf.crs, source=ctx.providers.OpenStreetMap.Mapnik)

# %% [markdown]
# # Calculate Residuals 
# With modern LIDAR DEM

# %%
groups = profiles_gdf.groupby('Location').apply(lambda x: x.groupby('Date'))

# %%
# residual_date_key = 'reference_dem_highres'
residual_date_key = 'baker'

# %%
diff_df = pd.DataFrame()
for grouped_by_date, index in zip(groups, groups.index):    
    def create_diff_df(df, residual_df):
        merged = df.merge(residual_df, on='Upstream Distance')
        merged['Elevation Difference'] = merged['Elevation_y'] - merged['Elevation_x']
        return merged
    residual_base_df = grouped_by_date.get_group(residual_date_key)
    difference_df = grouped_by_date.apply(
        lambda per_date_and_loc_df: create_diff_df(per_date_and_loc_df, residual_base_df)    
    )
    diff_df = diff_df.append(difference_df)

# %%
diff_df = diff_df[['Date_x', 'Location_x', 'Upstream Distance', 'Elevation Difference']].rename(
    {'Date_x': 'Date', 'Location_x': 'Location'},
    axis=1
)

# %%
alt.Chart(diff_df).mark_line().encode(
    alt.X('Upstream Distance:Q'),
    alt.Y('Elevation Difference:Q', scale=alt.Scale(zero=False)),
    alt.Color('Date:O'),
    alt.Facet('Location', columns=3)
).resolve_scale(x='independent', y='independent')
