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

# %%
xsection_files

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
historical_dem_files = glob.glob( 
    # "/data2/elilouis/generate_ee_dems_baker/mixed_timesift/individual_clouds/*/dem.tif"
    "/data2/elilouis/timesift/baker-ee-many/mixed_timesift/individual_clouds_2_4/*/dem.tif"
)
modern_dem_files = [
    "/data2/elilouis/hsfm-geomorph/data/reference_dem_highres/baker/raw_tifs/baker_2015/2015.tif"
]

# %%
historical_dem_files

# %% [markdown]
# # Extract data from DEMs

# %%
profiles_gdf = gpd.GeoDataFrame()

# %%
for dem_file in historical_dem_files + modern_dem_files:
    date_str = dem_file.split('/')[-2]
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
            gdf['File Name'] = dem_file
            
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
profiles_gdf

# %%
profiles_gdf.Date.unique()

# %%
# profiles_gdf['Date'] = profiles_gdf['Date'].apply(lambda x: 2015 if x=='baker' else int('19' + x[:2]))

# %% [markdown]
# # Mark points as glacial/not

# %% [markdown]
# ## Load glacier polygons

# %%
from shapely.geometry.collection import GeometryCollection


# %%
glaciers_gdf = gpd.read_file('/data2/elilouis/hsfm-geomorph/data/mt_baker_mass_wasted/glacier_polygons_combined.geojson')
glaciers_gdf.geometry = glaciers_gdf.geometry.apply(lambda x: x if x else GeometryCollection())
glaciers_gdf = glaciers_gdf.to_crs(epsg=32610)

glaciers_gdf['year'] = glaciers_gdf['year'].astype(int)

# %% [markdown]
# ### Create a year column in the profiles_gdf column 
#
# this may need to be changed depending on how directories to the DEMs are labeled

# %%
profiles_gdf['year'] = profiles_gdf['Date'].apply(lambda s: int('19' + s[:2]) if not s == 'baker_2015' else s)

# %%
profiles_gdf.year.unique()

# %%
glaciers_gdf.year.unique()


# %% [markdown]
# ### Mark profile points as glacier/non-glacier

# %%
def point_is_glacier(year, point):
    this_years_glaciers = glaciers_gdf[glaciers_gdf.year == year]
    return any(
        this_years_glaciers.geometry.apply(lambda geom: geom.contains(point))
    )


# %%
profiles_gdf['glacial'] = profiles_gdf.apply(
    lambda row: point_is_glacier(row.year, row.geometry),
    axis=1
)

# %%
profiles_gdf.glacial.unique()

# %% [markdown]
# # Plot profile view

# %% [markdown]
# Take advantage of Altair's facet grid (obscures detail)

# %% [markdown]
# ## All locations

# %%
src = profiles_gdf.drop('geometry', axis=1)
alt.Chart(src).mark_line().encode(
    alt.X('Upstream Distance:Q'),
    alt.Y('Elevation:Q', scale=alt.Scale(zero=False)),
    alt.Color('Date:N'),
    alt.StrokeDash('glacial:N'),
    alt.Facet('Location', columns=3)
).resolve_scale(x='independent', y='independent')

# %% [markdown]
# ## Rainbow

# %%
src = profiles_gdf.drop('geometry', axis=1)
src = src[src['Location'].str.contains('rainbow')]
src = src[src['Date'].isin([
    '70_9.0_29.0',
    '79_10.0_6.0',
    '91_9.0_9.0'
])]
alt.Chart(src).mark_line().encode(
    alt.X('Upstream Distance:Q'),
    alt.Y('Elevation:Q', scale=alt.Scale(zero=False)),
    alt.Color('Date:N'),
    alt.StrokeDash('glacial:N'),
    alt.Facet('Location', columns=3)
)

# %%
src = profiles_gdf.drop('geometry', axis=1)
src = src[src['Location'].str.contains('rainbow')]
src = src[~src.glacial]
alt.Chart(src).mark_line(size=1).encode(
    alt.X('Upstream Distance:Q'),
    alt.Y('Elevation:Q', scale=alt.Scale(zero=False)),
    alt.Color('Date:N'),
    alt.Facet('Location', columns=3)
)

# %% [markdown]
# ## Roosevelt

# %%
src = profiles_gdf.drop('geometry', axis=1)
src = src[src['Location'].str.contains('roosevelt')]
src = src[src['Date'].isin([
    '70_9.0_29.0',
    '79_10.0_6.0',
    '92_9.0_18.0'
])]
alt.Chart(src).mark_line().encode(
    alt.X('Upstream Distance:Q'),
    alt.Y('Elevation:Q', scale=alt.Scale(zero=False)),
    alt.Color('Date:N'),
    alt.StrokeDash('glacial:N'),
    alt.Facet('Location', columns=3)
)

# %%
src = profiles_gdf.drop('geometry', axis=1)
src = src[src['Location'].str.contains('roosevelt')]
src = src[~src.glacial]
alt.Chart(src).mark_line(size=1).encode(
    alt.X('Upstream Distance:Q'),
    alt.Y('Elevation:Q', scale=alt.Scale(zero=False)),
    alt.Color('Date:N'),
    alt.Facet('Location', columns=3)
)

# %% [markdown]
# ## Coleman

# %%
src = profiles_gdf.drop('geometry', axis=1)
src = src[src['Location'].str.contains('coleman')]
alt.Chart(src).mark_line().encode(
    alt.X('Upstream Distance:Q'),
    alt.Y('Elevation:Q', scale=alt.Scale(zero=False)),
    alt.Color('Date:N'),
    alt.StrokeDash('glacial:N'),
    alt.Facet('Location', columns=3)
).resolve_scale(x='independent', y='independent')

# %%
src = profiles_gdf.drop('geometry', axis=1)
src = src[src['Location'].str.contains('coleman')]
src = src[~src.glacial]
alt.Chart(src).mark_line(size=1).encode(
    alt.X('Upstream Distance:Q'),
    alt.Y('Elevation:Q', scale=alt.Scale(zero=False)),
    alt.Color('Date:N'),
    alt.Facet('Location', columns=3)
).resolve_scale(x='independent', y='independent')

# %% [markdown]
# ## Deming

# %%
src = profiles_gdf.drop('geometry', axis=1)
src = src[src['Location'] == 'deming']
src = src[src['Date'].isin([
    '70_9.0_29.0',
    '79_10.0_6.0',
    '91_9.0_9.0'
])]
alt.Chart(src).mark_line().encode(
    alt.X('Upstream Distance:Q'),
    alt.Y('Elevation:Q', scale=alt.Scale(zero=False)),
    alt.Color('Date:N'),
    alt.StrokeDash('glacial:N'),
    alt.Facet('Location', columns=3)
).resolve_scale(x='independent', y='independent')

# %%
src = profiles_gdf.drop('geometry', axis=1)
src = src[src['Location'] == 'deming']
src = src[~src.glacial]
alt.Chart(src).mark_line(size=1).encode(
    alt.X('Upstream Distance:Q'),
    alt.Y('Elevation:Q', scale=alt.Scale(zero=False)),
    alt.Color('Date:N'),
    alt.Facet('Location', columns=3)
).resolve_scale(x='independent', y='independent')

# %% [markdown]
# ## Mazama

# %%
src = profiles_gdf.drop('geometry', axis=1)
src = src[src['Location'] == 'mazama']
src = src[src['Date'].isin([
    '70_9.0_29.0',
    '79_10.0_6.0',
    '92_9.0_18.0'
])]
alt.Chart(src).mark_line().encode(
    alt.X('Upstream Distance:Q'),
    alt.Y('Elevation:Q', scale=alt.Scale(zero=False)),
    alt.Color('Date:N'),
    alt.StrokeDash('glacial:N'),
    alt.Facet('Location', columns=3)
).resolve_scale(x='independent', y='independent')

# %%
src = profiles_gdf.drop('geometry', axis=1)
src = src[src['Location'] == 'mazama']
alt.Chart(src).mark_line().encode(
    alt.X('Upstream Distance:Q'),
    alt.Y('Elevation:Q', scale=alt.Scale(zero=False)),
    alt.Color('Date:N'),
    alt.StrokeDash('glacial:N'),
    alt.Facet('Location', columns=3)
).resolve_scale(x='independent', y='independent')

# %%
src = profiles_gdf.drop('geometry', axis=1)
src = src[src['Location'] == 'mazama']
src = src[~src.glacial]
alt.Chart(src).mark_line(size=1).encode(
    alt.X('Upstream Distance:Q'),
    alt.Y('Elevation:Q', scale=alt.Scale(zero=False)),
    alt.Color('Date:N'),
    alt.StrokeDash('glacial:N'),
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
profiles_gdf.Date.unique()

# %%
test = pd.concat([
    profiles_gdf.groupby(['Location', 'Date']).get_group(('boulder1', '70_9.0_29.0')),
    profiles_gdf.groupby(['Location', 'Date']).get_group(('boulder1', 'baker_2015'))
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

# %% [markdown]
# ## Mazama

# %%
LOCATION = 'mazama'
xsection_file = [f for f in xsection_files if LOCATION in f][0]
xsection_gdf = gpd.read_file(xsection_file)
xsection_gdf
ax = xsection_gdf.plot(color='red')
ax.set_xlim(
    ax.get_xlim()[0]-750,
    ax.get_xlim()[1]+750
)   
plt.title(LOCATION)
ctx.add_basemap(ax, zoom=14, crs=xsection_gdf.crs, 
#                     source=ctx.providers.OpenStreetMap.Mapnik
                    source=ctx.providers.Esri.WorldImagery
                   )

# %%
LOCATION = 'mazama'
src = profiles_gdf[profiles_gdf['Location'] == LOCATION]
src = src[src['Date'].isin([
    '70_9.0_29.0',
    '79_10.0_6.0',
    '92_9.0_18.0',
    'baker_2015'
])]
src.geometry = src.geometry.apply(lambda point: None if all([np.isnan(point.x), np.isnan(point.y)]) else point)
line_gdf = gpd.GeoDataFrame(src.groupby('Date').apply(lambda x: geometry.LineString(x.geometry.dropna().tolist())).reset_index().rename({0:'geometry'}, axis=1), crs=src.crs)
line_gdf = line_gdf[~line_gdf.geometry.is_empty]
line_gdf['Location'] = LOCATION
ax = line_gdf.plot(column='Date', categorical=True, legend=True)
ax.set_xlim(
    ax.get_xlim()[0]-750,
    ax.get_xlim()[1]+750
)   
ax.set_xticks([])
ax.set_yticks([])
plt.title(LOCATION)
ctx.add_basemap(ax, zoom=14, crs=line_gdf.crs, 
#                     source=ctx.providers.OpenStreetMap.Mapnik
                    source=ctx.providers.Esri.WorldImagery
                   )

# %%
LOCATION = 'mazama'
src = profiles_gdf[profiles_gdf['Location'] == LOCATION]
src = src[~src['Date'].isin(['47_9.0_14.0', '50_9.0_2.0', '79_9.0_14.0'])]

src.geometry = src.geometry.apply(lambda point: None if all([np.isnan(point.x), np.isnan(point.y)]) else point)
line_gdf = gpd.GeoDataFrame(src.groupby('Date').apply(lambda x: geometry.LineString(x.geometry.dropna().tolist())).reset_index().rename({0:'geometry'}, axis=1), crs=src.crs)
line_gdf = line_gdf[~line_gdf.geometry.is_empty]
line_gdf['Location'] = LOCATION
ax = line_gdf.plot(column='Date', categorical=True, legend=True)
ax.set_xlim(
    ax.get_xlim()[0]-750,
    ax.get_xlim()[1]+750
)   
plt.title(LOCATION)
ctx.add_basemap(ax, zoom=14, crs=line_gdf.crs, 
#                     source=ctx.providers.OpenStreetMap.Mapnik
                    source=ctx.providers.Esri.WorldImagery
                   )

# %%
old_ylims = ax.get_ylim()
old_xlims = ax.get_xlim()

src_glaciers = glaciers_gdf[glaciers_gdf['Name'].notnull()]
src_glaciers = src_glaciers[src_glaciers['Name'].str.contains('Mazama')]
src_glaciers = src_glaciers[src_glaciers.year.isin([1970, 1977, 1990])]
ax = src_glaciers.plot(
    column='year', categorical=True, facecolor="none", edgecolor='k', legend=True, linewidth=2,
    figsize=(5,5)
)
ax.set_xlim(old_xlims)
ax.set_ylim( 5.408e6, 5.410e6)
ax.set_yticks([])
ax.set_xticks([])
ctx.add_basemap(ax, zoom=14, crs=line_gdf.crs, source=ctx.providers.Esri.WorldImagery)

# %% [markdown]
# ## Deming

# %%
LOCATION = 'deming'
xsection_file = [f for f in xsection_files if LOCATION in f][0]
xsection_gdf = gpd.read_file(xsection_file)
xsection_gdf
ax = xsection_gdf.plot(color='red')
plt.title(LOCATION)
ctx.add_basemap(ax, zoom=14, crs=line_gdf.crs, 
#                     source=ctx.providers.OpenStreetMap.Mapnik
                    source=ctx.providers.Esri.WorldImagery
                   )

# %%
LOCATION = 'deming'
src = profiles_gdf[profiles_gdf['Location'] == LOCATION]
src = src[src['Date'].isin([
    '70_9.0_29.0',
    '79_10.0_6.0',
    '91_9.0_9.0'
])]
src.geometry = src.geometry.apply(lambda point: None if all([np.isnan(point.x), np.isnan(point.y)]) else point)
line_gdf = gpd.GeoDataFrame(src.groupby('Date').apply(lambda x: geometry.LineString(x.geometry.dropna().tolist())).reset_index().rename({0:'geometry'}, axis=1), crs=src.crs)
line_gdf = line_gdf[~line_gdf.geometry.is_empty]
line_gdf['Location'] = LOCATION
ax = line_gdf.plot(column='Date', categorical=True, legend=True)
plt.title(LOCATION)
ctx.add_basemap(ax, zoom=14, crs=line_gdf.crs, 
#                     source=ctx.providers.OpenStreetMap.Mapnik
                    source=ctx.providers.Esri.WorldImagery
                   )

# %%
old_ylims = ax.get_ylim()
old_xlims = ax.get_xlim()

src_glaciers = glaciers_gdf[glaciers_gdf['Name'].notnull()]
src_glaciers = src_glaciers[src_glaciers['Name'].str.contains('Deming')]
src_glaciers = src_glaciers[src_glaciers.year.isin([1970, 1987, 1990, 1991])]
ax = src_glaciers.plot(column='year', categorical=True, facecolor="none", edgecolor='k', legend=True, linewidth=2)
ax.set_xlim(old_xlims)
ax.set_ylim(old_ylims)
ax.set_yticks([])
ax.set_xticks([])
ctx.add_basemap(ax, zoom=14, crs=line_gdf.crs, source=ctx.providers.Esri.WorldImagery)

# %% [markdown]
# ## Coleman and Roosevelt

# %%
LOCATION = 'coleman1'
LOCATION2 = 'coleman2'
LOCATION3 = 'roosevelt'

xsection_file = [f for f in xsection_files if LOCATION in f][0]
xsection_file2 = [f for f in xsection_files if LOCATION2 in f][0]
xsection_file3 = [f for f in xsection_files if LOCATION3 in f][0]

xsection_gdf = gpd.read_file(xsection_file)
xsection_gdf2 = gpd.read_file(xsection_file2)
xsection_gdf3 = gpd.read_file(xsection_file3)

xsection_gdf = xsection_gdf.append(xsection_gdf2).append(xsection_gdf3)

ax = xsection_gdf.plot(color='red')
plt.title('Coleman and Roosevelt')
ctx.add_basemap(ax, zoom=14, crs=line_gdf.crs, 
#                     source=ctx.providers.OpenStreetMap.Mapnik
                    source=ctx.providers.Esri.WorldImagery
                   )

# %%
LOCATION = 'coleman1'
LOCATION2 = 'coleman2'
LOCATION3 = 'roosevelt'

src = profiles_gdf[profiles_gdf['Location'] == LOCATION]
src2 = profiles_gdf[profiles_gdf['Location'] == LOCATION2]
src3 = profiles_gdf[profiles_gdf['Location'] == LOCATION3]

src.geometry = src.geometry.apply(lambda point: None if all([np.isnan(point.x), np.isnan(point.y)]) else point)
src2.geometry = src2.geometry.apply(lambda point: None if all([np.isnan(point.x), np.isnan(point.y)]) else point)
src3.geometry = src3.geometry.apply(lambda point: None if all([np.isnan(point.x), np.isnan(point.y)]) else point)

line_gdf = gpd.GeoDataFrame(src.groupby('Date').apply(lambda x: geometry.LineString(x.geometry.dropna().tolist())).reset_index().rename({0:'geometry'}, axis=1), crs=src.crs)
line_gdf2 = gpd.GeoDataFrame(src2.groupby('Date').apply(lambda x: geometry.LineString(x.geometry.dropna().tolist())).reset_index().rename({0:'geometry'}, axis=1), crs=src.crs)
line_gdf3 = gpd.GeoDataFrame(src3.groupby('Date').apply(lambda x: geometry.LineString(x.geometry.dropna().tolist())).reset_index().rename({0:'geometry'}, axis=1), crs=src.crs)

line_gdf = line_gdf[~line_gdf.geometry.is_empty]
line_gdf2 = line_gdf2[~line_gdf2.geometry.is_empty]
line_gdf3 = line_gdf3[~line_gdf3.geometry.is_empty]

ax = line_gdf.plot(column='Date', categorical=True, legend=True)
line_gdf2.plot(column='Date', categorical=True, legend=True, ax=ax)
line_gdf3.plot(column='Date', categorical=True, legend=True, ax=ax)

plt.title('Coleman and Roosevelt')
ctx.add_basemap(ax, zoom=14, crs=line_gdf.crs, 
#                     source=ctx.providers.OpenStreetMap.Mapnik
                    source=ctx.providers.Esri.WorldImagery
                   )

# %%
old_ylims = ax.get_ylim()
old_xlims = ax.get_xlim()

src_glaciers = glaciers_gdf[glaciers_gdf['Name'].notnull()]
src_glaciers1 = src_glaciers[src_glaciers['Name'].str.contains('Coleman')]
src_glaciers2 = src_glaciers[src_glaciers['Name'].str.contains('Roosevelt')]

src_glaciers = src_glaciers1.append(src_glaciers2)

src_glaciers = src_glaciers[src_glaciers.year.isin([1970, 1979, 1987, 1990])]
ax = src_glaciers.plot(column='year', categorical=True, facecolor="none", edgecolor='k', legend=True, linewidth=2)
ax.set_xlim(old_xlims)
ax.set_ylim(old_ylims)
ax.set_yticks([])
ax.set_xticks([])
ctx.add_basemap(ax, zoom=14, crs=line_gdf.crs, source=ctx.providers.Esri.WorldImagery)

# %% [markdown]
# ## Rainbow

# %%
LOCATION = 'rainbow'
xsection_file = [f for f in xsection_files if LOCATION in f][0]
xsection_gdf = gpd.read_file(xsection_file)
xsection_gdf
ax = xsection_gdf.plot(color='red')
plt.title(LOCATION)
ctx.add_basemap(ax, zoom=14, crs=line_gdf.crs, 
#                     source=ctx.providers.OpenStreetMap.Mapnik
                    source=ctx.providers.Esri.WorldImagery
                   )

# %%
LOCATION = 'rainbow'
src = profiles_gdf[profiles_gdf['Location'] == LOCATION]
src.geometry = src.geometry.apply(lambda point: None if all([np.isnan(point.x), np.isnan(point.y)]) else point)
line_gdf = gpd.GeoDataFrame(src.groupby('Date').apply(lambda x: geometry.LineString(x.geometry.dropna().tolist())).reset_index().rename({0:'geometry'}, axis=1), crs=src.crs)
line_gdf = line_gdf[~line_gdf.geometry.is_empty]
line_gdf['Location'] = LOCATION
ax = line_gdf.plot(column='Date', categorical=True, legend=True)
plt.title(LOCATION)
ctx.add_basemap(ax, zoom=14, crs=line_gdf.crs, 
#                     source=ctx.providers.OpenStreetMap.Mapnik
                    source=ctx.providers.Esri.WorldImagery
                   )

# %%
old_ylims = ax.get_ylim()
old_xlims = ax.get_xlim()

src_glaciers = glaciers_gdf[glaciers_gdf['Name'].notnull()]
src_glaciers = src_glaciers[src_glaciers['Name'].str.contains('Rainbow')]
src_glaciers = src_glaciers[src_glaciers.year.isin([1970, 1977, 1990, 1991])]
ax = src_glaciers.plot(column='year', categorical=True, facecolor="none", edgecolor='k', legend=True, linewidth=2)
ax.set_xlim(old_xlims)
ax.set_ylim(old_ylims)
ax.set_yticks([])
ax.set_xticks([])
ctx.add_basemap(ax, zoom=14, crs=line_gdf.crs, source=ctx.providers.Esri.WorldImagery)

# %% [markdown]
# ## All locations

# %%
for loc, gdf in profiles_gdf.groupby('Location'):
    # Replace Point(nan, nan) with None
    gdf.geometry = gdf.geometry.apply(lambda point: None if all([np.isnan(point.x), np.isnan(point.y)]) else point)

    line_gdf = gpd.GeoDataFrame(gdf.groupby('Date').apply(lambda x: geometry.LineString(x.geometry.dropna().tolist())).reset_index().rename({0:'geometry'}, axis=1), crs=gdf.crs)
    line_gdf = line_gdf[~line_gdf.geometry.is_empty]
    line_gdf['Location'] = loc
    ax = line_gdf.plot(column='Date', categorical=True, legend=True)
    plt.title(loc)
    ctx.add_basemap(ax, zoom=14, crs=line_gdf.crs, 
#                     source=ctx.providers.OpenStreetMap.Mapnik
                    source=ctx.providers.Esri.WorldImagery
                   )

# %% [markdown]
# # Calculate Residuals 
# With modern LIDAR DEM

# %%
groups = profiles_gdf.groupby('Location').apply(lambda x: x.groupby('Date'))

# %%
residual_date_key = 'baker_2015'

# %%
diff_df = pd.DataFrame()
for grouped_by_date, index in zip(groups, groups.index):    
    def create_diff_df(df, residual_df):
        merged = df.merge(residual_df, on='Upstream Distance')
        merged['Elevation Difference'] = (merged['Elevation_y'] - merged['Elevation_x'])
        return merged
    residual_base_df = grouped_by_date.get_group(residual_date_key)
    difference_df = grouped_by_date.apply(
        lambda per_date_and_loc_df: create_diff_df(per_date_and_loc_df, residual_base_df)    
    )
    diff_df = diff_df.append(difference_df)

# %%
diff_df = diff_df[[
    'Date_x', 'Location_x', 'Upstream Distance', 'Elevation Difference', 'glacial_x', 'glacial_y'
]].rename(
    {'Date_x': 'Date', 'Location_x': 'Location'},
    axis=1
)
diff_df['glacial'] = diff_df.apply(
    lambda row:
    row['glacial_x'] or row['glacial_y'],
    axis=1
)

# %% [markdown]
# # Plot all locations

# %%
alt.Chart(diff_df).mark_line().encode(
    alt.X('Upstream Distance:Q'),
    alt.Y('Elevation Difference:Q', scale=alt.Scale(zero=False)),
    alt.Color('Date:O'),
    alt.Facet('Location', columns=3)
).resolve_scale(x='independent', y='independent')

# %% [markdown]
# # Plot Single Locations

# %% [markdown]
# ## Mazama

# %%
src = diff_df[diff_df.Location == 'mazama']
src = src[src['Date'].isin([
    '70_9.0_29.0',
    '79_10.0_6.0',
    '92_9.0_18.0'
])]
src = src[np.abs(src['Elevation Difference']) < 5]
alt.Chart(src).mark_line().encode(
    alt.X('Upstream Distance:Q'),
    alt.Y('Elevation Difference:Q', scale=alt.Scale(zero=False),
          title='2015 Elevation - Historical Elevation (m)',
         ),
    alt.Color('Date:O', scale = alt.Scale(scheme='viridis')),
    alt.StrokeDash('glacial:N')
).resolve_scale(x='independent', y='independent')

# %%
src = diff_df[diff_df.Location == 'mazama']
src = src[src['Date'].isin([
    '70_9.0_29.0',
    '79_10.0_6.0',
    '92_9.0_18.0'
])]
src = src[np.abs(src['Elevation Difference']) < 5]

alt.Chart(src).mark_line().transform_window(
    rolling_mean='mean(Elevation Difference)',
    frame=[-10, 10]
).encode(
    alt.X('Upstream Distance:Q'),
    alt.Y(
        'rolling_mean:Q', 
        title='2015 Elevation - Historical Elevation (m)',
        scale=alt.Scale(zero=False, domain=[-5, 5])
    ),
    alt.Color('Date:O', scale = alt.Scale(scheme='viridis')),
    alt.StrokeDash('glacial:N')
).resolve_scale(x='independent', y='independent')


# %% [markdown]
# ## Deming

# %%
src = diff_df[diff_df.Location == 'deming']
src = src[src['Date'].isin([
    '70_9.0_29.0',
    '79_10.0_6.0',
    '91_9.0_9.0'
])]
src = src[np.abs(src['Elevation Difference']) < 5]
alt.Chart(src).mark_line().encode(
    alt.X('Upstream Distance:Q'),
    alt.Y('Elevation Difference:Q', scale=alt.Scale(zero=False),
          title='2015 Elevation - Historical Elevation (m)',
         ),
    alt.Color('Date:O', scale = alt.Scale(scheme='viridis')),
    alt.StrokeDash('glacial:N')
).resolve_scale(x='independent', y='independent')

# %%
src = diff_df[diff_df.Location == 'deming']
src = src[src['Date'].isin([
    '70_9.0_29.0',
    '79_10.0_6.0',
    '91_9.0_9.0'
])]
src = src[np.abs(src['Elevation Difference']) < 5]

alt.Chart(src).mark_line().transform_window(
    rolling_mean='mean(Elevation Difference)',
    frame=[-10, 10]
).encode(
    alt.X('Upstream Distance:Q'),
    alt.Y(
        'rolling_mean:Q', 
        title='2015 Elevation - Historical Elevation (m)',
        scale=alt.Scale(zero=False, domain=[-5, 5])
    ),
    alt.Color('Date:O', scale = alt.Scale(scheme='viridis')),
    alt.StrokeDash('glacial:N')
).resolve_scale(x='independent', y='independent')

# %%
