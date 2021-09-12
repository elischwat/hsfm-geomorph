# %%
import os
import io
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path

import altair as alt
import rioxarray as rio
from pyproj import Proj, transform
import contextily as ctx
import rioxarray as rix

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 10.0)

import sys
sys.path.append('/home/elilouis/hsfm-geomorph/dem-analysis/')
import profiling_tools

# %%
# !find . -name "*.json"

# %%
import json
with open("hood_eliot_config.json") as f:
    input_config = json.load(f)

# %%
input_config

# %% [markdown]
# # Generate DEM of difference datasets

# %%
import hsfm

# %%
len(list(sorted(input_config['dem file paths'].items())))


# %%
diff_dem_fns = []
dem_list = list(sorted(input_config['dem file paths'].items()))
for i in range(0, len(dem_list) - 1):
    print(f'DEM of difference from: {dem_list[i][0]} and {dem_list[i+1][0]}')
    diff_dem = hsfm.utils.difference_dems(dem_list[i+1][1], dem_list[i][1])
    print(f'Generated difference dem at {diff_dem}')
    new_diff_dem_fn = diff_dem.replace('.tif', f'-{Path(dem_list[i][1]).stem}.tif')
    os.rename(diff_dem, new_diff_dem_fn)
    print(f'Final DoD at {new_diff_dem_fn}')
    diff_dem_fns.append(new_diff_dem_fn)
    print()

# %%
diff_dem_fns

# %% [markdown]
# ### Create dictionary of difference rasters

# %% [markdown]
# Create Keys

# %%
keys = list(sorted(input_config['dem file paths'].keys()))

# %%
diff_names = [
    f'{keys[i]}/{keys[i+1]}' for i in range(len(keys)-1)
]
diff_names

# %%
list(zip(diff_dem_fns, diff_names))

# %% [markdown]
# Populate the dictionary

# %%
diff_rasters = {}
for fn, key in zip(diff_dem_fns, diff_names):
    diff_rasters[
        key
    ] = rix.open_rasterio(fn , masked=True)

# %%
for r in diff_rasters.values():
    print(r.rio.resolution())

# %%
import math

# %%
rasters = diff_rasters.copy()


# %%
# fig, axes = plt.subplots(
#         ncols=5,
#         nrows=math.ceil(len(diff_rasters)/5),
#         figsize=(20,5),
#         sharex = True,
#         sharey = True,
#         constrained_layout=True
#     )
# ims = []
# cmap = plt.cm.PuOr
# cmap.set_bad('grey',1.)

# %%
def plot_n(rasters):
    fig, axes = plt.subplots(
        ncols=4,
        nrows=math.ceil(len(diff_rasters)/5),
        figsize=(16,5),
        sharex = True,
        sharey = True,
        constrained_layout=True
    )
    ims = []
    cmap = plt.cm.PuOr
    cmap.set_bad('grey',1.)
    axes = axes.flatten()
    for (date_str,raster), ax in zip(rasters.items(), axes):
        ax.set_title(date_str)
        ax.set_aspect('equal')
        ax.set_facecolor('grey')
        im = raster.plot(
            ax=ax,
            add_labels = False,
            add_colorbar = False,
            vmin = -10,
            vmax = 10,
            cmap=cmap
        )
        ims.append(im)

    cbar = fig.colorbar(ims[0], ax=axes, shrink=0.5, cmap=cmap, label='Elevation Change (m)')
    plt.locator_params(nbins=4)


# %%
type(list(diff_rasters.items())[0][1])

# %%
plot_n(diff_rasters)

# %% [markdown]
# # Mask DEMs

# %% [markdown]
# ## Crop Out Glacier Polygon (maximum polygon)

# %%
all_glacier_polygons = gpd.read_file(input_config['glacier polygons'])

# %%
all_glacier_polygons

# %%
all_glacier_polygons = gpd.read_file(input_config['glacier polygons'])
all_glacier_polygons.id = all_glacier_polygons.id.astype('int')
year_of_glacial_max = input_config["glacial maximum year"]
print(f'Using maximum glacier polygons from {year_of_glacial_max}')
glacier_polygons = all_glacier_polygons.loc[all_glacier_polygons['id'] == year_of_glacial_max].geometry

# %%
diff_rasters = {
    k: v.rio.clip(glacier_polygons, invert=True) for k, v in diff_rasters.items()

}


# %%
plot_n(diff_rasters)

# %% [markdown]
# ## Mask with Study Area Boundary

# %%
lia_boundary_polygon = gpd.read_file(input_config['study area boundary']).geometry

# %%
diff_rasters = {
    k: v.rio.clip(lia_boundary_polygon) for k, v in diff_rasters.items()

}


# %%
plot_n(diff_rasters)

# %% [markdown]
# # Analysis and Viz

# %% [markdown]
# ## Net Change and Total Area of diff datasets

# %%
areas = []
net_changes = []

# %% [markdown]
# # WHATS UP THE NEGATIVE RESOLUTION???

# %%
for raster in diff_rasters.values():    
    values = raster.values.flatten()
    values = values[~np.isnan(values)]
    print('N Values: ' + str(len(values)))
    res = raster.rio.resolution()
    print('Resolution: ' + str(res))
    total_area = (np.full(len(values), 1)*res[0]*res[1]).sum()
    net_change = (-(res[0]*res[1])*values).sum()
    net_changes.append(net_change)
    areas.append(total_area)
    print('Total Area Measured: ' + str(total_area))
    print('Net Change: ' + str(net_change))

# %%
diff_rasters.keys()

# %%
calculation_df = pd.DataFrame({
        'Total Net Change': np.array(net_changes),
        'Measured Area': -np.array(areas),
        'Date Intervals': list(diff_rasters.keys())
})
calculation_df = calculation_df.reset_index()

# %%
from datetime import timedelta

# %%
calculation_df
calculation_df['Start Date'] = pd.to_datetime(
    calculation_df['Date Intervals'].str.split('/').apply(lambda x: x[0].split('-')[0]).astype(int), 
    format='%Y'
)
calculation_df['End Date'] = pd.to_datetime(
    calculation_df['Date Intervals'].str.split('/').apply(lambda x: x[1].split('-')[0]).astype(int), 
    format='%Y'
)
calculation_df['Years in Interval'] = ((
    calculation_df['End Date'] - calculation_df['Start Date']
) / timedelta(days=365)).astype(int)
calculation_df['Annual Net Change'] = calculation_df['Total Net Change']/calculation_df['Years in Interval']
calculation_df

# %%
alt.Chart(calculation_df).mark_bar().encode(
    x=alt.X('Start Date:T', title='Date'),
    x2='End Date:T',
    y=alt.Y('Annual Net Change:Q', title='Annual Rate of Volumetric Change  (m^3/year)')
)

# %%
alt.Chart(calculation_df).mark_bar(
).encode(
    x='index:O',
    y='Measured Area:Q'
)

# %%
calculation_df['Start Date'] = calculation_df['Date Intervals'].apply(lambda x: x.split('/')[0])
calculation_df['End Date'] = calculation_df['Date Intervals'].apply(lambda x: x.split('/')[1])

# %%
calculation_df

# %% [markdown]
# ## Distributions of DoDs

# %%
alt.data_transformers.enable('default', max_rows=None)

# %%
df = pd.DataFrame()

# %%
for dates, raster in diff_rasters.items():
    ls = []
    data = raster.values.flatten()
    data = data[~np.isnan(data)]
    sub_df = pd.DataFrame(
        data
    ).dropna().rename({0:'values'}, axis=1)
    sub_df['date'] = dates
    df = df.append(
        sub_df
    )

# %% [markdown]
# alt.Chart(df).mark_bar().encode(
#     x=alt.X("values", bin=alt.Bin(extent=[-20, 20], step=0.5)),    
#     y='count()'
# ).properties(width=250, height=200).facet('date:O', columns=2)

# %% [markdown]
# ## Profiles

# %%
lines_gdf = gpd.read_file(input_config['profiles'])
lines_gdf

# %%
ax = lines_gdf.plot()
ctx.add_basemap(ax, crs=lines_gdf.crs, source=ctx.providers.OpenStreetMap.Mapnik)
lines_gdf.apply(lambda x: ax.annotate(s=x['id'], xy=x.geometry.centroid.coords[0], size=10),axis=1);
plt.xticks([]); plt.yticks([])

# %%
dem_files = list(input_config['dem file paths'].values())
dem_files

# %% [markdown]
# Query DEMs for profile data and generate a big dataframe

# %%
df_list = []

for key, linestring in lines_gdf.iterrows():
    for dem_file in dem_files:
        one_profile_df = profiling_tools.get_raster_values_for_line(
            profiling_tools.increase_line_complexity(linestring.geometry, 100), 
            dem_file, 
            band=1
        )
        one_profile_df['Date'] = Path(dem_file).stem
        one_profile_df['Profile'] = linestring.id
        df_list.append(one_profile_df)
profiles_df = pd.concat(df_list)

# %% [markdown]
#

# %% [markdown]
# Define glacial/nonglacial for each point in all profiles

# %% [markdown]
# Define function for querying the glacier polygons to label each point as "glacial" or not. Uses the glacier polygons df./

# %%
all_glacier_polygons.id = all_glacier_polygons.id.astype('int')


# %%
def point_and_date_is_glacial(point, date):
    #Extract year to match the date-identifier in the glacial polygons df
    date = int(date.split('-')[0])
    relevant_glacier_polygons = all_glacier_polygons[all_glacier_polygons['id'] == date]
    return any(relevant_glacier_polygons.geometry.apply(lambda x: x.contains(point)))


# %%
all_glacier_polygons

# %%
profiles_df

# %%
profiles_df['geometry'] = gpd.points_from_xy(profiles_df.X, profiles_df.Y)
profiles_df['Glacial'] = profiles_df.apply(lambda row: point_and_date_is_glacial(row.geometry, row.Date), axis=1)
profiles_df.head()

# %% [markdown]
# Because facet charts cannot be layered, we can only look at one profile at a time

# %%
src = profiles_df.drop('geometry', axis=1)

alt.Chart(src).mark_line().encode(
    alt.X('path_distance:Q', title='Pathwise Distance (m)'),
    alt.Y('raster_value:Q', title='Elevation (m)', scale=alt.Scale(zero=False)),
    alt.Color('Date:O'),
    alt.Facet('Profile:O', columns=3, sort='ascending'),
    strokeDash='Glacial',
#     detail = 'Date:O' #this doesnt help???
).properties(width=300, height=200).resolve_scale(y='independent', x='independent')

# %%
src = profiles_df.drop('geometry', axis=1)
alt.Chart(src).mark_line().encode(
    alt.X('path_distance'), 
    alt.Y('raster_value', scale=alt.Scale(zero=False)),
    alt.Color('Date:O'),
    alt.Facet('Profile', columns=3)
).properties(width=300, height=200).resolve_scale(y='independent', x='independent')

# %% [markdown]
# ## Valley Centerline Elevation

# %% [markdown]
# Read in manually generated cross-section lines

# %% [markdown]
# NOTE: These cross sections must be in order from upstream to downstream!!! And the upstream most cross sections must intersect ALL DEMs for this to work (ie the extent of the DEMs only varies in the downstream area)

# %% [markdown]
# NOW THIS MUST ITERATE OVER A LIST OF FILES

# %%
valley_xsection_selected = input_config["valley_xsections"][0]

# %%
xsection_geoms = gpd.read_file(valley_xsection_selected)

# %%
ax = xsection_geoms.plot()
ctx.add_basemap(ax, crs=xsection_geoms.crs, source=ctx.providers.OpenStreetMap.Mapnik)
plt.xticks([]); plt.yticks([])


# %% [markdown]
# Convert MultiLineStrings to LineStrings

# %%
def multilinestring_to_linestring(mls):
    if type(mls)=='MultiLineString':
        assert len(mls) == 1, "MultiLineString can only be converted if size 1."
        return mls[0]
    else:
        return mls
xsection_geoms.geometry = xsection_geoms.geometry.apply(multilinestring_to_linestring)


# %%
profiles_gdf = gpd.GeoDataFrame()
for key, dem_fn in input_config['dem file paths'].items():
    gdf = profiling_tools.get_valley_lowline_from_xsections(xsection_geoms.geometry, dem_fn)
    gdf['Date'] = key
    profiles_gdf = profiles_gdf.append(gdf)

# %%
alt.Chart(profiles_gdf).mark_line().encode(
    alt.X('path_distance:Q', title='Upstream Distance (m)'),
    alt.Y('raster_value:Q', title='Elevation (m)', scale=alt.Scale(zero=False)),
    alt.Color('Date:N')
)

# %% [markdown]
# Make lines from all the points so we can look at how the "valley low/center lines" change laterally in time

# %%
from shapely.geometry import LineString


# %%
def linestring_from_points(df):
    return LineString(df.geometry)
valley_centerlines_gdf = profiles_gdf.groupby('Date').agg({'geometry': linestring_from_points})
valley_centerlines_gdf = gpd.GeoDataFrame(
    valley_centerlines_gdf
).set_crs(
    epsg=32610
).reset_index()

# %%
ax = valley_centerlines_gdf.plot(column='Date', legend=True, legend_kwds={'loc': "upper left"})
ctx.add_basemap(ax, crs=valley_centerlines_gdf.crs, source=ctx.providers.OpenStreetMap.Mapnik)
plt.xticks([]); plt.yticks([])

# %% [markdown]
# Recreate the above profiles_gdf plot but with dashed lines where the elevation is on glacier (use glacier polygons)

# %%
all_glacier_polygons

# %%
profiles_gdf['Glacial'] = profiles_gdf.apply(
    lambda row: point_and_date_is_glacial(row.geometry, row['Date']), axis=1
)

# %%
alt.Chart(profiles_gdf).mark_line().encode(
    alt.X('path_distance:Q', title='Upstream Distance (m)'),
    alt.Y('raster_value:Q', title='Elevation (m)', scale=alt.Scale(zero=False)),
    alt.Color('Date:N'),
    strokeDash='Glacial',
)
