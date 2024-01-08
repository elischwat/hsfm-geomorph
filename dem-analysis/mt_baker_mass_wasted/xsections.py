# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3.9.2 ('xdem')
#     language: python
#     name: python3
# ---

# %%
import geopandas as gpd
import shapely
import shapely
import rasterio
import glob
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append("..")
import datetime
import profiling_tools
import altair as alt
from altair import datum
import os
import json
import math

# %% [markdown]
# # Inputs
#
# Provide:
#
# * Input file path to file with cross section lines/polygons to extract low points/stream profile from
# * Output file path where low points will be saved
# * Input directory path to location of DEMs
# * Parameter LINE_COMPLEXITY which is the number of points that each cross-section line is split into. LINE_COMPLEXITY elevation points will be extracted from the DEM for each cross section line

# %% [markdown]
# If you use the arg, you must run from CLI like this
#
# ```
# HSFM_GEOMORPH_INPUT_FILE='inputs/mazama_inputs.json' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/xsections.ipynb  --output outputs/xsections_mazama.html
# ```

# %%
BASE_PATH = os.environ.get("HSFM_GEOMORPH_DATA_PATH")
print(f"retrieved base path: {BASE_PATH}")

# %%
# Or set an env arg:
if os.environ.get('HSFM_GEOMORPH_INPUT_FILE'):
    json_file_path = os.environ['HSFM_GEOMORPH_INPUT_FILE']
else:
    json_file_path = 'inputs/mazama_inputs.json'

# %%
with open(json_file_path, 'r') as j:
     params = json.loads(j.read())

# %%
params

# %%
TO_DROP = params['inputs']['TO_DROP']
valley_name = params['inputs']['valley_name']
# if this is defined, only data from these dates are analyed
XSECTIONS_INCLUDE = params["inputs"]["XSECTIONS_INCLUDE"]
input_xsections_file = os.path.join(BASE_PATH, params['xsections']['input_xsections_file'])
output_lowpoints_file = os.path.join(BASE_PATH, params['xsections']['output_lowpoints_file'])
output_streamlines_file = os.path.join(BASE_PATH, params['xsections']['output_streamlines_file'])
input_dems_path = os.path.join(BASE_PATH, params['inputs']['dems_path'])
glacier_polygons_file = os.path.join(BASE_PATH, params['inputs']['glacier_polygons_file'])
LINE_COMPLEXITY = params['xsections']['line_complexity']

group_slope_meters = params['xsections']['group_slope_meters']

# Used to strip date from dem file names
strip_time_format = params['inputs']['strip_time_format']

reference_dem_date = datetime.datetime.strptime(
    params['inputs']['reference_dem_date'], 
    strip_time_format
)

# %%
raster_fns = glob.glob(os.path.join(input_dems_path, "*.tif"))
if XSECTIONS_INCLUDE:
    raster_fns = [fn for fn in raster_fns if Path(fn).stem in XSECTIONS_INCLUDE]
else:
    raster_fns = [fn for fn in raster_fns if Path(fn).stem not in TO_DROP]
raster_fns

# %% [markdown]
# # Extract profiles from DEMs
#
# Along each cross-section, extract point with lowest elevation and calculate "path distance", the distance from the furthest downstream cross section line.

# %%
# read cross sections file into GeoDataframe
gdf = gpd.read_file(input_xsections_file)
# Increase the number of points in each line
gdf.geometry = gdf.geometry.apply(lambda g: profiling_tools.increase_line_complexity(g, LINE_COMPLEXITY))
# Find the centroid of each line
gdf['centroid'] = gdf.geometry.apply(lambda x: x.centroid)
# Get all points from the cross section lines and create a row for each point. 
gdf['coords'] = gdf.geometry.apply(lambda x: list(x.coords))
gdf = gdf.explode('coords', ignore_index=True)
# Make the coords column a shapely.geometry.Point type and drop the cross section geometries which we no longer need.
gdf['coords'] = gdf['coords'].apply(shapely.geometry.Point)
gdf.drop(columns=["geometry"])

combined_gdf = gpd.GeoDataFrame()

for raster in raster_fns:
    print(raster)
    # Extract an elevation value for each point
    with rasterio.open(raster) as src:
        new_gdf = gdf.copy()
        # pnts['values'] = [sample[0] for sample in src.sample(coords)]
        # gdf['elevation'] = gdf['coords'].apply(lambda x: [sample for sample in src.sample(x)])
        new_gdf['elevation'] = pd.Series([sample[0] for sample in src.sample(new_gdf["coords"].apply(lambda x: x.xy))])
        new_gdf['elevation'] = new_gdf['elevation'].apply(lambda x: np.nan if x == src.nodata else x)
        
    # Convert file name to datetime as per the provided format
    date = datetime.datetime.strptime(Path(raster).stem, strip_time_format)
    new_gdf['time'] = date

    # Find the point in each cross section line (identified by the ID column, with 0 meaning furthest downstream) with the lowest elevation
    new_gdf = new_gdf.sort_values('elevation').groupby('id').apply(pd.DataFrame.head, n=1)
    new_gdf['low_point_coords'] = new_gdf.apply(lambda row: None if np.isnan(row['elevation']) else row['coords'], axis=1)

    # Set the geometry to the centroid (of the cross-section lines) to calculate "path distance", distance upstream from the furthest downstream cross-section
    new_gdf.geometry = new_gdf["centroid"]
    new_gdf['path_distance'] = pd.Series(new_gdf.distance(
            gpd.GeoDataFrame(new_gdf.shift(1), crs=new_gdf.crs)
        ).fillna(0)).cumsum()
    
    combined_gdf = pd.concat([combined_gdf, new_gdf])

combined_gdf = combined_gdf.set_crs(crs=gdf.crs)


# %%
def create_path_dist_from_glaciers(df):
    path_distance_at_glacier = df.loc[df['n_from_glacial_max']==0, 'path_distance'].iloc[0]
    df['path_distance_from_glacier'] = path_distance_at_glacier - df['path_distance']
    return df
combined_gdf = combined_gdf.groupby('time').apply(create_path_dist_from_glaciers)

# %% [markdown]
# # Mark points as (non)glacial

# %%
glaciers_gdf = gpd.read_file(glacier_polygons_file)
glaciers_gdf = glaciers_gdf.to_crs(combined_gdf.crs)
glaciers_gdf['time'] = glaciers_gdf['year'].apply(lambda d: datetime.datetime.strptime(d, strip_time_format))


# %%

combined_gdf['glacial'] = combined_gdf.apply(
    lambda row: any(glaciers_gdf.loc[glaciers_gdf['time'] == row["time"], 'geometry'].apply(lambda g: g.contains(row['coords']))),
    axis=1
)

# %% [markdown]
# Plot elevation profiles (small)

# %%
src = combined_gdf[[ "time", "path_distance_from_glacier", "elevation", "glacial"]].reset_index(drop=True)
src['time'] = src['time'].apply(lambda x: x.strftime(strip_time_format))

alt.Chart(
    src
).mark_line().encode(
    alt.X("path_distance_from_glacier:Q", title="Distance downstream from observed glacial maximum"),
    alt.Y("elevation:Q", scale=alt.Scale(zero=False), impute=alt.ImputeParams(value=None), title="Valley floor elevation, in meters"),
    alt.Color("time:O"),
    alt.StrokeDash('glacial:N')
).properties(
    width = 600,
    # height = 600
)

# %%
src = combined_gdf[[ "time", "path_distance_from_glacier", "elevation", "glacial"]].reset_index(drop=True)
src['time'] = src['time'].apply(lambda x: x.strftime(strip_time_format))
alt.Chart(
    src
).mark_line().transform_filter(
    (datum.glacial == False)
).encode(
    alt.X("path_distance_from_glacier:Q", title="Distance downstream from observed glacial maximum"),
    alt.Y("elevation:Q", scale=alt.Scale(zero=False), impute=alt.ImputeParams(value=None), title="Valley floor elevation, in meters"),
    alt.Color("time:O"),
    alt.StrokeDash('glacial:N')
).properties(
    width = 600,
    # height = 600
)


# %% [markdown]
# # Calculate Residuals
#
# Calculate so accumulation is always positive with time, erosion is negative with time.

# %%
diff_df = pd.DataFrame()

combined_gdf_grouped = combined_gdf.reset_index(drop=True).groupby("time")
reference_group = combined_gdf_grouped.get_group(reference_dem_date)

for timestamp, df in combined_gdf_grouped:    
    if timestamp != reference_dem_date:
        print(timestamp)
        this_diff_df = df.copy()
        merged = df.merge(reference_group, on='path_distance_from_glacier')
        if timestamp > reference_dem_date:
            residual_values = merged['elevation_x'] - merged['elevation_y']
        else:
            residual_values = merged['elevation_y'] - merged['elevation_x']
        assert len(this_diff_df) == len(residual_values)
        this_diff_df['elevation_residual'] = list(residual_values)
        diff_df = pd.concat([diff_df, this_diff_df])

# %% [markdown]
# Plot elevation residuals, exclude glacier signals (large)

# %%
src = diff_df[['elevation', 'time', 'path_distance_from_glacier', 'glacial', 'elevation_residual', 'n_from_glacial_max']].reset_index().dropna()
src['time'] = src['time'].apply(lambda x: x.strftime(strip_time_format))
alt.Chart(
    src
).transform_filter(
    (datum.glacial == False)
).mark_circle().encode(
    alt.X("path_distance_from_glacier:Q"),
    alt.Y("elevation_residual:Q", scale=alt.Scale(zero=False), title='Elevation Residuals (rolling mean, 10 meter window', impute=alt.ImputeParams(value=None)),
    alt.Color("time:O", scale=alt.Scale(scheme='viridis')),
    alt.StrokeDash('glacial:N'),
    tooltip=['n_from_glacial_max', 'time']

).properties(
    width = 1400,
    height = 600,
    title="Elevation Residuals, relative to 2015 data."
).configure_legend(
    titleColor='black', 
    titleFontSize=12, 
    labelFontSize=16, 
    symbolStrokeWidth=4
).interactive()

# %% [markdown]
# Plot elevation residuals, exclude glacier signals (small)

# %%
src = diff_df[['elevation', 'time', 'path_distance_from_glacier', 'glacial', 'elevation_residual']].reset_index().dropna()
src['time'] = src['time'].apply(lambda x: x.strftime(strip_time_format))
alt.Chart(
    src
).transform_filter(
    (datum.glacial == False)
).mark_line().encode(
    alt.X("path_distance_from_glacier:Q", title="Distance downstream from observed glacial maximum"),
    alt.Y("elevation_residual:Q", scale=alt.Scale(zero=False), title="Valley floor elevation residuals relative to 2015 data, in meters", impute=alt.ImputeParams(value=None)),
    
    alt.Color("time:O", scale=alt.Scale(scheme='viridis')),
    alt.StrokeDash('glacial:N')
).properties(
    width = 600,
    # height = 600,
    title="Valley floor elevation residuals relative to 2015 data, in meters"
).configure_legend(
    titleColor='black', 
    titleFontSize=12, 
    labelFontSize=16, 
    symbolStrokeWidth=4
)

# %% [markdown]
# Plot elevation residuals, exclude glacier signals, rolling mean (small)

# %%
src = diff_df[['elevation', 'time', 'path_distance_from_glacier', 'glacial', 'elevation_residual']].reset_index().dropna()
src['time'] = src['time'].apply(lambda x: x.strftime(strip_time_format))
alt.Chart(
    src
).transform_filter(
    (datum.glacial == False)
).transform_window(
    rolling_mean='mean(elevation_residual)',
    groupby=['time'],
    frame=[-5,5]
).mark_line().encode(
    alt.X("path_distance_from_glacier:Q", title="Distance downstream from observed glacial maximum"),
    alt.Y(
        "rolling_mean:Q", 
        scale=alt.Scale(zero=False), 
        title=['Elevation residuals relative to 2015 data, in meters,',  '(rolling mean, 10 meter window)'],
        impute=alt.ImputeParams(value=None)
    ),
    alt.Color("time:O", scale=alt.Scale(scheme='viridis')),
    alt.StrokeDash('glacial:N')
).properties(
    width = 600,
    # height = 600,
    title="Elevation Residuals, relative to 2015 data."
).configure_legend(
    titleColor='black', 
    titleFontSize=12, 
    labelFontSize=16, 
    symbolStrokeWidth=4
)

# %% [markdown]
# Plot elevation residuals, include glacier signals (small)

# %%
src = diff_df[['elevation', 'time', 'path_distance_from_glacier', 'glacial', 'elevation_residual']].reset_index().dropna()
src['time'] = src['time'].apply(lambda x: x.strftime(strip_time_format))
alt.Chart(
    src
).mark_line().encode(
    alt.X("path_distance_from_glacier:Q", title="Distance downstream from observed glacial maximum"),
    alt.Y(
        "elevation_residual:Q", 
        scale=alt.Scale(zero=False), 
        title=['Elevation residuals relative to 2015 data, in meters'],
        impute=alt.ImputeParams(value=None)
    ),
    alt.Color("time:O", scale=alt.Scale(scheme='viridis')),
    alt.StrokeDash('glacial:N')
).properties(
    width = 600,
    # height = 600,
    title="Elevation Residuals, relative to 2015 data."
).configure_legend(
    titleColor='black', 
    titleFontSize=12, 
    labelFontSize=16, 
    symbolStrokeWidth=4
)

# %% [markdown]
# Plot elevation residuals, include glacier signals, rolling mean (small)

# %%
src = diff_df[['elevation', 'time', 'path_distance_from_glacier', 'glacial', 'elevation_residual']].reset_index().dropna()
src['time'] = src['time'].apply(lambda x: x.strftime(strip_time_format))
alt.Chart(
    src
).transform_window(
    rolling_mean='mean(elevation_residual)',
    groupby=['time'],
    frame=[-5,5]
).mark_line().encode(
    alt.X("path_distance_from_glacier:Q", title="Distance downstream from observed glacial maximum"),
        alt.Y(
        "rolling_mean:Q", 
        scale=alt.Scale(zero=False), 
        title=['Elevation residuals relative to 2015 data, in meters,',  '(rolling mean, 10 meter window)'],
        impute=alt.ImputeParams(value=None)
    ),
    
    alt.Color("time:O", scale=alt.Scale(scheme='viridis')),
    alt.StrokeDash('glacial:N')
).properties(
    width = 600,
    # height = 600,
    title="Elevation Residuals, relative to 2015 data."
).configure_legend(
    titleColor='black', 
    titleFontSize=12, 
    labelFontSize=16, 
    symbolStrokeWidth=4
)


# %% [markdown]
# # Calculate slope (negative)

# %%
def calculate_gradient(df):
    df['slope'] = - np.gradient(df['elevation'], df['path_distance_from_glacier'])
    return df

# slope_df = combined_gdf.groupby('time').apply(lambda df: calculate_gradient(df))
slope_df = combined_gdf.query('glacial == False').reset_index(drop=True).groupby('time').apply(lambda df: calculate_gradient(df))


# %% [markdown]
# Plot slope (small)

# %%
src = slope_df.reset_index(drop=True)[['elevation', 'time', 'path_distance_from_glacier', 'glacial', 'slope']].reset_index().dropna()
src['time'] = src['time'].apply(lambda x: x.strftime(strip_time_format))
alt.Chart(
    src
).transform_filter(
    (datum.glacial == False)
).mark_line().encode(
    alt.X("path_distance_from_glacier:Q", title="Distance downstream from observed glacial maximum"),
    alt.Y("slope:Q", scale=alt.Scale(zero=False), title='Valley floor slope', impute=alt.ImputeParams(value=None)),
    alt.Color("time:O", scale=alt.Scale(scheme='viridis')),
    alt.StrokeDash('glacial:N')
).properties(
    width = 600,
    # height = 600,
    title="Valley floor gradient"
).configure_legend(
    titleColor='black', 
    titleFontSize=12, 
    labelFontSize=16, 
    symbolStrokeWidth=4
)

# %% [markdown]
# Plot slope, rolling mean (small)

# %%
src = slope_df.reset_index(drop=True)[['elevation', 'time', 'path_distance_from_glacier', 'glacial', 'slope']].reset_index().dropna()
src['time'] = src['time'].apply(lambda x: x.strftime(strip_time_format))
alt.Chart(
    src
).transform_filter(
    (datum.glacial == False)
).transform_window(
    rolling_mean='mean(slope)',
    frame=[-5, 5],
    groupby=['time']
).mark_line().encode(
    alt.X("path_distance_from_glacier:Q", title="Distance downstream from observed glacial maximum"),
    alt.Y("rolling_mean:Q", scale=alt.Scale(zero=False), title='Valley floor slope (rolling mean, 10 meter window)', impute=alt.ImputeParams(value=None)),
    alt.Color("time:O", scale=alt.Scale(scheme='viridis')),
    alt.StrokeDash('glacial:N')
).properties(
    width = 600,
    # height = 600,
    title="Valley floor gradient"
).configure_legend(
    titleColor='black', 
    titleFontSize=12, 
    labelFontSize=16, 
    symbolStrokeWidth=4
)

# %%
src = slope_df.reset_index(drop=True)[['elevation', 'time', 'path_distance_from_glacier', 'glacial', 'slope']].reset_index().dropna()
src['time'] = src['time'].apply(lambda x: x.strftime(strip_time_format))
alt.Chart(
    src
).transform_filter(
    (datum.glacial == False)
).transform_window(
    rolling_mean='mean(slope)',
    frame=[-5, 5],
    groupby=['time']
).mark_line().encode(
    alt.X("path_distance_from_glacier:Q", title="Distance downstream from observed glacial maximum"),
    alt.Y("rolling_mean:Q", scale=alt.Scale(zero=False), title='Valley floor slope (rolling mean, 10 meter window)', impute=alt.ImputeParams(value=None)),
    alt.Color("time:O", scale=alt.Scale(scheme='viridis')),
    alt.StrokeDash('glacial:N')
).properties(
    width = 600,
    # height = 600,
    title="Valley floor gradient"
).configure_legend(
    titleColor='black', 
    titleFontSize=12, 
    labelFontSize=16, 
    symbolStrokeWidth=4
)

# %% [markdown]
# ## Group by kilometer upslope/downslope from glacier

# %%
grouped_km = slope_df[['elevation', 'time', 'path_distance_from_glacier', 'glacial', 'slope']].reset_index(drop=True).dropna()
grouped_km = grouped_km[~grouped_km.glacial]
grouped_km['Kilometer downstream from glacier'] = grouped_km['path_distance_from_glacier'].apply(lambda x: math.floor(x/1000))
groups = grouped_km.groupby(['time', 'Kilometer downstream from glacier'])
# remove data points if you weren't able to average slope over more than 500 meters
grouped_km = groups.filter(lambda df: 
    (df['path_distance_from_glacier'].max() - df['path_distance_from_glacier'].min()) > 1000*(2/3)
)
grouped_km = grouped_km.groupby(['time', 'Kilometer downstream from glacier']).mean().reset_index()

# %%
alt.Chart(grouped_km).mark_line(point=True).encode(
    alt.X('time:T', title=""),
    alt.Y('slope:Q', title="Valley floor slope"),
    alt.Facet('Kilometer downstream from glacier:O', title='Kilometer downstream from glacier')
).properties(width=200)

# %% [markdown]
# ## Group by provided distance upslope/downslope from glacier

# %%
grouped_halfkm = slope_df[['elevation', 'time', 'path_distance_from_glacier', 'glacial', 'slope']].reset_index(drop=True).dropna()
grouped_halfkm = grouped_halfkm[~grouped_halfkm.glacial]
grouped_halfkm['Half kilometer downstream from glacier'] = grouped_halfkm['path_distance_from_glacier'].apply(lambda x: math.floor(x/group_slope_meters))
groups = grouped_halfkm.groupby(['time', 'Half kilometer downstream from glacier'])
# remove data points if you weren't able to average slope over more than half of the averaging distance
grouped_halfkm = groups.filter(lambda df: 
    (df['path_distance_from_glacier'].max() - df['path_distance_from_glacier'].min()) > group_slope_meters*(2/3)
)
grouped_halfkm = grouped_halfkm.groupby(['time', 'Half kilometer downstream from glacier']).mean().reset_index()

# %%
alt.Chart(grouped_halfkm).mark_line(point=True).encode(
    alt.X('time:T', title=""),
    alt.Y('slope:Q', title="Valley floor slope"),
    alt.Facet('Half kilometer downstream from glacier:O', title='Half kilometer downstream from glacier')
).properties(width=200)

# %% [markdown]
# # Export low points
#

# %%
combined_gdf.geometry = combined_gdf['low_point_coords']

combined_gdf_noglacial = combined_gdf.query("not glacial")

combined_gdf_noglacial[
    ['geometry', 'path_distance_from_glacier', 'elevation', 'id', 'time']
].reset_index(drop=True).to_file(
    output_lowpoints_file,
    driver="GeoJSON"
)

# %% [markdown]
# # Create streamlines from low points

# %%
from shapely.geometry import Point, LineString
streamlines = combined_gdf_noglacial.reset_index(drop=True).groupby("time").apply(lambda df: LineString([point for point in df.geometry.tolist() if point]))

# %%
streamlines_gdf = gpd.GeoDataFrame(geometry=streamlines, crs=combined_gdf.crs)

# %%
streamlines_gdf.to_file(output_streamlines_file)


# %% [markdown]
# # Save dataframes

# %%
elevation_profiles = combined_gdf.reset_index(drop=True)[[ "time", "path_distance_from_glacier", "elevation", "glacial", "n_from_glacial_max"]].reset_index()
elevation_profiles['time'] = elevation_profiles['time'].apply(lambda x: x.strftime(strip_time_format))


dfs = [
    grouped_km,
    grouped_halfkm,
    elevation_profiles
]
names = [
    'slope_grouped_km',
    'slope_grouped_halfkm',
    'elevation_profiles'
]

for df,name in zip(dfs, names):
    df['valley'] = valley_name
    outdir = os.path.join("outputs", name)
    outfile = os.path.join(outdir, valley_name + ".pickle")
    os.makedirs(outdir, exist_ok=True)
    print(outfile)
    df.to_pickle(outfile)
