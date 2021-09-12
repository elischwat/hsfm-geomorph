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

# !find /home/elilouis/hsfm-geomorph/ -name "*mosaic*"

import geopandas as gpd
import altair as alt
import io   
from profiling_tools import *
pd.options.display.max_rows = 100
pd.options.display.max_columns = 100
alt.data_transformers.disable_max_rows()

# # Input Requirements
# A csv of DEM filenames, a date for each DEM filename, and an area/region/generic label.

n_points = 1000
rasters_crs = 'EPSG:32610'
profiles_shapefile = "/data2/elilouis/hsfm-geomorph/data/profiles/paradise_road_xsection.shp"
files_data = """
area,            date,    filename
rainier,   1973-09-24,    /data2/elilouis/rainier_friedrich/collection/73V3/73V3_mosaic.tif
rainier,   1979-10-05,    /data2/elilouis/rainier_friedrich/collection/79V5/79V5_mosaic.tif
rainier,   1990-09-13,    /data2/elilouis/rainier_friedrich/collection/90V3/90V3_mosaic.tif
rainier,   1992-07-28,    /data2/elilouis/rainier_friedrich/collection/92V3/92V3_mosaic.tif
rainier,   1992-10-06,    /data2/elilouis/rainier_friedrich/collection/92V5/92V5_mosaic.tif
rainier,      2007/08,    /data2/elilouis/hsfm-geomorph/data/reference_dem_highres/rainier_lidar_dsm-adj.tif
"""

# # Prep data

# ## Create dataframe of DEM file names

dem_files_df = pd.read_csv(io.StringIO(files_data), skipinitialspace=True)

dem_files_df.head(3)

# ## Create geodataframe of profile lines
# Convert to CRS of the rasters

gdf = gpd.read_file(profiles_shapefile)
gdf = gdf.to_crs(rasters_crs)

gdf.head(3)

# ### Increase complexity of paths to sample more DEM pixels. 
# Probably won't need to do this after properly creating profile lines (from valley cross section minimums)

gdf.geometry = gdf.geometry.apply(
    lambda line: 
    increase_line_complexity(line, n_points = n_points)
)

gdf.plot()

# ## Generate all possible profiles
# For each linestring, find all DEM files in that area and retrieve profile data.

dem_files_df['geometry'] = gdf.geometry.iloc[0]

profile_and_dem_df = gpd.GeoDataFrame(dem_files_df)

profile_df_list = []
for i, row in profile_and_dem_df.iterrows():
    profile_df = get_raster_values_for_line(row.geometry, row.filename)
    profile_df['area'] = row['area']
    profile_df['date'] = row['date']
    profile_df_list.append(profile_df)
profile_df = pd.concat(profile_df_list)

alt.Chart(profile_df).mark_line().encode(
    x = alt.X('path_distance:Q', title='Pathwise Distance (m)'),
    y = alt.Y('raster_value:Q', title='Elevation (m)', scale=alt.Scale(zero=False)),
    color='date:N'
).properties(
    height=300, width=500
).resolve_scale(
    x='independent',
    y='independent'
)

profile_df.date.unique()

profile_df.head()

profile_df.date.unique()

# +
src = profile_df[profile_df['date'] == '2007/08']
df_1973 = pd.DataFrame({
    'X': src['X'],
    'Y': src['Y'],
    'path_distance': src['path_distance'],
    'difference': profile_df[profile_df['date'] == '2007/08']['raster_value'] - profile_df[profile_df['date'] == '1973-09-24']['raster_value'],
})
df_1973['name'] = '2007/08 - 1973'

df_1979 = pd.DataFrame({
    'X': src['X'],
    'Y': src['Y'],
    'path_distance': src['path_distance'],
    'difference': profile_df[profile_df['date'] == '2007/08']['raster_value'] - profile_df[profile_df['date'] == '1979-10-05']['raster_value']    
})
df_1979['name'] = '2007/08 - 1979'

# df_1990 = pd.DataFrame({
#     'X': src['X'],
#     'Y': src['Y'],
#     'path_distance': src['path_distance'],
#     'difference': profile_df[profile_df['date'] == '2007/08']['raster_value'] - profile_df[profile_df['date'] == '1990-09-13']['raster_value']    
# })
# df_1990['name'] = '2007/08 - 1990'

# df_1992a = pd.DataFrame({
#     'X': src['X'],
#     'Y': src['Y'],
#     'path_distance': src['path_distance'],
#     'difference': profile_df[profile_df['date'] == '2007/08']['raster_value'] - profile_df[profile_df['date'] == '1992-07-28']['raster_value']    
# })
# df_1992a['name'] = '2007/08 - 1992-07'

# df_1992b = pd.DataFrame({
#     'X': src['X'],
#     'Y': src['Y'],
#     'path_distance': src['path_distance'],
#     'difference': profile_df[profile_df['date'] == '2007/08']['raster_value'] - profile_df[profile_df['date'] == '1992-10-06']['raster_value']    
# })
# df_1992b['name'] = '2007/08 - 1992-10'

# df = pd.concat([df_1973, df_1979, df_1990, df_1992a, df_1992b])
df = pd.concat([df_1973, df_1979])
# -

alt.Chart(df).mark_line().encode(
    x = alt.X('path_distance:Q', title='Pathwise Distance (m)'),
    y = alt.Y('difference:Q', title='Elevation Difference (m)', scale=alt.Scale(zero=False)),
    color=alt.Color('name:N', title='DEM of Difference')
).properties(
    height=300, width=500
).resolve_scale(
    x='independent',
    y='independent'
)

import contextily as ctx
gdf_transformed = gdf.to_crs(epsg=3857)
ax = gdf_transformed.plot()
ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)


