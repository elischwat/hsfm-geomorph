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
profiles_shapefile = "/data2/elilouis/hsfm-geomorph/data/profiles/profiles.shp"
files_data = """
area,            date,    filename
baker,     1970-09-29,    /data2/elilouis/baker_friedrich/1970-09-29_DEM.tif
baker,     1977-09-27,    /data2/elilouis/baker_friedrich/1977-09-27_DEM.tif
baker,     1979-10-06,    /data2/elilouis/baker_friedrich/1979-10-06_DEM.tif
baker,     1987-08-21,    /data2/elilouis/baker_friedrich/1987-08-21_DEM.tif
baker,     1990-09-05,    /data2/elilouis/baker_friedrich/1990-09-05_DEM.tif
baker,     1991-09-09,    /data2/elilouis/baker_friedrich/1991-09-09_DEM.tif
baker,           2015,    /data2/elilouis/hsfm-geomorph/data/reference_dem_highres/baker/baker_2015_utm_m.tif
rainier,   1970-08-29,    /data2/elilouis/rainier_friedrich/collection/70V1/70V1_mosaic.tif
rainier,   1973-09-24,    /data2/elilouis/rainier_friedrich/collection/73V3/73V3_mosaic.tif
rainier,   1979-10-05,    /data2/elilouis/rainier_friedrich/collection/79V5/79V5_mosaic.tif
rainier,   1987-08-21,    /data2/elilouis/rainier_friedrich/collection/87V1/87V1_mosaic.tif
rainier,   1988-08-21,    /data2/elilouis/rainier_friedrich/collection/88V1/88V1_mosaic.tif
rainier,   1980-10-09,    /data2/elilouis/rainier_friedrich/collection/80V1/80V1_mosaic.tif
rainier,   1990-09-13,    /data2/elilouis/rainier_friedrich/collection/90V3/90V3_mosaic.tif
rainier,   1991-09-09,    /data2/elilouis/rainier_friedrich/collection/91V3/91V3_mosaic.tif
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

gdf[gdf['area']=='rainier'].plot(), gdf[gdf['area']=='baker'].plot()

# ## Generate all possible profiles
# For each linestring, find all DEM files in that area and retrieve profile data.

profile_and_dem_df = gdf.merge(dem_files_df, on='area')

profile_and_dem_df.head()

profile_df_list = []
for i, row in profile_and_dem_df.iterrows():
    profile_df = get_raster_values_for_line(row.geometry, row.filename)
    profile_df['terminus_x'] = row['terminus_x']
    profile_df['terminus_y'] = row['terminus_y']
    profile_df['area'] = row['area']
    profile_df['name'] = row['name']
    profile_df['date'] = row['date']
    profile_df_list.append(profile_df)
profile_df = pd.concat(profile_df_list)

profile_df.head()

# ## Split into Rainier and Baker geodataframes

rainier_profiles_df = profile_df[profile_df['area'] == 'rainier']
baker_profiles_df = profile_df[profile_df['area'] == 'baker']

len(rainier_profiles_df), len(baker_profiles_df)

# # Plot data by mountain

rainier_along_path_plot = alt.Chart(rainier_profiles_df).mark_line().encode(
    x = alt.X('path_distance:Q', title='Pathwise Distance (m)'),
    y = alt.Y('raster_value:Q', title='Elevation (m)', scale=alt.Scale(zero=False)),
    color='date:O'
).properties(
    height=200
).facet(
    row = 'name:N'
).resolve_scale(
    x='independent',
    y='independent'
)
rainier_along_path_plot

baker_along_path_plot = alt.Chart(baker_profiles_df).mark_line().encode(
    x = alt.X('path_distance:Q', title='Pathwise Distance (m)'),
    y = alt.Y('raster_value:Q', title='Elevation (m)', scale=alt.Scale(zero=False)),
    color='date:O'
).properties(
    height=200
).facet(
    row = 'name:N'
).resolve_scale(
    x='independent',
    y='independent'
)
baker_along_path_plot

# # Altair example of adding a vertical/horizontal line

long_rule = alt.Chart(terminus_df).mark_rule().encode(
    x=alt.X('lon', scale=alt.Scale(zero=False)),
#     color = 'year:O'
)
long_rule + long_plot
