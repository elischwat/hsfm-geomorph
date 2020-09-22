# -*- coding: utf-8 -*-
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

import pandas as pd
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt

# # Find 60s imagery (no KML file to use)

# Read in compiled list of images

pids_df = pd.read_csv('glacier_names_pids.csv')

# Read in AOI geometries

aoi_gdf = gpd.read_file('aois.geojson')
aoi_gdf = aoi_gdf.to_crs(epsg=3857)
baker_polygon = aoi_gdf[aoi_gdf.name == 'Mt. Baker'].geometry.iloc[0]
glacier_polygon = aoi_gdf[aoi_gdf.name == 'Glacier Peak'].geometry.iloc[0]
rainier_polygon = aoi_gdf[aoi_gdf.name == 'Mt. Rainier'].geometry.iloc[0]

# ## Find 60s images listed in the `glacier_names_pids.csv` dataset

images_60s = pids_df[pids_df.Year < 1970]
images_60s.head()

len(images_60s), len(images_60s[~images_60s.Location.isna()]), len(images_60s[~images_60s.Longitude.isna()]), len(images_60s[~images_60s.Latitude.isna()])

# ## Find 60s images on Mt. Rainier

# ### Open the glacier polygon dataset, which will help us identify images lacking lat/long info

rgi_gdf = gpd.read_file('/home/elilouis/02_rgi60_WesternCanadaUS.shp')
rgi_gdf = rgi_gdf.to_crs(epsg=3857)

rainier_glaciers_gdf = rgi_gdf[rgi_gdf.geometry.within(rainier_polygon)]

ax = rainier_glaciers_gdf.plot(column='Name')
ctx.add_basemap(ax, url=ctx.providers.Esri.WorldImagery)
plt.gcf().set_size_inches(8,8)

rainier_glaciers_gdf.Name.unique()

rainier_glaciers_gdf.Name = rainier_glaciers_gdf.Name.apply(lambda x: x.replace('WA','').strip())
rainier_glacier_names = rainier_glaciers_gdf.Name.unique()
rainier_glacier_names

rainier_images_60s = images_60s[images_60s.Location.isin(rainier_glacier_names)]
rainier_images_60s

# ### Plot glaciers we have images for

rainier_glaciers_with_images_gdf = rainier_glaciers_gdf[rainier_glaciers_gdf.Name.isin(rainier_images_60s.Location)]

ax = rainier_glaciers_with_images_gdf.plot(column='Name', legend=True)
ctx.add_basemap(ax, url=ctx.providers.Esri.WorldImagery)
plt.gcf().set_size_inches(5,5)

# And how many images do i have per glacier from the 60s? Not many

rainier_images_60s.Location.value_counts()

# So if we want 60s data, maybe the Carbon and Frying Pan Watersheds are preferable over Kautz.

rainier_images_60s

rainier_images_60s_gdf = gpd.GeoDataFrame(
    rainier_images_60s, geometry=gpd.points_from_xy(rainier_images_60s.Longitude, rainier_images_60s.Latitude))
rainier_images_60s_gdf.crs = {'init' :'epsg:4326'}
rainier_images_60s_gdf = rainier_images_60s_gdf.to_crs(epsg=3857)

ax = rainier_glaciers_with_images_gdf.plot(column='Name', legend=True)
ctx.add_basemap(ax, url=ctx.providers.Esri.WorldImagery)
rainier_images_60s_gdf.plot(ax=ax, color='red')

# It seems that these images were all from two points, so maybe we don't have many 1960s images at all for Mt Rainier. We should examine the thumbnails to be sure.
