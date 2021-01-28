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
import math
import numpy as np
import pandas as pd
import geopandas as gpd

import matplotlib.pyplot as plt

from itertools import chain

from shapely.geometry import shape
from shapely import geometry

from shapely.ops import unary_union

import rasterio
from rasterio import features
from rasterio.plot import show as rioshow

import contextily as ctx
import os
# -

# # Inputs Required:

# Rainier Inputs
glacier_polygon_file = "/data2/elilouis/hsfm-geomorph/data/02_rgi60_WesternCanadaUS/02_rgi60_WesternCanadaUS.shp"
mosaics_path_directory = "/data2/elilouis/rainier_friedrich/collection/"
mosaic_tif_suffix = "*mosaic_30m.tif"
year_from_filename = lambda s: f"19{s[:2]}"

# Baker Inputs
glacier_polygon_file = "/data2/elilouis/hsfm-geomorph/data/02_rgi60_WesternCanadaUS/02_rgi60_WesternCanadaUS.shp"
mosaics_path_directory = "/data2/elilouis/baker_friedrich/"
mosaic_tif_suffix = "*DEM_30m.tif"
year_from_filename = lambda s: s[:4]

mosaic_file_names = !find {mosaics_path_directory} -type f -name {mosaic_tif_suffix}
print(f"Found DEM files:")
_ = [print('\t' + fn) for fn in mosaic_file_names]


# ## Plot Annual DEM Coverage Outline Polygons

def outline_from_raster(raster_dataset, simplify_tolerance = 0.0001, buffer = 0.001):
    """
    raster_dataset (rasterio dataset)
    simplify_tolerance (float): in units of the CRS of raster_dataset.
    buffer (float): in units of the CRS of raster_dataset.
    """
    # Read the dataset's valid data mask as a ndarray.
    mask = raster_dataset.dataset_mask()
    geoms = []
    vals = []
    # Extract feature shapes and values from the array.
    for geom, val in rasterio.features.shapes(
            mask, transform=raster_dataset.transform):

        # Transform shapes from the dataset's own coordinate
        # reference system to CRS84 (EPSG:4326).
        geom = rasterio.warp.transform_geom(
            raster_dataset.crs, 'EPSG:4326', geom, precision=6)

        # Print GeoJSON shapes to stdout.
        geoms.append(geom)
        vals.append(val)
    filtered_geoms = [geom for geom, val in zip(geoms,vals) if val == 255]
    gdf = gpd.GeoDataFrame(geometry = [shape(g) for g in filtered_geoms])
    gdf.geometry = gdf.geometry.apply(lambda g: g.simplify(simplify_tolerance))
    gdf.geometry = gdf.geometry.apply(lambda g: g.buffer(buffer))
    union = unary_union(gdf.geometry)
    return gpd.GeoDataFrame(geometry=[union])


# %%time
gdf_list = []
for fn in mosaic_file_names:
    dataset = rasterio.open(fn)
    gdf = outline_from_raster(dataset)
    gdf['year'] = fn.split('/')[-1].split(".")[0]
    gdf_list.append(gdf)
gdf_all_years = pd.concat(gdf_list)

gdf_all_years = gdf_all_years.set_crs(epsg=4326)
gdf_all_years = gdf_all_years.to_crs(epsg=3857)

# Plot just the bounds first and we will use the automatically determined bounds to filter the RGI glacier polygons and plot those after

ax = gpd.GeoDataFrame(gdf_all_years).plot(
    column='year',
    facecolor="none",
    edgecolor='k',
    linewidth=2.5,
    legend=True,
    figsize=(10,10)
)

xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
bounds = geometry.Polygon([[xmin, ymax], [xmax, ymax], [xmax, ymin], [xmin, ymin]])

glacier_gdf = gpd.read_file(glacier_polygon_file)
glacier_gdf = glacier_gdf.to_crs(epsg=3857)
rainier_glacier_gdf = glacier_gdf[glacier_gdf.geometry.within(bounds)]

gdf_all_years['year'] = gdf_all_years['year'].apply(year_from_filename)

profiles_shapefile = "/data2/elilouis/hsfm-geomorph/data/profiles/profiles.shp"
profiles_gdf = gpd.read_file(profiles_shapefile)
profiles_gdf = profiles_gdf[profiles_gdf['area']=='rainier']
profiles_gdf = profiles_gdf.to_crs(epsg=3857)

ax = rainier_glacier_gdf.plot(figsize=(10,10), alpha=0.35, label='name')
gpd.GeoDataFrame(gdf_all_years).plot(
    column='year',
    facecolor="none",
    edgecolor='k',
    linewidth=2.5,
    legend=True,
    ax=ax
)
profiles_gdf.plot(ax=ax, color='red', linewidth=3)
ctx.add_basemap(ax, source=ctx.providers.Stamen.Terrain, alpha=0.6)
plt.title("NAGAP Historical DEM Coverage", fontsize=16)
plt.show()

output_shp_fn = os.path.join(mosaics_path_directory, "dem_coverage_polygons.geojson")
print(f"Saving DEM coverage polygons to {output_shp_fn}")
gdf_all_years.to_file(output_shp_fn, driver="GeoJSON")
