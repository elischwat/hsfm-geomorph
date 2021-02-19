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
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import OrderedDict

import hsfm
import hipp

import contextily as ctx
from contextily.tile import bounds2raster
import geopandas as gpd
import rasterio as rio
import rioxarray as riox
from shapely.geometry import box
import rasterio.mask
import rasterio.plot
from rasterio.warp import array_bounds, reproject, Resampling, calculate_default_transform

from pyproj import Proj, Transformer
# -

# Notes:
# * Currently this notebook expects 3 input DEMs and will only plot two difference datasets
# * Contextily tile downloads can be finnicky, especially if the polygon for masking is very small

# # Input Files

# +
output_path = "/Volumes/MyDrive/hsfm-geomorph/data/gcas/"

input_dems = {
    1979: "/Volumes/GoogleDrive/.shortcut-targets-by-id/1qrTqp2neZpyQkDkX88TC0wzZ3Fu2jXG_/nagap_testing/baker/input_data/79V6/10/06/sfm/cluster_001/metashape1/baker_cluster_001-DEM_dem_align/baker_cluster_001-DEM_baker_2015_utm_m_nuth_x+0.18_y+0.11_z+1.82_align.tif",
    1970: "/Volumes/GoogleDrive/.shortcut-targets-by-id/1qrTqp2neZpyQkDkX88TC0wzZ3Fu2jXG_/nagap_testing/baker/input_data/70V2/09/29/sfm/cluster_000/metashape1/baker_cluster_000-DEM_dem_align/baker_cluster_000-DEM_baker_2015_utm_m_nuth_x+0.78_y+1.06_z+1.06_align.tif",
    1977: "/Volumes/GoogleDrive/.shortcut-targets-by-id/1qrTqp2neZpyQkDkX88TC0wzZ3Fu2jXG_/nagap_testing/baker/input_data/77V6/09/27/sfm/cluster_003/metashape1/baker_cluster_003-DEM_dem_align/baker_cluster_003-DEM_baker_2015_utm_m_nuth_x+2.43_y+1.77_z+1.60_align.tif",
}

mosaic_file_1970 = "/Volumes/GoogleDrive/.shortcut-targets-by-id/1qrTqp2neZpyQkDkX88TC0wzZ3Fu2jXG_/nagap_testing/baker/input_data/70V2/09/29/sfm/cluster_000/metashape1/baker_cluster_000_orthomosaic.tif"
mosaic_file_1977 = "/Volumes/GoogleDrive/.shortcut-targets-by-id/1qrTqp2neZpyQkDkX88TC0wzZ3Fu2jXG_/nagap_testing/baker/input_data/77V6/09/27/sfm/cluster_003/metashape1/baker_cluster_003_orthomosaic.tif"
mosaic_file_1979 = "/Volumes/GoogleDrive/.shortcut-targets-by-id/1qrTqp2neZpyQkDkX88TC0wzZ3Fu2jXG_/nagap_testing/baker/input_data/79V6/10/06/sfm/cluster_001/metashape1/baker_cluster_001_orthomosaic.tif"

area_polygon = gpd.read_file("/Volumes/MyDrive/hsfm-geomorph/data/gcas/mazama_gca.geojson").iloc[0:1]


# -

# ## Create DoDs with DEMs

# +
def difference_dems(older_dem_path, newer_dem_path, dest_path):
    """ 
    Difference calculated
    newer_dem_path - older_dem_path
    """
    
    src = hsfm.utils.difference_dems(older_dem_path, newer_dem_path)
    os.replace(src, dest_path)
    return dest_path

def dem_difference_stack(input_dem_dict, output_path):
    """
    Generate a time series of DEMs of Difference. Create a difference with every
    chronological pair of DEMs
    inputs_dems (dict[int, str]): identifying integer (like the year) and file path
    """
    input_dems = list(OrderedDict(sorted(input_dem_dict.items())).items())
    
    diff_dem_paths = []
    
    first_year, first_dem_path = input_dems.pop(0)
    while len(input_dems) > 0:
        second_year, second_dem_path = input_dems.pop(0)
        dest = difference_dems(first_dem_path, second_dem_path, os.path.join(output_path, f"{second_year}_{first_year}_difference.tif"))
        diff_dem_paths.append(dest)
        first_year, first_dem_path = (second_year, second_dem_path)
    return diff_dem_paths


# -

diff_dem_paths = dem_difference_stack(input_dems, output_path)
diff_dem_paths

# ## Mask DoDs with Polygons
# #### IE Create DoD dataset for uncertainty analysis

shapes = [area_polygon.geometry.iloc[0]]

# #### Process 2 DoDs

len(diff_dem_paths)

dod_1 = riox.open_rasterio(diff_dem_paths[0], masked=True)
dod_clipped_1 = dod_1.rio.clip(shapes).squeeze()

sns.distplot(dod_clipped_1.values, kde=False)
sns.despine(offset=5, trim=True);
plt.title("1970-1977")
plt.show()

dod_2 = riox.open_rasterio(diff_dem_paths[1], masked=True)
dod_clipped_2 = dod_2.rio.clip(shapes).squeeze()

sns.distplot(dod_clipped_2.values, kde=False)
sns.despine(offset=5, trim=True);
plt.title("1977-1979")
plt.show()


# #### Plot map of differences

def plot_raster(data, transform, cmap, vmin, vmax, title, ax):
    """
    You need to plot these two separate times to get a nice colorbar 
    and nice axes in the correct CRS
    """
    image_hidden = ax.imshow(
        data,                      
        cmap=cmap,                      
        vmin=vmin,                      
        vmax=vmax
    )
    # plot on the same axis with rio.plot.show
    image = rio.plot.show(
        data, 
        transform=transform, 
        ax=ax, 
        cmap=cmap,
        vmin=vmin, 
        vmax=vmax,
        title=title
    )
    return image_hidden


# +
fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 5))
img = plot_raster(dod_clipped_1.values, dod_clipped_1.rio.transform(), 'PuOr', -15, 15, '1970-1977', axes[0])
img = plot_raster(dod_clipped_2.values, dod_clipped_2.rio.transform(), 'PuOr', -15, 15, '1977-1979', axes[1])

for ax in axes:
    ax.ticklabel_format(useOffset=False, style='plain')
fig.colorbar(img, ax=axes)
plt.show()
# -

# ## Retrieve Satellite Imagery Tile

# #### Get bounds of masked dataset, in Web Mercator because the bounds2raster method requires it

west, south, east, north = dod_clipped_1.rio.reproject("EPSG:3857").rio.bounds()

_ = bounds2raster(
    west - 0.00001, 
    south - 0.00001,
    east + 0.00001, 
    north + 0.00001, 
    "sample_tile.tif", 
    zoom = "auto", 
#     source=ctx.providers.Esri.WorldImagery
    source = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
)

# #### Reproject the satellite tile to our preferred EPSG:32610

sat_tile = riox.open_rasterio("sample_tile.tif").rio.reproject("EPSG:32610")

# #### Plot Satellite Imagery Tile

fig, ax = plt.subplots(figsize=(5, 5))
image = rio.plot.show(
    sat_tile.values,
    transform =sat_tile.rio.transform(), 
    ax=ax
)
ax.ticklabel_format(useOffset=False, style='plain')

# ## Plot DoD and Satellite Imagery

# +
fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15, 5))

image = rio.plot.show(
    sat_tile.values,
    transform=sat_tile.rio.transform(), 
    ax=axes[0],
    title="Modern Imagery"
)

img = plot_raster(dod_clipped_1.values, dod_clipped_1.rio.transform(), 'PuOr', -15, 15, '1970-1977', axes[1])
img = plot_raster(dod_clipped_2.values, dod_clipped_2.rio.transform(), 'PuOr', -15, 15, '1977-1979', axes[2])

for ax in axes:
    ax.ticklabel_format(useOffset=False, style='plain')
fig.colorbar(img, ax=axes)
plt.show()


# -

# ## Incorporate Mosaics

# Read in file, mask with geometry, convert from 4326 to 32610, plot it like the tile above (one step)

def get_mosaic_chunk(orthomosaic_file, bounds, dst_crs):
    """really a clip and reproject method"""
    return riox.open_rasterio(
        orthomosaic_file
    ).rio.clip_box(
        *bounds
    ).rio.reproject(
        dst_crs
    )


# Create shape for clipping (bounds of the downloaded satellite tile)

geom = gpd.GeoDataFrame(
    geometry=[box(*(west, south, east, north))], 
    crs='epsg:3857'
).to_crs('epsg:4326').geometry.iloc[0]

# #### Plot Mosaics

fig, axes = plt.subplots(1,3,sharex=True, sharey=True, figsize=(20,8))
for ax, file, year in zip(
    axes, 
    [mosaic_file_1970, mosaic_file_1977, mosaic_file_1979],
    [1970, 1977, 1979]
):
    dataset = get_mosaic_chunk(file, geom.bounds, rasterio.crs.CRS.from_epsg(32610))
    array, transform = (dataset[0].values, dataset.rio.transform())
    array = hipp.image.clahe_equalize_image(array)
    rio.plot.show(array,transform=transform, ax=ax, cmap="Greys", title=year, clim=(0,255))
    ax.ticklabel_format(useOffset=False, style='plain')
plt.suptitle("NAGAP Mosaics", fontsize=14)

# ## Plot Mosaics, Satellite Imagery, and DoD together

# +
fig, axes = plt.subplots(1, 6, sharex=True, sharey=True, figsize=(24,5))

# PLOT THE SAT IMAGERY
image = rio.plot.show(
    sat_tile.values,
    transform=sat_tile.rio.transform(), 
    ax=axes[0],
    title='Modern Imagery'
)

# PLOT MOSAICS (mosaic_file_1970, mosaic_file_1977, mosaic_file_1979)
dataset = get_mosaic_chunk(mosaic_file_1970, geom.bounds, rasterio.crs.CRS.from_epsg(32610))
rio.plot.show(
    hipp.image.clahe_equalize_image(dataset[0].values),
    transform=transform, 
    ax=axes[1], 
    cmap="Greys", 
    title='1970', 
    clim=(0,255)
)

dataset = get_mosaic_chunk(mosaic_file_1977, geom.bounds, rasterio.crs.CRS.from_epsg(32610))
rio.plot.show(
    hipp.image.clahe_equalize_image(dataset[0].values),
    transform=transform, 
    ax=axes[2], 
    cmap="Greys", 
    title='1977', 
    clim=(0,255)
)

dataset = get_mosaic_chunk(mosaic_file_1979, geom.bounds, rasterio.crs.CRS.from_epsg(32610))
rio.plot.show(
    hipp.image.clahe_equalize_image(dataset[0].values),
    transform=transform, 
    ax=axes[3], 
    cmap="Greys", 
    title='1979', 
    clim=(0,255)
)

img = plot_raster(dod_clipped_1.values, dod_clipped_1.rio.transform(), 'PuOr', -15, 15, '1970-1977', axes[4])
img = plot_raster(dod_clipped_2.values, dod_clipped_2.rio.transform(), 'PuOr', -15, 15, '1977-1979', axes[5])

# add colorbar using the now hidden image
# pass list of axes so space is "stolen" equally from all subplots
fig.colorbar(img, ax=axes, pad = 0.01)
for ax in axes:
    ax.ticklabel_format(useOffset=False, style='plain')
plt.show()
