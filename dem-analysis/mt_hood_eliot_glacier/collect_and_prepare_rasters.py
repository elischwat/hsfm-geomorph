# %%
import os
import io
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path

import altair as alt
import profiling_tools
import rioxarray as rio
from pyproj import Proj, transform
import contextily as ctx
import rioxarray as rix

import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = (10.0, 10.0)

# %% [markdown]
# # Gather data from disparate sources into one project directory

# %% [markdown]
# Original DEM files, from HSFM output:
#
# >
#     area,     date,       filename
#     hood,     1975-09,    /data2/elilouis/mt_hood_timesift/individual_clouds/75_09/cluster1/1/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/spoint2point_bareground-trans_source-DEM_reference_dem_clipped_nuth_x-3.82_y-3.01_z+1.66_align.tif
#     hood,     1967-09,    /data2/elilouis/mt_hood_timesift/individual_clouds/67_9/cluster1/1/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/spoint2point_bareground-trans_source-DEM_reference_dem_clipped_nuth_x-4.90_y-5.23_z+4.21_align.tif
#     hood,     1977-10,    /data2/elilouis/mt_hood_timesift/individual_clouds/77_10/cluster0/1/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/spoint2point_bareground-trans_source-DEM_reference_dem_clipped_nuth_x-3.13_y-3.96_z+2.31_align.tif
#     hood,     1980-10,    /data2/elilouis/mt_hood_timesift/individual_clouds/80_10/cluster0/1/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/spoint2point_bareground-trans_source-DEM_reference_dem_clipped_nuth_x+0.38_y-2.59_z+1.67_align.tif
#     hood,     1990-09,    /data2/elilouis/mt_hood_timesift/individual_clouds/90_09/cluster0/1/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/spoint2point_bareground-trans_source-DEM_reference_dem_clipped_nuth_x-2.79_y-2.00_z+1.61_align.tif
#     hood,     2015,       /data2/elilouis/hsfm-geomorph/data/reference_dem_highres/hood/2009.tif
#     """

# %%
# mkdir /data2/elilouis/hsfm-geomorph/data/mt_hood_eliot_glacier/input_dems/orthos

# %%
# !cp /data2/elilouis/mt_hood_timesift/individual_clouds/75_09/cluster1/1/project_orthomosaic.tif \
#                             /data2/elilouis/hsfm-geomorph/data/mt_hood_eliot_glacier/orthos/75_09_ortho.tif
# !cp /data2/elilouis/mt_hood_timesift/individual_clouds/67_9/cluster1/1/project_orthomosaic.tif  \
#                             /data2/elilouis/hsfm-geomorph/data/mt_hood_eliot_glacier/orthos/67_9_ortho.tif
# !cp /data2/elilouis/mt_hood_timesift/individual_clouds/77_10/cluster0/1/project_orthomosaic.tif \
#                             /data2/elilouis/hsfm-geomorph/data/mt_hood_eliot_glacier/orthos/77_10_ortho.tif
# !cp /data2/elilouis/mt_hood_timesift/individual_clouds/80_10/cluster0/1/project_orthomosaic.tif \
#                             /data2/elilouis/hsfm-geomorph/data/mt_hood_eliot_glacier/orthos/80_10_ortho.tif
# !cp /data2/elilouis/mt_hood_timesift/individual_clouds/90_09/cluster0/1/project_orthomosaic.tif \
#                             /data2/elilouis/hsfm-geomorph/data/mt_hood_eliot_glacier/orthos/90_09_ortho.tif

# %%
# ls /data2/elilouis/mt_hood_timesift/individual_clouds/75_09/cluster1/1/project_orthomosaic.tif

# %% [markdown]
# Make copies of files for this project, put in our directory for this project
#
# `/data2/elilouis/hsfm-geomorph/data/mt_hood_eliot_glacier`

# %%
# !cp \
#     /data2/elilouis/mt_hood_timesift/individual_clouds/75_09/cluster1/1/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/spoint2point_bareground-trans_source-DEM_reference_dem_clipped_nuth_x-3.82_y-3.01_z+1.66_align.tif  \
#     /data2/elilouis/hsfm-geomorph/data/mt_hood_eliot_glacier/input_dems/1975-09.tif 
# !cp \
#     /data2/elilouis/mt_hood_timesift/individual_clouds/67_9/cluster1/1/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/spoint2point_bareground-trans_source-DEM_reference_dem_clipped_nuth_x-4.90_y-5.23_z+4.21_align.tif \
#     /data2/elilouis/hsfm-geomorph/data/mt_hood_eliot_glacier/input_dems/1967-09.tif 
# !cp \
#     /data2/elilouis/mt_hood_timesift/individual_clouds/77_10/cluster0/1/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/spoint2point_bareground-trans_source-DEM_reference_dem_clipped_nuth_x-3.13_y-3.96_z+2.31_align.tif \
#     /data2/elilouis/hsfm-geomorph/data/mt_hood_eliot_glacier/input_dems/1977-10.tif 
# !cp \
#     /data2/elilouis/mt_hood_timesift/individual_clouds/80_10/cluster0/1/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/spoint2point_bareground-trans_source-DEM_reference_dem_clipped_nuth_x+0.38_y-2.59_z+1.67_align.tif \
#     /data2/elilouis/hsfm-geomorph/data/mt_hood_eliot_glacier/input_dems/1980-10.tif 
# !cp \
#     /data2/elilouis/mt_hood_timesift/individual_clouds/90_09/cluster0/1/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/spoint2point_bareground-trans_source-DEM_reference_dem_clipped_nuth_x-2.79_y-2.00_z+1.61_align.tif \
#     /data2/elilouis/hsfm-geomorph/data/mt_hood_eliot_glacier/input_dems/1990-09.tif 
# !cp \
#     /data2/elilouis/hsfm-geomorph/data/reference_dem_highres/hood/2009.tif \
#     /data2/elilouis/hsfm-geomorph/data/mt_hood_eliot_glacier/input_dems/2009.tif    


# %%
# ls /data2/elilouis/hsfm-geomorph/data/mt_hood_eliot_glacier/input_dems/

# %% [markdown]
# ## Create dataframe of dem files for later use

# %%
files_df = pd.read_csv(io.StringIO(
    """
    date,       filename
    1975-09,    /data2/elilouis/hsfm-geomorph/data/mt_hood_eliot_glacier/input_dems/1975-09.tif
    1967-09,    /data2/elilouis/hsfm-geomorph/data/mt_hood_eliot_glacier/input_dems/1967-09.tif
    1977-10,    /data2/elilouis/hsfm-geomorph/data/mt_hood_eliot_glacier/input_dems/1977-10.tif
    1980-10,    /data2/elilouis/hsfm-geomorph/data/mt_hood_eliot_glacier/input_dems/1980-10.tif
    1990-09,    /data2/elilouis/hsfm-geomorph/data/mt_hood_eliot_glacier/input_dems/1990-09.tif
    2009,       /data2/elilouis/hsfm-geomorph/data/mt_hood_eliot_glacier/input_dems/2009.tif
    """
    ), 
    sep="\s*[,]\s*",
    engine='python'
) 

# %% [markdown]
# # Create Slope and Area Dataset from Mt Hood Dem

# %% [markdown]
# ## Extract watershed boundary from the NHD datasets

# %%
gdf = gpd.read_file(
	"/data2/elilouis/hsfm-geomorph/data/NHDPLUS_H_1707_HU4_GDB/NHDPLUS_H_1707_HU4_GDB.gdb", 
	layer='WBDHU12'
)

# %% [markdown]
# Can filter for relevant watersheds: WBDHU12 175 &468
# Or determine those programatically below

# %% [markdown]
# ## Get the center of one DEM to identify the watershed geometrically

# %%
example_raster = rio.open_rasterio(files_df['filename'].iloc[0])


# %%
raster_diff= example_raster.rio.reproject(gdf.crs)
coords = raster_diff.rio.bounds()
centerx,centery = (np.average(coords[::2]), np.average(coords[1::2]))
centerx,centery

# %%
from shapely.geometry import Point

# %%
watershed_boundaries = gdf[gdf.geometry.intersects(Point(centerx,centery).buffer(0.02))]
watershed_boundaries.Name

# %%
ax = watershed_boundaries.to_crs(epsg=3785).plot()
ctx.add_basemap(ax)

# %% [markdown]
# ## Crop reference DEM to the relevant watersheds

# %%
#identify the LIDAR reference DEM file path
reference_dem_fn = files_df.filename.iloc[-1]
cropped_fn = reference_dem_fn.replace(
    '.tif', '_cropped_to_watersheds.tif'
)
reference_dem_fn, cropped_fn

# %%
from shapely.ops import cascaded_union
# open the DEM
ref_dem = rix.open_rasterio(
    reference_dem_fn
)
# convert watershed vector boundaries to the CRS of the raster and merge into a single geometry
joined_watershed_boundaries = cascaded_union(
    watershed_boundaries.to_crs(ref_dem.rio.crs).geometry
)
# clip raster to merged bounds
ref_dem.rio.clip(
    [joined_watershed_boundaries]
).rio.to_raster(
    cropped_fn
)

# %% [markdown]
# ## Generate Slope and Area datasets from a downsampled reference LIDAR DEM

# %% [markdown]
# Create a 5 meter version of the reference DEM for drainage area and slope calculations.

# %%
reference_dem_fn

# %%
downsampled_reference_dem = (
    "/data2/elilouis/hsfm-geomorph/data/mt_hood_eliot_glacier/input_dems/2009_5m.tif"
)
downsampled_reference_slope = (
    "/data2/elilouis/hsfm-geomorph/data/mt_hood_eliot_glacier/rasters/2009_5m_slope.tif"
)
downsampled_reference_flowdir = (
    "/data2/elilouis/hsfm-geomorph/data/mt_hood_eliot_glacier/rasters/2009_5m_flowdir.tif"
)

# %%
ref_dem_raster = rix.open_rasterio(reference_dem_fn)
ref_dem_raster_downsampled = ref_dem_raster.rio.reproject(ref_dem_raster.rio.crs, resolution=5)
ref_dem_raster_downsampled.rio.to_raster(
    downsampled_reference_dem
)

# %% [markdown]
# Create drainage raster with downsampled reference_dem

# %%
from pysheds.grid import Grid

grid = Grid.from_raster(downsampled_reference_dem, data_name='dem')
grid.fill_depressions(data='dem', out_name='filled')
grid.resolve_flats(data='filled', out_name='dem_inflated')
grid.flowdir('dem_inflated', out_name='dir', routing='d8')
grid.accumulation(data='dir', out_name='acc')

# %%
from matplotlib import colors, cm, pyplot as plt
# norm = colors.LogNorm(acc.min(), acc.max())
plt.imshow(grid.acc, cmap=cm.gray, norm=colors.LogNorm())
plt.colorbar()

# %%
grid.to_raster(
    'acc', 
    downsampled_reference_flowdir, 
    dtype='float32'
)

# %% [markdown]
# Create slope raster with downsampled reference DEM

# %%
# !gdaldem slope \
#     $downsampled_reference_dem \
#     $downsampled_reference_slope

# %%
slope= rix.open_rasterio(downsampled_reference_slope)
slope_masked = slope.values
slope_masked[slope_masked==-9999] = np.nan

from matplotlib import colors, cm, pyplot as plt
# norm = colors.LogNorm(acc.min(), acc.max())
plt.imshow(slope.values[0], cmap=cm.gray)
plt.colorbar()

# %% [markdown]
# # Regrid all datasets to one grid and crop to study area

# %% [markdown]
# ## Load study area polygon

# %%
study_area_geometry_fn = "/data2/elilouis/hsfm-geomorph/data/mt_hood_eliot_glacier/eliot_glacier_study_area.geojson"

# %%
study_boundary_geom = gpd.read_file(study_area_geometry_fn).geometry.iloc[0]
study_boundary_geom

# %% [markdown]
# ## Determine resolution/grid to match all other rasters too (largest resolution)

# %%
largest_resolution_file = ""
largest_resolution = 0
for file in files_df.filename:
    if file.endswith("tif"):
        resolution = rio.open_rasterio(file).rio.resolution()
        print(os.path.basename(file), resolution)
        if resolution[0] > largest_resolution:
            largest_resolution = resolution[0]
            largest_resolution_file = file
print(f"Largest resolution is from file {os.path.basename(largest_resolution_file)}, {largest_resolution}")

# %% [markdown]
# ## Reproject the DEM with the largest resolution to a round number resolution, then reproject all other data to match
# Use 2 meter resolution based on output above

# %% [markdown]
# Regrid and save one raster that will serve as the "master"

# %%
master_grid_raster = rio.open_rasterio(largest_resolution_file, masked=True)
master_grid_raster = master_grid_raster.rio.reproject(master_grid_raster.rio.crs, resolution=2)
master_grid_raster_fn = largest_resolution_file.replace(
    '/input_dems/','/rasters_regridded/'
)
master_grid_raster.rio.to_raster(
    master_grid_raster_fn
)
master_grid_raster.shape, master_grid_raster.rio.resolution()

# %% [markdown]
# Open all files (slope, flowdir, and all DEMs)

# %%
slope_raster = rio.open_rasterio(downsampled_reference_slope, masked=True)
flowdir_raster = rio.open_rasterio(downsampled_reference_flowdir, masked=True)
dem_rasters_fns = [f for f in files_df.filename if f is not largest_resolution_file]
dem_rasters = [rio.open_rasterio(f, masked=True) for f in dem_rasters_fns]

# %%
dem_rasters_fns

# %% [markdown]
# Reproject slope and flowdir rasters

# %%
from rasterio.warp import Resampling
slope_raster = slope_raster.rio.reproject_match(
    master_grid_raster,
    Resampling.cubic
)
flowdir_raster = flowdir_raster.rio.reproject_match(
    master_grid_raster,
    Resampling.cubic
)

# %%
slope_raster.shape, slope_raster.rio.resolution(), flowdir_raster.shape, flowdir_raster.rio.resolution()

# %% [markdown]
# Save reprojected slope and flowdir rasters

# %%
slope_raster.rio.to_raster(
    "/data2/elilouis/hsfm-geomorph/data/mt_hood_eliot_glacier/rasters_regridded/slope.tif"
)
flowdir_raster.rio.to_raster(
    "/data2/elilouis/hsfm-geomorph/data/mt_hood_eliot_glacier/rasters_regridded/flowdir.tif"
)

# %% [markdown]
# Reproject all and save all DEMs

# %%
regridded_dems = {}
for raster, fn in zip(dem_rasters, dem_rasters_fns):
    new_fn = fn.replace('/input_dems/','/rasters_regridded/')
    new_raster = raster.rio.reproject_match(
        master_grid_raster,
        Resampling.cubic
    )
    regridded_dems[Path(new_fn).stem] = new_raster
    new_raster.rio.to_raster(new_fn)

# %%
for k,v in regridded_dems.items():
    print(k)
    print(v.rio.shape)
    print(v.rio.resolution())
    print()

# %%
regridded_dems[
    Path(master_grid_raster_fn).stem
] = master_grid_raster


# %% [markdown]
# # Generate DEM of difference datasets

# %%
def adjascent_wise(iterable):
    """iterate over a list, grabbing adjascent items at a time"""
    return zip(iterable[:-1], iterable[1:])


# %%
sorted(regridded_dems)

# %%
difference_dem_dir = "/data2/elilouis/hsfm-geomorph/data/mt_hood_eliot_glacier/difference_dems/"

# %%
list(adjascent_wise(sorted(regridded_dems)))

# %%
for old,new in adjascent_wise(sorted(regridded_dems)):
    new_raster = regridded_dems[new]
    old_raster = regridded_dems[old]
    difference_fn = os.path.join(difference_dem_dir, new + '-' + old + '.tif')
    print('Differencing:')
    print(old, new)
    print('Shape of older/newer DEMs:')
    print(old_raster.shape, new_raster.shape)
    print('Resolution of older/newer DEMs:')
    print(old_raster.rio.resolution(), new_raster.rio.resolution())
    print(f'Saving to: {difference_fn}')
    difference_dem = (new_raster - old_raster)    
    print('Shape of generated difference DEM:')
    print(difference_dem.shape)
    print('Resolution of generated difference DEM:')
    print(difference_dem.rio.resolution())
    # WHY IS THIS NECESSARY??? 
    # Note that if the "new_raster" is used, the 2009 tif (reference DEM) will have a bad nodata value
#     difference_dem = difference_dem.rio.set_attrs(old_raster.attrs)
    difference_dem.rio.to_raster(difference_fn)
    print()

# %% [markdown]
# ### Create dictionary of difference rasters

# %%
# ls -lah $difference_dem_dir

# %%
diff_rasters = {}
for f in os.listdir(difference_dem_dir):
    if f.endswith('tif'):
        print(f)
        diff_rasters[Path(f).stem] = rix.open_rasterio(
            os.path.join(difference_dem_dir, f)
        )

# %% [markdown]
# # Find common study area for all DODs

# %% [markdown]
# ## Create a mask that masks pixels if any of the datasets are NaN for a given pixel

# %%
masks = []
for key, raster in diff_rasters.items():
    masks.append(raster.to_masked_array().mask)
combined_mask = masks[0]
for mask in masks:
    combined_mask = np.logical_or(combined_mask, mask)

# %% [markdown]
# ## Apply mask to all rasters and produce datasets with area-in-common

# %%
common_diff_rasters = {}
for key, raster in diff_rasters.items():
    common_diff_rasters[key] = raster.copy().where(~combined_mask)

# %% [markdown]
# # Mask with glacier polygons

# %% [markdown]
# ## Load glacier polygons

# %%
glacier_polygons = gpd.read_file("/data2/elilouis/hsfm-geomorph/data/mt_hood_eliot_glacier/glacier_polygons.shp")
glacier_polygons

# %% [markdown]
# ## Match DOD timespan to the glacier polygon of the earliest year (to exclude glacier mass changes)

# %%
for k,v in common_diff_rasters.items():
    print(k)

# %%
dod_key_glacier_polygon_year = {
    "1975-09-1967-09" : 1967,
    "1977-10-1975-09" : 1975,
    "1980-10-1977-10" : 1977,
    "1990-09-1980-10" : 1980,
    "2009-1990-09" : 1990
}

# %% [markdown]
# ## Create dictionary of masked difference rasters

# %%
difference_dem_glacier_masked_dir = difference_dem_dir.replace("difference_dems", "difference_dems_glacier_masked")

# %%
glacier_masked_diff_rasters = {}
for k,v in common_diff_rasters.items():
    glacier_masked_raster = v.rio.clip(
        glacier_polygons[
            glacier_polygons.id==dod_key_glacier_polygon_year[k]
        ].geometry,
        drop=False, # maintain the shape of the dataset
        invert=True # "mask" instead of "clip to"
    )
    glacier_masked_raster.rio.to_raster(os.path.join(difference_dem_glacier_masked_dir, k +'.tif'))
    glacier_masked_diff_rasters[k] = glacier_masked_raster

# %% [markdown]
# # Mask with LIA Boundary

# %%
lia_boundary_gdf = gpd.read_file('/data2/elilouis/hsfm-geomorph/data/mt_hood_eliot_glacier/study_area_boundary.shp')

# %%
lia_boundary_gdf.geometry.iloc[0]

# %%
difference_dem_glacier_studyarea_masked_dir = difference_dem_glacier_masked_dir.replace(
    "difference_dems_glacier_masked",
    "difference_dems_glacier_studyarea_masked"
)
difference_dem_glacier_studyarea_masked_dir

# %%
glacier_studyarea_masked_diff_rasters = {}
for k, v in glacier_masked_diff_rasters.items():
    result = v.rio.clip(
            lia_boundary_gdf.geometry,
            drop=False, # maintain the shape of the dataset
        )
    result.rio.to_raster(os.path.join(difference_dem_glacier_studyarea_masked_dir, k + '.tif'))
    glacier_studyarea_masked_diff_rasters[k] = result

# %% [markdown]
# # Mask with NLCD
#
# This uses hsfm so requires some annoying file manipulation

# %%
from hsfm.utils import mask_dem

# %%
difference_dem_glacier_nlcd_masked_dir = difference_dem_glacier_masked_dir.replace(
    "difference_dems_glacier_masked",
    "difference_dems_glacier_nlcd_masked"
)
difference_dem_glacier_nlcd_masked_dir

# %%
glacier_nlcd_masked_diff_rasters = {}
for f in os.listdir(difference_dem_glacier_masked_dir):
    result = mask_dem(
        os.path.join(difference_dem_glacier_masked_dir, f), 
        difference_dem_glacier_nlcd_masked_dir, masks=['--nlcd']
    )
    glacier_nlcd_masked_diff_rasters[
        f.replace('.tif','')
    ] = rix.open_rasterio(result)

# %%
glacier_nlcd_masked_diff_rasters
for k,v in glacier_nlcd_masked_diff_rasters.items():
    print(k)
    print(v.rio.shape)
    print(v.rio.resolution())
    print()

# %% [markdown]
# # Visualize DODs

# %% [markdown]
# ## Heatmaps

# %% [markdown]
# ### Full DODs

# %%
fig, axes = plt.subplots(
    ncols=5,
    nrows=int(len(diff_rasters)/5),
    figsize=(20,5),
    sharex = True,
    sharey = True,
    constrained_layout=True
)
ims = []
for (date_str,raster), ax in zip(diff_rasters.items(), axes):
    ax.set_title(date_str)
    ax.set_aspect('equal')
    im = raster.plot(
        ax=ax,
        add_labels = False,
        add_colorbar = False,
    #     norm,
        vmin = -20,
        vmax = 20,
        cmap='PuOr'
    )
    ims.append(im)
cbar = fig.colorbar(ims[0], ax=axes, shrink=0.5)

# %% [markdown]
# ### Full DODs with common extent

# %%
fig, axes = plt.subplots(
    ncols=5,
    nrows=int(len(diff_rasters)/5),
    figsize=(20,5),
    sharex = True,
    sharey = True,
    constrained_layout=True
)
ims = []
for (date_str,raster), ax in zip(common_diff_rasters.items(), axes):
    ax.set_title(date_str)
    ax.set_aspect('equal')
    im = raster.plot(
        ax=ax,
        add_labels = False,
        add_colorbar = False,
    #     norm,
        vmin = -20,
        vmax = 20,
        cmap='PuOr'
    )
    ims.append(im)
cbar = fig.colorbar(ims[0], ax=axes, shrink=0.5)

# %% [markdown]
# ### Glacier-masked DODs

# %%
fig, axes = plt.subplots(
    ncols=5,
    nrows=int(len(diff_rasters)/5),
    figsize=(20,5),
    sharex = True,
    sharey = True,
    constrained_layout=True
)
ims = []
for (date_str,raster), ax in zip(glacier_masked_diff_rasters.items(), axes):
    ax.set_title(date_str)
    ax.set_aspect('equal')
    im = raster.plot(
        ax=ax,
        add_labels = False,
        add_colorbar = False,
    #     norm,
        vmin = -20,
        vmax = 20,
        cmap='PuOr'
    )
    ims.append(im)
cbar = fig.colorbar(ims[0], ax=axes, shrink=0.5)

# %% [markdown]
# ### Glacier and Study area-masked DODs

# %%
fig, axes = plt.subplots(
    ncols=5,
    nrows=int(len(diff_rasters)/5),
    figsize=(20,5),
    sharex = True,
    sharey = True,
    constrained_layout=True
)
ims = []
for (date_str,raster), ax in zip(glacier_studyarea_masked_diff_rasters.items(), axes):
    ax.set_title(date_str)
    ax.set_aspect('equal')
    im = raster.plot(
        ax=ax,
        add_labels = False,
        add_colorbar = False,
    #     norm,
        vmin = -20,
        vmax = 20,
        cmap='PuOr'
    )
    ims.append(im)
cbar = fig.colorbar(ims[0], ax=axes, shrink=0.5)

# %% [markdown]
# ### Glacier and NLCD/Forest-masked DODs

# %%
fig, axes = plt.subplots(
    ncols=5,
    nrows=int(len(diff_rasters)/5),
    figsize=(20,5),
    sharex = True,
    sharey = True,
    constrained_layout=True
)
ims = []
for (date_str,raster), ax in zip(glacier_nlcd_masked_diff_rasters.items(), axes):
    ax.set_title(date_str)
    ax.set_aspect('equal')
    im = raster.plot(
        ax=ax,
        add_labels = False,
        add_colorbar = False,
    #     norm,
        vmin = -20,
        vmax = 20,
        cmap='PuOr'
    )
    ims.append(im)
cbar = fig.colorbar(ims[0], ax=axes, shrink=0.5)


# %% [markdown]
# ## Distributions

# %%
def plot_raster_distributions(raster_dict):
    fig, axes = plt.subplots(
        ncols=5,
        nrows=int(len(raster_dict)/5),
        figsize=(25,5),
        sharex = True,
        sharey = True,
    #     constrained_layout=True
    )
    ims = []
    for (date_str,raster), ax in zip(raster_dict.items(), axes):
        ax.set_title(date_str)
    #     ax.set_aspect('equal')
        im = sns.distplot(raster.values, ax=ax, kde=False, bins=500)
        ims.append(im)
        ax.set_xlim(-20, 20)


# %% [markdown]
# ### Full DODs

# %%
plot_raster_distributions(diff_rasters)

# %% [markdown]
# ### Full DODs with common extent

# %%
plot_raster_distributions(common_diff_rasters)

# %% [markdown]
# ### Glacier-masked DODs

# %%
plot_raster_distributions(glacier_masked_diff_rasters)

# %% [markdown]
# ### Glacier and Study Area masked DODs

# %%
plot_raster_distributions(glacier_studyarea_masked_diff_rasters)

# %% [markdown]
# ### Glacier and NLCD/Forest-masked DODs

# %%
plot_raster_distributions(glacier_nlcd_masked_diff_rasters)

# %%
print('Valid data in each masked DOD:')
for key, raster in glacier_studyarea_masked_diff_rasters.items():
    x = raster.values.flatten()
    print(key)
    print(len(x[~np.isnan(x)]))

# %%
print('Valid data in each masked DOD:')
for key, raster in glacier_nlcd_masked_diff_rasters.items():
    x = raster.values.flatten()
    print(key)
    print(len(x[~np.isnan(x)]))

# %%
